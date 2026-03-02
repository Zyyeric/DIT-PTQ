import logging
import time
from typing import Union
import numpy as np
import tqdm
from tqdm import trange
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_block import BaseQuantBlock, QuantDiffBTB, QuantDiffRB
from qdiff.quant_model import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer

logger = logging.getLogger(__name__)


# ── Distributed helpers ───────────────────────────────────────────────────────

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_dist_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_dist_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def sync_grads(opt_params):
    if not is_dist_initialized():
        return
    world_size = get_dist_world_size()
    for p in opt_params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# ── Data caching ──────────────────────────────────────────────────────────────

def save_inp_oup_data(model: QuantModel,
                      layer: Union[QuantModule, BaseQuantBlock],
                      cali_data: torch.Tensor,
                      asym: bool = False,
                      act_quant: bool = False,
                      batch_size: int = 32,
                      keep_gpu: bool = True,
                      cond: bool = False,
                      is_sm: bool = False):
    """
    Cache the input and output tensors of a specific layer/block over the
    full calibration dataset by running forward passes with hooks.

    :param model:      QuantModel
    :param layer:      QuantModule or BaseQuantBlock to cache for
    :param cali_data:  calibration data tuple (xs, ts[, cs[, ucs]])
    :param asym:       if True, store quantized input / FP output (AdaRound style)
    :param act_quant:  enable activation quantization during caching
    :param batch_size: mini-batch size for forward passes
    :param keep_gpu:   move cached tensors to GPU after collection
    :param cond:       conditional generation mode
    :param is_sm:      half-size caching to avoid OOM on large attention matrices
    :return:           (cached_inps, cached_outs)

    BUG FIX: keep_gpu path now moves ALL list entries (0..7) to device,
    not just indices 0 and 1 as in the original.
    """
    t_start = time.time()
    layer_name = getattr(layer, 'nametag', type(layer).__name__)
    device = next(model.parameters()).device
    logger.info(f"[save_inp_oup_data] layer={layer_name}  "
                f"batch_size={batch_size}  asym={asym}  act_quant={act_quant}  "
                f"cond={cond}  is_sm={is_sm}  keep_gpu={keep_gpu}")

    get_inp_out = GetLayerInpOut(model, layer, device=device,
                                 asym=asym, act_quant=act_quant)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()

    # Unpack calibration data
    if not cond:
        cali_xs, cali_ts = cali_data
        cali_conds = None
        logger.debug(f"  Uncond mode: xs={tuple(cali_xs.shape)}  ts={tuple(cali_ts.shape)}")
    elif len(cali_data) == 4:
        cali_xs, cali_ts, cali_conds, cali_ack = cali_data
        logger.debug(f"  Cond(4) mode: xs={tuple(cali_xs.shape)}  "
                     f"ts={tuple(cali_ts.shape)}  cs={tuple(cali_conds.shape)}")
    else:
        cali_xs, cali_ts, cali_conds = cali_data
        logger.debug(f"  Cond(3) mode: xs={tuple(cali_xs.shape)}  "
                     f"ts={tuple(cali_ts.shape)}  cs={tuple(cali_conds.shape)}")

    # For is_sm mode, subsample to half the data to save memory
    inds = None
    if is_sm:
        inds = np.random.choice(cali_xs.size(0), cali_xs.size(0) // 2, replace=False)
        logger.info(f"  is_sm=True: subsampled to {len(inds)} / {cali_xs.size(0)} samples")

    num = int(cali_xs.size(0) / batch_size)
    if is_sm:
        num //= 2
    logger.info(f"  Running {num} forward-pass batches to collect cache...")

    l_in_0, l_in_1, l_in, l_out = 0, 0, 0, 0
    for i in trange(num, desc=f"cache {layer_name[:30]}"):
        if not cond:
            sl = (inds[i * batch_size:(i + 1) * batch_size]
                  if is_sm else slice(i * batch_size, (i + 1) * batch_size))
            cur_inp, cur_out = get_inp_out(
                cali_xs[sl].to(device),
                cali_ts[sl].to(device))
        else:
            sl = (inds[i * batch_size:(i + 1) * batch_size]
                  if is_sm else slice(i * batch_size, (i + 1) * batch_size))
            cur_inp, cur_out = get_inp_out(
                cali_xs[sl].to(device),
                cali_ts[sl].to(device),
                cali_conds[sl].to(device))

        if isinstance(cur_inp, tuple):
            if len(cur_inp) == 8:
                # Diffusers QuantDiffBTB — 8-element input tuple
                if not is_sm:
                    cached_batches.append((tuple(cur_inp), cur_out.cpu()))
                else:
                    # is_sm not supported for 8-tuple path — warn and fall back
                    logger.warning("is_sm not fully supported for 8-tuple input — "
                                   "using normal caching for this batch")
                    cached_batches.append((tuple(cur_inp), cur_out.cpu()))
            else:
                # 2-element (x, t) tuple — e.g. QuantResBlock
                cur_x, cur_t = cur_inp
                if not is_sm:
                    cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
                else:
                    if cached_inps is None:
                        l_in_0 = cur_x.shape[0] * num
                        l_in_1 = cur_t.shape[0] * num
                        cached_inps = [
                            torch.zeros(l_in_0, *cur_x.shape[1:]),
                            torch.zeros(l_in_1, *cur_t.shape[1:]),
                        ]
                    cached_inps[0].index_copy_(
                        0, torch.arange(i * cur_x.shape[0], (i + 1) * cur_x.shape[0]),
                        cur_x.cpu())
                    cached_inps[1].index_copy_(
                        0, torch.arange(i * cur_t.shape[0], (i + 1) * cur_t.shape[0]),
                        cur_t.cpu())
        else:
            # Plain tensor input
            if not is_sm:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
            else:
                if cached_inps is None:
                    l_in = cur_inp.shape[0] * num
                    cached_inps = torch.zeros(l_in, *cur_inp.shape[1:])
                cached_inps.index_copy_(
                    0, torch.arange(i * cur_inp.shape[0], (i + 1) * cur_inp.shape[0]),
                    cur_inp.cpu())

        if is_sm:
            if cached_outs is None:
                l_out = cur_out.shape[0] * num
                cached_outs = torch.zeros(l_out, *cur_out.shape[1:])
            cached_outs.index_copy_(
                0, torch.arange(i * cur_out.shape[0], (i + 1) * cur_out.shape[0]),
                cur_out.cpu())

    # Concatenate batches
    if not is_sm:
        if isinstance(cached_batches[0][0], tuple) and len(cached_batches[0][0]) == 8:
            # 8-tuple path — cat each slot individually, handle None slots
            cached_inps = [
                torch.cat([x[0][0] for x in cached_batches]),
                (torch.cat([x[0][1] for x in cached_batches])
                 if cached_batches[0][0][1] is not None
                 else [None] * len(cached_batches)),
                (torch.cat([x[0][2] for x in cached_batches])
                 if cached_batches[0][0][2] is not None
                 else [None] * len(cached_batches)),
                (torch.cat([x[0][3] for x in cached_batches])
                 if cached_batches[0][0][3] is not None
                 else [None] * len(cached_batches)),
                (torch.cat([x[0][4] for x in cached_batches])
                 if cached_batches[0][0][4] is not None
                 else [None] * len(cached_batches)),
                [x[0][5] for x in cached_batches],   # cross_attention_kwargs — list of dicts
                (torch.cat([x[0][6] for x in cached_batches])
                 if cached_batches[0][0][6] is not None
                 else [None] * len(cached_batches)),
                (torch.cat([x[0][7] for x in cached_batches])
                 if cached_batches[0][0][7] is not None
                 else [None] * len(cached_batches)),
            ]
        elif isinstance(cached_batches[0][0], tuple):
            cached_inps = [
                torch.cat([x[0][0] for x in cached_batches]),
                torch.cat([x[0][1] for x in cached_batches]),
            ]
        else:
            cached_inps = torch.cat([x[0] for x in cached_batches])
        cached_outs = torch.cat([x[1] for x in cached_batches])

    # Log shapes
    if isinstance(cached_inps, list):
        for idx, ci in enumerate(cached_inps):
            if torch.is_tensor(ci):
                logger.info(f"  cached_inps[{idx}] shape: {tuple(ci.shape)}  dtype={ci.dtype}")
            else:
                logger.info(f"  cached_inps[{idx}]: non-tensor ({type(ci).__name__}, len={len(ci) if hasattr(ci, '__len__') else 'N/A'})")
    else:
        logger.info(f"  cached_inps shape: {tuple(cached_inps.shape)}  dtype={cached_inps.dtype}")
    logger.info(f"  cached_outs shape: {tuple(cached_outs.shape)}  dtype={cached_outs.dtype}")

    torch.cuda.empty_cache()

    if keep_gpu:
        if isinstance(cached_inps, list):
            # BUG FIX: original only moved indices 0 and 1 to device.
            # All tensor slots must be moved; non-tensor slots (None lists, dict lists) are left as-is.
            for idx in range(len(cached_inps)):
                if torch.is_tensor(cached_inps[idx]):
                    cached_inps[idx] = cached_inps[idx].to(device)
                    logger.debug(f"  Moved cached_inps[{idx}] to {device}")
        else:
            cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        logger.debug(f"  Moved cached_outs to {device}")

    logger.info(f"[save_inp_oup_data] Done in {time.time()-t_start:.1f}s")
    return cached_inps, cached_outs


def save_grad_data(model: QuantModel,
                   layer: Union[QuantModule, BaseQuantBlock],
                   cali_data: torch.Tensor,
                   damping: float = 1.,
                   act_quant: bool = False,
                   batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Cache gradient data of a layer/block over the calibration dataset.
    Uses KL divergence between FP and quantized outputs to compute Fisher-like gradients.

    :param model:      QuantModel
    :param layer:      QuantModule or BaseQuantBlock
    :param cali_data:  calibration data tensor
    :param damping:    add to FIM diagonal for numerical stability (unused currently)
    :param act_quant:  enable activation quantization
    :param batch_size: mini-batch size
    :param keep_gpu:   move result to GPU
    :return:           gradient tensor (abs + 1.0 stabilisation)
    """
    t_start = time.time()
    layer_name = getattr(layer, 'nametag', type(layer).__name__)
    device = next(model.parameters()).device
    logger.info(f"[save_grad_data] layer={layer_name}  "
                f"batch_size={batch_size}  act_quant={act_quant}")

    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    num = int(cali_data.size(0) / batch_size)
    for i in trange(num, desc=f"grad {layer_name[:30]}"):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat(cached_batches)
    cached_grads = cached_grads.abs() + 1.0
    logger.info(f"  cached_grads shape: {tuple(cached_grads.shape)}  "
                f"mean={cached_grads.mean():.4f}  max={cached_grads.max():.4f}")

    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)

    logger.info(f"[save_grad_data] Done in {time.time()-t_start:.1f}s")
    return cached_grads


# ── Forward / backward hooks ──────────────────────────────────────────────────

class StopForwardException(Exception):
    """Thrown to abort a forward pass early once the target layer has been reached."""
    pass


class DataSaverHook:
    """Forward hook that stores the input and/or output of a module."""
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input  = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store  = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
            # For QuantDiffBTB: reconstruct full 8-element input from cached side-channel attrs
            if hasattr(module, "ts_cache"):
                self.input_store = (
                    input_batch[0],
                    module.am_cache,
                    module.ehs_cache,
                    module.eam_cache,
                    module.ts_cache,
                    module.cak_cache,
                    module.class_labels,
                    module.added_cond_kwargs,
                )
                logger.debug(f"DataSaverHook: reconstructed 8-tuple input for QuantDiffBTB")
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    """
    Runs a forward pass and captures the input/output of a target layer via hook.
    Supports conditional (PixArt / Diffusers) and unconditional models.

    BUG FIX: original had `assert context is not None` which always crashed
    in unconditional mode. Now context is optional and only passed when not None.
    """
    def __init__(self, model: QuantModel,
                 layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device,
                 asym: bool = False,
                 act_quant: bool = False):
        self.model      = model
        self.layer      = layer
        self.asym       = asym
        self.device     = device
        self.act_quant  = act_quant
        self.data_saver = DataSaverHook(
            store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, timesteps, context=None):
        self.model.eval()
        self.model.set_quant_state(False, False)

        # BUG FIX: removed `assert context is not None` — unconditional models
        # pass context=None and must not crash here.
        if context is None:
            logger.debug("GetLayerInpOut: context=None (unconditional mode)")

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                if context is not None:
                    _ = self.model(
                        x, timestep=timesteps,
                        encoder_hidden_states=context,
                        added_cond_kwargs=pixart_alpha_aca_dict(x))
                else:
                    _ = self.model(x, timestep=timesteps)
            except StopForwardException:
                pass

            if self.asym:
                # Re-run with quantized weights to get the quantized input
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    if context is not None:
                        _ = self.model(
                            x, timestep=timesteps,
                            encoder_hidden_states=context,
                            added_cond_kwargs=pixart_alpha_aca_dict(x))
                    else:
                        _ = self.model(x, timestep=timesteps)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        # Return appropriate tuple based on layer type
        if isinstance(self.layer, QuantDiffBTB):
            inp = self.data_saver.input_store
            logger.debug(f"GetLayerInpOut (QuantDiffBTB): returning 8-tuple  "
                         f"[0]={tuple(inp[0].shape) if torch.is_tensor(inp[0]) else 'None'}")
            return (
                inp[0].detach().cpu(),
                inp[1].detach().cpu() if inp[1] is not None else None,
                inp[2].detach().cpu() if inp[2] is not None else None,
                inp[3].detach().cpu() if inp[3] is not None else None,
                inp[4].detach().cpu() if inp[4] is not None else None,
                inp[5].detach().cpu() if inp[5] is not None else None,
                inp[6].detach().cpu() if inp[6] is not None else None,
                inp[7].detach().cpu() if inp[7] is not None else None,
            ), self.data_saver.output_store.detach().cpu()

        elif (len(self.data_saver.input_store) > 1
              and torch.is_tensor(self.data_saver.input_store[1])):
            inp = self.data_saver.input_store
            logger.debug(f"GetLayerInpOut (2-tuple): [0]={tuple(inp[0].shape)}  "
                         f"[1]={tuple(inp[1].shape)}")
            return (inp[0].detach(), inp[1].detach()), self.data_saver.output_store.detach()

        else:
            inp = self.data_saver.input_store[0]
            logger.debug(f"GetLayerInpOut (tensor): shape={tuple(inp.shape)}")
            return inp.detach(), self.data_saver.output_store.detach()


class GradSaverHook:
    """Backward hook that stores the output gradient of a module."""
    def __init__(self, store_grad=True):
        self.store_grad    = store_grad
        self.stop_backward = False
        self.grad_out      = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    """
    Computes KL-divergence gradient between FP and quantized model outputs,
    used for Fisher-weighted reconstruction in BRECQ.
    """
    def __init__(self, model: QuantModel,
                 layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device,
                 act_quant: bool = False):
        self.model     = model
        self.layer     = layer
        self.device    = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        self.model.eval()
        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q  = self.model(inputs)
                loss   = F.kl_div(
                    F.log_softmax(out_q, dim=1),
                    F.softmax(out_fp, dim=1),
                    reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass
        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


# ── Quantization helpers ──────────────────────────────────────────────────────

def quantize_model_till(model: QuantModule,
                        layer: Union[QuantModule, BaseQuantBlock],
                        act_quant: bool = False):
    """
    Enable quantization for all modules up to and including `layer`.
    Assumes modules are iterated in forward order.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break


# ── Calibration data helpers ──────────────────────────────────────────────────

def get_train_samples(args, sample_data, custom_steps=None):
    num_samples, num_st = args.cali_n, args.cali_st
    custom_steps = args.custom_steps if custom_steps is None else custom_steps
    if num_st == 1:
        xs = sample_data[:num_samples]
        ts = torch.ones(num_samples) * 800
        return xs, ts
    nsteps    = len(sample_data["ts"])
    assert nsteps >= custom_steps, f"nsteps={nsteps} < custom_steps={custom_steps}"
    timesteps = list(range(0, nsteps, nsteps // num_st))
    logger.info(f"get_train_samples: selected {len(timesteps)} steps from {nsteps} "
                f"(every {nsteps//num_st})")
    xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
    ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]
    if args.cond:
        xs_lst   += xs_lst
        ts_lst   += ts_lst
        conds_lst = ([sample_data["cs"][i][:num_samples]  for i in timesteps]
                   + [sample_data["ucs"][i][:num_samples] for i in timesteps])
        conds = torch.cat(conds_lst, dim=0)
    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)
    if args.cond:
        return xs, ts, conds
    return xs, ts


def get_train_samples_custom(args, sample_data, custom_steps=None):
    """
    Wrapper that delegates to get_train_samples_custom_ucs.

    BUG FIX: original had a dead code block below the delegate call
    (unreachable code after return) with different and incorrect behaviour
    (missing the ucs doubling). Removed entirely.
    """
    return get_train_samples_custom_ucs(args, sample_data, custom_steps)


def get_train_samples_custom_ucs(args, sample_data, custom_steps=None):
    """
    Build calibration tensors from the pre-saved sample_data dict.

    For conditional PixArt with unconditional conditioning (ucs), doubles the
    batch with both conditional and unconditional embeddings.

    BUG FIX: original accessed conds_lst unconditionally at the end even when
    it was only assigned inside the if-branch, causing UnboundLocalError when
    args.cond=False or "ucs" is missing from sample_data.
    """
    num_samples, num_st = args.cali_n, args.cali_st

    nsteps = len(sample_data["ts"])
    assert nsteps >= custom_steps, \
        f"nsteps={nsteps} < custom_steps={custom_steps}"
    timesteps = list(range(0, nsteps, nsteps // num_st))
    logger.info(f"get_train_samples_custom_ucs: {len(timesteps)} timesteps from {nsteps}  "
                f"num_samples={num_samples}  cond={args.cond}  "
                f"has_ucs={'ucs' in sample_data}")

    xs_lst = [sample_data["xs"][i][:num_samples] for i in timesteps]
    ts_lst = [sample_data["ts"][i][:num_samples] for i in timesteps]

    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)

    # BUG FIX: conds_lst was only assigned in the branch but used unconditionally.
    # Now we always return xs, ts[, conds] consistently.
    if args.cond and "ucs" in sample_data:
        xs_lst   += xs_lst
        ts_lst   += ts_lst
        conds_lst = ([sample_data["cs"][i][:num_samples]  for i in timesteps]
                   + [sample_data["ucs"][i][:num_samples] for i in timesteps])
        xs    = torch.cat(xs_lst,    dim=0)
        ts    = torch.cat(ts_lst,    dim=0)
        conds = torch.cat(conds_lst, dim=0)
        logger.info(f"  Doubled with ucs: xs={tuple(xs.shape)}  "
                    f"ts={tuple(ts.shape)}  conds={tuple(conds.shape)}")
        return xs, ts, conds

    elif args.cond and "cs" in sample_data:
        # Cond mode but no ucs — use only conditional embeddings
        logger.warning("get_train_samples_custom_ucs: args.cond=True but 'ucs' not in "
                       "sample_data — using conditional embeddings only (no doubling)")
        conds_lst = [sample_data["cs"][i][:num_samples] for i in timesteps]
        conds = torch.cat(conds_lst, dim=0)
        logger.info(f"  Cond only: xs={tuple(xs.shape)}  ts={tuple(ts.shape)}  "
                    f"conds={tuple(conds.shape)}")
        return xs, ts, conds

    else:
        logger.info(f"  Uncond: xs={tuple(xs.shape)}  ts={tuple(ts.shape)}")
        return xs, ts


def get_train_samples_sdxl(args, sample_data, custom_steps=None):
    """SDXL variant — also returns text_embeds and time_ids."""
    num_samples, num_st = args.cali_n, args.cali_st
    nsteps    = len(sample_data["ts"])
    assert nsteps >= custom_steps, \
        f"nsteps={nsteps} < custom_steps={custom_steps}"
    timesteps = list(range(0, nsteps, nsteps // num_st))
    logger.info(f"get_train_samples_sdxl: {len(timesteps)} steps  num_samples={num_samples}")
    xs_lst    = [sample_data["xs"][i][:num_samples] for i in timesteps]
    ts_lst    = [sample_data["ts"][i][:num_samples] for i in timesteps]
    conds_lst = [sample_data["cs"][i][:num_samples] for i in timesteps]
    tes       = sample_data["text_embeds"][:num_samples]
    tid       = sample_data["time_ids"]
    xs    = torch.cat(xs_lst,    dim=0)
    ts    = torch.cat(ts_lst,    dim=0)
    conds = torch.cat(conds_lst, dim=0)
    logger.info(f"  xs={tuple(xs.shape)}  ts={tuple(ts.shape)}  conds={tuple(conds.shape)}  "
                f"tes={tuple(tes.shape)}  tid={tuple(tid.shape) if torch.is_tensor(tid) else tid}")
    return xs, ts, conds, tes, tid


# ── AdaRound conversion and checkpoint resume ─────────────────────────────────

def convert_adaround(model):
    """
    Replace all QuantModule weight quantizers with AdaRoundQuantizer in-place.
    Skips layers/blocks with ignore_reconstruction=True.
    """
    n_converted = 0
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction:
                logger.debug(f"convert_adaround: skipping layer {name}")
                continue
            module.weight_quantizer = AdaRoundQuantizer(
                uaq=module.weight_quantizer,
                round_mode='learned_hard_sigmoid',
                weight_tensor=module.org_weight.data)
            n_converted += 1
            logger.debug(f"convert_adaround: converted layer {name}")
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction:
                logger.debug(f"convert_adaround: skipping block {name}")
                continue
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, QuantModule):
                    if sub_module.split != 0:
                        sub_module.weight_quantizer = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer,
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data[:, :sub_module.split, ...])
                        sub_module.weight_quantizer_0 = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer_0,
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data[:, sub_module.split:, ...])
                    else:
                        sub_module.weight_quantizer = AdaRoundQuantizer(
                            uaq=sub_module.weight_quantizer,
                            round_mode='learned_hard_sigmoid',
                            weight_tensor=sub_module.org_weight.data)
                    n_converted += 1
                    logger.debug(f"convert_adaround: converted block.{sub_name}")
        else:
            convert_adaround(module)
    if n_converted > 0:
        logger.info(f"convert_adaround: converted {n_converted} QuantModules")


def resume_cali_model(qnn, ckpt_path, cali_data,
                      quant_act=False, act_quant_mode='qdiff', cond=False):
    """
    Resume quantization from a saved checkpoint.

    Initializes weight quantizer parameters by running a short forward pass,
    converts to AdaRound, loads the checkpoint, then optionally initializes
    activation quantizers if quant_act=True.

    BUG FIX: the `elif len(cali_data) == 4` branch in the original unpacked
    3 values from a length-4 tuple (cali_xs, cali_ts, cali_cs = cali_data)
    — silent wrong unpack, last element silently dropped. Now correctly
    unpacks all 4 elements.
    """
    logger.info(f"resume_cali_model: loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    logger.info(f"  Checkpoint keys: {len(ckpt)}  "
                f"(first 5: {list(ckpt.keys())[:5]})")

    logger.info("  Initializing weight quantization parameters via forward pass...")
    qnn.set_quant_state(True, False)

    if not cond:
        cali_xs, cali_ts = cali_data
        logger.debug(f"  Uncond init: xs={tuple(cali_xs[:1].shape)}")
        _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda())

    elif len(cali_data) == 5:
        # SDXL
        cali_xs, cali_ts, cali_cs, cali_tes, cali_tid = cali_data
        logger.debug(f"  SDXL(5) init: xs={tuple(cali_xs[:1].shape)}")
        _ = qnn(
            cali_xs[:1].cuda(), cali_ts[:1].cuda(),
            encoder_hidden_states=cali_cs[:1].cuda(),
            added_cond_kwargs={
                "text_embeds": cali_tes[:1].cuda(),
                "time_ids":    cali_tid.cuda()},
            cross_attention_kwargs={}, return_dict=False)

    elif len(cali_data) == 4:
        # BUG FIX: original unpacked only 3 values from a 4-element tuple,
        # silently dropping the 4th element (cali_ack / added_cond_kwargs).
        cali_xs, cali_ts, cali_cs, cali_ack = cali_data
        logger.debug(f"  Cond(4) init: xs={tuple(cali_xs[:2].shape)}")
        with torch.no_grad():
            _ = qnn(
                cali_xs[:2].cuda(),
                timestep=cali_ts[:2].cuda(),
                encoder_hidden_states=cali_cs[:2].cuda(),
                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[:2]))

    else:
        # PixArt (3-element)
        cali_xs, cali_ts, cali_cs = cali_data
        cali_xs = cali_xs.to(torch.float16)
        cali_cs = cali_cs.to(torch.float16)
        logger.debug(f"  PixArt(3) init: xs={tuple(cali_xs[:2].shape)}")
        with torch.no_grad():
            _ = qnn(
                cali_xs[:2].cuda(),
                timestep=cali_ts[:2].cuda(),
                encoder_hidden_states=cali_cs[:2].cuda(),
                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[:2]))

    logger.info("  Converting weight quantizers to AdaRound...")
    convert_adaround(qnn)

    # Promote zero_point and delta to Parameters for state_dict compatibility
    n_promoted = 0
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            m.zero_point = nn.Parameter(m.zero_point)
            m.delta      = nn.Parameter(m.delta)
            n_promoted  += 1
    logger.info(f"  Promoted {n_promoted} AdaRound zero_point/delta to Parameters")

    # Strip activation quantizer keys before loading
    act_keys = [k for k in ckpt if "act" in k]
    logger.info(f"  Removing {len(act_keys)} activation quantizer keys from checkpoint")
    for k in act_keys:
        del ckpt[k]

    strict = (act_quant_mode == 'qdiff')
    logger.info(f"  Loading state dict (strict={strict})...")
    missing, unexpected = qnn.load_state_dict(ckpt, strict=strict)
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    qnn.set_quant_state(weight_quant=True, act_quant=False)

    # Demote Parameters back to plain tensors/scalars for inference
    for m in qnn.model.modules():
        if isinstance(m, AdaRoundQuantizer):
            zp = m.zero_point.data; delattr(m, "zero_point"); m.zero_point = zp
            dl = m.delta.data;      delattr(m, "delta");      m.delta      = dl
    logger.info("  Weight-only resume complete")

    # ── Optional: activation quantization init ────────────────────────────────
    if quant_act:
        logger.info("  Initializing activation quantization parameters...")
        qnn.set_quant_state(True, True)

        if not cond:
            _ = qnn(cali_xs[:1].cuda(), cali_ts[:1].cuda())
        elif len(cali_data) == 4:
            # BUG FIX (same as above): was indexing cali_data dict-style which
            # would crash since cali_data is a tuple here. Use unpacked vars.
            cali_xs = cali_xs.to(torch.float16)
            cali_cs = cali_cs.to(torch.float16)
            with torch.no_grad():
                _ = qnn(
                    cali_xs[:2].cuda(),
                    timestep=cali_ts[:2].cuda(),
                    encoder_hidden_states=cali_cs[:2].cuda(),
                    added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[:2]))
        else:
            # PixArt
            cali_xs = cali_xs.to(torch.float16)
            cali_cs = cali_cs.to(torch.float16)
            with torch.no_grad():
                _ = qnn(
                    cali_xs[:2].cuda(),
                    timestep=cali_ts[:2].cuda(),
                    encoder_hidden_states=cali_cs[:2].cuda(),
                    added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[:2]))

        logger.info("  Promoting all quantizer params for act checkpoint load...")
        n_promoted = 0
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                m.zero_point = nn.Parameter(m.zero_point)
                m.delta      = nn.Parameter(m.delta)
                n_promoted  += 1
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    if not torch.is_tensor(m.zero_point):
                        m.zero_point = nn.Parameter(
                            torch.tensor(float(m.zero_point)))
                    else:
                        m.zero_point = nn.Parameter(m.zero_point)
                    n_promoted += 1
        logger.info(f"  Promoted {n_promoted} params")

        logger.info(f"  Re-loading checkpoint for full quant state: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        qnn.load_state_dict(ckpt)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        # Demote back to plain values for inference
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                zp = m.zero_point.data; delattr(m, "zero_point"); m.zero_point = zp
                dl = m.delta.data;      delattr(m, "delta");      m.delta      = dl
            elif isinstance(m, UniformAffineQuantizer):
                if m.zero_point is not None:
                    zp_val = m.zero_point.item()
                    delattr(m, "zero_point")
                    assert int(zp_val) == zp_val, \
                        f"zero_point {zp_val} is not an integer — unexpected for INT quant"
                    m.zero_point = int(zp_val)
        logger.info("  Full quant resume complete")


# ── Misc ──────────────────────────────────────────────────────────────────────

def greedy_core_set_selection(unique_points, size, dist_func, verbose=True):
    """
    Greedy coreset selection (Algorithm 1 from SmallGAN).
    Selects `size` points that maximally cover the input set.
    """
    assert len(unique_points) >= size, \
        f"Core set size {size} > number of points {len(unique_points)}"
    bar = None
    if verbose:
        from tqdm import tqdm
        bar = tqdm(total=size, desc="Greedy core set selection", ascii=True)

    remaining = [v for v in unique_points]
    random.shuffle(remaining)
    core_set  = [remaining[0]]
    remaining = remaining[1:]
    memo = {}

    while len(core_set) < size:
        max_dist      = None
        cand_idx      = None
        for pi, p1 in enumerate(remaining):
            min_dist = min(dist_func(p1, p2, memo) for p2 in core_set)
            if max_dist is None or min_dist > max_dist:
                max_dist = min_dist
                cand_idx = pi
        assert cand_idx is not None
        core_set.append(remaining[cand_idx])
        if bar is not None:
            bar.update(1)
        del remaining[cand_idx]

    if bar is not None:
        bar.close()
    logger.info(f"greedy_core_set_selection: selected {len(core_set)} points")
    return core_set, memo


def pixart_alpha_aca_dict(x):
    """
    Build the `added_cond_kwargs` dict expected by PixArt-Alpha's transformer.
    Infers resolution from the latent spatial size (x.shape[2] * 8).
    """
    bs     = x.shape[0]
    hw     = x.shape[2] * 8
    device = x.device
    dtype  = x.dtype if torch.is_floating_point(x) else torch.float16
    return {
        'resolution':   torch.tensor([[hw, hw]], device=device, dtype=dtype).expand(bs, -1),
        'aspect_ratio': torch.tensor([[1.0]],    device=device, dtype=dtype).expand(bs, -1),
    }