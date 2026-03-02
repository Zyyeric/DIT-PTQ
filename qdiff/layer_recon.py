import time
import logging
import torch
from qdiff.quant_layer import QuantModule, StraightThrough, lp_loss
from qdiff.quant_model import QuantModel
from qdiff.block_recon import LinearTempDecay
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import (
    save_grad_data,
    save_inp_oup_data,
    sync_grads,
    get_dist_rank,
)

logger = logging.getLogger(__name__)


def layer_reconstruction(model: QuantModel, layer: QuantModule,
                         cali_data: torch.Tensor,
                         batch_size: int = 32,
                         iters: int = 20000,
                         weight: float = 0.001,
                         opt_mode: str = 'mse',
                         asym: bool = False,
                         include_act_func: bool = True,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         act_quant: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         multi_gpu: bool = False,
                         cond: bool = False,
                         is_sm: bool = False,
                         sequential: bool = False,
                         no_adaround: bool = False):
    """
    Layer-level reconstruction to optimize rounding or activation scale.

    BUG FIX: save_inp_oup_data now uses the passed batch_size argument instead
    of the hardcoded value of 8 that was in the original code.
    """
    t_start = time.time()
    layer_name = getattr(layer, 'nametag', type(layer).__name__)
    logger.info(f"[LayerRecon] Starting: {layer_name}  "
                f"iters={iters}  batch_size={batch_size}  act_quant={act_quant}  "
                f"asym={asym}  opt_mode={opt_mode}  warmup={warmup}")

    model.set_quant_state(False, False)
    if sequential:
        model.set_quant_state(True, act_quant)
        logger.debug(f"[LayerRecon] Sequential mode — full model quant state set")
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        if no_adaround:
            logger.info(f"[LayerRecon] no_adaround=True — skipping {layer_name}")
            return

        # Replace weight quantizer with AdaRoundQuantizer
        if layer.split != 0:
            layer.weight_quantizer = AdaRoundQuantizer(
                uaq=layer.weight_quantizer, round_mode=round_mode,
                weight_tensor=layer.org_weight.data[:, :layer.split, ...])
            layer.weight_quantizer_0 = AdaRoundQuantizer(
                uaq=layer.weight_quantizer_0, round_mode=round_mode,
                weight_tensor=layer.org_weight.data[:, layer.split:, ...])
        else:
            layer.weight_quantizer = AdaRoundQuantizer(
                uaq=layer.weight_quantizer, round_mode=round_mode,
                weight_tensor=layer.org_weight.data)
        layer.weight_quantizer.soft_targets = True

        opt_params = [layer.weight_quantizer.alpha]
        if layer.split != 0:
            opt_params += [layer.weight_quantizer_0.alpha]
        optimizer = torch.optim.Adam(opt_params, lr=1e-3)
        scheduler = None
        logger.debug(f"[LayerRecon] Weight mode — AdaRound alpha params: {len(opt_params)}")
    else:
        opt_params = [layer.act_quantizer.delta]
        if layer.split != 0 and layer.act_quantizer_0.delta is not None:
            opt_params += [layer.act_quantizer_0.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=0.)
        logger.debug(f"[LayerRecon] Act mode — delta params: {len(opt_params)}  lr={lr}")

    loss_mode = 'none' if act_quant else 'relaxation'
    loss_func = LossFunction(
        layer, round_loss=loss_mode, weight=weight,
        max_count=iters, rec_loss=opt_mode,
        b_range=b_range, decay_start=0, warmup=warmup, p=p)

    # BUG FIX: original hardcoded batch_size=8 here, ignoring the argument.
    logger.debug(f"[LayerRecon] Caching inp/out data (batch_size={batch_size})...")
    t_cache = time.time()
    cached_inps, cached_outs = save_inp_oup_data(
        model, layer, cali_data, asym, act_quant,
        batch_size,          # FIX: was hardcoded 8
        keep_gpu=False, cond=cond, is_sm=is_sm)
    logger.debug(f"[LayerRecon] Cache built in {time.time()-t_cache:.1f}s  "
                 f"inps={tuple(cached_inps.shape) if hasattr(cached_inps,'shape') else type(cached_inps)}  "
                 f"outs={tuple(cached_outs.shape) if hasattr(cached_outs,'shape') else type(cached_outs)}")

    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, layer, cali_data, act_quant,
                                      batch_size=batch_size)
    else:
        cached_grads = None

    device = 'cuda'
    rank   = get_dist_rank()
    rng    = torch.Generator(device='cpu')
    rng.manual_seed(1337 + rank)

    layer = layer.to(device, torch.float32)
    t_opt = time.time()
    for i in range(iters):
        idx     = torch.randperm(cached_inps.size(0), generator=rng)[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad(set_to_none=True)
        out_quant = layer(cur_inp)
        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward()
        if multi_gpu:
            sync_grads(opt_params)
        optimizer.step()
        if scheduler:
            scheduler.step()

    logger.debug(f"[LayerRecon] Optimization loop done in {time.time()-t_opt:.1f}s")

    layer = layer.to(device, torch.float16)
    torch.cuda.empty_cache()

    # Finalize — switch to hard rounding
    layer.weight_quantizer.soft_targets = False
    if layer.split != 0:
        layer.weight_quantizer_0.soft_targets = False

    if not include_act_func:
        layer.activation_function = org_act_func

    logger.info(f"[LayerRecon] Done: {layer_name}  total={time.time()-t_start:.1f}s")


class LossFunction:
    def __init__(self, layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):
        self.layer      = layer
        self.round_loss = round_loss
        self.weight     = weight
        self.rec_loss   = rec_loss
        self.loss_start = max_count * warmup
        self.p          = p
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        self.count += 1

        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a   = (pred - tgt).abs()
            g   = grad.abs()
            bdp = torch.sum(a * g, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (bdp * a * g).mean() / 100
        else:
            raise ValueError(f"Unknown rec_loss: {self.rec_loss}")

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss = self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError(f"Unknown round_loss: {self.round_loss}")

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info(f"[LayerRecon] iter={self.count}  "
                        f"total={float(total_loss):.4f}  "
                        f"rec={float(rec_loss):.4f}  "
                        f"round={float(round_loss):.4f}  "
                        f"b={b:.3f}")
        return total_loss