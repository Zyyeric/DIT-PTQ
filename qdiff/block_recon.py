import time
import logging
import torch
from qdiff.quant_layer import QuantModule, StraightThrough, lp_loss
from qdiff.quant_model import QuantModel
from qdiff.quant_block import BaseQuantBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import (
    save_grad_data,
    save_inp_oup_data,
    sync_grads,
    get_dist_rank,
)

logger = logging.getLogger(__name__)


def block_reconstruction(model: QuantModel, block: BaseQuantBlock,
                         cali_data: torch.Tensor,
                         batch_size: int = 32,
                         iters: int = 20000,
                         weight: float = 0.01,
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
    Block-level reconstruction to optimize rounding or activation scale.

    BUG FIX: save_inp_oup_data now uses the passed batch_size argument instead
    of the hardcoded value of 8 that was in the original code.
    """
    t_start    = time.time()
    block_name = getattr(block, 'nametag', type(block).__name__)
    n_qm = sum(1 for m in block.modules() if isinstance(m, QuantModule))
    logger.info(f"[BlockRecon] Starting: {block_name}  "
                f"iters={iters}  batch_size={batch_size}  act_quant={act_quant}  "
                f"n_QuantModules={n_qm}  asym={asym}  opt_mode={opt_mode}")

    model.set_quant_state(False, False)
    if sequential:
        model.set_quant_state(True, act_quant)
        logger.debug(f"[BlockRecon] Sequential mode — full model quant state set")
    block.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        if no_adaround:
            logger.info(f"[BlockRecon] no_adaround=True — skipping {block_name}")
            return

        # Replace all QuantModule weight quantizers with AdaRoundQuantizer
        n_adaround = 0
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.split != 0:
                    module.weight_quantizer = AdaRoundQuantizer(
                        uaq=module.weight_quantizer, round_mode=round_mode,
                        weight_tensor=module.org_weight.data[:, :module.split, ...])
                    module.weight_quantizer_0 = AdaRoundQuantizer(
                        uaq=module.weight_quantizer_0, round_mode=round_mode,
                        weight_tensor=module.org_weight.data[:, module.split:, ...])
                else:
                    module.weight_quantizer = AdaRoundQuantizer(
                        uaq=module.weight_quantizer, round_mode=round_mode,
                        weight_tensor=module.org_weight.data)
                module.weight_quantizer.soft_targets = True
                if module.split != 0:
                    module.weight_quantizer_0.soft_targets = True
                n_adaround += 1

        logger.debug(f"[BlockRecon] Replaced {n_adaround} weight quantizers with AdaRound")

        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params.append(module.weight_quantizer.alpha)
                if module.split != 0:
                    opt_params.append(module.weight_quantizer_0.alpha)

        if len(opt_params) > 0:
            optimizer = torch.optim.Adam(opt_params, lr=1e-3)
            logger.debug(f"[BlockRecon] Weight mode — {len(opt_params)} alpha params")
        else:
            optimizer = None
            logger.warning(f"[BlockRecon] No opt_params found for {block_name} — "
                           "block has no QuantModules to optimize")
        scheduler = None

    else:
        # Activation quantization: collect all delta params from block
        opt_params = []

        if hasattr(block, 'act_quantizer') and block.act_quantizer.delta is not None:
            opt_params.append(block.act_quantizer.delta)

        if hasattr(block, 'attn1'):
            opt_params += [
                block.attn1.act_quantizer_q.delta,
                block.attn1.act_quantizer_k.delta,
                block.attn1.act_quantizer_v.delta,
            ]
            if hasattr(block, 'attn2') and block.attn2 is not None:
                opt_params += [
                    block.attn2.act_quantizer_q.delta,
                    block.attn2.act_quantizer_k.delta,
                    block.attn2.act_quantizer_v.delta,
                ]
            if block.attn1.act_quantizer_w.n_bits != 16:
                opt_params.append(block.attn1.act_quantizer_w.delta)
            if (hasattr(block, 'attn2') and block.attn2 is not None
                    and block.attn2.act_quantizer_w.n_bits != 16):
                opt_params.append(block.attn2.act_quantizer_w.delta)

        if hasattr(block, 'act_quantizer_q'):
            opt_params += [block.act_quantizer_q.delta, block.act_quantizer_k.delta]
        if hasattr(block, 'act_quantizer_w'):
            opt_params.append(block.act_quantizer_v.delta)
            if block.act_quantizer_w.n_bits != 16:
                opt_params.append(block.act_quantizer_w.delta)

        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.delta is not None:
                    opt_params.append(module.act_quantizer.delta)
                if module.split != 0 and module.act_quantizer_0.delta is not None:
                    opt_params.append(module.act_quantizer_0.delta)

        # Remove None entries safely
        opt_params = [p for p in opt_params if p is not None]
        logger.debug(f"[BlockRecon] Act mode — {len(opt_params)} delta params  lr={lr}")
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    loss_func = LossFunction(
        block, round_loss=loss_mode, weight=weight,
        max_count=iters, rec_loss=opt_mode,
        b_range=b_range, decay_start=0, warmup=warmup, p=p)

    # BUG FIX: original hardcoded batch_size=8 here, ignoring the argument.
    logger.debug(f"[BlockRecon] Caching inp/out data (batch_size={batch_size})...")
    t_cache = time.time()
    cached_inps, cached_outs = save_inp_oup_data(
        model, block, cali_data, asym, act_quant,
        batch_size,          # FIX: was hardcoded 8
        keep_gpu=False, cond=cond, is_sm=is_sm)
    logger.debug(f"[BlockRecon] Cache built in {time.time()-t_cache:.1f}s")

    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, block, cali_data, act_quant,
                                      batch_size=batch_size)
    else:
        cached_grads = None

    device = 'cuda'
    rank   = get_dist_rank()
    rng    = torch.Generator(device='cpu')
    rng.manual_seed(2024 + rank)

    block  = block.to(device, torch.float32)
    t_opt  = time.time()

    if optimizer is not None:
        for i in range(iters):
            if isinstance(cached_inps, list):
                idx = torch.randperm(cached_inps[0].size(0), generator=rng)[:batch_size]
                if len(cached_inps) == 8:
                    cur_x   = cached_inps[0][idx].to(device, torch.float32)
                    cur_am  = (cached_inps[1][idx].to(device, torch.float32)
                               if isinstance(cached_inps[1], torch.Tensor) else None)
                    cur_ehs = (cached_inps[2][idx].to(device, torch.float32)
                               if isinstance(cached_inps[2], torch.Tensor) else None)
                    cur_eam = (cached_inps[3][idx].to(device, torch.float32)
                               if isinstance(cached_inps[3], torch.Tensor) else None)
                    cur_ts  = (cached_inps[4][idx].to(device)
                               if isinstance(cached_inps[4], torch.Tensor) else None)
                    cur_cak = (cached_inps[5][idx]
                               if cached_inps[5][0] is not None else None)
                    cur_cl  = (cached_inps[6][idx].to(device)
                               if isinstance(cached_inps[6], torch.Tensor) else None)
                    cur_ack = (cached_inps[7][idx].to(device)
                               if isinstance(cached_inps[7], torch.Tensor) else None)
                    cur_inp = (cur_x, cur_am, cur_ehs, cur_eam, cur_ts, cur_cak, cur_cl, cur_ack)
                else:
                    cur_x   = cached_inps[0][idx].to(device)
                    cur_t   = cached_inps[1][idx].to(device)
                    cur_inp = (cur_x, cur_t)
            else:
                idx     = torch.randperm(cached_inps.size(0), generator=rng)[:batch_size]
                cur_inp = cached_inps[idx].to(device)

            cur_out  = cached_outs[idx].to(device)
            cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

            optimizer.zero_grad(set_to_none=True)
            if isinstance(cur_inp, tuple):
                out_quant = block(*cur_inp) if len(cur_inp) == 8 else block(cur_inp[0], cur_inp[1])
            else:
                out_quant = block(cur_inp)

            err = loss_func(out_quant, cur_out, cur_grad)
            err.backward()
            if multi_gpu:
                sync_grads(opt_params)
            optimizer.step()
            if scheduler:
                scheduler.step()
    else:
        logger.warning(f"[BlockRecon] optimizer is None — skipping optimization loop for {block_name}")

    logger.debug(f"[BlockRecon] Optimization loop done in {time.time()-t_opt:.1f}s")
    block = block.to(device, torch.float16)
    torch.cuda.empty_cache()

    # Finalize — hard rounding
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
            if module.split != 0:
                module.weight_quantizer_0.soft_targets = False

    if not include_act_func:
        block.activation_function = org_act_func

    logger.info(f"[BlockRecon] Done: {block_name}  total={time.time()-t_start:.1f}s")


class LossFunction:
    def __init__(self, block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):
        self.block      = block
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
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals  = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (
                        1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError(f"Unknown round_loss: {self.round_loss}")

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info(f"[BlockRecon] iter={self.count}  "
                        f"total={float(total_loss):.4f}  "
                        f"rec={float(rec_loss):.4f}  "
                        f"round={float(round_loss):.4f}  "
                        f"b={b:.3f}")
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2,
                 start_b: int = 10, end_b: int = 2):
        self.t_max       = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b     = start_b
        self.end_b       = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))