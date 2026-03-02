import math
import gc
import logging
import torch
import torch.nn as nn
import transformers
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = logging.getLogger(__name__)


def _tstr(t):
    """Format a tensor or scalar for debug logging without printing the full array."""
    if torch.is_tensor(t):
        if t.numel() == 1:
            return f"{t.item():.4e}"
        return f"min={t.min():.4e} max={t.max():.4e} shape={tuple(t.shape)}"
    return str(t)


def quantize_gptq(x, scale, zero, maxq, channel_group, quant_type="int", q_min=0):
    """
    Quantize tensor x given scale/zero/maxq parameters.

    :param x:             input tensor (weight column or group)
    :param scale:         quantization step size
    :param zero:          zero-point offset
    :param maxq:          upper integer clamp bound (e.g. 15 for INT4)
    :param channel_group: number of channels per group (>1 for group quant)
    :param quant_type:    reserved for future non-int types
    :param q_min:         lower integer clamp bound.
                          0 for asymmetric, -(2^(n-1)) for symmetric INT.
                          IMPORTANT: hardcoding 0 here is wrong for symmetric
                          quantization — always pass this explicitly.
    """
    if maxq < 0:
        # Binary / ternary special case
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero

    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, \
            "only support 2D input when using multiple channel groups"
        x = x.reshape((int(x.shape[0] / channel_group), -1))

    q = torch.clamp(torch.round(x / scale) + zero, q_min, maxq)
    q = scale * (q - zero)
    return q.reshape(shape)


class Quantizer_GPTQ(nn.Module):
    """
    Standalone GPTQ quantizer using find_params-style scale/zero discovery.
    Used when the layer's weight_quantizer is NOT a UniformAffineQuantizer
    (e.g. when GPTQ is configured independently of the qdiff stack).
    """
    def __init__(self, shape=1):
        super(Quantizer_GPTQ, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, channel_group=1, sym=True,
                  mse=False, norm=2.4, grid=100, maxshrink=0.8,
                  clip_ratio=1.0, quant_type="int"):
        self.maxq          = torch.tensor(2 ** bits - 1)
        self.perchannel    = perchannel
        self.channel_group = channel_group
        self.sym           = sym
        self.mse           = mse
        self.norm          = norm
        self.grid          = grid
        self.maxshrink     = maxshrink
        self.clip_ratio    = clip_ratio
        self.quant_type    = quant_type
        logger.debug(f"Quantizer_GPTQ configured: bits={bits} perchannel={perchannel} "
                     f"sym={sym} clip_ratio={clip_ratio} quant_type={quant_type}")

    def find_params(self, x, weight=False):
        dev   = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape

        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.channel_group > 1:
                    x = x.reshape(int(shape[0] / self.channel_group), -1)
            else:
                if len(shape) == 4: x = x.permute([1, 0, 2, 3]).flatten(1)
                if len(shape) == 3: x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2: x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp  = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp  = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero  = xmin
        else:
            self.scale = (xmax - xmin) * self.clip_ratio / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            tmp        = shape[0] if weight else (shape[1] if len(shape) != 3 else shape[2])
            self.scale = self.scale.repeat(tmp)
            self.zero  = self.zero.repeat(tmp)

        if weight:
            shape      = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero  = self.zero.reshape(shape)
            return

        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero  = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero  = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero  = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            # Quantizer_GPTQ is always asymmetric (q_min=0 correct here)
            return quantize_gptq(x, self.scale, self.zero, self.maxq,
                                  self.channel_group, self.quant_type, q_min=0)
        return x

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:
    def __init__(self, layer):
        self.layer    = layer
        self.dev      = self.layer.weight.device
        W             = self.layer.weight.data.clone()

        import torch.nn.functional as F
        # Flatten Conv weights the same way add_batch does — must be consistent
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(self.layer, QuantModule) and self.layer.fwd_func in (F.conv2d, F.conv1d):
            W = W.flatten(1)
        elif isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows     = W.shape[0]
        self.columns  = W.shape[1]
        self.H        = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.n_nonout = W.shape[1]
        del W
        logger.debug(f"GPTQ init: layer={type(layer).__name__}  "
                     f"rows={self.rows}  columns={self.columns}")

    def add_batch(self, inp, out):
        """Accumulate Hessian from one batch of layer inputs."""
        if inp is None or inp.numel() == 0:
            logger.warning("add_batch: received empty input — skipping")
            return

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        # Determine the true underlying op — critical for QuantModule wrappers.
        # A QuantModule can wrap either Linear or Conv2d, and we must NOT use
        # isinstance(QuantModule) to assume Linear: pos_embed.proj is a Conv2d
        # wrapped in QuantModule, and calling .t() on its 4D input crashes.
        import torch.nn.functional as F
        layer = self.layer
        is_conv = (
            isinstance(layer, nn.Conv2d)
            or (isinstance(layer, QuantModule) and layer.fwd_func in (F.conv2d, F.conv1d))
        )
        is_linear = (
            isinstance(layer, (nn.Linear, transformers.Conv1D))
            or (isinstance(layer, QuantModule) and layer.fwd_func is F.linear)
        )

        if is_linear:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        elif is_conv:
            # kernel_size is NOT stored in QuantModule.fwd_kwargs — only stride,
            # padding, dilation, groups are. Read kernel_size from weight shape
            # directly: Conv2d weight is (out_ch, in_ch/groups, kH, kW).
            # This works for both nn.Conv2d and QuantModule wrapping Conv2d.
            w = layer.weight
            kernel_size = (w.shape[2], w.shape[3]) if w.dim() == 4 else (w.shape[2],)
            if isinstance(layer, QuantModule):
                kw = layer.fwd_kwargs
                unfold = nn.Unfold(
                    kernel_size=kernel_size,
                    dilation=kw.get('dilation', (1, 1)),
                    padding=kw.get('padding', (0, 0)),
                    stride=kw.get('stride', (1, 1)))
            else:
                unfold = nn.Unfold(
                    kernel_size, dilation=layer.dilation,
                    padding=layer.padding, stride=layer.stride)
            inp = unfold(inp).permute([1, 0, 2]).flatten(1)
        else:
            logger.warning(f"add_batch: unrecognized layer type {type(layer).__name__} "
                           "— skipping Hessian update")
            return

        self.H        *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp            = math.sqrt(2 / self.nsamples) * inp.float()
        self.H        += inp.matmul(inp.t())

    def _resolve_quant_params(self, W_group):
        """
        Extract (scale, zero, q_min, q_max, channel_group, quant_type) from
        whichever quantizer type is attached to this layer, initializing it
        from W_group if not yet ready.

        Supports:
          - UniformAffineQuantizer (UAQ): qdiff stack quantizer.
            Uses .delta / .zero_point / .q_min / .q_max.
            Calls init_quantization_scale() if not yet inited.
          - Quantizer_GPTQ: standalone quantizer.
            Uses .scale / .zero / .maxq.
            Calls find_params() if not yet ready.

        BUG FIX vs uploaded version: uses isinstance() not hasattr('delta').
        hasattr is fragile — delta=None exists on UAQ before init, and could
        accidentally exist on Quantizer_GPTQ.

        BUG FIX: handles scalar (per-tensor) UAQ delta safely.
        Only reshapes scale/zero when ndim > 1 (per-channel).
        Scalar delta (ndim==0 or 1) must NOT be reshaped to (-1,1) as that
        would produce shape (1,1) which may broadcast incorrectly against
        w.unsqueeze(1) of shape (rows, 1).
        """
        q = self.quantizer

        if isinstance(q, UniformAffineQuantizer):
            if not q.inited:
                d, z     = q.init_quantization_scale(
                    W_group, channel_wise=q.channel_wise)
                q.delta      = d
                q.zero_point = z
                q.inited     = True
                logger.debug(f"  UAQ init: delta=[{_tstr(d)}]  zero_point=[{_tstr(z)}]")

            scale  = q.delta
            zero   = q.zero_point
            q_min  = q.q_min
            q_max  = q.q_max
            ch_grp = 1        # UAQ handles group quant internally
            qtype  = "int"

            # Per-channel: reshape for column-wise broadcast.
            # Per-tensor (scalar/0-d): leave as-is — broadcasts naturally.
            if torch.is_tensor(scale) and scale.ndim > 1:
                scale = scale.reshape(-1, 1)
            if torch.is_tensor(zero) and zero.ndim > 1:
                zero  = zero.reshape(-1, 1)

            return scale, zero, q_min, q_max, ch_grp, qtype

        elif isinstance(q, Quantizer_GPTQ):
            if not q.ready():
                q.find_params(W_group, weight=True)
                logger.debug(f"  Quantizer_GPTQ init: scale=[{_tstr(q.scale)}]")

            # Quantizer_GPTQ encodes symmetric via zero=(maxq+1)/2 offset,
            # so q_min=0 is still correct for the clamp range here.
            return (q.scale, q.zero, 0, int(q.maxq.item()),
                    q.channel_group, q.quant_type)

        else:
            raise TypeError(
                f"GPTQ.quantizer is {type(q).__name__} — "
                "expected UniformAffineQuantizer or Quantizer_GPTQ.")

    def _refresh_group_params(self, W_group):
        """Re-compute scale/zero for a new column group (called at group boundaries)."""
        q = self.quantizer
        if isinstance(q, UniformAffineQuantizer):
            d, z         = q.init_quantization_scale(
                W_group, channel_wise=q.channel_wise)
            q.delta      = d
            q.zero_point = z
            logger.debug(f"  UAQ group refresh: delta=[{_tstr(d)}]")
        elif isinstance(q, Quantizer_GPTQ):
            q.find_params(W_group, weight=True)
            logger.debug(f"  Quantizer_GPTQ group refresh: scale=[{_tstr(q.scale)}]")
        else:
            raise TypeError(f"Unsupported quantizer type: {type(q).__name__}")

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1):
        """
        Run GPTQ quantization on self.layer using the accumulated Hessian.

        :param blocksize:  number of weight columns processed per outer block
        :param percdamp:   Hessian damping factor (adds percdamp * mean(diag(H)) to diagonal)
        :param groupsize:  columns per quantization group (-1 = whole row = no grouping)
        """
        if self.nsamples == 0:
            logger.warning("fasterquant: nsamples=0 — Hessian is zero, "
                           "skipping this layer to avoid NaN weights")
            return

        logger.debug(f"fasterquant: rows={self.rows} cols={self.columns} "
                     f"blocksize={blocksize} percdamp={percdamp} "
                     f"groupsize={groupsize} nsamples={self.nsamples}")

        import torch.nn.functional as F
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(self.layer, QuantModule) and self.layer.fwd_func in (F.conv2d, F.conv1d):
            W = W.flatten(1)
        elif isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Initialize quantizer from the full weight slab
        scale, zero, q_min, q_max, ch_grp, qtype = self._resolve_quant_params(
            W[:, :self.n_nonout])

        # Clone and immediately free Hessian to recover VRAM
        H = self.H.clone()
        del self.H

        dead   = torch.diag(H) == 0
        n_dead = dead.sum().item()
        if n_dead > 0:
            logger.warning(f"  {n_dead}/{self.columns} dead Hessian columns — "
                           "those weights will be zeroed")
        H[dead, dead] = 1
        W[:, dead]    = 0

        Losses = torch.zeros_like(W)
        Q      = torch.zeros_like(W)

        damp           = percdamp * torch.mean(torch.diag(H))
        diag           = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        logger.debug(f"  Hessian damp: {damp.item():.6f}")

        H    = torch.linalg.cholesky(H)
        H    = torch.cholesky_inverse(H)
        H    = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        total_loss = 0.0
        for i1 in range(0, self.n_nonout, blocksize):
            i2    = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1      = W[:, i1:i2].clone()
            Q1      = torch.zeros_like(W1)
            Err1    = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1   = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Per-group scale update at every group boundary
                if groupsize > 0 and (i1 + i) % groupsize == 0:
                    W_group = W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)]
                    self._refresh_group_params(W_group)
                    scale, zero, q_min, q_max, ch_grp, qtype = \
                        self._resolve_quant_params(W_group)

                q = quantize_gptq(
                    w.unsqueeze(1),
                    scale, zero, q_max, ch_grp, qtype,
                    q_min=q_min    # FIX: was hardcoded 0, wrong for symmetric INT
                ).flatten()

                Q1[:, i]      = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1          = (w - q) / d
                W1[:, i:]    -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i]    = err1

            Q[:, i1:i2]      = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:]       -= Err1.matmul(Hinv[i1:i2, i2:])
            total_loss       += Losses1.sum().item()

        torch.cuda.synchronize()
        n_weights = self.rows * self.columns
        logger.debug(f"  GPTQ done: total_loss={total_loss:.6f}  "
                     f"mean_per_weight={total_loss/n_weights:.6f}")

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype)
        if hasattr(self.layer, "org_weight"):
            self.layer.org_weight.data = self.layer.weight.data.clone()

        del H, Losses, W, Q   # Q was absent from the original del

    def free(self):
        """Release Hessian and clear CUDA cache. Safe to call even after fasterquant."""
        if hasattr(self, 'H') and self.H is not None:
            del self.H
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPTQ.free() complete")
