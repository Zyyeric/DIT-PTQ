import math
import gc
import logging
import torch
import torch.nn as nn
import transformers
from qdiff.quant_layer import QuantModule

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = logging.getLogger(__name__)


def quantize_gptq(x, scale, zero, maxq, channel_group, quant_type="int", q_min=0):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, "only support 2D input when using multiple channel groups"
        x = x.reshape((int(x.shape[0] / channel_group), -1))

    q = torch.clamp(torch.round(x / scale) + zero, q_min, maxq)
    q = scale * (q - zero)
    return q.reshape(shape)


class Quantizer_GPTQ(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer_GPTQ, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, channel_group=1, sym=True,
                  mse=False, norm=2.4, grid=100, maxshrink=0.8,
                  clip_ratio=1.0, quant_type="int"):
        self.maxq         = torch.tensor(2 ** bits - 1)
        self.perchannel   = perchannel
        self.channel_group = channel_group
        self.sym          = sym
        self.mse          = mse
        self.norm         = norm
        self.grid         = grid
        self.maxshrink    = maxshrink
        self.clip_ratio   = clip_ratio
        self.quant_type   = quant_type
        logger.debug(f"Quantizer_GPTQ configured: bits={bits} perchannel={perchannel} "
                     f"sym={sym} clip_ratio={clip_ratio} quant_type={quant_type}")

    def find_params(self, x, weight=False):
        dev = x.device
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
            return quantize_gptq(x, self.scale, self.zero, self.maxq,
                                  self.channel_group, self.quant_type)
        return x

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:
    def __init__(self, layer):
        self.layer   = layer
        self.dev     = self.layer.weight.device
        W            = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):         W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()

        self.rows    = W.shape[0]
        self.columns = W.shape[1]
        self.H       = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.n_nonout = W.shape[1]
        del W
        logger.debug(f"GPTQ init: rows={self.rows} columns={self.columns}")

    def add_batch(self, inp, out):
        # BUG FIX: guard against empty inputs
        if inp is None or inp.numel() == 0:
            logger.warning("add_batch received empty input — skipping")
            return

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, (nn.Linear, transformers.Conv1D, QuantModule)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size, dilation=self.layer.dilation,
                padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp).permute([1, 0, 2]).flatten(1)

        self.H        *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp            = math.sqrt(2 / self.nsamples) * inp.float()
        self.H        += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1):
        # BUG FIX: guard against zero samples — would produce NaN Hessian
        if self.nsamples == 0:
            logger.warning("fasterquant called with nsamples=0 — Hessian is all zeros, "
                           "skipping quantization for this layer")
            return

        logger.debug(f"fasterquant: rows={self.rows} columns={self.columns} "
                     f"blocksize={blocksize} percdamp={percdamp} groupsize={groupsize} "
                     f"nsamples={self.nsamples}")

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):         W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): W = W.t()
        W = W.float()

        is_uaq = hasattr(self.quantizer, 'delta')
        if is_uaq:
            if not self.quantizer.inited:
                d, z = self.quantizer.init_quantization_scale(
                    W[:, :self.n_nonout], channel_wise=self.quantizer.channel_wise)
                self.quantizer.delta = d
                self.quantizer.zero_point = z
                self.quantizer.inited = True
        elif not self.quantizer.ready():
             self.quantizer.find_params(W[:, :self.n_nonout], weight=True)
             logger.debug(f"  Scale range: [{self.quantizer.scale.min():.4f}, {self.quantizer.scale.max():.4f}]")

        H    = self.H.clone()
        del self.H  # free immediately — BUG FIX: was kept alive until free()

        dead = torch.diag(H) == 0
        n_dead = dead.sum().item()
        if n_dead > 0:
            logger.debug(f"  Dead columns (zero diagonal): {n_dead}/{self.columns}")
        H[dead, dead] = 1
        W[:, dead]    = 0

        Losses = torch.zeros_like(W)
        Q      = torch.zeros_like(W)

        damp       = percdamp * torch.mean(torch.diag(H))
        diag       = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        logger.debug(f"  Damping value: {damp.item():.6f}")

        H    = torch.linalg.cholesky(H)
        H    = torch.cholesky_inverse(H)
        H    = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        total_loss = 0.0
        for i1 in range(0, self.n_nonout, blocksize):
            i2    = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1    = W[:, i1:i2].clone()
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize > 0 and (i1 + i) % groupsize == 0:
                    if is_uaq:
                        d_new, z_new = self.quantizer.init_quantization_scale(
                            W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)],
                            channel_wise=self.quantizer.channel_wise)
                        self.quantizer.delta = d_new
                        self.quantizer.zero_point = z_new
                    else:
                        self.quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + groupsize), self.n_nonout)],
                            weight=True)

                if is_uaq:
                    scale = self.quantizer.delta
                    zero = self.quantizer.zero_point
                    maxq = self.quantizer.q_max
                    q_min = self.quantizer.q_min
                    channel_group = 1
                    quant_type = "int"
                    if scale.ndim > 1: scale = scale.reshape(-1, 1)
                    if zero.ndim > 1: zero = zero.reshape(-1, 1)
                else:
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero
                    maxq = self.quantizer.maxq
                    q_min = 0
                    channel_group = self.quantizer.channel_group
                    quant_type = self.quantizer.quant_type

                q = quantize_gptq(
                    w.unsqueeze(1),
                    scale, zero,
                    maxq, channel_group,
                    quant_type, q_min=q_min).flatten()

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
        logger.debug(f"  Total quant loss (sum): {total_loss:.6f}")

        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # Overwrite weight — BUG FIX: also clean up Q after assignment
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if hasattr(self.layer, "org_weight"):
            self.layer.org_weight.data = self.layer.weight.data.clone()

        del H, Losses, W, Q  # BUG FIX: Q was not deleted in original

    def free(self):
        # BUG FIX: original set self.H = None after del self.H in fasterquant
        # Now safe — H is already deleted in fasterquant, this is just cleanup
        if hasattr(self, 'H') and self.H is not None:
            del self.H
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPTQ.free() called — cache cleared")