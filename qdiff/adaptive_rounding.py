import logging
import torch
from torch import nn
from qdiff.quant_layer import UniformAffineQuantizer, round_ste
from qdiff.quantizers.fp8_quantizer import (
    FPQuantizer, quantize_to_fp8_ste_MM,
    quantize_to_fp8_ste_MM_soft_targets,
    quantize_to_fp8_rest_scale,
)

logger = logging.getLogger(__name__)


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer.
    Optimizes the rounding policy by reconstructing intermediate layer outputs.

    BUG FIX: init_alpha now guards against NaN via epsilon clamping when
    weight values land exactly on a grid point (rest == gamma).
    """

    def __init__(self, uaq: UniformAffineQuantizer,
                 weight_tensor: torch.Tensor,
                 round_mode: str = 'learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()

        # Copy all attributes from the source UAQ
        self.n_bits    = uaq.n_bits
        self.sym       = uaq.sym
        self.delta     = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels  = uaq.n_levels
        self.fp        = uaq.fp
        self.mantissa_bits = uaq.mantissa_bits
        self.sign_bits = 1

        # Strict Q-DiT integer bounds
        self.q_min = uaq.q_min
        self.q_max = uaq.q_max

        if hasattr(uaq, 'maxval'):
            self.maxval = uaq.maxval
        elif self.fp:
            self.maxval = uaq.delta

        # Inherit grouped quantization attributes
        if hasattr(uaq, 'group_quant'):
            self.group_quant = uaq.group_quant
            self.group_size  = uaq.group_size
        if hasattr(uaq, 'fp_biased_adaround'):
            self.fp_biased_adaround = uaq.fp_biased_adaround

        self.round_mode  = round_mode
        self.alpha       = None
        self.soft_targets = False

        # Sigmoid shaping parameters
        self.gamma = -0.1
        self.zeta  =  1.1
        self.beta  =  2 / 3

        logger.debug(f"AdaRound init: n_bits={self.n_bits} sym={self.sym} fp={self.fp} "
                     f"round_mode={round_mode} "
                     f"group_quant={getattr(self, 'group_quant', False)} "
                     f"weight_tensor shape={tuple(weight_tensor.shape)}")

        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        x_shape_old = x.shape
        is_group = getattr(self, 'group_quant', False) and x.ndim == 2

        # Handle group quantization reshaping
        if is_group:
            x = x.view(-1, self.group_size)

        if self.fp:
            x_dequant = quantize_to_fp8_ste_MM_soft_targets(
                x, self.n_bits, self.delta, self.mantissa_bits,
                self.sign_bits, self.get_soft_targets())
            if is_group:
                x_dequant = x_dequant.view(x_shape_old)
            return x_dequant

        # Standard INT adaptive rounding
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            x_int = x_floor + self.get_soft_targets()
        else:
            x_int = torch.round(x / self.delta)

        x_quant   = torch.clamp(x_int + self.zero_point, self.q_min, self.q_max)
        x_float_q = (x_quant - self.zero_point) * self.delta

        if is_group:
            x_float_q = x_float_q.view(x_shape_old)
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma,
            0, 1)

    def init_alpha(self, x: torch.Tensor):
        is_group = getattr(self, 'group_quant', False) and x.ndim == 2
        if is_group:
            x = x.view(-1, self.group_size)

        if self.fp:
            rest, self.scale = quantize_to_fp8_rest_scale(
                x, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits)

            # BUG FIX: clamp rest to avoid log(0) or log(negative) → NaN
            eps  = 1e-6
            rest = rest.clamp(self.gamma + eps, self.zeta - eps)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)

            if getattr(self, 'fp_biased_adaround', False):
                alpha = self.scale * alpha
            self.alpha = nn.Parameter(alpha)

            n_nan = torch.isnan(alpha).sum().item()
            if n_nan > 0:
                logger.warning(f"init_alpha (FP): {n_nan} NaN values after init — "
                               "check delta/scale values")
            logger.debug(f"init_alpha FP: alpha range=[{alpha.min():.4f}, {alpha.max():.4f}]")
            return

        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.delta) - x_floor   # fractional part in [0, 1)

            # BUG FIX: clamp rest away from gamma (=-0.1) to avoid log domain error.
            # When rest == gamma exactly, the argument to log() becomes 0 → -inf → NaN alpha.
            # This happens when a weight is exactly representable at integer multiples of delta.
            eps  = 1e-6
            rest = rest.clamp(self.gamma + eps, self.zeta - eps)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)

            n_nan = torch.isnan(alpha).sum().item()
            if n_nan > 0:
                logger.warning(f"init_alpha (INT): {n_nan} NaN values after epsilon clamp — "
                               f"delta={self.delta}")
            logger.debug(f"init_alpha INT: rest range=[{rest.min():.4f}, {rest.max():.4f}]  "
                         f"alpha range=[{alpha.min():.4f}, {alpha.max():.4f}]")
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError(f"Unsupported round_mode: {self.round_mode}")