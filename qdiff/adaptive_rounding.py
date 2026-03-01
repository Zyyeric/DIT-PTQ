import torch
from torch import nn
import logging
from qdiff.quant_layer import UniformAffineQuantizer, round_ste
from qdiff.quantizers.fp8_quantizer import FPQuantizer, quantize_to_fp8_ste_MM, quantize_to_fp8_ste_MM_soft_targets, quantize_to_fp8_rest_scale

logger = logging.getLogger(__name__)


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.fp = uaq.fp
        self.mantissa_bits = uaq.mantissa_bits
        self.sign_bits = 1
        
        # === THE AGENT's FIX: Strict Q-DiT Bounds ===
        self.q_min = uaq.q_min
        self.q_max = uaq.q_max
        # ============================================
        
        if hasattr(uaq, 'maxval'):
            self.maxval = uaq.maxval

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.fp:
            # Native FP4 optimization
            x_dequant = quantize_to_fp8_ste_MM_soft_targets(
                x, self.n_bits, self.delta, self.mantissa_bits, self.sign_bits, self.get_soft_targets()
            )
            return x_dequant
        else:
            # Standard INT rounding
            x_floor = torch.floor(x / self.delta)
            if self.round_mode == 'learned_hard_sigmoid':
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = torch.round(x / self.delta)
            
            # === THE AGENT's FIX: Applied to the clamp ===
            x_quant = torch.clamp(x_int + self.zero_point, self.q_min, self.q_max)
            x_float_q = (x_quant - self.zero_point) * self.delta
            
            return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        if hasattr(self, 'group_quant') and self.group_quant == True:
            x_old = x
            x = x.view(-1, self.group_size)
            
        if self.fp == True:
            rest, self.scale = quantize_to_fp8_rest_scale(x, self.n_bits, self.maxval, self.mantissa_bits, self.sign_bits)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            if hasattr(self, 'fp_biased_adaround') and self.fp_biased_adaround == True:
                alpha = self.scale * alpha 
            self.alpha = nn.Parameter(alpha)
            return
            
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            rest = (x / self.delta) - x_floor  
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1) 
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError('Please supply a valid round_mode')