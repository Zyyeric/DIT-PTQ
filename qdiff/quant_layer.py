import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from qdiff.quantizers.fp8_quantizer import quantize_to_fp8_ste_MM

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding."""
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """L_p norm loss."""
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    Asymmetric uniform affine quantizer with straight-through gradient estimator.
    Supports per-channel, per-group, FP8, and integer quantization modes.
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False,
                 channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False,
                 fp: bool = False, mantissa_bits: int = None,
                 weight_group_size: int = None, fp_biased_adaround: bool = True,
                 online_act_quant: bool = False, group_quant: bool = False,
                 clip_ratio: float = 1.0, **kwargs):
        super(UniformAffineQuantizer, self).__init__()

        self.sym          = symmetric
        self.n_bits       = n_bits
        self.clip_ratio   = clip_ratio

        # n_levels kept for AdaRound compatibility
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

        # Integer clamp bounds
        if self.sym:
            self.q_max =  (2 ** (self.n_bits - 1) - 1)
            self.q_min = -(2 ** (self.n_bits - 1))
        else:
            self.q_max = (2 ** self.n_bits) - 1
            self.q_min = 0

        self.delta          = None
        self.zero_point     = None
        self.inited         = False
        self.leaf_param     = leaf_param
        self.channel_wise   = channel_wise
        self.scale_method   = scale_method
        self.running_stat   = False
        self.always_zero    = always_zero
        self.fp_biased_adaround = fp_biased_adaround

        self.sign_bits = 1
        self.fp = fp
        if mantissa_bits is None:
            if   n_bits == 8: mantissa_bits = 4
            elif n_bits == 6: mantissa_bits = 2
            else:             mantissa_bits = 1
        self.mantissa_bits = torch.Tensor([float(mantissa_bits)])

        if self.leaf_param:
            self.x_min, self.x_max = None, None
        if self.scale_method == "mse":
            self.alpha_dict = {}
        elif self.scale_method in ["kmeans", "coreset"]:
            self.dist_dev_list = []
            self.c_k = []
        elif self.scale_method == "kmeans_all":
            self.k_all = None
        elif "quantile" in self.scale_method:
            self.c_k = []

        self.online_act_quant = online_act_quant
        self.dynamic_idx      = None
        self.dynamic_mantissa = None
        self.dim              = 0
        self.group_quant      = False
        self.group_size       = weight_group_size
        if not self.leaf_param:
            self.group_quant = group_quant

        logger.debug(f"UAQ init: n_bits={n_bits} sym={symmetric} channel_wise={channel_wise} "
                     f"scale_method={scale_method} fp={fp} mantissa_bits={mantissa_bits} "
                     f"group_quant={self.group_quant} group_size={weight_group_size} "
                     f"q_min={self.q_min} q_max={self.q_max}")

    def forward(self, x: torch.Tensor):
        if self.group_quant:
            x_old = x
            x = x.view(-1, self.group_size)

        if self.online_act_quant and self.leaf_param:
            return self.online_per_tensor_quant(x)

        if not self.inited:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(
                    x, self.channel_wise, print_stats=True)
                if self.fp:
                    delta = torch.abs(
                        torch.tensor([torch.max(self.x_min.abs(), self.x_max.abs())])
                    ).to(x.device)
                    mantissa_bits = self.mantissa_bits.to(x.device)
                    if self.dynamic_idx is None:
                        self.delta = nn.Parameter(delta)
                    else:
                        array = [[delta]] * self.dynamic_idx
                        self.delta = nn.Parameter(array)
                else:
                    self.delta = nn.Parameter(delta)
            else:
                if self.fp:
                    self.delta, self.zero_point = self.init_quantization_scale_fp(
                        x, self.channel_wise)
                else:
                    self.delta, self.zero_point = self.init_quantization_scale(
                        x, self.channel_wise, print_stats=True)
            self.inited = True
            logger.debug(f"UAQ inited: delta range=[{self.delta.min():.4e}, "
                         f"{self.delta.max().item():.4e}]  zero_point={self.zero_point}")

        if self.fp:
            if self.leaf_param and self.dynamic_idx is not None:
                x_dequant = quantize_to_fp8_ste_MM(
                    x, self.n_bits, self.delta[self.dynamic_idx],
                    self.dynamic_mantissa[self.dynamic_idx], self.sign_bits)
            else:
                x_dequant = quantize_to_fp8_ste_MM(
                    x, self.n_bits, self.delta, self.mantissa_bits, self.sign_bits)
            if self.group_quant:
                x_dequant = x_dequant.view(x_old.shape)
            return x_dequant

        if self.running_stat:
            self.act_momentum_update(x)

        if self.scale_method in ["kmeans", "coreset"]:
            return self.dequant_kmeans(x)
        elif self.scale_method == "kmeans_all":
            return self.k_all
        elif "quantile" in self.scale_method:
            return self.dequant_kmeans(x)

        # Standard INT forward
        x_int    = round_ste(x / self.delta) + self.zero_point
        x_quant  = torch.clamp(x_int, self.q_min, self.q_max)
        x_dequant = (x_quant - self.zero_point) * self.delta

        if self.group_quant:
            x_dequant = x_dequant.view(x_old.shape)
        return x_dequant

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert self.inited
        assert self.leaf_param
        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.q_max
        else:
            delta = ((self.x_max - self.x_min) / self.q_max
                     if not self.always_zero else self.x_max / self.q_max)
        delta = torch.clamp(delta, min=1e-8)

        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not self.always_zero else 0

        # BUG FIX: avoid creating a new Parameter every call — just update .data
        if isinstance(self.delta, nn.Parameter):
            self.delta.data = delta.to(self.delta.device)
        else:
            self.delta = nn.Parameter(delta)

    def online_per_tensor_quant(self, x: torch.Tensor):
        x_shape = x.shape
        if len(x.shape) == 3:
            Cin = x.shape[-1]
            x = x.reshape(-1, Cin)
        elif len(x.shape) == 4:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])

        x_min    = x.min(dim=-1)[0]
        x_max    = x.max(dim=-1)[0]
        x_absmax = torch.maximum(x_min.abs(), x_max.abs())
        x_dequant = quantize_to_fp8_ste_MM(
            x, self.n_bits, x_absmax, self.mantissa_bits, self.sign_bits)
        return x_dequant.reshape(x_shape)

    def init_quantization_scale_fp(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone   = x.clone().detach()
            n_channels = x_clone.shape[0]
            if   len(x.shape) == 4: x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3: x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:                   x_max = x_clone.abs().max(dim=-1)[0]
            delta      = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale_fp(
                    x_clone[c], channel_wise=False)
            if   len(x.shape) == 4: delta = delta.view(-1, 1, 1, 1); zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3: delta = delta.view(-1, 1, 1);    zero_point = zero_point.view(-1, 1, 1)
            else:                   delta = delta.view(-1, 1);        zero_point = zero_point.view(-1, 1)
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()
            x_min  = min(x.min().item(), 0)
            x_max  = max(x.max().item(), 0)
            x_max  = max(abs(x_max), abs(x_min))
            delta  = x_max
            zero_point = torch.zeros(1)
        return delta, zero_point

    def init_quantization_scale(self, x: torch.Tensor,
                                channel_wise: bool = False,
                                print_stats: bool = False):
        delta, zero_point = None, None

        if channel_wise:
            x_clone    = x.clone().detach()
            n_channels = x_clone.shape[0]
            if   len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                # BUG FIX: original used .abs().min() — wrong for asymmetric quant.
                # Signed minimum is needed to correctly compute the asymmetric range.
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]  # BUG FIX
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
                x_min = x_clone.min(dim=-1)[0]  # BUG FIX
            delta      = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(
                    x_clone[c], channel_wise=False)
            if   len(x.shape) == 4: delta = delta.view(-1, 1, 1, 1); zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3: delta = delta.view(-1, 1, 1);    zero_point = zero_point.view(-1, 1, 1)
            else:                   delta = delta.view(-1, 1);        zero_point = zero_point.view(-1, 1)

            if print_stats:
                logger.debug(f"Per-channel scale init: delta range=[{delta.min():.4e}, "
                             f"{delta.max():.4e}]  shape={tuple(delta.shape)}")
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min   = min(x.min().item(), 0)
                x_max   = max(x.max().item(), 0)
                x_absmax = max(abs(x_min), x_max)

                if self.sym:
                    if self.clip_ratio < 1.0:
                        x_absmax *= self.clip_ratio
                    delta = x_absmax / self.q_max
                else:
                    if self.clip_ratio < 1.0:
                        x_min *= self.clip_ratio
                        x_max *= self.clip_ratio
                    delta = float(x_max - x_min) / self.q_max

                delta = max(delta, 1e-8)
                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)

                if print_stats:
                    logger.debug(f"Max scale init: x_min={x_min:.4f} x_max={x_max:.4f} "
                                 f"delta={delta.item():.6e} zero_point={zero_point}")

            elif self.scale_method == 'mse':
                w_absmax   = x.abs().max().clamp(min=1e-5)
                w_min      = x.min()
                w_max      = x.max()
                best_score = 1e10
                best_min, best_max, best_absmax = w_min.clone(), w_max.clone(), w_absmax.clone()

                for i in range(100):
                    if self.sym:
                        new_max = w_absmax * (1.0 - i * 0.001)
                        scales  = new_max / self.q_max
                        base    = 0
                    else:
                        new_max = w_max * (1.0 - i * 0.001)
                        new_min = w_min * (1.0 - i * 0.001)
                        scales  = max((new_max - new_min).item(), 1e-5) / self.q_max
                        base    = round(float(-new_min / scales))
                        base    = max(min(base, self.q_max), self.q_min)

                    x_q    = (torch.clamp(torch.round(x / scales) + base,
                                          self.q_min, self.q_max) - base) * scales
                    score  = lp_loss(x, x_q, p=2.4, reduction='all')

                    if score < best_score:
                        best_score = score
                        if self.sym:
                            best_absmax = new_max
                        else:
                            best_min, best_max = new_min, new_max

                if self.sym:
                    delta      = best_absmax / self.q_max
                    zero_point = 0
                else:
                    delta      = max((best_max - best_min).item(), 1e-5) / self.q_max
                    zero_point = round(float(-best_min / delta))
                    zero_point = max(min(zero_point, self.q_max), self.q_min)

                delta = torch.tensor(delta).type_as(x)
                if print_stats:
                    logger.debug(f"MSE scale init: best_score={best_score:.6f} "
                                 f"delta={delta.item():.6e} zero_point={zero_point}")

            elif self.scale_method == 'kmeans':
                from sklearn.cluster import KMeans
                x_np  = x.clone().detach().cpu().view(1, -1).numpy()
                mykm  = KMeans(n_clusters=min(2 ** self.n_bits, x_np.shape[1]),
                               max_iter=100).fit(x_np.T)
                for i in range(x_np.shape[1]):
                    x_np[0, i] = mykm.cluster_centers_[mykm.labels_[i], :]
                x_b2t = torch.from_numpy(x_np).to(x.device, dtype=x.dtype).view(x.shape)
                self.c_k.append(x_b2t.unsqueeze(0))
                centers_dist = mykm.cluster_centers_[:, 0]
                centers_dist.sort()
                distances = [abs(centers_dist[i] - centers_dist[i - 1])
                             for i in range(1, len(centers_dist))]
                self.dist_dev_list.append(np.std(distances))
                delta, zero_point = 1, 0
                logger.debug(f"KMeans scale init: n_clusters={len(centers_dist)} "
                             f"dist_std={self.dist_dev_list[-1]:.4f}")
            else:
                raise NotImplementedError(f"Unknown scale_method: {self.scale_method}")

        return delta, zero_point

    def quantize(self, x, max, min):
        delta      = ((max - min) / self.q_max if not self.always_zero
                      else max / self.q_max)
        zero_point = ((-min / delta).round() if not self.always_zero else 0)
        x_int    = torch.round(x / delta)
        x_quant  = torch.clamp(x_int + zero_point, self.q_min, self.q_max)
        return (x_quant - zero_point) * delta

    def dequant_kmeans(self, x):
        if isinstance(self.c_k, list):
            self.c_k = torch.cat(self.c_k, dim=0)
        return self.c_k


class QuantModule(nn.Module):
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {},
                 disable_act_quant: bool = False,
                 act_quant_mode: str = 'qdiff'):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params    = act_quant_params

        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride, padding=org_module.padding,
                dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=org_module.stride, padding=org_module.padding,
                dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func   = F.linear

        self.weight    = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias     = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias     = None
            self.org_bias = None

        self.use_weight_quant  = False
        self.use_act_quant     = False
        self.act_quant_mode    = act_quant_mode
        self.disable_act_quant = disable_act_quant
        self.weight_quantizer  = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)
        self.split = 0

        self.activation_function  = StraightThrough()
        self.ignore_reconstruction = False
        self.extra_repr            = org_module.extra_repr

        if hasattr(org_module, "in_features"): self.in_features = org_module.in_features
        self.nametag   = getattr(org_module, "nametag", None)
        self.run_prints = True

        logger.debug(f"QuantModule wrapping {type(org_module).__name__}  "
                     f"weight={tuple(self.weight.shape)}  "
                     f"act_quant_mode={act_quant_mode}")

    def forward(self, input: torch.Tensor, split: int = 0):
        og_dtype = input.dtype

        if split != 0 and self.split != 0:
            assert split == self.split
        elif split != 0:
            self.split = split
            self.set_split()

        if not self.disable_act_quant and self.use_act_quant:
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    input = self.act_quantizer(input)

        if self.use_weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight   = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias   = self.org_bias

        # dtype alignment
        if weight.dtype != input.dtype:
            weight = weight.to(input.dtype)
        if bias is not None and bias.dtype != input.dtype:
            bias = bias.to(input.dtype)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)
        return out.to(og_dtype)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant    = act_quant

    def set_split(self):
        self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer.running_stat = running_stat
            if self.split != 0:
                self.act_quantizer_0.running_stat = running_stat