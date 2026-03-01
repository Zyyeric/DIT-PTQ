# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn
import torch
from torch.autograd import Function

# For FP adaround
class FloorStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad

class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class StochasticRoundSTE(Function):
    @staticmethod
    def forward(ctx, x):
        # Sample noise between [0, 1)
        noise = torch.rand_like(x)
        return torch.floor(x + noise)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class ScaleGradient(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None


class EWGSFunctional(Function):
    """
    x_in: float input
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """

    @staticmethod
    def forward(ctx, x_in, scaling_factor):
        x_int = torch.round(x_in)
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in - x_int)
        return x_int

    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g) * diff
        return g * scale, None, None


class StackSigmoidFunctional(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Apply round to nearest in the forward pass
        ctx.save_for_backward(x, alpha)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        sig_min = torch.sigmoid(alpha / 2)
        sig_scale = 1 - 2 * sig_min
        x_base = torch.floor(x).detach()
        x_rest = x - x_base - 0.5
        stacked_sigmoid_grad = (
            torch.sigmoid(x_rest * -alpha)
            * (1 - torch.sigmoid(x_rest * -alpha))
            * -alpha
            / sig_scale
        )
        return stacked_sigmoid_grad * grad_output, None


class StochasticRounding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return stochastic_round_ste_func(x)
        else:
            return round_ste_func(x)


round_ste_func = RoundStraightThrough.apply
stacked_sigmoid_func = StackSigmoidFunctional.apply
scale_grad_func = ScaleGradient.apply
stochastic_round_ste_func = StochasticRoundSTE.apply
ewgs_func = EWGSFunctional.apply
floor_ste_func = FloorStraightThrough.apply
