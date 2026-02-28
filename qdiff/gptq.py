"""
GPTQ implementation for DIT-PTQ.
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.quant_layer import QuantModule
import logging

logger = logging.getLogger(__name__)

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        
        # Handle QuantModule wrapping
        if isinstance(self.layer, QuantModule):
            if self.layer.fwd_func == F.conv2d:
                W = W.flatten(1)
            elif self.layer.fwd_func == F.conv1d:
                W = W.t()
            # Linear weights are already (out, in)
        elif isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(self.layer, nn.Linear):
            W = W.data.clone()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out=None):
        # inp: (B, ...)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        # Handle QuantModule
        is_conv2d = isinstance(self.layer, nn.Conv2d) or (isinstance(self.layer, QuantModule) and self.layer.fwd_func == F.conv2d)
        is_linear = isinstance(self.layer, nn.Linear) or (isinstance(self.layer, QuantModule) and self.layer.fwd_func == F.linear)

        if is_linear:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        if is_conv2d:
            if isinstance(self.layer, QuantModule):
                kwargs = self.layer.fwd_kwargs
                kernel_size = self.layer.weight.shape[2:]
                stride = kwargs.get('stride', 1)
                padding = kwargs.get('padding', 0)
                dilation = kwargs.get('dilation', 1)
            else:
                kernel_size = self.layer.kernel_size
                stride = self.layer.stride
                padding = self.layer.padding
                dilation = self.layer.dilation
            
            unfold = nn.Unfold(
                kernel_size,
                dilation=dilation,
                padding=padding,
                stride=stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        W = self.layer.weight.data.clone()
        is_conv2d = isinstance(self.layer, nn.Conv2d) or (isinstance(self.layer, QuantModule) and self.layer.fwd_func == F.conv2d)
        is_conv1d = isinstance(self.layer, QuantModule) and self.layer.fwd_func == F.conv1d

        if is_conv2d:
            W = W.flatten(1)
        if is_conv1d:
            W = W.t()
        
        W = W.float()
        
        if not self.quantizer.inited:
             self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = self.quantizer.quantize_gptq(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if is_conv1d:
            Q = Q.t()
            
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
