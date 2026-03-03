"""
GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

Implementation of the GPTQ algorithm (Frantar et al., 2023) for INT weight quantization.
This replaces BRECQ+AdaRound for the INT4 path in the DIT-PTQ comparison.

Algorithm:
    For each layer:
    1. Collect Hessian H = 2 * X^T X from calibration data
    2. Apply dampening: H += damp * I
    3. Compute H_inv via Cholesky decomposition
    4. For each column (or group of columns in a block):
       a. Quantize weight column using the existing UniformAffineQuantizer (INT mode)
       b. Compute quantization error
       c. Compensate remaining columns using H_inv

Reference: https://arxiv.org/abs/2210.17323
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Union
from tqdm import tqdm

from qdiff.quant_layer import QuantModule, UniformAffineQuantizer
from qdiff.quant_block import BaseQuantBlock
from qdiff.quant_model import QuantModel

logger = logging.getLogger(__name__)


class GPTQ:
    """
    GPTQ quantizer for a single QuantModule (nn.Linear or nn.Conv2d wrapped).
    
    Collects Hessian information from calibration data, then performs
    column-by-column weight quantization with error compensation.
    """

    def __init__(self, layer: QuantModule):
        self.layer = layer
        W = layer.org_weight.data.clone()
        # Flatten conv weights to 2D: (out_features, in_features)
        if len(W.shape) == 4:  # Conv2d
            self.rows = W.shape[0]
            self.columns = W.shape[1] * W.shape[2] * W.shape[3]
        elif len(W.shape) == 2:  # Linear
            self.rows = W.shape[0]
            self.columns = W.shape[1]
        else:
            raise ValueError(f"Unsupported weight shape: {W.shape}")

        self.H = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float32)
        self.nsamples = 0
        self.original_shape = W.shape
        
        # Store conv metadata for proper input unfolding in add_batch
        self.is_conv2d = (len(W.shape) == 4)
        if self.is_conv2d:
            self.kernel_size = (W.shape[2], W.shape[3])
            # Get conv parameters from the QuantModule
            fwd_kwargs = layer.fwd_kwargs
            self.stride = fwd_kwargs.get('stride', (1, 1))
            self.padding = fwd_kwargs.get('padding', (0, 0))
            self.dilation = fwd_kwargs.get('dilation', (1, 1))

    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate Hessian H = X^T X from a batch of input activations.
        
        :param inp: input activations to this layer, shape (batch, ..., in_features)
        """
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.detach().float()
        
        # Flatten to 2D: (total_tokens, in_features)
        if self.is_conv2d:
            # Conv2d input: (batch, C_in, H, W) → unfold to patches
            # Each patch has C_in * kH * kW features, matching self.columns
            inp = torch.nn.functional.unfold(
                inp, self.kernel_size, 
                dilation=self.dilation, padding=self.padding, stride=self.stride
            )  # → (batch, C_in*kH*kW, n_patches)
            inp = inp.permute(0, 2, 1)  # → (batch, n_patches, C_in*kH*kW)
            inp = inp.reshape(-1, inp.shape[-1])  # → (total_patches, columns)
        elif len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        elif len(inp.shape) == 2:
            pass  # Already (batch, in_features)
        
        n = inp.shape[0]
        
        # H += X^T X (accumulated, normalized later)
        self.H = self.H.to(inp.device)
        self.H += inp.t() @ inp
        self.nsamples += n
        if self.nsamples % 100 < n:
            logger.debug("  Hessian accumulation: %d samples collected so far", self.nsamples)

    def _quantize_column(self, w_col: torch.Tensor, weight_quantizer: UniformAffineQuantizer):
        """
        Quantize a single weight column using the weight quantizer.
        
        The quantizer's init_quantization_scale expects the weight in a shape that
        can be channel-wise processed. For per-column quantization, we pass a 2D
        tensor with shape (out_features, 1) and use per-tensor quantization.
        
        :param w_col: weight column, shape (out_features,)
        :param weight_quantizer: the UniformAffineQuantizer to use
        :return: quantized column, shape (out_features,)
        """
        # Quantize via round-to-nearest using the quantizer's existing delta & zero_point
        if not weight_quantizer.inited:
            # This shouldn't happen in the normal GPTQ flow since we initialize
            # the quantizer before calling quantize()
            logger.warning("Weight quantizer not initialized, doing per-column init")
            weight_quantizer(w_col.unsqueeze(1))
        
        delta = weight_quantizer.delta
        zero_point = weight_quantizer.zero_point
        n_levels = weight_quantizer.n_levels
        sym = weight_quantizer.sym
        
        # Apply quantize-dequantize to this column
        from qdiff.quant_layer import round_ste
        x_int = round_ste(w_col / delta.squeeze()) + (zero_point.squeeze() if not sym else 0)
        if sym:
            x_quant = torch.clamp(x_int, -n_levels - 1, n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, n_levels - 1)
        x_dequant = (x_quant - (zero_point.squeeze() if not sym else 0)) * delta.squeeze()
        
        return x_dequant

    def quantize(self, blocksize: int = 128, percdamp: float = 0.01, 
                 group_size: int = -1):
        """
        Run the GPTQ algorithm to quantize weights.
        
        :param blocksize: number of columns to process together
        :param percdamp: dampening percentage for Hessian diagonal
        :param group_size: group size for group quantization (-1 = per-channel)
        :return: quantized weight tensor
        """
        W = self.layer.org_weight.data.clone().float()
        W_shape = W.shape
        
        # Flatten to 2D
        if len(W_shape) == 4:
            W = W.reshape(self.rows, self.columns)
        
        device = W.device
        
        # Finalize Hessian: H = (1/n) * X^T X
        H = self.H.to(device)
        if self.nsamples > 0:
            H /= self.nsamples
        
        # Dampening
        damp = percdamp * torch.mean(torch.diag(H)).item()
        if damp < 1e-10:
            damp = 1e-6  # safety floor
        diag = torch.arange(self.columns, device=device)
        H[diag, diag] += damp
        
        # Cholesky decomposition of H_inv
        # GPTQ needs the upper Cholesky factor of H^{-1}
        try:
            H_inv = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_inv)
            H_inv = torch.linalg.cholesky(H_inv, upper=True)
        except torch.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed for layer %s, "
                         "falling back to pseudo-inverse", 
                         getattr(self.layer, 'nametag', 'unknown'))
            H_inv = torch.linalg.pinv(H)
            H_inv = torch.linalg.cholesky(H_inv + 1e-6 * torch.eye(self.columns, device=device), 
                                          upper=True)
        
        Hinv = H_inv
        
        Q = torch.zeros_like(W)
        Losses = torch.zeros(self.rows, device=device)
        
        # Get the weight quantizer and initialize it with the full weight
        weight_quantizer = self.layer.weight_quantizer
        
        # GPTQ manages column groups itself, so we must disable the quantizer's
        # internal group_quant reshape (it would double-reshape the data).
        # We also use 'max' for scale init since GPTQ handles error compensation.
        orig_group_quant = weight_quantizer.group_quant
        orig_scale_method = weight_quantizer.scale_method
        weight_quantizer.group_quant = False
        weight_quantizer.scale_method = 'max'
        
        # Initialize quantizer scale/zero_point from the full weight tensor
        # This gives us the per-channel (or per-tensor) delta and zero_point
        # NOTE: Must use plain Python bool, NOT torch.tensor(False), because
        # the quantizer checks `if self.inited is False:` — Python `is` checks
        # identity, and torch.tensor(False) is not the Python False singleton.
        weight_quantizer.inited = False
        _ = weight_quantizer(W.reshape(self.original_shape) if len(self.original_shape) == 4 else W)
        weight_quantizer.inited = True
        
        # Process columns in blocks
        num_blocks = (self.columns + blocksize - 1) // blocksize
        import time as _time
        _quant_start = _time.time()
        for block_idx, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            if block_idx % max(num_blocks // 5, 1) == 0:
                elapsed = _time.time() - _quant_start
                logger.info("  GPTQ block %d/%d (cols %d-%d/%d) | elapsed=%.1fs",
                           block_idx + 1, num_blocks, i1, i2, self.columns, elapsed)
            
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for j in range(count):
                w = W1[:, j]
                d = Hinv1[j, j]
                
                # Re-initialize quantizer for new group if needed
                if group_size > 0 and (i1 + j) % group_size == 0:
                    group_start = i1 + j
                    group_end = min(group_start + group_size, self.columns)
                    group_w = W[:, group_start:group_end]
                    # Re-init quantizer with this group's range
                    # (group_quant is already disabled, so no double-reshape)
                    weight_quantizer.inited = False
                    _ = weight_quantizer(group_w)
                    weight_quantizer.inited = True
                
                # Quantize the column directly (bypass the quantizer's forward
                # to avoid re-initialization issues — use stored delta/zero_point)
                q = self._quantize_column(w, weight_quantizer)
                
                Q1[:, j] = q
                Losses1[:, j] = (w - q) ** 2 / (d ** 2)
                
                err = (w - q) / d
                Err1[:, j] = err
                
                # Compensate remaining columns in this block
                W1[:, j:] -= err.unsqueeze(1) @ Hinv1[j, j:].unsqueeze(0)
            
            Q[:, i1:i2] = Q1
            Losses += Losses1.sum(dim=1)
            
            # Compensate remaining blocks
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        
        avg_loss = torch.sum(Losses).item() / self.rows
        logger.info("GPTQ quantization loss for %s: %.6f",
                    getattr(self.layer, 'nametag', 'unknown'), avg_loss)
        
        # Restore quantizer settings that GPTQ temporarily modified
        weight_quantizer.group_quant = orig_group_quant
        weight_quantizer.scale_method = orig_scale_method
        
        # Reshape back
        Q = Q.reshape(W_shape)
        return Q

    def free(self):
        """Free cached Hessian memory."""
        del self.H
        torch.cuda.empty_cache()


class GPTQLayerInputCapture:
    """
    Forward hook to capture layer inputs for Hessian computation.
    Registered as a forward hook: __call__(module, input, output).
    The hook receives `input` as a tuple of all positional args.
    """
    def __init__(self, gptq: GPTQ):
        self.gptq = gptq
    
    def __call__(self, module, inp, out):
        # inp is a tuple of positional arguments; first element is the activation tensor
        if isinstance(inp, tuple):
            self.gptq.add_batch(inp[0])
        else:
            self.gptq.add_batch(inp)


def gptq_quantize_model(
    qnn: QuantModel,
    cali_data,
    batch_size: int = 8,
    blocksize: int = 128,
    percdamp: float = 0.01,
    group_size: int = -1,
    cond: bool = True,
):
    """
    Apply GPTQ weight quantization to all QuantModules in a QuantModel.
    
    This replaces the BRECQ+AdaRound pipeline for INT weight quantization.
    It processes layers sequentially: for each layer, collect Hessian from calibration
    data, run GPTQ, then update the layer weights with the quantized values.
    
    After GPTQ, the quantized weights are stored as the layer's `weight.data` so that
    during inference, `QuantModule.forward()` applies the weight_quantizer on already-
    GPTQ-optimized weights (the quantizer becomes a no-op since the weights are already
    on the quantization grid).
    
    :param qnn: QuantModel wrapping the transformer
    :param cali_data: calibration data tuple (xs, ts, conds)
    :param batch_size: batch size for calibration passes
    :param blocksize: GPTQ block size (columns processed together)
    :param percdamp: dampening percentage for Hessian
    :param group_size: group quantization size (-1 for per-channel)
    :param cond: whether using conditional generation
    """
    from qdiff.utils import pixart_alpha_aca_dict
    
    device = next(qnn.parameters()).device

    # Unpack calibration data
    if cond and len(cali_data) == 3:
        cali_xs, cali_ts, cali_cs = cali_data
    elif not cond:
        cali_xs, cali_ts = cali_data[:2]
        cali_cs = None
    else:
        raise ValueError(f"Unexpected cali_data length: {len(cali_data)}")

    # Collect all QuantModules in order
    quant_modules = []
    for name, module in qnn.model.named_modules():
        if isinstance(module, QuantModule):
            quant_modules.append((name, module))
    
    logger.info("GPTQ: Found %d QuantModules to quantize", len(quant_modules))
    
    # Disable weight quantization during Hessian collection so we see the
    # true (unquantized) activations flowing through the model
    qnn.set_quant_state(False, False)
    
    # Process each layer
    for layer_idx, (layer_name, layer) in enumerate(tqdm(quant_modules, desc="GPTQ layers")):
        logger.info("GPTQ: Processing layer %d/%d: %s", 
                    layer_idx + 1, len(quant_modules), layer_name)
        
        gptq = GPTQ(layer)
        
        # Register hook to capture inputs
        hook_capture = GPTQLayerInputCapture(gptq)
        handle = layer.register_forward_hook(hook_capture)
        
        # Run calibration data through the model to collect Hessian
        qnn.eval()
        num_batches = math.ceil(cali_xs.shape[0] / batch_size)
        
        with torch.no_grad():
            n_fwd_batches = min(num_batches, 16)
            logger.info("GPTQ: Running %d calibration batches (batch_size=%d) for Hessian",
                       n_fwd_batches, batch_size)
            for i in range(n_fwd_batches):  # Cap at 16 batches for speed
                start = i * batch_size
                end = min((i + 1) * batch_size, cali_xs.shape[0])
                x_batch = cali_xs[start:end].to(device, torch.float16)
                t_batch = cali_ts[start:end].to(device)
                
                try:
                    if cali_cs is not None:
                        c_batch = cali_cs[start:end].to(device, torch.float16)
                        _ = qnn(x_batch, timestep=t_batch, 
                               encoder_hidden_states=c_batch,
                               added_cond_kwargs=pixart_alpha_aca_dict(x_batch))
                    else:
                        _ = qnn(x_batch, t_batch)
                except Exception as e:
                    logger.warning("Forward pass failed at batch %d: %s", i, str(e))
                    break
        
        handle.remove()
        logger.info("GPTQ: Hessian collection done for %s: %d samples collected", layer_name, gptq.nsamples)
        
        # Run GPTQ quantization
        if gptq.nsamples > 0:
            Q = gptq.quantize(blocksize=blocksize, percdamp=percdamp,
                             group_size=group_size)
            
            # Update BOTH weight and org_weight with the quantized values.
            # - weight.data: used by QuantModule.forward() when use_weight_quant=True
            #   (weight_quantizer(self.weight) will re-quantize, but since values are
            #   already on the grid, the re-quantization is nearly a no-op)
            # - org_weight: used by QuantModule.forward() when use_weight_quant=False
            #   (for any non-quantized inference or as the baseline)
            Q_typed = Q.to(layer.weight.dtype)
            layer.weight.data = Q_typed
            layer.org_weight.data = Q_typed.clone()
        else:
            logger.warning("GPTQ: No samples collected for layer %s, skipping", layer_name)
        
        gptq.free()
        torch.cuda.empty_cache()
    
    # Re-enable weight quantization for inference
    qnn.set_quant_state(True, False)
    
    logger.info("GPTQ quantization complete for all layers")
