import logging
import random
import numpy as np
import torch
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import (
    QuantBasicTransformerBlock, QuantResBlock,
    QuantQKMatMul, QuantSMVMatMul,
    QuantAttnBlock, QuantHunyuanBlock,
    QuantDiffBTB, QuantDiffRB,
)
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module,
                 weight_quant_params: dict = {},
                 act_quant_params: dict = {},
                 **kwargs):
        super().__init__()
        self.model    = model
        self.sm_abit  = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])

        # Diffusers / SDXL compatibility
        if hasattr(model, "dtype"):
            self.dtype = model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            if hasattr(model, "add_embedding"):
                self.forward = self.forward_diffusers
                logger.info("QuantModel: using forward_diffusers (Diffusers pipeline detected)")
            else:
                self.forward = model.forward
                logger.info("QuantModel: using model.forward directly")

        logger.info("QuantModel: refactoring modules...")
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        self.set_attn_weight_mantissa_bits(weight_quant_params)
        self.set_ff1_weight_mantissa_bits(weight_quant_params)
        self.set_asym_for_sm(act_quant_params)

        n_qm = sum(1 for m in self.modules() if isinstance(m, QuantModule))
        n_qb = sum(1 for m in self.modules() if isinstance(m, BaseQuantBlock))
        logger.info(f"QuantModel ready: QuantModules={n_qm}  QuantBlocks={n_qb}  "
                    f"sm_abit={self.sm_abit}")

    def set_asym_for_sm(self, act_quant_params: dict = {}):
        # ── 1. Q-DiT (INT) Logic ──────────────────────────────────────────
        if not act_quant_params.get('fp', False):
            # Recover 50% wasted resolution for Q-DiT Softmax
            logger.info("QuantModel: INT4 mode. Forcing asymmetric bounds for Softmax.")
            n = 0
            for m in self.model.modules():
                if isinstance(m, QuantDiffBTB):
                    # Loop safely over both attention blocks
                    for attn in [m.attn1, m.attn2]:
                        if attn is not None:
                            # [FIXED]: Update clamp bounds alongside symmetry!
                            attn.act_quantizer_w.sym = False
                            attn.act_quantizer_w.always_zero = True
                            attn.act_quantizer_w.q_max = (2 ** attn.act_quantizer_w.n_bits) - 1
                            attn.act_quantizer_w.q_min = 0
                    n += 1
            return  # <--- Exits here during Q-DiT runs!

        if act_quant_params.get('asym_softmax', False):
            logger.info("QuantModel: using Symmetric quantization for Softmax")
            return
        logger.info("QuantModel: using Asymmetric quantization for Softmax")
        n = 0
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.attn1.act_quantizer_w.sign_bits = 0
                if m.attn2 is not None:
                    m.attn2.act_quantizer_w.sign_bits = 0
                n += 1
        logger.debug(f"  set_asym_for_sm: modified {n} QuantDiffBTB blocks")

    def set_attn_weight_mantissa_bits(self, weight_quant_params: dict = {}):
        num = weight_quant_params.get('attn_weight_mantissa', None)
        if num is None:
            logger.info("QuantModel: no specific attention mantissa bits set")
            return
        logger.info(f"QuantModel: setting attention weight mantissa_bits={num}")
        n = 0
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.attn1.to_q.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_k.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_v.weight_quantizer.mantissa_bits[0] = num
                m.attn1.to_out[0].weight_quantizer.mantissa_bits[0] = num
                if m.attn2 is not None:
                    m.attn2.to_q.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_k.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_v.weight_quantizer.mantissa_bits[0] = num
                    m.attn2.to_out[0].weight_quantizer.mantissa_bits[0] = num
                n += 1
        logger.debug(f"  set_attn_weight_mantissa_bits: modified {n} blocks")

    def set_ff1_weight_mantissa_bits(self, weight_quant_params: dict = {}):
        num = weight_quant_params.get('ff_weight_mantissa', None)
        if num is None:
            logger.info("QuantModel: no specific FF mantissa bits set")
            return
        logger.info(f"QuantModel: setting FF first-layer weight mantissa_bits={num}")
        n = 0
        for m in self.model.modules():
            if isinstance(m, QuantDiffBTB):
                m.ff.net[0].proj.weight_quantizer.mantissa_bits[0] = num
                n += 1
        logger.debug(f"  set_ff1_weight_mantissa_bits: modified {n} blocks")

    def quant_module_refactor(self, module: nn.Module,
                              weight_quant_params: dict = {},
                              act_quant_params: dict = {}):
        """Recursively replace Conv2d/Conv1d/Linear with QuantModule."""
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name])

            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                logger.debug(f"  Replacing {child_module.nametag} with QuantModule")
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module,
                             weight_quant_params: dict = {},
                             act_quant_params: dict = {}):
        """Recursively replace known block types with their Quant equivalents."""
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                special_cls = self.specials[type(child_module)]
                logger.debug(f"  Replacing {type(child_module).__name__} → {special_cls.__name__}")
                if special_cls in [QuantBasicTransformerBlock, QuantAttnBlock,
                                   QuantDiffBTB, QuantHunyuanBlock]:
                    setattr(module, name, special_cls(
                        child_module, act_quant_params, sm_abit=self.sm_abit))
                elif special_cls == QuantSMVMatMul:
                    setattr(module, name, special_cls(
                        act_quant_params, sm_abit=self.sm_abit))
                elif special_cls == QuantQKMatMul:
                    setattr(module, name, special_cls(act_quant_params))
                else:
                    setattr(module, name, special_cls(child_module, act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

    def forward_diffusers(self,
                          latent_model_input,
                          timestep=None,
                          encoder_hidden_states=None,
                          cross_attention_kwargs=None,
                          added_cond_kwargs=None,
                          return_dict=False):
        """
        Diffusers-compatible forward. 
        NOTE: original used positional arg 't' but PixArt calls with 'timestep' kwarg —
        unified here to avoid silent mismatches.
        """
        return self.model(
            sample=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict)

    def set_running_stat(self, running_stat: bool, sm_only=False):
        n = 0
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    for q in [m.attn1.act_quantizer_q, m.attn1.act_quantizer_k,
                              m.attn1.act_quantizer_v, m.attn1.act_quantizer_w,
                              m.attn2.act_quantizer_q, m.attn2.act_quantizer_k,
                              m.attn2.act_quantizer_v, m.attn2.act_quantizer_w]:
                        q.running_stat = running_stat
                n += 1
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    for q in [m.attn1.act_quantizer_q, m.attn1.act_quantizer_k,
                              m.attn1.act_quantizer_v, m.attn1.act_quantizer_w]:
                        q.running_stat = running_stat
                    if m.attn2 is not None:
                        for q in [m.attn2.act_quantizer_q, m.attn2.act_quantizer_k,
                                  m.attn2.act_quantizer_v, m.attn2.act_quantizer_w]:
                            q.running_stat = running_stat
                n += 1
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)
        logger.debug(f"set_running_stat({running_stat}, sm_only={sm_only}): "
                     f"updated {n} attn blocks")

    def set_grad_ckpt(self, grad_ckpt: bool):
        n = 0
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                m.checkpoint = grad_ckpt
                n += 1
        logger.info(f"set_grad_ckpt({grad_ckpt}): updated {n} transformer blocks")