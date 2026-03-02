import logging
import torch
from diffusers.models.attention import Attention
from typing import Optional
from diffusers.utils import deprecate

logger = logging.getLogger(__name__)


class QuantAttnProcessor:
    """
    Drop-in replacement for Diffusers AttnProcessor / AttnProcessor2_0.
    Injects activation quantization for Q, K, V, and softmax output (W).

    Compatible with Diffusers >= 0.29.2.
    Uses the 1.0-style head_to_batch_dim path (not scaled_dot_product_attention)
    so that intermediate tensors are accessible for quantization.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecate(
                "scale", "1.0.0",
                "The `scale` argument is deprecated and will be ignored. "
                "Remove it to avoid an error in a future release.")

        original_dtype = hidden_states.dtype
        residual       = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Use head_to_batch_dim so Q/K/V are accessible for quantization
        # (scaled_dot_product_attention fuses everything and hides intermediates)
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attn.use_act_quant:
            logger.debug(f"QuantAttnProcessor: quantizing Q/K/V  "
                         f"shapes Q={tuple(query.shape)} K={tuple(key.shape)} "
                         f"V={tuple(value.shape)}")
            query = attn.act_quantizer_q(query).to(original_dtype)
            key   = attn.act_quantizer_k(key).to(original_dtype)
            value = attn.act_quantizer_v(value).to(original_dtype)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Quantize softmax output (attention weights) before matmul with V.
        # Only applied when act_quant is on and n_bits <= 8 to avoid FP overflow.
        if attn.use_act_quant and attn.act_quantizer_w.n_bits <= 8:
            logger.debug(f"QuantAttnProcessor: quantizing attention weights  "
                         f"shape={tuple(attention_probs.shape)}  "
                         f"n_bits={attn.act_quantizer_w.n_bits}")
            attention_probs = attn.act_quantizer_w(attention_probs).to(original_dtype)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states