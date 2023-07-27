# Copyright © 2023 Abacus.AI. All rights reserved.

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig


def fixed_pos_embedding(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, dtype=torch.float32) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float32), inv_freq)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, head_dim, max_position_embeddings, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.max_seq_len_cached = max_position_embeddings
        self.cached = False

    def set_scale_base(self, new_base):
        self.scale_base = new_base
        self.cached = False

    def cache_buffers(self, length: int):
        self.register_buffer(
            "scale", (torch.arange(0, self.head_dim, 2, dtype=torch.float32) + 0.4 * self.head_dim) / (1.4 * self.head_dim)
        )
        # The reason for not doing a simple [0, length) is because of float16 limitations.
        min_pos = -length // 2
        max_pos = length + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(length, self.head_dim // 2)
        self.register_buffer('scale_cached', scale, persistent=False)
        self.register_buffer('inv_scale_cached', 1.0 / scale, persistent=False)
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
        self.cached = True

    def forward(self, x: torch.Tensor, offset, downscale=False):
        with torch.autocast('cuda', enabled=False), torch.device(x.device):
            length = offset + x.shape[2]
            if not self.cached:
                self.cache_buffers(self.max_seq_len_cached)
            # It is unsafe to grow the buffers after allocation due to the offset issue.
            if length > self.max_seq_len_cached:
                raise NotImplementedError('Cannot increase buffer after initialization.')
                # self.cache_buffers(length)

            scale = self.inv_scale_cached if downscale else self.scale_cached
            cos = self.cos_cached
            sin = self.sin_cached
            if scale.shape[0] > length:
                scale = scale[offset:length]
                sin = sin[offset:length]
                cos = cos[offset:length]

            return apply_rotary_pos_emb(x, sin, cos, scale)


class LlamaXPosAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, attn: LlamaAttention, xpos: XPOS):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.xpos = xpos

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset

        with torch.autocast('cuda', enabled=False):
            key_states = self.xpos(key_states.to(torch.float32), offset, downscale=True)
            query_states = self.xpos(query_states.to(torch.float32), offset, downscale=False)
            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # TODO(siddartha): It seems like it would be a lot more numerically stable to do the scaling
            # as part of this operation. The matmul is doing a dot head_dim vectors. The function above
            # is a kernel with K(q_i, k_j, c_k) = ((0.4 + (k + 1) / dim) / 1.4)^|i - j|
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @staticmethod
    def patch(model, config=None, scale_base=512):
        if config is None:
            config = model.config
        model = getattr(model, 'base_model', model)
        dim = config.hidden_size / config.num_attention_heads
        xpos = XPOS(dim, config.max_position_embeddings, scale_base=scale_base)
        for decoder in model.layers:
            assert hasattr(decoder, 'self_attn')
            decoder.self_attn = LlamaXPosAttention(config, decoder.self_attn, xpos)
        return xpos
