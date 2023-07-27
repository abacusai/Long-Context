# Copyright © 2023 Abacus.AI. All rights reserved.

from numpy import pi as PI
import torch
from transformers.models.llama import modeling_llama


def truncate_frequency(f, t, low, z):
    ft = torch.where(f < t, low, f)
    ft = torch.where(f < t / z, 0, ft)
    return ft


def batch_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids = position_ids.squeeze(0)
    cos = cos[:, :, position_ids, :]  # [bs, 1, seq_len, dim]
    sin = sin[:, :, position_ids, :]  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (modeling_llama.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_llama.rotate_half(k) * sin)
    return q_embed, k_embed


class ScaledLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, scale: float = 1.0, scale_power: float = 0.0, truncate: int = 0, randomize: bool = False, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.scale = scale
        self.scale_power = scale_power
        self.randomize = randomize
        freq_2pi = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        if self.scale > 1.0:
            freq_2pi /= self.scale
        if self.scale_power > 0:
            freq_2pi *= (1.0 - torch.arange(len(freq_2pi)) / len(freq_2pi)) ** scale_power
        if (truncate or 0) > 0:
            cutoff = 2 * PI / truncate
            freq_2pi = truncate_frequency(freq_2pi, cutoff, 2 * PI / (truncate * 16), 8)
        self.register_buffer("freq_2pi", freq_2pi)
        # Build here to make `torch.jit.trace` work.
        self.cache_buffers(max_position_embeddings)
        self.rebuild_random = True

    def cache_buffers(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.freq_2pi.device, dtype=self.freq_2pi.dtype)
        pos_x_freq = torch.einsum("i,j->ij", t, self.freq_2pi)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((pos_x_freq, pos_x_freq), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def random_init_forward(self, x):
        if not self.rebuild_random:
            return

        t = torch.cumsum(torch.rand(
            (x.shape[0], x.shape[2]), device=self.freq_2pi.device, dtype=self.freq_2pi.dtype) * 2, -1)
        limit = torch.tensor(x.shape[2])
        t *= limit / torch.maximum(t[:, -1:], limit)
        pos_x_freq = torch.einsum("bi,j->bij", t, self.freq_2pi)
        emb = torch.cat((pos_x_freq, pos_x_freq), dim=-1)
        self.register_buffer('cos_random', emb.cos().to(dtype=x.dtype)[:, None, ...])
        self.register_buffer('sin_random', emb.sin().to(dtype=x.dtype)[:, None, ...])
        self.rebuild_random = False

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.randomize and self.training:
            self.random_init_forward(x)
            return self.cos_random, self.sin_random

        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.cache_buffers(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype))

    @staticmethod
    def patch(model: torch.nn.Module, scale: float = 1.0, scale_power: float = 0.0, truncate: int = 0, randomize: bool = False, config=None):
        if config is None:
            config = model.config
        model = getattr(model, 'base_model', model)
        dim = config.hidden_size / config.num_attention_heads
        rotary_emb = ScaledLlamaRotaryEmbedding(
            dim,
            scale=scale,
            scale_power=scale_power,
            truncate=truncate,
            randomize=randomize,
            max_position_embeddings=config.max_position_embeddings)
        for decoder in model.layers:
            assert hasattr(decoder, 'self_attn') and hasattr(decoder.self_attn, 'rotary_emb')
            assert decoder.self_attn.rotary_emb.inv_freq.shape == rotary_emb.freq_2pi.shape
            decoder.self_attn.rotary_emb = rotary_emb
        if randomize:
            try:
                from fastchat.train import llama_flash_attn_monkey_patch
                llama_flash_attn_monkey_patch.apply_rotary_pos_emb = batch_apply_rotary_pos_emb
            except ImportError:
                pass
            modeling_llama.saved_apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
            modeling_llama.apply_rotary_pos_emb = batch_apply_rotary_pos_emb

            def randomize_hook(module: torch.nn.Module, _: tuple):
                # Need kwargs to do this without the lazy caching with rebuild_random
                if module.training:
                    rotary_emb.rebuild_random = True
            rotary_emb.random_init_hook = model.register_forward_pre_hook(randomize_hook)
        return rotary_emb
