"""Multi-head Latent Attention (MLA) with a compressed KV cache.

Keys and values are reconstructed from a shared low-rank latent c_kv, so the
inference cache stores only c_kv (d_c floats) plus a small decoupled RoPE key
per token instead of full per-head K and V. This cuts KV memory by roughly
(2 * n_heads * d_head) / (d_c + d_rope) while keeping multi-head expressiveness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLAConfig:
    d_model: int = 2048
    n_heads: int = 16
    d_head: int = 128
    d_latent_kv: int = 512      # compressed KV dimension d_c
    d_latent_q: int = 1024      # compressed query dimension
    d_rope: int = 64            # decoupled rotary dimension per head
    max_seq_len: int = 32768
    rope_base: float = 10000.0


def rope_cache(seq_len: int, dim: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        h, dh, dr = cfg.n_heads, cfg.d_head, cfg.d_rope
        # Query path: down-project, norm, up-project into content + rope parts.
        self.q_down = nn.Linear(cfg.d_model, cfg.d_latent_q, bias=False)
        self.q_norm = RMSNorm(cfg.d_latent_q)
        self.q_up = nn.Linear(cfg.d_latent_q, h * (dh + dr), bias=False)
        # KV path: a single latent shared by all heads, plus one shared rope key.
        self.kv_down = nn.Linear(cfg.d_model, cfg.d_latent_kv + dr, bias=False)
        self.kv_norm = RMSNorm(cfg.d_latent_kv)
        self.k_up = nn.Linear(cfg.d_latent_kv, h * dh, bias=False)
        self.v_up = nn.Linear(cfg.d_latent_kv, h * dh, bias=False)
        self.out = nn.Linear(h * dh, cfg.d_model, bias=False)
        cos, sin = rope_cache(cfg.max_seq_len, dr, cfg.rope_base, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        h, dh, dr = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_rope
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]

        q = self.q_up(self.q_norm(self.q_down(x))).view(B, T, h, dh + dr)
        q_c, q_r = q.split([dh, dr], dim=-1)
        q_r = apply_rope(q_r, cos[:, None], sin[:, None])

        kv = self.kv_down(x)
        c_kv, k_r = kv.split([self.cfg.d_latent_kv, dr], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_r = apply_rope(k_r, cos, sin)                       # (B, T, dr), shared by heads

        if cache is not None:                                  # cache only the latent + rope key
            if "c_kv" in cache:
                c_kv = torch.cat([cache["c_kv"], c_kv], dim=1)
                k_r = torch.cat([cache["k_r"], k_r], dim=1)
            cache["c_kv"], cache["k_r"] = c_kv, k_r

        S = c_kv.shape[1]
        k_c = self.k_up(c_kv).view(B, S, h, dh)
        v = self.v_up(c_kv).view(B, S, h, dh)
        k = torch.cat([k_c, k_r[:, :, None].expand(B, S, h, dr)], dim=-1)
        qf = torch.cat([q_c, q_r], dim=-1)

        attn = F.scaled_dot_product_attention(
            qf.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=cache is None or T > 1, scale=1.0 / math.sqrt(dh + dr),
        )
        return self.out(attn.transpose(1, 2).reshape(B, T, h * dh))


def kv_bytes_per_token(cfg: MLAConfig, dtype_bytes: int = 2) -> dict[str, int]:
    mha = 2 * cfg.n_heads * cfg.d_head * dtype_bytes
    mla = (cfg.d_latent_kv + cfg.d_rope) * dtype_bytes
    return {"mha": mha, "mla": mla, "ratio_x": mha // mla}


if __name__ == "__main__":
    cfg = MLAConfig()
    layer = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 128, cfg.d_model)
    cache: dict = {}
    y = layer(x, cache=cache)
    y_next = layer(torch.randn(2, 1, cfg.d_model), cache=cache, start_pos=128)
    print(y.shape, y_next.shape, cache["c_kv"].shape, kv_bytes_per_token(cfg))
