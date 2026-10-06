import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class HybridConfig:
    vocab_size: int = 50304
    d_model: int = 2048
    n_layers: int = 32
    n_heads: int = 16
    d_state: int = 128          # SSM state dimension (Mamba-2 style)
    d_conv: int = 4
    expand: int = 2
    attention_every: int = 6    # one attention block per N SSM blocks
    max_seq_len: int = 131072
    rope_theta: float = 500000.0


class SelectiveScan(nn.Module):
    """Mamba-2 selective state-space block with input-dependent A, B, C."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_model * cfg.expand
        self.d_inner = d_inner
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=cfg.d_conv,
                                groups=d_inner, padding=cfg.d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, cfg.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, cfg.d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        b, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)
        x_in = rearrange(self.conv1d(rearrange(x_in, "b l d -> b d l"))[..., :seq_len], "b d l -> b l d")
        x_in = F.silu(x_in)

        proj = self.x_proj(x_in)
        B, C, dt = proj.split([self.cfg.d_state, self.cfg.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (b, l, d_inner)
        A = -torch.exp(self.A_log.float())                     # (d_inner, d_state)

        y, state = self._scan(x_in, dt, A, B, C, state)
        y = y + x_in * self.D
        return self.out_proj(y * F.silu(gate)), state

    def _scan(self, u, dt, A, B, C, state):
        """Associative sequential scan; swap for the fused CUDA kernel in prod."""
        b, seq_len, d_inner = u.shape
        if state is None:
            state = u.new_zeros(b, d_inner, self.cfg.d_state)
        outputs = []
        dA = torch.exp(dt.unsqueeze(-1) * A)                   # (b, l, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        for t in range(seq_len):
            state = dA[:, t] * state + dBu[:, t]
            outputs.append(torch.einsum("bdn,bn->bd", state, C[:, t]))
        return torch.stack(outputs, dim=1), state


class SlidingWindowAttention(nn.Module):
    """Periodic attention layer that restores exact token-to-token recall."""

    def __init__(self, cfg: HybridConfig, window: int = 4096):
        super().__init__()
        self.cfg, self.window = cfg, window
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.register_buffer("inv_freq", 1.0 / (cfg.rope_theta ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.cfg.n_heads) for t in (q, k, v))
        q, k = self._rope(q, k, positions)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(rearrange(out, "b h l d -> b l (h d)"))

    def _rope(self, q, k, positions):
        freqs = torch.einsum("l,d->ld", positions.float(), self.inv_freq)
        cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]

        def rotate(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

        return rotate(q), rotate(k)


class HybridBlock(nn.Module):
    def __init__(self, cfg: HybridConfig, layer_idx: int):
        super().__init__()
        self.use_attention = (layer_idx + 1) % cfg.attention_every == 0
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mixer = SlidingWindowAttention(cfg) if self.use_attention else SelectiveScan(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False), nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False))

    def forward(self, x, positions, state=None):
        if self.use_attention:
            x = x + self.mixer(self.norm1(x), positions)
        else:
            delta, state = self.mixer(self.norm1(x), state)
            x = x + delta
        return x + self.mlp(self.norm2(x)), state


class HybridSSMLanguageModel(nn.Module):
    """Interleaved Mamba-2 / attention backbone for 128k-token pretraining."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([HybridBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                states: Optional[list] = None):
        x = self.embed(input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        states = states or [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            x, states[i] = block(x, positions, states[i])
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        return {"logits": logits, "loss": loss, "states": states}
