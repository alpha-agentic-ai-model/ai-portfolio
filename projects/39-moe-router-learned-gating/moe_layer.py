import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class MoEConfig:
    dim: int = 768
    num_experts: int = 8
    top_k: int = 2
    ffn_multiplier: int = 4
    capacity_factor: float = 1.25
    aux_loss_weight: float = 0.01
    expert_dropout: float = 0.0
    jitter_noise: float = 0.0


class TopKGating(nn.Module):
    """Learned top-k gating network with load-balancing auxiliary loss."""

    def __init__(self, dim: int, num_experts: int, top_k: int = 2, jitter_noise: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.jitter_noise = jitter_noise

        # Initialize gate weights uniformly
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            weights: (batch, seq_len, top_k)
            indices: (batch, seq_len, top_k)
            aux_loss: scalar load-balancing loss
        """
        logits = self.gate(x)  # (B, S, E)

        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        scores = F.softmax(logits, dim=-1)
        top_vals, top_idx = scores.topk(self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)

        # Load-balancing auxiliary loss (Switch Transformer style)
        # Encourages uniform expert utilization
        density = scores.mean(dim=(0, 1))  # Average routing probability per expert
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(top_idx[:, :, 0], self.num_experts).float()
        frequency = one_hot.mean(dim=(0, 1))
        aux_loss = self.num_experts * (density * frequency).sum()

        return top_vals, top_idx, aux_loss


class ExpertFFN(nn.Module):
    """Feed-forward network for a single expert."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class MoELayer(nn.Module):
    """Sparse Mixture of Experts layer with capacity-constrained routing."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gating = TopKGating(
            dim=config.dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            jitter_noise=config.jitter_noise,
        )
        ffn_dim = config.dim * config.ffn_multiplier
        self.experts = nn.ModuleList([
            ExpertFFN(config.dim, ffn_dim, config.expert_dropout)
            for _ in range(config.num_experts)
        ])
        self.layer_norm = nn.LayerNorm(config.dim)

        # Expert utilization tracking
        self.register_buffer("expert_counts", torch.zeros(config.num_experts))
        self.register_buffer("total_tokens", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            output: (batch, seq_len, dim)
            aux_loss: scalar auxiliary loss for load balancing
        """
        residual = x
        x = self.layer_norm(x)
        B, S, D = x.shape

        weights, indices, aux_loss = self.gating(x)
        output = torch.zeros_like(x)

        # Capacity constraint: max tokens per expert
        capacity = int(self.config.capacity_factor * S * B / self.config.num_experts)

        for k in range(self.config.top_k):
            expert_idx = indices[:, :, k]  # (B, S)
            expert_wt = weights[:, :, k].unsqueeze(-1)  # (B, S, 1)

            for e_idx in range(self.config.num_experts):
                mask = (expert_idx == e_idx)  # (B, S)
                if not mask.any():
                    continue

                # Apply capacity constraint
                flat_mask = mask.view(-1)
                active_indices = flat_mask.nonzero(as_tuple=True)[0]
                if len(active_indices) > capacity:
                    active_indices = active_indices[:capacity]
                    limited_mask = torch.zeros_like(flat_mask)
                    limited_mask[active_indices] = True
                    mask = limited_mask.view(B, S)

                # Route tokens to expert
                tokens = x[mask]  # (num_tokens, D)
                expert_out = self.experts[e_idx](tokens)
                output[mask] += expert_wt[mask] * expert_out

                # Track utilization
                if self.training:
                    self.expert_counts[e_idx] += mask.sum().item()

        if self.training:
            self.total_tokens += B * S

        return residual + output, aux_loss * self.config.aux_loss_weight

    @property
    def expert_utilization(self) -> dict[int, float]:
        """Get expert utilization ratios."""
        total = self.total_tokens.item()
        if total == 0:
            return {}
        return {
            i: count / total
            for i, count in enumerate(self.expert_counts.tolist())
        }

    def reset_utilization_stats(self):
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE feed-forward layer."""

    def __init__(self, dim: int, num_heads: int, moe_config: MoEConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.moe = MoELayer(moe_config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        normed = self.attn_norm(x)
        attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        moe_out, aux_loss = self.moe(x)
        return moe_out, aux_loss
