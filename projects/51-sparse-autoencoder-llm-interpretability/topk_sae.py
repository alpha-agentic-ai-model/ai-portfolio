"""Sparse autoencoder for LLM interpretability.

Trains a TopK sparse autoencoder on residual-stream activations captured from a
transformer layer, so each latent tends to represent one human-interpretable
feature. Includes dead-latent resampling, normalised decoder columns and a
helper that surfaces the top-activating contexts for any feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer


@dataclass
class SAEConfig:
    d_model: int = 768
    expansion: int = 16
    k: int = 32
    lr: float = 4e-4
    batch_tokens: int = 4096
    dead_after_tokens: int = 2_000_000


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        d_sae = cfg.d_model * cfg.expansion
        self.k = cfg.k
        self.b_pre = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_enc = nn.Parameter(torch.empty(cfg.d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, cfg.d_model))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)
            self.W_enc.data = self.W_dec.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_pre) @ self.W_enc + self.b_enc
        vals, idx = torch.topk(F.relu(pre), self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z @ self.W_dec + self.b_pre, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


class SAETrainer:
    def __init__(self, model_name: str, layer: int, cfg: SAEConfig, device: str = "cuda"):
        self.lm = HookedTransformer.from_pretrained(model_name, device=device)
        self.hook = f"blocks.{layer}.hook_resid_post"
        self.cfg, self.device = cfg, device
        self.sae = TopKSAE(cfg).to(device)
        self.opt = torch.optim.Adam(self.sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
        self.tokens_since_fired = torch.zeros(cfg.d_model * cfg.expansion, device=device)

    @torch.no_grad()
    def activations(self, tokens: torch.Tensor) -> torch.Tensor:
        _, cache = self.lm.run_with_cache(tokens, names_filter=self.hook)
        return cache[self.hook].reshape(-1, self.cfg.d_model)

    @torch.no_grad()
    def resample_dead(self, x: torch.Tensor, recon: torch.Tensor) -> int:
        dead = (self.tokens_since_fired > self.cfg.dead_after_tokens).nonzero().squeeze(-1)
        if dead.numel() == 0:
            return 0
        err = (x - recon).pow(2).sum(-1)
        picks = torch.multinomial(err, dead.numel(), replacement=True)
        new_dirs = F.normalize(x[picks] - self.sae.b_pre, dim=-1)
        self.sae.W_dec.data[dead] = new_dirs
        self.sae.W_enc.data[:, dead] = new_dirs.T * 0.2
        self.sae.b_enc.data[dead] = 0.0
        self.tokens_since_fired[dead] = 0
        return dead.numel()

    def step(self, tokens: torch.Tensor) -> dict[str, float]:
        x = self.activations(tokens.to(self.device))
        recon, z = self.sae(x)
        loss = F.mse_loss(recon, x)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sae.normalize_decoder()
        fired = (z > 0).any(0)
        self.tokens_since_fired = torch.where(fired, 0, self.tokens_since_fired + x.shape[0])
        fvu = (x - recon).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
        revived = self.resample_dead(x, recon.detach())
        return {"mse": loss.item(), "fvu": fvu.item(), "revived": revived,
                "dead_frac": (self.tokens_since_fired > self.cfg.dead_after_tokens).float().mean().item()}

    @torch.no_grad()
    def top_contexts(self, feature: int, tokens: torch.Tensor, n: int = 10) -> list[str]:
        z = self.sae.encode(self.activations(tokens.to(self.device)))[:, feature]
        flat = tokens.reshape(-1)
        best = torch.topk(z, n).indices
        return [self.lm.to_string(flat[max(0, i - 8): i + 2]) for i in best.tolist()]


if __name__ == "__main__":
    trainer = SAETrainer("gpt2-small", layer=8, cfg=SAEConfig())
    batch = trainer.lm.to_tokens(["The Eiffel Tower is in Paris."] * 8)
    print(trainer.step(batch))
