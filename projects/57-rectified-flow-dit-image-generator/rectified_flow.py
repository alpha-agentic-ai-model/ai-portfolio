"""Rectified-flow (flow matching) image generator: a small DiT learns the
velocity field that transports Gaussian noise to data along straight paths,
so sampling needs only a handful of Euler steps."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * 1000 * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)          # adaLN-Zero modulation
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class FlowDiT(nn.Module):
    def __init__(self, img: int = 32, ch: int = 3, patch: int = 2, dim: int = 384,
                 depth: int = 12, heads: int = 6, classes: int = 10) -> None:
        super().__init__()
        self.patch, self.grid, self.ch = patch, img // patch, ch
        self.embed = nn.Linear(ch * patch * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.y_emb = nn.Embedding(classes + 1, dim)  # last index = unconditional (CFG)
        self.blocks = nn.ModuleList(DiTBlock(dim, heads) for _ in range(depth))
        self.out = nn.Linear(dim, ch * patch * patch)
        self.null_class = classes

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = self.patch
        tokens = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        h = self.embed(tokens) + self.pos
        c = self.t_mlp(timestep_embedding(t, h.shape[-1])) + self.y_emb(y)
        for blk in self.blocks:
            h = blk(h, c)
        v = self.out(h)
        return rearrange(v, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=self.grid, p1=p, p2=p, c=self.ch)


def flow_matching_loss(model: FlowDiT, x1: torch.Tensor, y: torch.Tensor, p_uncond: float = 0.1) -> torch.Tensor:
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(x1.size(0), device=x1.device))   # logit-normal t sampling
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    y = torch.where(torch.rand_like(t) < p_uncond, torch.full_like(y, model.null_class), y)
    target_v = x1 - x0                                               # straight-line velocity
    return F.mse_loss(model(xt, t, y), target_v)


@torch.no_grad()
def sample(model: FlowDiT, y: torch.Tensor, steps: int = 8, cfg: float = 2.0) -> torch.Tensor:
    x = torch.randn(y.size(0), model.ch, model.grid * model.patch, model.grid * model.patch, device=y.device)
    null = torch.full_like(y, model.null_class)
    for i in range(steps):
        t = torch.full((y.size(0),), i / steps, device=y.device)
        v_c, v_u = model(x, t, y), model(x, t, null)
        x = x + (v_u + cfg * (v_c - v_u)) / steps                     # Euler step
    return x.clamp(-1, 1)


def train(epochs: int = 100, device: str = "cuda") -> FlowDiT:
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    loader = DataLoader(datasets.CIFAR10("data", download=True, transform=tf),
                        batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    model = FlowDiT().to(device)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = flow_matching_loss(model, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update_parameters(model)
        print(f"epoch {epoch} loss {loss.item():.4f}")
    return ema.module


if __name__ == "__main__":
    net = train()
    imgs = sample(net, torch.arange(10, device="cuda").repeat_interleave(8), steps=8)
    torch.save(imgs.cpu(), "samples.pt")
