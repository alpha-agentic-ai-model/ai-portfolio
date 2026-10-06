import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class DraftProfile:
    """Runtime statistics for one candidate draft model."""
    name: str
    params_b: float
    gamma: int = 5                     # tokens proposed per verification step
    accepted: int = 0
    proposed: int = 0
    draft_latency_ms: deque = field(default_factory=lambda: deque(maxlen=256))

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.5

    @property
    def mean_draft_ms(self) -> float:
        return sum(self.draft_latency_ms) / len(self.draft_latency_ms) if self.draft_latency_ms else 1.0

    def expected_speedup(self, target_ms: float) -> float:
        """Leviathan-style expected tokens per verification pass, cost adjusted."""
        a = min(max(self.acceptance_rate, 1e-3), 0.999)
        expected_tokens = (1 - a ** (self.gamma + 1)) / (1 - a)
        cost = target_ms + self.gamma * self.mean_draft_ms
        return expected_tokens * target_ms / max(cost, 1e-6)


class AdaptiveDraftSelector:
    """Thompson-sampling bandit over draft models, keyed by request class."""

    def __init__(self, profiles: list[DraftProfile], explore_sigma: float = 0.08):
        self.profiles = {p.name: p for p in profiles}
        self.explore_sigma = explore_sigma
        self.target_latency_ms = 22.0

    def select(self, prompt_len: int, requested_tokens: int) -> DraftProfile:
        best, best_score = None, -math.inf
        for profile in self.profiles.values():
            score = profile.expected_speedup(self.target_latency_ms)
            score += torch.randn(1).item() * self.explore_sigma      # exploration
            if prompt_len > 8192:
                score -= 0.05 * profile.params_b                     # long prefills favour tiny drafts
            if score > best_score:
                best, best_score = profile, score
        return best

    def tune_gamma(self, profile: DraftProfile) -> None:
        """Walk gamma toward the acceptance-rate optimum: gamma* ~ 1/(1-alpha)."""
        a = profile.acceptance_rate
        target = int(round(min(max(1.0 / max(1 - a, 0.08), 2), 12)))
        profile.gamma += (1 if target > profile.gamma else -1) if target != profile.gamma else 0


class SpeculativeEngine:
    """Draft-and-verify decoding loop with lossless rejection sampling."""

    def __init__(self, target_model, draft_models: dict, selector: AdaptiveDraftSelector,
                 temperature: float = 0.7):
        self.target = target_model
        self.drafts = draft_models
        self.selector = selector
        self.temperature = temperature
        self.metrics = {"tokens": 0, "verify_passes": 0, "rejections": 0}

    @torch.inference_mode()
    async def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256) -> dict:
        profile = self.selector.select(input_ids.size(-1), max_new_tokens)
        draft = self.drafts[profile.name]
        tokens = input_ids
        target_cache, draft_cache = None, None
        started = time.monotonic()

        while tokens.size(-1) - input_ids.size(-1) < max_new_tokens:
            t0 = time.monotonic()
            proposals, draft_probs, draft_cache = self._draft(draft, tokens, draft_cache, profile.gamma)
            profile.draft_latency_ms.append((time.monotonic() - t0) * 1000 / profile.gamma)

            target_probs, target_cache = self._verify(tokens, proposals, target_cache)
            accepted, bonus = self._rejection_sample(proposals, draft_probs, target_probs)

            profile.proposed += proposals.size(-1)
            profile.accepted += accepted.numel()
            self.metrics["verify_passes"] += 1
            self.metrics["rejections"] += proposals.size(-1) - accepted.numel()

            tokens = torch.cat([tokens, accepted.unsqueeze(0), bonus.unsqueeze(0)], dim=-1)
            if accepted.numel() < proposals.size(-1):
                draft_cache = self._rollback(draft_cache, tokens.size(-1))
            self.selector.tune_gamma(profile)
            if bonus.item() == self.target.config.eos_token_id:
                break
            await asyncio.sleep(0)

        generated = tokens.size(-1) - input_ids.size(-1)
        elapsed = time.monotonic() - started
        return {"tokens": tokens, "generated": generated,
                "draft_model": profile.name, "gamma": profile.gamma,
                "acceptance_rate": round(profile.acceptance_rate, 3),
                "tokens_per_second": round(generated / max(elapsed, 1e-6), 1)}

    def _draft(self, draft, tokens, cache, gamma: int):
        proposals, probs = [], []
        cursor = tokens
        for _ in range(gamma):
            out = draft(cursor, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            dist = F.softmax(out.logits[:, -1] / self.temperature, dim=-1)
            token = torch.multinomial(dist, num_samples=1)
            proposals.append(token)
            probs.append(dist)
            cursor = token
        return torch.cat(proposals, dim=-1)[0], torch.stack(probs, dim=1)[0], cache

    def _verify(self, tokens, proposals, cache):
        full = torch.cat([tokens, proposals.unsqueeze(0)], dim=-1)
        out = self.target(full, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -(proposals.size(-1) + 1):]
        return F.softmax(logits / self.temperature, dim=-1)[0], out.past_key_values

    def _rejection_sample(self, proposals, draft_probs, target_probs):
        """Lossless acceptance: keep token with p=min(1, q_target/q_draft)."""
        accepted = []
        for i, token in enumerate(proposals):
            p_target = target_probs[i, token]
            p_draft = draft_probs[i, token]
            if torch.rand(1, device=token.device) < torch.clamp(p_target / (p_draft + 1e-9), max=1.0):
                accepted.append(token)
                continue
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            residual = residual / residual.sum().clamp(min=1e-9)
            return torch.stack(accepted) if accepted else proposals[:0], torch.multinomial(residual, 1)[0]
        bonus = torch.multinomial(target_probs[-1], num_samples=1)[0]
        return torch.stack(accepted), bonus

    def _rollback(self, cache, keep_len: int):
        if cache is None:
            return None
        return tuple(tuple(t[..., : keep_len - 1, :] for t in layer) for layer in cache)
