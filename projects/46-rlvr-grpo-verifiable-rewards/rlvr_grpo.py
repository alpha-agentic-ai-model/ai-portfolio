import asyncio
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class Rollout:
    prompt: str
    completion: str
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    logprobs: Optional[torch.Tensor] = None
    advantage: float = 0.0


@dataclass
class GRPOConfig:
    group_size: int = 8              # completions sampled per prompt
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    reward_std_floor: float = 1e-4
    pass_threshold: float = 0.6      # curriculum promotion gate


class Verifier:
    """Base class for deterministic, programmatically checkable rewards."""

    name: str = "verifier"
    weight: float = 1.0

    async def score(self, prompt: str, completion: str, meta: dict) -> float:
        raise NotImplementedError


class UnitTestVerifier(Verifier):
    """Executes model-written code against hidden tests inside a sandbox."""

    name, weight = "unit_tests", 1.0

    def __init__(self, image: str = "ghcr.io/rlvr/pytest-sandbox:latest", timeout_s: int = 20):
        self.image, self.timeout_s = image, timeout_s

    async def score(self, prompt: str, completion: str, meta: dict) -> float:
        code = self._extract_code(completion)
        if not code:
            return 0.0
        payload = json.dumps({"solution": code, "tests": meta.get("tests", "")})
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "-i", "--network=none", "--cap-drop=ALL",
            "--memory=512m", "--pids-limit=64", self.image,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(payload.encode()), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return 0.0
        try:
            report = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            return 0.0
        total = max(report.get("total", 0), 1)
        return report.get("passed", 0) / total

    @staticmethod
    def _extract_code(text: str) -> str:
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
        return blocks[-1].strip() if blocks else ""


class FormatVerifier(Verifier):
    """Cheap shaping reward: enforces the required reasoning/answer structure."""

    name, weight = "format", 0.15

    async def score(self, prompt: str, completion: str, meta: dict) -> float:
        has_reasoning = bool(re.search(r"<think>.*?</think>", completion, flags=re.S))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, flags=re.S))
        length_ok = 32 <= len(completion.split()) <= 2048
        return (has_reasoning + has_answer + length_ok) / 3.0


class RewardRouter:
    """Aggregates verifier scores into a single scalar, logging each component."""

    def __init__(self, verifiers: list[Verifier]):
        self.verifiers = verifiers
        self.total_weight = sum(v.weight for v in verifiers) or 1.0

    async def evaluate(self, rollout: Rollout, meta: dict) -> Rollout:
        scores = await asyncio.gather(*[
            v.score(rollout.prompt, rollout.completion, meta) for v in self.verifiers])
        rollout.reward_breakdown = {v.name: float(s) for v, s in zip(self.verifiers, scores)}
        rollout.reward = sum(v.weight * s for v, s in zip(self.verifiers, scores)) / self.total_weight
        return rollout


class GRPOTrainer:
    """Group-Relative Policy Optimisation over verifiable rewards.

    GRPO drops the value network: advantages are the z-scored rewards inside a
    group of completions sampled from the same prompt, which makes the loop
    cheap enough to run against a live vLLM sampler.
    """

    def __init__(self, policy, reference, sampler, router: RewardRouter, cfg: GRPOConfig):
        self.policy, self.reference = policy, reference
        self.sampler, self.router, self.cfg = sampler, router, cfg
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.history: list[dict] = []

    async def collect_group(self, prompt: str, meta: dict) -> list[Rollout]:
        completions = await self.sampler.generate(
            prompt, n=self.cfg.group_size, temperature=self.cfg.temperature)
        rollouts = [Rollout(prompt=prompt, completion=c.text, logprobs=c.logprobs)
                    for c in completions]
        rollouts = await asyncio.gather(*[self.router.evaluate(r, meta) for r in rollouts])
        rewards = torch.tensor([r.reward for r in rollouts])
        baseline, spread = rewards.mean(), rewards.std().clamp(min=self.cfg.reward_std_floor)
        for rollout, reward in zip(rollouts, rewards):
            rollout.advantage = float((reward - baseline) / spread)
        return rollouts

    def grpo_loss(self, rollouts: list[Rollout]) -> torch.Tensor:
        losses = []
        for rollout in rollouts:
            tokens = self.policy.tokenize(rollout.prompt, rollout.completion)
            new_logprobs = self.policy.token_logprobs(tokens)
            with torch.no_grad():
                ref_logprobs = self.reference.token_logprobs(tokens)
            ratio = torch.exp(new_logprobs - rollout.logprobs.to(new_logprobs.device))
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            pg = -torch.min(ratio * rollout.advantage, clipped * rollout.advantage).mean()
            kl = (torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1).mean()
            losses.append(pg + self.cfg.kl_coef * kl)
        return torch.stack(losses).mean()

    async def step(self, batch: list[dict]) -> dict:
        started = time.monotonic()
        groups = await asyncio.gather(*[
            self.collect_group(item["prompt"], item) for item in batch])
        flat = [r for group in groups for r in group]
        loss = self.grpo_loss(flat)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        await self.sampler.sync_weights(self.policy)
        pass_rate = sum(r.reward_breakdown.get("unit_tests", 0) == 1.0 for r in flat) / len(flat)
        metrics = {"loss": float(loss), "grad_norm": float(grad_norm),
                   "mean_reward": sum(r.reward for r in flat) / len(flat),
                   "pass_rate": round(pass_rate, 3),
                   "promote_curriculum": pass_rate >= self.cfg.pass_threshold,
                   "step_seconds": round(time.monotonic() - started, 2)}
        self.history.append(metrics)
        return metrics
