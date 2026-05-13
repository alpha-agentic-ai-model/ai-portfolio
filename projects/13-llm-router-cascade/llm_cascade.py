"""
Intelligent LLM Router with Cost-Aware Cascade
Routes prompts to optimal models based on complexity, cost, and quality requirements.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: float
    max_tokens: int
    quality_tier: int  # 1=highest, 3=lowest


@dataclass
class RouteDecision:
    model: str
    estimated_cost: float
    confidence: float
    fallback_chain: list[str]
    reasoning: str = ""


@dataclass
class RoutingMetrics:
    total_requests: int = 0
    total_cost: float = 0.0
    cascade_triggers: int = 0
    model_distribution: dict = field(default_factory=dict)
    avg_latency_ms: float = 0.0


AVAILABLE_MODELS = {
    "claude-haiku-4-5": ModelConfig(
        name="claude-haiku-4-5",
        provider="anthropic",
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        avg_latency_ms=200,
        max_tokens=8192,
        quality_tier=3,
    ),
    "claude-sonnet-4-6": ModelConfig(
        name="claude-sonnet-4-6",
        provider="anthropic",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=800,
        max_tokens=8192,
        quality_tier=2,
    ),
    "claude-opus-4-6": ModelConfig(
        name="claude-opus-4-6",
        provider="anthropic",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=2000,
        max_tokens=8192,
        quality_tier=1,
    ),
}

CASCADE_CHAINS = {
    Complexity.SIMPLE: ["claude-haiku-4-5", "claude-sonnet-4-6"],
    Complexity.MODERATE: ["claude-sonnet-4-6", "claude-opus-4-6"],
    Complexity.COMPLEX: ["claude-opus-4-6"],
}


class FeatureExtractor:
    """Extracts routing-relevant features from prompts."""

    def extract(self, prompt: str) -> np.ndarray:
        features = [
            len(prompt),
            len(prompt.split()),
            prompt.count("\n"),
            prompt.count("```"),
            sum(1 for c in prompt if c.isupper()) / max(len(prompt), 1),
            len(set(prompt.split())) / max(len(prompt.split()), 1),
            self._count_technical_terms(prompt),
            self._estimate_reasoning_depth(prompt),
            prompt.lower().count("step by step"),
            prompt.lower().count("analyze"),
            prompt.lower().count("compare"),
            prompt.lower().count("explain"),
        ]
        return np.array(features, dtype=np.float32)

    def _count_technical_terms(self, text: str) -> int:
        technical = {
            "algorithm", "optimize", "architecture", "implement",
            "debug", "refactor", "benchmark", "evaluate", "deploy",
            "kubernetes", "docker", "api", "database", "schema",
        }
        words = set(text.lower().split())
        return len(words & technical)

    def _estimate_reasoning_depth(self, text: str) -> float:
        depth_markers = [
            "why", "how", "compare", "trade-off", "pros and cons",
            "step by step", "in detail", "thoroughly", "comprehensive",
        ]
        count = sum(1 for m in depth_markers if m in text.lower())
        return min(count / 3.0, 1.0)


class ComplexityClassifier:
    """ML-based prompt complexity classification."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.thresholds = {"simple_max": 0.35, "moderate_max": 0.7}

    def classify(self, prompt: str) -> tuple[Complexity, float]:
        features = self.feature_extractor.extract(prompt)
        score = self._compute_complexity_score(features)
        if score <= self.thresholds["simple_max"]:
            return Complexity.SIMPLE, 1.0 - score
        elif score <= self.thresholds["moderate_max"]:
            return Complexity.MODERATE, 0.7
        else:
            return Complexity.COMPLEX, score

    def _compute_complexity_score(self, features: np.ndarray) -> float:
        weights = np.array([
            0.001, 0.005, 0.02, 0.15, 0.1, -0.1,
            0.08, 0.3, 0.1, 0.05, 0.05, 0.03,
        ])
        raw = float(np.dot(features, weights))
        return max(0.0, min(1.0, raw))


class CostMonitor:
    """Tracks API costs and provides estimates."""

    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.daily_spend: float = 0.0
        self.request_log: list[dict] = []

    def estimate(self, model_name: str, prompt: str) -> float:
        config = AVAILABLE_MODELS.get(model_name)
        if not config:
            return 0.0
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = min(input_tokens * 0.8, config.max_tokens)
        cost = (
            (input_tokens / 1000) * config.cost_per_1k_input
            + (output_tokens / 1000) * config.cost_per_1k_output
        )
        return round(cost, 6)

    def record(self, model: str, input_tokens: int, output_tokens: int):
        config = AVAILABLE_MODELS.get(model)
        if not config:
            return
        cost = (
            (input_tokens / 1000) * config.cost_per_1k_input
            + (output_tokens / 1000) * config.cost_per_1k_output
        )
        self.daily_spend += cost
        self.request_log.append({
            "model": model, "cost": cost, "timestamp": time.time()
        })

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.daily_budget - self.daily_spend)


class QualityGate:
    """Evaluates response quality for cascade decisions."""

    def __init__(self, min_length: int = 50, coherence_threshold: float = 0.6):
        self.min_length = min_length
        self.coherence_threshold = coherence_threshold

    def evaluate(self, prompt: str, response: str) -> tuple[bool, float]:
        if len(response.strip()) < self.min_length:
            return False, 0.2
        score = self._compute_quality_score(prompt, response)
        return score >= self.coherence_threshold, score

    def _compute_quality_score(self, prompt: str, response: str) -> float:
        length_score = min(len(response) / max(len(prompt), 1), 2.0) / 2.0
        has_structure = any(
            m in response for m in ["1.", "- ", "```", "##", "**"]
        )
        structure_score = 0.3 if has_structure else 0.0
        relevance = self._keyword_overlap(prompt, response)
        return min(1.0, length_score * 0.4 + structure_score + relevance * 0.3)

    def _keyword_overlap(self, prompt: str, response: str) -> float:
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        if not prompt_words:
            return 0.0
        return len(prompt_words & response_words) / len(prompt_words)


class LLMRouter:
    """Intelligent router with cost-aware cascading."""

    def __init__(self, daily_budget: float = 100.0):
        self.classifier = ComplexityClassifier()
        self.cost_tracker = CostMonitor(daily_budget)
        self.quality_gate = QualityGate()
        self.metrics = RoutingMetrics()

    async def route(self, prompt: str) -> RouteDecision:
        complexity, confidence = self.classifier.classify(prompt)
        chain = CASCADE_CHAINS[complexity]
        model = chain[0]
        if self.cost_tracker.budget_remaining < self.cost_tracker.estimate(model, prompt):
            model = chain[-1] if len(chain) > 1 else model
            logger.warning(f"Budget constrained, downgrading to {model}")
        return RouteDecision(
            model=model,
            estimated_cost=self.cost_tracker.estimate(model, prompt),
            confidence=confidence,
            fallback_chain=chain[1:],
            reasoning=f"Complexity={complexity.value}, confidence={confidence:.2f}",
        )

    async def execute_with_cascade(
        self, prompt: str, llm_client: Any
    ) -> dict:
        decision = await self.route(prompt)
        self.metrics.total_requests += 1
        models_to_try = [decision.model] + decision.fallback_chain
        for i, model in enumerate(models_to_try):
            start = time.time()
            response = await llm_client.generate(model=model, prompt=prompt)
            latency = (time.time() - start) * 1000
            passed, quality = self.quality_gate.evaluate(prompt, response)
            self.metrics.model_distribution[model] = (
                self.metrics.model_distribution.get(model, 0) + 1
            )
            if passed or i == len(models_to_try) - 1:
                self.cost_tracker.record(
                    model, len(prompt.split()), len(response.split())
                )
                return {
                    "response": response,
                    "model_used": model,
                    "quality_score": quality,
                    "latency_ms": latency,
                    "cascaded": i > 0,
                    "cost": decision.estimated_cost,
                }
            logger.info(f"Quality gate failed for {model}, cascading...")
            self.metrics.cascade_triggers += 1
        return {"response": response, "model_used": models_to_try[-1]}
