import hashlib
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from sentence_transformers import SentenceTransformer
from enum import Enum


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class CacheEntry:
    prompt_hash: str
    embedding: np.ndarray
    response: str
    model: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    ttl_seconds: int = 3600


@dataclass
class ProviderHealth:
    name: str
    status: ProviderStatus = ProviderStatus.HEALTHY
    last_check: float = 0.0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    cost_per_1k_tokens: float = 0.0


@dataclass
class GatewayMetrics:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_errors: dict[str, int] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0


class SemanticCache:
    """Embedding-based semantic cache for LLM responses."""

    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2", threshold: float = 0.92, max_entries: int = 10000):
        self.encoder = SentenceTransformer(encoder_model)
        self.threshold = threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []

    async def lookup(self, prompt: str) -> Optional[str]:
        """Find semantically similar cached response."""
        emb = self.encoder.encode([prompt])[0]
        now = time.time()

        best_match = None
        best_sim = 0.0

        for entry in self.entries:
            # Skip expired entries
            if now - entry.created_at > entry.ttl_seconds:
                continue

            sim = np.dot(emb, entry.embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(entry.embedding)
            )
            if sim >= self.threshold and sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match:
            best_match.hit_count += 1
            return best_match.response
        return None

    async def store(self, prompt: str, response: str, model: str = "", ttl: int = 3600):
        """Store a new cache entry."""
        emb = self.encoder.encode([prompt])[0]
        entry = CacheEntry(
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            embedding=emb,
            response=response,
            model=model,
            ttl_seconds=ttl,
        )
        self.entries.append(entry)

        # Evict oldest entries if over capacity
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.created_at)
            self.entries = self.entries[-self.max_entries:]

    def evict_expired(self):
        """Remove expired cache entries."""
        now = time.time()
        self.entries = [e for e in self.entries if now - e.created_at <= e.ttl_seconds]

    @property
    def hit_rate(self) -> float:
        total_hits = sum(e.hit_count for e in self.entries)
        return total_hits / max(len(self.entries), 1)


class LLMProvider:
    """Base class for LLM provider integration."""

    def __init__(self, name: str, client: Any, cost_per_1k: float):
        self.name = name
        self.client = client
        self.cost_per_1k = cost_per_1k

    async def call(self, prompt: str, model: Optional[str] = None) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    async def call(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or "claude-sonnet-4-6"
        response = await self.client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    async def call(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or "gpt-4o"
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class LLMGateway:
    """Multi-provider LLM gateway with semantic caching and failover."""

    def __init__(self, providers: list[LLMProvider], cache: SemanticCache,
                 max_retries: int = 2, health_check_interval: int = 60):
        self.providers = providers
        self.cache = cache
        self.max_retries = max_retries
        self.health_check_interval = health_check_interval
        self.health: dict[str, ProviderHealth] = {
            p.name: ProviderHealth(name=p.name, cost_per_1k_tokens=p.cost_per_1k)
            for p in providers
        }
        self.metrics = GatewayMetrics()
        self._dedup_lock: dict[str, asyncio.Event] = {}
        self._dedup_results: dict[str, str] = {}

    def get_healthy_providers(self) -> list[LLMProvider]:
        """Return providers sorted by health and cost."""
        healthy = [
            p for p in self.providers
            if self.health[p.name].status != ProviderStatus.DOWN
        ]
        healthy.sort(key=lambda p: (
            0 if self.health[p.name].status == ProviderStatus.HEALTHY else 1,
            p.cost_per_1k,
        ))
        return healthy

    async def complete(self, prompt: str, model: Optional[str] = None,
                       skip_cache: bool = False) -> dict:
        """Route completion request with caching and failover."""
        self.metrics.total_requests += 1
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Request deduplication
        if prompt_hash in self._dedup_lock:
            await self._dedup_lock[prompt_hash].wait()
            if prompt_hash in self._dedup_results:
                return {"response": self._dedup_results[prompt_hash], "cached": True, "deduped": True}

        # Check semantic cache
        if not skip_cache:
            cached = await self.cache.lookup(prompt)
            if cached:
                self.metrics.cache_hits += 1
                return {"response": cached, "cached": True, "provider": "cache"}

        self.metrics.cache_misses += 1
        self._dedup_lock[prompt_hash] = asyncio.Event()

        try:
            for provider in self.get_healthy_providers():
                for attempt in range(self.max_retries):
                    try:
                        start = time.perf_counter()
                        resp = await provider.call(prompt, model)
                        latency = (time.perf_counter() - start) * 1000

                        # Update health metrics
                        health = self.health[provider.name]
                        health.avg_latency_ms = (health.avg_latency_ms * 0.9) + (latency * 0.1)
                        health.error_count = max(0, health.error_count - 1)
                        health.status = ProviderStatus.HEALTHY

                        # Cache the response
                        await self.cache.store(prompt, resp, model or provider.name)

                        self._dedup_results[prompt_hash] = resp
                        return {
                            "response": resp,
                            "cached": False,
                            "provider": provider.name,
                            "latency_ms": round(latency, 1),
                        }
                    except Exception as e:
                        health = self.health[provider.name]
                        health.error_count += 1
                        if health.error_count >= 3:
                            health.status = ProviderStatus.DOWN
                        elif health.error_count >= 1:
                            health.status = ProviderStatus.DEGRADED
                        self.metrics.provider_errors[provider.name] = (
                            self.metrics.provider_errors.get(provider.name, 0) + 1
                        )

            raise RuntimeError("All LLM providers unavailable")
        finally:
            self._dedup_lock[prompt_hash].set()
            self._dedup_lock.pop(prompt_hash, None)
            # Clean up dedup results after a delay
            asyncio.get_event_loop().call_later(5, self._dedup_results.pop, prompt_hash, None)
