"""Online retrieval-quality monitor for production RAG.

Samples live (query, retrieved chunks) pairs, labels chunk relevance with an
LLM grader, and tracks precision@k, nDCG@k and an embedding-drift score per
time window. Fires an alert when a metric drops beyond a CUSUM threshold, so
index regressions surface before users notice degraded answers.
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from anthropic import Anthropic
from prometheus_client import Gauge

client = Anthropic()
GRADER_MODEL = "claude-haiku-4-5"

PRECISION = Gauge("rag_precision_at_k", "LLM-graded precision@k", ["index"])
NDCG = Gauge("rag_ndcg_at_k", "LLM-graded nDCG@k", ["index"])
DRIFT = Gauge("rag_query_embedding_drift", "Cosine distance of query centroid vs baseline", ["index"])


@dataclass
class RetrievalEvent:
    query: str
    query_embedding: np.ndarray
    chunks: list[str]


@dataclass
class CusumDetector:
    """One-sided CUSUM that flags sustained downward shifts from a target mean."""
    target: float
    slack: float = 0.02
    threshold: float = 0.25
    s: float = 0.0

    def update(self, x: float) -> bool:
        self.s = max(0.0, self.s + (self.target - x - self.slack))
        return self.s > self.threshold


def grade_chunks(query: str, chunks: list[str]) -> list[int]:
    """Return graded relevance 0/1/2 for each chunk, in order."""
    listing = "\n\n".join(f"[{i}] {c[:1200]}" for i, c in enumerate(chunks))
    prompt = (
        f"Query: {query}\n\nPassages:\n{listing}\n\n"
        "Rate each passage: 2 = directly answers, 1 = partially relevant, 0 = irrelevant.\n"
        'Reply with JSON only: {"grades": [int, ...]}'
    )
    resp = client.messages.create(model=GRADER_MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}])
    text = resp.content[0].text
    grades = json.loads(text[text.index("{"): text.rindex("}") + 1])["grades"]
    return [int(max(0, min(2, g))) for g in grades[: len(chunks)]]


def precision_at_k(grades: list[int]) -> float:
    return sum(g > 0 for g in grades) / max(1, len(grades))


def ndcg_at_k(grades: list[int]) -> float:
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    ideal = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(grades, reverse=True)))
    return dcg / ideal if ideal > 0 else 0.0


@dataclass
class RetrievalMonitor:
    index_name: str
    baseline_centroid: np.ndarray
    baseline_ndcg: float
    sample_rate: float = 0.05
    window: int = 200
    ndcg_hist: deque = field(default_factory=lambda: deque(maxlen=200))
    prec_hist: deque = field(default_factory=lambda: deque(maxlen=200))
    emb_hist: deque = field(default_factory=lambda: deque(maxlen=200))
    alerts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cusum = CusumDetector(target=self.baseline_ndcg)

    def observe(self, ev: RetrievalEvent) -> None:
        self.emb_hist.append(ev.query_embedding / np.linalg.norm(ev.query_embedding))
        if random.random() > self.sample_rate or not ev.chunks:
            return
        grades = grade_chunks(ev.query, ev.chunks)
        n = ndcg_at_k(grades)
        self.ndcg_hist.append(n)
        self.prec_hist.append(precision_at_k(grades))
        if self.cusum.update(n):
            self.alerts.append(f"{self.index_name}: sustained nDCG drop (cusum={self.cusum.s:.2f})")
            self.cusum.s = 0.0
        self.publish()

    def drift(self) -> float:
        if len(self.emb_hist) < 20:
            return 0.0
        centroid = np.mean(self.emb_hist, axis=0)
        centroid /= np.linalg.norm(centroid)
        return float(1.0 - centroid @ self.baseline_centroid)

    def publish(self) -> None:
        PRECISION.labels(self.index_name).set(float(np.mean(self.prec_hist)))
        NDCG.labels(self.index_name).set(float(np.mean(self.ndcg_hist)))
        DRIFT.labels(self.index_name).set(self.drift())


def wrap_retriever(retriever, embed, monitor: RetrievalMonitor):
    """Decorate any retriever(query, k) -> list[str] so every call is monitored."""
    def monitored(query: str, k: int = 5) -> list[str]:
        chunks = retriever(query, k)
        try:
            monitor.observe(RetrievalEvent(query, np.asarray(embed(query)), chunks))
        except Exception as exc:  # monitoring must never break serving
            monitor.alerts.append(f"monitor error: {exc}")
        return chunks
    return monitored
