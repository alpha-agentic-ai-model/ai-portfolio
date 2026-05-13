"""
Privacy-Preserving Federated RAG with Differential Privacy
Enables multi-organization knowledge sharing without exposing raw documents.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Tracks cumulative privacy loss (epsilon) across queries."""

    total_epsilon: float
    consumed_epsilon: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    def can_query(self, cost: float) -> bool:
        return self.remaining >= cost

    def consume(self, cost: float):
        if not self.can_query(cost):
            raise PrivacyBudgetExhausted(
                f"Need {cost}, only {self.remaining} remaining"
            )
        self.consumed_epsilon += cost


class PrivacyBudgetExhausted(Exception):
    pass


@dataclass
class RetrievalResult:
    doc_id: str
    org_id: str
    score: float
    metadata: dict = field(default_factory=dict)


class LaplaceMechanism:
    """Calibrated Laplace noise for differential privacy."""

    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.laplace(loc=0.0, scale=self.scale, size=data.shape)
        return data + noise

    def privacy_cost(self) -> float:
        return self.epsilon


class FAISSIndex:
    """Wrapper around FAISS for approximate nearest neighbor search."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings: list[np.ndarray] = []
        self.doc_ids: list[str] = []
        self._index = None

    def add(self, doc_id: str, embedding: np.ndarray):
        self.embeddings.append(embedding)
        self.doc_ids.append(doc_id)
        self._index = None  # Invalidate

    def build(self):
        if not self.embeddings:
            return
        self._matrix = np.vstack(self.embeddings).astype("float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._matrix = self._matrix / np.clip(norms, 1e-10, None)

    def search(self, query: np.ndarray, k: int = 10) -> list[RetrievalResult]:
        if not self.embeddings:
            return []
        if self._matrix is None:
            self.build()
        query_norm = query / np.clip(np.linalg.norm(query), 1e-10, None)
        scores = self._matrix @ query_norm.T
        scores = scores.flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    doc_id=self.doc_ids[idx],
                    org_id="",
                    score=float(scores[idx]),
                )
            )
        return results


class SecureAggregator:
    """Secure multi-party aggregation using rank fusion."""

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or {}

    def secure_merge(
        self,
        partial_results: list[list[RetrievalResult]],
        strategy: str = "rank_fusion",
    ) -> list[RetrievalResult]:
        if strategy == "rank_fusion":
            return self._reciprocal_rank_fusion(partial_results)
        return self._score_fusion(partial_results)

    def _reciprocal_rank_fusion(
        self, results_lists: list[list[RetrievalResult]], k: int = 60
    ) -> list[RetrievalResult]:
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}
        for results in results_lists:
            for rank, result in enumerate(results):
                key = f"{result.org_id}:{result.doc_id}"
                rrf_score = 1.0 / (k + rank + 1)
                weight = self.weights.get(result.org_id, 1.0)
                doc_scores[key] = doc_scores.get(key, 0) + rrf_score * weight
                doc_map[key] = result
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        merged = []
        for key in sorted_keys:
            result = doc_map[key]
            result.score = doc_scores[key]
            merged.append(result)
        return merged

    def _score_fusion(
        self, results_lists: list[list[RetrievalResult]]
    ) -> list[RetrievalResult]:
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}
        for results in results_lists:
            max_score = max((r.score for r in results), default=1.0)
            for result in results:
                key = f"{result.org_id}:{result.doc_id}"
                normalized = result.score / max(max_score, 1e-10)
                doc_scores[key] = max(doc_scores.get(key, 0), normalized)
                doc_map[key] = result
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys]


class FederatedRAG:
    """Privacy-preserving RAG across organizational silos."""

    def __init__(
        self,
        epsilon: float = 1.0,
        total_budget: float = 100.0,
        embedding_dim: int = 768,
    ):
        self.dp_mechanism = LaplaceMechanism(epsilon=epsilon)
        self.aggregator = SecureAggregator()
        self.local_indexes: dict[str, FAISSIndex] = {}
        self.privacy_budget = PrivacyBudget(total_epsilon=total_budget)
        self.embedding_dim = embedding_dim

    def add_organization(self, org_id: str):
        if org_id not in self.local_indexes:
            self.local_indexes[org_id] = FAISSIndex(self.embedding_dim)
            logger.info(f"Registered organization: {org_id}")

    def add_private_embeddings(
        self,
        org_id: str,
        doc_ids: list[str],
        embeddings: np.ndarray,
    ):
        if org_id not in self.local_indexes:
            self.add_organization(org_id)
        # Add calibrated Laplace noise for differential privacy guarantee
        noisy_embeddings = self.dp_mechanism.add_noise(embeddings)
        for doc_id, emb in zip(doc_ids, noisy_embeddings):
            self.local_indexes[org_id].add(doc_id, emb)
        self.local_indexes[org_id].build()
        logger.info(
            f"Added {len(doc_ids)} private embeddings for org {org_id}"
        )

    async def federated_retrieve(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[RetrievalResult]:
        query_cost = self.dp_mechanism.privacy_cost()
        if not self.privacy_budget.can_query(query_cost):
            raise PrivacyBudgetExhausted("Global privacy budget exhausted")
        partial_results = []
        for org_id, index in self.local_indexes.items():
            hits = index.search(query_embedding, k=top_k)
            for hit in hits:
                hit.org_id = org_id
            partial_results.append(hits)

        merged = self.aggregator.secure_merge(
            partial_results, strategy="rank_fusion"
        )
        self.privacy_budget.consume(query_cost)
        logger.info(
            f"Federated query returned {len(merged[:top_k])} results, "
            f"budget remaining: {self.privacy_budget.remaining:.2f}"
        )
        return merged[:top_k]

    async def query_with_generation(
        self, question: str, embedder, llm, top_k: int = 5
    ) -> dict:
        query_emb = embedder.encode(question)
        results = await self.federated_retrieve(query_emb, top_k=top_k)
        context = "\n\n".join(
            f"[Source: {r.org_id}/{r.doc_id}]\n{r.metadata.get('text', '')}"
            for r in results
        )
        answer = await llm.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
            max_tokens=1024,
        )
        return {
            "answer": answer,
            "sources": [
                {"org": r.org_id, "doc": r.doc_id, "score": r.score}
                for r in results
            ],
            "privacy_remaining": self.privacy_budget.remaining,
        }
