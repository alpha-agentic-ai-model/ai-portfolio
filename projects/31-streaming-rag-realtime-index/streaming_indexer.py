"""Streaming RAG with Real-Time Index Updates."""
import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    version: int
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class Document:
    id: str
    content: str
    source: str
    timestamp: float
    version: int = 1


class ChunkingEngine:
    """Split documents into overlapping chunks with versioning."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"{doc.id}:{idx}:{doc.version}".encode()
                ).hexdigest()[:16]
                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    doc_id=doc.id,
                    version=doc.version,
                    metadata={"source": doc.source, "chunk_idx": idx},
                ))
                idx += 1
            start = end - self.overlap

        return chunks


class EmbeddingService:
    """Generate embeddings with batching support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Simulated embeddings (replace with real model)
        return [
            np.random.randn(self.dim).tolist()
            for _ in texts
        ]

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query])[0]


class VectorIndex:
    """In-memory vector store with upsert and eviction."""

    def __init__(self):
        self.vectors: dict[str, tuple[list[float], dict]] = {}
        self.doc_chunks: dict[str, set[str]] = {}

    def upsert(self, chunk_id: str, vector: list[float], metadata: dict):
        self.vectors[chunk_id] = (vector, metadata)
        doc_id = metadata.get("doc_id", "")
        self.doc_chunks.setdefault(doc_id, set()).add(chunk_id)

    def evict_document(self, doc_id: str):
        chunk_ids = self.doc_chunks.pop(doc_id, set())
        for cid in chunk_ids:
            self.vectors.pop(cid, None)
        return len(chunk_ids)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        scores = []
        qv = np.array(query_vector)
        for cid, (vec, meta) in self.vectors.items():
            sim = float(np.dot(qv, np.array(vec)) / (
                np.linalg.norm(qv) * np.linalg.norm(vec) + 1e-8
            ))
            scores.append({"id": cid, "score": sim, **meta})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    @property
    def size(self) -> int:
        return len(self.vectors)


class BM25Index:
    """Keyword-based BM25 index for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: dict[str, list[str]] = {}
        self.avg_dl: float = 0.0

    def add(self, doc_id: str, text: str):
        tokens = text.lower().split()
        self.docs[doc_id] = tokens
        total = sum(len(t) for t in self.docs.values())
        self.avg_dl = total / max(len(self.docs), 1)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = query.lower().split()
        scores = []
        for doc_id, tokens in self.docs.items():
            score = self._score(query_tokens, tokens)
            scores.append({"id": doc_id, "score": score})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        dl = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = doc_tokens.count(qt)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
            score += numerator / max(denominator, 1e-8)
        return score


class HybridRetriever:
    """Combine dense vector search with BM25 for hybrid retrieval."""

    def __init__(self, alpha: float = 0.7):
        self.vector_index = VectorIndex()
        self.bm25_index = BM25Index()
        self.embedder = EmbeddingService()
        self.alpha = alpha  # weight for dense vs sparse

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qvec = self.embedder.encode_query(query)
        dense = self.vector_index.search(qvec, top_k=top_k * 2)
        sparse = self.bm25_index.search(query, top_k=top_k * 2)

        # Reciprocal rank fusion
        fused = {}
        for rank, item in enumerate(dense):
            fused[item["id"]] = self.alpha / (rank + 60)
        for rank, item in enumerate(sparse):
            fused.setdefault(item["id"], 0)
            fused[item["id"]] += (1 - self.alpha) / (rank + 60)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [{"id": cid, "score": score} for cid, score in ranked[:top_k]]


class StreamingRAGIndexer:
    """Main indexer that processes document streams."""

    def __init__(self):
        self.chunker = ChunkingEngine()
        self.retriever = HybridRetriever()
        self.processed = 0

    def ingest(self, doc: Document):
        # Evict old version
        self.retriever.vector_index.evict_document(doc.id)
        # Chunk and embed
        chunks = self.chunker.chunk(doc)
        texts = [c.text for c in chunks]
        embeddings = self.retriever.embedder.encode(texts)
        # Index
        for chunk, emb in zip(chunks, embeddings):
            self.retriever.vector_index.upsert(
                chunk.id, emb,
                {"doc_id": doc.id, "text": chunk.text, **chunk.metadata}
            )
            self.retriever.bm25_index.add(chunk.id, chunk.text)
        self.processed += 1

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        return self.retriever.retrieve(text, top_k)


if __name__ == "__main__":
    indexer = StreamingRAGIndexer()
    for i in range(5):
        doc = Document(
            id=f"doc-{i}", content=f"AI document {i} about machine learning " * 50,
            source="stream", timestamp=time.time(), version=1
        )
        indexer.ingest(doc)

    results = indexer.query("machine learning techniques")
    print(f"Indexed {indexer.processed} docs, {indexer.retriever.vector_index.size} chunks")
    print(f"Top results: {results[:3]}")
