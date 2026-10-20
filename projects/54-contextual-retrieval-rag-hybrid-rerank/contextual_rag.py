"""Contextual retrieval: each chunk is prefixed with an LLM-written sentence
that situates it inside its source document, then indexed in both BM25 and a
dense index. Results are fused with Reciprocal Rank Fusion and reranked."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import anthropic
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CONTEXT_PROMPT = """<document>
{doc}
</document>
Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>
Write one or two sentences that situate this chunk within the overall document
(entity, section, time period) to improve search retrieval. Answer with only the context."""


@dataclass
class Chunk:
    doc_id: str
    text: str
    context: str = ""

    @property
    def indexed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


def split(doc: str, size: int = 800, overlap: int = 120) -> list[str]:
    words, out, i = doc.split(), [], 0
    while i < len(words):
        out.append(" ".join(words[i:i + size]))
        i += size - overlap
    return out


class ContextualIndex:
    def __init__(self, embedder: str = "BAAI/bge-base-en-v1.5",
                 reranker: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.client = anthropic.Anthropic()
        self.embedder = SentenceTransformer(embedder)
        self.reranker = CrossEncoder(reranker)
        self.chunks: list[Chunk] = []

    def _contextualize(self, doc: str, chunk: str) -> str:
        # The full document goes in a cached system block, so N chunks of the
        # same doc pay full input price only once.
        resp = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=120,
            system=[{"type": "text", "text": "You write retrieval context.",
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": CONTEXT_PROMPT.format(doc=doc, chunk=chunk)}],
        )
        return resp.content[0].text.strip()

    def add(self, doc_id: str, doc: str) -> None:
        for piece in split(doc):
            self.chunks.append(Chunk(doc_id, piece, self._contextualize(doc, piece)))

    def build(self) -> None:
        corpus = [c.indexed_text for c in self.chunks]
        self.bm25 = BM25Okapi([t.lower().split() for t in corpus])
        self.vectors = self.embedder.encode(corpus, normalize_embeddings=True, batch_size=64)

    def _dense(self, query: str, k: int) -> list[int]:
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        return list(np.argsort(-self.vectors @ q)[:k])

    def _sparse(self, query: str, k: int) -> list[int]:
        return list(np.argsort(-self.bm25.get_scores(query.lower().split()))[:k])

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] += 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, query: str, top_k: int = 5, pool: int = 50) -> list[tuple[Chunk, float]]:
        fused = self.rrf([self._dense(query, pool), self._sparse(query, pool)])[:pool]
        pairs = [(query, self.chunks[i].indexed_text) for i in fused]
        scores = self.reranker.predict(pairs, batch_size=16)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(self.chunks[fused[i]], float(scores[i])) for i in order]

    def answer(self, query: str) -> str:
        hits = self.search(query)
        context = "\n\n".join(f"[{i + 1}] ({c.doc_id}) {c.text}" for i, (c, _) in enumerate(hits))
        resp = self.client.messages.create(
            model="claude-sonnet-4-5", max_tokens=800,
            messages=[{"role": "user", "content":
                       f"Answer using only the sources and cite them like [1].\n\n{context}\n\nQ: {query}"}],
        )
        return resp.content[0].text


if __name__ == "__main__":
    idx = ContextualIndex()
    idx.add("acme-10q-q2", open("data/acme_10q_q2_2026.txt").read())
    idx.build()
    print(idx.answer("How did ACME's revenue growth change versus the prior quarter?"))
