from dataclasses import dataclass
from typing import Optional

@dataclass
class Proposition:
    text: str
    source_chunk_id: str
    confidence: float

@dataclass
class RetrievalResult:
    propositions: list[Proposition]
    reasoning_trace: str
    relevance_score: float

class VectorlessRetriever:
    """Reasoning-based retrieval without vector embeddings."""
    def __init__(self, llm, document_store):
        self.llm = llm
        self.store = document_store
        self.fts = FullTextSearch(document_store)

    async def extract_propositions(self, document: str) -> list[Proposition]:
        prompt = f"""Extract atomic propositions from this document.
        Each proposition should be a single, verifiable fact.
        Document: {document}"""
        response = await self.llm.generate(prompt)
        return self.parse_propositions(response)

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        # Step 1: BM25 candidate retrieval
        candidates = self.fts.search(query, limit=top_k * 5)

        # Step 2: Reasoning-based relevance scoring
        scored = []
        for chunk in candidates:
            relevance = await self.reason_about_relevance(query, chunk)
            if relevance.is_relevant:
                scored.append((chunk, relevance.score))

        # Step 3: Iterative refinement
        scored.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [c for c, _ in scored[:top_k]]
        refined = await self.iterative_refine(query, top_chunks)
        return RetrievalResult(
            propositions=refined,
            reasoning_trace=self.build_trace(),
            relevance_score=self.aggregate_score(scored)
        )

    async def reason_about_relevance(self, query, chunk):
        prompt = f"""Given the query: "{query}"
        Is this passage relevant? Explain your reasoning step by step.
        Passage: {chunk.text}"""
        return await self.llm.generate(prompt, schema=RelevanceJudgment)