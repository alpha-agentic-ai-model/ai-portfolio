from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import BM25Retriever
import chromadb


class HybridRetriever:
    def __init__(self, vector_store, sparse_index, alpha=0.7):
        self.vector_retriever = vector_store.as_retriever(similarity_top_k=10)
        self.sparse_retriever = BM25Retriever.from_defaults(
            index=sparse_index, similarity_top_k=10
        )
        self.alpha = alpha

    async def retrieve(self, queries: list[str]):
        all_docs = []
        for query in queries:
            semantic = await self.vector_retriever.aretrieve(query)
            sparse = await self.sparse_retriever.aretrieve(query)
            merged = self._reciprocal_rank_fusion(semantic, sparse)
            all_docs.extend(merged)
        return self._deduplicate(all_docs)

    def _reciprocal_rank_fusion(self, semantic, sparse, k=60):
        scores = {}
        for rank, doc in enumerate(semantic):
            scores[doc.id_] = self.alpha / (k + rank)
        for rank, doc in enumerate(sparse):
            scores[doc.id_] = scores.get(doc.id_, 0) + (1 - self.alpha) / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HallucinationGrader:
    def __init__(self, llm):
        self.llm = llm

    async def verify(self, answer: str, context: list) -> bool:
        prompt = f"""Given the context, determine if the answer is grounded in facts.
        Context: {context}
        Answer: {answer}
        Return JSON: {{"grounded": true/false, "confidence": 0.0-1.0}}"""
        result = await self.llm.agenerate(prompt)
        return result["grounded"] and result["confidence"] > 0.8


class AgenticRAG:
    def __init__(self, llm, vector_store, bm25_index):
        self.llm = llm
        self.hybrid_retriever = HybridRetriever(
            vector=vector_store, sparse=bm25_index, alpha=0.7
        )
        self.reranker = CohereRerank(top_n=5)
        self.hallucination_checker = HallucinationGrader(llm)

    async def query(self, question: str, max_retries: int = 2):
        sub_queries = await self.decompose(question)
        docs = await self.hybrid_retriever.retrieve(sub_queries)
        ranked = self.reranker.rerank(question, docs)

        for attempt in range(max_retries):
            answer = await self.llm.generate(question, ranked)
            if await self.hallucination_checker.verify(answer, ranked):
                return {"answer": answer, "sources": ranked, "attempts": attempt + 1}
            ranked = await self.re_retrieve(question)

        return {"answer": answer, "sources": ranked, "warning": "max retries reached"}
