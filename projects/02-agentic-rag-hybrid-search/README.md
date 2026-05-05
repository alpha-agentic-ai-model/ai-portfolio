# Agentic RAG with Hybrid Search & Self-Correction

An advanced RAG pipeline that implements hybrid search (BM25 + semantic), cross-encoder re-ranking, query decomposition, and a self-corrective loop that detects hallucinations and re-retrieves context.

## Architecture
```
[Query] -> [Decomposer] -> [Hybrid Search (BM25 + Semantic)]
                                    |
          [Re-ranker] -> [Generator] -> [Hallucination Check]
                                              |
                                    (if fails) -> [Re-retrieve]
```

## Tech Stack
- LlamaIndex, ChromaDB, Cohere Rerank, FastAPI, Claude API

## Key Features
- Hybrid retrieval with configurable alpha weighting
- Cross-encoder re-ranking for precision
- Self-corrective generation with hallucination grading
- Query decomposition for complex questions
