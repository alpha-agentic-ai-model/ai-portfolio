# Contextual Retrieval RAG with Hybrid Fusion and Reranking

> **Category:** RAG
> **Project #54** in the AI Engineer Portfolio

## Overview

Before indexing, each chunk is prefixed with a short LLM-written sentence that places it in its source document (entity, section, period), using prompt caching so the full document is paid for once. Chunks go into both BM25 and a dense index, results are merged with Reciprocal Rank Fusion, and a cross-encoder reranks the pool before cited answer generation.

## Architecture

```
[Document] -> [Chunker] -> [Claude Haiku Context Writer (cached)]
          |
[BM25] -> [Dense bge Embeddings] -> [RRF Fusion]
          |
[Cross-Encoder Rerank] -> [Cited Answer]
```

## Tech Stack

Claude API, rank-bm25, sentence-transformers, BGE Reranker, NumPy

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/54-contextual-retrieval-rag-hybrid-rerank

# Install dependencies
pip install -r requirements.txt

# Run the project
python contextual_rag.py
```

## Author

**Manikanta Pudoka** — AI Engineer
