# Streaming RAG with Real-Time Index Updates

## Overview
A RAG pipeline that ingests documents via Kafka streams and maintains a real-time vector index with zero-downtime updates. Supports hybrid BM25+dense retrieval with automatic chunk versioning and stale-document eviction.

## Architecture
```
[Kafka Stream] → [Chunker] → [Embedder]
  |
[Vector DB] → [Index Manager] → [BM25 Index] → [Hybrid Retriever]
```

## Tech Stack
Kafka, Qdrant, LangChain, sentence-transformers, FastAPI, Python

## Key Features
- Production-ready implementation with error handling
- Comprehensive type annotations and documentation
- Modular architecture for easy extension
- Built for scalability and performance
