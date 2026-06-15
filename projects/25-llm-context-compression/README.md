# Adaptive LLM Context Compression Engine

## Overview
A production middleware layer that dynamically compresses LLM context windows using extractive summarization, semantic deduplication, and importance-weighted token pruning.

## Architecture
```
[Input Context] → [Importance Scorer] → [Semantic Deduplicator]
 ↓
[Extractive Compressor] → [Token Pruner] → [Compressed Context]
 ↓
[Quality Monitor] → [Ratio Tuner] → [Feedback Loop]
```

## Key Features
- 40-60% token reduction with minimal information loss
- Semantic deduplication removes redundant context
- Importance scoring combines query relevance + positional weighting
- Adaptive compression ratio tuning via quality feedback loops

## Tech Stack
- **Claude API** — LLM inference target
- **tiktoken** — Accurate token counting
- **sentence-transformers** — Semantic similarity & deduplication
- **Redis** — Quality feedback cache
- **FastAPI** — Middleware API layer
- **NumPy** — Numerical operations for scoring
