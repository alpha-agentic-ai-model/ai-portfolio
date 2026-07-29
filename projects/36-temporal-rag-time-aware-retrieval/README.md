# Temporal RAG with Time-Aware Knowledge Retrieval

> **Category:** RAG  
> **Project #36** in the AI Engineer Portfolio

## Overview

A time-sensitive RAG system that indexes documents with temporal metadata, performs decay-weighted retrieval favoring recent information, and resolves temporal conflicts when multiple documents contain contradictory time-bound facts. Supports temporal query understanding and automatic knowledge expiration.

## Architecture

```
[Query] → [Temporal Parser] → [Time-Aware Retriever]
          ↓
[Decay Scorer] → [Conflict Resolver] → [Temporal Ranker]
          ↓
[Context Assembler] → [LLM + Time Grounding] → [Dated Answer]
```

## Tech Stack

LlamaIndex, Qdrant, Claude API, dateutil, FastAPI, Pydantic

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/36-temporal-rag-time-aware-retrieval

# Install dependencies
pip install -r requirements.txt

# Run the project
python temporal_retriever.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
