# Late-Interaction Retrieval Engine with Residual Compression

> **Category:** RAG
> **Project #42** in the AI Engineer Portfolio

## Overview

A ColBERT-style multi-vector retriever that scores queries against per-token document embeddings via MaxSim instead of collapsing passages into one vector. A PLAID-inspired two-stage index prunes candidates by centroid, then rescores exactly on 2-bit residual codes, recovering the recall of late interaction at single-vector storage cost.

## Architecture

```
[Query] -> [Token Encoder] -> [128-d Normalised Vectors]
          |
[Centroid Pruning (IVF)] -> [Candidate Depth 1024]
          |
[Residual Decode] -> [Exact MaxSim Rescore] -> [Cross-Encoder Rerank]
```

## Tech Stack

PyTorch, ColBERTv2, PLAID, FAISS, NumPy, FastAPI

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/42-late-interaction-retrieval-engine

# Install dependencies
pip install -r requirements.txt

# Run the project
python late_interaction_engine.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
