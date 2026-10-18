# Citation-Grounded RAG with Claim-Level Verification

> **Category:** RAG
> **Project #48** in the AI Engineer Portfolio

## Overview

A RAG pipeline that forces a citation on every sentence, splits each sentence into atomic claims and checks them against the cited passage with an NLI cross-encoder. Unsupported claims are rewritten or dropped before the answer ships, and a per-answer groundedness score feeds hallucination monitoring.

## Architecture

```
[Query] -> [Qdrant Retriever] -> [Cited Generation [S#]]
          |
[Claim Decomposer] -> [NLI Cross-Encoder] -> [Support Check]
          |
[Repair Pass] -> [Groundedness Score] -> [Answer + Citations]
```

## Tech Stack

Claude API, Qdrant, sentence-transformers, DeBERTa-v3 NLI, FastAPI

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/48-citation-grounded-rag-claim-verification

# Install dependencies
pip install -r requirements.txt

# Run the project
python citation_rag.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
