# Online Retrieval Quality Monitor with LLM-Graded Relevance

> **Category:** RAG
> **Project #60** in the AI Engineer Portfolio

## Overview

Samples live RAG traffic, grades each retrieved chunk with an LLM, and tracks precision@k, nDCG@k and query-embedding drift as Prometheus metrics. A CUSUM detector raises an alert on a sustained nDCG drop, so a bad re-index or embedding change shows up before answer quality complaints do.

## Architecture

```
[Retriever Call] -> [Monitor Wrapper] -> [Sampler (5%)]
          |
[LLM Relevance Grader] -> [precision@k / nDCG@k] -> [CUSUM Detector]
          |
[Embedding Centroid Drift] -> [Prometheus] -> [Alerts]
```

## Tech Stack

Claude API, NumPy, Prometheus, CUSUM, Python

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/60-rag-retrieval-quality-monitor-drift

# Install dependencies
pip install -r requirements.txt

# Run the project
python retrieval_monitor.py
```

## Author

**Manikanta Pudoka** — AI Engineer
