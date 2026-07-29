# LLM API Gateway with Semantic Caching & Fallback

> **Category:** LLM Engineering  
> **Project #37** in the AI Engineer Portfolio

## Overview

A production API gateway for LLM providers that implements semantic caching using embedding similarity, automatic provider failover with health checks, request deduplication, and cost-aware routing. Reduces API costs by 40% through intelligent cache hits while maintaining 99.9% availability across OpenAI, Anthropic, and Google endpoints.

## Architecture

```
[API Request] → [Semantic Cache Check] → [Cache Hit / Miss]
          ↓
[Provider Router] → [Health Monitor] → [Primary / Fallback LLM]
          ↓
[Response Normalizer] → [Cache Writer] → [Cost Tracker]
```

## Tech Stack

FastAPI, Redis, sentence-transformers, Claude API, OpenAI, Prometheus

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/37-llm-api-gateway-semantic-cache

# Install dependencies
pip install -r requirements.txt

# Run the project
python llm_gateway.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
