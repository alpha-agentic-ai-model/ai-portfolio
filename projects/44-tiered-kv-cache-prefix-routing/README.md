# Tiered KV-Cache Service with Prefix-Aware Routing

> **Category:** MLOps
> **Project #44** in the AI Engineer Portfolio

## Overview

A fleet-level KV-cache control plane that content-addresses prompt prefixes into 16-token blocks and cascades them HBM to DRAM to NVMe to object store rather than discarding them. A radix trie over block hashes routes each request to the replica that already holds the longest matching prefix, turning repeated system prompts into near-zero prefill cost.

## Architecture

```
[Prompt Tokens] -> [Rolling Block Hash] -> [Prefix Radix Trie]
          |
[Replica Router (longest match)] -> [Admission Control]
          |
[HBM] -> [DRAM] -> [NVMe] -> [Object Store] -> [Prometheus]
```

## Tech Stack

vLLM, LMCache, Redis, Kubernetes, Prometheus, NumPy

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/44-tiered-kv-cache-prefix-routing

# Install dependencies
pip install -r requirements.txt

# Run the project
python kv_tier_manager.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
