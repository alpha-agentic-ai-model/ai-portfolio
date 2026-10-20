# Multi-Tenant LoRA Adapter Serving with LRU Hot-Swap

> **Category:** MLOps
> **Project #56** in the AI Engineer Portfolio

## Overview

A serving gateway that runs one shared base model on vLLM and loads each tenant's LoRA adapter on demand from S3 into a bounded pool of GPU slots. Idle adapters are evicted least-recently-used, busy ones are protected by in-flight counts, and API keys map requests to the right adapter so dozens of fine-tunes share one GPU.

## Architecture

```
[Client + API Key] -> [FastAPI Gateway] -> [Tenant Lookup]
          |
[Adapter Pool (LRU)] -> [S3 Adapter Store]
          |
[vLLM Base Model + LoRA] -> [Response]
```

## Tech Stack

vLLM, FastAPI, boto3 / S3, LoRA, asyncio

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/56-multi-tenant-lora-adapter-serving

# Install dependencies
pip install -r requirements.txt

# Run the project
python lora_router.py
```

## Author

**Manikanta Pudoka** — AI Engineer
