# Multi-Head Latent Attention with Compressed KV Cache

> **Category:** Deep Learning
> **Project #63** in the AI Engineer Portfolio

## Overview

A PyTorch implementation of multi-head latent attention, where keys and values are rebuilt from one shared low-rank latent and a decoupled RoPE key. The inference cache stores only that latent per token, cutting KV memory by about 7x versus standard multi-head attention in the default config.

## Architecture

```
[Hidden States] -> [Q Down/Up Proj] -> [Content + RoPE Query]
          |
[KV Down Proj] -> [Latent c_kv + RoPE Key] -> [KV Cache (latent only)]
          |
[K/V Up Proj] -> [SDPA] -> [Output Proj]
```

## Tech Stack

PyTorch, RoPE, RMSNorm, SDPA, bf16

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/63-multi-head-latent-attention-compressed-kv

# Install dependencies
pip install -r requirements.txt

# Run the project
python mla.py
```

## Author

**Manikanta Pudoka** — AI Engineer
