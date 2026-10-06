# Hybrid Mamba-Transformer for 128k Context Pretraining

> **Category:** Deep Learning
> **Project #45** in the AI Engineer Portfolio

## Overview

A language-model backbone that interleaves Mamba-2 selective state-space blocks with a periodic sliding-window attention layer, trading quadratic attention for linear-time recurrence on most layers. The retained attention layers preserve exact token-to-token recall, which is what pure SSM stacks lose on retrieval-style long-context evaluations.

## Architecture

```
[Tokens] -> [Embedding] -> [Selective Scan Block x5]
          |
[Sliding-Window Attention + RoPE] -> [RMSNorm]
          |
[Gated MLP] -> [Recurrent State Carry] -> [Tied LM Head]
```

## Tech Stack

PyTorch, Mamba-2, FlashAttention-2, einops, DeepSpeed, Triton

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/45-hybrid-mamba-transformer-long-context

# Install dependencies
pip install -r requirements.txt

# Run the project
python hybrid_ssm.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
