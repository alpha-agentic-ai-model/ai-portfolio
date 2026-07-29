# Mixture of Experts Router with Learned Gating

> **Category:** Deep Learning  
> **Project #39** in the AI Engineer Portfolio

## Overview

A custom Mixture of Experts (MoE) transformer layer with a learned top-k gating network that dynamically routes tokens to specialized expert sub-networks. Implements load-balanced routing with auxiliary loss, expert capacity constraints, and gradient-based expert utilization tracking for efficient sparse model training.

## Architecture

```
[Input Tokens] → [Gating Network] → [Top-K Expert Selection]
          ↓
[Expert 1] | [Expert 2] | ... | [Expert N]
          ↓
[Weighted Combine] → [Load Balancer Loss] → [Output]
```

## Tech Stack

PyTorch, DeepSpeed, CUDA, Weights & Biases, Hugging Face, einops

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/39-moe-router-learned-gating

# Install dependencies
pip install -r requirements.txt

# Run the project
python moe_layer.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
