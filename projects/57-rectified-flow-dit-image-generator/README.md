# Rectified Flow Image Generator with a Diffusion Transformer

> **Category:** Deep Learning
> **Project #57** in the AI Engineer Portfolio

## Overview

A class-conditional image generator trained with flow matching: a DiT with adaLN-Zero blocks learns the velocity that moves Gaussian noise to data along straight lines. Logit-normal timestep sampling, EMA weights and classifier-free guidance let it produce CIFAR-10 samples in about 8 Euler steps instead of hundreds.

## Architecture

```
[Noise x0] -> [Interpolate xt = (1-t)x0 + t x1] -> [DiT (adaLN-Zero)]
          |
[Velocity Loss v = x1 - x0] -> [EMA Weights] -> [8-step Euler + CFG Sampler]
```

## Tech Stack

PyTorch, einops, torchvision, bf16 Autocast, Flow Matching

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/57-rectified-flow-dit-image-generator

# Install dependencies
pip install -r requirements.txt

# Run the project
python rectified_flow.py
```

## Author

**Manikanta Pudoka** — AI Engineer
