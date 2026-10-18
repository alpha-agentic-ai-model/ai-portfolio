# TopK Sparse Autoencoder for LLM Interpretability

> **Category:** Deep Learning
> **Project #51** in the AI Engineer Portfolio

## Overview

Trains a TopK sparse autoencoder on residual-stream activations so individual latents line up with human-interpretable features. Includes unit-norm decoder columns, dead-latent resampling from high-error tokens, fraction-of-variance-unexplained tracking and a top-activating-context viewer for labelling features.

## Architecture

```
[Text Corpus] -> [HookedTransformer] -> [Residual Stream (layer L)]
          |
[TopK Encoder] -> [Sparse Latents] -> [Linear Decoder]
          |
[MSE / FVU] -> [Dead-Latent Resampler] -> [Feature Explorer]
```

## Tech Stack

PyTorch, TransformerLens, CUDA, einops, Weights & Biases

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/51-sparse-autoencoder-llm-interpretability

# Install dependencies
pip install -r requirements.txt

# Run the project
python topk_sae.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
