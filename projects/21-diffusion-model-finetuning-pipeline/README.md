# Diffusion Model Fine-Tuning with DreamBooth & LoRA

## Overview
An end-to-end pipeline for fine-tuning Stable Diffusion models using DreamBooth and LoRA adapters on custom image datasets. Supports automated dataset curation, prior-preservation training, hyperparameter sweeps, and one-click deployment with A/B model comparison.

## Architecture
```
[Image Dataset] → [Auto Captioner] → [Data Augmenter]
       ↓
[DreamBooth + LoRA Trainer] → [W&B Sweep]
       ↓
[Model Registry] → [Gradio Endpoint] → [A/B Compare]
```

## Tech Stack
- **Diffusers** — Hugging Face diffusion model library
- **PyTorch** — Deep learning framework
- **LoRA / PEFT** — Parameter-efficient fine-tuning
- **Weights & Biases** — Experiment tracking and sweeps
- **Gradio** — Inference UI and model comparison
- **CUDA** — GPU-accelerated training

## Key Features
- DreamBooth training with LoRA for memory-efficient fine-tuning
- Automated dataset augmentation and captioning
- Prior-preservation loss for subject fidelity
- W&B integration for hyperparameter sweeps
- Cosine annealing with warmup scheduling
- Checkpoint saving and model registry integration
