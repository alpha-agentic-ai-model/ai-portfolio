# Continuous Pre-Training Pipeline for Domain LLMs

## Overview
An end-to-end pipeline for continuously pre-training open-source LLMs on domain-specific corpora with catastrophic forgetting mitigation.

## Architecture
```
[Domain Corpus] → [Data Quality Filter] → [Curriculum Scheduler]
 ↓
[DeepSpeed ZeRO-3 Trainer] → [EWC Regularizer]
 ↓
[Eval Harness] → [Domain Benchmark] → [Model Registry]
```

## Key Features
- Curriculum learning with progressive data mixing strategies
- Elastic Weight Consolidation (EWC) prevents catastrophic forgetting
- DeepSpeed ZeRO-3 for distributed training across multi-GPU clusters
- Automated data quality filtering removes low-quality training samples
- Continuous evaluation against domain-specific benchmarks

## Tech Stack
- **PyTorch** — Training framework
- **DeepSpeed** — ZeRO-3 distributed optimization
- **HuggingFace** — Model & tokenizer loading
- **Weights & Biases** — Experiment tracking
- **datasets** — Data loading & preprocessing
- **CUDA** — GPU acceleration
