# SLO-Driven Autoscaler for LLM Inference Replicas

> **Category:** MLOps
> **Project #50** in the AI Engineer Portfolio

## Overview

A Kubernetes controller that scales vLLM replicas on the signals that matter for LLM serving: queue depth, time-to-first-token p95 and KV-cache pressure, instead of CPU. It sizes against a TTFT SLO with asymmetric cooldowns, one-step scale-down to protect warm caches, and a GPU cost ceiling.

## Architecture

```
[vLLM Pods] -> [Prometheus] -> [Signal Reader]
          |
[Queue / TTFT / KV Sizing] -> [Hysteresis + Cooldown] -> [Budget Cap]
          |
[K8s Scale API] -> [Deployment] -> [Grafana SLO Board]
```

## Tech Stack

Kubernetes, vLLM, Prometheus, Python, Grafana, Helm

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/50-slo-driven-llm-inference-autoscaler

# Install dependencies
pip install -r requirements.txt

# Run the project
python autoscaler.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
