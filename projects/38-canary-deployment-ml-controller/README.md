# Canary Deployment Controller for ML Models

> **Category:** MLOps  
> **Project #38** in the AI Engineer Portfolio

## Overview

An intelligent ML model deployment controller that implements canary releases with automatic traffic shifting based on real-time performance metrics. Monitors prediction quality, latency percentiles, and error rates during gradual rollouts, with automatic rollback if the canary model underperforms the baseline by configurable thresholds.

## Architecture

```
[New Model] → [Canary Controller] → [Traffic Splitter (5%→100%)]
          ↓
[Baseline Metrics] ↔ [Canary Metrics] → [Comparator]
          ↓
[Auto-Promote / Rollback] → [Model Registry] → [Alert Manager]
```

## Tech Stack

Kubernetes, Istio, MLflow, Prometheus, Python, FastAPI

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/38-canary-deployment-ml-controller

# Install dependencies
pip install -r requirements.txt

# Run the project
python canary_controller.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
