# Speculative Decoding Server with Adaptive Draft Selection

> **Category:** LLM Engineering
> **Project #43** in the AI Engineer Portfolio

## Overview

A draft-and-verify decoding loop that keeps generation lossless through rejection sampling while a Thompson-sampling bandit picks the draft model and speculation depth per request. Acceptance rate and draft latency are tracked online, so gamma walks toward its 1/(1-alpha) optimum instead of sitting at a hand-tuned constant that degrades on long prefills.

## Architecture

```
[Request] -> [Draft Bandit] -> [Draft Model (gamma tokens)]
          |
[Target Verify Pass] -> [Rejection Sampler]
          |
[Accepted Prefix + Bonus Token] -> [KV Rollback] -> [Gamma Tuner]
```

## Tech Stack

vLLM, PyTorch, EAGLE-2, CUDA, Ray Serve, Triton

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/43-speculative-decoding-adaptive-drafts

# Install dependencies
pip install -r requirements.txt

# Run the project
python speculative_engine.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
