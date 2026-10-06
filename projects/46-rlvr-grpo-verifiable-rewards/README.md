# RLVR Training Loop with GRPO and Sandboxed Verifiers

> **Category:** LLM Engineering
> **Project #46** in the AI Engineer Portfolio

## Overview

A reinforcement-learning-from-verifiable-rewards pipeline where reward comes from executing the model's own code against hidden tests in a network-isolated sandbox, not from a learned reward model. GRPO replaces the value network with group-relative advantages, and a pass-rate gate promotes the curriculum once the policy clears the difficulty tier it is training on.

## Architecture

```
[Prompt Batch] -> [vLLM Sampler (group of 8)] -> [Rollouts]
          |
[Unit-Test Verifier (sandbox)] -> [Format Verifier] -> [Reward Router]
          |
[Group-Relative Advantage] -> [Clipped PG + KL] -> [Weight Sync]
```

## Tech Stack

PyTorch, TRL GRPO, vLLM, Docker, Ray, Weights & Biases

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/46-rlvr-grpo-verifiable-rewards

# Install dependencies
pip install -r requirements.txt

# Run the project
python rlvr_grpo.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
