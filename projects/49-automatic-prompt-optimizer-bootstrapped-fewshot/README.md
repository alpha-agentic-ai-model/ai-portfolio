# Automatic Prompt Optimizer with Bootstrapped Few-Shot Search

> **Category:** LLM Engineering
> **Project #49** in the AI Engineer Portfolio

## Overview

Treats a prompt as a compilable program of instruction plus demonstrations, in the spirit of DSPy. Demos are bootstrapped from the model's own correct traces, a meta-prompt proposes instruction variants, and successive halving on a dev split selects the best combination at a fraction of a full grid search's cost.

## Architecture

```
[Train Split] -> [Bootstrap Demos (correct traces)] -> [Demo Pool]
          |
[Meta-Prompt] -> [Instruction Candidates] -> [Candidate Sampler]
          |
[Successive Halving (dev)] -> [Metric] -> [Compiled Prompt]
```

## Tech Stack

Claude API, DSPy concepts, Python, Optuna, MLflow

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/49-automatic-prompt-optimizer-bootstrapped-fewshot

# Install dependencies
pip install -r requirements.txt

# Run the project
python prompt_optimizer.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
