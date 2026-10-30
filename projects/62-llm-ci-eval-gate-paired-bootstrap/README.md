# CI Evaluation Gate for LLM Changes with Paired Bootstrap

> **Category:** MLOps
> **Project #62** in the AI Engineer Portfolio

## Overview

Runs a baseline and a candidate prompt or model over the same eval set, scores each example with deterministic checks and a rubric grader, and computes a paired-bootstrap confidence interval per metric. The build fails only when a regression is statistically clear, and the result table is posted to the GitHub Actions summary.

## Architecture

```
[Pull Request] -> [GitHub Actions] -> [Eval Dataset (JSONL)]
          |
[Baseline Variant] -> [Candidate Variant] -> [Scorers + Rubric Grader]
          |
[Paired Bootstrap CI] -> [Pass / Fail Gate] -> [Step Summary]
```

## Tech Stack

GitHub Actions, Claude API, NumPy, PyYAML, Paired Bootstrap

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/62-llm-ci-eval-gate-paired-bootstrap

# Install dependencies
pip install -r requirements.txt

# Run the project
python eval_gate.py
```

## Author

**Manikanta Pudoka** — AI Engineer
