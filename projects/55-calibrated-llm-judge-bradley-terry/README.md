# Calibrated LLM-as-Judge with Bradley-Terry Ranking

> **Category:** LLM Engineering
> **Project #55** in the AI Engineer Portfolio

## Overview

An evaluation harness that compares model outputs pairwise, asking the judge twice with positions swapped so only consistent verdicts count. Wins are fitted with a Bradley-Terry model to produce a global ranking of systems, and Kendall tau against a human-labelled anchor set reports how far the judge can be trusted.

## Architecture

```
[System Outputs] -> [Pairwise Sampler] -> [Judge (A,B)] -> [Judge (B,A)]
          |
[Consistency Filter] -> [Bradley-Terry MM Fit] -> [Ranking + Kendall tau vs Humans]
```

## Tech Stack

Claude API, NumPy, SciPy, Bradley-Terry, Python

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/55-calibrated-llm-judge-bradley-terry

# Install dependencies
pip install -r requirements.txt

# Run the project
python judge_calibration.py
```

## Author

**Manikanta Pudoka** — AI Engineer
