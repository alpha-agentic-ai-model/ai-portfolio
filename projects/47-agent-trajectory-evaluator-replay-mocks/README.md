# Agent Trajectory Evaluator with Replayable Tool Mocks

> **Category:** Agentic AI
> **Project #47** in the AI Engineer Portfolio

## Overview

An evaluation harness that scores how an agent reached its answer, not just whether it did: tool-call precision and recall, redundant calls, and recovery from injected tool failures. Tool outputs are replayed from recorded fixtures keyed by a hash of the call, so agent regressions are caught deterministically in CI without hitting live APIs.

## Architecture

```
[Eval Case Suite] -> [Agent Under Test] -> [Replay Toolbox (fixtures)]
          |
[Fault Injector] -> [Trajectory Recorder] -> [Scorer]
          |
[Precision / Recall] -> [Fault-Recovery Rate] -> [CI Gate]
```

## Tech Stack

Python, Claude Agent SDK, pytest, OpenTelemetry, DuckDB, GitHub Actions

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/47-agent-trajectory-evaluator-replay-mocks

# Install dependencies
pip install -r requirements.txt

# Run the project
python trajectory_eval.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
