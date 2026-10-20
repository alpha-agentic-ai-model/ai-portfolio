# Durable Agent Executor with Checkpointed Step Replay

> **Category:** Agentic AI
> **Project #53** in the AI Engineer Portfolio

## Overview

A long-running agent runtime that journals every model call and tool call with an idempotency key. If the process crashes or is redeployed mid-task, it replays completed steps from the journal instead of re-executing side effects, and raises a non-determinism error if the replayed code diverges from history.

## Architecture

```
[Task] -> [Agent Loop] -> [Step Journal (SQLite/Postgres)]
          |
[Replay Hit?] -> [Cached Output]
          |
[Execute LLM / Tool] -> [Append Record]
```

## Tech Stack

Claude API, Python, SQLite / Postgres, Tool Use, Durable Execution

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/53-durable-agent-executor-checkpointed-replay

# Install dependencies
pip install -r requirements.txt

# Run the project
python durable_agent.py
```

## Author

**Manikanta Pudoka** — AI Engineer
