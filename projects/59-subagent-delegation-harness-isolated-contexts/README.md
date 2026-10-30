# Sub-Agent Delegation Harness with Isolated Contexts

> **Category:** Agentic AI
> **Project #59** in the AI Engineer Portfolio

## Overview

An orchestrator plans a goal into subtasks and hands each one to a fresh sub-agent with its own message history, tool allowlist and turn budget. Only a short, JSON-validated report comes back, so the orchestrator's context holds conclusions instead of raw tool output and long jobs stay within the window.

## Architecture

```
[Goal] -> [Orchestrator Planner] -> [Subtask List]
          |
[Sub-Agent Pool (isolated contexts)] -> [Scoped Tool Calls] -> [JSON Reports]
          |
[Report Digest] -> [Final Synthesis]
```

## Tech Stack

Claude API, Tool Use, asyncio, Python dataclasses, JSON Schema

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/59-subagent-delegation-harness-isolated-contexts

# Install dependencies
pip install -r requirements.txt

# Run the project
python delegation_harness.py
```

## Author

**Manikanta Pudoka** — AI Engineer
