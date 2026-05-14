# AWS Strands Agent Orchestrator with Dynamic Task Routing

## Overview
A multi-agent orchestration framework built on the AWS Strands Agents SDK that dynamically routes tasks to specialized sub-agents based on intent classification. Features native Bedrock model integration, tool-use chains with retry logic, shared memory across agent teams, and automatic load balancing with agent health monitoring.

## Architecture
```
[User Request] → [Intent Classifier] → [Agent Router]
       ↓
[Research Agent] | [Code Agent] | [Data Agent]
       ↓
[Shared Memory] → [Result Aggregator] → [Response]
```

## Tech Stack
- **Strands Agents SDK** — AWS model-driven agent framework
- **AWS Bedrock** — Managed model inference
- **Claude API** — Primary reasoning model
- **DynamoDB** — Shared agent memory store
- **Python / asyncio** — Async execution runtime

## Key Features
- Intent-based dynamic routing to specialized agents
- Tool-use chains with automatic retry and fallback
- Agent health monitoring with automatic load balancing
- Shared memory layer across agent teams via DynamoDB
- Native AWS Bedrock integration for model inference
