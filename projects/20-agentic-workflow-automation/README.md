# Agentic Workflow Engine with Human-in-the-Loop

## Overview
An enterprise workflow automation engine where AI agents plan, execute, and monitor multi-step business processes with built-in human approval gates. Features DAG-based workflow definition, parallel branch execution, automatic error recovery with checkpoint rollback, and a real-time audit trail.

## Architecture
```
[Workflow DSL] → [DAG Compiler] → [Execution Engine]
       ↓
[Agent Step] → [Human Gate] → [Parallel Branch]
       ↓
[Checkpoint Store] → [Recovery Manager] → [Audit Log]
```

## Tech Stack
- **LangGraph** — Agent execution framework
- **Temporal.io** — Durable workflow orchestration
- **Claude API** — Agent reasoning model
- **PostgreSQL** — Workflow state and audit trail
- **Redis** — Checkpoint and queue management
- **FastAPI** — Workflow API and approval endpoints

## Key Features
- DAG-based workflow definition with parallel branches
- Human-in-the-loop approval gates for sensitive operations
- Automatic retry with checkpoint rollback on failure
- Real-time compliance audit trail
- Topological batch execution for maximum parallelism
