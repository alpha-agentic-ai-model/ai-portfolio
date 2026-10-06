# Sandboxed Agent Runtime with Capability-Scoped Permissions

> **Category:** Agentic AI
> **Project #41** in the AI Engineer Portfolio

## Overview

A supervisor that runs untrusted agent-generated code inside a gVisor-isolated container where every side effect must first clear a time-boxed, resource-scoped capability grant. Policy decisions are delegated to Open Policy Agent with deny-by-default fallback, so a prompt injection can only reach the narrow set of files, hosts and tools issued for that run.

## Architecture

```
[Agent Plan] -> [Capability Broker] -> [OPA Policy Plane]
          |
[Grant Ledger (TTL + budget)] -> [Syscall Interceptor]
          |
[gVisor Sandbox] -> [Egress Filter] -> [Append-Only Audit Log]
```

## Tech Stack

MCP SDK, gVisor, Open Policy Agent, Docker, asyncio, Pydantic

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/41-sandboxed-agent-runtime-capability-scoped

# Install dependencies
pip install -r requirements.txt

# Run the project
python sandbox_supervisor.py
```

## Author

**Manikanta Pudoka** — AI Engineer
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
