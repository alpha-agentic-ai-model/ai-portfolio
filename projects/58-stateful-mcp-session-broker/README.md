# Stateful MCP Session Broker with Context Stitching

> **Category:** Agentic AI
> **Project #58** in the AI Engineer Portfolio

## Overview

An MCP server that proxies tool calls to downstream MCP servers and records each call and result in a per-conversation ledger in Redis. Agents can pin important entries and read a budgeted, stitched context back as an MCP resource, so state survives reconnects and is shared across tools.

## Architecture

```
[Agent] -> [Session Broker (FastMCP)] -> [Downstream MCP Servers]
          |
[Redis Ledger] -> [Pin / Recency Ordering] -> [context:// Resource]
```

## Tech Stack

MCP Python SDK, FastMCP, Redis, asyncio, Streamable HTTP

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/58-stateful-mcp-session-broker

# Install dependencies
pip install -r requirements.txt

# Run the project
python session_broker.py
```

## Author

**Manikanta Pudoka** — AI Engineer
