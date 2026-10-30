# Human-Approval MCP Server with Elicitation and Audit Log

> **Category:** Agentic AI
> **Project #64** in the AI Engineer Portfolio

## Overview

An MCP server that wraps risky actions such as refunds, deletions and outbound email, and pauses mid-call to ask the human through MCP elicitation to approve, edit or reject them. Low-risk calls under a policy threshold run automatically, and every decision is written to a hashed, append-only audit log.

## Architecture

```
[Agent] -> [MCP Tool Call] -> [Risk Policy]
          |
[Auto-Approve (low risk)] -> [ctx.elicit() Approval Form] -> [Human Reviewer]
          |
[Append-only Audit Log] -> [Side Effect] -> [audit://recent]
```

## Tech Stack

MCP Python SDK, FastMCP, Elicitation, Pydantic, Streamable HTTP

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/64-mcp-human-approval-elicitation-server

# Install dependencies
pip install -r requirements.txt

# Run the project
python approval_server.py
```

## Author

**Manikanta Pudoka** — AI Engineer
