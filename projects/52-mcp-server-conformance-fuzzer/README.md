# MCP Server Conformance Fuzzer with Schema-Driven Tests

> **Category:** Agentic AI
> **Project #52** in the AI Engineer Portfolio

## Overview

A test harness that connects to any MCP server over stdio, reads each tool's JSON Schema and generates both valid and adversarial arguments with Hypothesis. It flags crashes, invalid inputs that get accepted, malformed responses and latency-budget breaches, producing a conformance report before a server reaches production agents.

## Architecture

```
[MCP Server (stdio)] -> [list_tools] -> [JSON Schemas]
          |
[Hypothesis Valid Args] -> [Mutated Invalid Args] -> [call_tool]
          |
[Conformance Checks] -> [Latency Budget] -> [Findings Report]
```

## Tech Stack

MCP Python SDK, Hypothesis, hypothesis-jsonschema, jsonschema, asyncio

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/52-mcp-server-conformance-fuzzer

# Install dependencies
pip install -r requirements.txt

# Run the project
python mcp_fuzzer.py
```

## Author

**Manikanta Pudoka** — AI Engineer
- Portfolio: https://alpha-agentic-ai-model.github.io/ai-portfolio/
- LinkedIn: https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/
- Email: manikanta.pudoka.ai@gmail.com
