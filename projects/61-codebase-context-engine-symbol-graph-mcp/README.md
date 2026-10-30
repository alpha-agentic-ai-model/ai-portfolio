# Codebase Context Engine with Symbol-Graph Ranking

> **Category:** LLM Engineering
> **Project #61** in the AI Engineer Portfolio

## Overview

Parses a repository with tree-sitter into definitions and cross-file call edges, then ranks files with personalized PageRank seeded from what the coding agent is editing. The top signatures are packed into a token budget and served over MCP, giving agents a compact repo map instead of whole files.

## Architecture

```
[Repository] -> [tree-sitter Parser] -> [Symbols + Call Edges]
          |
[Symbol Graph (networkx)] -> [Personalized PageRank] -> [Token-Budget Packer]
          |
[MCP Tool: repo_map] -> [Coding Agent]
```

## Tech Stack

tree-sitter, networkx, tiktoken, MCP Python SDK, FastMCP

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/61-codebase-context-engine-symbol-graph-mcp

# Install dependencies
pip install -r requirements.txt

# Run the project
python repo_map.py
```

## Author

**Manikanta Pudoka** — AI Engineer
