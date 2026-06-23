# Multi-Agent Code Review Pipeline

## Overview
A crew of specialized AI agents that perform comprehensive code reviews. SecurityAgent finds vulnerabilities, PerformanceAgent identifies bottlenecks, StyleAgent enforces conventions, and SynthesisAgent merges findings into actionable PR comments.

## Architecture
```
[PR Diff] → [Dispatcher] → [Security Agent]

[Style Agent] → [Perf Agent] → [Synthesis Agent] → [PR Comments]
```

## Tech Stack
CrewAI, Claude API, Tree-sitter, GitHub API, Python, AST

## Key Features
- Production-ready implementation with error handling
- Comprehensive type annotations and documentation
- Modular architecture for easy extension
- Built for scalability and performance
