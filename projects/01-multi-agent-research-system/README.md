# Multi-Agent Research & Report Generator

An orchestrated multi-agent system where specialized agents (Researcher, Analyst, Writer, Editor) collaborate to produce comprehensive research reports from a single topic prompt.

## Architecture
```
[User Query] -> [Orchestrator] -> [Researcher]
                                      |
                    [Analyst] -> [Writer] -> [Editor] -> [Final Report]
```

## Tech Stack
- LangGraph (state management & agent handoffs)
- GPT-4o (LLM backbone)
- Tavily API (web research)
- Python + Async IO

## Key Features
- Stateful agent orchestration with LangGraph StateGraph
- Quality gate loop: Editor can send drafts back to Writer
- Parallel research across multiple sub-queries
- Structured output with citations
