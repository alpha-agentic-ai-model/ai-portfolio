# Persistent Agent Memory & Episodic Recall System

## Overview
A sophisticated memory management system for AI agents implementing hierarchical memory (working, episodic, semantic, procedural) with time-weighted decay and associative recall.

## Architecture
```
[Agent Action] → [Memory Encoder] → [Importance Scorer]
 ↓
[Working Memory] → [Episodic Store] → [Semantic Consolidator]
 ↓
[Associative Retrieval] → [Temporal Decay] → [Memory Replay]
```

## Key Features
- Four-tier hierarchical memory: working, episodic, semantic, procedural
- Time-weighted decay with importance-based preservation
- Associative recall combining relevance, recency, importance, and frequency
- Working memory consolidation (inspired by human sleep-based memory)
- Cross-session persistence for long-running agent deployments

## Tech Stack
- **Claude API** — Agent reasoning & importance scoring
- **PostgreSQL** — Persistent memory storage
- **pgvector** — Vector similarity search for recall
- **LangGraph** — Agent workflow integration
- **Redis** — Working memory cache
