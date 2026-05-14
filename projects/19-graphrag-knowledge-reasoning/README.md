# GraphRAG: Knowledge Graph Reasoning Engine

## Overview
A graph-based retrieval-augmented generation system that builds a knowledge graph from unstructured documents using entity extraction and relation linking, then performs multi-hop reasoning over the graph to answer complex queries spanning multiple documents.

## Architecture
```
[Documents] → [Entity Extractor] → [Relation Linker]
       ↓
[Knowledge Graph (Neo4j)] → [Multi-Hop Traversal]
       ↓
[Context Assembler] → [LLM Reasoner] → [Cited Answer]
```

## Tech Stack
- **Neo4j** — Native graph database for knowledge storage
- **LangChain** — LLM orchestration framework
- **Claude API** — Entity extraction and reasoning
- **spaCy** — NLP preprocessing and NER
- **FastAPI** — Query and ingestion API
- **NetworkX** — Graph analysis and path scoring

## Key Features
- Automatic entity extraction and relation linking from documents
- Multi-hop graph traversal for complex reasoning
- Path scoring and ranking for relevance
- Cited answers with entity and relationship evidence
- Incremental knowledge graph updates
