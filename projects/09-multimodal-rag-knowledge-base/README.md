# Multi-Modal RAG for Enterprise Knowledge Base

Multimodal RAG system ingesting PDFs, images, tables, and code repos into a unified knowledge graph. Uses ColPali for visual document retrieval with graph-based reasoning.

## Architecture
```
[PDFs + Images + Code] -> [Multi-Modal Embedder]
                                  |
[Knowledge Graph] -> [Graph + Vector Retrieval] -> [Answer + Visual Citations]
```

## Tech Stack
- ColPali, Neo4j, LlamaIndex, Qdrant, Unstructured.io
