"""GraphRAG: Knowledge Graph Reasoning Engine

Graph-based RAG system that builds a knowledge graph from unstructured documents
and performs multi-hop reasoning to answer complex queries.
"""

from neo4j import GraphDatabase
from dataclasses import dataclass, field
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    entity_type: str
    properties: dict = field(default_factory=dict)
    source_doc: Optional[str] = None


@dataclass
class Relation:
    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    evidence: str = ""


@dataclass
class ReasoningPath:
    entities: list[Entity]
    relations: list[Relation]
    score: float = 0.0

    @property
    def hop_count(self) -> int:
        return len(self.relations)

    def to_text(self) -> str:
        parts = []
        for i, rel in enumerate(self.relations):
            parts.append(
                f"{rel.source.name} --[{rel.relation_type}]--> {rel.target.name}"
            )
        return " | ".join(parts)


class EntityExtractor:
    """Extracts entities and relations from text using LLM."""

    EXTRACTION_PROMPT = """Extract all entities and relationships from the text.
Return JSON with format:
{"entities": [{"name": "...", "type": "...", "properties": {...}}],
 "relations": [{"source": "...", "target": "...", "type": "...", "confidence": 0.9}]}

Text: {text}"""

    def __init__(self, llm):
        self.llm = llm

    async def extract(self, text: str) -> list[Entity]:
        response = await self.llm.generate(
            self.EXTRACTION_PROMPT.format(text=text[:4000])
        )
        data = json.loads(response)
        return [
            Entity(name=e["name"], entity_type=e["type"],
                   properties=e.get("properties", {}))
            for e in data.get("entities", [])
        ]

    async def link_relations(self, text: str,
                              entities: list[Entity]) -> list[Relation]:
        entity_names = [e.name for e in entities]
        entity_map = {e.name: e for e in entities}
        response = await self.llm.generate(
            self.EXTRACTION_PROMPT.format(text=text[:4000])
        )
        data = json.loads(response)
        relations = []
        for r in data.get("relations", []):
            src = entity_map.get(r["source"])
            tgt = entity_map.get(r["target"])
            if src and tgt:
                relations.append(Relation(
                    source=src, target=tgt,
                    relation_type=r["type"],
                    confidence=r.get("confidence", 0.8),
                ))
        return relations


class GraphRAGEngine:
    """Knowledge graph reasoning engine with multi-hop traversal."""

    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, llm):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.llm = llm
        self.extractor = EntityExtractor(llm)

    async def ingest_document(self, doc_id: str, text: str):
        entities = await self.extractor.extract(text)
        relations = await self.extractor.link_relations(text, entities)

        logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations "
                     f"from doc {doc_id}")

        with self.driver.session() as session:
            for entity in entities:
                session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "SET e.type = $type, e.doc = $doc",
                    name=entity.name, type=entity.entity_type, doc=doc_id,
                )
            for rel in relations:
                session.run(
                    "MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt}) "
                    "MERGE (a)-[r:RELATES {type: $rel_type}]->(b) "
                    "SET r.confidence = $conf, r.doc = $doc",
                    src=rel.source.name, tgt=rel.target.name,
                    rel_type=rel.relation_type,
                    conf=rel.confidence, doc=doc_id,
                )

        return {"entities": len(entities), "relations": len(relations)}

    def traverse(self, seed_entities: list[str],
                 max_hops: int = 3) -> list[ReasoningPath]:
        paths = []
        with self.driver.session() as session:
            for seed in seed_entities:
                result = session.run(
                    f"MATCH path = (start:Entity {{name: $seed}})"
                    f"-[*1..{max_hops}]-(end:Entity) "
                    "RETURN path, length(path) as hops "
                    "ORDER BY hops LIMIT 20",
                    seed=seed,
                )
                for record in result:
                    neo_path = record["path"]
                    entities_in_path = [
                        Entity(name=node["name"],
                               entity_type=node.get("type", "unknown"))
                        for node in neo_path.nodes
                    ]
                    relations_in_path = [
                        Relation(
                            source=Entity(name=rel.start_node["name"],
                                          entity_type=""),
                            target=Entity(name=rel.end_node["name"],
                                          entity_type=""),
                            relation_type=rel.get("type", "related"),
                            confidence=rel.get("confidence", 0.5),
                        )
                        for rel in neo_path.relationships
                    ]
                    paths.append(ReasoningPath(
                        entities=entities_in_path,
                        relations=relations_in_path,
                        score=sum(r.confidence for r in relations_in_path),
                    ))
        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:10]

    def assemble_context(self, paths: list[ReasoningPath]) -> str:
        context_parts = []
        for i, path in enumerate(paths):
            context_parts.append(f"Path {i + 1} (score={path.score:.2f}):")
            context_parts.append(f"  {path.to_text()}")
        return "\n".join(context_parts)

    async def multi_hop_query(self, question: str, max_hops: int = 3) -> str:
        seed_entities = await self._extract_query_entities(question)
        logger.info(f"Query entities: {seed_entities}")

        paths = self.traverse(seed_entities, max_hops=max_hops)
        context = self.assemble_context(paths)

        prompt = (
            f"Using the following knowledge graph paths as context, "
            f"answer the question. Cite specific entities and relationships.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return await self.llm.generate(prompt)

    async def _extract_query_entities(self, question: str) -> list[str]:
        response = await self.llm.generate(
            f"Extract key entity names from this question as a JSON list: {question}"
        )
        return json.loads(response)

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        engine = GraphRAGEngine(
            neo4j_uri="bolt://localhost:7687",
            neo4j_auth=("neo4j", "password"),
            llm=None,  # Replace with your LLM client
        )
        # Example usage
        await engine.ingest_document("doc-1", "Sample document text...")
        answer = await engine.multi_hop_query(
            "How does entity A relate to entity C through entity B?"
        )
        print(answer)
        engine.close()

    asyncio.run(main())
