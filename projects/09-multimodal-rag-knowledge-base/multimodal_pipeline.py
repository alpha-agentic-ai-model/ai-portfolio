from llama_index.core import KnowledgeGraphIndex, VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from unstructured.partition.auto import partition
import qdrant_client


class MultiModalDoc:
    def __init__(self, path: str, doc_type: str):
        self.path = path
        self.doc_type = doc_type
        self.elements = []
        self.embeddings = None


class DocumentProcessor:
    def __init__(self):
        self.supported_types = ['pdf', 'image', 'code', 'table']

    def process(self, doc: MultiModalDoc) -> list:
        elements = partition(filename=doc.path)
        processed = []
        for el in elements:
            processed.append({
                'text': str(el),
                'type': el.category,
                'metadata': el.metadata.to_dict(),
            })
        return processed


class MultiModalRAG:
    def __init__(self, neo4j_url='bolt://localhost:7687', qdrant_url='localhost:6333'):
        self.graph_store = Neo4jGraphStore(
            url=neo4j_url, username='neo4j', password='password'
        )
        self.vector_client = qdrant_client.QdrantClient(url=qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=self.vector_client, collection_name='multimodal_docs'
        )
        self.processor = DocumentProcessor()
        self.kg_index = KnowledgeGraphIndex(
            graph_store=self.graph_store,
        )
        self.vector_index = VectorStoreIndex.from_vector_store(self.vector_store)

    async def ingest(self, docs: list[MultiModalDoc]):
        for doc in docs:
            elements = self.processor.process(doc)
            nodes = self.extract_entities(elements)
            relations = self.extract_relations(nodes)
            self.kg_index.insert(nodes, relations)

            embeddings = await self.embed_multimodal(elements)
            self.vector_index.insert(embeddings)

    async def query(self, q: str, top_k: int = 10):
        # Hybrid: graph + vector retrieval
        graph_results = self.kg_index.as_retriever().retrieve(q)
        vector_results = self.vector_index.as_retriever(
            similarity_top_k=top_k
        ).retrieve(q)

        merged = self.merge_results(graph_results, vector_results)
        answer = await self.generate_with_citations(q, merged)
        return answer

    def extract_entities(self, elements):
        entities = []
        for el in elements:
            if el['type'] in ('Title', 'NarrativeText'):
                extracted = self.ner_model.predict(el['text'])
                entities.extend(extracted)
        return entities

    def extract_relations(self, nodes):
        relations = []
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                rel = self.relation_model.predict(node_a, node_b)
                if rel.confidence > 0.7:
                    relations.append(rel)
        return relations

    async def generate_with_citations(self, query, context):
        prompt = f'Answer based on context. Cite sources.\nQuery: {query}\nContext: {context}'
        response = await self.llm.agenerate(prompt)
        return {
            'answer': response,
            'citations': self.extract_citations(response, context),
        }
