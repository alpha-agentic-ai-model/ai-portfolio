import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class MemoryEntry:
    content: str
    memory_type: MemoryType
    importance: float
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    associations: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None

class AgentMemorySystem:
    """Hierarchical memory with temporal decay and associative recall."""
    def __init__(self, db, embedding_model, decay_rate: float = 0.995):
        self.db = db
        self.embedder = embedding_model
        self.decay_rate = decay_rate
        self.working_memory: list[MemoryEntry] = []

    def compute_retrieval_score(self, memory: MemoryEntry, query_emb) -> float:
        age_hours = (time.time() - memory.timestamp) / 3600
        recency = math.pow(self.decay_rate, age_hours)
        relevance = self.cosine_sim(memory.embedding, query_emb)
        frequency = math.log(1 + memory.access_count)
        return (0.4 * relevance + 0.3 * recency +
                0.2 * memory.importance + 0.1 * frequency)

    async def store(self, content: str, mem_type: MemoryType):
        embedding = await self.embedder.encode(content)
        importance = await self.score_importance(content)
        entry = MemoryEntry(
            content=content, memory_type=mem_type,
            importance=importance, embedding=embedding
        )
        if mem_type == MemoryType.WORKING:
            self.working_memory.append(entry)
            if len(self.working_memory) > 7:
                await self.consolidate_to_episodic()
        else:
            await self.db.insert(entry)

    async def recall(self, query: str, top_k: int = 5):
        query_emb = await self.embedder.encode(query)
        candidates = await self.db.search_similar(query_emb, limit=50)
        scored = [(m, self.compute_retrieval_score(m, query_emb))
                  for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        for mem, _ in scored[:top_k]:
            mem.access_count += 1
        return [m for m, _ in scored[:top_k]]