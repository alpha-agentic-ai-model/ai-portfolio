import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from dateutil import parser as date_parser
import re
import numpy as np


@dataclass
class TemporalDocument:
    content: str
    created_at: datetime
    valid_until: Optional[datetime] = None
    source: str = ""
    relevance: float = 0.0
    topic_hash: str = ""
    embedding: Optional[np.ndarray] = None


@dataclass
class TemporalQuery:
    original: str
    normalized: str
    time_reference: datetime
    time_window: Optional[tuple[datetime, datetime]] = None
    requires_latest: bool = False


@dataclass
class TemporalResult:
    documents: list[TemporalDocument]
    conflicts_resolved: int
    expired_filtered: int
    time_reference: datetime
    answer: str = ""


class TemporalQueryParser:
    """Parse temporal intent from natural language queries."""

    RELATIVE_PATTERNS = {
        r"today": timedelta(days=0),
        r"yesterday": timedelta(days=-1),
        r"last week": timedelta(weeks=-1),
        r"last month": timedelta(days=-30),
        r"last year": timedelta(days=-365),
        r"this week": timedelta(days=0),
        r"this month": timedelta(days=0),
    }

    def parse(self, query: str, now: Optional[datetime] = None) -> TemporalQuery:
        now = now or datetime.utcnow()
        requires_latest = any(kw in query.lower() for kw in [
            "latest", "current", "newest", "most recent", "up to date",
            "now", "today", "right now"
        ])

        time_ref = now
        time_window = None

        for pattern, delta in self.RELATIVE_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                time_ref = now + delta
                break

        # Try explicit date parsing
        date_match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", query)
        if date_match:
            try:
                time_ref = date_parser.parse(date_match.group(1))
            except ValueError:
                pass

        # Detect range queries
        range_match = re.search(r"between (.+?) and (.+?)(?:\s|$)", query, re.IGNORECASE)
        if range_match:
            try:
                start = date_parser.parse(range_match.group(1))
                end = date_parser.parse(range_match.group(2))
                time_window = (start, end)
            except ValueError:
                pass

        return TemporalQuery(
            original=query,
            normalized=re.sub(r"(latest|current|recent|today|yesterday)", "", query, flags=re.IGNORECASE).strip(),
            time_reference=time_ref,
            time_window=time_window,
            requires_latest=requires_latest,
        )


class TemporalRAG:
    """Time-aware RAG with decay scoring and conflict resolution."""

    def __init__(self, vector_store, llm, half_life_days: int = 30):
        self.store = vector_store
        self.llm = llm
        self.half_life = half_life_days
        self.query_parser = TemporalQueryParser()

    def temporal_score(self, doc: TemporalDocument, query_time: datetime) -> float:
        """Score document relevance with exponential time decay."""
        age_days = max((query_time - doc.created_at).days, 0)
        decay = math.exp(-0.693 * age_days / self.half_life)

        # Penalize expired documents heavily
        if doc.valid_until and query_time > doc.valid_until:
            decay *= 0.1

        return doc.relevance * decay

    def is_within_window(self, doc: TemporalDocument, window: tuple[datetime, datetime]) -> bool:
        """Check if document falls within a time window."""
        start, end = window
        return start <= doc.created_at <= end

    async def retrieve(self, query: str, top_k: int = 10) -> TemporalResult:
        """Retrieve time-aware documents with conflict resolution."""
        parsed = self.query_parser.parse(query)
        candidates = await self.store.search(parsed.normalized, limit=top_k * 3)

        # Filter by time window if specified
        if parsed.time_window:
            candidates = [d for d in candidates if self.is_within_window(d, parsed.time_window)]

        # Filter expired documents
        now = datetime.utcnow()
        active = [d for d in candidates if not d.valid_until or d.valid_until >= now]
        expired_count = len(candidates) - len(active)

        # Score with temporal decay
        scored = []
        for doc in active:
            score = self.temporal_score(doc, parsed.time_reference)
            if parsed.requires_latest:
                # Boost very recent documents
                age_hours = (now - doc.created_at).total_seconds() / 3600
                recency_boost = 1.0 + max(0, 1.0 - age_hours / 168)  # boost within 7 days
                score *= recency_boost
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Resolve temporal conflicts
        resolved, conflicts = self.resolve_conflicts(scored[:top_k * 2])

        # Generate time-grounded answer
        context = "

".join(
            f"[{d.created_at.strftime('%Y-%m-%d')}] {d.content}"
            for d in resolved[:top_k]
        )
        answer = await self.llm.generate(
            f"Answer based on the following time-stamped sources. "
            f"Prefer more recent information when sources conflict.

"
            f"Query: {query}

Sources:
{context}"
        )

        return TemporalResult(
            documents=resolved[:top_k],
            conflicts_resolved=conflicts,
            expired_filtered=expired_count,
            time_reference=parsed.time_reference,
            answer=answer,
        )

    def resolve_conflicts(self, scored_docs: list[tuple[TemporalDocument, float]]) -> tuple[list[TemporalDocument], int]:
        """When documents contradict on the same topic, keep the most recent."""
        seen_topics: dict[str, tuple[TemporalDocument, float]] = {}
        conflicts = 0

        for doc, score in scored_docs:
            topic = doc.topic_hash or self.extract_topic_hash(doc)
            if topic not in seen_topics:
                seen_topics[topic] = (doc, score)
            elif doc.created_at > seen_topics[topic][0].created_at:
                seen_topics[topic] = (doc, score)
                conflicts += 1

        resolved = sorted(
            seen_topics.values(), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in resolved], conflicts

    def extract_topic_hash(self, doc: TemporalDocument) -> str:
        """Extract a rough topic identifier from document content."""
        words = doc.content.lower().split()[:20]
        key_words = sorted(set(w for w in words if len(w) > 4))[:5]
        return "|".join(key_words)
