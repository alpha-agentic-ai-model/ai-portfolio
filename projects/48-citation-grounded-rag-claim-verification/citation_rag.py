"""Citation-grounded RAG with claim-level verification.

The generator must attach a source id to every sentence. Each sentence is then
split into atomic claims and checked against its cited passage with an NLI
cross-encoder; unsupported claims are rewritten or dropped before the answer
is returned, and a groundedness score is attached for monitoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import anthropic
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

CITE_RE = re.compile(r"\[(S\d+)\]")


@dataclass
class Passage:
    sid: str
    text: str
    url: str


@dataclass
class Claim:
    text: str
    source: str | None
    entailment: float = 0.0


@dataclass
class GroundedAnswer:
    text: str
    citations: dict[str, str]
    claims: list[Claim] = field(default_factory=list)

    @property
    def groundedness(self) -> float:
        if not self.claims:
            return 0.0
        return sum(c.entailment >= 0.5 for c in self.claims) / len(self.claims)


class CitationRAG:
    def __init__(self, collection: str, threshold: float = 0.5):
        self.llm = anthropic.Anthropic()
        self.qdrant = QdrantClient("localhost", port=6333)
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        self.collection = collection
        self.threshold = threshold

    def retrieve(self, query: str, k: int = 6) -> list[Passage]:
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self.qdrant.search(self.collection, query_vector=vec, limit=k)
        return [Passage(f"S{i+1}", h.payload["text"], h.payload["url"]) for i, h in enumerate(hits)]

    def _ask(self, system: str, user: str, max_tokens: int = 800) -> str:
        msg = self.llm.messages.create(
            model="claude-sonnet-4-5", max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text

    def generate(self, query: str, passages: list[Passage]) -> str:
        context = "\n\n".join(f"[{p.sid}] {p.text}" for p in passages)
        system = ("Answer only from the sources. End every sentence with one citation "
                  "like [S2]. If the sources do not contain the answer, say so.")
        return self._ask(system, f"Sources:\n{context}\n\nQuestion: {query}")

    def decompose(self, sentence: str) -> list[str]:
        out = self._ask("Split the sentence into minimal standalone factual claims, one per line.",
                        sentence, max_tokens=300)
        return [line.strip("-• ").strip() for line in out.splitlines() if line.strip()]

    def verify(self, draft: str, passages: list[Passage]) -> list[Claim]:
        by_id = {p.sid: p for p in passages}
        claims: list[Claim] = []
        for sentence in re.split(r"(?<=[.!?])\s+", draft.strip()):
            cites = CITE_RE.findall(sentence)
            source = cites[0] if cites else None
            for atom in self.decompose(CITE_RE.sub("", sentence)):
                claim = Claim(atom, source)
                if source in by_id:
                    logits = self.nli.predict([(by_id[source].text, atom)], apply_softmax=True)[0]
                    claim.entailment = float(logits[1])  # index 1 = entailment
                claims.append(claim)
        return claims

    def repair(self, draft: str, unsupported: list[Claim]) -> str:
        bad = "\n".join(f"- {c.text}" for c in unsupported)
        return self._ask("Remove or soften the listed unsupported claims. Keep all citations "
                         "for the remaining sentences unchanged.",
                         f"Draft:\n{draft}\n\nUnsupported:\n{bad}")

    def answer(self, query: str) -> GroundedAnswer:
        passages = self.retrieve(query)
        draft = self.generate(query, passages)
        claims = self.verify(draft, passages)
        unsupported = [c for c in claims if c.entailment < self.threshold]
        if unsupported:
            draft = self.repair(draft, unsupported)
            claims = self.verify(draft, passages)
        used = set(CITE_RE.findall(draft))
        citations = {p.sid: p.url for p in passages if p.sid in used}
        return GroundedAnswer(draft, citations, claims)


if __name__ == "__main__":
    rag = CitationRAG("docs")
    result = rag.answer("What changed in the v2 billing API?")
    print(result.text)
    print(f"groundedness={result.groundedness:.2f}", result.citations)
