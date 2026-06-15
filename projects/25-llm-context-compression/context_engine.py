import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float

class ContextCompressionEngine:
    """Adaptive context compression with quality feedback."""
    def __init__(self, target_ratio: float = 0.5):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.target_ratio = target_ratio
        self.quality_history: list[float] = []

    def score_importance(self, sentences: list[str], query: str) -> np.ndarray:
        query_emb = self.encoder.encode([query])
        sent_embs = self.encoder.encode(sentences)
        similarities = np.dot(sent_embs, query_emb.T).flatten()
        position_weights = np.linspace(1.0, 0.7, len(sentences))
        return similarities * position_weights

    def deduplicate(self, sentences: list[str], threshold: float = 0.92):
        embeddings = self.encoder.encode(sentences)
        keep = [0]
        for i in range(1, len(sentences)):
            sims = np.dot(embeddings[i], embeddings[keep].T)
            if sims.max() < threshold:
                keep.append(i)
        return [sentences[i] for i in keep]

    def compress(self, context: str, query: str) -> CompressionResult:
        sentences = self.split_sentences(context)
        deduped = self.deduplicate(sentences)
        scores = self.score_importance(deduped, query)
        budget = int(len(deduped) * self.adaptive_ratio())
        top_idx = np.argsort(scores)[-budget:]
        top_idx.sort()
        compressed = " ".join(deduped[i] for i in top_idx)
        return CompressionResult(
            compressed_text=compressed,
            original_tokens=self.count_tokens(context),
            compressed_tokens=self.count_tokens(compressed),
            compression_ratio=len(compressed) / len(context),
            quality_score=self.estimate_quality(scores, top_idx)
        )