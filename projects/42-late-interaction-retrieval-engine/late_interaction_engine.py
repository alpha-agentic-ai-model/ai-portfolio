import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class RetrievalConfig:
    model_name: str = "colbert-ir/colbertv2.0"
    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 300
    nbits: int = 2              # residual compression bits per dimension
    ncells: int = 32            # IVF cells probed during candidate generation
    candidate_depth: int = 1024
    device: str = "cuda"


class LateInteractionEncoder(nn.Module):
    """ColBERT-style encoder producing L2-normalised per-token embeddings."""

    def __init__(self, cfg: RetrievalConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        self.project = nn.Linear(self.backbone.config.hidden_size, cfg.dim, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                keep_punctuation: bool = True) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.project(hidden)
        emb = F.normalize(emb, p=2, dim=-1)
        mask = attention_mask.unsqueeze(-1).to(emb.dtype)
        if not keep_punctuation:
            mask = mask * self._content_mask(input_ids).unsqueeze(-1)
        return emb * mask

    def _content_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        skip = torch.tensor(self.cfg_skiplist, device=input_ids.device)
        return (~torch.isin(input_ids, skip)).to(input_ids.dtype)

    cfg_skiplist: list[int] = []


class ResidualCodec:
    """Centroid + residual compression: 1 byte centroid id, nbits/dim residual."""

    def __init__(self, centroids: np.ndarray, nbits: int = 2):
        self.centroids = centroids.astype(np.float32)
        self.nbits = nbits
        self.bucket_cutoffs = np.quantile(
            np.linspace(-1, 1, 2 ** nbits + 1)[1:-1], q=np.linspace(0, 1, 2 ** nbits - 1))

    def encode(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = embeddings @ self.centroids.T
        codes = sims.argmax(axis=1).astype(np.int32)
        residual = embeddings - self.centroids[codes]
        quantised = np.digitize(residual, self.bucket_cutoffs).astype(np.uint8)
        return codes, quantised

    def decode(self, codes: np.ndarray, quantised: np.ndarray) -> np.ndarray:
        levels = np.concatenate([[-1.0], self.bucket_cutoffs, [1.0]])
        mids = (levels[:-1] + levels[1:]) / 2.0
        return self.centroids[codes] + mids[quantised]


class PlaidIndex:
    """Two-stage PLAID index: centroid pruning, then exact MaxSim rescoring."""

    def __init__(self, codec: ResidualCodec, cfg: RetrievalConfig):
        self.codec = codec
        self.cfg = cfg
        self.doc_codes: dict[int, np.ndarray] = {}
        self.doc_residuals: dict[int, np.ndarray] = {}
        self.inverted: dict[int, set[int]] = {}

    def add(self, doc_id: int, embeddings: np.ndarray) -> None:
        codes, residual = self.codec.encode(embeddings)
        self.doc_codes[doc_id] = codes
        self.doc_residuals[doc_id] = residual
        for centroid_id in np.unique(codes):
            self.inverted.setdefault(int(centroid_id), set()).add(doc_id)

    def generate_candidates(self, query_emb: np.ndarray) -> list[int]:
        sims = query_emb @ self.codec.centroids.T
        top_cells = np.argsort(-sims, axis=1)[:, : self.cfg.ncells].ravel()
        scores: dict[int, float] = {}
        for cell in top_cells:
            for doc_id in self.inverted.get(int(cell), ()):  # centroid-pruned fan-in
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[: self.cfg.candidate_depth]

    def maxsim(self, query_emb: np.ndarray, doc_id: int) -> float:
        doc = self.codec.decode(self.doc_codes[doc_id], self.doc_residuals[doc_id])
        return float(np.max(query_emb @ doc.T, axis=1).sum())


class LateInteractionRetriever:
    """End-to-end late-interaction retrieval with optional cross-encoder rerank."""

    def __init__(self, cfg: RetrievalConfig, index: PlaidIndex, reranker: Optional[object] = None):
        self.cfg = cfg
        self.index = index
        self.reranker = reranker
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.encoder = LateInteractionEncoder(cfg).to(cfg.device).eval()

    @torch.inference_mode()
    def embed_query(self, query: str) -> np.ndarray:
        batch = self.tokenizer(f"[Q] {query}", max_length=self.cfg.query_maxlen,
                               padding="max_length", truncation=True, return_tensors="pt")
        batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
        emb = self.encoder(batch["input_ids"], batch["attention_mask"])
        return emb[0].float().cpu().numpy()

    def search(self, query: str, k: int = 10) -> list[dict]:
        query_emb = self.embed_query(query)
        candidates = self.index.generate_candidates(query_emb)
        scored = [(doc_id, self.index.maxsim(query_emb, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        head = scored[: max(k * 4, 40)]
        if self.reranker is not None:
            head = self.reranker.rerank(query, [doc_id for doc_id, _ in head])
        return [{"doc_id": doc_id, "score": round(score, 4),
                 "normalised": round(score / self.cfg.query_maxlen, 4)}
                for doc_id, score in head[:k]]


def in_batch_contrastive_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """MaxSim-based contrastive objective used for distillation fine-tuning."""
    scores = torch.einsum("bqd,cnd->bcqn", q, d).max(dim=-1).values.sum(dim=-1)
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores / temperature, labels)
