import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Tier(IntEnum):
    HBM = 0      # on-GPU paged KV blocks
    DRAM = 1     # pinned host memory
    NVME = 2     # local block store
    REMOTE = 3   # shared object store across the fleet


@dataclass
class BlockMeta:
    block_hash: str
    tier: Tier
    n_tokens: int
    bytes_size: int
    last_access: float
    hits: int = 0
    layer_range: tuple[int, int] = (0, 0)

    def score(self, now: float, half_life_s: float = 90.0) -> float:
        """Frequency-recency score (GDSF-like) used for eviction ordering."""
        decay = 0.5 ** ((now - self.last_access) / half_life_s)
        return (self.hits + 1) * decay / max(self.bytes_size / 1e6, 0.1)


@dataclass
class TierBudget:
    capacity_bytes: int
    used_bytes: int = 0
    promote_latency_ms: float = 0.0

    @property
    def utilisation(self) -> float:
        return self.used_bytes / max(self.capacity_bytes, 1)


class PrefixTrie:
    """Radix trie over block hashes; maps shared prompt prefixes to replicas."""

    def __init__(self):
        self.children: dict[str, "PrefixTrie"] = {}
        self.replicas: set[str] = set()
        self.depth = 0

    def insert(self, chain: list[str], replica: str) -> None:
        node = self
        for i, block_hash in enumerate(chain):
            node = node.children.setdefault(block_hash, PrefixTrie())
            node.depth = i + 1
            node.replicas.add(replica)

    def longest_match(self, chain: list[str]) -> tuple[int, set[str]]:
        node, matched, replicas = self, 0, set()
        for block_hash in chain:
            nxt = node.children.get(block_hash)
            if nxt is None:
                break
            node, matched, replicas = nxt, matched + 1, nxt.replicas
        return matched, replicas


class KVTierManager:
    """Tiered KV-cache service: hashes prefixes, places blocks, routes requests.

    Blocks are content-addressed by the chain of token ids that produced them,
    so any replica holding the same prefix can serve a prefill hit. Cold blocks
    cascade HBM -> DRAM -> NVMe -> object store instead of being discarded.
    """

    BLOCK_TOKENS = 16

    def __init__(self, budgets: dict[Tier, TierBudget], bytes_per_token: int = 2 * 32 * 128 * 2):
        self.budgets = budgets
        self.bytes_per_token = bytes_per_token
        self.blocks: dict[str, BlockMeta] = {}
        self.trie = PrefixTrie()
        self.stats = {"hits": 0, "misses": 0, "promotions": 0, "evictions": 0, "spills": 0}

    def hash_chain(self, token_ids: list[int]) -> list[str]:
        chain, running = [], hashlib.blake2b(digest_size=16)
        for start in range(0, len(token_ids) - self.BLOCK_TOKENS + 1, self.BLOCK_TOKENS):
            block = token_ids[start: start + self.BLOCK_TOKENS]
            running.update(np.asarray(block, dtype=np.int32).tobytes())
            chain.append(running.hexdigest())
        return chain

    def route(self, token_ids: list[int], candidates: list[str]) -> dict:
        """Prefix-aware routing: send the request where the KV already lives."""
        chain = self.hash_chain(token_ids)
        matched, replicas = self.trie.longest_match(chain)
        eligible = [r for r in candidates if r in replicas] or candidates
        hit_ratio = matched * self.BLOCK_TOKENS / max(len(token_ids), 1)
        self.stats["hits" if matched else "misses"] += 1
        return {"replica": eligible[0], "prefix_hit_tokens": matched * self.BLOCK_TOKENS,
                "prefix_hit_ratio": round(hit_ratio, 3), "chain_len": len(chain)}

    async def admit(self, chain: list[str], replica: str, layer_range: tuple[int, int]) -> None:
        now = time.time()
        size = self.BLOCK_TOKENS * self.bytes_per_token
        for block_hash in chain:
            meta = self.blocks.get(block_hash)
            if meta is not None:
                meta.hits += 1
                meta.last_access = now
                if meta.tier > Tier.HBM:
                    await self.promote(meta)
                continue
            await self._make_room(Tier.HBM, size)
            self.blocks[block_hash] = BlockMeta(block_hash, Tier.HBM, self.BLOCK_TOKENS,
                                                size, now, layer_range=layer_range)
            self.budgets[Tier.HBM].used_bytes += size
        self.trie.insert(chain, replica)

    async def promote(self, meta: BlockMeta) -> None:
        target = Tier(max(meta.tier - 1, Tier.HBM))
        await self._make_room(target, meta.bytes_size)
        self.budgets[meta.tier].used_bytes -= meta.bytes_size
        self.budgets[target].used_bytes += meta.bytes_size
        meta.tier = target
        self.stats["promotions"] += 1

    async def _make_room(self, tier: Tier, needed: int) -> None:
        budget = self.budgets[tier]
        if budget.used_bytes + needed <= budget.capacity_bytes:
            return
        now = time.time()
        victims = sorted((m for m in self.blocks.values() if m.tier is tier),
                         key=lambda m: m.score(now))
        for victim in victims:
            await self._spill(victim)
            if budget.used_bytes + needed <= budget.capacity_bytes:
                return

    async def _spill(self, meta: BlockMeta) -> None:
        nxt = Tier(min(meta.tier + 1, Tier.REMOTE))
        self.budgets[meta.tier].used_bytes -= meta.bytes_size
        if nxt is meta.tier:                       # already at the coldest tier: drop it
            self.blocks.pop(meta.block_hash, None)
            self.stats["evictions"] += 1
            return
        await self._make_room(nxt, meta.bytes_size)
        self.budgets[nxt].used_bytes += meta.bytes_size
        meta.tier = nxt
        self.stats["spills"] += 1

    def prometheus_metrics(self) -> str:
        lines = [f'kv_cache_hit_ratio {self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1):.4f}']
        for tier, budget in self.budgets.items():
            lines.append(f'kv_tier_utilisation{{tier="{tier.name.lower()}"}} {budget.utilisation:.4f}')
        for key, value in self.stats.items():
            lines.append(f"kv_cache_{key}_total {value}")
        return "\n".join(lines) + "\n"
