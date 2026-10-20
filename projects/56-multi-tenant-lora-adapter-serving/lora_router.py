"""Multi-tenant LoRA serving: one shared base model on vLLM, per-tenant LoRA
adapters loaded on demand into a bounded GPU slot pool with LRU eviction,
and a FastAPI front door that maps API keys to adapters."""
from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import boto3
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MAX_GPU_LORAS = int(os.getenv("MAX_GPU_LORAS", "16"))
ADAPTER_BUCKET = os.getenv("ADAPTER_BUCKET", "tenant-adapters")
CACHE_DIR = "/var/cache/adapters"


@dataclass
class AdapterSlot:
    tenant: str
    int_id: int
    path: str
    last_used: float
    in_flight: int = 0


class AdapterPool:
    """Keeps at most MAX_GPU_LORAS adapters resident; evicts least-recently-used idle ones."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.slots: OrderedDict[str, AdapterSlot] = OrderedDict()
        self.s3 = boto3.client("s3")
        self.lock = asyncio.Lock()
        self._next_id = 1

    def _download(self, tenant: str) -> str:
        local = os.path.join(CACHE_DIR, tenant)
        if not os.path.exists(os.path.join(local, "adapter_config.json")):
            os.makedirs(local, exist_ok=True)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                self.s3.download_file(ADAPTER_BUCKET, f"{tenant}/{name}", os.path.join(local, name))
        return local

    async def acquire(self, tenant: str) -> AdapterSlot:
        async with self.lock:
            if tenant in self.slots:
                slot = self.slots.pop(tenant)
            else:
                if len(self.slots) >= self.capacity:
                    victim = next((t for t, s in self.slots.items() if s.in_flight == 0), None)
                    if victim is None:
                        raise HTTPException(503, "all adapter slots busy; retry")
                    del self.slots[victim]       # vLLM drops it from its LoRA cache lazily
                path = await asyncio.to_thread(self._download, tenant)
                slot = AdapterSlot(tenant, self._next_id, path, time.time())
                self._next_id += 1
            slot.in_flight += 1
            slot.last_used = time.time()
            self.slots[tenant] = slot          # move to MRU end
            return slot

    async def release(self, slot: AdapterSlot) -> None:
        async with self.lock:
            slot.in_flight -= 1


engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=BASE_MODEL, enable_lora=True, max_loras=MAX_GPU_LORAS,
    max_lora_rank=64, max_cpu_loras=MAX_GPU_LORAS * 4, gpu_memory_utilization=0.9,
))
pool = AdapterPool(MAX_GPU_LORAS)
TENANTS = {k: v for k, v in (p.split(":") for p in os.getenv("TENANT_KEYS", "").split(",") if p)}
app = FastAPI(title="Multi-Tenant LoRA Gateway")


class Completion(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


@app.post("/v1/completions")
async def complete(body: Completion, x_api_key: str = Header(...)) -> dict:
    tenant = TENANTS.get(x_api_key)
    if tenant is None:
        raise HTTPException(401, "unknown API key")
    slot = await pool.acquire(tenant)
    try:
        lora = LoRARequest(lora_name=tenant, lora_int_id=slot.int_id, lora_path=slot.path)
        params = SamplingParams(max_tokens=body.max_tokens, temperature=body.temperature)
        request_id = f"{tenant}-{time.time_ns()}"
        final = None
        async for out in engine.generate(body.prompt, params, request_id, lora_request=lora):
            final = out
        text = final.outputs[0].text if final else ""
        return {"tenant": tenant, "adapter_id": slot.int_id, "text": text,
                "usage": {"completion_tokens": len(final.outputs[0].token_ids) if final else 0}}
    finally:
        await pool.release(slot)


@app.get("/v1/adapters")
async def adapters() -> list[dict]:
    return [{"tenant": s.tenant, "id": s.int_id, "in_flight": s.in_flight,
             "idle_s": round(time.time() - s.last_used, 1)} for s in pool.slots.values()]
