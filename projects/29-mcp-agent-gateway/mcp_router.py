"""MCP Agent Gateway with Dynamic Tool Discovery."""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel


@dataclass
class ToolDefinition:
    name: str
    description: str
    endpoint: str
    input_schema: dict
    permissions: list[str] = field(default_factory=list)
    rate_limit: int = 100  # requests per minute


class ToolCallRequest(BaseModel):
    agent_id: str
    tool_name: str
    parameters: dict[str, Any]
    trace_id: Optional[str] = None


class AuditLogger:
    def __init__(self, sink: str = "postgres"):
        self.sink = sink
        self.buffer: list[dict] = []

    async def log(self, agent_id: str, tool_name: str,
                  params: dict, status: str, latency_ms: float):
        entry = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "tool": tool_name,
            "params_hash": hash(str(sorted(params.items()))),
            "status": status,
            "latency_ms": latency_ms,
        }
        self.buffer.append(entry)
        if len(self.buffer) >= 50:
            await self.flush()

    async def flush(self):
        if not self.buffer:
            return
        # Batch insert to PostgreSQL
        batch = self.buffer[:]
        self.buffer.clear()
        print(f"Flushed {len(batch)} audit entries to {self.sink}")


class ToolRegistry:
    def __init__(self, registry_url: str):
        self.url = registry_url
        self._cache: dict[str, ToolDefinition] = {}

    async def discover(self, tool_name: str) -> Optional[ToolDefinition]:
        if tool_name in self._cache:
            return self._cache[tool_name]
        # Discovery via MCP protocol handshake
        tool = await self._fetch_from_registry(tool_name)
        if tool:
            self._cache[tool_name] = tool
        return tool

    async def _fetch_from_registry(self, name: str) -> Optional[ToolDefinition]:
        # Simulated registry lookup
        return ToolDefinition(
            name=name,
            description=f"Auto-discovered tool: {name}",
            endpoint=f"{self.url}/tools/{name}",
            input_schema={},
        )

    async def list_tools(self) -> list[str]:
        return list(self._cache.keys())


class AuthLayer:
    def __init__(self):
        self.scopes: dict[str, set[str]] = {}

    async def verify_scope(self, agent_id: str, tool_name: str):
        allowed = self.scopes.get(agent_id, set())
        if tool_name not in allowed and "*" not in allowed:
            raise PermissionError(
                f"Agent {agent_id} lacks permission for {tool_name}"
            )

    def grant(self, agent_id: str, tools: list[str]):
        self.scopes.setdefault(agent_id, set()).update(tools)


class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.window = 60  # seconds

    async def check(self, agent_id: str, tool_name: str, limit: int):
        key = f"rl:{agent_id}:{tool_name}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, self.window)
        if count > limit:
            raise HTTPException(429, "Rate limit exceeded")


class MCPGateway:
    def __init__(self, registry_url: str, redis_url: str = "redis://localhost"):
        self.registry = ToolRegistry(registry_url)
        self.auth = AuthLayer()
        self.rate_limiter: Optional[RateLimiter] = None
        self.audit = AuditLogger(sink="postgres")
        self.redis_url = redis_url

    async def startup(self):
        client = redis.from_url(self.redis_url)
        self.rate_limiter = RateLimiter(client)

    async def route_tool_call(self, request: ToolCallRequest) -> dict:
        start = time.monotonic()
        await self.auth.verify_scope(request.agent_id, request.tool_name)

        tool = await self.registry.discover(request.tool_name)
        if not tool:
            raise HTTPException(404, f"Tool not found: {request.tool_name}")

        await self.rate_limiter.check(
            request.agent_id, request.tool_name, tool.rate_limit
        )

        # Execute the tool call
        result = {"output": f"Executed {tool.name}", "status": "success"}
        latency = (time.monotonic() - start) * 1000

        await self.audit.log(
            request.agent_id, request.tool_name,
            request.parameters, "success", latency
        )
        return result


# FastAPI application
app = FastAPI(title="MCP Agent Gateway")
gateway = MCPGateway(registry_url="http://localhost:8500")


@app.on_event("startup")
async def on_startup():
    await gateway.startup()
    gateway.auth.grant("agent-alpha", ["*"])


@app.post("/v1/tools/call")
async def call_tool(request: ToolCallRequest):
    return await gateway.route_tool_call(request)


@app.get("/v1/tools")
async def list_tools():
    return {"tools": await gateway.registry.list_tools()}
