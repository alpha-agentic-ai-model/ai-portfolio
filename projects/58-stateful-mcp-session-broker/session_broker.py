"""Stateful MCP session broker: sits between agents and downstream MCP servers,
keeps a per-conversation context ledger (retrieved passages, tool calls,
results) and exposes it back to the agent as MCP resources, so state survives
reconnects and is shared across tools."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field

import redis.asyncio as redis
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import Context, FastMCP

LEDGER_TTL_S = 6 * 3600
MAX_ENTRY_CHARS = 4000

DOWNSTREAM = {
    "search": StdioServerParameters(command="uvx", args=["mcp-server-docs-search"]),
    "github": StdioServerParameters(command="npx", args=["-y", "@modelcontextprotocol/server-github"]),
}


@dataclass
class LedgerEntry:
    id: str
    server: str
    tool: str
    args: dict
    result: str
    ts: float = field(default_factory=time.time)
    pinned: bool = False


class Ledger:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self.r = redis.from_url(url, decode_responses=True)

    def _key(self, conv: str) -> str:
        return f"mcp:ledger:{conv}"

    async def append(self, conv: str, entry: LedgerEntry) -> None:
        await self.r.rpush(self._key(conv), json.dumps(asdict(entry)))
        await self.r.expire(self._key(conv), LEDGER_TTL_S)

    async def entries(self, conv: str) -> list[LedgerEntry]:
        return [LedgerEntry(**json.loads(x)) for x in await self.r.lrange(self._key(conv), 0, -1)]

    async def stitch(self, conv: str, budget_chars: int = 12000) -> str:
        """Pinned entries first, then most recent, until the budget is spent."""
        items = await self.entries(conv)
        ordered = [e for e in items if e.pinned] + [e for e in reversed(items) if not e.pinned]
        out, used = [], 0
        for e in ordered:
            block = f"## {e.server}.{e.tool}({json.dumps(e.args)[:200]})\n{e.result}"
            if used + len(block) > budget_chars:
                break
            out.append(block)
            used += len(block)
        return "\n\n".join(out)


class Broker:
    def __init__(self) -> None:
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}

    async def connect(self) -> None:
        for name, params in DOWNSTREAM.items():
            read, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session

    async def call(self, server: str, tool: str, args: dict) -> str:
        result = await self.sessions[server].call_tool(tool, args)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        return ("ERROR: " if result.isError else "") + text


ledger, broker = Ledger(), Broker()
mcp = FastMCP("session-broker")


@mcp.tool()
async def call_tool(conversation_id: str, server: str, tool: str, args: dict, ctx: Context) -> str:
    """Proxy a tool call to a downstream MCP server and record it in the conversation ledger."""
    if not broker.sessions:
        await broker.connect()
    if server not in broker.sessions:
        return f"unknown server '{server}'; available: {sorted(broker.sessions)}"
    await ctx.info(f"{server}.{tool} for {conversation_id}")
    result = await broker.call(server, tool, args)
    entry = LedgerEntry(uuid.uuid4().hex[:8], server, tool, args, result[:MAX_ENTRY_CHARS])
    await ledger.append(conversation_id, entry)
    return f"[ledger:{entry.id}]\n{result}"


@mcp.tool()
async def pin(conversation_id: str, entry_id: str) -> str:
    """Pin a ledger entry so it is always included in the stitched context."""
    key = ledger._key(conversation_id)
    for i, raw in enumerate(await ledger.r.lrange(key, 0, -1)):
        data = json.loads(raw)
        if data["id"] == entry_id:
            data["pinned"] = True
            await ledger.r.lset(key, i, json.dumps(data))
            return f"pinned {entry_id}"
    return f"no entry {entry_id}"


@mcp.resource("context://{conversation_id}")
async def stitched_context(conversation_id: str) -> str:
    """Budgeted, ordered context for a conversation — read this after a reconnect."""
    return await ledger.stitch(conversation_id)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
