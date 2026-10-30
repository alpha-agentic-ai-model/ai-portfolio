"""Sub-agent delegation harness with isolated contexts.

An orchestrator agent plans work, then delegates each subtask to a fresh
sub-agent that has its own message history, its own tool allowlist and its
own token budget. Only a compact, schema-checked report flows back to the
orchestrator, so its context window holds results rather than raw tool noise.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-5"
client = AsyncAnthropic()

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    fn: ToolFn

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


@dataclass
class SubAgentSpec:
    role: str
    task: str
    tools: list[Tool]
    max_turns: int = 8
    max_output_tokens: int = 2048


@dataclass
class SubAgentReport:
    role: str
    status: str                     # "done" | "budget_exhausted" | "error"
    summary: str
    evidence: list[str] = field(default_factory=list)
    turns_used: int = 0
    input_tokens: int = 0


REPORT_INSTRUCTIONS = (
    "When finished, reply with ONLY a JSON object: "
    '{"summary": str (<= 120 words), "evidence": [short quotes or file paths]}'
)


class SubAgent:
    """Runs one delegated task in a context that the orchestrator never sees."""

    def __init__(self, spec: SubAgentSpec):
        self.spec = spec
        self.tools = {t.name: t for t in spec.tools}
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": spec.task}]

    async def run(self) -> SubAgentReport:
        used_in = 0
        for turn in range(1, self.spec.max_turns + 1):
            resp = await client.messages.create(
                model=MODEL,
                max_tokens=self.spec.max_output_tokens,
                system=f"You are the {self.spec.role} sub-agent. {REPORT_INSTRUCTIONS}",
                tools=[t.spec() for t in self.tools.values()],
                messages=self.messages,
            )
            used_in += resp.usage.input_tokens
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return self._parse_report(resp, turn, used_in)
            results = await asyncio.gather(*[self._exec(b) for b in resp.content if b.type == "tool_use"])
            self.messages.append({"role": "user", "content": list(results)})
        return SubAgentReport(self.spec.role, "budget_exhausted", "Turn budget exhausted.", [], turn, used_in)

    async def _exec(self, block: Any) -> dict[str, Any]:
        tool = self.tools.get(block.name)
        if tool is None:   # tool not in this sub-agent's allowlist
            out, err = f"tool '{block.name}' not permitted for {self.spec.role}", True
        else:
            try:
                out, err = (await tool.fn(block.input))[:8000], False
            except Exception as exc:  # surface failures to the model, not the orchestrator
                out, err = f"{type(exc).__name__}: {exc}", True
        return {"type": "tool_result", "tool_use_id": block.id, "content": out, "is_error": err}

    def _parse_report(self, resp: Any, turn: int, used_in: int) -> SubAgentReport:
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        try:
            data = json.loads(text[text.index("{"): text.rindex("}") + 1])
            return SubAgentReport(self.spec.role, "done", data["summary"], data.get("evidence", [])[:10], turn, used_in)
        except (ValueError, KeyError):
            return SubAgentReport(self.spec.role, "error", text[:600], [], turn, used_in)


class Orchestrator:
    def __init__(self, toolbox: dict[str, Tool], concurrency: int = 4):
        self.toolbox = toolbox
        self.sem = asyncio.Semaphore(concurrency)

    async def plan(self, goal: str) -> list[dict[str, Any]]:
        prompt = (
            f"Goal: {goal}\nAvailable tools: {sorted(self.toolbox)}\n"
            'Return JSON list of subtasks: [{"role": str, "task": str, "tools": [tool names]}]. Max 5.'
        )
        resp = await client.messages.create(model=MODEL, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text
        return json.loads(text[text.index("["): text.rindex("]") + 1])

    async def _delegate(self, sub: dict[str, Any]) -> SubAgentReport:
        tools = [self.toolbox[n] for n in sub.get("tools", []) if n in self.toolbox]
        async with self.sem:
            return await SubAgent(SubAgentSpec(sub["role"], sub["task"], tools)).run()

    async def run(self, goal: str) -> str:
        subtasks = await self.plan(goal)
        reports = await asyncio.gather(*[self._delegate(s) for s in subtasks])
        digest = "\n".join(
            f"- [{r.role}] ({r.status}, {r.turns_used} turns) {r.summary} | evidence: {r.evidence}" for r in reports
        )
        final = await client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": f"Goal: {goal}\nSub-agent reports:\n{digest}\nWrite the final answer."}],
        )
        return final.content[0].text
