"""MCP server conformance fuzzer.

Connects to any MCP server over stdio, lists its tools, and generates valid and
adversarial arguments from each tool's JSON Schema with Hypothesis. Every call
is checked for protocol conformance: schema-valid inputs must not crash the
server, invalid inputs must be rejected cleanly, and responses must be well
formed and inside a latency budget.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class Finding:
    tool: str
    kind: str  # crash | accepted_invalid | malformed_response | slow
    args: dict
    detail: str


@dataclass
class Report:
    server: str
    tools_tested: int = 0
    calls: int = 0
    findings: list[Finding] = field(default_factory=list)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for f in self.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        return {"server": self.server, "tools": self.tools_tested,
                "calls": self.calls, "findings": kinds}


def invalid_args(schema: dict) -> st.SearchStrategy[dict]:
    """Mutate a schema-valid object: drop required keys, flip types, add junk."""
    props = list(schema.get("properties", {}))
    required = schema.get("required", [])
    junk = st.one_of(st.integers(), st.text(max_size=5), st.none(), st.lists(st.integers(), max_size=2))
    drop = st.just({k: "x" for k in props if k not in required[:1]}) if required else st.nothing()
    flip = st.fixed_dictionaries({p: junk for p in props}) if props else st.nothing()
    extra = st.just({"__unexpected__": "\u0000" * 3})
    return st.one_of(drop, flip, extra)


class ConformanceFuzzer:
    def __init__(self, command: str, args: list[str], latency_budget_s: float = 5.0,
                 examples: int = 40):
        self.params = StdioServerParameters(command=command, args=args)
        self.budget = latency_budget_s
        self.examples = examples
        self.report = Report(server=" ".join([command, *args]))

    async def _call(self, session: ClientSession, tool: str, args: dict, valid: bool) -> None:
        self.report.calls += 1
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(session.call_tool(tool, args), timeout=self.budget * 2)
        except asyncio.TimeoutError:
            self.report.findings.append(Finding(tool, "slow", args, "no response within 2x budget"))
            return
        except Exception as exc:  # transport-level failure = server crashed or broke protocol
            self.report.findings.append(Finding(tool, "crash", args, repr(exc)[:300]))
            return
        elapsed = time.perf_counter() - start
        if elapsed > self.budget:
            self.report.findings.append(Finding(tool, "slow", args, f"{elapsed:.2f}s"))
        if not isinstance(result.content, list):
            self.report.findings.append(Finding(tool, "malformed_response", args, "content not a list"))
        if not valid and not result.isError:
            self.report.findings.append(
                Finding(tool, "accepted_invalid", args, "invalid input returned isError=false"))
        if valid and result.isError:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            if "Traceback" in text or "Internal" in text:
                self.report.findings.append(Finding(tool, "crash", args, text[:300]))

    def _sample(self, strategy: st.SearchStrategy[dict]) -> list[dict]:
        samples: list[dict] = []

        @settings(max_examples=self.examples, database=None, deadline=None,
                  suppress_health_check=list(HealthCheck))
        @given(strategy)
        def collect(x: dict) -> None:
            samples.append(x)

        collect()
        return samples

    async def run(self) -> Report:
        async with stdio_client(self.params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                self.report.tools_tested = len(tools)
                for tool in tools:
                    schema = tool.inputSchema or {"type": "object"}
                    Draft202012Validator.check_schema(schema)
                    validator = Draft202012Validator(schema)
                    for args in self._sample(from_schema(schema)):
                        await self._call(session, tool.name, args, valid=True)
                    for args in self._sample(invalid_args(schema)):
                        is_valid = validator.is_valid(args)
                        await self._call(session, tool.name, args, valid=is_valid)
        return self.report


if __name__ == "__main__":
    fuzzer = ConformanceFuzzer("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    report = asyncio.run(fuzzer.run())
    print(json.dumps(report.summary(), indent=2))
