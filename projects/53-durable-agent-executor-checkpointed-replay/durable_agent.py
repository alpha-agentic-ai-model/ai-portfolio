"""Durable agent executor: every model call and tool call is journaled so a
crashed or redeployed long-running agent resumes from its last completed step
instead of re-running side effects."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

MODEL = "claude-sonnet-4-5"


@dataclass
class StepRecord:
    run_id: str
    seq: int
    kind: str          # "llm" | "tool"
    key: str           # idempotency key (hash of inputs)
    output: Any


class Journal:
    """Append-only step journal backed by SQLite (swap for Postgres in prod)."""

    def __init__(self, path: str = "journal.db") -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS steps (run_id TEXT, seq INTEGER, kind TEXT,"
            " key TEXT, output TEXT, ts REAL, PRIMARY KEY (run_id, seq))"
        )

    def get(self, run_id: str, seq: int) -> StepRecord | None:
        row = self.db.execute(
            "SELECT kind, key, output FROM steps WHERE run_id=? AND seq=?", (run_id, seq)
        ).fetchone()
        return StepRecord(run_id, seq, row[0], row[1], json.loads(row[2])) if row else None

    def put(self, rec: StepRecord) -> None:
        self.db.execute(
            "INSERT INTO steps VALUES (?,?,?,?,?,?)",
            (rec.run_id, rec.seq, rec.kind, rec.key, json.dumps(rec.output), time.time()),
        )
        self.db.commit()


class NonDeterminismError(RuntimeError):
    """Raised when replayed code diverges from the journaled history."""


@dataclass
class DurableAgent:
    run_id: str
    tools: dict[str, Callable[..., Any]]
    tool_specs: list[dict]
    journal: Journal = field(default_factory=Journal)
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)
    _seq: int = 0

    def _step(self, kind: str, payload: dict, fn: Callable[[], Any]) -> Any:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        self._seq += 1
        cached = self.journal.get(self.run_id, self._seq)
        if cached is not None:
            if cached.key != key:
                raise NonDeterminismError(f"step {self._seq}: history={cached.key} now={key}")
            return cached.output                      # replay: no side effects
        output = fn()                                 # first execution
        self.journal.put(StepRecord(self.run_id, self._seq, kind, key, output))
        return output

    def _llm(self, messages: list[dict]) -> dict:
        def call() -> dict:
            resp = self.client.messages.create(
                model=MODEL, max_tokens=2048, tools=self.tool_specs, messages=messages
            )
            return resp.model_dump(include={"content", "stop_reason"})
        return self._step("llm", {"messages": messages}, call)

    def _tool(self, name: str, args: dict) -> str:
        return self._step("tool", {"name": name, "args": args},
                          lambda: str(self.tools[name](**args)))

    def run(self, task: str, max_turns: int = 30) -> str:
        messages: list[dict] = [{"role": "user", "content": task}]
        for _ in range(max_turns):
            resp = self._llm(messages)
            messages.append({"role": "assistant", "content": resp["content"]})
            if resp["stop_reason"] != "tool_use":
                return "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")
            results = [
                {"type": "tool_result", "tool_use_id": b["id"], "content": self._tool(b["name"], b["input"])}
                for b in resp["content"] if b["type"] == "tool_use"
            ]
            messages.append({"role": "user", "content": results})
        raise TimeoutError(f"run {self.run_id} exceeded {max_turns} turns")


def send_invoice(customer_id: str, amount: float) -> dict:
    # side-effecting tool: must run exactly once even across crashes
    print(f"[side effect] invoicing {customer_id} for {amount}")
    return {"invoice_id": f"inv_{customer_id}", "amount": amount}


if __name__ == "__main__":
    specs = [{
        "name": "send_invoice",
        "description": "Send an invoice to a customer",
        "input_schema": {"type": "object", "properties": {
            "customer_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["customer_id", "amount"]},
    }]
    agent = DurableAgent(run_id="billing-2026-10-20", tools={"send_invoice": send_invoice}, tool_specs=specs)
    # Kill the process mid-run and start it again: completed steps replay from the journal.
    print(agent.run("Invoice customer acme-42 for 1200 USD and confirm."))
