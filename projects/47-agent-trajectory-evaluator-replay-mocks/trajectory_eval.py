"""Agent trajectory evaluator with replayable tool mocks.

Records every tool call an agent makes, replays them deterministically against
recorded fixtures, and scores the trajectory on goal completion, tool-call
precision, redundant steps and recovery from injected tool failures.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    output: Any = None
    error: str | None = None

    @property
    def key(self) -> str:
        blob = json.dumps({"n": self.name, "a": self.args}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class Trajectory:
    task_id: str
    calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None


class ReplayToolbox:
    """Serves recorded tool outputs; optionally injects faults to test recovery."""

    def __init__(self, fixture_dir: Path, fault_plan: dict[str, int] | None = None):
        self.fixtures: dict[str, Any] = {}
        for f in fixture_dir.glob("*.json"):
            self.fixtures.update(json.loads(f.read_text()))
        self.fault_plan = fault_plan or {}
        self.seen: dict[str, int] = {}

    def call(self, name: str, args: dict[str, Any]) -> ToolCall:
        tc = ToolCall(name, args)
        count = self.seen.get(name, 0)
        self.seen[name] = count + 1
        if self.fault_plan.get(name) == count:
            tc.error = f"InjectedFault: {name} timed out"
        elif tc.key in self.fixtures:
            tc.output = self.fixtures[tc.key]
        else:
            tc.error = f"UnrecordedCall: {name}({args})"
        return tc


@dataclass
class EvalCase:
    task_id: str
    prompt: str
    expected_tools: list[str]
    answer_check: Callable[[str], bool]
    max_steps: int = 12


@dataclass
class Score:
    task_id: str
    success: bool
    tool_precision: float
    tool_recall: float
    redundant_calls: int
    recovered_from_faults: bool
    steps: int

    def as_row(self) -> dict[str, Any]:
        return self.__dict__


class TrajectoryEvaluator:
    def __init__(self, agent_factory: Callable[[ReplayToolbox], Any], fixture_dir: Path):
        self.agent_factory = agent_factory
        self.fixture_dir = fixture_dir

    def run_case(self, case: EvalCase, fault_plan: dict[str, int] | None = None) -> Score:
        toolbox = ReplayToolbox(self.fixture_dir, fault_plan)
        agent = self.agent_factory(toolbox)
        traj: Trajectory = agent.run(case.prompt, max_steps=case.max_steps)
        used = [c.name for c in traj.calls]
        relevant = [n for n in used if n in case.expected_tools]
        precision = len(relevant) / len(used) if used else 0.0
        recall = len(set(relevant)) / len(set(case.expected_tools)) if case.expected_tools else 1.0
        keys = [c.key for c in traj.calls if c.error is None]
        redundant = len(keys) - len(set(keys))
        faults_hit = any(c.error and c.error.startswith("InjectedFault") for c in traj.calls)
        success = bool(traj.final_answer) and case.answer_check(traj.final_answer)
        return Score(case.task_id, success, round(precision, 3), round(recall, 3),
                     redundant, faults_hit and success, len(traj.calls))

    def run_suite(self, cases: list[EvalCase], with_faults: bool = True) -> dict[str, Any]:
        rows = []
        for case in cases:
            rows.append(self.run_case(case).as_row())
            if with_faults and case.expected_tools:
                fault = {case.expected_tools[0]: 0}
                row = self.run_case(case, fault).as_row()
                row["task_id"] += ":fault"
                rows.append(row)
        n = len(rows) or 1
        return {
            "pass_rate": sum(r["success"] for r in rows) / n,
            "mean_precision": sum(r["tool_precision"] for r in rows) / n,
            "mean_recall": sum(r["tool_recall"] for r in rows) / n,
            "fault_recovery_rate": (
                sum(r["recovered_from_faults"] for r in rows if r["task_id"].endswith(":fault"))
                / max(1, sum(r["task_id"].endswith(":fault") for r in rows))
            ),
            "rows": rows,
        }


if __name__ == "__main__":
    print("Wire an agent_factory that accepts a ReplayToolbox and returns .run(prompt, max_steps)")
