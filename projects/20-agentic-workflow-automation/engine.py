"""Agentic Workflow Engine with Human-in-the-Loop

Enterprise workflow automation engine where AI agents plan, execute, and monitor
multi-step business processes with built-in human approval gates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Optional
import asyncio
import time
import logging
import json

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class WorkflowStep:
    name: str
    agent_fn: Callable
    requires_approval: bool = False
    retry_limit: int = 3
    timeout_seconds: int = 300
    depends_on: list[str] = field(default_factory=list)
    rollback_fn: Optional[Callable] = None


@dataclass
class StepResult:
    step_name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    attempts: int = 1
    approved_by: Optional[str] = None


@dataclass
class AuditEntry:
    timestamp: float
    workflow_id: str
    step_name: str
    action: str
    details: dict = field(default_factory=dict)


class CheckpointStore:
    """Persistent checkpoint storage for workflow state."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def save(self, workflow_id: str, step_name: str, state: dict):
        key = f"{workflow_id}:{step_name}"
        self._store[key] = {
            "state": state,
            "timestamp": time.time(),
        }
        logger.info(f"Checkpoint saved: {key}")

    def load(self, workflow_id: str, step_name: str) -> Optional[dict]:
        key = f"{workflow_id}:{step_name}"
        entry = self._store.get(key)
        return entry["state"] if entry else None

    def get_latest(self, workflow_id: str) -> Optional[str]:
        prefix = f"{workflow_id}:"
        matching = [k for k in self._store if k.startswith(prefix)]
        if not matching:
            return None
        latest = max(matching, key=lambda k: self._store[k]["timestamp"])
        return latest.split(":", 1)[1]


class AuditLogger:
    """Compliance audit trail for workflow execution."""

    def __init__(self):
        self._entries: list[AuditEntry] = []

    def record(self, workflow_id: str, step_name: str,
               action: str, details: dict = None):
        entry = AuditEntry(
            timestamp=time.time(),
            workflow_id=workflow_id,
            step_name=step_name,
            action=action,
            details=details or {},
        )
        self._entries.append(entry)
        logger.info(f"Audit: [{workflow_id}] {step_name} -> {action}")

    def get_trail(self, workflow_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.workflow_id == workflow_id]


class ApprovalGate:
    """Human approval gate for sensitive workflow steps."""

    def __init__(self):
        self._pending: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, tuple[bool, str]] = {}

    async def request_approval(self, workflow_id: str,
                                step_name: str, context: dict,
                                timeout: int = 3600) -> tuple[bool, str]:
        key = f"{workflow_id}:{step_name}"
        self._pending[key] = asyncio.Event()
        logger.info(f"Approval requested: {key}")

        try:
            await asyncio.wait_for(
                self._pending[key].wait(), timeout=timeout
            )
            approved, approver = self._decisions.get(key, (False, "timeout"))
            return approved, approver
        except asyncio.TimeoutError:
            return False, "timeout"

    def approve(self, workflow_id: str, step_name: str, approver: str):
        key = f"{workflow_id}:{step_name}"
        self._decisions[key] = (True, approver)
        if key in self._pending:
            self._pending[key].set()

    def reject(self, workflow_id: str, step_name: str, approver: str):
        key = f"{workflow_id}:{step_name}"
        self._decisions[key] = (False, approver)
        if key in self._pending:
            self._pending[key].set()


class WorkflowEngine:
    """DAG-based workflow engine with human-in-the-loop and checkpointing."""

    def __init__(self):
        self.steps: dict[str, WorkflowStep] = {}
        self.checkpoints = CheckpointStore()
        self.audit_log = AuditLogger()
        self.approval_gate = ApprovalGate()

    def add_step(self, step: WorkflowStep):
        self.steps[step.name] = step

    def compile_dag(self) -> dict[str, list[str]]:
        dag = {}
        for name, step in self.steps.items():
            dag[name] = step.depends_on
        return dag

    def topological_batches(self, dag: dict) -> list[list[str]]:
        in_degree = {n: 0 for n in dag}
        for deps in dag.values():
            for d in deps:
                if d in in_degree:
                    in_degree[d] = in_degree.get(d, 0)

        remaining = dict(dag)
        batches = []
        completed = set()

        while remaining:
            batch = [
                n for n, deps in remaining.items()
                if all(d in completed for d in deps)
            ]
            if not batch:
                raise ValueError("Circular dependency detected in workflow DAG")
            batches.append(batch)
            completed.update(batch)
            for n in batch:
                del remaining[n]

        return batches

    async def run_step(self, step_name: str, workflow_id: str,
                       context: dict) -> StepResult:
        step = self.steps[step_name]
        self.audit_log.record(workflow_id, step_name, "started")

        for attempt in range(1, step.retry_limit + 1):
            try:
                start = time.perf_counter()
                output = await asyncio.wait_for(
                    step.agent_fn(context),
                    timeout=step.timeout_seconds,
                )
                duration = (time.perf_counter() - start) * 1000

                if step.requires_approval:
                    self.audit_log.record(workflow_id, step_name,
                                          "awaiting_approval")
                    approved, approver = await self.approval_gate.request_approval(
                        workflow_id, step_name, {"output": str(output)[:500]}
                    )
                    if not approved:
                        self.audit_log.record(
                            workflow_id, step_name, "rejected",
                            {"approver": approver},
                        )
                        return StepResult(
                            step_name=step_name,
                            status=StepStatus.FAILED,
                            error=f"Rejected by {approver}",
                            duration_ms=duration,
                            attempts=attempt,
                        )
                    self.audit_log.record(
                        workflow_id, step_name, "approved",
                        {"approver": approver},
                    )

                self.checkpoints.save(workflow_id, step_name,
                                       {"output": str(output)[:1000]})
                self.audit_log.record(workflow_id, step_name, "completed")

                return StepResult(
                    step_name=step_name,
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration_ms=duration,
                    attempts=attempt,
                )

            except Exception as e:
                logger.error(f"Step '{step_name}' failed (attempt {attempt}): {e}")
                if attempt == step.retry_limit:
                    self.audit_log.record(
                        workflow_id, step_name, "failed",
                        {"error": str(e), "attempts": attempt},
                    )
                    return StepResult(
                        step_name=step_name,
                        status=StepStatus.FAILED,
                        error=str(e),
                        attempts=attempt,
                    )

    async def execute(self, workflow_id: str, context: dict) -> list[StepResult]:
        dag = self.compile_dag()
        batches = self.topological_batches(dag)
        all_results = []

        self.audit_log.record(workflow_id, "__workflow__", "started")

        for batch in batches:
            tasks = [
                self.run_step(step_name, workflow_id, context)
                for step_name in batch
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            failed = [r for r in results if r.status == StepStatus.FAILED]
            if failed:
                logger.error(f"Workflow {workflow_id} failed at: "
                             f"{[f.step_name for f in failed]}")
                break

        self.audit_log.record(workflow_id, "__workflow__", "completed")
        return all_results
