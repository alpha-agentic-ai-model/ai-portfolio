import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Capability(Enum):
    """Fine-grained capabilities an agent may request at runtime."""
    FS_READ = "fs.read"
    FS_WRITE = "fs.write"
    NET_EGRESS = "net.egress"
    PROC_SPAWN = "proc.spawn"
    SECRET_READ = "secret.read"
    MCP_TOOL_CALL = "mcp.tool.call"


@dataclass(frozen=True)
class CapabilityGrant:
    """A time-boxed, resource-scoped grant issued to a single agent run."""
    capability: Capability
    resource_glob: str
    expires_at: float
    max_invocations: int = 100

    def permits(self, resource: str, now: float) -> bool:
        from fnmatch import fnmatch
        return now < self.expires_at and fnmatch(resource, self.resource_glob)


@dataclass
class SandboxSpec:
    image: str = "ghcr.io/agent-runtime/python-slim:3.12"
    runtime: str = "runsc"            # gVisor syscall interception
    cpu_millis: int = 1000
    memory_mb: int = 1024
    pids_limit: int = 64
    read_only_root: bool = True
    network_mode: str = "none"        # flipped to a filtered CNI when NET_EGRESS granted
    wall_clock_timeout_s: int = 120


class PolicyDenied(Exception):
    pass


class PolicyEngine:
    """Thin client over Open Policy Agent; falls back to deny-by-default."""

    def __init__(self, opa_url: str, bundle: str = "agent/authz"):
        self.opa_url = opa_url.rstrip("/")
        self.bundle = bundle
        self._decision_cache: dict[tuple, bool] = {}

    async def evaluate(self, principal: str, capability: Capability, resource: str) -> bool:
        key = (principal, capability.value, resource)
        if key in self._decision_cache:
            return self._decision_cache[key]
        payload = {"input": {"principal": principal, "cap": capability.value, "resource": resource}}
        allowed = await self._post(f"/v1/data/{self.bundle}/allow", payload)
        self._decision_cache[key] = allowed
        return allowed

    async def _post(self, path: str, payload: dict) -> bool:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.opa_url + path, json=payload, timeout=2) as r:
                    body = await r.json()
                    return bool(body.get("result", False))
        except Exception:
            return False  # deny-by-default on policy-plane failure


class AuditLog:
    """Append-only structured audit trail; one record per capability decision."""

    def __init__(self, sink):
        self.sink = sink

    async def record(self, **fields: Any) -> None:
        fields["ts"] = time.time()
        await self.sink.write(json.dumps(fields, separators=(",", ":")) + "\n")


class SandboxSupervisor:
    """Supervises untrusted agent code inside a capability-scoped sandbox.

    Every side effect the agent attempts is routed through check() before it
    reaches a real syscall, so the blast radius of a prompt injection is
    bounded by the grants issued for that specific run.
    """

    def __init__(self, policy: PolicyEngine, audit: AuditLog, spec: Optional[SandboxSpec] = None):
        self.policy = policy
        self.audit = audit
        self.spec = spec or SandboxSpec()
        self.grants: dict[str, list[CapabilityGrant]] = {}
        self.usage: dict[tuple[str, str], int] = {}

    def issue(self, run_id: str, capability: Capability, resource_glob: str, ttl_s: int = 300) -> CapabilityGrant:
        grant = CapabilityGrant(capability, resource_glob, time.time() + ttl_s)
        self.grants.setdefault(run_id, []).append(grant)
        return grant

    async def check(self, run_id: str, principal: str, capability: Capability, resource: str) -> None:
        now = time.time()
        local = [g for g in self.grants.get(run_id, []) if g.capability is capability]
        if not any(g.permits(resource, now) for g in local):
            await self.audit.record(run=run_id, cap=capability.value, resource=resource, verdict="no_grant")
            raise PolicyDenied(f"no active grant for {capability.value} on {resource}")
        counter_key = (run_id, capability.value)
        self.usage[counter_key] = self.usage.get(counter_key, 0) + 1
        if self.usage[counter_key] > min(g.max_invocations for g in local):
            raise PolicyDenied(f"invocation budget exhausted for {capability.value}")
        if not await self.policy.evaluate(principal, capability, resource):
            await self.audit.record(run=run_id, cap=capability.value, resource=resource, verdict="opa_deny")
            raise PolicyDenied(f"policy denied {capability.value} on {resource}")
        await self.audit.record(run=run_id, cap=capability.value, resource=resource, verdict="allow")

    async def run(self, principal: str, code: str, grants: list[tuple[Capability, str]]) -> dict:
        run_id = uuid.uuid4().hex[:12]
        for capability, glob in grants:
            self.issue(run_id, capability, glob)
        spec = self.spec
        if any(c is Capability.NET_EGRESS for c, _ in grants):
            spec = SandboxSpec(**{**spec.__dict__, "network_mode": "filtered"})
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(self._exec(run_id, spec, code), spec.wall_clock_timeout_s)
            status = "ok"
        except asyncio.TimeoutError:
            result, status = {"error": "wall_clock_timeout"}, "timeout"
        except PolicyDenied as exc:
            result, status = {"error": str(exc)}, "denied"
        await self.audit.record(run=run_id, principal=principal, status=status,
                                duration_ms=round((time.monotonic() - started) * 1000, 2))
        self.grants.pop(run_id, None)
        return {"run_id": run_id, "status": status, "result": result}

    async def _exec(self, run_id: str, spec: SandboxSpec, code: str) -> dict:
        argv = [
            "docker", "run", "--rm", f"--runtime={spec.runtime}",
            f"--cpus={spec.cpu_millis / 1000:.2f}", f"--memory={spec.memory_mb}m",
            f"--pids-limit={spec.pids_limit}", f"--network={spec.network_mode}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
        ]
        if spec.read_only_root:
            argv += ["--read-only", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m"]
        argv += [spec.image, "python", "-c", code]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return {"exit_code": proc.returncode,
                "stdout": out.decode(errors="replace")[-8192:],
                "stderr": err.decode(errors="replace")[-4096:]}
