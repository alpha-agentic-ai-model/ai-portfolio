"""MCP Secure Data Gateway with Policy-Based Access

Model Context Protocol gateway providing AI agents with secure, policy-controlled
access to enterprise databases, APIs, and file systems.
"""

from mcp.server.fastmcp import FastMCP
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from dataclasses import dataclass, field
from typing import Optional
import time
import json
import logging
import re

logger = logging.getLogger(__name__)

mcp = FastMCP("secure-data-gateway")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


@dataclass
class AccessPolicy:
    """Policy governing an agent's data access permissions."""
    agent_id: str
    allowed_tables: list[str] = field(default_factory=list)
    allowed_operations: list[str] = field(default_factory=lambda: ["SELECT"])
    max_rows: int = 1000
    redact_pii: bool = True
    rate_limit_rpm: int = 60
    allowed_columns: Optional[dict[str, list[str]]] = None


@dataclass
class AuditRecord:
    timestamp: float
    agent_id: str
    operation: str
    resource: str
    rows_returned: int
    pii_redacted: bool
    policy_applied: str
    latency_ms: float


class RateLimiter:
    """Token bucket rate limiter per agent identity."""

    def __init__(self):
        self._buckets: dict[str, list[float]] = {}

    def check(self, agent_id: str, limit_rpm: int) -> bool:
        now = time.time()
        if agent_id not in self._buckets:
            self._buckets[agent_id] = []

        # Clean old entries (older than 60s)
        self._buckets[agent_id] = [
            t for t in self._buckets[agent_id] if now - t < 60
        ]

        if len(self._buckets[agent_id]) >= limit_rpm:
            return False

        self._buckets[agent_id].append(now)
        return True


class PolicyEngine:
    """OPA-based policy engine for access control decisions."""

    def __init__(self, opa_url: str = "http://localhost:8181"):
        self.opa_url = opa_url
        self._cache: dict[str, tuple[AccessPolicy, float]] = {}
        self.cache_ttl = 300  # 5 minutes

    def evaluate(self, agent_id: str, resource: str) -> Optional[AccessPolicy]:
        cache_key = f"{agent_id}:{resource}"
        if cache_key in self._cache:
            policy, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.cache_ttl:
                return policy

        # In production, query OPA:
        # response = httpx.post(f"{self.opa_url}/v1/data/gateway/allow",
        #     json={"input": {"agent": agent_id, "resource": resource}})
        # For demo, return a default policy
        policy = AccessPolicy(
            agent_id=agent_id,
            allowed_tables=["users", "orders", "products", "analytics"],
            max_rows=500,
            redact_pii=True,
            rate_limit_rpm=30,
        )
        self._cache[cache_key] = (policy, time.time())
        return policy


class AuditLogger:
    """Compliance-grade audit logging for all data access."""

    def __init__(self):
        self._records: list[AuditRecord] = []

    def record(self, agent_id: str, operation: str, resource: str,
               rows: int, pii_redacted: bool, policy: str,
               latency_ms: float):
        entry = AuditRecord(
            timestamp=time.time(),
            agent_id=agent_id,
            operation=operation,
            resource=resource,
            rows_returned=rows,
            pii_redacted=pii_redacted,
            policy_applied=policy,
            latency_ms=latency_ms,
        )
        self._records.append(entry)
        logger.info(
            f"AUDIT: agent={agent_id} op={operation} resource={resource} "
            f"rows={rows} pii_redacted={pii_redacted}"
        )

    def get_agent_history(self, agent_id: str,
                          limit: int = 100) -> list[AuditRecord]:
        return [r for r in self._records
                if r.agent_id == agent_id][-limit:]


class PIIRedactor:
    """Detects and redacts PII from query results."""

    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> tuple[str, int]:
        results = self.analyzer.analyze(
            text=text, language="en",
            entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
                       "US_SSN", "PERSON", "IP_ADDRESS"],
        )
        if not results:
            return text, 0
        anonymized = self.anonymizer.anonymize(
            text=text, analyzer_results=results,
        )
        return anonymized.text, len(results)


def extract_tables(sql: str) -> list[str]:
    """Extract table names from a SQL query."""
    pattern = r"\bFROM\s+([\w.]+)|\bJOIN\s+([\w.]+)"
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return list(set(m[0] or m[1] for m in matches))


def validate_readonly(sql: str) -> bool:
    """Ensure SQL query is read-only."""
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                  "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
    sql_upper = sql.upper().strip()
    return not any(sql_upper.startswith(kw) for kw in forbidden)


# Initialize components
engine = PolicyEngine()
rate_limiter = RateLimiter()
audit_log = AuditLogger()
pii_redactor = PIIRedactor()


@mcp.tool()
async def query_database(sql: str, agent_id: str) -> str:
    """Execute a read-only database query with policy enforcement and PII redaction."""
    start = time.perf_counter()

    if not validate_readonly(sql):
        raise PermissionError("Only read-only queries are allowed")

    tables = extract_tables(sql)
    policy = engine.evaluate(agent_id, ",".join(tables))
    if not policy:
        raise PermissionError(f"No access policy found for agent {agent_id}")

    if not rate_limiter.check(agent_id, policy.rate_limit_rpm):
        raise PermissionError(f"Rate limit exceeded for agent {agent_id}")

    # Verify table access
    unauthorized = [t for t in tables if t not in policy.allowed_tables]
    if unauthorized:
        raise PermissionError(f"Access denied to tables: {unauthorized}")

    # Execute query (in production, use SQLAlchemy)
    # result = await db.execute_readonly(sql, max_rows=policy.max_rows)
    result_text = f"Query executed: {sql[:100]}..."  # placeholder

    pii_count = 0
    if policy.redact_pii:
        result_text, pii_count = pii_redactor.redact(result_text)

    latency = (time.perf_counter() - start) * 1000
    audit_log.record(
        agent_id=agent_id, operation="SELECT",
        resource=",".join(tables), rows=0,
        pii_redacted=pii_count > 0,
        policy=f"policy:{agent_id}", latency_ms=latency,
    )

    return result_text


@mcp.tool()
async def get_access_report(agent_id: str) -> str:
    """Get audit trail for an agent's data access history."""
    history = audit_log.get_agent_history(agent_id)
    return json.dumps([{
        "timestamp": r.timestamp,
        "operation": r.operation,
        "resource": r.resource,
        "rows": r.rows_returned,
        "pii_redacted": r.pii_redacted,
    } for r in history], indent=2)


if __name__ == "__main__":
    mcp.run()
