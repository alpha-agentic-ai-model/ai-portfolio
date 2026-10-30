"""Human-approval MCP server using elicitation.

Wraps risky operations (payments, deletions, outbound email, deploys) as MCP
tools that pause mid-call and ask the human, through the client's elicitation
UI, to approve, edit or reject the exact action. Decisions are written to an
append-only audit log, and low-risk calls under a policy threshold run without
interrupting the user.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("human-approval")
AUDIT_LOG = Path("audit/approvals.jsonl")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

AUTO_APPROVE_USD = 50.0
PROTECTED_PATHS = ("prod/", "billing/", "secrets/")


class Approval(BaseModel):
    decision: Literal["approve", "reject"] = Field(description="Approve or reject this action")
    note: str = Field(default="", description="Optional reason, shown in the audit log")


class RefundApproval(Approval):
    amount_usd: float = Field(description="Amount to refund; edit to change it")


def audit(action: str, payload: dict, decision: str, actor: str, note: str = "") -> str:
    record = {"ts": time.time(), "action": action, "payload": payload, "decision": decision,
              "actor": actor, "note": note}
    record["digest"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record["digest"]


async def ask(ctx: Context, message: str, schema: type[BaseModel]):
    """Elicit a structured decision; treat decline/cancel as a rejection."""
    result = await ctx.elicit(message=message, schema=schema)
    if result.action != "accept" or result.data is None:
        return None
    return result.data


@mcp.tool()
async def issue_refund(order_id: str, amount_usd: float, reason: str, ctx: Context) -> str:
    """Refund a customer order. Amounts above the policy threshold require human approval."""
    payload = {"order_id": order_id, "amount_usd": amount_usd, "reason": reason}
    if amount_usd <= AUTO_APPROVE_USD:
        ref = audit("issue_refund", payload, "auto_approved", "policy")
        return f"Refunded ${amount_usd:.2f} on {order_id} (auto, audit {ref})"
    data = await ask(ctx, f"Refund ${amount_usd:.2f} on order {order_id}?\nReason: {reason}", RefundApproval)
    if data is None or data.decision == "reject":
        ref = audit("issue_refund", payload, "rejected", "human", getattr(data, "note", "declined"))
        return f"Refund rejected by reviewer (audit {ref}). Do not retry without new information."
    payload["amount_usd"] = data.amount_usd
    ref = audit("issue_refund", payload, "approved", "human", data.note)
    # payments.refund(order_id, data.amount_usd)  # real side effect goes here
    return f"Refunded ${data.amount_usd:.2f} on {order_id} (approved, audit {ref})"


@mcp.tool()
async def delete_path(path: str, recursive: bool, ctx: Context) -> str:
    """Delete a storage path. Protected prefixes always require approval."""
    payload = {"path": path, "recursive": recursive}
    if not path.startswith(PROTECTED_PATHS) and not recursive:
        ref = audit("delete_path", payload, "auto_approved", "policy")
        return f"Deleted {path} (audit {ref})"
    data = await ask(ctx, f"Delete {'recursively ' if recursive else ''}'{path}'? This cannot be undone.", Approval)
    if data is None or data.decision == "reject":
        ref = audit("delete_path", payload, "rejected", "human", getattr(data, "note", "declined"))
        return f"Deletion of {path} rejected (audit {ref})."
    ref = audit("delete_path", payload, "approved", "human", data.note)
    return f"Deleted {path} (approved, audit {ref})"


@mcp.tool()
async def send_email(to: list[str], subject: str, body: str, ctx: Context) -> str:
    """Send an outbound email. Always shows the human the final draft first."""
    payload = {"to": to, "subject": subject, "body_sha": hashlib.sha256(body.encode()).hexdigest()[:12]}
    preview = f"To: {', '.join(to)}\nSubject: {subject}\n\n{body[:1500]}"
    data = await ask(ctx, f"Send this email?\n\n{preview}", Approval)
    if data is None or data.decision == "reject":
        ref = audit("send_email", payload, "rejected", "human", getattr(data, "note", "declined"))
        return f"Email not sent (audit {ref}). Reviewer note: {getattr(data, 'note', '')}"
    ref = audit("send_email", payload, "approved", "human", data.note)
    return f"Email sent to {len(to)} recipient(s) (audit {ref})"


@mcp.resource("audit://recent")
def recent_decisions() -> str:
    """Last 50 approval decisions, newest first."""
    if not AUDIT_LOG.exists():
        return "[]"
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-50:]
    return json.dumps([json.loads(l) for l in reversed(lines)], indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
