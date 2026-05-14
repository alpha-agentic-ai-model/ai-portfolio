# MCP Secure Data Gateway with Policy-Based Access

## Overview
A Model Context Protocol gateway that provides AI agents with secure, policy-controlled access to enterprise databases, APIs, and file systems. Features OPA-based access policies, automatic PII redaction, query audit logging, rate limiting per agent identity, and data lineage tracking.

## Architecture
```
[Agent Request] → [Auth & Identity] → [Policy Engine (OPA)]
       ↓
[PII Redactor] → [Data Source Router] → [DB / API / Files]
       ↓
[Audit Logger] → [Rate Limiter] → [Lineage Tracker]
```

## Tech Stack
- **MCP SDK** — Model Context Protocol server
- **Open Policy Agent** — Policy evaluation engine
- **FastAPI** — Gateway API layer
- **SQLAlchemy** — Database abstraction
- **Presidio** — PII detection and redaction
- **Redis** — Rate limiting and caching

## Key Features
- OPA-based access policies per agent identity
- Automatic PII detection and redaction using Presidio
- Read-only query enforcement with table-level ACLs
- Per-agent rate limiting with token bucket algorithm
- Compliance-grade audit trail for all data access
- Policy caching for sub-millisecond evaluation
