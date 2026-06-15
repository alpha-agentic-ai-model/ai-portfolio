# Real-Time Feature Store with Online-Offline Consistency

## Overview
A production feature store guaranteeing consistency between offline training features and online serving features with sub-10ms latency.

## Architecture
```
[Event Stream] → [Stream Processor] → [Online Store (Redis)]
 ↓
[Batch Processor] → [Offline Store (Parquet)] → [Training Pipeline]
 ↓
[Consistency Checker] → [Drift Monitor] → [Feature Catalog]
```

## Key Features
- Sub-10ms online feature serving via Redis
- Point-in-time correct joins for training data consistency
- Automated feature drift monitoring with alerting
- Self-serve feature catalog with full lineage tracking
- Unified computation engine ensures online-offline parity

## Tech Stack
- **Apache Kafka** — Event streaming for real-time features
- **Redis** — Online feature store with sub-ms reads
- **Apache Spark** — Batch feature computation
- **DuckDB** — Lightweight offline analytics
- **FastAPI** — Feature serving API
- **Prometheus** — Latency & drift monitoring
