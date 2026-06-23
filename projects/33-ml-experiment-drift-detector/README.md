# ML Experiment Drift Detection Platform

## Overview
An automated platform that monitors production ML models for data drift, concept drift, and performance degradation. Uses statistical tests (PSI, KS, Jensen-Shannon) with adaptive thresholds and triggers automatic retraining.

## Architecture
```
[Prediction Logs] → [Feature Store] → [Drift Analyzer]
  |
[Alert Engine] → [Retrain Trigger] → [Model Registry] → [Dashboard]
```

## Tech Stack
Evidently AI, MLflow, Prometheus, Grafana, Python, scikit-learn

## Key Features
- Production-ready implementation with error handling
- Comprehensive type annotations and documentation
- Modular architecture for easy extension
- Built for scalability and performance
