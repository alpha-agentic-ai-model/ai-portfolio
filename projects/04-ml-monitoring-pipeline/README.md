# Real-Time ML Model Monitoring & Auto-Retraining Pipeline

A complete MLOps pipeline that monitors deployed models for data drift, concept drift, and performance degradation. Automatically triggers retraining with canary deployments.

## Architecture
```
[Prod Model] -> [Metrics Collector] -> [Drift Detector]
                                             |
               (drift detected)              v
[Data Pipeline] -> [Auto Retrain] -> [Canary Deploy]
```

## Tech Stack
- MLflow, Evidently AI, Kubernetes, Prometheus, Grafana, Apache Kafka
