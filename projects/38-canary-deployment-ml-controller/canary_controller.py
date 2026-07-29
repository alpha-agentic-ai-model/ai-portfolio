import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class RolloutPhase(Enum):
    CANARY_5 = 5
    CANARY_25 = 25
    CANARY_50 = 50
    CANARY_75 = 75
    FULL = 100


class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryMetrics:
    accuracy: float
    p50_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    sample_count: int
    throughput_rps: float = 0.0
    memory_mb: float = 0.0


@dataclass
class DeploymentRecord:
    model_id: str
    status: DeploymentStatus
    current_phase: RolloutPhase
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    rollback_reason: Optional[str] = None
    phase_history: list[dict] = field(default_factory=list)


class MetricsCollector:
    """Collect and aggregate model serving metrics from Prometheus."""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url

    async def get_metrics(self, model_id: str, window_minutes: int = 5) -> CanaryMetrics:
        """Query Prometheus for model serving metrics."""
        accuracy = await self._query_gauge(
            f'model_accuracy{{model_id="{model_id}"}}', window_minutes
        )
        p50_latency = await self._query_histogram(
            f'inference_latency_seconds{{model_id="{model_id}"}}', 0.5, window_minutes
        )
        p99_latency = await self._query_histogram(
            f'inference_latency_seconds{{model_id="{model_id}"}}', 0.99, window_minutes
        )
        error_rate = await self._query_rate(
            f'inference_errors_total{{model_id="{model_id}"}}', window_minutes
        )
        sample_count = await self._query_counter(
            f'inference_requests_total{{model_id="{model_id}"}}', window_minutes
        )
        throughput = sample_count / (window_minutes * 60) if window_minutes > 0 else 0

        return CanaryMetrics(
            accuracy=accuracy,
            p50_latency_ms=p50_latency * 1000,
            p99_latency_ms=p99_latency * 1000,
            error_rate=error_rate,
            sample_count=int(sample_count),
            throughput_rps=throughput,
        )

    async def _query_gauge(self, query: str, window: int) -> float:
        # Prometheus query implementation
        return 0.0

    async def _query_histogram(self, query: str, quantile: float, window: int) -> float:
        return 0.0

    async def _query_rate(self, query: str, window: int) -> float:
        return 0.0

    async def _query_counter(self, query: str, window: int) -> float:
        return 0.0


class TrafficManager:
    """Manage traffic splitting between baseline and canary models via Istio."""

    def __init__(self, k8s_client: Any, namespace: str = "ml-serving"):
        self.k8s = k8s_client
        self.namespace = namespace

    async def set_weight(self, model_id: str, canary_pct: int):
        """Update Istio VirtualService weights for canary traffic splitting."""
        baseline_pct = 100 - canary_pct
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {"name": f"model-{model_id}", "namespace": self.namespace},
            "spec": {
                "http": [{
                    "route": [
                        {"destination": {"host": f"model-{model_id}-baseline"}, "weight": baseline_pct},
                        {"destination": {"host": f"model-{model_id}-canary"}, "weight": canary_pct},
                    ]
                }]
            },
        }
        await self.k8s.apply(virtual_service)
        logger.info(f"Traffic split for {model_id}: baseline={baseline_pct}%, canary={canary_pct}%")

    async def route_all_to_baseline(self, model_id: str):
        """Route 100% traffic to baseline (rollback)."""
        await self.set_weight(model_id, 0)

    async def route_all_to_canary(self, model_id: str):
        """Route 100% traffic to canary (promotion)."""
        await self.set_weight(model_id, 100)


class CanaryController:
    """Automated canary deployment controller with metric-based promotion."""

    def __init__(
        self,
        model_registry: Any,
        traffic_mgr: TrafficManager,
        metrics_collector: MetricsCollector,
        accuracy_threshold: float = 0.02,
        latency_threshold: float = 1.2,
        error_rate_threshold: float = 1.5,
        soak_time_seconds: int = 300,
        min_samples: int = 100,
    ):
        self.registry = model_registry
        self.traffic = traffic_mgr
        self.metrics = metrics_collector
        self.accuracy_thr = accuracy_threshold
        self.latency_thr = latency_threshold
        self.error_rate_thr = error_rate_threshold
        self.soak_time = soak_time_seconds
        self.min_samples = min_samples
        self.deployments: dict[str, DeploymentRecord] = {}

    async def deploy_canary(self, model_id: str) -> DeploymentRecord:
        """Execute a full canary deployment with phased rollout."""
        record = DeploymentRecord(
            model_id=model_id,
            status=DeploymentStatus.IN_PROGRESS,
            current_phase=RolloutPhase.CANARY_5,
        )
        self.deployments[model_id] = record

        baseline_metrics = await self.metrics.get_metrics(f"{model_id}-baseline", window_minutes=30)
        logger.info(f"Baseline metrics: accuracy={baseline_metrics.accuracy:.4f}, "
                     f"p99={baseline_metrics.p99_latency_ms:.1f}ms")

        phases = list(RolloutPhase)
        for phase in phases:
            record.current_phase = phase
            await self.traffic.set_weight(model_id, phase.value)
            logger.info(f"Phase {phase.name}: traffic at {phase.value}%")

            # Soak period
            await asyncio.sleep(self.soak_time)

            # Collect canary metrics
            canary_metrics = await self.metrics.get_metrics(f"{model_id}-canary")

            phase_record = {
                "phase": phase.name,
                "timestamp": time.time(),
                "canary_accuracy": canary_metrics.accuracy,
                "canary_p99_ms": canary_metrics.p99_latency_ms,
                "canary_error_rate": canary_metrics.error_rate,
                "canary_samples": canary_metrics.sample_count,
            }
            record.phase_history.append(phase_record)

            # Check minimum samples
            if canary_metrics.sample_count < self.min_samples and phase != RolloutPhase.FULL:
                logger.warning(f"Insufficient samples ({canary_metrics.sample_count}), extending soak")
                await asyncio.sleep(self.soak_time)
                canary_metrics = await self.metrics.get_metrics(f"{model_id}-canary")

            # Evaluate health
            healthy, reason = self.evaluate_health(canary_metrics, baseline_metrics)
            if not healthy:
                await self.rollback(model_id, reason)
                record.status = DeploymentStatus.ROLLED_BACK
                record.rollback_reason = reason
                record.completed_at = time.time()
                return record

        # Full promotion
        await self.registry.promote(model_id)
        record.status = DeploymentStatus.PROMOTED
        record.completed_at = time.time()
        logger.info(f"Model {model_id} promoted to production")
        return record

    def evaluate_health(self, canary: CanaryMetrics, baseline: CanaryMetrics) -> tuple[bool, str]:
        """Compare canary metrics against baseline with thresholds."""
        accuracy_delta = baseline.accuracy - canary.accuracy
        if accuracy_delta > self.accuracy_thr:
            return False, f"Accuracy degraded by {accuracy_delta:.4f} (threshold: {self.accuracy_thr})"

        latency_ratio = canary.p99_latency_ms / max(baseline.p99_latency_ms, 0.1)
        if latency_ratio > self.latency_thr:
            return False, f"P99 latency {latency_ratio:.2f}x baseline (threshold: {self.latency_thr}x)"

        error_ratio = canary.error_rate / max(baseline.error_rate, 1e-6)
        if error_ratio > self.error_rate_thr:
            return False, f"Error rate {error_ratio:.2f}x baseline (threshold: {self.error_rate_thr}x)"

        return True, "All metrics within thresholds"

    async def rollback(self, model_id: str, reason: str = ""):
        """Rollback canary deployment and alert."""
        logger.warning(f"Rolling back {model_id}: {reason}")
        await self.traffic.route_all_to_baseline(model_id)
        await self.registry.mark_failed(model_id, reason)
