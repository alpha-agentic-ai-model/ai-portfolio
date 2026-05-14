"""LLM Observability & Trace Analytics Platform

Production-grade distributed tracing for multi-model LLM pipelines.
Captures token usage, latency, cost attribution, and hallucination metrics.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from dataclasses import dataclass, field
from typing import Optional
import time
import json
import logging

logger = logging.getLogger(__name__)

COST_PER_1K_TOKENS = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.00025, "output": 0.00125},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

@dataclass
class LLMSpan:
    """Represents a single LLM invocation trace span."""
    trace_id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    hallucination_score: float = 0.0
    metadata: dict = field(default_factory=dict)

@dataclass
class AnomalyAlert:
    metric: str
    current_value: float
    threshold: float
    severity: str  # "warning" | "critical"
    span: LLMSpan = None


class AnomalyDetector:
    """Detects anomalies in LLM metrics using rolling statistics."""

    def __init__(self, window_size: int = 100, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._latency_buffer: list[float] = []
        self._cost_buffer: list[float] = []
        self._hallucination_buffer: list[float] = []

    def check(self, span: LLMSpan) -> list[AnomalyAlert]:
        alerts = []
        alerts.extend(self._check_metric(
            "latency_ms", span.latency_ms, self._latency_buffer, span
        ))
        alerts.extend(self._check_metric(
            "cost_usd", span.cost_usd, self._cost_buffer, span
        ))
        alerts.extend(self._check_metric(
            "hallucination_score", span.hallucination_score,
            self._hallucination_buffer, span
        ))
        return alerts

    def _check_metric(self, name, value, buffer, span) -> list[AnomalyAlert]:
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)
        if len(buffer) < 10:
            return []
        mean = sum(buffer) / len(buffer)
        std = (sum((x - mean) ** 2 for x in buffer) / len(buffer)) ** 0.5
        if std == 0:
            return []
        z_score = abs(value - mean) / std
        if z_score > self.z_threshold:
            severity = "critical" if z_score > 4.0 else "warning"
            return [AnomalyAlert(
                metric=name, current_value=value,
                threshold=mean + self.z_threshold * std,
                severity=severity, span=span,
            )]
        return []


class LLMTraceCollector:
    """Collects and exports LLM trace data with cost attribution."""

    def __init__(self, service_name: str, exporter=None):
        self.provider = TracerProvider()
        if exporter:
            self.provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(service_name)
        self.anomaly_detector = AnomalyDetector()
        self._spans: list[LLMSpan] = []

    def _calculate_cost(self, model: str, prompt_tokens: int,
                        completion_tokens: int) -> float:
        rates = COST_PER_1K_TOKENS.get(model, {"input": 0.001, "output": 0.002})
        return (prompt_tokens * rates["input"] / 1000 +
                completion_tokens * rates["output"] / 1000)

    def trace_llm_call(self, model: str, prompt_tokens: int,
                       completion_tokens: int,
                       hallucination_score: float = 0.0,
                       metadata: Optional[dict] = None) -> LLMSpan:
        with self.tracer.start_as_current_span("llm_call") as otel_span:
            start = time.perf_counter()

            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

            otel_span.set_attribute("llm.model", model)
            otel_span.set_attribute("llm.tokens.prompt", prompt_tokens)
            otel_span.set_attribute("llm.tokens.completion", completion_tokens)
            otel_span.set_attribute("llm.cost_usd", cost)
            otel_span.set_attribute("llm.hallucination_score", hallucination_score)

            latency = (time.perf_counter() - start) * 1000

            span = LLMSpan(
                trace_id=format(otel_span.get_span_context().trace_id, "032x"),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency,
                cost_usd=cost,
                hallucination_score=hallucination_score,
                metadata=metadata or {},
            )

            alerts = self.anomaly_detector.check(span)
            for alert in alerts:
                logger.warning(
                    f"Anomaly detected: {alert.metric}={alert.current_value:.4f} "
                    f"(threshold={alert.threshold:.4f}, severity={alert.severity})"
                )
                otel_span.add_event(
                    "anomaly_detected",
                    attributes={"metric": alert.metric, "severity": alert.severity},
                )

            self._spans.append(span)
            return span

    def get_cost_report(self, last_n: int = 100) -> dict:
        recent = self._spans[-last_n:]
        by_model = {}
        for s in recent:
            if s.model not in by_model:
                by_model[s.model] = {"count": 0, "total_cost": 0.0,
                                     "total_tokens": 0, "avg_latency": 0.0}
            entry = by_model[s.model]
            entry["count"] += 1
            entry["total_cost"] += s.cost_usd
            entry["total_tokens"] += s.prompt_tokens + s.completion_tokens
            entry["avg_latency"] = (
                (entry["avg_latency"] * (entry["count"] - 1) + s.latency_ms)
                / entry["count"]
            )
        return {
            "total_cost": sum(e["total_cost"] for e in by_model.values()),
            "total_requests": len(recent),
            "by_model": by_model,
        }


if __name__ == "__main__":
    collector = LLMTraceCollector("portfolio-demo")
    # Simulate traces
    for i in range(20):
        span = collector.trace_llm_call(
            model="claude-sonnet-4-6",
            prompt_tokens=500 + i * 10,
            completion_tokens=200 + i * 5,
            hallucination_score=0.05 + (0.01 * i),
        )
        print(f"Span {span.trace_id[:8]}: cost=${span.cost_usd:.4f}")

    report = collector.get_cost_report()
    print(f"\nCost report: ${report['total_cost']:.4f} across {report['total_requests']} requests")
