"""SLO-driven autoscaler for LLM inference replicas.

CPU utilisation is a poor scaling signal for GPU-bound LLM servers. This
controller scrapes queue depth, time-to-first-token (TTFT) p95 and KV-cache
usage from vLLM's Prometheus metrics, then sizes the Deployment against a
TTFT SLO with hysteresis, cooldowns and a cost ceiling.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import requests
from kubernetes import client, config

PROM = "http://prometheus.monitoring:9090/api/v1/query"


@dataclass
class SLO:
    ttft_p95_s: float = 0.8
    max_queue_per_replica: int = 16
    kv_usage_high: float = 0.90


@dataclass
class Policy:
    min_replicas: int = 1
    max_replicas: int = 12
    scale_up_cooldown_s: int = 60
    scale_down_cooldown_s: int = 300
    scale_down_headroom: float = 0.5  # only shrink when well under SLO
    gpu_hour_cost: float = 2.5
    hourly_budget: float = 30.0


@dataclass
class Signals:
    replicas: int
    queue: float
    ttft_p95: float
    kv_usage: float


def prom(query: str) -> float:
    r = requests.get(PROM, params={"query": query}, timeout=5)
    r.raise_for_status()
    result = r.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else 0.0


class InferenceAutoscaler:
    def __init__(self, namespace: str, deployment: str, slo: SLO, policy: Policy):
        config.load_incluster_config()
        self.apps = client.AppsV1Api()
        self.ns, self.name = namespace, deployment
        self.slo, self.policy = slo, policy
        self.last_up = self.last_down = 0.0

    def read_signals(self) -> Signals:
        sel = f'model_name="{self.name}"'
        scale = self.apps.read_namespaced_deployment_scale(self.name, self.ns)
        return Signals(
            replicas=scale.status.replicas or 1,
            queue=prom(f"sum(vllm:num_requests_waiting{{{sel}}})"),
            ttft_p95=prom(
                f"histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket{{{sel}}}[2m])) by (le))"),
            kv_usage=prom(f"avg(vllm:gpu_cache_usage_perc{{{sel}}})"),
        )

    def desired(self, s: Signals) -> int:
        by_queue = math.ceil(s.queue / self.slo.max_queue_per_replica) if s.queue else 0
        by_ttft = math.ceil(s.replicas * s.ttft_p95 / self.slo.ttft_p95_s) if s.ttft_p95 else 0
        by_kv = s.replicas + 1 if s.kv_usage > self.slo.kv_usage_high else 0
        target = max(by_queue, by_ttft, by_kv, self.policy.min_replicas)
        budget_cap = int(self.policy.hourly_budget // self.policy.gpu_hour_cost)
        return min(target, self.policy.max_replicas, budget_cap)

    def decide(self, s: Signals, now: float) -> int:
        target = self.desired(s)
        if target > s.replicas and now - self.last_up >= self.policy.scale_up_cooldown_s:
            self.last_up = now
            return target
        under_slo = (s.ttft_p95 < self.slo.ttft_p95_s * self.policy.scale_down_headroom
                     and s.queue == 0)
        if (target < s.replicas and under_slo
                and now - self.last_down >= self.policy.scale_down_cooldown_s):
            self.last_down = now
            return s.replicas - 1  # shrink one step at a time to protect warm KV caches
        return s.replicas

    def apply(self, replicas: int) -> None:
        self.apps.patch_namespaced_deployment_scale(
            self.name, self.ns, {"spec": {"replicas": replicas}})

    def run(self, interval_s: int = 15) -> None:
        while True:
            try:
                s = self.read_signals()
                target = self.decide(s, time.time())
                if target != s.replicas:
                    print(f"scale {s.replicas}->{target} queue={s.queue:.0f} "
                          f"ttft_p95={s.ttft_p95:.2f}s kv={s.kv_usage:.0%}")
                    self.apply(target)
            except Exception as exc:  # keep the loop alive; alerting handles persistence
                print(f"autoscaler error: {exc}")
            time.sleep(interval_s)


if __name__ == "__main__":
    InferenceAutoscaler("llm", "llama-3-70b", SLO(), Policy()).run()
