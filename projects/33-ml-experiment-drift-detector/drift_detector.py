"""ML Experiment Drift Detection Platform."""
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np
from scipy import stats


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftReport:
    feature: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    js_divergence: float
    is_drifted: bool
    severity: DriftSeverity
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelHealth:
    model_id: str
    drift_reports: list[DriftReport] = field(default_factory=list)
    overall_health: float = 1.0
    needs_retrain: bool = False
    last_checked: float = field(default_factory=time.time)


class StatisticalTests:
    """Collection of drift detection statistical tests."""

    @staticmethod
    def population_stability_index(
        baseline: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        bin_edges = np.histogram_bin_edges(baseline, bins=bins)
        b_hist = np.histogram(baseline, bins=bin_edges)[0]
        c_hist = np.histogram(current, bins=bin_edges)[0]

        b_pct = b_hist / max(len(baseline), 1)
        c_pct = c_hist / max(len(current), 1)

        b_pct = np.clip(b_pct, 1e-6, None)
        c_pct = np.clip(c_pct, 1e-6, None)

        return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))

    @staticmethod
    def kolmogorov_smirnov(
        baseline: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        stat, p_value = stats.ks_2samp(baseline, current)
        return float(stat), float(p_value)

    @staticmethod
    def jensen_shannon_divergence(
        baseline: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        bin_edges = np.histogram_bin_edges(
            np.concatenate([baseline, current]), bins=bins
        )
        p = np.histogram(baseline, bins=bin_edges)[0].astype(float)
        q = np.histogram(current, bins=bin_edges)[0].astype(float)

        p /= max(p.sum(), 1e-8)
        q /= max(q.sum(), 1e-8)

        m = 0.5 * (p + q)
        js = 0.5 * stats.entropy(p + 1e-8, m + 1e-8) + \
             0.5 * stats.entropy(q + 1e-8, m + 1e-8)
        return float(js)


class AdaptiveThreshold:
    """Dynamically adjust drift thresholds based on history."""

    def __init__(self, base_psi: float = 0.2, base_ks: float = 0.05,
                 window_size: int = 50):
        self.base_psi = base_psi
        self.base_ks = base_ks
        self.history: list[float] = []
        self.window_size = window_size

    def update(self, psi_value: float):
        self.history.append(psi_value)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    @property
    def psi_threshold(self) -> float:
        if len(self.history) < 10:
            return self.base_psi
        mean = np.mean(self.history)
        std = np.std(self.history)
        return max(self.base_psi, mean + 2 * std)

    @property
    def ks_threshold(self) -> float:
        return self.base_ks


class DriftDetector:
    """Main drift detection engine."""

    def __init__(self, psi_threshold: float = 0.2, ks_alpha: float = 0.05):
        self.tests = StatisticalTests()
        self.threshold = AdaptiveThreshold(base_psi=psi_threshold, base_ks=ks_alpha)

    def analyze_feature(self, feature: str,
                        baseline: np.ndarray,
                        current: np.ndarray) -> DriftReport:
        psi = self.tests.population_stability_index(baseline, current)
        ks_stat, ks_p = self.tests.kolmogorov_smirnov(baseline, current)
        js_div = self.tests.jensen_shannon_divergence(baseline, current)

        self.threshold.update(psi)
        is_drifted = psi > self.threshold.psi_threshold or ks_p < self.threshold.ks_threshold
        severity = self._classify_severity(psi)

        return DriftReport(
            feature=feature,
            psi_score=psi,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            js_divergence=js_div,
            is_drifted=is_drifted,
            severity=severity,
        )

    def _classify_severity(self, psi: float) -> DriftSeverity:
        if psi < 0.1:
            return DriftSeverity.NONE
        elif psi < 0.2:
            return DriftSeverity.LOW
        elif psi < 0.35:
            return DriftSeverity.MEDIUM
        elif psi < 0.5:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL


class RetrainTrigger:
    """Decide when to trigger model retraining."""

    def __init__(self, drift_threshold: int = 3, health_floor: float = 0.6):
        self.drift_threshold = drift_threshold
        self.health_floor = health_floor

    def evaluate(self, health: ModelHealth) -> bool:
        drifted_count = sum(1 for r in health.drift_reports if r.is_drifted)
        critical = any(
            r.severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)
            for r in health.drift_reports
        )
        return drifted_count >= self.drift_threshold or critical or \
               health.overall_health < self.health_floor


class DriftMonitor:
    """Orchestrate drift detection across all features."""

    def __init__(self):
        self.detector = DriftDetector()
        self.trigger = RetrainTrigger()

    def check_model(self, model_id: str,
                    baselines: dict[str, np.ndarray],
                    currents: dict[str, np.ndarray]) -> ModelHealth:
        reports = []
        for feature in baselines:
            if feature not in currents:
                continue
            report = self.detector.analyze_feature(
                feature, baselines[feature], currents[feature]
            )
            reports.append(report)

        health_score = 1.0 - (
            sum(1 for r in reports if r.is_drifted) / max(len(reports), 1)
        )
        model_health = ModelHealth(
            model_id=model_id,
            drift_reports=reports,
            overall_health=health_score,
        )
        model_health.needs_retrain = self.trigger.evaluate(model_health)
        return model_health


if __name__ == "__main__":
    np.random.seed(42)
    monitor = DriftMonitor()

    baselines = {
        "age": np.random.normal(35, 10, 1000),
        "income": np.random.lognormal(10, 1, 1000),
        "score": np.random.uniform(0, 1, 1000),
    }
    # Simulate drift in income feature
    currents = {
        "age": np.random.normal(36, 10, 500),
        "income": np.random.lognormal(11, 1.5, 500),  # shifted
        "score": np.random.uniform(0, 1, 500),
    }

    health = monitor.check_model("fraud-detector-v3", baselines, currents)
    print(f"Model: {health.model_id}")
    print(f"Health: {health.overall_health:.2f}")
    print(f"Needs retrain: {health.needs_retrain}")
    for r in health.drift_reports:
        flag = " ** DRIFT **" if r.is_drifted else ""
        print(f"  {r.feature}: PSI={r.psi_score:.4f} KS={r.ks_statistic:.4f}{flag}")
