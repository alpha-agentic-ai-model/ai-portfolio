from evidently.metrics import DataDriftPreset, DataQualityPreset
from evidently import Report
from prometheus_client import Gauge, start_http_server
import mlflow
import logging

logger = logging.getLogger(__name__)

drift_gauge = Gauge("model_drift_score", "Current drift score", ["model_name"])
quality_gauge = Gauge("data_quality_score", "Data quality score", ["model_name"])


class DriftMonitor:
    def __init__(self, reference_data, model_name: str, threshold: float = 0.15):
        self.reference = reference_data
        self.model_name = model_name
        self.threshold = threshold
        self.consecutive_drifts = 0

    def check_drift(self, current_batch) -> dict:
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=self.reference, current_data=current_batch)

        result = report.as_dict()
        drift_score = result["metrics"][0]["result"]["share_of_drifted_columns"]
        quality_score = result["metrics"][1]["result"]["current"]["share_of_missing_values"]

        drift_gauge.labels(model_name=self.model_name).set(drift_score)
        quality_gauge.labels(model_name=self.model_name).set(1 - quality_score)

        if drift_score > self.threshold:
            self.consecutive_drifts += 1
            logger.warning(
                f"Drift detected: {drift_score:.3f} > {self.threshold} "
                f"(consecutive: {self.consecutive_drifts})"
            )
            if self.consecutive_drifts >= 3:
                self.trigger_retraining(current_batch)
                self.alert_team(drift_score)
        else:
            self.consecutive_drifts = 0

        return {
            "drift_score": drift_score,
            "quality_score": 1 - quality_score,
            "drifted": drift_score > self.threshold,
            "retraining_triggered": self.consecutive_drifts >= 3,
        }

    def trigger_retraining(self, new_data):
        with mlflow.start_run(run_name=f"auto-retrain-{self.model_name}"):
            mlflow.log_param("trigger", "drift_detected")
            mlflow.log_param("model_name", self.model_name)
            logger.info(f"Retraining triggered for {self.model_name}")

    def alert_team(self, drift_score: float):
        logger.critical(
            f"ALERT: Model {self.model_name} drift score {drift_score:.3f} "
            f"exceeded threshold for 3 consecutive batches"
        )


class CanaryDeployer:
    def __init__(self, k8s_client, traffic_split: float = 0.1):
        self.k8s = k8s_client
        self.traffic_split = traffic_split

    async def deploy_canary(self, model_name: str, new_version: str):
        await self.k8s.create_canary(
            service=model_name,
            version=new_version,
            weight=int(self.traffic_split * 100),
        )
        logger.info(
            f"Canary deployed: {model_name}@{new_version} "
            f"with {self.traffic_split*100}% traffic"
        )

    async def promote_or_rollback(self, model_name: str, metrics: dict):
        if metrics["error_rate"] < 0.01 and metrics["latency_p99"] < 200:
            await self.k8s.promote_canary(model_name)
            logger.info(f"Canary promoted for {model_name}")
        else:
            await self.k8s.rollback_canary(model_name)
            logger.warning(f"Canary rolled back for {model_name}")
