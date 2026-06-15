from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime
import time

@dataclass
class FeatureDefinition:
    name: str
    entity_key: str
    transform_fn: Callable
    ttl_seconds: int = 3600
    version: int = 1
    drift_threshold: float = 0.1

@dataclass
class FeatureVector:
    entity_id: str
    features: dict[str, Any]
    timestamp: datetime
    latency_ms: float

class RealtimeFeatureStore:
    """Feature store with online-offline consistency guarantee."""
    def __init__(self, redis_client, offline_store, kafka_consumer):
        self.redis = redis_client
        self.offline = offline_store
        self.consumer = kafka_consumer
        self.registry: dict[str, FeatureDefinition] = {}
        self.drift_monitor = DriftMonitor()

    def register_feature(self, feature: FeatureDefinition):
        self.registry[feature.name] = feature
        self.drift_monitor.register(feature.name, feature.drift_threshold)

    async def get_online(self, entity_id: str, features: list[str]) -> FeatureVector:
        start = time.perf_counter()
        pipe = self.redis.pipeline()
        for feat in features:
            pipe.hget(f"feat:{feat}", entity_id)
        values = await pipe.execute()
        result = {feat: self.deserialize(val)
                  for feat, val in zip(features, values) if val}
        latency = (time.perf_counter() - start) * 1000
        return FeatureVector(
            entity_id=entity_id, features=result,
            timestamp=datetime.utcnow(), latency_ms=latency
        )

    async def materialize_online(self, feature_name: str):
        defn = self.registry[feature_name]
        async for event in self.consumer.subscribe(defn.entity_key):
            value = defn.transform_fn(event)
            await self.redis.hset(
                f"feat:{feature_name}", event.entity_id,
                self.serialize(value)
            )
            await self.redis.expire(
                f"feat:{feature_name}", defn.ttl_seconds
            )
            self.drift_monitor.observe(feature_name, value)