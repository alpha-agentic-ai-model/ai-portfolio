"""
GPU Cluster Orchestrator with Spot Instance Optimization
Manages distributed training across mixed spot/on-demand GPU instances.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class InstanceType(str, Enum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"
    RESERVED = "reserved"


class GPUType(str, Enum):
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100 = "h100"
    L4 = "l4"
    T4 = "t4"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPUNode:
    node_id: str
    gpu_type: GPUType
    instance_type: InstanceType
    region: str
    num_gpus: int
    hourly_cost: float
    available: bool = True
    preemption_probability: float = 0.0


@dataclass
class Checkpoint:
    checkpoint_id: str
    job_id: str
    path: str
    step: int
    timestamp: datetime
    size_mb: float
    includes_optimizer_state: bool = True


@dataclass
class TrainingJob:
    job_id: str
    name: str
    num_gpus_required: int
    gpu_type: GPUType
    max_budget: float
    train_fn: Optional[Callable] = None
    status: JobStatus = JobStatus.PENDING
    current_step: int = 0
    total_steps: int = 10000
    assigned_nodes: list[GPUNode] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    total_cost: float = 0.0
    preemption_count: int = 0
    started_at: Optional[datetime] = None


class SpotBidOptimizer:
    """Predicts spot prices and optimizes bidding strategy."""

    def __init__(self, max_price_multiplier: float = 0.7):
        self.price_history: dict[str, list[float]] = {}
        self.max_price_multiplier = max_price_multiplier

    def record_price(self, region: str, gpu_type: str, price: float):
        key = f"{region}:{gpu_type}"
        if key not in self.price_history:
            self.price_history[key] = []
        self.price_history[key].append(price)

    def predict_price(self, region: str, gpu_type: str) -> float:
        key = f"{region}:{gpu_type}"
        history = self.price_history.get(key, [])
        if not history:
            return 0.0
        weights = [0.5 ** i for i in range(len(history))]
        weights.reverse()
        total = sum(w * p for w, p in zip(weights, history))
        return total / sum(weights)

    def optimal_bid(self, region: str, gpu_type: str, on_demand_price: float) -> float:
        predicted = self.predict_price(region, gpu_type)
        if predicted == 0:
            return on_demand_price * self.max_price_multiplier
        return min(predicted * 1.15, on_demand_price * self.max_price_multiplier)

    def estimate_preemption_risk(self, bid_price: float, region: str, gpu_type: str) -> float:
        predicted = self.predict_price(region, gpu_type)
        if predicted == 0 or bid_price == 0:
            return 0.5
        ratio = bid_price / predicted
        if ratio > 1.5:
            return 0.05
        elif ratio > 1.2:
            return 0.15
        elif ratio > 1.0:
            return 0.30
        else:
            return 0.60


class CheckpointManager:
    """Manages training checkpoints with gradient state preservation."""

    def __init__(self, storage_path: str = "/checkpoints", max_kept: int = 3):
        self.storage_path = Path(storage_path)
        self.max_kept = max_kept
        self.checkpoints: dict[str, list[Checkpoint]] = {}

    async def save_checkpoint(
        self, job: TrainingJob, model_state: dict, optimizer_state: dict
    ) -> Checkpoint:
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid4()),
            job_id=job.job_id,
            path=str(self.storage_path / job.job_id / f"step_{job.current_step}"),
            step=job.current_step,
            timestamp=datetime.utcnow(),
            size_mb=self._estimate_size(model_state, optimizer_state),
            includes_optimizer_state=True,
        )
        if job.job_id not in self.checkpoints:
            self.checkpoints[job.job_id] = []
        self.checkpoints[job.job_id].append(checkpoint)
        self._prune_old_checkpoints(job.job_id)
        job.checkpoints.append(checkpoint)
        logger.info(f"Saved checkpoint at step {job.current_step} for {job.name}")
        return checkpoint

    def get_latest_checkpoint(self, job_id: str) -> Optional[Checkpoint]:
        checkpoints = self.checkpoints.get(job_id, [])
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda c: c.step)

    def _prune_old_checkpoints(self, job_id: str):
        checkpoints = self.checkpoints.get(job_id, [])
        if len(checkpoints) > self.max_kept:
            checkpoints.sort(key=lambda c: c.step)
            self.checkpoints[job_id] = checkpoints[-self.max_kept:]

    def _estimate_size(self, model_state: dict, optimizer_state: dict) -> float:
        return 500.0  # Placeholder MB estimate


class FaultRecovery:
    """Handles preemption and fault recovery for training jobs."""

    def __init__(self, checkpoint_manager: CheckpointManager, max_retries: int = 5):
        self.checkpoint_manager = checkpoint_manager
        self.max_retries = max_retries

    async def handle_preemption(self, job: TrainingJob, allocator: "GPUAllocator") -> bool:
        job.status = JobStatus.PREEMPTED
        job.preemption_count += 1
        logger.warning(f"Job {job.name} preempted (count: {job.preemption_count})")
        if job.preemption_count > self.max_retries:
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.name} exceeded max preemption retries")
            return False
        latest_ckpt = self.checkpoint_manager.get_latest_checkpoint(job.job_id)
        if latest_ckpt:
            job.current_step = latest_ckpt.step
            logger.info(f"Resuming {job.name} from step {latest_ckpt.step}")
        new_nodes = await allocator.allocate(
            num_gpus=job.num_gpus_required,
            gpu_type=job.gpu_type,
            prefer_on_demand=job.preemption_count >= 3,
        )
        if new_nodes:
            job.assigned_nodes = new_nodes
            job.status = JobStatus.RUNNING
            return True
        return False


class GPUAllocator:
    """Allocates GPUs across regions with spot/on-demand mix."""

    def __init__(self, nodes: list[GPUNode]):
        self.nodes = {n.node_id: n for n in nodes}
        self.spot_optimizer = SpotBidOptimizer()

    async def allocate(
        self,
        num_gpus: int,
        gpu_type: GPUType,
        prefer_on_demand: bool = False,
    ) -> list[GPUNode]:
        candidates = [
            n for n in self.nodes.values()
            if n.gpu_type == gpu_type and n.available
        ]
        if prefer_on_demand:
            candidates.sort(key=lambda n: (n.instance_type != InstanceType.ON_DEMAND, n.hourly_cost))
        else:
            candidates.sort(key=lambda n: n.hourly_cost)
        allocated = []
        gpus_allocated = 0
        for node in candidates:
            if gpus_allocated >= num_gpus:
                break
            node.available = False
            allocated.append(node)
            gpus_allocated += node.num_gpus
        if gpus_allocated < num_gpus:
            for node in allocated:
                node.available = True
            return []
        return allocated

    def release(self, nodes: list[GPUNode]):
        for node in nodes:
            if node.node_id in self.nodes:
                self.nodes[node.node_id].available = True


class GPUOrchestrator:
    """Main orchestrator for distributed GPU training jobs."""

    def __init__(self, nodes: list[GPUNode], checkpoint_path: str = "/checkpoints"):
        self.allocator = GPUAllocator(nodes)
        self.checkpoint_manager = CheckpointManager(checkpoint_path)
        self.fault_recovery = FaultRecovery(self.checkpoint_manager)
        self.active_jobs: dict[str, TrainingJob] = {}
        self.completed_jobs: list[TrainingJob] = []

    async def submit_job(self, job: TrainingJob) -> str:
        nodes = await self.allocator.allocate(
            num_gpus=job.num_gpus_required,
            gpu_type=job.gpu_type,
        )
        if not nodes:
            raise RuntimeError(f"Cannot allocate {job.num_gpus_required} GPUs")
        job.assigned_nodes = nodes
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        self.active_jobs[job.job_id] = job
        logger.info(
            f"Launched job {job.name} on {len(nodes)} nodes "
            f"({sum(n.num_gpus for n in nodes)} GPUs)"
        )
        return job.job_id

    async def monitor_jobs(self):
        for job_id, job in list(self.active_jobs.items()):
            if job.status == JobStatus.RUNNING:
                cost_per_hour = sum(n.hourly_cost for n in job.assigned_nodes)
                if job.total_cost >= job.max_budget:
                    logger.warning(f"Job {job.name} hit budget limit")
                    job.status = JobStatus.FAILED
                    self.allocator.release(job.assigned_nodes)
                    self.completed_jobs.append(job)
                    del self.active_jobs[job_id]

    def get_cluster_stats(self) -> dict:
        total_gpus = sum(n.num_gpus for n in self.allocator.nodes.values())
        used_gpus = sum(
            n.num_gpus for n in self.allocator.nodes.values() if not n.available
        )
        total_cost = sum(j.total_cost for j in self.active_jobs.values())
        return {
            "total_gpus": total_gpus,
            "used_gpus": used_gpus,
            "utilization": used_gpus / max(total_gpus, 1),
            "active_jobs": len(self.active_jobs),
            "total_spend": total_cost,
        }
