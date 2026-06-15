import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Pheromone:
    task_id: str
    intensity: float
    deposited_by: str
    decay_rate: float = 0.95

@dataclass
class SwarmAgent:
    agent_id: str
    role: str = "generalist"
    energy: float = 1.0
    memory: list = field(default_factory=list)

class StigmergyMemory:
    """Shared memory layer for agent coordination via pheromones."""
    def __init__(self, redis_client):
        self.redis = redis_client
        self.pheromones: dict[str, list[Pheromone]] = {}

    async def deposit(self, pheromone: Pheromone):
        key = f"pheromone:{pheromone.task_id}"
        await self.redis.zadd(key, {pheromone.deposited_by: pheromone.intensity})

    async def sense(self, task_id: str, radius: float = 0.5) -> float:
        key = f"pheromone:{task_id}"
        signals = await self.redis.zrangebyscore(key, radius, "+inf")
        return sum(float(s) for s in signals)

class AgentSwarm:
    """Bio-inspired swarm with emergent collective intelligence."""
    def __init__(self, num_agents: int = 50, llm_client=None):
        self.agents = [SwarmAgent(f"agent-{i}") for i in range(num_agents)]
        self.stigmergy = StigmergyMemory(redis_client)
        self.llm = llm_client

    async def solve(self, problem: str, max_rounds: int = 10):
        sub_tasks = await self.decompose(problem)
        for round_num in range(max_rounds):
            tasks = await asyncio.gather(*[
                self.agent_step(agent, sub_tasks)
                for agent in self.agents
            ])
            consensus = await self.check_consensus(tasks)
            if consensus.confidence > 0.85:
                return consensus.solution
            await self.decay_pheromones()
        return self.best_effort_solution()