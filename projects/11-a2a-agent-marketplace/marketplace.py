"""
A2A Protocol Agent Marketplace & Discovery Service
Enables autonomous agent discovery, capability negotiation, and task delegation.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SkillCategory(str, Enum):
    NLP = "natural_language_processing"
    VISION = "computer_vision"
    DATA = "data_analysis"
    CODE = "code_generation"
    RESEARCH = "research"
    CREATIVE = "creative"


@dataclass
class Skill:
    name: str
    category: SkillCategory
    proficiency: float  # 0.0 - 1.0
    description: str


@dataclass
class AgentCard:
    agent_id: str
    name: str
    description: str
    skills: list[Skill]
    endpoint_url: str
    trust_score: float = 0.5
    total_tasks: int = 0
    successful_tasks: int = 0
    registered_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks


class TaskRequest(BaseModel):
    task_id: str = ""
    description: str
    required_skills: list[str]
    timeout_seconds: int = 300
    max_budget: float = 10.0
    priority: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        if not self.task_id:
            self.task_id = str(uuid4())


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    agent_id: str = ""
    duration_ms: float = 0


class TrustEngine:
    """Bayesian trust scoring for agent reputation."""

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.alpha = prior_alpha
        self.beta = prior_beta

    def calculate_score(self, successes: int, failures: int) -> float:
        alpha = self.alpha + successes
        beta = self.beta + failures
        return alpha / (alpha + beta)

    def rank(self, agents: list[AgentCard]) -> list[AgentCard]:
        scored = []
        for agent in agents:
            failures = agent.total_tasks - agent.successful_tasks
            score = self.calculate_score(agent.successful_tasks, failures)
            agent.trust_score = score
            scored.append(agent)
        return sorted(scored, key=lambda a: a.trust_score, reverse=True)

    def update(self, agent: AgentCard, result: TaskResponse):
        agent.total_tasks += 1
        if result.status == "success":
            agent.successful_tasks += 1
        failures = agent.total_tasks - agent.successful_tasks
        agent.trust_score = self.calculate_score(
            agent.successful_tasks, failures
        )


class SkillRouter:
    """Routes tasks to agents based on skill matching."""

    def match_score(self, required: list[str], agent: AgentCard) -> float:
        agent_skills = {s.name.lower() for s in agent.skills}
        required_set = {r.lower() for r in required}
        if not required_set:
            return 0.0
        overlap = agent_skills & required_set
        coverage = len(overlap) / len(required_set)
        proficiency_sum = sum(
            s.proficiency for s in agent.skills if s.name.lower() in overlap
        )
        avg_proficiency = proficiency_sum / max(len(overlap), 1)
        return coverage * 0.6 + avg_proficiency * 0.4


class AgentRegistry:
    """Central registry for A2A-compliant agents."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.agents: dict[str, AgentCard] = {}

    async def register(self, card: AgentCard) -> str:
        self.agents[card.agent_id] = card
        await self.redis_client.hset(
            "a2a:agents", card.agent_id, json.dumps(card.__dict__, default=str)
        )
        for skill in card.skills:
            await self.redis_client.sadd(
                f"a2a:skills:{skill.name.lower()}", card.agent_id
            )
        return card.agent_id

    async def search(
        self, skills: list[str], min_trust_score: float = 0.5
    ) -> list[AgentCard]:
        candidate_ids: set[str] = set()
        for skill in skills:
            ids = await self.redis_client.smembers(f"a2a:skills:{skill.lower()}")
            candidate_ids.update(id.decode() if isinstance(id, bytes) else id for id in ids)
        candidates = [
            self.agents[aid]
            for aid in candidate_ids
            if aid in self.agents and self.agents[aid].trust_score >= min_trust_score
        ]
        return candidates


class CollaborationSession:
    """Multi-turn collaboration session between agents."""

    def __init__(self, task: TaskRequest, agent: AgentCard):
        self.session_id = str(uuid4())
        self.task = task
        self.agent = agent
        self.messages: list[dict] = []
        self.status = "negotiating"

    async def negotiate(self) -> bool:
        self.messages.append({
            "role": "requester",
            "content": f"Task: {self.task.description}",
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.status = "accepted"
        return True

    async def execute(self, timeout: int = 300, callback=None) -> TaskResponse:
        self.status = "executing"
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)  # Simulate execution
        elapsed = (asyncio.get_event_loop().time() - start) * 1000
        self.status = "completed"
        return TaskResponse(
            task_id=self.task.task_id,
            status="success",
            result={"output": f"Completed by {self.agent.name}"},
            agent_id=self.agent.agent_id,
            duration_ms=elapsed,
        )


class AgentMarketplace:
    """A2A-compliant agent marketplace with discovery and delegation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.registry = AgentRegistry(redis_url)
        self.trust_scorer = TrustEngine()
        self.router = SkillRouter()
        self.active_sessions: dict[str, CollaborationSession] = {}

    async def register_agent(self, card: AgentCard) -> str:
        return await self.registry.register(card)

    async def discover_agents(
        self, required_skills: list[str], top_k: int = 5
    ) -> list[AgentCard]:
        candidates = await self.registry.search(
            skills=required_skills, min_trust_score=0.3
        )
        ranked = self.trust_scorer.rank(candidates)
        for agent in ranked:
            agent.trust_score *= (
                1 + self.router.match_score(required_skills, agent)
            ) / 2
        ranked.sort(key=lambda a: a.trust_score, reverse=True)
        return ranked[:top_k]

    async def delegate_task(self, task: TaskRequest) -> TaskResponse:
        agents = await self.discover_agents(task.required_skills)
        if not agents:
            raise HTTPException(status_code=404, detail="No suitable agents found")
        best_agent = agents[0]
        session = CollaborationSession(task, best_agent)
        self.active_sessions[session.session_id] = session
        accepted = await session.negotiate()
        if not accepted:
            raise HTTPException(status_code=409, detail="Agent declined task")
        result = await session.execute(timeout=task.timeout_seconds)
        self.trust_scorer.update(best_agent, result)
        del self.active_sessions[session.session_id]
        return result


# FastAPI application
app = FastAPI(title="A2A Agent Marketplace", version="1.0.0")
marketplace = AgentMarketplace()


@app.post("/agents/register")
async def register_agent(card_data: dict):
    skills = [
        Skill(
            name=s["name"],
            category=SkillCategory(s.get("category", "research")),
            proficiency=s.get("proficiency", 0.7),
            description=s.get("description", ""),
        )
        for s in card_data.get("skills", [])
    ]
    card = AgentCard(
        agent_id=str(uuid4()),
        name=card_data["name"],
        description=card_data.get("description", ""),
        skills=skills,
        endpoint_url=card_data["endpoint_url"],
    )
    agent_id = await marketplace.register_agent(card)
    return {"agent_id": agent_id, "status": "registered"}


@app.post("/tasks/delegate")
async def delegate_task(task: TaskRequest):
    result = await marketplace.delegate_task(task)
    return result


@app.get("/agents/discover")
async def discover_agents(skills: str, top_k: int = 5):
    skill_list = [s.strip() for s in skills.split(",")]
    agents = await marketplace.discover_agents(skill_list, top_k)
    return [{"id": a.agent_id, "name": a.name, "trust": a.trust_score} for a in agents]
