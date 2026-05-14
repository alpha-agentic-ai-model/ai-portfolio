"""AWS Strands Agent Orchestrator with Dynamic Task Routing

Multi-agent orchestration framework built on the Strands Agents SDK.
Dynamically routes tasks to specialized sub-agents based on intent classification.
"""

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands import tool
from typing import Any, Optional
from dataclasses import dataclass, field
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class AgentHealth:
    name: str
    total_requests: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def failure_rate(self) -> float:
        return self.failures / max(self.total_requests, 1)

    @property
    def is_healthy(self) -> bool:
        return self.failure_rate < 0.3 and (time.time() - self.last_heartbeat) < 60


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant context."""
    # In production, connect to your vector store
    results = vector_store.similarity_search(query, k=5)
    return "\n".join(r.page_content for r in results)

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information."""
    results = tavily_client.search(query, max_results=5)
    return "\n".join(f"- {r['title']}: {r['content'][:200]}" for r in results['results'])

@tool
def run_code(code: str, language: str = "python") -> str:
    """Execute code in a sandboxed environment."""
    result = sandbox.execute(code, language=language, timeout=30)
    return f"Output: {result.stdout}\nErrors: {result.stderr}"

@tool
def analyze_repo(repo_url: str) -> str:
    """Analyze a code repository structure and key files."""
    repo = git_client.clone(repo_url, depth=1)
    structure = repo.get_tree()
    return f"Repository structure:\n{structure}"

@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the analytics database."""
    results = db_client.execute_readonly(sql)
    return results.to_markdown()


class IntentClassifier:
    """Classifies user requests into agent categories."""

    INTENTS = ["research", "code", "data", "general"]

    def __init__(self, model):
        self.classifier = Agent(
            model=model,
            system_prompt=(
                "Classify the user request into exactly one category: "
                "research, code, data, or general. "
                "Respond with only the category name."
            ),
        )

    def classify(self, request: str) -> str:
        result = self.classifier(f"Classify: {request}")
        intent = result.message.strip().lower()
        return intent if intent in self.INTENTS else "general"


class StrandsOrchestrator:
    """Multi-agent orchestrator with dynamic routing and health monitoring."""

    def __init__(self, region: str = "us-east-1"):
        self.model = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6-v1:0",
            region_name=region,
        )
        self.intent_classifier = IntentClassifier(self.model)
        self.health: dict[str, AgentHealth] = {}

        # Initialize specialized agents
        self.agents = {
            "research": Agent(
                model=self.model,
                system_prompt=(
                    "You are a research specialist. Synthesize information "
                    "from multiple sources into clear, cited summaries."
                ),
                tools=[search_knowledge_base, web_search],
            ),
            "code": Agent(
                model=self.model,
                system_prompt=(
                    "You are a senior software engineer. Write clean, "
                    "tested code and explain your design decisions."
                ),
                tools=[run_code, analyze_repo],
            ),
            "data": Agent(
                model=self.model,
                system_prompt=(
                    "You are a data analyst. Query databases, analyze "
                    "results, and present insights with evidence."
                ),
                tools=[query_database, run_code],
            ),
            "general": Agent(
                model=self.model,
                system_prompt="You are a helpful AI assistant.",
                tools=[search_knowledge_base, web_search],
            ),
        }

        for name in self.agents:
            self.health[name] = AgentHealth(name=name)

    def _select_agent(self, intent: str) -> tuple[str, Agent]:
        primary = self.health.get(intent)
        if primary and primary.is_healthy:
            return intent, self.agents[intent]

        # Fallback to healthiest agent
        logger.warning(f"Agent '{intent}' unhealthy, finding fallback")
        healthy = sorted(
            [(n, h) for n, h in self.health.items() if h.is_healthy],
            key=lambda x: x[1].avg_latency_ms,
        )
        if healthy:
            name = healthy[0][0]
            return name, self.agents[name]
        return "general", self.agents["general"]

    def execute(self, request: str, max_retries: int = 2) -> str:
        intent = self.intent_classifier.classify(request)
        logger.info(f"Classified intent: {intent}")

        agent_name, agent = self._select_agent(intent)
        health = self.health[agent_name]

        for attempt in range(max_retries + 1):
            try:
                start = time.perf_counter()
                result = agent(request)
                latency = (time.perf_counter() - start) * 1000

                health.total_requests += 1
                health.avg_latency_ms = (
                    (health.avg_latency_ms * (health.total_requests - 1) + latency)
                    / health.total_requests
                )
                health.last_heartbeat = time.time()

                logger.info(
                    f"Agent '{agent_name}' responded in {latency:.0f}ms "
                    f"(attempt {attempt + 1})"
                )
                return result.message

            except Exception as e:
                health.failures += 1
                logger.error(f"Agent '{agent_name}' failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    agent_name, agent = self._select_agent("general")
                    health = self.health[agent_name]

        return "I encountered an error processing your request. Please try again."

    def get_health_report(self) -> dict:
        return {
            name: {
                "healthy": h.is_healthy,
                "requests": h.total_requests,
                "failure_rate": f"{h.failure_rate:.1%}",
                "avg_latency_ms": f"{h.avg_latency_ms:.0f}",
            }
            for name, h in self.health.items()
        }


if __name__ == "__main__":
    orchestrator = StrandsOrchestrator()
    response = orchestrator.execute("What are the latest trends in quantum computing?")
    print(response)
    print("\nHealth Report:", orchestrator.get_health_report())
