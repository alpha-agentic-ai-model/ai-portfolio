from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents import researcher, analyst, writer, editor


class ResearchState(TypedDict):
    topic: str
    sources: list
    analysis: str
    draft: str
    final_report: str
    quality_score: float


def quality_gate(state: ResearchState) -> str:
    """Route based on quality score from editor."""
    if state["quality_score"] >= 0.85:
        return "pass"
    return "revise"


def build_research_workflow():
    workflow = StateGraph(ResearchState)

    workflow.add_node("research", researcher.run)
    workflow.add_node("analyze", analyst.run)
    workflow.add_node("write", writer.run)
    workflow.add_node("edit", editor.run)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "write")
    workflow.add_edge("write", "edit")
    workflow.add_conditional_edges(
        "edit",
        quality_gate,
        {"pass": END, "revise": "write"}
    )

    return workflow.compile()


async def generate_report(topic: str) -> str:
    app = build_research_workflow()
    result = await app.ainvoke({"topic": topic})
    return result["final_report"]


if __name__ == "__main__":
    import asyncio
    report = asyncio.run(generate_report("AI Agent Frameworks in 2026"))
    print(report)
