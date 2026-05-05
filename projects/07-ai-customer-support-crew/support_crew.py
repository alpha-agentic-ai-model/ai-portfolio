from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import redis


@tool('zendesk_reader')
def zendesk_reader(ticket_id: str) -> str:
    """Read ticket details from Zendesk."""
    # Zendesk API integration
    return get_ticket_details(ticket_id)


@tool('sentiment_tool')
def sentiment_tool(text: str) -> dict:
    """Analyze customer sentiment."""
    return analyze_sentiment(text)


@tool('kb_search')
def kb_search(query: str) -> list:
    """Search knowledge base for relevant articles."""
    return search_knowledge_base(query)


def build_support_crew():
    classifier = Agent(
        role='Ticket Classifier',
        goal='Classify tickets by urgency and category with 98% accuracy',
        backstory='Expert at understanding customer intent and urgency levels',
        llm='claude-sonnet-4-6',
        tools=[zendesk_reader, sentiment_tool],
    )

    responder = Agent(
        role='Response Drafter',
        goal='Draft empathetic, accurate responses that resolve issues',
        backstory='Senior support engineer with 10 years of experience',
        llm='claude-sonnet-4-6',
        tools=[kb_search],
    )

    qa_agent = Agent(
        role='QA Reviewer',
        goal='Ensure responses are accurate, empathetic, and complete',
        backstory='Quality assurance specialist for customer communications',
        llm='claude-sonnet-4-6',
    )

    classify_task = Task(
        description='Classify the incoming ticket: {ticket}',
        expected_output='JSON with category, urgency, sentiment',
        agent=classifier,
    )

    draft_task = Task(
        description='Draft a response based on classification and KB search',
        expected_output='Professional response email',
        agent=responder,
    )

    review_task = Task(
        description='Review the draft for accuracy and tone',
        expected_output='Approved response or revision notes',
        agent=qa_agent,
    )

    crew = Crew(
        agents=[classifier, responder, qa_agent],
        tasks=[classify_task, draft_task, review_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def handle_ticket(ticket_data: dict):
    crew = build_support_crew()
    result = crew.kickoff(inputs={'ticket': ticket_data})

    cache = redis.Redis()
    cache.setex(
        f"response:{ticket_data['id']}",
        3600,
        str(result),
    )
    return result


if __name__ == '__main__':
    sample_ticket = {
        'id': 'TKT-1234',
        'subject': 'Cannot access my account after password reset',
        'body': 'I reset my password but still cannot log in. This is urgent!',
        'customer_tier': 'enterprise',
    }
    result = handle_ticket(sample_ticket)
    print(result)
