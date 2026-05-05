# AI Customer Support Crew with CrewAI

Multi-agent customer support system using CrewAI with specialized agents for ticket classification, sentiment analysis, response drafting, and escalation routing. Integrates Zendesk and Slack.

## Architecture
```
[Incoming Ticket] -> [Classifier Agent] -> [Sentiment Agent]
                                                |
[Response Drafter] -> [QA Agent] -> [Human Approval / Auto-send]
```

## Tech Stack
- CrewAI, Claude API, Zendesk API, FastAPI, Redis
