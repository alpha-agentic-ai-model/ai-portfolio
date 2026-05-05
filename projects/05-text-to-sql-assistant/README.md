# Text-to-SQL AI Assistant with Schema-Aware Prompting

A natural language to SQL converter with database schema injection, chain-of-thought reasoning, sandbox validation, and 94% accuracy on Spider benchmark.

## Architecture
```
[NL Question] -> [Schema Injector] -> [SQL Generator]
                                            |
[Sandbox Validator] -> [Error Corrector] -> [Results + Explanation]
```

## Tech Stack
- Claude API, SQLAlchemy, Streamlit, PostgreSQL, Pydantic
