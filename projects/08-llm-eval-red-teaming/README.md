# LLM Evaluation & Red-Teaming Framework

Comprehensive evaluation framework for LLMs benchmarking accuracy, hallucination rates, latency, and safety. Includes automated red-teaming with adversarial prompt generation.

## Architecture
```
[Test Suite] -> [Multi-Model Runner] -> [Eval Pipeline]
                                              |
[Red Team Generator] -> [Safety Scorer] -> [Dashboard + Reports]
```

## Tech Stack
- DeepEval, LangSmith, Pydantic, Pytest, Streamlit, OpenAI / Claude
