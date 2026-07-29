# Multi-Modal Screen Agent with Visual Grounding

> **Category:** Agentic AI  
> **Project #40** in the AI Engineer Portfolio

## Overview

A vision-language agent that observes application screenshots, understands UI elements through visual grounding, plans multi-step actions, and executes them via keyboard/mouse automation. Uses Set-of-Mark prompting for precise element targeting and a reflexion loop for self-correction when actions fail to achieve the intended state.

## Architecture

```
[Screenshot] → [Vision Encoder] → [SoM Annotator]
          ↓
[Action Planner (LLM)] → [Grounding Verifier] → [Executor]
          ↓
[State Observer] → [Reflexion Loop] → [Goal Checker]
```

## Tech Stack

Claude API, PyAutoGUI, Florence-2, Pillow, asyncio, Pydantic

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/40-multimodal-screen-agent-visual-grounding

# Install dependencies
pip install -r requirements.txt

# Run the project
python screen_agent.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
