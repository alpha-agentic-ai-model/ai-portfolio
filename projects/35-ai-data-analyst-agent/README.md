# AI Data Analyst Agent with Automated EDA

> **Category:** Agentic AI  
> **Project #35** in the AI Engineer Portfolio

## Overview

An autonomous data analysis agent that ingests CSV/Parquet datasets, performs automated exploratory data analysis, generates statistical summaries, detects outliers, creates visualizations, and produces narrative insights. Uses a plan-execute loop with tool calling for pandas operations and Matplotlib chart generation.

## Architecture

```
[Dataset Upload] → [Schema Detector] → [EDA Planner]
          ↓
[Stats Engine] → [Outlier Detector] → [Chart Generator]
          ↓
[Insight Narrator] → [Report Builder] → [Interactive Dashboard]
```

## Tech Stack

Claude API, Pandas, Matplotlib, FastAPI, Streamlit, SciPy

## Getting Started

```bash
# Clone the repository
git clone https://github.com/alpha-agentic-ai-model/ai-portfolio.git
cd ai-portfolio/projects/35-ai-data-analyst-agent

# Install dependencies
pip install -r requirements.txt

# Run the project
python data_analyst.py
```

## Author

**Manikanta Pudoka** — AI Engineer  
[GitHub](https://github.com/alpha-agentic-ai-model) | [LinkedIn](https://www.linkedin.com/in/pudoka-manikanta-3477a11b1/) | [Email](mailto:manikanta.pudoka.ai@gmail.com)
