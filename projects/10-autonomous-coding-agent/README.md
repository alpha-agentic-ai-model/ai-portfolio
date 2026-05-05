# Autonomous Coding Agent with Planning & Execution

AI coding agent inspired by SWE-bench that takes a GitHub issue, plans a solution, navigates the codebase, writes code with tests, and submits a pull request using a plan-act-observe loop.

## Architecture
```
[GitHub Issue] -> [Planner Agent] -> [Code Navigator]
                                          |
[Code Writer] -> [Test Runner] -> [PR Submitter]
```

## Tech Stack
- OpenAI Agents SDK, Tree-sitter, GitHub API, Docker, Python AST
