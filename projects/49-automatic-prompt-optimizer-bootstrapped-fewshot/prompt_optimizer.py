"""Automatic prompt optimizer with bootstrapped few-shot search.

Treats a prompt as a program: an instruction plus a set of demonstrations.
Candidate demos are bootstrapped from the model's own correct traces on the
training split, instructions are proposed by a meta-prompt, and a successive-
halving search picks the best (instruction, demo-set) pair on a dev split.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import anthropic

Example = dict[str, str]  # {"input": ..., "output": ...}


@dataclass(frozen=True)
class Candidate:
    instruction: str
    demos: tuple[int, ...]


class PromptProgram:
    def __init__(self, model: str = "claude-haiku-4-5"):
        self.client = anthropic.Anthropic()
        self.model = model

    def render(self, cand: Candidate, pool: list[Example], x: str) -> str:
        shots = "\n\n".join(f"Input: {pool[i]['input']}\nOutput: {pool[i]['output']}"
                            for i in cand.demos)
        return f"{cand.instruction}\n\n{shots}\n\nInput: {x}\nOutput:"

    def __call__(self, cand: Candidate, pool: list[Example], x: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=256, temperature=0,
            messages=[{"role": "user", "content": self.render(cand, pool, x)}])
        return msg.content[0].text.strip()


class PromptOptimizer:
    def __init__(self, program: PromptProgram, metric: Callable[[str, str], float],
                 seed: int = 7):
        self.program = program
        self.metric = metric
        self.rng = random.Random(seed)
        self.meta = anthropic.Anthropic()

    def bootstrap_demos(self, train: list[Example], seed_instruction: str,
                        max_demos: int = 24) -> list[Example]:
        """Keep only training examples the zero-shot program already gets right."""
        base = Candidate(seed_instruction, ())
        kept = []
        for ex in self.rng.sample(train, min(len(train), max_demos * 3)):
            pred = self.program(base, [], ex["input"])
            if self.metric(pred, ex["output"]) >= 1.0:
                kept.append({"input": ex["input"], "output": pred})
            if len(kept) >= max_demos:
                break
        return kept

    def propose_instructions(self, task_desc: str, demos: list[Example], n: int = 6) -> list[str]:
        sample = "\n".join(f"- {d['input']} -> {d['output']}" for d in demos[:8])
        msg = self.meta.messages.create(
            model="claude-sonnet-4-5", max_tokens=1200, temperature=1.0,
            messages=[{"role": "user", "content": (
                f"Task: {task_desc}\nExamples:\n{sample}\n\n"
                f"Write {n} distinct, concise system instructions for this task. "
                "Vary strategy (step-by-step, rules-first, format-first). One per line.")}])
        lines = [l.strip("-0123456789. ").strip() for l in msg.content[0].text.splitlines()]
        return [l for l in lines if len(l) > 20][:n]

    def score(self, cand: Candidate, pool: list[Example], data: list[Example]) -> float:
        preds = [self.program(cand, pool, ex["input"]) for ex in data]
        return sum(self.metric(p, ex["output"]) for p, ex in zip(preds, data)) / len(data)

    def compile(self, task_desc: str, seed_instruction: str, train: list[Example],
                dev: list[Example], k_shots: int = 4, n_candidates: int = 24) -> Candidate:
        pool = self.bootstrap_demos(train, seed_instruction)
        instructions = [seed_instruction] + self.propose_instructions(task_desc, pool)
        candidates = {
            Candidate(self.rng.choice(instructions),
                      tuple(sorted(self.rng.sample(range(len(pool)), min(k_shots, len(pool))))))
            for _ in range(n_candidates)
        }
        survivors, budget = list(candidates), 8
        # successive halving: evaluate on growing dev slices, keep the top half each round
        while len(survivors) > 1 and budget <= len(dev):
            batch = dev[:budget]
            ranked = sorted(survivors, key=lambda c: self.score(c, pool, batch), reverse=True)
            survivors = ranked[: max(1, len(ranked) // 2)]
            budget *= 2
        best = survivors[0]
        print(f"best dev score={self.score(best, pool, dev):.3f} demos={best.demos}")
        return best


def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


if __name__ == "__main__":
    data = [{"input": f"{a}+{b}", "output": str(a + b)} for a in range(20) for b in range(5)]
    random.shuffle(data)
    opt = PromptOptimizer(PromptProgram(), exact_match)
    best = opt.compile("Add two integers", "Return only the sum.", data[:60], data[60:])
    print(best.instruction)
