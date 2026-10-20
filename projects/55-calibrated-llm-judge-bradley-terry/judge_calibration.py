"""Calibrated LLM-as-judge: pairwise comparisons with position swapping to
cancel order bias, Bradley-Terry fitting to turn wins into a global ranking,
and agreement measured against a small human-labelled anchor set."""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass

import anthropic
import numpy as np
from scipy.stats import kendalltau

JUDGE_PROMPT = """You are comparing two responses to the same user prompt.
<prompt>{prompt}</prompt>
<response_a>{a}</response_a>
<response_b>{b}</response_b>
Judge on correctness first, then helpfulness, then concision. Ignore length.
Return JSON only: {{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


@dataclass
class Verdict:
    i: int
    j: int
    score: float          # 1.0 = i wins, 0.0 = j wins, 0.5 = tie or inconsistent
    consistent: bool


class PairwiseJudge:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        self.client = anthropic.Anthropic()
        self.model = model

    def _ask(self, prompt: str, a: str, b: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=150, temperature=0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)}],
        )
        try:
            return json.loads(resp.content[0].text)["winner"]
        except (json.JSONDecodeError, KeyError):
            return "tie"

    def compare(self, prompt: str, outputs: list[str], i: int, j: int) -> Verdict:
        # Ask twice with swapped positions; only agreement counts as a win.
        first = self._ask(prompt, outputs[i], outputs[j])
        second = self._ask(prompt, outputs[j], outputs[i])
        to_score = {"A": 1.0, "B": 0.0, "tie": 0.5}
        s1, s2 = to_score.get(first, 0.5), 1.0 - to_score.get(second, 0.5)
        consistent = s1 == s2
        return Verdict(i, j, s1 if consistent else 0.5, consistent)


def bradley_terry(n: int, verdicts: list[Verdict], iters: int = 200, eps: float = 1e-9) -> np.ndarray:
    """MM algorithm (Hunter 2004). Ties count as half a win for each side."""
    wins = np.zeros((n, n))
    for v in verdicts:
        wins[v.i, v.j] += v.score
        wins[v.j, v.i] += 1.0 - v.score
    games = wins + wins.T
    p = np.ones(n)
    for _ in range(iters):
        denom = np.array([
            sum(games[i, j] / (p[i] + p[j]) for j in range(n) if j != i) for i in range(n)
        ])
        new_p = (wins.sum(axis=1) + eps) / (denom + eps)
        new_p /= new_p.sum()
        if np.abs(new_p - p).max() < 1e-8:
            break
        p = new_p
    return np.log(p)


def rank_systems(judge: PairwiseJudge, prompts: list[str], outputs: dict[str, list[str]],
                 sample_pairs: int | None = None) -> tuple[list[tuple[str, float]], float]:
    names = list(outputs)
    verdicts: list[Verdict] = []
    pairs = list(itertools.combinations(range(len(names)), 2))
    for p_idx, prompt in enumerate(prompts):
        chosen = random.sample(pairs, sample_pairs) if sample_pairs else pairs
        answers = [outputs[n][p_idx] for n in names]
        verdicts += [judge.compare(prompt, answers, i, j) for i, j in chosen]
    strengths = bradley_terry(len(names), verdicts)
    consistency = float(np.mean([v.consistent for v in verdicts]))
    ranking = sorted(zip(names, strengths.tolist()), key=lambda x: -x[1])
    return ranking, consistency


def human_agreement(judge_rank: list[str], human_rank: list[str]) -> float:
    pos = {n: k for k, n in enumerate(human_rank)}
    tau, _ = kendalltau(range(len(judge_rank)), [pos[n] for n in judge_rank])
    return float(tau)


if __name__ == "__main__":
    data = json.load(open("data/eval_outputs.json"))   # {"prompts": [...], "outputs": {system: [...]}}
    ranking, consistency = rank_systems(PairwiseJudge(), data["prompts"], data["outputs"])
    for name, s in ranking:
        print(f"{name:<24} {s:+.3f}")
    print(f"position-swap consistency: {consistency:.1%}")
    if "human_rank" in data:
        print(f"Kendall tau vs humans: {human_agreement([n for n, _ in ranking], data['human_rank']):.2f}")
