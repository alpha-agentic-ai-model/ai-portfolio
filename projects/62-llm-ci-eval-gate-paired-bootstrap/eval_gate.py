"""CI evaluation gate for LLM changes.

Runs the candidate (new prompt, model or retriever config) and the baseline
over the same eval set, scores both per example, and uses a paired bootstrap
to decide whether the change is a statistically significant regression.
Designed to run in GitHub Actions and fail the build with a markdown report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from anthropic import Anthropic

client = Anthropic()


@dataclass
class Variant:
    name: str
    model: str
    system_prompt: str
    temperature: float = 0.0


@dataclass
class GateResult:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    passed: bool


def load_variant(path: str) -> Variant:
    cfg = yaml.safe_load(Path(path).read_text())
    return Variant(cfg["name"], cfg["model"], Path(cfg["system_prompt_file"]).read_text(), cfg.get("temperature", 0.0))


def generate(v: Variant, prompt: str) -> str:
    resp = client.messages.create(
        model=v.model, max_tokens=1024, temperature=v.temperature,
        system=v.system_prompt, messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def score(example: dict, output: str) -> dict[str, float]:
    """Cheap deterministic checks plus an optional rubric grade."""
    s = {"contains_expected": float(all(k.lower() in output.lower() for k in example.get("must_include", [])))}
    s["format_ok"] = float(not example.get("json") or _is_json(output))
    if rubric := example.get("rubric"):
        g = client.messages.create(
            model="claude-haiku-4-5", max_tokens=10,
            messages=[{"role": "user", "content": f"Rubric: {rubric}\nAnswer: {output}\nScore 1-5. Reply with the digit only."}],
        ).content[0].text.strip()[:1]
        s["rubric"] = (int(g) - 1) / 4 if g.isdigit() else 0.0
    return s


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def paired_bootstrap(base: np.ndarray, cand: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = cand - base
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_variant(v: Variant, dataset: list[dict], workers: int = 8) -> list[dict[str, float]]:
    with ThreadPoolExecutor(workers) as pool:
        outputs = list(pool.map(lambda ex: generate(v, ex["input"]), dataset))
    return [score(ex, out) for ex, out in zip(dataset, outputs)]


def gate(base_scores, cand_scores, tolerance: float) -> list[GateResult]:
    results = []
    for metric in base_scores[0]:
        b = np.array([s[metric] for s in base_scores])
        c = np.array([s[metric] for s in cand_scores])
        lo, hi = paired_bootstrap(b, c)
        # Fail only if we are confident the candidate is worse by more than the tolerance.
        passed = hi >= -tolerance
        results.append(GateResult(metric, b.mean(), c.mean(), c.mean() - b.mean(), lo, hi, passed))
    return results


def report(results: list[GateResult]) -> str:
    rows = ["| Metric | Baseline | Candidate | Δ | 95% CI | Gate |", "|---|---|---|---|---|---|"]
    for r in results:
        rows.append(f"| {r.metric} | {r.baseline_mean:.3f} | {r.candidate_mean:.3f} | {r.delta:+.3f} | "
                    f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {'✅' if r.passed else '❌'} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()
    dataset = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    base = run_variant(load_variant(args.baseline), dataset)
    cand = run_variant(load_variant(args.candidate), dataset)
    results = gate(base, cand, args.tolerance)
    md = report(results)
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary).write_text(md, encoding="utf-8")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
