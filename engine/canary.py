"""Canary / model-version comparison.

Runs the same suite through two pipelines (model A vs model B) and produces
a paired comparison: same queries, same judges, so per-case scores pair up.
Includes a paired t-test on the per-case total score (no scipy needed).
"""
from __future__ import annotations

import datetime
import math

from .harness import Suite, _judge_label
from .judges import build_judge
from .pipeline import RAGPipeline


def _case_total_score(case, pipeline) -> tuple[float, int, int]:
    """Returns (total_score, n_judges, n_passed) for one case on one pipeline."""
    result = pipeline.answer(case.query)
    total, passed = 0.0, 0
    for spec in case.judge_specs:
        judge = build_judge(spec, pipeline.provider, pipeline.retriever)
        v = judge.evaluate(result)
        total += v.score
        passed += 1 if v.passed else 0
    return total, len(case.judge_specs), passed


def _paired_t(diffs: list[float]) -> dict:
    n = len(diffs)
    if n < 2:
        return {"t": None, "p": None, "mean_delta": diffs[0] if diffs else None}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return {"t": 0.0, "p": 1.0, "mean_delta": mean}
    t = mean / math.sqrt(var / n)
    # two-sided p via normal approximation (fine for n >= 5 with a caveat)
    z = abs(t)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return {"t": round(t, 3), "p": round(p, 4), "mean_delta": round(mean, 4)}


def compare_versions(suite: Suite, pipeline_a: RAGPipeline, pipeline_b: RAGPipeline,
                    label_a: str = "baseline", label_b: str = "canary") -> dict:
    rows, diffs = [], []
    wins_a = wins_b = ties = 0
    for case in suite.cases:
        ta, ja, pa = _case_total_score(case, pipeline_a)
        tb, jb, pb = _case_total_score(case, pipeline_b)
        norm_a, norm_b = ta / max(1, ja), tb / max(1, jb)
        diffs.append(norm_b - norm_a)
        if norm_b > norm_a:
            wins_b += 1
        elif norm_a > norm_b:
            wins_a += 1
        else:
            ties += 1
        rows.append({
            "case": case.id, "n_judges": ja,
            "score_a": round(norm_a, 3), "score_b": round(norm_b, 3),
            "passed_a": f"{pa}/{ja}", "passed_b": f"{pb}/{jb}",
            "delta": round(norm_b - norm_a, 3),
        })

    stats = _paired_t(diffs)
    import datetime

    return {
        "created_at": datetime.datetime.now(),
        "suite": suite.name,
        "label_a": label_a, "label_b": label_b,
        "model_a": pipeline_a.provider.model,
        "model_b": pipeline_b.provider.model,
        "cases": rows,
        "wins": {label_a: wins_a, label_b: wins_b, "tie": ties},
        "paired_t": stats,
        "verdict": "BETTER" if stats.get("mean_delta", 0) > 0.05 else
                   ("WORSE" if stats.get("mean_delta", 0) < -0.05 else "NO_SIGNIFICANT_DIFFERENCE"),
        "promote": stats.get("mean_delta", 0) > 0.05 and stats.get("p", 1) < 0.05,
    }
