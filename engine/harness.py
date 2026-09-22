"""Versioned evaluation suites + regression detection.

A suite is a YAML file of cases; each case runs through the RAG pipeline and
is scored by its judges. Every run is compared against the last *passing*
baseline run of the same suite+version -- a judge that passed before and
fails now (or drops hard) is a regression. `run_regression.py` exits
non-zero on regression, so this doubles as a CI gate.
"""
from __future__ import annotations

import datetime

import yaml

from .judges import Verdict, build_judge
from .pipeline import RAGPipeline
from .storage import Storage

# A score drop of this much on an already-passing judge counts as regression.
SCORE_DROP_THRESHOLD = 0.15


class Case:
    def __init__(self, case_id: str, query: str, judge_specs: list[dict]):
        self.id = case_id
        self.query = query
        self.judge_specs = judge_specs


class Suite:
    def __init__(self, name: str, version: int, cases: list[Case], threshold: float = 1.0):
        self.name = name
        self.version = version
        self.cases = cases
        self.threshold = threshold  # required overall pass rate

    @classmethod
    def from_yaml(cls, path: str) -> "Suite":
        with open(path) as f:
            data = yaml.safe_load(f)
        cases = [
            Case(c["id"], c["query"], c.get("judges", []))
            for c in data.get("cases", [])
        ]
        return cls(data["suite"], int(data["version"]), cases, float(data.get("threshold", 1.0)))


def _judge_label(spec: dict) -> str:
    args = spec.get("args", {})
    if spec["type"] == "llm_judge":
        return "llm_judge"
    if spec["type"] == "contains":
        kws = args.get("keywords", [])
        kws = kws if isinstance(kws, list) else [kws]
        return "contains:" + ",".join(map(str, kws))[:40]
    return spec["type"]


def run_suite(suite: Suite, pipeline: RAGPipeline, storage: Storage | None = None,
              label: str = "") -> dict:
    """Execute every case, judge it, compare against baseline, persist."""
    started_at = datetime.datetime.now()
    results = []          # rows for persistence
    regressions = []      # summary of failures vs baseline
    baseline = storage.latest_passing_run(suite.name, suite.version) if storage else None
    baseline_scores = {}
    if baseline:
        for r in storage.run_results(baseline["id"]):
            baseline_scores[(r["case_id"], r["judge"])] = r

    for case in suite.cases:
        result = pipeline.answer(case.query)
        for spec in case.judge_specs:
            judge = build_judge(spec, pipeline.provider, pipeline.retriever)
            verdict: Verdict = judge.evaluate(result)
            jl = _judge_label(spec)
            reg = False
            prev = baseline_scores.get((case.id, jl))
            if prev is not None:
                if prev["passed"] and not verdict.passed:
                    reg = True
                elif verdict.score < prev["score"] - SCORE_DROP_THRESHOLD:
                    reg = True
            row = {
                "case_id": case.id, "judge": jl, "score": verdict.score,
                "passed": verdict.passed, "detail": verdict.detail, "regression": reg,
            }
            results.append(row)
            if reg:
                regressions.append({
                    "case": case.id, "judge": jl,
                    "baseline_score": prev["score"], "current_score": verdict.score,
                    "baseline_passed": prev["passed"], "current_passed": verdict.passed,
                })

    n_checks = len(results) or 1
    n_passed = sum(1 for r in results if r["passed"])
    pass_rate = n_passed / n_checks
    suite_passed = pass_rate >= suite.threshold and not regressions

    report = {
        "suite": suite.name,
        "suite_version": suite.version,
        "label": label,
        "model": pipeline.provider.model,
        "started_at": started_at,
        "cases": len(suite.cases),
        "checks": len(results),
        "checks_passed": n_passed,
        "pass_rate": round(pass_rate, 4),
        "suite_passed": suite_passed,
        "regressions": regressions,
        "baseline_run_id": baseline["id"] if baseline else None,
        "results": results,
    }

    if storage:
        run_id = storage.insert_eval_run(suite.name, suite.version, pipeline.provider.model, suite_passed, report)
        for r in results:
            storage.insert_eval_result(run_id, r["case_id"], r["judge"], r["score"], r["passed"],
                                        r["detail"], r["regression"])
        report["run_id"] = run_id
    return report
