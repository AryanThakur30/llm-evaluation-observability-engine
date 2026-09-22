"""Unit tests for the eval engine.

Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.drift import ks_two_sample, psi, semantic_drift
from engine.harness import Suite, run_suite
from engine.judges import GroundednessJudge, LLMJudge, ContainsJudge, build_judge
from engine.pipeline import RAGPipeline
from engine.providers import MockLLM
from engine.storage import Storage
from engine.tracer import Tracer
from scripts.seed_data import DOCS

SUITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "suites", "supportbot_qa.yaml")


# -- statistics ----------------------------------------------------------------

def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert psi(x, x) < 0.01


def test_psi_shifted_distributions_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(4, 1, 500)
    assert psi(a, b) > 0.25


def test_ks_same_distribution_high_p():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
    _, p = ks_two_sample(a, b)
    assert p > 0.05


def test_ks_different_distribution_low_p():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 300), rng.normal(3, 1, 300)
    _, p = ks_two_sample(a, b)
    assert p < 0.001


def test_semantic_drift_flags_topic_change():
    ref = ["refund policy covers 30 days for unused items",
           "standard shipping takes three to five business days",
           "password reset uses an email link valid thirty minutes"]
    cur_same = ["refunds apply within thirty days for unused items",
                "shipping takes five business days standard"]
    cur_off = ["football match postponed due to rain",
               "stock market volatility increased today"]
    s_same = semantic_drift(ref, cur_same)
    s_off = semantic_drift(ref, cur_off)
    assert s_same["mean_similarity"] > s_off["mean_similarity"]


# -- tracer --------------------------------------------------------------------

def test_tracer_nesting_and_ids():
    tr = Tracer()
    with tr.span("root") as root:
        with tr.span("child") as child:
            with tr.span("grand") as grand:
                pass
    assert root.trace_id == child.trace_id == grand.trace_id
    assert child.parent_id == root.span_id
    assert grand.parent_id == child.span_id
    assert len(tr.finished) == 3
    assert all(s.end_ns >= s.start_ns for s in tr.finished)


# -- providers -----------------------------------------------------------------

def test_mock_llm_is_deterministic():
    m1, m2 = MockLLM(), MockLLM()
    g1 = m1.generate("sys", "user CONTEXT:\nsome docs\n\nQUESTION:\nq")
    g2 = m2.generate("sys", "user CONTEXT:\nsome docs\n\nQUESTION:\nq")
    assert g1.text == g2.text


def test_mock_llm_grounded_mode_uses_context():
    m = MockLLM()
    ctx = "Refunds are available within 30 days of delivery for unused items."
    out = m.generate("sys", f"CONTEXT:\n{ctx}\n\nQUESTION:\nHow do refunds work?")
    assert "30 days" in out.text


# -- judges --------------------------------------------------------------------

def test_contains_judge():
    j = ContainsJudge(["refund", "30 days"])
    v_ok = j.evaluate({"response": "You can get a refund within 30 days."})
    v_no = j.evaluate({"response": "Contact support for details."})
    assert v_ok.passed and not v_no.passed


def test_llm_judge_parses_and_penalizes_evasive():
    judge = LLMJudge(MockLLM(), "accurate answer", min_score=4)
    good = judge.evaluate({"query": "q", "response": "Based on the documentation, refunds are available within 30 days of delivery for unused items in original packaging."})
    bad = judge.evaluate({"query": "q", "response": "I'm sorry, I don't have that information right now. Please contact support."})
    assert bad.score < good.score
    assert "reasoning" in bad.detail


def test_groundedness_scores_grounding():
    pipe = RAGPipeline(DOCS, MockLLM())
    res = pipe.answer("How do I get a refund for my order?")
    g = GroundednessJudge(pipe.retriever)
    v = g.evaluate(res)
    assert v.score >= 0.7
    # A response unrelated to retrieved context must score low.
    res2 = dict(res, response="The capital of France is Paris and the Eiffel Tower is lovely.")
    assert g.evaluate(res2).score < 0.5


# -- end to end ------------------------------------------------------------------

@pytest.fixture()
def tmp_storage(tmp_path):
    return Storage(str(tmp_path / "test.duckdb"))


def test_suite_baseline_then_regression(tmp_storage):
    suite = Suite.from_yaml(SUITE_PATH)
    good = run_suite(suite, RAGPipeline(DOCS, MockLLM("mock/v1")), storage=tmp_storage, label="base")
    assert good["suite_passed"], [r for r in good["results"] if not r["passed"]]

    bad = run_suite(suite, RAGPipeline(DOCS, MockLLM("mock/v2", degraded=True)), storage=tmp_storage, label="bad")
    assert not bad["suite_passed"]
    assert len(bad["regressions"]) > 0
    assert bad["pass_rate"] < good["pass_rate"]


def test_build_judge_factory():
    p, r = MockLLM(), RAGPipeline(DOCS, MockLLM())
    assert isinstance(build_judge({"type": "contains", "args": {"keywords": ["x"]}}, p, r), ContainsJudge)
    assert isinstance(build_judge({"type": "llm_judge", "args": {"criterion": "c"}}, p, r), LLMJudge)
    with pytest.raises(ValueError):
        build_judge({"type": "nope"}, p, r)
