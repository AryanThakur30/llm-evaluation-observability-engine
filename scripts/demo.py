"""End-to-end demo: the whole platform in ~2 minutes, fully offline.

Story it tells (the classic silent-regression incident):
  1. Baseline    -- suite passes on the healthy model.
  2. Traffic     -- 14 days of simulated requests. Days 1-7 healthy,
                    days 8-14 the provider is "updated" to a degraded model
                    (nobody notices; latency & cost look normal).
  3. Regression  -- re-running the suite flags exactly what broke.
  4. Drift       -- statistical analysis of the traffic log detects and
                    localises the change to day 8.
  5. Canary      -- old vs new model, head-to-head with a paired t-test.
  6. Dashboard   -- everything above is queryable at http://localhost:8000
"""
from __future__ import annotations

import datetime
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.canary import compare_versions
from engine.drift import analyze_traffic
from engine.harness import Suite, run_suite
from engine.pipeline import RAGPipeline
from engine.providers import MockLLM
from engine.storage import Storage
from scripts.seed_data import DOCS

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo_eval.duckdb")
SUITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "suites", "supportbot_qa.yaml")

# Extra "organic" traffic queries so the traffic log isn't just the suite.
NOISE_QUERIES = [
    "can I pay with UPI?", "do you offer gift cards?", "how do I cancel my order?",
    "what payment methods do you accept?", "can I exchange for a larger size?",
    "is there a price match policy?", "bulk order discount?", "cash on delivery available?",
    "gift card validity?", "what is your exchange policy?",
]


def banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    # -- clean slate --------------------------------------------------------
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    storage = Storage(DB_PATH)
    suite = Suite.from_yaml(SUITE_PATH)
    rng = random.Random(42)

    good = MockLLM(model="mock/supportbot-v1")
    bad = MockLLM(model="mock/supportbot-v1.1-regressed", degraded=True)

    # 1. Baseline run -------------------------------------------------------
    banner("STEP 1 - Baseline evaluation suite (healthy model)")
    base_pipeline = RAGPipeline(DOCS, good, storage=storage, log_traffic=False)
    baseline = run_suite(suite, base_pipeline, storage=storage, label="baseline")
    print(f"  model={baseline['model']}")
    print(f"  checks passed: {baseline['checks_passed']}/{baseline['checks']} "
          f"({baseline['pass_rate']:.0%}) -> suite_passed={baseline['suite_passed']}")

    # 2. Simulated production traffic ---------------------------------------
    banner("STEP 2 - Replaying 14 days of simulated production traffic")
    prod_pipeline = RAGPipeline(DOCS, good, storage=storage, log_traffic=True)
    queries = [c.query for c in suite.cases] + NOISE_QUERIES
    t0 = datetime.datetime(2026, 9, 7, 10, 0, 0)
    for day in range(1, 15):  # days 1..14
        # On day 8 a silent "model update" ships. Traffic keeps flowing.
        current = prod_pipeline if day <= 7 else RAGPipeline(DOCS, bad, storage=storage, log_traffic=True)
        for i in range(6):  # 6 requests/day
            q = rng.choice(queries)
            ts = t0 + datetime.timedelta(days=day - 1, hours=2 * i, minutes=rng.randint(0, 55))
            current.answer(q, ts=ts, day=day)
    n = storage.con.execute("SELECT count(*) FROM traffic").fetchone()[0]
    print(f"  logged {n} requests across 14 days")
    print(f"  ... days 1-7: healthy model, days 8-14: DEGRADED model (silent update)")

    # 3. Regression check ----------------------------------------------------
    banner("STEP 3 - Regression suite on the updated model")
    bad_pipeline = RAGPipeline(DOCS, bad, storage=storage, log_traffic=False)
    run = run_suite(suite, bad_pipeline, storage=storage, label="post-update")
    print(f"  model={run['model']}")
    print(f"  checks passed: {run['checks_passed']}/{run['checks']} ({run['pass_rate']:.0%})")
    print(f"  regressions vs baseline: {len(run['regressions'])}")
    for r in run["regressions"]:
        print(f"   [REGRESSION] {r['case']:<22} {r['judge']:<24} "
              f"{r['baseline_score']:.2f} -> {r['current_score']:.2f} "
              f"(passed={r['baseline_passed']} -> {r['current_passed']})")

    # 4. Drift detection -----------------------------------------------------
    banner("STEP 4 - Drift detection over the traffic log (week 1 vs week 2)")
    drift = analyze_traffic(storage, reference_days=(1, 7), current_days=(8, 14))
    storage.insert_report("drift", drift)
    for metric, m in drift["metrics"].items():
        if metric == "semantic":
            print(f"  semantic: similarity={m['mean_similarity']} vocab_jaccard={m['vocab_jaccard']}")
        else:
            print(f"  {metric:<18} PSI={m['psi']:<8} KS_p={m['ks_p']:<10} "
                  f"mean {m['ref_mean']} -> {m['cur_mean']}  [{m['verdict']}]")
    print(f"  drift_detected={drift['drift_detected']}, alerts={len(drift['alerts'])}")
    for a in drift["alerts"]:
        print(f"   [ALERT] {a}")

    # 5. Canary comparison ---------------------------------------------------
    banner("STEP 5 - Canary comparison (old model vs new model, paired)")
    canary = compare_versions(suite, base_pipeline, bad_pipeline,
                              label_a="supportbot-v1", label_b="supportbot-v1.1-regressed")
    storage.insert_report("canary", canary)
    print(f"  {canary['label_a']} ({canary['model_a']}) vs {canary['label_b']} ({canary['model_b']})")
    print(f"  wins: {canary['wins']}")
    print(f"  paired t={canary['paired_t']['t']} p={canary['paired_t']['p']} "
          f"mean_delta={canary['paired_t']['mean_delta']}")
    print(f"  verdict: {canary['verdict']}  promote={canary['promote']}")

    # 6. Done ----------------------------------------------------------------
    banner("RESULT - the platform caught the silent regression end to end")
    print(f"  regression suite : {len(run['regressions'])} judge regressions flagged")
    print(f"  drift detection  : {len(drift['alerts'])} statistical alerts")
    print(f"  canary           : {canary['verdict']}, promote={canary['promote']}")
    print(f"\n  data written to : {DB_PATH}")
    print("  launch dashboard:  uvicorn api.app:app --port 8000")
    print("  then open       :  http://localhost:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
