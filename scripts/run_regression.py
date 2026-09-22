"""CI entrypoint: run a suite, exit non-zero if it regresses.

Usage:
    python scripts/run_regression.py --suite suites/supportbot_qa.yaml \
        [--db data/eval.duckdb] [--provider mock|ollama|openai] [--model NAME]

Intended to be wired into GitHub Actions / pre-push hooks -- see README.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.harness import Suite, run_suite
from engine.pipeline import RAGPipeline
from engine.providers import MockLLM, OllamaProvider, OpenAICompatProvider
from engine.storage import Storage
from scripts.seed_data import DOCS


def make_provider(kind: str, model: str):
    if kind == "mock":
        return MockLLM(model=model or "mock/supportbot-v1")
    if kind == "ollama":
        return OllamaProvider(model=model or "llama3.2:3b")
    if kind == "openai":
        return OpenAICompatProvider(model=model or "gpt-4o-mini")
    raise SystemExit(f"unknown provider: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--db", default=os.path.join("data", "eval.duckdb"))
    ap.add_argument("--provider", default="mock", choices=["mock", "ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="ci")
    args = ap.parse_args()

    storage = Storage(args.db)
    suite = Suite.from_yaml(args.suite)
    provider = make_provider(args.provider, args.model)
    pipeline = RAGPipeline(DOCS, provider, storage=storage, log_traffic=False)

    report = run_suite(suite, pipeline, storage=storage, label=args.label)

    print(f"suite={report['suite']} v{report['suite_version']} model={report['model']}")
    print(f"checks passed: {report['checks_passed']}/{report['checks']} ({report['pass_rate']:.0%})")
    if report["regressions"]:
        print("\nREGRESSIONS DETECTED:")
        for r in report["regressions"]:
            print(f"  - {r['case']} / {r['judge']}: {r['baseline_score']:.2f} -> {r['current_score']:.2f}")
        sys.exit(1)
    print("no regressions vs baseline")
    sys.exit(0)


if __name__ == "__main__":
    main()
