# LLM Evaluation & Observability Engine

A production-grade, self-contained platform for evaluating and monitoring LLM applications — versioned regression suites, LLM-as-judge, groundedness/hallucination scoring, statistical drift detection over live traffic, cost/latency/token tracing, and canary model comparison, all queryable through a live dashboard.

**Zero external services required.** Runs fully offline on a deterministic mock model; plugs into Ollama (local open-weights) or any OpenAI-compatible API with one flag.

```
┌────────────────────────────────────────────────────────────────────────┐
│                            TRAFFIC (production or replay)              │
└──────────────┬─────────────────────────────────────────────────────────┘
               │
      ┌────────▼─────────┐     spans (OTel-style: trace/span/parent ids)
      │  RAGPipeline     │──────────────────────────────┐
      │  retrieval → gen │                              │
      └────────┬─────────┘                              │
               │ rows: tokens, cost, latency,          │
               │       retrieval scores                 ▼
               │                        ┌───────────────────────────┐
     ┌─────────▼─────────┐              │  DuckDB (single file)     │
     │  EVALUATION       │  query/write │  spans · traffic ·         │
     │  ──────────────── │─────────────►│  eval_runs · eval_results ·│
     │  suite (YAML)     │              │  reports                   │
     │  judges:          │              └───────────┬───────────────┬─┘
     │   · deterministic │                          │               │
     │   · LLM-as-judge  │              ┌───────────▼───┐   ┌───────▼───────┐
     │   · groundedness  │              │ DRIFT DETECTION│   │  DASHBOARD    │
     │  regression vs    │              │ PSI · KS test  │   │  FastAPI +    │
     │  last baseline    │              │ semantic drift │   │  Chart.js     │
     └─────────┬─────────┘              └───────────────┘   └───────────────┘
               │
     ┌─────────▼─────────┐
     │  CANARY           │  model A vs B, paired per-case,
     │  COMPARISON       │  paired t-test on score deltas
     └───────────────────┘
```

## Why this exists

Almost every team shipping an LLM feature has no evaluation infrastructure. The model gets "updated," quality silently degrades, and nothing catches it because latency is fine and no errors are thrown. This project is the platform that catches exactly that — and the demo proves it end-to-end:

1. **Baseline** — suite passes 16/16 checks on the healthy model.
2. **Incident** — 14 days of simulated traffic; on day 8 a "model update" ships. Latency and cost look normal.
3. **Regression suite** — flags 13 judge regressions with case-level detail.
4. **Drift detection** — PSI + Kolmogorov–Smirnov tests over the traffic log detect the change and localise it; semantic drift shows responses moved 90% away from the reference corpus.
5. **Canary** — old vs new model, head-to-head: paired t = −8.9, verdict WORSE, promote = False.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Run the full end-to-end demo (offline, ~30s)
python scripts/demo.py

# 2. Run the test suite
pytest tests/ -v

# 3. Launch the dashboard
uvicorn api.app:app --port 8000     # then open http://localhost:8000
```

## Using real models

```bash
# Local open-weights via Ollama (free, offline)
ollama pull llama3.2:3b
python scripts/run_regression.py --suite suites/supportbot_qa.yaml --provider ollama --model llama3.2:3b

# Any OpenAI-compatible API (OpenAI, Groq, Together, vLLM serve, ...)
export OPENAI_API_KEY=sk-...
python scripts/run_regression.py --suite suites/supportbot_qa.yaml --provider openai --model gpt-4o-mini
```

Providers are pluggable (`engine/providers.py`); every generation is instrumented for tokens, cost, and latency uniformly.

## CI integration (the quality gate)

`run_regression.py` exits non-zero on regression, so the suite becomes a build gate:

```yaml
# .github/workflows/eval-gate.yml
name: llm-quality-gate
on: [push]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python scripts/run_regression.py --suite suites/supportbot_qa.yaml
      # store data/eval.duckdb as an artifact to persist baselines between runs
```

## Project layout

```
engine/
  providers.py    # MockLLM (deterministic, offline) · Ollama · OpenAI-compatible
  tracer.py       # OTel-concepts span tracer (trace/span/parent ids, DuckDB export)
  storage.py      # DuckDB store: spans, traffic, eval runs/results, reports
  retriever.py    # TF-IDF retrieval + sentence-level similarity (groundedness)
  pipeline.py     # RAG pipeline: retrieve → generate, auto-traced + logged
  judges.py       # deterministic checks · LLM-as-judge · groundedness scoring
  harness.py      # versioned YAML suites, regression detection vs baseline
  drift.py        # PSI · two-sample KS test · TF-IDF semantic drift
  canary.py       # paired model comparison with paired t-test
suites/           # eval suites (version these like code)
scripts/          # demo.py · run_regression.py (CI gate) · seed_data.py
api/app.py       # FastAPI JSON + dashboard server
dashboard/       # single-file Chart.js dashboard
tests/            # 13 tests: statistics, tracer, judges, end-to-end regression
```

## Concepts demonstrated (interview talking points)

- **Evaluation methodology** — offline suites vs online drift, deterministic checks vs LLM-as-judge, inter-check agreement, regression gating vs last passing baseline
- **Statistics, from scratch** — Population Stability Index (quantile binning), two-sample Kolmogorov–Smirnov with the asymptotic Kolmogorov distribution, paired t-test — no scipy, the math is explicit and auditable
- **Groundedness / faithfulness** — sentence-level claim support against retrieved context (the NLI-model upgrade path is documented in the code)
- **Observability design** — span trees with trace/span/parent IDs mirroring OpenTelemetry's data model, exported to a columnar store; the OTel-exporter swap is mechanical
- **Cost attribution** — per-model pricing tables, cost-per-token, cost anomaly surfaces in the same drift framework as quality metrics

## Roadmap (documented next steps)

- [ ] Real OTel exporter (`opentelemetry-sdk`) alongside the built-in tracer
- [ ] NLI-based groundedness (DeBERTa entailment) behind the same `Judge` interface
- [ ] Embedding-based retrieval (bge-small / nomic) behind `TfidfRetriever`'s interface
- [ ] Online evals: interleaved feedback from production thumbs-up/down
- [ ] Multi-tenant projects + auth on the API
- [ ] Alert routing (Slack/webhook) when `analyze_traffic` fires

## License

MIT — do whatever you want, attribution appreciated.
