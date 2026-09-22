"""LLM Evaluation & Observability Engine.

A production-grade, self-contained platform for evaluating and monitoring
LLM applications:
- versioned evaluation harness with regression detection (CI-ready)
- deterministic checks + LLM-as-judge + groundedness/hallucination scoring
- statistical drift detection over live traffic (PSI, KS, semantic)
- cost/latency/token instrumentation with OpenTelemetry-style span tracing
- canary comparison between model versions
- FastAPI dashboard over a DuckDB analytics store

Works fully offline via a deterministic MockLLM; plugs into Ollama or any
OpenAI-compatible API for real models.
"""

from .canary import compare_versions
from .drift import analyze_traffic, psi, ks_two_sample, semantic_drift
from .harness import Suite, run_suite
from .judges import build_judge, LLMJudge, GroundednessJudge
from .pipeline import RAGPipeline
from .providers import MockLLM, OllamaProvider, OpenAICompatProvider
from .storage import Storage
from .tracer import Tracer

__all__ = [
    "RAGPipeline", "Storage", "Tracer", "MockLLM", "OllamaProvider",
    "OpenAICompatProvider", "Suite", "run_suite", "compare_versions",
    "analyze_traffic", "psi", "ks_two_sample", "semantic_drift",
    "build_judge", "LLMJudge", "GroundednessJudge",
]
