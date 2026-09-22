"""RAG pipeline with automatic tracing, cost/latency accounting and traffic logging.

Every answer() call produces a span tree:

    rag.pipeline
      ├── retrieval   (docs, top score)
      └── generation  (model, tokens, cost, latency)

...exported to DuckDB, plus a row in the traffic table that drift detection
and the dashboard consume.
"""
from __future__ import annotations

import datetime

from .providers import BaseProvider
from .retriever import TfidfRetriever
from .storage import Storage
from .tracer import Tracer

SYSTEM_PROMPT = (
    "You are a helpful support assistant. Answer the user's question using "
    "only the documentation provided in CONTEXT. If the context does not "
    "contain the answer, say so."
)


class RAGPipeline:
    def __init__(
        self,
        docs: list[str],
        provider: BaseProvider,
        storage: Storage | None = None,
        tracer: Tracer | None = None,
        k: int = 3,
        log_traffic: bool = True,
    ):
        self.retriever = TfidfRetriever(docs)
        self.provider = provider
        self.k = k
        self.storage = storage
        self.log_traffic = log_traffic and storage is not None
        self.tracer = tracer or Tracer()
        if storage is not None:
            self.tracer.exporter = storage.insert_span

    def answer(self, query: str, ts=None, day: int | None = None) -> dict:
        with self.tracer.span("rag.pipeline", query=query[:200]) as span:
            with self.tracer.span("retrieval", k=self.k) as rsp:
                hits = self.retriever.retrieve(query, self.k)
                rsp.set("doc_indices", [h["index"] for h in hits])
                rsp.set("top_score", round(hits[0]["score"], 4) if hits else 0.0)

            context = "\n\n".join(h["text"] for h in hits)
            user = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

            with self.tracer.span("generation", model=self.provider.model) as gsp:
                gen = self.provider.generate(SYSTEM_PROMPT, user)
                gsp.set("prompt_tokens", gen.prompt_tokens)
                gsp.set("completion_tokens", gen.completion_tokens)
                gsp.set("cost_usd", round(gen.cost_usd, 6))
                gsp.set("latency_ms", round(gen.latency_ms, 2))

            span.set("completion_tokens", gen.completion_tokens)
            span.set("cost_usd", round(gen.cost_usd, 6))

            if self.log_traffic:
                self.storage.insert_traffic(
                    {
                        "trace_id": span.trace_id,
                        "ts": ts or datetime.datetime.now(),
                        "day": day,
                        "query": query,
                        "response": gen.text,
                        "model": gen.model,
                        "prompt_tokens": gen.prompt_tokens,
                        "completion_tokens": gen.completion_tokens,
                        "cost_usd": gen.cost_usd,
                        "latency_ms": gen.latency_ms,
                        "n_docs": len(hits),
                        "top_retrieval_score": hits[0]["score"] if hits else 0.0,
                    }
                )

            return {
                "query": query,
                "response": gen.text,
                "retrieved": hits,
                "generation": gen,
                "trace_id": span.trace_id,
            }
