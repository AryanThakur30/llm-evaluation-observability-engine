"""LLM provider abstraction.

Three implementations:
- MockLLM: deterministic, offline, free. Powers tests and the demo without
  any model download. Simulates both a grounded RAG model and a *degraded*
  one, so the eval engine has a real regression to catch.
- OllamaProvider: local open-weights models via http://localhost:11434.
- OpenAICompatProvider: any OpenAI-compatible endpoint (OpenAI, Groq,
  Together, vLLM serve, ...).

All providers return a `Generation` carrying token counts, latency and cost,
so instrumentation works uniformly.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

# USD per 1K tokens: (prompt, completion). Local models are free.
PRICING = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
}
FREE_PREFIXES = ("mock/", "ollama/")


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    cost_usd: float = 0.0


def approx_tokens(text: str) -> int:
    """Cheap token proxy (~4 chars/token). Swap for tiktoken in production."""
    return max(1, (len(text) + 3) // 4)


def price(model: str, pt: int, ct: int) -> float:
    if model.startswith(FREE_PREFIXES):
        return 0.0
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return pt / 1000.0 * pin + ct / 1000.0 * pout


class BaseProvider(ABC):
    model: str = "unknown"

    @abstractmethod
    def generate(self, system: str, user: str) -> Generation:
        """Generate a completion. Must record its own latency."""


class MockLLM(BaseProvider):
    """Deterministic offline model.

    - Grounded mode: answers using the CONTEXT block embedded in the prompt.
    - Degraded mode: ignores context, emits canned evasive text -- this is
      the "silent model regression" the platform exists to catch.
    - Judge mode: acts as LLM-as-judge when the system prompt carries the
      evaluator marker; penalises evasive, ungrounded responses.

    Same (system, user) input always yields the same output => reproducible
    evaluations, which real eval platforms need and most demos lack.
    """

    JUDGE_MARKER = "strict evaluator"

    EVASIVE = (
        "I'm sorry, I don't have that information right now. Please contact support.",
        "Great question! There are many factors to consider and it depends on your specific situation.",
        "That's an interesting topic. Generally speaking, policies may vary. Is there anything else I can help with?",
        "I'm not sure about the details, but our team is happy to assist you further via email.",
    )

    def __init__(self, model: str = "mock/supportbot-v1", degraded: bool = False):
        self.model = model
        self.degraded = degraded

    # -- internals ---------------------------------------------------------

    def _rng(self, system: str, user: str) -> random.Random:
        seed = hashlib.sha256((self.model + system + user).encode()).hexdigest()
        return random.Random(seed)

    def _grounded_answer(self, user: str) -> str:
        context = user.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0].strip()
        question = user.split("QUESTION:", 1)[-1].strip()
        sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
        qwords = {w for w in question.lower().split() if len(w) > 3}
        ranked = sorted(
            sentences,
            key=lambda s: -len({w for w in s.lower().split() if len(w) > 3} & qwords),
        )
        picked = ranked[:3]
        body = ". ".join(picked).rstrip(".") + "."
        return f"Based on the documentation: {body}"

    def _degraded_answer(self, system: str, user: str, rng: random.Random) -> str:
        return rng.choice(self.EVASIVE)

    def _judge(self, system: str, user: str, rng: random.Random) -> str:
        response = user.split("RESPONSE:", 1)[-1]
        lowered = response.lower()
        evasive = any(
            m in lowered for m in ("don't have", "not sure", "depends on your", "may vary")
        )
        grounded = "based on the documentation" in lowered or len(response.split()) > 15
        if evasive:
            score = rng.randint(1, 2)
            reason = "The response is evasive and does not address the question."
        elif not grounded:
            score = 2
            reason = "The response is too thin to be useful."
        else:
            score = rng.randint(4, 5)
            reason = "The response addresses the question clearly and uses the documentation."
        return json.dumps({"score": score, "reasoning": reason})

    # -- API ---------------------------------------------------------------

    def generate(self, system: str, user: str) -> Generation:
        rng = self._rng(system, user)
        if self.JUDGE_MARKER in system:
            text = self._judge(system, user, rng)
        elif not self.degraded and "CONTEXT:" in user:
            text = self._grounded_answer(user)
        else:
            text = self._degraded_answer(system, user, rng)
        latency = 180 + rng.random() * 420
        pt, ct = approx_tokens(system + user), approx_tokens(text)
        return Generation(text, pt, ct, latency, self.model, price(self.model, pt, ct))


class OllamaProvider(BaseProvider):
    """Local open-weights models. Free, offline, reproducible-ish."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434", temperature: float = 0.2):
        self.model = "ollama/" + model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def generate(self, system: str, user: str) -> Generation:
        body = json.dumps(
            {
                "model": self.model.split("/", 1)[1],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        latency = (time.perf_counter() - t0) * 1000
        pt = data.get("prompt_eval_count") or approx_tokens(system + user)
        ct = data.get("eval_count") or approx_tokens(data["message"]["content"])
        return Generation(data["message"]["content"], pt, ct, latency, self.model, price(self.model, pt, ct))


class OpenAICompatProvider(BaseProvider):
    """Any OpenAI-compatible API: OpenAI, Groq, Together, vLLM, ..."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None, temperature: float = 0.2):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(self, system: str, user: str) -> Generation:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        latency = (time.perf_counter() - t0) * 1000
        msg = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", approx_tokens(system + user))
        ct = usage.get("completion_tokens", approx_tokens(msg))
        return Generation(msg, pt, ct, latency, self.model, price(self.model, pt, ct))
