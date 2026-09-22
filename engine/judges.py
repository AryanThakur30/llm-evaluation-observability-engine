"""Judges: deterministic checks, LLM-as-judge, and groundedness scoring.

Every judge returns a Verdict(passed, score in [0,1], detail) so the harness,
canary comparison and dashboard can treat them uniformly.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .providers import BaseProvider
from .retriever import TfidfRetriever, split_sentences


@dataclass
class Verdict:
    passed: bool
    score: float          # normalised [0, 1]
    detail: dict


class Judge(ABC):
    name = "judge"

    @abstractmethod
    def evaluate(self, result: dict) -> Verdict: ...


# -- deterministic checks ----------------------------------------------------


class ContainsJudge(Judge):
    """Checks that required keywords appear in the response."""

    def __init__(self, keywords, mode: str = "all"):
        self.keywords = [keywords] if isinstance(keywords, str) else list(keywords)
        self.mode = mode

    def evaluate(self, result):
        lowered = result["response"].lower()
        present = [k for k in self.keywords if k.lower() in lowered]
        score = len(present) / len(self.keywords) if self.keywords else 1.0
        ok = (len(present) == len(self.keywords)) if self.mode == "all" else (len(present) > 0)
        return Verdict(ok, score, {"required": self.keywords, "found": present, "mode": self.mode})


class RegexJudge(Judge):
    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def evaluate(self, result):
        found = bool(self.pattern.search(result["response"]))
        return Verdict(found, 1.0 if found else 0.0, {"pattern": self.pattern.pattern, "found": found})


class LengthJudge(Judge):
    """Response length bounds, in words."""

    def __init__(self, min_words: int = 5, max_words: int = 400):
        self.min_words, self.max_words = min_words, max_words

    def evaluate(self, result):
        n = len(result["response"].split())
        ok = self.min_words <= n <= self.max_words
        return Verdict(ok, min(1.0, n / max(1, self.min_words)), {"words": n, "bounds": [self.min_words, self.max_words]})


class JsonValidJudge(Judge):
    """Response must parse as JSON (for structured-output pipelines)."""

    def evaluate(self, result):
        try:
            json.loads(result["response"])
            return Verdict(True, 1.0, {})
        except json.JSONDecodeError as e:
            return Verdict(False, 0.0, {"error": str(e)})


class RetrievalScoreJudge(Judge):
    """The retriever itself must clear a relevance bar for this query."""

    def __init__(self, min_score: float = 0.15):
        self.min_score = min_score

    def evaluate(self, result):
        top = result["retrieved"][0]["score"] if result["retrieved"] else 0.0
        return Verdict(top >= self.min_score, top, {"top_score": top, "min_score": self.min_score})


# -- LLM-as-judge -------------------------------------------------------------


class LLMJudge(Judge):
    """Asks a model to score the response 1-5 against a criterion."""

    name = "llm_judge"

    SYSTEM = (
        "You are a strict evaluator for a support assistant. Score the "
        "RESPONSE against the CRITERION with an integer from 1 (worst) to "
        "5 (best). Respond ONLY with JSON: "
        '{"score": <int>, "reasoning": "<one sentence>"}.'
    )

    def __init__(self, provider: BaseProvider, criterion: str, min_score: int = 4):
        self.provider = provider
        self.criterion = criterion
        self.min_score = min_score

    def evaluate(self, result):
        user = (
            f"CRITERION: {self.criterion}\n\n"
            f"QUESTION: {result['query']}\n\n"
            f"RESPONSE: {result['response']}\n\n"
            "Return the JSON verdict now."
        )
        gen = self.provider.generate(self.SYSTEM, user)
        score, reasoning = self._parse(gen.text)
        passed = score >= self.min_score
        return Verdict(
            passed,
            score / 5.0,
            {"raw_score": score, "min_score": self.min_score, "reasoning": reasoning,
             "judge_model": self.provider.model},
        )

    @staticmethod
    def _parse(text: str):
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                return int(d.get("score", 1)), str(d.get("reasoning", ""))[:300]
            except (ValueError, TypeError):
                pass
        return 1, "Judge output could not be parsed"


# -- groundedness / hallucination scoring -------------------------------------


class GroundednessJudge(Judge):
    """Faithfulness of the answer to the retrieved context.

    Splits the answer into atomic sentences and scores each by TF-IDF
    cosine similarity against the retrieved context. Sentences above
    `support_threshold` count as supported; the final score is the
    fraction of supported sentences. Embedding models or an entailment
    NLI model are the production upgrade path.
    """

    name = "groundedness"

    def __init__(self, retriever: TfidfRetriever, min_score: float = 0.7, support_threshold: float = 0.10):
        self.retriever = retriever
        self.min_score = min_score
        self.support_threshold = support_threshold

    def evaluate(self, result):
        context = "\n".join(h["text"] for h in result["retrieved"])
        sentences = split_sentences(result["response"])
        if not sentences:
            return Verdict(False, 0.0, {"error": "empty response"})
        per_sentence = []
        for s in sentences:
            sim = self.retriever.sentence_support(s, context)
            per_sentence.append({"sentence": s[:120], "similarity": round(sim, 4),
                                 "supported": sim >= self.support_threshold})
        score = sum(1 for p in per_sentence if p["supported"]) / len(per_sentence)
        return Verdict(score >= self.min_score, score,
                       {"sentence_scores": per_sentence, "min_score": self.min_score})


# -- factory -------------------------------------------------------------------


def build_judge(spec: dict, provider: BaseProvider, retriever: TfidfRetriever) -> Judge:
    """spec example: {type: contains, args: {keywords: [refund, 30 days]}}"""
    t, args = spec["type"], spec.get("args", {})
    if t == "contains":
        return ContainsJudge(args.get("keywords", []), args.get("mode", "all"))
    if t == "regex":
        return RegexJudge(args["pattern"])
    if t == "length":
        return LengthJudge(args.get("min_words", 5), args.get("max_words", 400))
    if t == "json_valid":
        return JsonValidJudge()
    if t == "retrieval_score":
        return RetrievalScoreJudge(args.get("min_score", 0.15))
    if t == "llm_judge":
        return LLMJudge(provider, args.get("criterion", "The response is accurate and helpful."),
                        args.get("min_score", 4))
    if t == "groundedness":
        return GroundednessJudge(retriever, args.get("min_score", 0.7), args.get("support_threshold", 0.10))
    raise ValueError(f"unknown judge type: {t}")
