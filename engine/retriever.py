"""TF-IDF retrieval + sentence-level similarity used for RAG and groundedness.

In production you would swap in an embedding model (bge-small, nomic-embed,
OpenAI embeddings). TF-IDF keeps the demo dependency-free and makes the
retrieval-quality math explicit and auditable.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class TfidfRetriever:
    def __init__(self, docs: list[str]):
        self.docs = list(docs)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(self.docs)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(-sims)[:k]
        return [{"index": int(i), "text": self.docs[i], "score": float(sims[i])} for i in order]

    def sentence_support(self, sentence: str, context: str) -> float:
        """Cosine similarity of one sentence against a context corpus."""
        s = self.vec.transform([sentence])
        c = self.vec.transform([context])
        return float(cosine_similarity(s, c)[0, 0])
