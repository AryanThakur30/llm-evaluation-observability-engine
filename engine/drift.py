"""Statistical drift detection over the traffic log.

Two families of signals:
- Per-metric distribution shift: PSI (Population Stability Index) and a
  two-sample Kolmogorov-Smirnov test with the asymptotic p-value -- no scipy
  needed, the math is explicit and auditable.
- Semantic drift: TF-IDF similarity of current responses to the reference
  corpus + vocabulary churn (Jaccard overlap, new-term rate).

PSI conventions: <0.10 stable, 0.10-0.25 moderate, >0.25 major shift.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EPS = 1e-6


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two samples via quantile binning."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_two_sample(a, b):
    """Two-sample KS statistic and asymptotic p-value."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) < 5 or len(b) < 5:
        return 0.0, 1.0
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    if d < 1e-12:  # identical empirical CDFs
        return 0.0, 1.0
    en = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(en) + 0.12 + 0.11 / math.sqrt(en)) * d
    # Kolmogorov distribution: Q(l) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 l^2)
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def semantic_drift(reference_texts, current_texts) -> dict:
    """How far current responses have moved from the reference corpus."""
    if not reference_texts or not current_texts:
        return {"mean_similarity": None, "vocab_jaccard": None, "new_term_rate": None}
    vec = TfidfVectorizer(stop_words="english")
    ref = vec.fit_transform(reference_texts)
    cur = vec.transform(current_texts)
    centroid = np.asarray(ref.mean(axis=0))
    sims = cosine_similarity(cur, centroid)[:, 0]
    ref_vocab = set(vec.get_feature_names_out())
    cur_terms = set()
    for t in current_texts:
        cur_terms |= set(t.lower().split())
    inter = ref_vocab & cur_terms
    union = ref_vocab | cur_terms
    return {
        "mean_similarity": round(float(sims.mean()), 4),
        "vocab_jaccard": round(len(inter) / len(union) if union else 1.0, 4),
        "new_term_rate": round(len(cur_terms - ref_vocab) / max(1, len(cur_terms)), 4),
    }


def _psi_verdict(v: float) -> str:
    if v > 0.25:
        return "MAJOR_DRIFT"
    if v > 0.10:
        return "MODERATE_DRIFT"
    return "STABLE"


def analyze_traffic(storage, reference_days: tuple[int, int], current_days: tuple[int, int]) -> dict:
    """Full drift report comparing two day ranges of the traffic log."""
    ref = storage.traffic_metrics_between_days(*reference_days)
    cur = storage.traffic_metrics_between_days(*current_days)

    metrics = {}
    alerts = []
    for metric in ("latency_ms", "completion_tokens", "cost_usd", "response_words"):
        a, b = ref[metric], cur[metric]
        if not a or not b:
            continue
        p = psi(a, b)
        d, kp = ks_two_sample(a, b)
        drifted = p > 0.10 or kp < 0.05
        metrics[metric] = {
            "psi": round(p, 4), "ks_d": round(d, 4), "ks_p": round(kp, 5),
            "ref_mean": round(float(np.mean(a)), 3), "cur_mean": round(float(np.mean(b)), 3),
            "verdict": _psi_verdict(p),
        }
        if drifted:
            alerts.append(
                f"{metric}: {_psi_verdict(p)} (PSI={p:.3f}, KS p={kp:.4f}, "
                f"mean {np.mean(a):.2f} -> {np.mean(b):.2f})"
            )

    sem = semantic_drift(ref["responses"], cur["responses"])
    if sem["mean_similarity"] is not None:
        metrics["semantic"] = sem
        if sem["mean_similarity"] < 0.25 or sem["vocab_jaccard"] < 0.5:
            alerts.append(
                f"semantic drift: similarity to reference corpus fell to "
                f"{sem['mean_similarity']}, vocab jaccard {sem['vocab_jaccard']}"
            )

    import datetime

    return {
        "created_at": datetime.datetime.now(),
        "reference_days": list(reference_days),
        "current_days": list(current_days),
        "n_reference": len(ref["latency_ms"]),
        "n_current": len(cur["latency_ms"]),
        "metrics": metrics,
        "alerts": alerts,
        "drift_detected": len(alerts) > 0,
    }
