"""Retrieval metrics.

Pure functions over ranked id lists and relevant id sets, with binary relevance.
Kept free of any project types so the arithmetic is directly unit-testable
against worked examples.

Conventions
-----------
* Ranks are 1-based in reported output, 0-based internally.
* A query whose relevant set is empty is *undefined* for recall and is excluded
  by :func:`aggregate` rather than being scored as 0, which would silently drag
  every average down.
* ``retrieved`` must not contain duplicates; the runner de-duplicates.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    The denominator is ``k`` even when fewer than ``k`` results were returned:
    returning 3 results of which 3 are relevant is not the same as returning 10
    of which 3 are relevant, and precision should reflect the shortfall.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant products found within the top ``k``."""
    if not relevant:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def reciprocal_rank(
    retrieved: Sequence[str], relevant: set[str], k: int | None = None
) -> float:
    """Reciprocal of the rank of the first relevant result, else 0.

    Args:
        retrieved: Ranked product ids.
        relevant: Relevant product ids.
        k: Optional cut-off; a first hit beyond ``k`` scores 0.
    """
    limit = len(retrieved) if k is None else min(k, len(retrieved))
    for index in range(limit):
        if retrieved[index] in relevant:
            return 1.0 / (index + 1)
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain with binary gains and log2 discount."""
    return sum(
        1.0 / math.log2(index + 2) for index, pid in enumerate(retrieved[:k]) if pid in relevant
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised DCG.

    The ideal ranking places ``min(k, len(relevant))`` relevant items first, so a
    query with fewer relevant items than ``k`` can still reach 1.0.
    """
    if not relevant or k <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    if ideal == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at ``k``: mean precision at each relevant hit."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, pid in enumerate(retrieved[:k]):
        if pid in relevant:
            hits += 1
            total += hits / (index + 1)
    denominator = min(len(relevant), k)
    return total / denominator if denominator else 0.0


@dataclass(slots=True)
class QueryScore:
    """Metrics for a single query."""

    query_id: str
    query_type: str
    relevant_count: int
    retrieved_count: int
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    average_precision: float = 0.0
    reciprocal_rank: float = 0.0
    first_relevant_rank: int | None = None
    latency_ms: float = 0.0

    def as_row(self, cutoffs: Sequence[int]) -> dict[str, object]:
        """Flatten to a serialisable record."""
        row: dict[str, object] = {
            "query_id": self.query_id,
            "type": self.query_type,
            "relevant": self.relevant_count,
            "retrieved": self.retrieved_count,
            "mrr": round(self.reciprocal_rank, 4),
            "ap": round(self.average_precision, 4),
            "first_relevant_rank": self.first_relevant_rank,
            "latency_ms": round(self.latency_ms, 2),
        }
        for k in cutoffs:
            row[f"p@{k}"] = round(self.precision.get(k, 0.0), 4)
            row[f"r@{k}"] = round(self.recall.get(k, 0.0), 4)
            row[f"ndcg@{k}"] = round(self.ndcg.get(k, 0.0), 4)
        return row


def score_query(
    *,
    query_id: str,
    query_type: str,
    retrieved: Sequence[str],
    relevant: set[str],
    cutoffs: Sequence[int],
    latency_ms: float = 0.0,
) -> QueryScore:
    """Compute every metric for one query."""
    largest = max(cutoffs) if cutoffs else len(retrieved)
    first_rank: int | None = None
    for index, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            first_rank = index
            break

    return QueryScore(
        query_id=query_id,
        query_type=query_type,
        relevant_count=len(relevant),
        retrieved_count=len(retrieved),
        precision={k: precision_at_k(retrieved, relevant, k) for k in cutoffs},
        recall={k: recall_at_k(retrieved, relevant, k) for k in cutoffs},
        ndcg={k: ndcg_at_k(retrieved, relevant, k) for k in cutoffs},
        average_precision=average_precision(retrieved, relevant, largest),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        first_relevant_rank=first_rank,
        latency_ms=latency_ms,
    )


@dataclass(slots=True)
class MetricSummary:
    """Metrics averaged over a set of queries."""

    queries: int
    cutoffs: list[int]
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    map_score: float
    median_latency_ms: float
    p95_latency_ms: float
    skipped: int = 0

    def headline(self, k: int = 10) -> str:
        """One-line summary at a given cut-off."""
        return (
            f"n={self.queries}  R@{k}={self.recall.get(k, 0.0):.3f}  "
            f"P@{k}={self.precision.get(k, 0.0):.3f}  "
            f"nDCG@{k}={self.ndcg.get(k, 0.0):.3f}  MRR={self.mrr:.3f}"
        )

    def to_dict(self) -> dict[str, object]:
        """Serialisable representation."""
        return {
            "queries": self.queries,
            "skipped": self.skipped,
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "precision": {f"@{k}": round(v, 4) for k, v in sorted(self.precision.items())},
            "recall": {f"@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "ndcg": {f"@{k}": round(v, 4) for k, v in sorted(self.ndcg.items())},
            "latency_ms": {
                "median": round(self.median_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
            },
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of ``values``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def aggregate(
    scores: Sequence[QueryScore], cutoffs: Sequence[int], skipped: int = 0
) -> MetricSummary:
    """Average per-query scores into a summary.

    Uses the macro average (each query weighted equally) rather than pooling hits
    across queries, so a query with a large relevant set cannot dominate.
    """
    cutoff_list = list(cutoffs)
    if not scores:
        return MetricSummary(
            queries=0,
            cutoffs=cutoff_list,
            precision=dict.fromkeys(cutoff_list, 0.0),
            recall=dict.fromkeys(cutoff_list, 0.0),
            ndcg=dict.fromkeys(cutoff_list, 0.0),
            mrr=0.0,
            map_score=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            skipped=skipped,
        )

    latencies = [s.latency_ms for s in scores]
    return MetricSummary(
        queries=len(scores),
        cutoffs=cutoff_list,
        precision={
            k: statistics.fmean(s.precision.get(k, 0.0) for s in scores) for k in cutoff_list
        },
        recall={k: statistics.fmean(s.recall.get(k, 0.0) for s in scores) for k in cutoff_list},
        ndcg={k: statistics.fmean(s.ndcg.get(k, 0.0) for s in scores) for k in cutoff_list},
        mrr=statistics.fmean(s.reciprocal_rank for s in scores),
        map_score=statistics.fmean(s.average_precision for s in scores),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
        skipped=skipped,
    )
