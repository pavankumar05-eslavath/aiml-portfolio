"""Evaluation runner.

Executes benchmark queries through the real :class:`SearchService` in-process -
the same code path the HTTP API uses, minus the network. Running the service
directly rather than over HTTP keeps runs reproducible and lets a parameter sweep
override fusion settings per query without restarting anything.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.config import FusionStrategy, ScoreNormalization
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.evaluation.dataset import QueryType, ResolvedQuery
from app.evaluation.metrics import MetricSummary, QueryScore, aggregate, score_query
from app.schemas.search import FusionOverrides, SearchFilters
from app.services.search_service import SearchQuery, SearchService

logger = get_logger(__name__)

DEFAULT_CUTOFFS: tuple[int, ...] = (1, 5, 10, 20)


@dataclass(slots=True)
class RunConfig:
    """One configuration under test."""

    label: str
    strategy: FusionStrategy | None = None
    normalization: ScoreNormalization | None = None
    image_weight: float | None = None
    text_weight: float | None = None
    lexical_weight: float | None = None
    cross_modal_weight: float | None = None
    embedding_fusion_alpha: float | None = None
    rrf_k: int | None = None
    top_k: int = 20
    #: Restrict to these query types; empty means all.
    query_types: tuple[QueryType, ...] = ()
    #: Restrict to these query groups (e.g. "contradiction"); empty means all.
    groups: tuple[str, ...] = ()

    def to_overrides(self) -> FusionOverrides:
        """Render as per-request fusion overrides."""
        return FusionOverrides(
            strategy=self.strategy,
            normalization=self.normalization,
            image_weight=self.image_weight,
            text_weight=self.text_weight,
            lexical_weight=self.lexical_weight,
            cross_modal_weight=self.cross_modal_weight,
            embedding_fusion_alpha=self.embedding_fusion_alpha,
            rrf_k=self.rrf_k,
        )

    def describe(self) -> dict[str, Any]:
        """Serialisable description of the knobs this config sets."""
        fields = {
            "strategy": self.strategy.value if self.strategy else None,
            "normalization": self.normalization.value if self.normalization else None,
            "image_weight": self.image_weight,
            "text_weight": self.text_weight,
            "lexical_weight": self.lexical_weight,
            "cross_modal_weight": self.cross_modal_weight,
            "embedding_fusion_alpha": self.embedding_fusion_alpha,
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
        }
        return {key: value for key, value in fields.items() if value is not None}


@dataclass(slots=True)
class RunResult:
    """Metrics for one configuration, overall and split by query type."""

    config: RunConfig
    overall: MetricSummary
    by_type: dict[str, MetricSummary] = field(default_factory=dict)
    scores: list[QueryScore] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self, include_queries: bool = False) -> dict[str, Any]:
        """Serialisable representation."""
        payload: dict[str, Any] = {
            "label": self.config.label,
            "config": self.config.describe(),
            "overall": self.overall.to_dict(),
            "by_type": {name: summary.to_dict() for name, summary in self.by_type.items()},
        }
        if self.errors:
            payload["errors"] = [{"query_id": q, "error": e} for q, e in self.errors]
        if self.skipped:
            payload["skipped"] = [{"query_id": q, "reason": r} for q, r in self.skipped]
        if include_queries:
            payload["queries"] = [score.as_row(self.overall.cutoffs) for score in self.scores]
        return payload


class EvaluationRunner:
    """Runs benchmark queries against a :class:`SearchService`."""

    def __init__(
        self,
        service: SearchService,
        *,
        cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    ) -> None:
        self._service = service
        self._cutoffs = list(cutoffs)

    async def run(self, resolved: Sequence[ResolvedQuery], config: RunConfig) -> RunResult:
        """Execute every runnable query under one configuration.

        A query that raises is recorded as an error and excluded from the averages
        rather than being scored 0: a crash is a defect to fix, not a retrieval
        result, and averaging it in would disguise the failure as poor quality.
        """
        scores: list[QueryScore] = []
        errors: list[tuple[str, str]] = []
        skipped: list[tuple[str, str]] = []
        overrides = config.to_overrides()
        # Retrieve enough depth to score the largest cut-off.
        top_k = max(config.top_k, max(self._cutoffs))

        for item in resolved:
            query = item.query
            if config.query_types and query.type not in config.query_types:
                continue
            if config.groups and (query.group or "") not in config.groups:
                continue
            if not item.is_runnable:
                skipped.append((query.id, item.skip_reason or "unknown"))
                continue

            search_query = SearchQuery(
                text=query.text if query.type is not QueryType.IMAGE else None,
                image_product_id=item.image_product_id,
                top_k=top_k,
                filters=SearchFilters(),
                overrides=overrides,
                # Score breakdowns are not needed for metrics; skipping them
                # avoids building objects for every result of every sweep.
                explain=False,
            )

            started = time.perf_counter()
            try:
                response = await self._service.search(search_query)
            except AppError as exc:
                errors.append((query.id, f"{exc.code}: {exc.message}"))
                continue
            except Exception as exc:
                errors.append((query.id, f"{type(exc).__name__}: {exc}"))
                logger.exception(
                    "evaluation query failed", extra={"context": {"query": query.id}}
                )
                continue
            latency_ms = (time.perf_counter() - started) * 1000

            retrieved = [result.product.id for result in response.results]
            scores.append(
                score_query(
                    query_id=query.id,
                    query_type=query.type.value,
                    retrieved=retrieved,
                    relevant=item.relevant_ids,
                    cutoffs=self._cutoffs,
                    latency_ms=latency_ms,
                )
            )

        by_type: dict[str, MetricSummary] = {}
        for query_type in (QueryType.TEXT, QueryType.IMAGE, QueryType.MULTIMODAL):
            subset = [s for s in scores if s.query_type == query_type.value]
            if subset:
                by_type[query_type.value] = aggregate(subset, self._cutoffs)

        return RunResult(
            config=config,
            overall=aggregate(scores, self._cutoffs, skipped=len(skipped)),
            by_type=by_type,
            scores=scores,
            errors=errors,
            skipped=skipped,
        )

    async def run_many(
        self, resolved: Sequence[ResolvedQuery], configs: Sequence[RunConfig]
    ) -> list[RunResult]:
        """Run several configurations over the same query set."""
        results: list[RunResult] = []
        for index, config in enumerate(configs, start=1):
            logger.info(
                "running configuration",
                extra={"context": {"label": config.label, "n": f"{index}/{len(configs)}"}},
            )
            results.append(await self.run(resolved, config))
        return results
