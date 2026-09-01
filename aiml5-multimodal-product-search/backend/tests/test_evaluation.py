"""Evaluation metric and ground-truth tests.

Metric implementations are verified against hand-worked examples. If these were
wrong, every number in the evaluation report would be wrong too, so they are
checked against values computed independently in the test body.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.dataset import (
    EvalQuery,
    QueryType,
    RelevanceRule,
    load_queries,
    resolve_queries,
)
from app.evaluation.metrics import (
    aggregate,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)
from tests.conftest import make_product

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_denominator_is_k_even_when_fewer_results(self):
        """Returning 2 of 10 asked-for results is worse than returning 2 of 2."""
        assert precision_at_k(["a", "b"], {"a", "b"}, 10) == 0.2

    def test_only_counts_within_the_cutoff(self):
        assert precision_at_k(["x", "x", "a"], {"a"}, 2) == 0.0

    def test_zero_k(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecall:
    def test_finds_everything(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0

    def test_finds_half(self):
        assert recall_at_k(["a"], {"a", "b"}, 5) == 0.5

    def test_bounded_by_the_cutoff(self):
        """R@k cannot exceed k/|relevant|, which is why R@10 looks low here."""
        relevant = {f"p{i}" for i in range(50)}
        retrieved = [f"p{i}" for i in range(50)]
        assert recall_at_k(retrieved, relevant, 10) == pytest.approx(10 / 50)

    def test_empty_relevant_set_is_zero(self):
        assert recall_at_k(["a"], set(), 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit_is_zero(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_cutoff_excludes_late_hits(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestNdcg:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_reversed_ranking_scores_lower(self):
        good = ndcg_at_k(["a", "b", "x", "y"], {"a", "b"}, 4)
        bad = ndcg_at_k(["x", "y", "a", "b"], {"a", "b"}, 4)
        assert good > bad

    def test_matches_a_hand_computation(self):
        # One relevant item at rank 2: DCG = 1/log2(3); ideal = 1/log2(2) = 1.
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(expected)

    def test_reaches_one_when_relevant_set_is_smaller_than_k(self):
        assert ndcg_at_k(["a", "x", "y"], {"a"}, 3) == pytest.approx(1.0)

    def test_empty_relevant_set(self):
        assert ndcg_at_k(["a"], set(), 5) == 0.0

    def test_dcg_discounts_by_position(self):
        assert dcg_at_k(["a"], {"a"}, 1) == pytest.approx(1.0)
        assert dcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / math.log2(3))


class TestAveragePrecision:
    def test_matches_a_hand_computation(self):
        # Hits at ranks 1 and 3: (1/1 + 2/3) / 2
        assert average_precision(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(
            (1.0 + 2 / 3) / 2
        )

    def test_perfect_ranking(self):
        assert average_precision(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_no_hits(self):
        assert average_precision(["x"], {"a"}, 1) == 0.0


class TestScoreAndAggregate:
    def test_score_query_populates_every_metric(self):
        score = score_query(
            query_id="q1",
            query_type="text",
            retrieved=["x", "a", "b"],
            relevant={"a", "b"},
            cutoffs=[1, 3],
            latency_ms=12.5,
        )
        assert score.first_relevant_rank == 2
        assert score.reciprocal_rank == pytest.approx(0.5)
        assert score.precision[1] == 0.0
        assert score.precision[3] == pytest.approx(2 / 3)
        assert score.recall[3] == 1.0
        assert score.latency_ms == 12.5

    def test_score_query_with_no_hits(self):
        score = score_query(
            query_id="q2",
            query_type="image",
            retrieved=["x", "y"],
            relevant={"a"},
            cutoffs=[5],
        )
        assert score.first_relevant_rank is None
        assert score.reciprocal_rank == 0.0

    def test_aggregate_uses_the_macro_average(self):
        """Each query weighs equally, so a large relevant set cannot dominate."""
        scores = [
            score_query(
                query_id="a",
                query_type="text",
                retrieved=["a"],
                relevant={"a"},
                cutoffs=[1],
            ),
            score_query(
                query_id="b",
                query_type="text",
                retrieved=["x"],
                relevant={"b"},
                cutoffs=[1],
            ),
        ]
        summary = aggregate(scores, [1])
        assert summary.queries == 2
        assert summary.precision[1] == 0.5
        assert summary.mrr == 0.5

    def test_aggregate_of_nothing_is_safe(self):
        summary = aggregate([], [1, 5])
        assert summary.queries == 0
        assert summary.mrr == 0.0
        assert summary.precision[5] == 0.0

    def test_aggregate_reports_latency_percentiles(self):
        scores = [
            score_query(
                query_id=str(i),
                query_type="text",
                retrieved=["a"],
                relevant={"a"},
                cutoffs=[1],
                latency_ms=float(i),
            )
            for i in range(1, 101)
        ]
        summary = aggregate(scores, [1])
        assert summary.median_latency_ms == pytest.approx(50.5)
        assert summary.p95_latency_ms == pytest.approx(95.0)

    def test_summary_serialises(self):
        summary = aggregate(
            [
                score_query(
                    query_id="a",
                    query_type="text",
                    retrieved=["a"],
                    relevant={"a"},
                    cutoffs=[1],
                )
            ],
            [1],
        )
        payload = summary.to_dict()
        assert payload["queries"] == 1
        assert "@1" in payload["recall"]


class TestRelevanceRule:
    def test_matches_on_a_single_attribute(self):
        rule = RelevanceRule(subcategories=["Sports Shoes"])
        assert rule.matches(make_product(subcategory="Sports Shoes"))
        assert not rule.matches(make_product(subcategory="Watches"))

    def test_is_case_insensitive(self):
        rule = RelevanceRule(colours=["black"])
        assert rule.matches(make_product(colour="Black"))

    def test_fields_are_anded(self):
        rule = RelevanceRule(subcategories=["Sports Shoes"], colours=["Black"])
        assert rule.matches(make_product(subcategory="Sports Shoes", colour="Black"))
        assert not rule.matches(make_product(subcategory="Sports Shoes", colour="White"))

    def test_values_within_a_field_are_ored(self):
        rule = RelevanceRule(genders=["Men", "Unisex"])
        assert rule.matches(make_product(gender="Men"))
        assert rule.matches(make_product(gender="Unisex"))
        assert not rule.matches(make_product(gender="Women"))

    def test_missing_attribute_fails_a_constrained_field(self):
        assert not RelevanceRule(colours=["Black"]).matches(make_product(colour=None))

    def test_excludes_named_external_ids(self):
        rule = RelevanceRule(colours=["Black"], exclude_external_ids=["skip"])
        assert not rule.matches(make_product(external_id="skip", colour="Black"))

    def test_empty_rule_is_flagged(self):
        assert RelevanceRule().is_empty()
        assert not RelevanceRule(colours=["Black"]).is_empty()

    def test_describe_is_human_readable(self):
        described = RelevanceRule(subcategories=["Heels"], colours=["Black"]).describe()
        assert "type in {Heels}" in described
        assert "AND" in described

    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError, match="unknown relevance field"):
            RelevanceRule.from_dict({"subcategorys": ["Heels"]})


class TestEvalQueryParsing:
    def test_parses_a_text_query(self):
        query = EvalQuery.from_dict(
            {
                "id": "t1",
                "type": "text",
                "query": "black shoes",
                "relevance": {"colours": ["Black"]},
            }
        )
        assert query.type is QueryType.TEXT
        assert query.text == "black shoes"

    def test_text_query_requires_text(self):
        with pytest.raises(ValueError, match="requires 'query' text"):
            EvalQuery.from_dict(
                {"id": "t2", "type": "text", "relevance": {"colours": ["Black"]}}
            )

    def test_image_query_requires_an_image_id(self):
        with pytest.raises(ValueError, match="requires 'image_external_id'"):
            EvalQuery.from_dict(
                {"id": "i1", "type": "image", "relevance": {"colours": ["Black"]}}
            )

    def test_multimodal_requires_both(self):
        with pytest.raises(ValueError):
            EvalQuery.from_dict(
                {
                    "id": "m1",
                    "type": "multimodal",
                    "query": "in black",
                    "relevance": {"colours": ["Black"]},
                }
            )

    def test_rejects_an_unconstrained_relevance_rule(self):
        """A rule matching everything would report a meaninglessly perfect score."""
        with pytest.raises(ValueError, match="matches everything"):
            EvalQuery.from_dict({"id": "x", "type": "text", "query": "q", "relevance": {}})

    def test_rejects_an_unknown_type(self):
        with pytest.raises(ValueError, match="invalid or missing 'type'"):
            EvalQuery.from_dict({"id": "x", "type": "audio", "query": "q"})

    def test_parses_the_group_field(self):
        query = EvalQuery.from_dict(
            {
                "id": "m1",
                "type": "multimodal",
                "query": "in black",
                "image_external_id": "1",
                "group": "contradiction",
                "relevance": {"colours": ["Black"]},
            }
        )
        assert query.group == "contradiction"


class TestShippedBenchmark:
    """The committed benchmark must stay loadable and internally consistent."""

    @pytest.fixture
    def path(self) -> Path:
        return REPO_ROOT / "evaluation" / "queries" / "benchmark.json"

    def test_loads_without_error(self, path: Path):
        queries = load_queries(path)
        assert len(queries) >= 30

    def test_ids_are_unique(self, path: Path):
        queries = load_queries(path)
        assert len({q.id for q in queries}) == len(queries)

    def test_covers_all_three_modes(self, path: Path):
        types = {q.type for q in load_queries(path)}
        assert types == {QueryType.TEXT, QueryType.IMAGE, QueryType.MULTIMODAL}

    def test_multimodal_queries_are_grouped(self, path: Path):
        """Both multimodal styles must be present, or the weight sweep is biased."""
        groups = {q.group for q in load_queries(path) if q.type is QueryType.MULTIMODAL}
        assert groups == {"contradiction", "agreement"}

    def test_every_query_has_a_constrained_rule(self, path: Path):
        assert all(not q.relevance.is_empty() for q in load_queries(path))

    def test_file_is_valid_json_with_documented_policy(self, path: Path):
        payload = json.loads(path.read_text())
        assert "relevance_policy" in payload

    def test_missing_file_raises_clearly(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_queries(tmp_path / "nope.json")

    def test_malformed_json_raises_clearly(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_queries(bad)


class TestResolveQueries:
    async def test_resolves_relevance_against_the_catalogue(
        self, session: AsyncSession, indexed_catalog
    ):
        query = EvalQuery.from_dict(
            {
                "id": "t1",
                "type": "text",
                "query": "black nike",
                "relevance": {"brands": ["Nike"], "colours": ["Black"]},
            }
        )
        resolved = await resolve_queries([query], session)
        assert len(resolved[0].relevant_ids) == 2
        assert resolved[0].is_runnable

    async def test_skips_queries_with_too_few_relevant_products(
        self, session: AsyncSession, indexed_catalog
    ):
        """Metrics over one product are noise, so such queries are excluded."""
        query = EvalQuery.from_dict(
            {
                "id": "t2",
                "type": "text",
                "query": "denim",
                "relevance": {"brands": ["Levis"]},
            }
        )
        resolved = await resolve_queries([query], session, min_relevant=2)
        assert not resolved[0].is_runnable
        assert "only 1 relevant" in resolved[0].skip_reason

    async def test_skips_when_the_query_image_is_absent(
        self, session: AsyncSession, indexed_catalog
    ):
        query = EvalQuery.from_dict(
            {
                "id": "i1",
                "type": "image",
                "image_external_id": "does-not-exist",
                "relevance": {"colours": ["Black"]},
            }
        )
        resolved = await resolve_queries([query], session)
        assert not resolved[0].is_runnable
        assert "not in catalogue" in resolved[0].skip_reason

    async def test_query_product_is_excluded_from_its_own_relevant_set(
        self, session: AsyncSession, indexed_catalog
    ):
        query = EvalQuery.from_dict(
            {
                "id": "i2",
                "type": "image",
                "image_external_id": "p1",
                "relevance": {"brands": ["Nike"]},
            }
        )
        resolved = await resolve_queries([query], session)
        target = next(p for p in indexed_catalog if p.external_id == "p1")
        assert target.id not in resolved[0].relevant_ids
        assert len(resolved[0].relevant_ids) == 1

    async def test_unindexed_products_are_not_counted_as_relevant(
        self, session: AsyncSession, sample_products
    ):
        """A product that search cannot return must not be in the ground truth."""
        for product in sample_products:
            session.add(product)
        await session.flush()

        query = EvalQuery.from_dict(
            {"id": "t3", "type": "text", "query": "nike", "relevance": {"brands": ["Nike"]}}
        )
        resolved = await resolve_queries([query], session)
        assert resolved[0].relevant_ids == set()
        assert not resolved[0].is_runnable
