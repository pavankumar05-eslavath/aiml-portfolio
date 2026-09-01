"""Repository tests: SQL access, lexical search, and the Qdrant layer.

The vector-store tests run against a real embedded Qdrant instance, so named
vectors, payload filtering and dimension guards are genuinely exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateProductError, NotFoundError, VectorStoreError
from app.models.product import Product
from app.repositories.product_repository import (
    ProductRepository,
    build_search_document_for,
    compute_content_hash,
)
from app.repositories.vector_repository import (
    IMAGE_VECTOR,
    TEXT_VECTOR,
    VectorRepository,
    build_qdrant_filter,
)
from app.schemas.product import ProductCreate, ProductUpdate
from app.schemas.search import SearchFilters, SortOption
from tests.conftest import FAKE_DIM, make_product


def unit_vector(seed: int, dim: int = FAKE_DIM) -> list[float]:
    """A deterministic unit vector, for exact similarity expectations."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=dim).astype(np.float32)
    return (values / np.linalg.norm(values)).tolist()


class TestContentHash:
    def test_is_stable_for_identical_input(self):
        product = make_product("Shoe")
        assert compute_content_hash(product, "m") == compute_content_hash(product, "m")

    def test_changes_when_an_embedding_field_changes(self):
        product = make_product("Shoe")
        before = compute_content_hash(product, "m")
        product.colour = "Crimson"
        assert compute_content_hash(product, "m") != before

    def test_ignores_fields_that_do_not_affect_the_embedding(self):
        """Price and stock do not feed the encoder, so they must not force re-embedding."""
        product = make_product("Shoe")
        before = compute_content_hash(product, "m")
        product.price = 1234.0
        product.in_stock = False
        assert compute_content_hash(product, "m") == before

    def test_changes_when_the_model_changes(self):
        """Switching MODEL_NAME must invalidate every stored vector."""
        product = make_product("Shoe")
        assert compute_content_hash(product, "clip") != compute_content_hash(product, "siglip")


class TestProductRepositoryWrites:
    async def test_create_persists_and_builds_document(
        self, products: ProductRepository, session: AsyncSession
    ):
        created = await products.create(
            ProductCreate(name="Nike Runner", brand="Nike", colour="Black", external_id="x1")
        )
        await session.commit()
        assert created.id
        assert created.search_document
        assert "Nike" in created.search_document

    async def test_create_rejects_duplicate_external_id(self, products: ProductRepository):
        await products.create(ProductCreate(name="A", external_id="dup"))
        with pytest.raises(DuplicateProductError):
            await products.create(ProductCreate(name="B", external_id="dup"))

    async def test_get_or_404_raises_for_missing(self, products: ProductRepository):
        with pytest.raises(NotFoundError):
            await products.get_or_404("nope")

    async def test_update_clears_hash_for_embedding_fields(
        self, products: ProductRepository, session: AsyncSession
    ):
        created = await products.create(ProductCreate(name="A", colour="Black"))
        created.content_hash = "stale-hash"
        await session.flush()

        updated = await products.update(created.id, ProductUpdate(colour="White"))
        assert updated.content_hash is None

    async def test_update_keeps_hash_for_non_embedding_fields(
        self, products: ProductRepository, session: AsyncSession
    ):
        created = await products.create(ProductCreate(name="A", colour="Black"))
        created.content_hash = "keep-me"
        await session.flush()

        updated = await products.update(created.id, ProductUpdate(price=42.0))
        assert updated.content_hash == "keep-me"

    async def test_bulk_upsert_inserts_then_updates(
        self, products: ProductRepository, session: AsyncSession
    ):
        payloads = [
            ProductCreate(name="One", external_id="e1", price=10.0),
            ProductCreate(name="Two", external_id="e2", price=20.0),
        ]
        touched, updated = await products.bulk_upsert(payloads)
        await session.commit()
        assert len(touched) == 2
        assert updated == 0

        # Re-running with a changed price must update in place, not duplicate.
        touched, updated = await products.bulk_upsert(
            [ProductCreate(name="One renamed", external_id="e1", price=11.0)]
        )
        await session.commit()
        assert updated == 1
        assert await products.count_all() == 2

    async def test_delete_removes_the_row(
        self, products: ProductRepository, session: AsyncSession
    ):
        created = await products.create(ProductCreate(name="Temp"))
        await session.commit()
        await products.delete(created.id)
        await session.commit()
        assert await products.get(created.id) is None


class TestProductRepositoryReads:
    @pytest.fixture
    async def stored(
        self, session: AsyncSession, sample_products: list[Product]
    ) -> list[Product]:
        for product in sample_products:
            product.search_document = build_search_document_for(product)
            session.add(product)
        await session.flush()
        return sample_products

    async def test_get_many_returns_a_mapping(self, products: ProductRepository, stored):
        ids = [stored[0].id, stored[2].id]
        found = await products.get_many(ids)
        assert set(found) == set(ids)

    async def test_get_many_with_no_ids_avoids_a_query(self, products: ProductRepository):
        assert await products.get_many([]) == {}

    async def test_list_returns_total_ignoring_pagination(
        self, products: ProductRepository, stored
    ):
        rows, total = await products.list_products(limit=2, offset=0)
        assert len(rows) == 2
        assert total == len(stored)

    async def test_filter_by_price_range(self, products: ProductRepository, stored):
        rows, total = await products.list_products(
            filters=SearchFilters(min_price=70, max_price=130), limit=50
        )
        assert total == 3
        assert all(70 <= row.price <= 130 for row in rows)

    async def test_filter_ids_narrows_a_candidate_set(
        self, products: ProductRepository, stored
    ):
        all_ids = [p.id for p in stored]
        kept = await products.filter_ids(SearchFilters(brands=["Nike"]), all_ids)
        assert len(kept) == 2

    async def test_sort_price_ascending_puts_nulls_last(
        self, products: ProductRepository, stored
    ):
        rows, _ = await products.list_products(sort=SortOption.PRICE_ASC, limit=50)
        prices = [row.price for row in rows]
        assert prices[-1] is None
        priced = [p for p in prices if p is not None]
        assert priced == sorted(priced)

    async def test_facets_report_counts_and_bounds(self, products: ProductRepository, stored):
        facets = await products.facets()
        assert dict(facets["categories"])["Footwear"] == 3
        assert facets["price_min"] == 60.0
        assert facets["price_max"] == 210.0

    async def test_stats_counts_indexing_state(self, products: ProductRepository, stored):
        stats = await products.stats()
        assert stats["total_products"] == len(stored)
        assert stats["indexed_products"] == 0
        assert stats["pending_products"] == len(stored)
        assert stats["with_image_vector"] == 0

    async def test_stats_on_an_empty_catalogue(self, products: ProductRepository):
        stats = await products.stats()
        assert stats["total_products"] == 0
        assert stats["with_text_vector"] == 0

    async def test_iter_for_indexing_selects_only_pending(
        self, products: ProductRepository, session: AsyncSession, stored
    ):
        pending = await products.iter_for_indexing()
        assert len(pending) == len(stored)

        await products.mark_indexed(
            stored[0],
            content_hash="h",
            document="d",
            has_image_vector=True,
            has_text_vector=True,
        )
        await session.flush()
        assert len(await products.iter_for_indexing()) == len(stored) - 1

    async def test_iter_for_indexing_force_selects_everything(
        self, products: ProductRepository, session: AsyncSession, stored
    ):
        for product in stored:
            await products.mark_indexed(
                product,
                content_hash="h",
                document="d",
                has_image_vector=True,
                has_text_vector=True,
            )
        await session.flush()
        assert await products.iter_for_indexing() == []
        assert len(await products.iter_for_indexing(force=True)) == len(stored)

    async def test_iter_for_indexing_respects_limit(self, products: ProductRepository, stored):
        assert len(await products.iter_for_indexing(limit=2)) == 2


class TestLexicalSearch:
    @pytest.fixture
    async def stored(
        self, session: AsyncSession, sample_products: list[Product]
    ) -> list[Product]:
        for product in sample_products:
            session.add(product)
        await session.flush()
        return sample_products

    async def test_matches_on_the_product_name(self, products: ProductRepository, stored):
        results = await products.lexical_search("nike")
        assert results
        assert all(0.0 <= score <= 1.0 for _, score in results)

    async def test_scores_are_ordered(self, products: ProductRepository, stored):
        results = await products.lexical_search("nike black shoes")
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    async def test_name_matches_outrank_facet_only_matches(
        self, products: ProductRepository, stored
    ):
        """A term in the title is stronger evidence than the same term in a facet."""
        results = dict(await products.lexical_search("watch"))
        watch = next(p for p in stored if "Watch" in p.name)
        assert results.get(watch.id, 0) > 0

    async def test_no_terms_returns_nothing(self, products: ProductRepository, stored):
        assert await products.lexical_search("!!! ???") == []

    async def test_respects_filters(self, products: ProductRepository, stored):
        results = await products.lexical_search("shoes", filters=SearchFilters(brands=["Puma"]))
        puma = next(p for p in stored if p.brand == "Puma")
        assert [pid for pid, _ in results] == [puma.id]

    async def test_respects_limit(self, products: ProductRepository, stored):
        assert len(await products.lexical_search("shoes", limit=1)) <= 1


class TestQdrantFilterTranslation:
    def test_empty_filters_produce_no_qdrant_filter(self):
        assert build_qdrant_filter(SearchFilters()) is None

    def test_single_facet_becomes_one_condition(self):
        built = build_qdrant_filter(SearchFilters(categories=["Footwear"]))
        assert built is not None
        assert len(built.must) == 1

    def test_multiple_facets_are_anded(self):
        built = build_qdrant_filter(
            SearchFilters(categories=["Footwear"], brands=["Nike"], colours=["Black"])
        )
        assert len(built.must) == 3

    def test_price_range_becomes_a_single_range_condition(self):
        built = build_qdrant_filter(SearchFilters(min_price=10, max_price=20))
        assert len(built.must) == 1
        assert built.must[0].range.gte == 10
        assert built.must[0].range.lte == 20

    def test_open_ended_price_range(self):
        built = build_qdrant_filter(SearchFilters(min_price=10))
        assert built.must[0].range.gte == 10
        assert built.must[0].range.lte is None

    def test_in_stock_flag(self):
        built = build_qdrant_filter(SearchFilters(in_stock_only=True))
        assert len(built.must) == 1


class TestVectorRepository:
    async def test_ensure_collection_is_idempotent(self, vectors: VectorRepository):
        assert await vectors.ensure_collection(FAKE_DIM) is False  # already created

    async def test_rejects_a_dimension_mismatch(self, vectors: VectorRepository):
        """Catches the "changed MODEL_NAME without reindexing" mistake loudly."""
        with pytest.raises(VectorStoreError, match="recreate it"):
            await vectors.ensure_collection(FAKE_DIM + 8)

    async def test_upsert_and_retrieve_named_vectors(self, vectors: VectorRepository):
        product_id = "11111111-1111-1111-1111-111111111111"
        await vectors.upsert(
            [
                (
                    product_id,
                    {IMAGE_VECTOR: unit_vector(1), TEXT_VECTOR: unit_vector(2)},
                    {"product_id": product_id, "category": "Footwear", "price": 10.0},
                )
            ]
        )
        stored = await vectors.get_vectors(product_id)
        assert set(stored) == {IMAGE_VECTOR, TEXT_VECTOR}
        assert len(stored[IMAGE_VECTOR]) == FAKE_DIM

    async def test_text_only_product_is_searchable(self, vectors: VectorRepository):
        """A product without a usable image must still be findable by text."""
        product_id = "22222222-2222-2222-2222-222222222222"
        await vectors.upsert(
            [(product_id, {TEXT_VECTOR: unit_vector(3)}, {"product_id": product_id})]
        )
        hits = await vectors.search(unit_vector(3), using=TEXT_VECTOR, limit=5)
        assert hits[0].product_id == product_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)

    async def test_search_returns_cosine_ordered_hits(self, vectors: VectorRepository):
        query = unit_vector(10)
        near = (0.98 * np.array(query) + 0.02 * np.array(unit_vector(11))).tolist()
        records = [
            (
                "aaaaaaaa-0000-0000-0000-000000000001",
                {IMAGE_VECTOR: query},
                {"product_id": "a"},
            ),
            ("aaaaaaaa-0000-0000-0000-000000000002", {IMAGE_VECTOR: near}, {"product_id": "b"}),
            (
                "aaaaaaaa-0000-0000-0000-000000000003",
                {IMAGE_VECTOR: unit_vector(99)},
                {"product_id": "c"},
            ),
        ]
        await vectors.upsert(records)
        hits = await vectors.search(query, using=IMAGE_VECTOR, limit=3)
        assert [hit.product_id for hit in hits] == ["a", "b", "c"]
        assert hits[0].score > hits[1].score > hits[2].score

    async def test_payload_filter_is_applied_by_the_engine(self, vectors: VectorRepository):
        await vectors.upsert(
            [
                (
                    f"bbbbbbbb-0000-0000-0000-00000000000{i}",
                    {IMAGE_VECTOR: unit_vector(i)},
                    {"product_id": f"p{i}", "category": "Footwear" if i < 2 else "Apparel"},
                )
                for i in range(4)
            ]
        )
        hits = await vectors.search(
            unit_vector(0),
            using=IMAGE_VECTOR,
            limit=10,
            filters=SearchFilters(categories=["Apparel"]),
        )
        assert {hit.payload["category"] for hit in hits} == {"Apparel"}

    async def test_search_many_runs_channels_concurrently(self, vectors: VectorRepository):
        product_id = "cccccccc-0000-0000-0000-000000000001"
        await vectors.upsert(
            [
                (
                    product_id,
                    {IMAGE_VECTOR: unit_vector(5), TEXT_VECTOR: unit_vector(6)},
                    {"product_id": product_id},
                )
            ]
        )
        results = await vectors.search_many(
            {
                "image_to_image": (IMAGE_VECTOR, unit_vector(5)),
                "text_to_text": (TEXT_VECTOR, unit_vector(6)),
            },
            limit=5,
        )
        assert set(results) == {"image_to_image", "text_to_text"}
        assert all(hits for hits in results.values())

    async def test_search_many_with_no_queries(self, vectors: VectorRepository):
        assert await vectors.search_many({}, limit=5) == {}

    async def test_delete_removes_points(self, vectors: VectorRepository):
        product_id = "dddddddd-0000-0000-0000-000000000001"
        await vectors.upsert(
            [(product_id, {TEXT_VECTOR: unit_vector(7)}, {"product_id": product_id})]
        )
        assert await vectors.count() == 1
        await vectors.delete([product_id])
        assert await vectors.count() == 0

    async def test_delete_with_no_ids_is_a_no_op(self, vectors: VectorRepository):
        assert await vectors.delete([]) == 0

    async def test_get_vectors_for_unknown_id(self, vectors: VectorRepository):
        assert await vectors.get_vectors("00000000-0000-0000-0000-000000000000") == {}

    async def test_health_reports_collection_state(self, vectors: VectorRepository):
        health = await vectors.health()
        assert health["exists"] is True
        assert health["mode"] == "embedded"
        assert health["points"] == 0

    async def test_upsert_with_no_records(self, vectors: VectorRepository):
        assert await vectors.upsert([]) == 0
