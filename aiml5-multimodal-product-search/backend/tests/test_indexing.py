"""Indexing pipeline tests: incremental behaviour, partial success, failure handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import IndexingError
from app.models.product import Product
from app.repositories.product_repository import ProductRepository
from app.repositories.vector_repository import IMAGE_VECTOR, TEXT_VECTOR, VectorRepository
from app.services.indexing_service import IndexingService
from tests.conftest import FakeEmbeddingService, make_image, make_product


async def store(session: AsyncSession, *products: Product) -> list[Product]:
    """Persist products without indexing them."""
    for product in products:
        session.add(product)
    await session.flush()
    return list(products)


class TestIndexing:
    async def test_indexes_both_vectors_when_an_image_exists(
        self,
        session: AsyncSession,
        indexing_service: IndexingService,
        vectors: VectorRepository,
        image_root: Path,
    ):
        make_image((10, 20, 30)).save(image_root / "a.jpg", format="JPEG")
        product = make_product("Indexed Shoe", external_id="i1", image_url="a.jpg")
        await store(session, product)

        report = await indexing_service.index_catalog()
        assert report.embedded == 1
        assert report.image_vectors == 1
        assert report.text_vectors == 1
        assert report.failed == 0

        stored = await vectors.get_vectors(product.id)
        assert set(stored) == {IMAGE_VECTOR, TEXT_VECTOR}

    async def test_records_indexing_state_on_the_row(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        make_image((1, 2, 3)).save(image_root / "b.jpg", format="JPEG")
        product = make_product("Stateful", external_id="i2", image_url="b.jpg")
        await store(session, product)

        await indexing_service.index_catalog()
        assert product.indexed_at is not None
        assert product.content_hash
        assert product.has_image_vector is True
        assert product.has_text_vector is True
        assert product.search_document

    async def test_second_run_does_nothing(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        """The whole point of the content fingerprint."""
        make_image((4, 5, 6)).save(image_root / "c.jpg", format="JPEG")
        await store(session, make_product("Stable", external_id="i3", image_url="c.jpg"))

        first = await indexing_service.index_catalog()
        second = await indexing_service.index_catalog()
        assert first.embedded == 1
        assert second.requested == 0
        assert second.embedded == 0

    async def test_force_recomputes_embeddings(
        self,
        session: AsyncSession,
        indexing_service: IndexingService,
        embedder: FakeEmbeddingService,
        image_root: Path,
    ):
        make_image((7, 8, 9)).save(image_root / "d.jpg", format="JPEG")
        await store(session, make_product("Forced", external_id="i4", image_url="d.jpg"))
        await indexing_service.index_catalog()

        calls_before = embedder.text_calls
        report = await indexing_service.index_catalog(force=True)
        assert report.embedded == 1
        assert embedder.text_calls > calls_before

    async def test_only_changed_products_are_re_embedded(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        for index in range(3):
            make_image((index * 40, 0, 0)).save(image_root / f"e{index}.jpg", format="JPEG")
        products = await store(
            session,
            *[
                make_product(f"Item {i}", external_id=f"m{i}", image_url=f"e{i}.jpg")
                for i in range(3)
            ],
        )
        await indexing_service.index_catalog()

        products[1].colour = "Chartreuse"
        products[1].content_hash = None
        await session.flush()

        report = await indexing_service.index_catalog()
        assert report.requested == 1
        assert report.embedded == 1

    async def test_batches_are_respected(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        """Batch embedding is the reason indexing is fast; verify it is used."""
        for index in range(20):
            make_image((index * 10, index * 5, 0)).save(
                image_root / f"f{index}.jpg", format="JPEG"
            )
        await store(
            session,
            *[
                make_product(f"Batch {i}", external_id=f"b{i}", image_url=f"f{i}.jpg")
                for i in range(20)
            ],
        )
        embedder = indexing_service._embedder
        report = await indexing_service.index_catalog()
        assert report.embedded == 20
        # INGEST_BATCH_SIZE is 8 in tests, so 20 products means 3 batches.
        assert embedder.text_calls == 3

    async def test_progress_callback_is_invoked(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        for index in range(10):
            make_image((index, index, index)).save(image_root / f"g{index}.jpg", format="JPEG")
        await store(
            session,
            *[
                make_product(f"P {i}", external_id=f"g{i}", image_url=f"g{i}.jpg")
                for i in range(10)
            ],
        )
        snapshots = []
        await indexing_service.index_catalog(progress=snapshots.append)
        assert snapshots
        assert snapshots[-1].processed == 10
        assert snapshots[-1].percent == 100.0
        assert snapshots[-1].rate > 0

    async def test_limit_caps_the_work(
        self, session: AsyncSession, indexing_service: IndexingService
    ):
        await store(session, *[make_product(f"L {i}", external_id=f"l{i}") for i in range(5)])
        report = await indexing_service.index_catalog(limit=2)
        assert report.requested == 2

    async def test_recreate_clears_state_and_reindexes(
        self,
        session: AsyncSession,
        indexing_service: IndexingService,
        vectors: VectorRepository,
        image_root: Path,
    ):
        make_image((9, 9, 9)).save(image_root / "h.jpg", format="JPEG")
        product = await store(
            session, make_product("Recreated", external_id="r1", image_url="h.jpg")
        )
        await indexing_service.index_catalog()
        assert await vectors.count() == 1

        report = await indexing_service.index_catalog(recreate=True)
        assert report.embedded == 1
        assert await vectors.count() == 1
        assert product[0].indexed_at is not None


class TestPartialSuccess:
    async def test_product_without_an_image_is_indexed_on_text(
        self,
        session: AsyncSession,
        indexing_service: IndexingService,
        vectors: VectorRepository,
    ):
        """Text search must keep working for products with no usable photo."""
        product = make_product("No Image Product", external_id="n1", image_url=None)
        await store(session, product)

        report = await indexing_service.index_catalog()
        assert report.embedded == 1
        assert report.text_vectors == 1
        assert report.image_vectors == 0
        assert report.failures
        assert "no image_url" in report.failures[0].reason

        stored = await vectors.get_vectors(product.id)
        assert set(stored) == {TEXT_VECTOR}
        assert product.has_text_vector is True
        assert product.has_image_vector is False

    async def test_missing_image_file_is_reported_not_fatal(
        self, session: AsyncSession, indexing_service: IndexingService
    ):
        product = make_product("Broken Path", external_id="n2", image_url="does/not/exist.jpg")
        await store(session, product)

        report = await indexing_service.index_catalog()
        assert report.embedded == 1
        assert report.image_vectors == 0
        assert product.index_error
        assert "not found" in product.index_error

    async def test_corrupt_image_file_is_reported_not_fatal(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        (image_root / "corrupt.jpg").write_bytes(b"this is not a JPEG at all")
        product = make_product("Corrupt Image", external_id="n3", image_url="corrupt.jpg")
        await store(session, product)

        report = await indexing_service.index_catalog()
        assert report.embedded == 1
        assert report.image_vectors == 0
        assert "unreadable" in (product.index_error or "")

    async def test_one_bad_product_does_not_block_the_batch(
        self, session: AsyncSession, indexing_service: IndexingService, image_root: Path
    ):
        make_image((3, 3, 3)).save(image_root / "good.jpg", format="JPEG")
        (image_root / "bad.jpg").write_bytes(b"garbage")
        await store(
            session,
            make_product("Good", external_id="ok1", image_url="good.jpg"),
            make_product("Bad", external_id="bad1", image_url="bad.jpg"),
        )

        report = await indexing_service.index_catalog()
        assert report.embedded == 2  # both got text vectors
        assert report.image_vectors == 1

    async def test_product_with_no_embeddable_text_fails_cleanly(
        self,
        session: AsyncSession,
        indexing_service: IndexingService,
        products: ProductRepository,
    ):
        blank = make_product(
            "x",
            external_id="blank",
            category=None,
            subcategory=None,
            brand=None,
            colour=None,
            gender=None,
            usage=None,
        )
        blank.name = ""
        blank.description = None
        await store(session, blank)

        report = await indexing_service.index_catalog()
        assert report.failed == 1
        assert report.embedded == 0
        assert blank.indexed_at is None
        assert blank.index_error

    async def test_failures_list_is_bounded(
        self, session: AsyncSession, indexing_service: IndexingService
    ):
        """A large broken import must not return a megabyte of failure detail."""
        await store(
            session,
            *[make_product(f"NoImg {i}", external_id=f"ni{i}") for i in range(60)],
        )
        report = await indexing_service.index_catalog()
        assert report.image_vectors == 0
        assert len(report.failures) <= 50


class TestEmbeddingFailure:
    async def test_batch_level_embedding_error_surfaces(
        self, session: AsyncSession, indexing_service: IndexingService, monkeypatch
    ):
        """A broken model is an operational fault, not per-product bad data."""
        from app.core.exceptions import EmbeddingError

        await store(session, make_product("Will Fail", external_id="wf1"))

        async def _boom(*_args, **_kwargs):
            raise EmbeddingError("inference exploded")

        monkeypatch.setattr(indexing_service._embedder, "embed_texts_async", _boom)
        with pytest.raises(IndexingError, match="Embedding failed"):
            await indexing_service.index_catalog()


class TestNoWork:
    async def test_empty_catalogue_reports_nothing_to_do(
        self, indexing_service: IndexingService
    ):
        report = await indexing_service.index_catalog()
        assert report.requested == 0
        assert report.embedded == 0
        assert report.failures == []

    async def test_specific_ids_can_be_targeted(
        self, session: AsyncSession, indexing_service: IndexingService
    ):
        products = await store(
            session, *[make_product(f"T {i}", external_id=f"t{i}") for i in range(4)]
        )
        report = await indexing_service.index_catalog(product_ids=[products[1].id])
        assert report.requested == 1
        assert products[1].indexed_at is not None
        assert products[0].indexed_at is None
