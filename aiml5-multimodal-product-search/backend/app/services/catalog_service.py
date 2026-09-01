"""Catalogue management: CRUD kept consistent with the vector store.

The rule this service enforces is that the database and Qdrant must not drift.
A product deleted through the API has its vectors removed in the same operation;
an edit that changes embedding-relevant text invalidates the fingerprint so the
next indexing run re-embeds it. Route handlers therefore contain no
orchestration logic - they translate HTTP to these calls and back.
"""

from __future__ import annotations

from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.models.product import Product
from app.repositories.product_repository import ProductRepository
from app.repositories.vector_repository import VectorRepository
from app.schemas.catalog import CatalogStats
from app.schemas.product import (
    CatalogFacets,
    FacetValue,
    ProductCreate,
    ProductUpdate,
)
from app.schemas.search import SearchFilters, SortOption
from app.services.embedding import EmbeddingService

logger = get_logger(__name__)


class CatalogService:
    """Create, update, delete and describe catalogue products."""

    def __init__(
        self,
        *,
        products: ProductRepository,
        vectors: VectorRepository,
        embedder: EmbeddingService,
    ) -> None:
        self._products = products
        self._vectors = vectors
        self._embedder = embedder

    async def get(self, product_id: str) -> Product:
        """Fetch a product or raise :class:`~app.core.exceptions.NotFoundError`."""
        return await self._products.get_or_404(product_id)

    async def list_products(
        self,
        *,
        filters: SearchFilters,
        search: str | None,
        sort: SortOption,
        limit: int,
        offset: int,
    ) -> tuple[list[Product], int]:
        """Page through the catalogue with optional filters."""
        return await self._products.list_products(
            filters=filters, search=search, sort=sort, limit=limit, offset=offset
        )

    async def create(self, payload: ProductCreate) -> Product:
        """Create a product.

        The product is *not* embedded here. Indexing is a separate, explicitly
        triggered step so that a burst of writes cannot serialise behind model
        inference, and so a bulk import pays the batched-embedding cost once.
        The product is returned with ``indexed_at`` unset, which the UI surfaces
        as "pending indexing".
        """
        product = await self._products.create(payload)
        logger.info(
            "product created",
            extra={"context": {"product_id": product.id, "name": product.name}},
        )
        return product

    async def update(self, product_id: str, payload: ProductUpdate) -> Product:
        """Apply a partial update.

        When embedding-relevant fields change, the repository clears the content
        hash so the next indexing run re-embeds the product.
        """
        product = await self._products.update(product_id, payload)
        logger.info(
            "product updated",
            extra={
                "context": {
                    "product_id": product.id,
                    "fields": sorted(payload.model_dump(exclude_unset=True)),
                    "reindex_required": product.content_hash is None,
                }
            },
        )
        return product

    async def delete(self, product_id: str) -> None:
        """Delete a product and its vectors.

        Vectors are removed first: an orphaned database row is invisible but
        harmless, whereas an orphaned vector would keep surfacing in search
        results that can no longer be hydrated.
        """
        await self._products.get_or_404(product_id)
        try:
            await self._vectors.delete([product_id])
        except VectorStoreError:
            logger.warning(
                "could not delete vectors; aborting product delete to avoid orphans",
                extra={"context": {"product_id": product_id}},
            )
            raise
        await self._products.delete(product_id)
        logger.info("product deleted", extra={"context": {"product_id": product_id}})

    async def facets(self) -> CatalogFacets:
        """Filter options for the UI, with counts."""
        raw = await self._products.facets()
        return CatalogFacets(
            categories=[FacetValue(value=v, count=c) for v, c in raw["categories"]],
            brands=[FacetValue(value=v, count=c) for v, c in raw["brands"]],
            colours=[FacetValue(value=v, count=c) for v, c in raw["colours"]],
            price_min=raw["price_min"],
            price_max=raw["price_max"],
        )

    async def stats(self) -> CatalogStats:
        """Catalogue and index counters.

        The Qdrant point count is best-effort: the admin view stays usable when
        the vector store is down, reporting ``-1`` rather than failing outright.
        """
        db_stats = await self._products.stats()
        try:
            points = await self._vectors.count()
        except VectorStoreError:
            logger.warning("vector store unavailable while collecting stats")
            points = -1

        return CatalogStats(
            total_products=db_stats["total_products"],
            indexed_products=db_stats["indexed_products"],
            pending_products=db_stats["pending_products"],
            failed_products=db_stats["failed_products"],
            with_image_vector=db_stats["with_image_vector"],
            with_text_vector=db_stats["with_text_vector"],
            vector_points=points,
            collection=self._vectors.collection,
            embedding_dim=(self._embedder.embedding_dim if self._embedder.is_loaded else None),
            model_name=self._embedder.configured_model_name,
            last_indexed_at=db_stats["last_indexed_at"],
        )
