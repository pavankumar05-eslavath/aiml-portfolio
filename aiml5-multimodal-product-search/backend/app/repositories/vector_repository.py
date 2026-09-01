"""Qdrant vector store access.

Collection layout
-----------------
One point per product, carrying two **named vectors** in the same collection:

* ``image`` - the product photo embedding
* ``text``  - the canonical product document embedding

Named vectors (rather than two collections, or one collection of fused vectors)
are what make the ranking layer possible. They keep both representations under a
single point id, so all four query channels are served from one collection with
one set of payload filters, and a product can never end up with its image in the
index but its text missing from a *different* index.

Payload filtering runs inside Qdrant, not in Python. Filters are passed with the
query so the HNSW traversal itself is constrained; retrieving a large candidate
set and discarding most of it in the API process would waste both bandwidth and
recall. The payload fields used for filtering are given explicit indexes, without
which Qdrant would fall back to scanning payloads.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger, log_duration
from app.schemas.search import SearchFilters

logger = get_logger(__name__)

IMAGE_VECTOR: Final = "image"
TEXT_VECTOR: Final = "text"

#: Payload fields that get a Qdrant index, with the schema type to use.
_PAYLOAD_INDEXES: Final[dict[str, models.PayloadSchemaType]] = {
    "category": models.PayloadSchemaType.KEYWORD,
    "subcategory": models.PayloadSchemaType.KEYWORD,
    "brand": models.PayloadSchemaType.KEYWORD,
    "colour": models.PayloadSchemaType.KEYWORD,
    "gender": models.PayloadSchemaType.KEYWORD,
    "price": models.PayloadSchemaType.FLOAT,
    "in_stock": models.PayloadSchemaType.BOOL,
}


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One scored point returned by Qdrant."""

    product_id: str
    score: float
    payload: dict[str, Any]


def build_qdrant_filter(filters: SearchFilters) -> models.Filter | None:
    """Translate API filters into a Qdrant payload filter.

    Multi-valued facets become ``MatchAny`` (OR within a facet); different facets
    are combined with ``must`` (AND across facets), which is the behaviour users
    expect from a faceted search UI.
    """
    if filters.is_empty():
        return None

    conditions: list[models.Condition] = []

    for key, values in (
        ("category", filters.categories),
        ("subcategory", filters.subcategories),
        ("brand", filters.brands),
        ("colour", filters.colours),
        ("gender", filters.genders),
    ):
        if values:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
            )

    if filters.min_price is not None or filters.max_price is not None:
        conditions.append(
            models.FieldCondition(
                key="price",
                range=models.Range(gte=filters.min_price, lte=filters.max_price),
            )
        )

    if filters.in_stock_only:
        conditions.append(
            models.FieldCondition(key="in_stock", match=models.MatchValue(value=True))
        )

    return models.Filter(must=conditions) if conditions else None


class VectorRepository:
    """Async wrapper around the Qdrant collection holding product embeddings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncQdrantClient | None = None
        self._lock = asyncio.Lock()
        self._collection = self._settings.qdrant_collection
        self._ensured_dim: int | None = None

    @property
    def collection(self) -> str:
        """Name of the backing collection."""
        return self._collection

    async def client(self) -> AsyncQdrantClient:
        """Return the shared client, connecting on first use.

        A single client is reused for the process lifetime so the HTTP connection
        pool (server mode) or storage handle (embedded mode) is not rebuilt per
        request.
        """
        client = self._client
        if client is not None:
            return client
        async with self._lock:
            # Double-checked locking: another task may have connected while this
            # one waited for the lock.
            client = self._client
            if client is not None:
                return client
            settings = self._settings
            try:
                if settings.uses_qdrant_server:
                    self._client = AsyncQdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=int(settings.qdrant_timeout),
                    )
                    logger.info(
                        "connected to qdrant server",
                        extra={"context": {"url": settings.qdrant_url}},
                    )
                else:
                    # Embedded mode: Qdrant runs in-process against a local path.
                    # Used for the no-container developer path and for tests.
                    self._client = AsyncQdrantClient(path=settings.qdrant_path)
                    logger.info(
                        "using embedded qdrant",
                        extra={"context": {"path": settings.qdrant_path}},
                    )
            except Exception as exc:
                raise VectorStoreError(f"Could not connect to Qdrant: {exc}") from exc
            return self._client

    async def close(self) -> None:
        """Release the client."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.warning("error while closing qdrant client", exc_info=True)
            self._client = None
            self._ensured_dim = None

    # ------------------------------------------------------------- bootstrap
    async def ensure_collection(self, embedding_dim: int, *, recreate: bool = False) -> bool:
        """Create the collection and payload indexes if needed.

        ``embedding_dim`` comes from the loaded model, never from configuration.
        If an existing collection was built with a different dimensionality the
        mismatch is raised rather than silently tolerated, because writes would
        fail per-point later with a far less obvious error.

        Returns:
            True if the collection was created by this call.
        """
        client = await self.client()
        try:
            exists = await client.collection_exists(self._collection)
            if exists and recreate:
                await client.delete_collection(self._collection)
                logger.warning(
                    "dropped existing collection",
                    extra={"context": {"collection": self._collection}},
                )
                exists = False

            if exists:
                await self._verify_dimensions(client, embedding_dim)
                self._ensured_dim = embedding_dim
                await self._ensure_payload_indexes(client)
                return False

            vector_params = models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=self._settings.qdrant_hnsw_m,
                    ef_construct=self._settings.qdrant_hnsw_ef_construct,
                ),
            )
            await client.create_collection(
                collection_name=self._collection,
                vectors_config={IMAGE_VECTOR: vector_params, TEXT_VECTOR: vector_params},
            )
            await self._ensure_payload_indexes(client)
            self._ensured_dim = embedding_dim
            logger.info(
                "created qdrant collection",
                extra={
                    "context": {
                        "collection": self._collection,
                        "dim": embedding_dim,
                        "vectors": [IMAGE_VECTOR, TEXT_VECTOR],
                    }
                },
            )
            return True
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to prepare Qdrant collection: {exc}") from exc

    async def _verify_dimensions(self, client: AsyncQdrantClient, expected: int) -> None:
        info = await client.get_collection(self._collection)
        params = info.config.params.vectors
        if not isinstance(params, dict):
            raise VectorStoreError(
                f"Collection {self._collection!r} does not use named vectors. "
                "Recreate it with 'python scripts/index_catalog.py --recreate'."
            )
        for name in (IMAGE_VECTOR, TEXT_VECTOR):
            if name not in params:
                raise VectorStoreError(
                    f"Collection {self._collection!r} is missing the {name!r} vector. "
                    "Recreate it with 'python scripts/index_catalog.py --recreate'."
                )
            actual = params[name].size
            if actual != expected:
                raise VectorStoreError(
                    f"Collection {self._collection!r} stores {actual}-d {name} vectors but "
                    f"the model produces {expected}-d. The collection was built with a "
                    "different model; recreate it with "
                    "'python scripts/index_catalog.py --recreate'."
                )

    async def _ensure_payload_indexes(self, client: AsyncQdrantClient) -> None:
        """Create payload indexes, ignoring those that already exist."""
        for field, schema in _PAYLOAD_INDEXES.items():
            try:
                await client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except (UnexpectedResponse, ValueError):
                # Already present: Qdrant server answers 4xx, embedded raises.
                continue
            except Exception:
                logger.warning(
                    "could not create payload index",
                    extra={"context": {"field": field}},
                )

    # ----------------------------------------------------------------- writes
    async def upsert(
        self,
        records: Sequence[tuple[str, dict[str, list[float]], dict[str, Any]]],
        *,
        wait: bool = True,
    ) -> int:
        """Upsert points.

        Args:
            records: ``(product_id, {vector_name: values}, payload)`` triples. A
                record may carry only the ``text`` vector when its image is
                missing, which keeps text search working for products without a
                usable photo.
            wait: Block until the write is visible to search.

        Returns:
            Number of points written.
        """
        if not records:
            return 0
        client = await self.client()
        points = [
            models.PointStruct(
                id=self._point_id(product_id),
                # cast: qdrant types this field with an invariant dict, so a
                # dict[str, list[float]] is rejected despite being valid.
                vector=cast("models.VectorStruct", vectors),
                payload=payload,
            )
            for product_id, vectors, payload in records
        ]
        try:
            # Debug level: one line per batch would drown out indexing progress.
            with log_duration(
                logger, "upserted vectors", level=logging.DEBUG, points=len(points)
            ):
                await client.upsert(collection_name=self._collection, points=points, wait=wait)
        except Exception as exc:
            raise VectorStoreError(f"Failed to write vectors to Qdrant: {exc}") from exc
        return len(points)

    async def delete(self, product_ids: Sequence[str]) -> int:
        """Delete points by product id."""
        if not product_ids:
            return 0
        client = await self.client()
        try:
            await client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(
                    points=[self._point_id(pid) for pid in product_ids]
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete vectors: {exc}") from exc
        return len(product_ids)

    async def drop_collection(self) -> None:
        """Delete the whole collection if it exists."""
        client = await self.client()
        try:
            if await client.collection_exists(self._collection):
                await client.delete_collection(self._collection)
        except Exception as exc:
            raise VectorStoreError(f"Failed to drop collection: {exc}") from exc
        self._ensured_dim = None

    # ------------------------------------------------------------------ reads
    async def search(
        self,
        vector: Sequence[float],
        *,
        using: str,
        limit: int,
        filters: SearchFilters | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorHit]:
        """Nearest-neighbour search against one named vector.

        Args:
            vector: L2-normalised query vector.
            using: ``"image"`` or ``"text"`` - which product vector to search.
            limit: Maximum hits to return.
            filters: Metadata constraints, evaluated inside Qdrant.
            score_threshold: Optional minimum cosine similarity.
        """
        client = await self.client()
        query_filter = build_qdrant_filter(filters) if filters else None
        try:
            response = await client.query_points(
                collection_name=self._collection,
                query=list(vector),
                using=using,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        return [
            VectorHit(
                product_id=str((point.payload or {}).get("product_id") or point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    async def search_many(
        self,
        queries: Mapping[str, tuple[str, Sequence[float]]],
        *,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> dict[str, list[VectorHit]]:
        """Run several named-vector searches concurrently.

        The channels of a multimodal query are independent, so issuing them
        together turns up to four sequential round-trips into one wall-clock wait.

        Args:
            queries: ``{channel_key: (vector_name, query_vector)}``.
            limit: Per-channel candidate limit.
            filters: Metadata constraints applied to every channel.

        Returns:
            ``{channel_key: hits}``, using the caller's keys. A channel that
            fails yields an empty list rather than failing the whole search, so
            one bad channel degrades ranking instead of returning an error.
        """
        if not queries:
            return {}

        async def _one(vector_name: str, vector: Sequence[float]) -> list[VectorHit]:
            return await self.search(vector, using=vector_name, limit=limit, filters=filters)

        keys = list(queries)
        outcomes = await asyncio.gather(
            *(_one(*queries[key]) for key in keys), return_exceptions=True
        )

        results: dict[str, list[VectorHit]] = {}
        failures: list[str] = []
        for key, outcome in zip(keys, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(key)
                results[key] = []
            else:
                results[key] = outcome

        if failures:
            if len(failures) == len(keys):
                raise VectorStoreError("All retrieval channels failed.")
            logger.warning(
                "some retrieval channels failed",
                extra={"context": {"channels": failures}},
            )
        return results

    async def get_vectors(self, product_id: str) -> dict[str, list[float]]:
        """Fetch the stored vectors for one product.

        Used by "more like this", where a catalogue product's own image embedding
        becomes the query and re-embedding it would be wasted compute.
        """
        client = await self.client()
        try:
            points = await client.retrieve(
                collection_name=self._collection,
                ids=[self._point_id(product_id)],
                with_vectors=True,
                with_payload=False,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to retrieve vectors: {exc}") from exc
        if not points:
            return {}
        vectors = points[0].vector
        if not isinstance(vectors, dict):
            return {}
        # Only dense named vectors are written by this project, but the client's
        # type also permits sparse and multi-vectors; skip anything else.
        dense: dict[str, list[float]] = {}
        for name, values in vectors.items():
            if isinstance(values, list) and all(isinstance(v, float | int) for v in values):
                dense[name] = [float(v) for v in values]  # type: ignore[arg-type]
        return dense

    async def count(self) -> int:
        """Number of points in the collection (0 when it does not exist)."""
        client = await self.client()
        try:
            if not await client.collection_exists(self._collection):
                return 0
            return int((await client.count(self._collection, exact=True)).count)
        except Exception as exc:
            raise VectorStoreError(f"Failed to count vectors: {exc}") from exc

    async def health(self) -> dict[str, Any]:
        """Return liveness information, raising on failure."""
        client = await self.client()
        try:
            exists = await client.collection_exists(self._collection)
            points = (
                int((await client.count(self._collection, exact=True)).count) if exists else 0
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unreachable: {exc}") from exc
        return {
            "collection": self._collection,
            "exists": exists,
            "points": points,
            "mode": "server" if self._settings.uses_qdrant_server else "embedded",
        }

    @staticmethod
    def _point_id(product_id: str) -> str:
        """Qdrant point id for a product.

        Product ids are UUID strings, which Qdrant accepts directly; using them
        verbatim keeps the vector store and the database trivially joinable.
        """
        return product_id


_repository: VectorRepository | None = None


def get_vector_repository() -> VectorRepository:
    """Return the process-wide vector repository."""
    global _repository
    if _repository is None:
        _repository = VectorRepository()
    return _repository


async def reset_vector_repository() -> None:
    """Close and drop the singleton. Used by tests."""
    global _repository
    if _repository is not None:
        await _repository.close()
    _repository = None
