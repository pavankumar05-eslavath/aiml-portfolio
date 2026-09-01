"""Multimodal search orchestration.

Pipeline for one request::

    validate -> embed query -> fan out retrieval channels (concurrent)
             -> fuse & rank -> filter -> paginate -> hydrate metadata -> respond

Design points
-------------
* **Filters are pushed into Qdrant**, not applied afterwards, so the candidate
  pool is already constrained (see
  :func:`app.repositories.vector_repository.build_qdrant_filter`).
* **Over-fetch before fusion.** Each channel retrieves ``top_k *
  CANDIDATE_MULTIPLIER`` candidates. Fusion can only promote a product that at
  least one channel surfaced, so a pool of exactly ``top_k`` would make fusion
  unable to rescue items ranked just outside a single channel's cut-off.
* **Metadata is hydrated once**, in a single batched query over the page being
  returned - not per result, and not for the whole candidate pool.
* **Channels degrade rather than fail.** If the lexical channel errors, the
  vector channels still answer.
"""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import FusionStrategy, ScoreNormalization, Settings, get_settings
from app.core.exceptions import (
    AppError,
    EmptyQueryError,
    InvalidImageError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.repositories.product_repository import ProductRepository
from app.repositories.vector_repository import (
    IMAGE_VECTOR,
    TEXT_VECTOR,
    VectorRepository,
)
from app.schemas.product import ProductSummary
from app.schemas.search import (
    FusionOverrides,
    RetrievalChannel,
    SearchFilters,
    SearchMode,
    SearchResponse,
    SearchResult,
    SearchTimings,
    SortOption,
)
from app.services import fusion as fusion_module
from app.services.embedding import EmbeddingService
from app.services.fusion import ChannelResult, FusedResult
from app.services.image_processing import ImageProcessor

logger = get_logger(__name__)


@dataclass(slots=True)
class EffectiveFusionConfig:
    """Ranking parameters for a single request, after applying overrides."""

    strategy: FusionStrategy
    normalization: ScoreNormalization
    image_weight: float
    text_weight: float
    lexical_weight: float
    cross_modal_weight: float
    embedding_fusion_alpha: float
    rrf_k: int

    @classmethod
    def resolve(
        cls, settings: Settings, overrides: FusionOverrides | None
    ) -> EffectiveFusionConfig:
        """Layer per-request overrides on top of the environment configuration."""
        o = overrides or FusionOverrides()
        return cls(
            strategy=o.strategy or settings.fusion_strategy,
            normalization=o.normalization or settings.score_normalization,
            image_weight=(settings.image_weight if o.image_weight is None else o.image_weight),
            text_weight=settings.text_weight if o.text_weight is None else o.text_weight,
            lexical_weight=(
                settings.lexical_weight if o.lexical_weight is None else o.lexical_weight
            ),
            cross_modal_weight=(
                settings.cross_modal_weight
                if o.cross_modal_weight is None
                else o.cross_modal_weight
            ),
            embedding_fusion_alpha=(
                settings.embedding_fusion_alpha
                if o.embedding_fusion_alpha is None
                else o.embedding_fusion_alpha
            ),
            rrf_k=settings.rrf_k if o.rrf_k is None else o.rrf_k,
        )


@dataclass(slots=True)
class SearchQuery:
    """A validated, modality-resolved search request."""

    text: str | None = None
    image: Image.Image | None = None
    image_product_id: str | None = None
    top_k: int = 20
    offset: int = 0
    filters: SearchFilters = field(default_factory=SearchFilters)
    sort: SortOption = SortOption.RELEVANCE
    overrides: FusionOverrides | None = None
    explain: bool = True

    @property
    def has_text(self) -> bool:
        """Whether a usable text query was supplied."""
        return bool(self.text and self.text.strip())

    @property
    def has_image(self) -> bool:
        """Whether a usable image query was supplied."""
        return self.image is not None or self.image_product_id is not None

    @property
    def mode(self) -> SearchMode:
        """Which search mode this query represents."""
        if self.has_text and self.has_image:
            return SearchMode.MULTIMODAL
        return SearchMode.IMAGE if self.has_image else SearchMode.TEXT


def decode_base64_image(payload: str) -> bytes:
    """Decode a base64 image payload, tolerating a ``data:`` URL prefix.

    Raises:
        InvalidImageError: if the payload is not valid base64.
    """
    cleaned = payload.strip()
    if cleaned.startswith("data:"):
        _, _, cleaned = cleaned.partition(",")
    try:
        return base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("image_base64 is not valid base64 data.") from exc


class SearchService:
    """Executes text, image and multimodal search against the catalogue."""

    def __init__(
        self,
        *,
        products: ProductRepository,
        vectors: VectorRepository,
        embedder: EmbeddingService,
        images: ImageProcessor,
        settings: Settings | None = None,
    ) -> None:
        self._products = products
        self._vectors = vectors
        self._embedder = embedder
        self._images = images
        self._settings = settings or get_settings()

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Run a search and return ranked, hydrated results.

        Raises:
            EmptyQueryError: neither text nor image was supplied.
            ValidationError: ``top_k`` exceeds the configured maximum.
        """
        started = time.perf_counter()
        timings = SearchTimings()
        warnings: list[str] = []

        if not (query.has_text or query.has_image):
            raise EmptyQueryError("Provide a text query, an image, or both.")
        if query.top_k > self._settings.max_top_k:
            raise ValidationError(
                f"top_k must not exceed {self._settings.max_top_k}.",
                details={"max_top_k": self._settings.max_top_k},
            )

        config = EffectiveFusionConfig.resolve(self._settings, query.overrides)
        requested_mode = query.mode

        if (
            config.strategy is FusionStrategy.EMBEDDING_FUSION
            and requested_mode is not SearchMode.MULTIMODAL
        ):
            # With a single modality there is nothing to blend; score-level
            # fusion over that modality's channels is the equivalent behaviour.
            warnings.append(
                "embedding_fusion requires both an image and a text query; "
                "fell back to weighted_sum for this single-modality request."
            )
            config.strategy = FusionStrategy.WEIGHTED_SUM

        # ---------------------------------------------------------- embedding
        embed_started = time.perf_counter()
        query_vectors = await self._build_query_vectors(query, config, warnings)
        timings.embed_ms = _elapsed_ms(embed_started)

        if not query_vectors:
            raise EmptyQueryError("The query produced no usable embedding.")

        # Derive the *actual* modalities from the vectors we managed to build, not
        # from what was requested. A "more like this" query against a product with
        # no indexed image legitimately degrades to a text-side query, and the
        # weights and the reported mode must both reflect that.
        has_image_side = RetrievalChannel.IMAGE_TO_IMAGE in query_vectors
        has_text_side = RetrievalChannel.TEXT_TO_TEXT in query_vectors
        has_fused = RetrievalChannel.FUSED_TO_IMAGE in query_vectors
        if has_fused or (has_image_side and has_text_side):
            mode = SearchMode.MULTIMODAL
        elif has_image_side:
            mode = SearchMode.IMAGE
        else:
            mode = SearchMode.TEXT

        weights = fusion_module.resolve_channel_weights(
            mode=mode,
            strategy=config.strategy,
            image_weight=config.image_weight,
            text_weight=config.text_weight,
            lexical_weight=config.lexical_weight,
            cross_modal_weight=config.cross_modal_weight,
            has_text=has_text_side or has_fused,
            has_image=has_image_side or has_fused,
        )
        # Keep only channels we hold a query vector for; the lexical channel needs
        # the raw query string rather than a vector.
        weights = {
            channel: weight
            for channel, weight in weights.items()
            if channel is RetrievalChannel.LEXICAL or channel in query_vectors
        }
        if RetrievalChannel.LEXICAL in weights and not query.has_text:
            del weights[RetrievalChannel.LEXICAL]
        if not weights:
            raise EmptyQueryError("No retrieval channel is available for this query.")
        total_weight = sum(weights.values())
        weights = {channel: weight / total_weight for channel, weight in weights.items()}

        # ---------------------------------------------------------- retrieval
        retrieve_started = time.perf_counter()
        pool_size = self._settings.candidate_pool_size(query.top_k + query.offset)
        channel_results = await self._retrieve(query, weights, query_vectors, pool_size)
        timings.retrieve_ms = _elapsed_ms(retrieve_started)

        # --------------------------------------------------------------- fuse
        fuse_started = time.perf_counter()
        fused = fusion_module.fuse(
            channel_results,
            weights,
            strategy=config.strategy,
            normalization=config.normalization,
            rrf_k=config.rrf_k,
            explain=query.explain,
        )
        timings.fuse_ms = _elapsed_ms(fuse_started)
        total_candidates = len(fused)

        if query.image_product_id:
            # "More like this" should not rank the query product against itself.
            fused = [item for item in fused if item.product_id != query.image_product_id]

        # ------------------------------------------------------------ hydrate
        hydrate_started = time.perf_counter()
        page, warnings_from_hydration = await self._hydrate_page(query, fused)
        warnings.extend(warnings_from_hydration)
        timings.hydrate_ms = _elapsed_ms(hydrate_started)
        timings.total_ms = _elapsed_ms(started)

        if not page:
            warnings.append(
                "No products matched this query with the current filters."
                if not query.filters.is_empty()
                else "No products matched this query."
            )

        results = [
            SearchResult(
                rank=index,
                product=item.product,
                score=round(item.fused.final_score, 6),
                breakdown=(
                    item.fused.to_breakdown(config.strategy, config.normalization)
                    if query.explain
                    else None
                ),
            )
            for index, item in enumerate(page, start=query.offset + 1)
        ]

        logger.info(
            "search completed",
            extra={
                "context": {
                    "mode": mode.value,
                    "strategy": config.strategy.value,
                    "channels": [c.value for c in weights],
                    "candidates": total_candidates,
                    "returned": len(results),
                    "top_k": query.top_k,
                    "filtered": not query.filters.is_empty(),
                    "total_ms": timings.total_ms,
                    "embed_ms": timings.embed_ms,
                    "retrieve_ms": timings.retrieve_ms,
                }
            },
        )

        return SearchResponse(
            mode=mode,
            query=query.text,
            has_image_query=query.has_image,
            total_candidates=total_candidates,
            returned=len(results),
            top_k=query.top_k,
            offset=query.offset,
            results=results,
            channels_used=list(weights),
            weights={channel.value: round(w, 6) for channel, w in weights.items()},
            strategy=config.strategy,
            timings=timings,
            model_name=self._embedder.info.name,
            warnings=warnings,
        )

    # ------------------------------------------------------------- embedding
    async def _build_query_vectors(
        self,
        query: SearchQuery,
        config: EffectiveFusionConfig,
        warnings: list[str],
    ) -> dict[RetrievalChannel, np.ndarray]:
        """Produce one query vector per active channel.

        A text query is embedded once and reused for both the ``text_to_text`` and
        ``text_to_image`` channels: the encoder output does not depend on which
        product vector it will be compared against.
        """
        vectors: dict[RetrievalChannel, np.ndarray] = {}

        text_vector: np.ndarray | None = None
        # `has_text` already guarantees a non-blank string; binding it explicitly
        # narrows the type without relying on `assert`, which `python -O` strips.
        query_text = (query.text or "").strip()
        if query_text:
            text_vector = await self._embedder.embed_text(query_text)

        image_vector: np.ndarray | None = None
        if query.image is not None:
            image_vector = await self._embedder.embed_image(query.image)
        elif query.image_product_id:
            image_vector, fallback_text = await self._vectors_for_product(
                query.image_product_id, warnings
            )
            # A product indexed on text only can still seed a "more like this"
            # query through its text vector, which beats refusing to answer.
            if image_vector is None and fallback_text is not None and text_vector is None:
                text_vector = fallback_text

        if (
            config.strategy is FusionStrategy.EMBEDDING_FUSION
            and image_vector is not None
            and text_vector is not None
        ):
            fused_vector = self._embedder.fuse(
                image_vector, text_vector, config.embedding_fusion_alpha
            )
            vectors[RetrievalChannel.FUSED_TO_IMAGE] = fused_vector
            vectors[RetrievalChannel.FUSED_TO_TEXT] = fused_vector
            return vectors

        if image_vector is not None:
            vectors[RetrievalChannel.IMAGE_TO_IMAGE] = image_vector
            vectors[RetrievalChannel.IMAGE_TO_TEXT] = image_vector
        if text_vector is not None:
            vectors[RetrievalChannel.TEXT_TO_TEXT] = text_vector
            vectors[RetrievalChannel.TEXT_TO_IMAGE] = text_vector
        return vectors

    async def _vectors_for_product(
        self, product_id: str, warnings: list[str]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reuse a catalogue product's stored embeddings as the query.

        Preference order for the image side: the stored image vector (free and
        exactly reproducible), then re-embedding the product's image file, then
        nothing. The stored text vector is returned alongside so the caller can
        fall back to a text-side query for products indexed on text only.

        Returns:
            ``(image_vector, text_vector)``, either of which may be ``None``.

        Raises:
            NotFoundError: if the product does not exist.
        """
        product = await self._products.get(product_id)
        if product is None:
            raise NotFoundError(f"No product with id {product_id!r}.")

        stored = await self._vectors.get_vectors(product_id)
        text_vector = (
            np.asarray(stored[TEXT_VECTOR], dtype=np.float32)
            if stored.get(TEXT_VECTOR)
            else None
        )

        if values := stored.get(IMAGE_VECTOR):
            return np.asarray(values, dtype=np.float32), text_vector

        reference = product.image_path or product.image_url
        if reference and (path := self._images.resolve_catalog_path(reference)):
            try:
                loaded = self._images.load_from_path(path)
            except AppError:
                logger.warning(
                    "could not load catalogue image for query",
                    extra={"context": {"product_id": product_id}},
                )
            else:
                return await self._embedder.embed_image(loaded.image), text_vector

        if text_vector is not None:
            warnings.append(
                f"Product {product_id} has no indexed image; matched on its "
                "description instead."
            )
        else:
            warnings.append(
                f"Product {product_id} has not been indexed yet; run catalogue "
                "indexing before searching for similar products."
            )
        return None, text_vector

    # ------------------------------------------------------------- retrieval
    async def _retrieve(
        self,
        query: SearchQuery,
        weights: dict[RetrievalChannel, float],
        query_vectors: dict[RetrievalChannel, np.ndarray],
        pool_size: int,
    ) -> list[ChannelResult]:
        """Fan out to every active channel and collect candidate lists."""
        vector_queries: dict[str, tuple[str, Sequence[float]]] = {}
        for channel in weights:
            if channel is RetrievalChannel.LEXICAL:
                continue
            vector = query_vectors.get(channel)
            if vector is None:
                continue
            vector_name = channel.product_vector
            # Only the lexical channel has no product vector, and it is skipped
            # above; this guard keeps the invariant explicit rather than asserted.
            if vector_name not in (IMAGE_VECTOR, TEXT_VECTOR):
                continue
            vector_queries[channel.value] = (vector_name, vector.tolist())

        hits = await self._vectors.search_many(
            vector_queries, limit=pool_size, filters=query.filters
        )

        results: list[ChannelResult] = [
            ChannelResult.from_scores(
                RetrievalChannel(key), [(hit.product_id, hit.score) for hit in channel_hits]
            )
            for key, channel_hits in hits.items()
        ]

        if RetrievalChannel.LEXICAL in weights and query.text:
            try:
                lexical = await self._products.lexical_search(
                    query.text, filters=query.filters, limit=pool_size
                )
            except AppError:
                # A keyword-channel outage must not take down semantic search.
                logger.warning("lexical channel unavailable; continuing without it")
            else:
                results.append(ChannelResult.from_scores(RetrievalChannel.LEXICAL, lexical))

        return results

    # --------------------------------------------------------------- hydrate
    async def _hydrate_page(
        self, query: SearchQuery, fused: list[FusedResult]
    ) -> tuple[list[_HydratedResult], list[str]]:
        """Apply sorting and pagination, then batch-load metadata for the page.

        Products present in Qdrant but missing from the database are dropped and
        reported as a warning: that combination means the two stores have drifted,
        which the operator should know about.
        """
        warnings: list[str] = []
        if not fused:
            return [], warnings

        if query.sort is SortOption.RELEVANCE:
            window = fused[query.offset : query.offset + query.top_k]
            products = await self._products.get_many([item.product_id for item in window])
            hydrated = [
                _HydratedResult(
                    fused=item, product=ProductSummary.model_validate(products[item.product_id])
                )
                for item in window
                if item.product_id in products
            ]
            if len(hydrated) < len(window):
                warnings.append(
                    f"{len(window) - len(hydrated)} result(s) were dropped because their "
                    "metadata is missing; the vector store and database are out of sync."
                )
            return hydrated, warnings

        # Non-relevance sorts must order the whole candidate set before slicing,
        # so metadata is needed for all candidates rather than just one page.
        products = await self._products.get_many([item.product_id for item in fused])
        enriched = [
            _HydratedResult(
                fused=item, product=ProductSummary.model_validate(products[item.product_id])
            )
            for item in fused
            if item.product_id in products
        ]
        enriched.sort(key=_sort_key(query.sort))
        return enriched[query.offset : query.offset + query.top_k], warnings


@dataclass(slots=True)
class _HydratedResult:
    """A fused result paired with its product metadata."""

    fused: FusedResult
    product: ProductSummary


def _sort_key(sort: SortOption) -> Any:
    """Build a sort key for post-fusion ordering.

    Products without a price sort last in both directions rather than being
    treated as free or as infinitely expensive.
    """
    match sort:
        case SortOption.PRICE_ASC:
            return lambda item: (
                item.product.price is None,
                item.product.price or 0.0,
                item.product.id,
            )
        case SortOption.PRICE_DESC:
            return lambda item: (
                item.product.price is None,
                -(item.product.price or 0.0),
                item.product.id,
            )
        case SortOption.NAME_ASC:
            return lambda item: (item.product.name.casefold(), item.product.id)
        case _:
            return lambda item: (-item.fused.final_score, item.product.id)


def _elapsed_ms(since: float) -> float:
    """Milliseconds elapsed since a ``perf_counter`` reading."""
    return round((time.perf_counter() - since) * 1000, 2)
