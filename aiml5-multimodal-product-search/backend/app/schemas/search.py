"""Search request/response schemas.

The response deliberately exposes the full score breakdown rather than a single
opaque number: the API contract for this project is that every ranking decision
is explainable, per channel, to the caller.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import FusionStrategy, ScoreNormalization
from app.schemas.product import ProductSummary


class SearchMode(StrEnum):
    """Which query modalities were supplied."""

    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class SortOption(StrEnum):
    """Ordering applied after fusion."""

    RELEVANCE = "relevance"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"
    NAME_ASC = "name_asc"


class RetrievalChannel(StrEnum):
    """A single source of ranking evidence.

    Named ``<query representation>_to_<product vector>``. ``lexical`` is the
    keyword channel served from the metadata database rather than from Qdrant.
    The ``fused_*`` channels are used only by
    :attr:`~app.core.config.FusionStrategy.EMBEDDING_FUSION`, where the image and
    text queries are blended into a single vector *before* retrieval and can no
    longer be attributed to a modality.
    """

    TEXT_TO_TEXT = "text_to_text"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_TO_TEXT = "image_to_text"
    FUSED_TO_IMAGE = "fused_to_image"
    FUSED_TO_TEXT = "fused_to_text"
    LEXICAL = "lexical"

    @property
    def is_image_side(self) -> bool:
        """Whether this channel carries image-query evidence."""
        return self in (RetrievalChannel.IMAGE_TO_IMAGE, RetrievalChannel.IMAGE_TO_TEXT)

    @property
    def is_text_side(self) -> bool:
        """Whether this channel carries text-query evidence."""
        return self in (RetrievalChannel.TEXT_TO_TEXT, RetrievalChannel.TEXT_TO_IMAGE)

    @property
    def product_vector(self) -> str | None:
        """Name of the product vector this channel searches against."""
        return {
            RetrievalChannel.TEXT_TO_TEXT: "text",
            RetrievalChannel.TEXT_TO_IMAGE: "image",
            RetrievalChannel.IMAGE_TO_IMAGE: "image",
            RetrievalChannel.IMAGE_TO_TEXT: "text",
            RetrievalChannel.FUSED_TO_IMAGE: "image",
            RetrievalChannel.FUSED_TO_TEXT: "text",
            RetrievalChannel.LEXICAL: None,
        }[self]


class SearchFilters(BaseModel):
    """Metadata constraints pushed down into the vector store payload filter."""

    model_config = ConfigDict(extra="forbid")

    categories: list[str] = Field(default_factory=list, max_length=32)
    subcategories: list[str] = Field(default_factory=list, max_length=32)
    brands: list[str] = Field(default_factory=list, max_length=32)
    colours: list[str] = Field(default_factory=list, max_length=32)
    genders: list[str] = Field(default_factory=list, max_length=16)
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    in_stock_only: bool = False

    @model_validator(mode="after")
    def _check_price_range(self) -> Self:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        return self

    def is_empty(self) -> bool:
        """Whether this filter set constrains anything at all."""
        return not (
            self.categories
            or self.subcategories
            or self.brands
            or self.colours
            or self.genders
            or self.in_stock_only
            or self.min_price is not None
            or self.max_price is not None
        )


class FusionOverrides(BaseModel):
    """Per-request overrides of the configured ranking behaviour.

    Exposed so the evaluation harness and the UI can sweep parameters without
    restarting the service. Omitted fields fall back to the environment config.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: FusionStrategy | None = None
    normalization: ScoreNormalization | None = None
    image_weight: Annotated[float, Field(ge=0, le=1)] | None = None
    text_weight: Annotated[float, Field(ge=0, le=1)] | None = None
    lexical_weight: Annotated[float, Field(ge=0, le=1)] | None = None
    cross_modal_weight: Annotated[float, Field(ge=0, le=1)] | None = None
    embedding_fusion_alpha: Annotated[float, Field(ge=0, le=1)] | None = None
    rrf_k: Annotated[int, Field(ge=1, le=1000)] | None = None


class BaseSearchRequest(BaseModel):
    """Common search parameters."""

    model_config = ConfigDict(extra="forbid")

    top_k: Annotated[int, Field(ge=1, le=100)] = 20
    offset: Annotated[int, Field(ge=0, le=10_000)] = 0
    filters: SearchFilters = Field(default_factory=SearchFilters)
    sort: SortOption = SortOption.RELEVANCE
    fusion: FusionOverrides | None = None
    explain: bool = Field(
        default=True, description="Include the per-channel score breakdown in results."
    )


class TextSearchRequest(BaseSearchRequest):
    """Text-only search."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "query": "black running shoes suitable for daily workouts",
                "top_k": 12,
                "filters": {"categories": ["Footwear"], "max_price": 120},
            }
        },
    )

    query: Annotated[str, Field(min_length=1, max_length=400)]


class MultimodalSearchRequest(BaseSearchRequest):
    """Image + optional text search, submitted as a JSON body.

    The image is provided either as a base64 data payload or as a reference to a
    catalogue product (``image_product_id``), which powers "more like this".
    Multipart upload is handled by the ``/search/image`` and
    ``/search/multimodal`` form endpoints instead.
    """

    query: Annotated[str, Field(max_length=400)] | None = None
    image_base64: str | None = Field(default=None, description="Base64-encoded image bytes.")
    image_product_id: str | None = Field(
        default=None, description="Use this catalogue product's image as the query image."
    )

    @model_validator(mode="after")
    def _require_some_input(self) -> Self:
        if not any((self.query, self.image_base64, self.image_product_id)):
            raise ValueError("provide at least one of: query, image_base64, image_product_id")
        if self.image_base64 and self.image_product_id:
            raise ValueError("provide either image_base64 or image_product_id, not both")
        return self


class ChannelScore(BaseModel):
    """One channel's contribution to a result's final score."""

    channel: RetrievalChannel
    raw: float = Field(description="Cosine similarity, or the lexical channel's own score.")
    normalized: float = Field(description="Score after per-channel normalisation.")
    weight: float = Field(description="Weight applied to the normalised score.")
    rank: int | None = Field(
        default=None, description="1-based rank within this channel's candidate list."
    )


class ScoreBreakdown(BaseModel):
    """How a result's final score was assembled."""

    final_score: float
    image_similarity: float | None = Field(
        default=None,
        description="Aggregated image-side similarity (same-modality and cross-modal).",
    )
    text_similarity: float | None = Field(
        default=None, description="Aggregated text-side similarity."
    )
    lexical_score: float | None = None
    channels: list[ChannelScore] = Field(default_factory=list)
    strategy: FusionStrategy
    normalization: ScoreNormalization


class SearchResult(BaseModel):
    """A single ranked product."""

    rank: int
    product: ProductSummary
    score: float
    breakdown: ScoreBreakdown | None = None


class SearchTimings(BaseModel):
    """Server-side latency breakdown in milliseconds."""

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    fuse_ms: float = 0.0
    hydrate_ms: float = 0.0
    total_ms: float = 0.0


class SearchResponse(BaseModel):
    """Search response envelope."""

    mode: SearchMode
    query: str | None = None
    has_image_query: bool = False
    total_candidates: int = Field(
        description="Distinct products retrieved across all channels before ranking."
    )
    returned: int
    top_k: int
    offset: int
    results: list[SearchResult]
    channels_used: list[RetrievalChannel]
    weights: dict[str, float] = Field(
        default_factory=dict, description="Effective per-channel weights for this request."
    )
    strategy: FusionStrategy
    timings: SearchTimings
    model_name: str
    warnings: list[str] = Field(default_factory=list)
