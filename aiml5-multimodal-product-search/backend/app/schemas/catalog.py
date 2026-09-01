"""Catalogue indexing and health schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class IndexRequest(BaseModel):
    """Request to (re)index catalogue products."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"force": False, "limit": None}},
    )

    product_ids: list[str] | None = Field(
        default=None,
        max_length=1000,
        description="Restrict indexing to these products. Omit to index the whole catalogue.",
    )
    force: bool = Field(
        default=False,
        description=(
            "Re-embed even when the stored content hash is unchanged. Use after "
            "switching MODEL_NAME."
        ),
    )
    limit: Annotated[int, Field(ge=1, le=100_000)] | None = Field(
        default=None, description="Process at most this many products."
    )


class IndexFailure(BaseModel):
    """A product that could not be indexed, and why."""

    product_id: str
    name: str | None = None
    reason: str


class IndexReport(BaseModel):
    """Outcome of an indexing run.

    Note on ``requested`` vs ``skipped_unchanged``: on the default incremental path
    products whose fingerprint is unchanged are excluded by the database query, so
    they are never loaded and do not appear in **either** count - ``requested`` is
    the size of the pending set, not of the catalogue. ``skipped_unchanged`` counts
    only products that were loaded and then found to be unchanged, which happens
    when specific ``product_ids`` are supplied and bypass that prefilter.

    A run that finds nothing to do therefore reports ``requested=0``, which is the
    signal that the index is already up to date.
    """

    requested: int = Field(
        description="Products loaded as candidates for indexing (the pending set)."
    )
    embedded: int = Field(description="Products for which new embeddings were computed.")
    skipped_unchanged: int = Field(
        description=(
            "Loaded products skipped because their content fingerprint already "
            "matched. Zero on the default path, where such products are excluded "
            "before loading."
        )
    )
    failed: int
    image_vectors: int = Field(description="Products that ended up with an image vector.")
    text_vectors: int = Field(description="Products that ended up with a text vector.")
    duration_ms: float
    failures: list[IndexFailure] = Field(default_factory=list)


class CatalogStats(BaseModel):
    """Aggregate catalogue and index state, used by the admin UI."""

    total_products: int
    indexed_products: int
    pending_products: int
    failed_products: int
    with_image_vector: int
    with_text_vector: int
    vector_points: int = Field(description="Points currently stored in Qdrant.")
    collection: str
    embedding_dim: int | None = None
    model_name: str
    last_indexed_at: datetime | None = None


class DependencyHealth(BaseModel):
    """Health of one downstream dependency."""

    name: str
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Aggregate service health."""

    status: Literal["ok", "degraded", "unavailable"]
    version: str
    environment: str
    model_name: str
    model_loaded: bool
    embedding_dim: int | None = None
    device: str | None = None
    dependencies: list[DependencyHealth] = Field(default_factory=list)
