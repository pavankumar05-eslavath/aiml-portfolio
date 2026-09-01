"""SQLAlchemy ORM model for catalogue products.

Design notes
------------
* ``id`` is a UUID string rather than an autoincrement integer. The same
  identifier keys the Postgres row and the Qdrant point, so it must be
  generatable client-side before either write happens.
* ``external_id`` preserves the source dataset's identifier and is uniquely
  constrained, which is how re-ingesting the same dataset becomes idempotent
  instead of duplicating the catalogue.
* ``content_hash`` fingerprints the embedding-relevant fields. Ingestion
  compares it to decide whether an existing product needs re-embedding, which is
  what makes incremental indexing cheap.
* ``indexed_at`` / ``index_error`` record the outcome of the last indexing
  attempt so failures are queryable rather than only present in logs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_product_id() -> str:
    """Generate a fresh product identifier."""
    return str(uuid.uuid4())


class Product(Base):
    """A single catalogue item and its indexing state."""

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_product_id)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)

    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    category: Mapped[str | None] = mapped_column(String(128), index=True)
    subcategory: Mapped[str | None] = mapped_column(String(128), index=True)
    brand: Mapped[str | None] = mapped_column(String(128), index=True)
    colour: Mapped[str | None] = mapped_column(String(64), index=True)
    gender: Mapped[str | None] = mapped_column(String(32))
    usage: Mapped[str | None] = mapped_column(String(64))
    season: Mapped[str | None] = mapped_column(String(32))
    year: Mapped[int | None] = mapped_column(Integer)

    price: Mapped[float | None] = mapped_column(Float, index=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    in_stock: Mapped[bool] = mapped_column(default=True, nullable=False)

    #: Public URL or path (relative to ``IMAGE_ROOT``) of the product image.
    image_url: Mapped[str | None] = mapped_column(String(1024))
    #: Local path actually used for embedding, resolved during ingestion.
    image_path: Mapped[str | None] = mapped_column(String(1024))

    #: The canonical document that was embedded, persisted for explainability.
    search_document: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    index_error: Mapped[str | None] = mapped_column(Text)
    has_image_vector: Mapped[bool] = mapped_column(default=False, nullable=False)
    has_text_vector: Mapped[bool] = mapped_column(default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("price IS NULL OR price >= 0", name="ck_products_price_non_negative"),
        # Supports the common "filter by category then order by price" browse path.
        Index("ix_products_category_price", "category", "price"),
        Index("ix_products_brand_price", "brand", "price"),
        # Lets the indexer find pending work without a full scan.
        Index("ix_products_indexed_at", "indexed_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Product id={self.id!r} name={self.name!r}>"

    def vector_payload(self) -> dict[str, Any]:
        """Metadata mirrored into the Qdrant payload.

        Only fields used for filtering or for rendering a result card without a
        database round-trip are duplicated; the database remains the system of
        record for everything else.
        """
        return {
            "product_id": self.id,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "colour": self.colour,
            "gender": self.gender,
            "usage": self.usage,
            "season": self.season,
            "price": self.price,
            "currency": self.currency,
            "in_stock": self.in_stock,
            "image_url": self.image_url,
        }
