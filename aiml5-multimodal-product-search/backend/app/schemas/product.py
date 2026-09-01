"""Product request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.text import clean_text, normalise_facet

Name = Annotated[str, Field(min_length=1, max_length=512)]
Price = Annotated[float, Field(ge=0, le=1_000_000)]


class ProductBase(BaseModel):
    """Fields shared by create and update payloads."""

    name: Name = Field(description="Display name of the product.")
    description: str | None = Field(default=None, max_length=4000)
    category: str | None = Field(default=None, max_length=128)
    subcategory: str | None = Field(default=None, max_length=128)
    brand: str | None = Field(default=None, max_length=128)
    colour: str | None = Field(default=None, max_length=64)
    gender: str | None = Field(default=None, max_length=32)
    usage: str | None = Field(default=None, max_length=64)
    season: str | None = Field(default=None, max_length=32)
    year: int | None = Field(default=None, ge=1900, le=2200)
    price: Price | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    in_stock: bool = True
    image_url: str | None = Field(
        default=None,
        max_length=1024,
        description="HTTP(S) URL, or a path relative to IMAGE_ROOT, of the product image.",
    )

    @field_validator("name", "description", mode="after")
    @classmethod
    def _clean(cls, v: str | None) -> str | None:
        return clean_text(v) or None if v is not None else None

    @field_validator(
        "category", "subcategory", "brand", "colour", "gender", "usage", "season", mode="after"
    )
    @classmethod
    def _facet(cls, v: str | None) -> str | None:
        return normalise_facet(v) or None if v is not None else None

    @field_validator("currency", mode="after")
    @classmethod
    def _currency(cls, v: str) -> str:
        code = v.strip().upper()
        if not code.isalpha():
            raise ValueError("currency must be a 3-letter alphabetic code")
        return code


class ProductCreate(ProductBase):
    """Payload for creating a product."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "external_id": "sku-10421",
                "name": "Nike Revolution 6 Running Shoe",
                "description": "Lightweight mesh running shoe with foam midsole.",
                "category": "Footwear",
                "subcategory": "Sports Shoes",
                "brand": "Nike",
                "colour": "Black",
                "gender": "Men",
                "usage": "Sports",
                "price": 74.99,
                "image_url": "sample/images/10421.jpg",
            }
        }
    )

    external_id: str | None = Field(
        default=None,
        max_length=128,
        description="Identifier from the source catalogue; must be unique when supplied.",
    )

    @field_validator("name", mode="after")
    @classmethod
    def _name_not_blank(cls, v: str | None) -> str:
        if not v or not v.strip():
            raise ValueError("name must not be blank")
        return v


class ProductUpdate(BaseModel):
    """Partial update payload; unset fields are left unchanged."""

    name: Name | None = None
    description: str | None = Field(default=None, max_length=4000)
    category: str | None = Field(default=None, max_length=128)
    subcategory: str | None = Field(default=None, max_length=128)
    brand: str | None = Field(default=None, max_length=128)
    colour: str | None = Field(default=None, max_length=64)
    gender: str | None = Field(default=None, max_length=32)
    usage: str | None = Field(default=None, max_length=64)
    season: str | None = Field(default=None, max_length=32)
    year: int | None = Field(default=None, ge=1900, le=2200)
    price: Price | None = None
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    in_stock: bool | None = None
    image_url: str | None = Field(default=None, max_length=1024)

    @field_validator("name", "description", mode="after")
    @classmethod
    def _clean(cls, v: str | None) -> str | None:
        return clean_text(v) or None if v is not None else None

    @field_validator(
        "category", "subcategory", "brand", "colour", "gender", "usage", "season", mode="after"
    )
    @classmethod
    def _facet(cls, v: str | None) -> str | None:
        return normalise_facet(v) or None if v is not None else None

    @field_validator("currency", mode="after")
    @classmethod
    def _currency(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v


class IndexingState(BaseModel):
    """Whether a product is currently searchable, and why not if it isn't."""

    indexed_at: datetime | None = None
    has_image_vector: bool = False
    has_text_vector: bool = False
    index_error: str | None = None

    @property
    def is_searchable(self) -> bool:
        """A product is searchable once at least one vector exists."""
        return self.has_image_vector or self.has_text_vector


class ProductRead(ProductBase):
    """Full product representation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    external_id: str | None = None
    search_document: str | None = Field(
        default=None, description="Canonical text that was embedded for this product."
    )
    indexed_at: datetime | None = None
    has_image_vector: bool = False
    has_text_vector: bool = False
    index_error: str | None = None
    created_at: datetime
    updated_at: datetime


class ProductSummary(BaseModel):
    """Compact representation used inside search results and listings."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None = None
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    colour: str | None = None
    gender: str | None = None
    usage: str | None = None
    price: float | None = None
    currency: str = "USD"
    in_stock: bool = True
    image_url: str | None = None


class CatalogFacets(BaseModel):
    """Distinct filter values with counts, for populating the UI filter panel."""

    categories: list[FacetValue]
    brands: list[FacetValue]
    colours: list[FacetValue]
    price_min: float | None = None
    price_max: float | None = None


class FacetValue(BaseModel):
    """One facet option and how many products carry it."""

    value: str
    count: int


CatalogFacets.model_rebuild()
