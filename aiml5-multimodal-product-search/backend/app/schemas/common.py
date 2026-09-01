"""Shared API schema components."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Machine-readable error body."""

    code: str = Field(description="Stable error identifier, e.g. 'invalid_image'.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, object] | None = Field(
        default=None, description="Optional structured context about the failure."
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "invalid_image",
                    "message": "The uploaded file is not a readable image.",
                    "details": {"content_type": "application/pdf"},
                }
            }
        }
    )

    error: ErrorDetail


class Page(BaseModel, Generic[T]):
    """Offset-paginated collection."""

    items: list[T]
    total: int = Field(description="Total rows matching the filters, ignoring pagination.")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        """Whether further rows exist beyond this page."""
        return self.offset + len(self.items) < self.total
