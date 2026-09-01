"""Application exception hierarchy.

Services raise these domain errors; a single set of FastAPI exception handlers
(see :mod:`app.api.errors`) turns them into consistent JSON payloads. Service
and repository code therefore never imports ``HTTPException``, which keeps the
business logic independent of the transport layer.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected application errors.

    Attributes:
        message: Human-readable, safe to return to the client.
        status_code: HTTP status the API layer should respond with.
        code: Stable machine-readable identifier for clients.
        details: Optional structured context (never include secrets).
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        """Render the error as the API's JSON error body."""
        body: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details:
            body["error"]["details"] = self.details
        return body


# --------------------------------------------------------------- 4xx errors
class ValidationError(AppError):
    """Caller supplied semantically invalid input."""

    status_code = 422
    code = "validation_error"


class InvalidImageError(ValidationError):
    """Uploaded bytes are not a usable image."""

    code = "invalid_image"


class ImageTooLargeError(ValidationError):
    """Uploaded image exceeds the configured size budget."""

    status_code = 413
    code = "image_too_large"


class EmptyQueryError(ValidationError):
    """Request carried neither usable text nor a usable image."""

    code = "empty_query"


class NotFoundError(AppError):
    """Requested entity does not exist."""

    status_code = 404
    code = "not_found"


class DuplicateProductError(AppError):
    """A product with the same identity already exists."""

    status_code = 409
    code = "duplicate_product"


# --------------------------------------------------------------- 5xx errors
class ServiceUnavailableError(AppError):
    """A required downstream dependency is unreachable."""

    status_code = 503
    code = "service_unavailable"


class VectorStoreError(ServiceUnavailableError):
    """Qdrant is unreachable or rejected an operation."""

    code = "vector_store_unavailable"


class DatabaseError(ServiceUnavailableError):
    """The metadata database is unreachable or rejected an operation."""

    code = "database_unavailable"


class ModelNotReadyError(ServiceUnavailableError):
    """The embedding model has not finished loading, or failed to load."""

    code = "model_not_ready"


class EmbeddingError(AppError):
    """Inference failed while producing an embedding."""

    status_code = 500
    code = "embedding_failed"


class IndexingError(AppError):
    """The ingestion pipeline failed in a way the caller should know about."""

    status_code = 500
    code = "indexing_failed"
