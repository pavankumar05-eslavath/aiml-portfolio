"""Exception handlers producing a single, consistent JSON error shape.

Every failure - domain error, framework validation error, or unexpected crash -
leaves the API as::

    {"error": {"code": "...", "message": "...", "details": {...}}}

Unhandled exceptions are logged with a traceback server-side and reported to the
client as a generic message carrying the request id. Stack traces are never sent
to clients: they leak file paths and library versions while telling the caller
nothing actionable.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger, get_request_id

logger = get_logger(__name__)


def _json(status_code: int, code: str, message: str, **details: object) -> JSONResponse:
    body: dict[str, object] = {"code": code, "message": message}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content={"error": body})


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the application's exception handlers."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        # Client faults are expected traffic; only server faults deserve a stack.
        if exc.status_code >= 500:
            logger.error(
                "request failed",
                exc_info=exc,
                extra={"context": {"code": exc.code, "status": exc.status_code}},
            )
        else:
            logger.info(
                "request rejected",
                extra={"context": {"code": exc.code, "status": exc.status_code}},
            )
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in error["loc"][1:]) or "body",
                "message": error["msg"],
            }
            for error in exc.errors()[:10]
        ]
        return _json(
            422,
            "validation_error",
            "The request body or parameters are invalid.",
            fields=fields,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Literal codes rather than `status.*` constants: Starlette has renamed
        # some of them across versions, and a deprecated alias raises under the
        # test suite's `filterwarnings = error` policy.
        codes = {
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            415: "unsupported_media_type",
            429: "rate_limited",
        }
        return _json(
            exc.status_code,
            codes.get(exc.status_code, "http_error"),
            str(exc.detail) if exc.detail else "Request failed.",
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.error("unhandled database error", exc_info=exc)
        return _json(
            503,
            "database_unavailable",
            "The metadata database is unavailable. Please retry.",
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        request_id = get_request_id()
        logger.critical("unhandled exception", exc_info=exc)
        return _json(
            500,
            "internal_error",
            "An unexpected error occurred. Quote the request id when reporting this.",
            request_id=request_id,
        )
