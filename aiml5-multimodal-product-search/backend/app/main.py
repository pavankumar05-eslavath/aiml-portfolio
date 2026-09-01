"""FastAPI application factory and process lifecycle.

Startup sequence
----------------
1. Configure logging.
2. Ensure the database schema exists.
3. Load the embedding model **in a worker thread**, so the event loop starts
   serving ``/health`` immediately while several hundred megabytes of weights are
   read. Endpoints needing inference return 503 until it finishes.
4. Ensure the Qdrant collection exists, sized from the loaded model.
5. Warm up the model with one tiny forward pass per tower, so the first real
   query does not absorb lazy-initialisation cost.

Startup is *tolerant*: if Qdrant or the model is unavailable the app still boots
and reports the problem through ``/health``. A container that refuses to start
tells an operator far less than one that starts and explains what is broken.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.errors import register_exception_handlers
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, set_request_id
from app.db.session import dispose_engine, init_models
from app.repositories.vector_repository import get_vector_repository
from app.services.embedding import get_embedding_service

logger = get_logger(__name__)

DESCRIPTION = """
Multimodal product search over a fashion catalogue using CLIP-family embeddings.

**Search modes**

* `POST /api/search/text` - natural-language query
* `POST /api/search/image` - upload an image
* `POST /api/search/multimodal` - image *and* text together
* `POST /api/search/similar/{product_id}` - more like an existing product

**How ranking works.** Products are indexed as two named vectors in one Qdrant
collection (`image` and `text`). A query is compared against both, producing up to
four similarity channels plus an optional lexical channel. Because same-modality
cosines (image-image 0.7305 mean) and cross-modal cosines (image-text 0.3124
mean) occupy different ranges, each channel is normalised over its own candidate
pool before the weighted sum, or combined by reciprocal rank fusion. Every
response includes the per-channel breakdown under `results[].breakdown`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of process-wide resources."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    started = time.perf_counter()

    logger.info(
        "starting application",
        extra={
            "context": {
                "env": settings.app_env,
                "model": settings.model_name,
                "database": "postgresql" if settings.is_postgres else "sqlite",
                "qdrant": "server" if settings.uses_qdrant_server else "embedded",
                "fusion": settings.fusion_strategy.value,
                "normalization": settings.score_normalization.value,
            }
        },
    )

    try:
        await init_models()
    except Exception:
        logger.exception("database initialisation failed; /health will report it")

    embedder = get_embedding_service()
    vectors = get_vector_repository()

    async def _load_model() -> None:
        """Load weights off the event loop, then prepare the collection."""
        try:
            info = await anyio.to_thread.run_sync(embedder.load)
        except Exception:
            logger.exception("model failed to load; inference endpoints will return 503")
            return
        try:
            await vectors.ensure_collection(info.embedding_dim)
        except Exception:
            logger.exception("qdrant collection unavailable; /health will report it")
        try:
            await anyio.to_thread.run_sync(embedder.warmup)
        except Exception:
            logger.warning("model warmup failed", exc_info=True)
        logger.info(
            "application ready",
            extra={"context": {"startup_ms": round((time.perf_counter() - started) * 1000, 2)}},
        )

    task = asyncio.create_task(_load_model())
    app.state.model_task = task

    try:
        yield
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Expected when shutting down before the model finished loading.
            logger.debug("model loading cancelled during shutdown")
        except Exception:
            # Never mask a shutdown failure: the app is going down either way,
            # but silently swallowing this would hide a genuine defect.
            logger.warning("model loading task failed during shutdown", exc_info=True)
        await vectors.close()
        await dispose_engine()
        embedder.unload()
        logger.info("shutdown complete")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Project repository", "url": "https://github.com/"},
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a request id and server timing to every response."""
        request_id = set_request_id(request.headers.get("X-Request-ID"))
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
        if request.url.path.startswith(settings.api_prefix):
            logger.debug(
                "request completed",
                extra={
                    "context": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(elapsed, 2),
                    }
                },
            )
        return response

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Send browsers hitting the root to the API documentation."""
        return RedirectResponse(url="/docs")

    return app


app = create_app()
