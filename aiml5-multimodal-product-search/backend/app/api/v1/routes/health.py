"""Health and readiness endpoints."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Response

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.db.session import check_database
from app.repositories.vector_repository import get_vector_repository
from app.schemas.catalog import DependencyHealth, HealthResponse
from app.services.embedding import get_embedding_service

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and dependency health",
    responses={503: {"description": "One or more dependencies are unavailable."}},
)
async def health(response: Response) -> HealthResponse:
    """Report the status of the model, the database and the vector store.

    Returns HTTP 200 when everything works, and 503 when a hard dependency is
    down, so container orchestrators can act on the status code alone. The model
    still loading counts as *degraded* rather than unavailable: metadata browsing
    works without it.
    """
    settings = get_settings()
    embedder = get_embedding_service()
    dependencies: list[DependencyHealth] = []

    started = time.perf_counter()
    try:
        await check_database()
    except AppError as exc:
        dependencies.append(
            DependencyHealth(name="database", status="unavailable", detail=exc.message)
        )
    else:
        dependencies.append(
            DependencyHealth(
                name="database",
                status="ok",
                detail="postgresql" if settings.is_postgres else "sqlite",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

    started = time.perf_counter()
    try:
        info = await get_vector_repository().health()
    except AppError as exc:
        dependencies.append(
            DependencyHealth(name="qdrant", status="unavailable", detail=exc.message)
        )
    else:
        empty = info["exists"] and info["points"] == 0
        dependencies.append(
            DependencyHealth(
                name="qdrant",
                status="degraded" if (not info["exists"] or empty) else "ok",
                detail=(
                    f"collection {info['collection']!r} missing; run the indexer"
                    if not info["exists"]
                    else f"{info['points']} points ({info['mode']} mode)"
                ),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

    dependencies.append(
        DependencyHealth(
            name="embedding_model",
            status="ok" if embedder.is_loaded else "degraded",
            detail=(
                f"{embedder.info.name} on {embedder.info.device}"
                if embedder.is_loaded
                else "not loaded"
            ),
        )
    )

    overall: Literal["ok", "degraded", "unavailable"]
    if any(d.status == "unavailable" for d in dependencies):
        overall = "unavailable"
        response.status_code = 503
    elif any(d.status == "degraded" for d in dependencies):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        version=API_VERSION,
        environment=settings.app_env,
        model_name=embedder.configured_model_name,
        model_loaded=embedder.is_loaded,
        embedding_dim=embedder.embedding_dim if embedder.is_loaded else None,
        device=embedder.info.device if embedder.is_loaded else None,
        dependencies=dependencies,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    status_code=200,
)
async def liveness() -> dict[str, str]:
    """Cheap liveness check that touches no dependency."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    responses={503: {"description": "Not ready to serve search traffic."}},
)
async def readiness(response: Response) -> dict[str, object]:
    """Report whether the service can serve *search* traffic.

    Distinct from liveness: readiness requires the model to be loaded, so a
    rolling deploy does not send queries to an instance still reading weights.
    """
    embedder = get_embedding_service()
    checks = {"model_loaded": embedder.is_loaded}
    try:
        await check_database()
        checks["database"] = True
    except AppError:
        checks["database"] = False
    try:
        await get_vector_repository().health()
        checks["qdrant"] = True
    except AppError:
        checks["qdrant"] = False

    ready = all(checks.values())
    if not ready:
        response.status_code = 503
    return {"ready": ready, "checks": checks}
