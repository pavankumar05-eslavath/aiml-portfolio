"""Catalogue indexing and statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CatalogServiceDep, IndexingServiceDep, SessionDep
from app.core.logging import get_logger
from app.schemas.catalog import CatalogStats, IndexReport, IndexRequest
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get(
    "/stats",
    response_model=CatalogStats,
    summary="Catalogue and index statistics",
)
async def catalog_stats(catalog: CatalogServiceDep) -> CatalogStats:
    """Report catalogue size and how much of it is indexed.

    A gap between ``total_products`` and ``indexed_products`` means products exist
    that search cannot return yet. ``vector_points`` is ``-1`` when the vector
    store could not be reached, so this endpoint stays useful during an outage.
    """
    return await catalog.stats()


@router.post(
    "/index",
    response_model=IndexReport,
    status_code=200,
    summary="Index or re-index the catalogue",
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse, "description": "Indexing failed."},
        503: {"model": ErrorResponse, "description": "Model or vector store unavailable."},
    },
)
async def index_catalog(
    payload: IndexRequest,
    indexer: IndexingServiceDep,
    session: SessionDep,
) -> IndexReport:
    """Embed products and write their vectors.

    Runs **synchronously** and returns a report. That is a deliberate choice for
    this project: the caller learns the real outcome, including per-product
    failures, instead of a job id that has to be polled. A catalogue large enough
    to need a background worker should use ``scripts/index_catalog.py``, which is
    the same service driven from a CLI - see ``docs/architecture.md`` for how this
    would move to a queue.

    By default only products whose content fingerprint changed are re-embedded.
    Pass ``force`` after switching ``MODEL_NAME``.
    """
    report = await indexer.index_catalog(
        product_ids=payload.product_ids, force=payload.force, limit=payload.limit
    )
    await session.commit()
    return report
