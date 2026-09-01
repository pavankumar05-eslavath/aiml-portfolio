"""FastAPI dependency wiring.

Services are constructed per request around a per-request database session, while
the expensive collaborators (the loaded model, the Qdrant client) are
process-wide singletons. This is the boundary that keeps model loading out of the
request path.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelNotReadyError
from app.db.session import get_db_session
from app.repositories.product_repository import ProductRepository
from app.repositories.vector_repository import VectorRepository, get_vector_repository
from app.services.catalog_service import CatalogService
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.image_processing import ImageProcessor, get_image_processor
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_product_repository(session: SessionDep) -> ProductRepository:
    """Product repository bound to the request's session."""
    return ProductRepository(session)


ProductRepositoryDep = Annotated[ProductRepository, Depends(get_product_repository)]
VectorRepositoryDep = Annotated[VectorRepository, Depends(get_vector_repository)]
ImageProcessorDep = Annotated[ImageProcessor, Depends(get_image_processor)]


def get_ready_embedding_service() -> EmbeddingService:
    """Return the embedding service, refusing the request if it is not loaded.

    Endpoints that need inference depend on this rather than on the raw service,
    so a request arriving during startup gets a clean 503 instead of blocking the
    event loop while several hundred megabytes of weights load.
    """
    service = get_embedding_service()
    if not service.is_loaded:
        raise ModelNotReadyError(
            "The embedding model is still loading. Retry shortly.",
        )
    return service


EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_ready_embedding_service)]


def get_search_service(
    products: ProductRepositoryDep,
    vectors: VectorRepositoryDep,
    embedder: EmbeddingServiceDep,
    images: ImageProcessorDep,
    settings: SettingsDep,
) -> SearchService:
    """Search service for the current request."""
    return SearchService(
        products=products,
        vectors=vectors,
        embedder=embedder,
        images=images,
        settings=settings,
    )


def get_catalog_service(
    products: ProductRepositoryDep,
    vectors: VectorRepositoryDep,
) -> CatalogService:
    """Catalogue service for the current request.

    Depends on the *unloaded-tolerant* embedding service so that listing and
    editing products keeps working while the model is still loading.
    """
    return CatalogService(products=products, vectors=vectors, embedder=get_embedding_service())


def get_indexing_service(
    products: ProductRepositoryDep,
    vectors: VectorRepositoryDep,
    embedder: EmbeddingServiceDep,
    images: ImageProcessorDep,
    settings: SettingsDep,
) -> IndexingService:
    """Indexing service for the current request."""
    return IndexingService(
        products=products,
        vectors=vectors,
        embedder=embedder,
        images=images,
        settings=settings,
    )


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
CatalogServiceDep = Annotated[CatalogService, Depends(get_catalog_service)]
IndexingServiceDep = Annotated[IndexingService, Depends(get_indexing_service)]
