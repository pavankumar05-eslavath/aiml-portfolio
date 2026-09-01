"""Product CRUD and catalogue browsing endpoints.

Handlers stay thin on purpose: parse and validate input, delegate to
:class:`~app.services.catalog_service.CatalogService`, shape the response. All
orchestration (keeping Qdrant consistent, invalidating fingerprints) lives in the
service so it is reachable from the ingestion scripts too.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Path, Query, Response

from app.api.deps import CatalogServiceDep, SessionDep
from app.schemas.common import ErrorResponse, Page
from app.schemas.product import (
    CatalogFacets,
    ProductCreate,
    ProductRead,
    ProductUpdate,
)
from app.schemas.search import SearchFilters, SortOption

router = APIRouter(prefix="/products", tags=["products"])

ProductIdPath = Annotated[str, Path(description="Product UUID.", min_length=1, max_length=36)]


@router.get(
    "",
    response_model=Page[ProductRead],
    summary="List products",
)
async def list_products(
    catalog: CatalogServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
    search: Annotated[
        str | None,
        Query(max_length=200, description="Substring match on name, brand or category."),
    ] = None,
    category: Annotated[list[str] | None, Query(description="Repeatable.")] = None,
    brand: Annotated[list[str] | None, Query(description="Repeatable.")] = None,
    colour: Annotated[list[str] | None, Query(description="Repeatable.")] = None,
    min_price: Annotated[float | None, Query(ge=0)] = None,
    max_price: Annotated[float | None, Query(ge=0)] = None,
    in_stock_only: bool = False,
    sort: SortOption = SortOption.RELEVANCE,
) -> Page[ProductRead]:
    """Page through the catalogue with metadata filters applied in SQL.

    This is the *browse* path and performs no embedding; ranked retrieval lives
    under ``/search``. With ``sort=relevance`` and no query to be relevant to,
    results are ordered newest-first.
    """
    filters = SearchFilters(
        categories=category or [],
        brands=brand or [],
        colours=colour or [],
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
    )
    products, total = await catalog.list_products(
        filters=filters, search=search, sort=sort, limit=limit, offset=offset
    )
    return Page[ProductRead](
        items=[ProductRead.model_validate(p) for p in products],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/facets",
    response_model=CatalogFacets,
    summary="Available filter values",
)
async def product_facets(catalog: CatalogServiceDep) -> CatalogFacets:
    """Return distinct categories, brands and colours with counts.

    Lets the UI build its filter panel from the data actually present rather than
    from a hardcoded list.
    """
    return await catalog.facets()


@router.get(
    "/{product_id}",
    response_model=ProductRead,
    summary="Get a product",
    responses={404: {"model": ErrorResponse}},
)
async def get_product(product_id: ProductIdPath, catalog: CatalogServiceDep) -> ProductRead:
    """Fetch a single product, including its indexing state."""
    return ProductRead.model_validate(await catalog.get(product_id))


@router.post(
    "",
    response_model=ProductRead,
    status_code=201,
    summary="Create a product",
    responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def create_product(
    payload: Annotated[ProductCreate, Body()],
    catalog: CatalogServiceDep,
    session: SessionDep,
) -> ProductRead:
    """Create a product.

    The product is created unindexed and is not yet searchable; call
    ``POST /catalog/index`` to embed it. Keeping the two separate means a bulk
    import is not serialised behind model inference.
    """
    product = await catalog.create(payload)
    await session.commit()
    return ProductRead.model_validate(product)


@router.put(
    "/{product_id}",
    response_model=ProductRead,
    summary="Update a product",
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def update_product(
    product_id: ProductIdPath,
    payload: Annotated[ProductUpdate, Body()],
    catalog: CatalogServiceDep,
    session: SessionDep,
) -> ProductRead:
    """Partially update a product.

    Changing a field that feeds the embedding (name, description, category,
    brand, colour, image) clears the content hash, so the next indexing run
    re-embeds this product and nothing else.
    """
    product = await catalog.update(product_id, payload)
    await session.commit()
    return ProductRead.model_validate(product)


@router.delete(
    "/{product_id}",
    status_code=204,
    summary="Delete a product",
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def delete_product(
    product_id: ProductIdPath,
    catalog: CatalogServiceDep,
    session: SessionDep,
) -> Response:
    """Delete a product and its vectors.

    If the vector store cannot be reached the delete is refused with 503 rather
    than leaving a vector that outlives its metadata.
    """
    await catalog.delete(product_id)
    await session.commit()
    return Response(status_code=204)
