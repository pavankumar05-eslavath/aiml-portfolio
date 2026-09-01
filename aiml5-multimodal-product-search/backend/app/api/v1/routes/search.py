"""Search endpoints: text, image, multimodal and "more like this".

Request shapes
--------------
Text search takes JSON. The image and multimodal endpoints accept
``multipart/form-data``, because that is what a browser file input produces and
it avoids the ~33% size inflation of base64. Since multipart fields are flat
strings, the structured part of the request (filters, fusion overrides) is
supplied as a JSON string in the ``options`` field - one parser, one schema,
identical validation to the JSON endpoints.

A base64 JSON variant is also provided at ``/search/multimodal/json`` for
scripted clients such as the evaluation harness.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, File, Form, Path, UploadFile

from app.api.deps import ImageProcessorDep, SearchServiceDep
from app.core.exceptions import EmptyQueryError, ValidationError
from app.core.logging import get_logger
from app.schemas.common import ErrorResponse
from app.schemas.search import (
    BaseSearchRequest,
    MultimodalSearchRequest,
    SearchResponse,
    TextSearchRequest,
)
from app.services.search_service import SearchQuery, decode_base64_image

logger = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

_SEARCH_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Uploaded image is too large."},
    422: {"model": ErrorResponse, "description": "Invalid image or empty query."},
    503: {"model": ErrorResponse, "description": "Model or a data store is unavailable."},
}


def _parse_options(raw: str | None) -> BaseSearchRequest:
    """Parse the ``options`` multipart field into a validated request object.

    Raises:
        ValidationError: if the field is not a JSON object or fails validation.
    """
    if not raw or not raw.strip():
        return BaseSearchRequest()
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"'options' is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValidationError("'options' must be a JSON object.")
    return BaseSearchRequest.model_validate(decoded)


@router.post(
    "/text",
    response_model=SearchResponse,
    summary="Text-only semantic search",
    responses=_SEARCH_ERRORS,
)
async def search_text(
    payload: TextSearchRequest,
    service: SearchServiceDep,
) -> SearchResponse:
    """Search the catalogue with natural language.

    The query is embedded once and compared against **both** product vectors: the
    product's text embedding (same-modality) and its image embedding
    (cross-modal). ``CROSS_MODAL_WEIGHT`` controls the balance, and the response
    reports each channel's contribution.
    """
    return await service.search(
        SearchQuery(
            text=payload.query,
            top_k=payload.top_k,
            offset=payload.offset,
            filters=payload.filters,
            sort=payload.sort,
            overrides=payload.fusion,
            explain=payload.explain,
        )
    )


@router.post(
    "/image",
    response_model=SearchResponse,
    summary="Image-only visual search",
    responses=_SEARCH_ERRORS,
)
async def search_image(
    service: SearchServiceDep,
    images: ImageProcessorDep,
    file: Annotated[UploadFile, File(description="Query image (JPEG, PNG or WEBP).")],
    options: Annotated[
        str | None,
        Form(description="JSON object with top_k, filters, sort, fusion, explain."),
    ] = None,
) -> SearchResponse:
    """Find visually similar products from an uploaded image.

    The upload is validated by decoding it, not by trusting its content type or
    extension, and is held in memory only for the duration of the request.
    """
    request = _parse_options(options)
    loaded = await images.load_from_upload(await file.read(), filename=file.filename)
    logger.info(
        "image query accepted",
        extra={
            "context": {
                "format": loaded.metadata.format,
                "width": loaded.metadata.width,
                "height": loaded.metadata.height,
                "size_bytes": loaded.metadata.size_bytes,
            }
        },
    )
    return await service.search(
        SearchQuery(
            image=loaded.image,
            top_k=request.top_k,
            offset=request.offset,
            filters=request.filters,
            sort=request.sort,
            overrides=request.fusion,
            explain=request.explain,
        )
    )


@router.post(
    "/multimodal",
    response_model=SearchResponse,
    summary="Combined image + text search",
    responses=_SEARCH_ERRORS,
)
async def search_multimodal(
    service: SearchServiceDep,
    images: ImageProcessorDep,
    query: Annotated[
        str | None, Form(max_length=400, description="Natural-language query.")
    ] = None,
    file: Annotated[UploadFile | None, File(description="Query image.")] = None,
    options: Annotated[str | None, Form()] = None,
) -> SearchResponse:
    """Search with an image and text together.

    This is the endpoint behind "something like this, but in black": the image
    supplies visual context while the text applies a modification. Supplying only
    one of the two is accepted and degrades to that single modality, so the
    frontend can use one endpoint for every mode.
    """
    request = _parse_options(options)
    text = (query or "").strip() or None

    image = None
    if file is not None and file.filename:
        loaded = await images.load_from_upload(await file.read(), filename=file.filename)
        image = loaded.image

    if image is None and not text:
        raise EmptyQueryError("Provide a text query, an image, or both.")

    return await service.search(
        SearchQuery(
            text=text,
            image=image,
            top_k=request.top_k,
            offset=request.offset,
            filters=request.filters,
            sort=request.sort,
            overrides=request.fusion,
            explain=request.explain,
        )
    )


@router.post(
    "/multimodal/json",
    response_model=SearchResponse,
    summary="Combined search with a base64 or catalogue image",
    responses=_SEARCH_ERRORS,
)
async def search_multimodal_json(
    payload: MultimodalSearchRequest,
    service: SearchServiceDep,
    images: ImageProcessorDep,
) -> SearchResponse:
    """JSON variant of multimodal search.

    Accepts either ``image_base64`` or ``image_product_id``. The latter reuses the
    catalogue product's *stored* embedding instead of re-encoding its image, which
    is both faster and exactly reproducible - the property the evaluation harness
    relies on for image queries.
    """
    image = None
    if payload.image_base64:
        loaded = await images.load_from_upload(decode_base64_image(payload.image_base64))
        image = loaded.image

    return await service.search(
        SearchQuery(
            text=payload.query,
            image=image,
            image_product_id=payload.image_product_id,
            top_k=payload.top_k,
            offset=payload.offset,
            filters=payload.filters,
            sort=payload.sort,
            overrides=payload.fusion,
            explain=payload.explain,
        )
    )


@router.post(
    "/similar/{product_id}",
    response_model=SearchResponse,
    summary="More products like this one",
    responses={404: {"model": ErrorResponse}, **_SEARCH_ERRORS},
)
async def search_similar(
    product_id: Annotated[str, Path(description="Product to find neighbours for.")],
    service: SearchServiceDep,
    payload: BaseSearchRequest | None = None,
) -> SearchResponse:
    """Find products similar to an existing catalogue item.

    Uses the product's stored image vector as the query and excludes the product
    itself from the results.
    """
    request = payload or BaseSearchRequest()
    return await service.search(
        SearchQuery(
            image_product_id=product_id,
            top_k=request.top_k,
            offset=request.offset,
            filters=request.filters,
            sort=request.sort,
            overrides=request.fusion,
            explain=request.explain,
        )
    )
