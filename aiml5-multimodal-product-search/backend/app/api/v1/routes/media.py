"""Serving catalogue product images.

Exists so the frontend can render locally-ingested images: the dataset ships
image *files*, not public URLs, so something has to serve them.

This is a development convenience, not a production image pipeline. In a real
deployment product images belong on a CDN or object store, and ``image_url``
would hold that absolute URL - which this endpoint already supports by leaving
``http(s)://`` references untouched. The trade-off is documented in
``docs/architecture.md``.

Security
--------
Two independent checks apply:

1. **Containment.** The requested path is fully resolved and then required to be
   inside ``IMAGE_ROOT``. Because resolution follows symlinks, this single check
   covers ``../`` sequences, absolute paths, *and* symlinks whose target lies
   outside the root.
2. **Suffix allow-list.** Only known image extensions are served, so even a file
   inside the root is not readable unless it is an image.

Symlinks that stay *within* the root are deliberately allowed: they cannot reach
anything the endpoint would not otherwise serve, and refusing them would break
deployments that mount an image directory by symlink.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Final

import anyio
from fastapi import APIRouter, Response
from fastapi.responses import FileResponse

from app.api.deps import SettingsDep
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.schemas.common import ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/media", tags=["media"])

_MEDIA_TYPES: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

#: Product images are immutable once ingested, so they can be cached hard.
_CACHE_CONTROL: Final = "public, max-age=86400"


class _Outcome(Enum):
    """Result of resolving a requested media path."""

    OK = auto()
    MISSING = auto()
    OUTSIDE_ROOT = auto()


@router.get(
    "/{image_path:path}",
    summary="Serve a catalogue product image",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "The image bytes."},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def get_media(image_path: str, settings: SettingsDep) -> Response:
    """Return a product image stored under ``IMAGE_ROOT``.

    Args:
        image_path: Path relative to ``IMAGE_ROOT``, e.g. ``sample/images/8420.jpg``.
        settings: Injected application settings, providing ``IMAGE_ROOT``.

    Raises:
        ValidationError: unsupported suffix, or a path escaping ``IMAGE_ROOT``.
        NotFoundError: no such file.
    """
    suffix = Path(image_path).suffix.lower()
    if suffix not in _MEDIA_TYPES:
        raise ValidationError(
            f"Unsupported image type {suffix or '(none)'}.",
            details={"allowed": sorted(_MEDIA_TYPES)},
        )

    # `resolve()` and `is_file()` are blocking stat calls. They are cheap
    # individually, but this endpoint serves every product thumbnail on a results
    # page, so they are offloaded rather than run on the event loop.
    def _resolve() -> tuple[_Outcome, Path]:
        root = Path(settings.image_root).resolve()
        candidate = (root / image_path).resolve()
        if not candidate.is_relative_to(root):
            return _Outcome.OUTSIDE_ROOT, candidate
        if not candidate.is_file():
            return _Outcome.MISSING, candidate
        return _Outcome.OK, candidate

    outcome, candidate = await anyio.to_thread.run_sync(_resolve)

    if outcome is _Outcome.OUTSIDE_ROOT:
        # Do not echo the attempted path back to the caller.
        logger.warning(
            "blocked media path traversal attempt",
            extra={"context": {"requested": image_path[:200]}},
        )
        raise ValidationError("Invalid image path.")

    if outcome is _Outcome.MISSING:
        raise NotFoundError(f"No image at {image_path!r}.")

    return FileResponse(
        candidate,
        media_type=_MEDIA_TYPES[suffix],
        headers={"Cache-Control": _CACHE_CONTROL},
        status_code=200,
    )
