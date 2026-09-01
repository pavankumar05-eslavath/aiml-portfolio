"""Image loading, validation and preprocessing.

Security posture
----------------
Uploaded bytes are never trusted:

* The declared ``Content-Type`` and file extension are treated as hints only;
  the real format comes from Pillow's decoder.
* Size is enforced on the byte length *before* decoding, so a malicious upload
  cannot force a large allocation.
* ``Image.verify()`` runs first to reject truncated/corrupt files, then the image
  is reopened - ``verify()`` leaves the file object unusable by design.
* Pillow's decompression-bomb guard is left enabled and its warning promoted to
  a hard rejection, defending against small files that decode to huge bitmaps.
* Uploads are processed in memory and never persisted. Only catalogue images
  (already on disk or fetched by the ingestion pipeline) touch the filesystem.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final

import anyio
from PIL import Image, ImageFile, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.core.exceptions import ImageTooLargeError, InvalidImageError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Refuse to reconstruct partially-truncated files: a silently half-decoded image
# would produce a misleading embedding.
ImageFile.LOAD_TRUNCATED_IMAGES = False

#: Pillow format name -> canonical extension, for the formats we accept.
FORMAT_EXTENSIONS: Final = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "GIF": ".gif",
}


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Facts about an accepted image."""

    format: str
    width: int
    height: int
    size_bytes: int
    mode: str


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """A validated, preprocessed image plus its original metadata."""

    image: Image.Image
    metadata: ImageMetadata


class ImageProcessor:
    """Validates and normalises images before they reach the encoder."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def allowed_formats(self) -> set[str]:
        """Pillow format names accepted by this instance."""
        return self._settings.allowed_image_format_set

    def validate_bytes(self, data: bytes, *, filename: str | None = None) -> ImageMetadata:
        """Validate raw bytes as an acceptable image.

        Args:
            data: Raw uploaded bytes.
            filename: Original filename, used only for error messages.

        Returns:
            Metadata describing the image.

        Raises:
            ImageTooLargeError: Byte length exceeds ``MAX_UPLOAD_BYTES``.
            InvalidImageError: Empty, corrupt, unsupported, or implausibly sized.
        """
        if not data:
            raise InvalidImageError("The uploaded file is empty.")
        limit = self._settings.max_upload_bytes
        if len(data) > limit:
            raise ImageTooLargeError(
                f"Image is {len(data) / 1_048_576:.2f} MB; the limit is "
                f"{limit / 1_048_576:.2f} MB.",
                details={"size_bytes": len(data), "limit_bytes": limit},
            )

        # Pass 1: structural verification. Promote Pillow warnings (notably
        # DecompressionBombWarning) to exceptions so they cannot be ignored.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(BytesIO(data)) as probe:
                    probe.verify()
        except UnidentifiedImageError as exc:
            raise InvalidImageError(
                "The uploaded file is not a recognised image.",
                details={"filename": filename} if filename else None,
            ) from exc
        except Exception as exc:
            raise InvalidImageError(f"The image could not be decoded: {exc}") from exc

        # Pass 2: reopen for real; verify() consumed the first handle.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(BytesIO(data)) as image:
                    fmt = (image.format or "").upper()
                    width, height = image.size
                    mode = image.mode
                    image.load()
        except Exception as exc:
            raise InvalidImageError(f"The image could not be decoded: {exc}") from exc

        if fmt not in self.allowed_formats:
            raise InvalidImageError(
                f"Unsupported image format {fmt or 'unknown'}. "
                f"Allowed: {', '.join(sorted(self.allowed_formats))}.",
                details={"detected_format": fmt or None},
            )

        smallest = self._settings.min_image_dimension
        if width < smallest or height < smallest:
            raise InvalidImageError(
                f"Image is too small ({width}x{height}); minimum edge is {smallest}px.",
                details={"width": width, "height": height},
            )

        return ImageMetadata(
            format=fmt, width=width, height=height, size_bytes=len(data), mode=mode
        )

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Convert to RGB and bound the longest edge.

        The encoder's own processor resizes to the model's input resolution
        anyway; downscaling first bounds the cost of that resize for very large
        uploads. Note the dataset images are 60x80, so upscaling is left to the
        model processor, which uses bicubic interpolation.
        """
        if image.mode != "RGB":
            # Flatten alpha onto white rather than dropping it, so transparent
            # product cut-outs do not acquire a black background.
            if image.mode in ("RGBA", "LA", "PA"):
                backdrop = Image.new("RGB", image.size, (255, 255, 255))
                backdrop.paste(image, mask=image.convert("RGBA").split()[-1])
                image = backdrop
            else:
                image = image.convert("RGB")

        longest = max(image.size)
        cap = self._settings.max_image_dimension
        if longest > cap:
            scale = cap / longest
            new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def load_from_bytes(self, data: bytes, *, filename: str | None = None) -> LoadedImage:
        """Validate and preprocess raw bytes in one step."""
        metadata = self.validate_bytes(data, filename=filename)
        with Image.open(BytesIO(data)) as image:
            image.load()
            processed = self.preprocess(image)
        return LoadedImage(image=processed, metadata=metadata)

    async def load_from_upload(
        self, data: bytes, *, filename: str | None = None
    ) -> LoadedImage:
        """Validate and preprocess an upload off the event loop."""
        return await anyio.to_thread.run_sync(
            lambda: self.load_from_bytes(data, filename=filename)
        )

    def load_from_path(self, path: str | Path) -> LoadedImage:
        """Load a catalogue image from disk.

        Raises:
            InvalidImageError: if the path is missing or the file is unusable.
        """
        resolved = Path(path)
        if not resolved.is_file():
            raise InvalidImageError(f"Image file not found: {resolved}")
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise InvalidImageError(f"Could not read image {resolved}: {exc}") from exc
        return self.load_from_bytes(data, filename=resolved.name)

    def resolve_catalog_path(self, image_reference: str) -> Path | None:
        """Resolve a stored image reference to a local file, if one exists.

        References may be absolute paths, or paths relative to ``IMAGE_ROOT``.
        HTTP(S) URLs return ``None`` - fetching them is the ingestion pipeline's
        job, not the request path's.

        Path traversal outside ``IMAGE_ROOT`` is rejected, so a crafted
        ``image_url`` cannot be used to read arbitrary files.
        """
        if not image_reference or image_reference.startswith(("http://", "https://")):
            return None

        root = Path(self._settings.image_root).resolve()
        candidate = Path(image_reference)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()

        if not resolved.is_relative_to(root):
            logger.warning(
                "rejected image path outside IMAGE_ROOT",
                extra={"context": {"reference": image_reference}},
            )
            return None
        return resolved if resolved.is_file() else None


_processor: ImageProcessor | None = None


def get_image_processor() -> ImageProcessor:
    """Return the process-wide image processor."""
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor
