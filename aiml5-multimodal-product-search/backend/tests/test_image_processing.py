"""Image validation and preprocessing tests.

Uploaded bytes are hostile input, so these tests focus on what must be *rejected*
as much as on what must be accepted.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import ImageTooLargeError, InvalidImageError
from app.services.image_processing import ImageProcessor
from tests.conftest import image_bytes, make_image


class TestValidation:
    def test_accepts_jpeg(self, image_processor: ImageProcessor):
        metadata = image_processor.validate_bytes(image_bytes(fmt="JPEG"))
        assert metadata.format == "JPEG"
        assert metadata.width == 64
        assert metadata.height == 64

    @pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
    def test_accepts_every_configured_format(self, image_processor: ImageProcessor, fmt: str):
        assert image_processor.validate_bytes(image_bytes(fmt=fmt)).format == fmt

    def test_rejects_unsupported_format(self, image_processor: ImageProcessor):
        """BMP decodes fine but is not in ALLOWED_IMAGE_FORMATS."""
        with pytest.raises(InvalidImageError, match="Unsupported image format"):
            image_processor.validate_bytes(image_bytes(fmt="BMP"))

    def test_rejects_empty_bytes(self, image_processor: ImageProcessor):
        with pytest.raises(InvalidImageError, match="empty"):
            image_processor.validate_bytes(b"")

    def test_rejects_plain_text(self, image_processor: ImageProcessor):
        with pytest.raises(InvalidImageError, match="not a recognised image"):
            image_processor.validate_bytes(b"This is a text file, not an image.")

    def test_rejects_truncated_jpeg(self, image_processor: ImageProcessor):
        """A half-decoded image would produce a misleading embedding."""
        with pytest.raises(InvalidImageError):
            image_processor.validate_bytes(image_bytes()[:50])

    def test_rejects_header_only_payload(self, image_processor: ImageProcessor):
        with pytest.raises(InvalidImageError):
            image_processor.validate_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    def test_rejects_oversized_payload(self, settings: Settings):
        """Size is checked on the byte length, before any decode is attempted."""
        settings.max_upload_bytes = 1024
        processor = ImageProcessor(settings)
        with pytest.raises(ImageTooLargeError) as caught:
            processor.validate_bytes(image_bytes(size=(600, 600)))
        assert caught.value.status_code == 413
        assert caught.value.details["limit_bytes"] == 1024

    def test_rejects_image_below_minimum_dimension(self, settings: Settings):
        settings.min_image_dimension = 32
        processor = ImageProcessor(settings)
        with pytest.raises(InvalidImageError, match="too small"):
            processor.validate_bytes(image_bytes(size=(8, 8)))

    def test_content_type_is_not_trusted(self, image_processor: ImageProcessor):
        """Validation depends on the decoder, not on a caller-supplied label."""
        with pytest.raises(InvalidImageError):
            image_processor.validate_bytes(b"not an image", filename="photo.jpg")

    def test_rejects_decompression_bomb(self, settings: Settings):
        """A small file that decodes to a huge bitmap must be refused.

        Pillow raises DecompressionBombWarning past its pixel limit; the processor
        promotes warnings to errors so the guard cannot be silently ignored.
        """
        settings.max_upload_bytes = 50 * 1024 * 1024
        processor = ImageProcessor(settings)
        original_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 1024  # 32x32
        try:
            with pytest.raises(InvalidImageError):
                processor.validate_bytes(image_bytes(size=(256, 256)))
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit

    def test_error_details_do_not_leak_internals(self, image_processor: ImageProcessor):
        with pytest.raises(InvalidImageError) as caught:
            image_processor.validate_bytes(b"garbage", filename="x.jpg")
        assert "Traceback" not in caught.value.message


class TestPreprocessing:
    def test_converts_greyscale_to_rgb(self, image_processor: ImageProcessor):
        grey = Image.new("L", (32, 32), 128)
        assert image_processor.preprocess(grey).mode == "RGB"

    def test_flattens_transparency_onto_white(self, image_processor: ImageProcessor):
        """A transparent cut-out must not acquire a black background."""
        transparent = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
        result = image_processor.preprocess(transparent)
        assert result.mode == "RGB"
        assert result.getpixel((8, 8)) == (255, 255, 255)

    def test_downscales_beyond_the_cap(self, settings: Settings):
        settings.max_image_dimension = 64
        processor = ImageProcessor(settings)
        result = processor.preprocess(make_image((10, 20, 30), size=(512, 256)))
        assert max(result.size) == 64
        # Aspect ratio is preserved.
        assert result.size == (64, 32)

    def test_leaves_small_images_untouched(self, settings: Settings):
        settings.max_image_dimension = 1024
        processor = ImageProcessor(settings)
        result = processor.preprocess(make_image((0, 0, 0), size=(60, 80)))
        assert result.size == (60, 80)

    def test_load_from_bytes_returns_processed_image_and_metadata(
        self, image_processor: ImageProcessor
    ):
        loaded = image_processor.load_from_bytes(image_bytes(fmt="PNG"))
        assert loaded.image.mode == "RGB"
        assert loaded.metadata.format == "PNG"
        assert loaded.metadata.size_bytes > 0


class TestCatalogPathResolution:
    def test_resolves_relative_path_inside_root(
        self, image_processor: ImageProcessor, image_root: Path
    ):
        target = image_root / "sub" / "item.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        make_image((1, 2, 3)).save(target, format="JPEG")
        assert image_processor.resolve_catalog_path("sub/item.jpg") == target.resolve()

    def test_returns_none_for_missing_file(
        self, image_processor: ImageProcessor, image_root: Path
    ):
        assert image_processor.resolve_catalog_path("nope.jpg") is None

    def test_returns_none_for_http_url(self, image_processor: ImageProcessor):
        """Remote images are the ingestion pipeline's job, not the request path's."""
        assert image_processor.resolve_catalog_path("https://cdn.example.com/a.jpg") is None

    @pytest.mark.parametrize(
        "hostile",
        [
            "../../../../etc/passwd",
            "sub/../../../../etc/passwd",
            "/etc/passwd",
            "....//....//etc/passwd",
        ],
    )
    def test_blocks_path_traversal(
        self, image_processor: ImageProcessor, image_root: Path, hostile: str
    ):
        """A crafted image_url must not be able to read files outside IMAGE_ROOT."""
        assert image_processor.resolve_catalog_path(hostile) is None

    def test_empty_reference_returns_none(self, image_processor: ImageProcessor):
        assert image_processor.resolve_catalog_path("") is None

    def test_load_from_path_rejects_missing_file(
        self, image_processor: ImageProcessor, tmp_path: Path
    ):
        with pytest.raises(InvalidImageError, match="not found"):
            image_processor.load_from_path(tmp_path / "absent.jpg")

    def test_load_from_path_rejects_non_image_file(
        self, image_processor: ImageProcessor, tmp_path: Path
    ):
        bogus = tmp_path / "fake.jpg"
        bogus.write_text("definitely not an image")
        with pytest.raises(InvalidImageError):
            image_processor.load_from_path(bogus)


async def test_load_from_upload_runs_off_the_event_loop(image_processor: ImageProcessor):
    """Decoding is CPU-bound and must not block the loop."""
    loaded = await image_processor.load_from_upload(image_bytes(), filename="a.jpg")
    assert loaded.image.size == (64, 64)


def test_animated_gif_is_rejected_by_default(image_processor: ImageProcessor):
    """GIF is not in the allowed set, so animation handling never arises."""
    buffer = BytesIO()
    frames = [make_image((i * 40, 0, 0), size=(32, 32)) for i in range(1, 4)]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    with pytest.raises(InvalidImageError, match="Unsupported image format"):
        image_processor.validate_bytes(buffer.getvalue())
