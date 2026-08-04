import io
import os

import pytest
from PIL import Image

from app.services.image_service import (
    ImageServiceError,
    ProcessedImage,
    process_image,
)

# ── helpers ──────────────────────────────────────────────────────────

SIZE_64 = 64
SIZE_3000 = 3000  # exceeds MAX_IMAGE_WIDTH / MAX_IMAGE_HEIGHT (2048)


def _make_image_bytes(fmt: str, mode: str = "RGB", size: int = SIZE_64) -> bytes:
    buf = io.BytesIO()
    img = Image.new(mode, (size, size), color=(255, 0, 0))
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── process_image ────────────────────────────────────────────────────


class TestProcessImage:
    def test_valid_jpeg(self):
        data = _make_image_bytes("JPEG")
        result = process_image(data, "photo.jpg", "image/jpeg")
        assert isinstance(result, ProcessedImage)
        assert result.media_type == "image/jpeg"
        assert result.width == SIZE_64
        assert result.height == SIZE_64
        assert len(result.data) > 0

    def test_valid_png(self):
        data = _make_image_bytes("PNG", "RGBA")
        result = process_image(data, "photo.png", "image/png")
        assert result.media_type == "image/jpeg"
        assert result.width == SIZE_64
        assert result.height == SIZE_64

    def test_valid_webp(self):
        data = _make_image_bytes("WEBP")
        result = process_image(data, "photo.webp", "image/webp")
        assert result.media_type == "image/jpeg"
        assert result.width == SIZE_64
        assert result.height == SIZE_64

    def test_empty_bytes_rejected(self):
        with pytest.raises(ImageServiceError, match="empty"):
            process_image(b"", "empty.jpg", "image/jpeg")

    def test_oversized_file_rejected(self):
        from app.core.config import settings

        big = b"A" * (settings.max_image_size_bytes + 1)
        with pytest.raises(ImageServiceError, match="exceeds"):
            process_image(big, "big.jpg", "image/jpeg")

    def test_bad_extension_rejected(self):
        data = _make_image_bytes("JPEG")
        with pytest.raises(ImageServiceError, match="Unsupported image format"):
            process_image(data, "photo.gif", "image/gif")

    def test_bad_content_type_rejected(self):
        data = _make_image_bytes("JPEG")
        with pytest.raises(ImageServiceError, match="Unsupported media type"):
            process_image(data, "photo.jpg", "image/gif")

    def test_corrupt_file_rejected(self):
        with pytest.raises(ImageServiceError, match="could not be recognised"):
            process_image(b"\xff\xd8\xff\x00", "corrupt.jpg", "image/jpeg")

    def test_rgba_converted_to_rgb(self):
        data = _make_image_bytes("PNG", "RGBA")
        result = process_image(data, "photo.png", "image/png")
        assert isinstance(result, ProcessedImage)

    def test_large_image_downscaled(self):
        data = _make_image_bytes("JPEG", size=SIZE_3000)
        result = process_image(data, "large.jpg", "image/jpeg")
        assert result.width <= 2048
        assert result.height <= 2048
        assert result.width < SIZE_3000

    def test_filename_without_extension_rejected(self):
        data = _make_image_bytes("JPEG")
        with pytest.raises(ImageServiceError, match="Unsupported image format"):
            process_image(data, "photo", "image/jpeg")

    def test_filename_uppercase_extension(self):
        data = _make_image_bytes("JPEG")
        result = process_image(data, "photo.JPG", "image/jpeg")
        assert isinstance(result, ProcessedImage)


# ── ProcessedImage dataclass ────────────────────────────────────────


class TestProcessedImage:
    def test_fields(self):
        pi = ProcessedImage(data=b"abc", media_type="image/jpeg", width=10, height=20)
        assert pi.data == b"abc"
        assert pi.media_type == "image/jpeg"
        assert pi.width == 10
        assert pi.height == 20
