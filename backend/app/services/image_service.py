import io
import logging
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class ImageServiceError(Exception):
    """Raised when image validation or processing fails."""


@dataclass
class ProcessedImage:
    """Normalised image data ready for LLM submission."""

    data: bytes
    media_type: str
    width: int
    height: int


def _detect_extension(filename: str) -> str:
    _, _, ext = filename.rpartition(".")
    return "." + ext.lower() if ext else ""


def process_image(file_bytes: bytes, filename: str, content_type: str | None) -> ProcessedImage:
    """Validate, normalise, and encode an uploaded product image.

    Pipeline
    1.  Size check
    2.  Extension and MIME-type validation
    3.  Pillow parse (catches corrupt / decompression-bomb files)
    4.  EXIF orientation correction
    5.  RGBA / CMYK / palette → RGB conversion
    6.  Downscale if any dimension exceeds ``max_image_width`` /
        ``max_image_height``
    7.  Re-encode as JPEG

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used for extension check).
        content_type: MIME type from the upload header.

    Returns:
        A :class:`ProcessedImage` with normalised JPEG data.

    Raises:
        ImageServiceError: If any validation or processing step fails.
    """
    max_bytes = settings.max_image_size_bytes
    if not file_bytes:
        raise ImageServiceError("Uploaded file is empty.")

    if len(file_bytes) > max_bytes:
        raise ImageServiceError(
            f"Image exceeds the maximum allowed size of "
            f"{settings.max_image_size_mb} MB."
        )

    ext = _detect_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageServiceError(
            "Unsupported image format. "
            "Accepted: .jpg, .jpeg, .png, .webp."
        )

    if content_type and content_type not in SUPPORTED_MIME_TYPES:
        raise ImageServiceError(
            f"Unsupported media type '{content_type}'. "
            f"Accepted: image/jpeg, image/png, image/webp."
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise ImageServiceError(
            "File could not be recognised as a valid image."
        ) from exc

    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ImageServiceError(
            "File could not be opened for processing."
        ) from exc

    detected = img.format or ""
    if detected.upper() not in ("JPEG", "PNG", "WEBP"):
        img.close()
        raise ImageServiceError(
            f"Detected format '{detected}' is not supported. "
            f"Accepted: JPEG, PNG, WebP."
        )

    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode in ("RGBA", "LA", "P"):
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            img_rgb.paste(img, mask=img.split()[3])
        else:
            img_rgb.paste(img)
        img = img_rgb
    elif img.mode != "RGB":
        img = img.convert("RGB")

    max_w = settings.max_image_width
    max_h = settings.max_image_height
    orig_w, orig_h = img.size
    if orig_w > max_w or orig_h > max_h:
        ratio = min(max_w / orig_w, max_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Image downscaled %dx%d → %dx%d", orig_w, orig_h, new_w, new_h)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    w, h = img.size
    img.close()

    logger.info(
        "Image processed — format=%s %dx%d → JPEG %dx%d %d bytes",
        detected,
        orig_w,
        orig_h,
        w,
        h,
        len(data),
    )

    return ProcessedImage(data=data, media_type="image/jpeg", width=w, height=h)
