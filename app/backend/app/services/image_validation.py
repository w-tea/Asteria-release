from collections.abc import Callable
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}
DEFAULT_MAX_IMAGE_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 8192 * 8192


class InvalidImageContentError(ValueError):
    pass


def _validate_decodable_image(
    opener: Callable[[], Image.Image],
    suffix: str,
    *,
    max_pixels: int,
) -> None:
    expected_format = SUPPORTED_IMAGE_FORMATS.get(suffix.lower())
    if expected_format is None:
        raise InvalidImageContentError("unsupported image extension")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with opener() as opened:
                decoded_format = opened.format
                width, height = opened.size
                if width < 1 or height < 1 or width * height > max_pixels:
                    raise InvalidImageContentError("image dimensions are invalid")
                opened.verify()
            with opener() as opened:
                opened.load()
        if decoded_format != expected_format:
            raise InvalidImageContentError("image format does not match its extension")
    except InvalidImageContentError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise InvalidImageContentError("image content is not decodable") from exc


def validate_image_bytes(
    data: bytes,
    suffix: str,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> None:
    if len(data) > max_bytes:
        raise InvalidImageContentError("image body is too large")
    _validate_decodable_image(lambda: Image.open(BytesIO(data)), suffix, max_pixels=max_pixels)


def validate_image_file(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> None:
    try:
        if path.stat().st_size > max_bytes:
            raise InvalidImageContentError("image file is too large")
    except OSError as exc:
        raise InvalidImageContentError("image file cannot be inspected") from exc
    _validate_decodable_image(lambda: Image.open(path), path.suffix, max_pixels=max_pixels)
