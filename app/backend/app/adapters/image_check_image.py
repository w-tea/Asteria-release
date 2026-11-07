from __future__ import annotations

import hashlib
import io
import os
import stat
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_PIXELS = 16_000_000
# Generated candidates are read sequentially from an operation-owned directory.
# Keep this finite and separate from the saved-image test's existing limits.
MAX_GENERATED_SOURCE_BYTES = 64 * 1024 * 1024
MAX_GENERATED_SOURCE_PIXELS = 32_000_000
MAX_PREPARED_EDGE = 1536
MAX_PREPARED_BYTES = 2 * 1024 * 1024
SUPPORTED_SOURCE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


class ImageCheckImageError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class ImageCheckImageReceipt:
    output_root: str
    source_path: str
    source_sha256: str
    source_bytes: int
    source_format: str
    source_width: int
    source_height: int
    exif_orientation: int
    oriented_width: int
    oriented_height: int
    prepared_sha256: str
    prepared_bytes: int
    prepared_width: int
    prepared_height: int
    resized: bool
    alpha_matte: str | None
    metadata_stripped: bool = True
    input_policy: str = "saved_image"


@dataclass(frozen=True)
class PreparedImageCheckImage:
    content: bytes = field(repr=False)
    media_type: str
    receipt: ImageCheckImageReceipt


def _read_source(source: str | Path, output_root: str | Path, *, max_bytes: int | None = None) -> tuple[Path, Path, bytes]:
    byte_limit = MAX_SOURCE_BYTES if max_bytes is None else max_bytes
    try:
        root = Path(output_root).resolve(strict=True)
        if not root.is_dir():
            raise OSError("output root is not a directory")
    except (OSError, RuntimeError):
        raise ImageCheckImageError("image_check_output_root_unavailable") from None
    try:
        path = Path(source).resolve(strict=True)
    except (OSError, RuntimeError):
        raise ImageCheckImageError("image_check_source_unavailable") from None
    if not path.is_relative_to(root):
        raise ImageCheckImageError("image_check_source_outside_output_root")
    try:
        with path.open("rb") as stream:
            source_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(source_stat.st_mode):
                raise ImageCheckImageError("image_check_source_unavailable")
            if source_stat.st_size > byte_limit:
                raise ImageCheckImageError("image_check_source_too_large")
            content = stream.read(byte_limit + 1)
    except OSError:
        raise ImageCheckImageError("image_check_source_unavailable") from None
    if len(content) > byte_limit:
        raise ImageCheckImageError("image_check_source_too_large")
    return root, path, content


def prepare_image_check_image(source: str | Path, *, output_root: str | Path,
                              generated_final: bool = False) -> PreparedImageCheckImage:
    """Prepare one bounded static image without writing or changing its source."""
    if generated_final:
        from app.services.image_check_finalization import is_private_image_candidate
        if not is_private_image_candidate(Path(source).resolve()):
            raise ImageCheckImageError("image_check_final_source_not_staged")
    root, path, source_content = _read_source(source, output_root,
        max_bytes=MAX_GENERATED_SOURCE_BYTES if generated_final else None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source_content)) as opened:
                source_format = opened.format or ""
                if source_format not in SUPPORTED_SOURCE_FORMATS:
                    raise ImageCheckImageError("image_check_source_format_unsupported")
                width, height = opened.size
                pixel_limit = MAX_GENERATED_SOURCE_PIXELS if generated_final else MAX_SOURCE_PIXELS
                if width * height > pixel_limit:
                    raise ImageCheckImageError("image_check_source_too_many_pixels")
                if getattr(opened, "n_frames", 1) != 1:
                    raise ImageCheckImageError("image_check_source_animated")
                orientation_value = opened.getexif().get(274, 1)
                orientation = orientation_value if isinstance(orientation_value, int) and 1 <= orientation_value <= 8 else 1
                with ImageOps.exif_transpose(opened) as oriented:
                    oriented_width, oriented_height = oriented.size
                    oriented.thumbnail((MAX_PREPARED_EDGE, MAX_PREPARED_EDGE), Image.Resampling.LANCZOS)
                    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                    # A fresh RGB image does not inherit prompt, EXIF or location metadata.
                    with Image.new("RGB", oriented.size, "white") as clean:
                        if has_alpha:
                            with oriented.convert("RGBA") as rgba, rgba.getchannel("A") as alpha:
                                clean.paste(rgba, mask=alpha)
                        else:
                            with oriented.convert("RGB") as rgb:
                                clean.paste(rgb)
                        encoded = io.BytesIO()
                        clean.save(encoded, format="JPEG", quality=90)
                        prepared_content = encoded.getvalue()
                        prepared_width, prepared_height = clean.size
    except ImageCheckImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageCheckImageError("image_check_source_invalid") from None
    if len(prepared_content) > MAX_PREPARED_BYTES:
        raise ImageCheckImageError("image_check_prepared_too_large")
    return PreparedImageCheckImage(
        content=prepared_content,
        media_type="image/jpeg",
        receipt=ImageCheckImageReceipt(
            output_root=str(root),
            source_path=str(path),
            source_sha256=hashlib.sha256(source_content).hexdigest(),
            source_bytes=len(source_content),
            source_format=source_format,
            source_width=width,
            source_height=height,
            exif_orientation=orientation,
            oriented_width=oriented_width,
            oriented_height=oriented_height,
            prepared_sha256=hashlib.sha256(prepared_content).hexdigest(),
            prepared_bytes=len(prepared_content),
            prepared_width=prepared_width,
            prepared_height=prepared_height,
            resized=(prepared_width, prepared_height) != (oriented_width, oriented_height),
            alpha_matte="white" if has_alpha else None,
            input_policy="generated_final" if generated_final else "saved_image",
        ),
    )
