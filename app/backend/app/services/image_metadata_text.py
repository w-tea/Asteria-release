from __future__ import annotations

import re
from pathlib import Path

from PIL import ExifTags, Image


def metadata_value_to_text(value: object) -> str:
    if isinstance(value, bytes):
        if value.startswith(b"UNICODE\x00"):
            payload = value[8:]
            if payload.startswith((b"\xfe\xff", b"\xff\xfe")):
                return payload.decode("utf-16", errors="ignore").strip("\x00")
            return payload.decode("utf-16-be", errors="ignore").strip("\x00")
        decoded = value.decode("utf-8", errors="ignore")
        if not decoded.strip():
            decoded = value.decode("latin-1", errors="ignore")
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", decoded)
        cleaned = re.sub(r"\s+([,;:])", r"\1", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()
    if isinstance(value, str):
        return value
    return ""


def read_image_metadata_values(path: Path) -> tuple[dict[str, object], list[object], str, int, int]:
    with Image.open(path) as image:
        image.load()
        info = dict(image.info)
        values: list[object] = list(info.values())
        try:
            exif = image.getexif()
        except (OSError, SyntaxError, ValueError):
            exif = None
        if exif:
            values.extend(exif.values())
            try:
                values.extend(exif.get_ifd(ExifTags.IFD.Exif).values())
            except (KeyError, OSError, TypeError, ValueError):
                pass
        format_name = str(image.format or "").lower()
        if format_name == "jpg":
            format_name = "jpeg"
        return info, values, format_name, image.width, image.height


def first_generation_parameters_text(
    image_info: dict[str, object],
    metadata_values: list[object],
) -> str:
    direct = metadata_value_to_text(image_info.get("parameters"))
    if direct.strip():
        return direct
    for value in metadata_values:
        text = metadata_value_to_text(value)
        if "Steps:" in text and any(marker in text for marker in ("Sampler:", "CFG scale:", "Seed:")):
            return text
    return ""
