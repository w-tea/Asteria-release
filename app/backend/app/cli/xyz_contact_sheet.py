from __future__ import annotations

import argparse
import binascii
import json
from pathlib import Path
import shutil
import struct
import sys
from typing import Any
import zlib

from PIL import Image, ImageDraw, ImageFont


PNG_MAX_DIMENSION = (2**31) - 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
WRITE_CHUNK_BYTES = 1024 * 1024


def stage_contact_sheet_row(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path, "asteria.xyz-contact-sheet-row.v1")
    staging_root = _directory(manifest["staging_root"])
    output = _contained_path(manifest["output_path"], (staging_root,), "xyz_contact_sheet_row_path_outside_staging")
    allowed_roots = tuple(_directory(value) for value in manifest.get("allowed_roots", []))
    images = [
        _contained_path(value, allowed_roots, "xyz_contact_sheet_image_path_outside_allowed_roots")
        for value in manifest.get("images", [])
    ]
    bounded_sizes = [_size_pair(value) for value in manifest.get("bounded_sizes", [])]
    display_ids = [str(value).strip() for value in manifest.get("display_ids", [])]
    x_labels = [str(value) for value in manifest.get("x_labels", [])]
    tile_width = _positive_int(manifest.get("tile_width"), "tile_width")
    tile_height = _positive_int(manifest.get("tile_height"), "tile_height")
    left_width = _non_negative_int(manifest.get("left_width"), "left_width")
    top_height = _non_negative_int(manifest.get("top_height"), "top_height")
    margin = _non_negative_int(manifest.get("margin"), "margin")
    if not images or len(images) != len(x_labels):
        raise ValueError("xyz_contact_sheet_row_cell_count_invalid")
    if bounded_sizes and len(bounded_sizes) != len(images):
        raise ValueError("xyz_contact_sheet_row_bounded_size_count_invalid")
    if display_ids and len(display_ids) != len(images):
        raise ValueError("xyz_contact_sheet_row_display_id_count_invalid")
    width = _checked_dimension(left_width + len(images) * tile_width + max(0, len(images) - 1) * margin)
    height = _checked_dimension(top_height + tile_height)
    size_bytes = _checked_bytes(width, height, 3)
    _require_disk_space(staging_root, size_bytes)
    temp = output.with_name(f".{output.name}.tmp")
    try:
        with temp.open("w+b") as handle:
            _fill_white(handle, size_bytes)
            if top_height:
                for index, label in enumerate(x_labels):
                    annotation = _label_image(tile_width, top_height, label)
                    _write_image(handle, annotation, width, left_width + index * (tile_width + margin), 0)
            if left_width:
                annotation = _label_image(left_width, tile_height, str(manifest.get("y_label") or ""))
                _write_image(handle, annotation, width, 0, top_height)
            for index, image_path in enumerate(images):
                with Image.open(image_path) as opened:
                    opened.load()
                    image = opened.convert("RGB")
                    bounded_width, bounded_height = bounded_sizes[index] if bounded_sizes else image.size
                    if bounded_width > image.width or bounded_height > image.height:
                        raise ValueError("xyz_contact_sheet_tile_upscale_forbidden")
                    if image.size != (bounded_width, bounded_height):
                        image = image.resize((bounded_width, bounded_height), Image.Resampling.LANCZOS)
                    if image.width > tile_width or image.height > tile_height:
                        raise ValueError("xyz_contact_sheet_tile_exceeds_declared_size")
                    x = left_width + index * (tile_width + margin) + (tile_width - image.width) // 2
                    y = top_height + (tile_height - image.height) // 2
                    if display_ids and display_ids[index]:
                        image = _draw_identity_badge(image, display_ids[index])
                    _write_image(handle, image, width, x, y)
            handle.flush()
        temp.replace(output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"width": width, "height": height, "path": str(output)}


def finalize_contact_sheet(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path, "asteria.xyz-contact-sheet-final.v1")
    staging_root = _directory(manifest["staging_root"])
    output = Path(str(manifest["output_path"])).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plane_width = _positive_int(manifest.get("plane_width"), "plane_width")
    tile_width = _positive_int(manifest.get("tile_width"), "tile_width")
    z_labels = [str(value) for value in manifest.get("z_labels", [])]
    z_header_height = _non_negative_int(manifest.get("z_header_height"), "z_header_height")
    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows or not z_labels or len(raw_rows) != len(z_labels):
        raise ValueError("xyz_contact_sheet_final_rows_invalid")
    row_groups: list[list[tuple[Path, int]]] = []
    expected_heights: list[int] | None = None
    for plane in raw_rows:
        if not isinstance(plane, list) or not plane:
            raise ValueError("xyz_contact_sheet_final_rows_invalid")
        rows: list[tuple[Path, int]] = []
        heights: list[int] = []
        for row in plane:
            if not isinstance(row, dict):
                raise ValueError("xyz_contact_sheet_final_rows_invalid")
            path = _contained_path(row.get("path"), (staging_root,), "xyz_contact_sheet_row_path_outside_staging")
            height = _positive_int(row.get("height"), "row_height")
            expected_size = _checked_bytes(plane_width, height, 3)
            if path.stat().st_size != expected_size:
                raise ValueError("xyz_contact_sheet_row_size_invalid")
            rows.append((path, height))
            heights.append(height)
        if expected_heights is None:
            expected_heights = heights
        elif heights != expected_heights:
            raise ValueError("xyz_contact_sheet_row_height_mismatch")
        row_groups.append(rows)
    assert expected_heights is not None
    width = _checked_dimension(plane_width * len(z_labels))
    height = _checked_dimension(z_header_height + sum(expected_heights))
    _require_disk_space(output.parent, max(1024, width * 4))
    temp = output.with_name(f".{output.name}.tmp")
    label_images = [
        _label_image(min(plane_width, max(tile_width, 1)), z_header_height, label)
        for label in z_labels
    ] if z_header_height else []
    try:
        with temp.open("wb") as target:
            target.write(b"\x89PNG\r\n\x1a\n")
            _png_chunk(target, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            compressor = zlib.compressobj(level=6)
            pending = bytearray()

            def emit(scanline: bytes) -> None:
                pending.extend(compressor.compress(b"\x00" + scanline))
                while len(pending) >= WRITE_CHUNK_BYTES:
                    _png_chunk(target, b"IDAT", bytes(pending[:WRITE_CHUNK_BYTES]))
                    del pending[:WRITE_CHUNK_BYTES]

            white_scanline = bytes([255]) * (width * 3)
            for y in range(z_header_height):
                scanline = bytearray(white_scanline)
                for z_index, label_image in enumerate(label_images):
                    label_row = label_image.crop((0, y, label_image.width, y + 1)).tobytes()
                    x = z_index * plane_width + (plane_width - label_image.width) // 2
                    start = x * 3
                    scanline[start : start + len(label_row)] = label_row
                emit(bytes(scanline))
            for row_index, row_height in enumerate(expected_heights):
                handles = [plane[row_index][0].open("rb") for plane in row_groups]
                try:
                    for _ in range(row_height):
                        scanline = bytearray()
                        for handle in handles:
                            chunk = handle.read(plane_width * 3)
                            if len(chunk) != plane_width * 3:
                                raise ValueError("xyz_contact_sheet_row_read_failed")
                            scanline.extend(chunk)
                        emit(bytes(scanline))
                finally:
                    for handle in handles:
                        handle.close()
            pending.extend(compressor.flush())
            while pending:
                chunk = bytes(pending[:WRITE_CHUNK_BYTES])
                del pending[:WRITE_CHUNK_BYTES]
                _png_chunk(target, b"IDAT", chunk)
            _png_chunk(target, b"IEND", b"")
            target.flush()
        max_image_pixels = Image.MAX_IMAGE_PIXELS
        try:
            # This file was encoded above from already validated, contained rows.
            # Pillow's global decompression-bomb limit would otherwise reject a
            # legitimate large XYZ board during this final self-verification.
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(temp) as image:
                if image.size != (width, height) or image.format != "PNG":
                    raise ValueError("xyz_contact_sheet_png_validation_failed")
                image.verify()
        finally:
            Image.MAX_IMAGE_PIXELS = max_image_pixels
        temp.replace(output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"width": width, "height": height, "output_path": str(output)}


def _read_manifest(path: Path, schema: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("xyz_contact_sheet_manifest_too_large")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError("xyz_contact_sheet_manifest_invalid")
    return payload


def _directory(value: Any) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_dir():
        raise ValueError("xyz_contact_sheet_directory_invalid")
    return path


def _contained_path(value: Any, roots: tuple[Path, ...], code: str) -> Path:
    path = Path(str(value)).resolve()
    if not roots or not any(path == root or root in path.parents for root in roots):
        raise ValueError(code)
    return path


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"xyz_contact_sheet_{name}_invalid")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"xyz_contact_sheet_{name}_invalid")
    return result


def _size_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("xyz_contact_sheet_bounded_size_invalid")
    return (
        _positive_int(value[0], "bounded_width"),
        _positive_int(value[1], "bounded_height"),
    )


def _checked_dimension(value: int) -> int:
    if value <= 0 or value > PNG_MAX_DIMENSION:
        raise ValueError("xyz_contact_sheet_dimension_overflow")
    return value


def _checked_bytes(width: int, height: int, channels: int) -> int:
    value = width * height * channels
    if value <= 0 or value > sys.maxsize:
        raise ValueError("xyz_contact_sheet_size_overflow")
    return value


def _require_disk_space(directory: Path, required: int) -> None:
    if shutil.disk_usage(directory).free < required:
        raise OSError("xyz_contact_sheet_insufficient_disk")


def _fill_white(handle, size_bytes: int) -> None:
    chunk = bytes([255]) * min(WRITE_CHUNK_BYTES, size_bytes)
    remaining = size_bytes
    while remaining:
        count = min(remaining, len(chunk))
        handle.write(chunk[:count])
        remaining -= count


def _write_image(handle, image: Image.Image, row_width: int, x: int, y: int) -> None:
    rgb = image.convert("RGB")
    for row in range(rgb.height):
        data = rgb.crop((0, row, rgb.width, row + 1)).tobytes()
        handle.seek(((y + row) * row_width + x) * 3)
        handle.write(data)


def _label_image(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (max(1, width), max(1, height)), "white")
    if not text.strip() or width <= 0 or height <= 0:
        return image
    draw = ImageDraw.Draw(image)
    font = _font(max(10, min(24, height // 2)))
    draw.multiline_text((width // 2, height // 2), text, fill="black", font=font, anchor="mm", align="center")
    return image


def _draw_identity_badge(source: Image.Image, text: str) -> Image.Image:
    image = source.copy()
    draw = ImageDraw.Draw(image)
    font = _font(max(8, min(24, image.height // 16)))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = max(1, bbox[2] - bbox[0])
    text_height = max(1, bbox[3] - bbox[1])
    padding_x = max(3, min(8, image.width // 40))
    padding_y = max(2, min(6, image.height // 60))
    badge_width = min(image.width, text_width + padding_x * 2)
    badge_height = min(image.height, text_height + padding_y * 2)
    draw.rounded_rectangle(
        (0, 0, max(0, badge_width - 1), max(0, badge_height - 1)),
        radius=min(5, badge_height // 3),
        fill=(16, 20, 24),
        outline=(255, 255, 255),
    )
    draw.text(
        (min(padding_x, badge_width // 2), min(padding_y, badge_height // 2)),
        text,
        fill="white",
        font=font,
    )
    return image


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/meiryo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _png_chunk(handle, kind: bytes, data: bytes) -> None:
    handle.write(struct.pack(">I", len(data)))
    handle.write(kind)
    handle.write(data)
    handle.write(struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage-row", "finalize"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = stage_contact_sheet_row(args.manifest) if args.mode == "stage-row" else finalize_contact_sheet(args.manifest)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
