from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math
from collections.abc import Sequence
import shutil
import subprocess
import sys
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class XYZGridAxis:
    key: str
    label: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class XYZGridCell:
    labels: dict[str, str]
    image_path: Path | None


class XYZContactSheetError(RuntimeError):
    pass


class XYZContactSheetStager:
    def __init__(
        self,
        *,
        axes: Sequence[XYZGridAxis],
        output_dir: Path,
        operation_id: str,
        margin: int = 0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.axes = tuple(axes[:3])
        self.output_dir = output_dir.resolve()
        self.operation_id = "".join(character for character in operation_id if character.isalnum() or character in "-_") or uuid4().hex
        self.margin = max(0, int(margin))
        self.timeout_seconds = timeout_seconds
        self.staging_dir = self.output_dir / "_xyz-staging" / self.operation_id
        self.staging_dir.mkdir(parents=True, exist_ok=False)
        self.warning_code = ""
        self.staged_row_count = 0
        self._buffer: list[XYZGridCell] = []
        self._rows: list[list[dict[str, object]]] = [[] for _ in self._z_values]
        self._tile_width = 0
        self._tile_height = 0
        self._plane_width = 0
        self._closed = False

    @property
    def _x_axis(self) -> XYZGridAxis:
        return self.axes[0] if self.axes else XYZGridAxis("", "", ("",))

    @property
    def _y_axis(self) -> XYZGridAxis:
        return self.axes[1] if len(self.axes) > 1 else XYZGridAxis("", "", ("",))

    @property
    def _z_axis(self) -> XYZGridAxis:
        return self.axes[2] if len(self.axes) > 2 else XYZGridAxis("", "", ("",))

    @property
    def _z_values(self) -> tuple[str, ...]:
        return self._z_axis.values or ("",)

    def add_cell(self, cell: XYZGridCell) -> None:
        if self.warning_code or self._closed:
            return
        self._buffer.append(cell)
        if len(self._buffer) < len(self._x_axis.values):
            return
        try:
            self._stage_buffered_row()
        except (OSError, ValueError, MemoryError, subprocess.SubprocessError, XYZContactSheetError):
            self.warning_code = "contact_sheet_creation_failed"
        finally:
            self._buffer.clear()

    def finalize(self) -> Path | None:
        if self.warning_code or self._closed:
            return None
        if self._buffer or self.staged_row_count != len(self._y_axis.values) * len(self._z_values):
            self.warning_code = "contact_sheet_creation_failed"
            return None
        output_path = self.output_dir / f"xyz_{self.operation_id}.png"
        z_header_height = max(16, min(64, self._tile_height // 8)) if self._z_axis.key else 0
        manifest = {
            "schema": "asteria.xyz-contact-sheet-final.v1",
            "staging_root": str(self.staging_dir),
            "output_path": str(output_path),
            "plane_width": self._plane_width,
            "tile_width": self._tile_width,
            "z_labels": list(self._z_values),
            "z_header_height": z_header_height,
            "rows": self._rows,
        }
        manifest_path = self.staging_dir / "final.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        try:
            result = self._run_child("finalize", manifest_path)
            resolved = Path(str(result.get("output_path") or "")).resolve()
            if resolved != output_path.resolve() or not resolved.is_file():
                raise XYZContactSheetError("contact sheet child returned an invalid output")
            return resolved
        except (OSError, ValueError, MemoryError, subprocess.SubprocessError, XYZContactSheetError):
            self.warning_code = "contact_sheet_creation_failed"
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        parent = self.staging_dir.parent
        try:
            parent.rmdir()
        except OSError:
            pass

    def _stage_buffered_row(self) -> None:
        if len(self._buffer) != len(self._x_axis.values):
            raise XYZContactSheetError("incomplete X row")
        image_paths: list[Path] = []
        for cell in self._buffer:
            if cell.image_path is None:
                raise XYZContactSheetError("candidate image is missing")
            image_path = cell.image_path.resolve()
            if self.output_dir not in image_path.parents:
                raise XYZContactSheetError("candidate image is outside output root")
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            if self._tile_width == 0:
                self._tile_width, self._tile_height = width, height
            elif width > self._tile_width or height > self._tile_height:
                raise XYZContactSheetError("candidate image exceeds the first tile size")
            image_paths.append(image_path)
        x_count = len(self._x_axis.values)
        y_count = len(self._y_axis.values)
        row_index = self.staged_row_count
        y_index = row_index % y_count
        z_index = row_index // y_count
        if z_index >= len(self._z_values):
            raise XYZContactSheetError("too many candidate rows")
        left_width = self._tile_width * 3 // 4 if self._y_axis.key else 0
        top_height = max(16, min(64, self._tile_height // 8)) if y_index == 0 else 0
        row_path = self.staging_dir / f"row-z{z_index:04d}-y{y_index:04d}.rgb"
        manifest = {
            "schema": "asteria.xyz-contact-sheet-row.v1",
            "staging_root": str(self.staging_dir),
            "allowed_roots": [str(self.output_dir)],
            "output_path": str(row_path),
            "images": [str(path) for path in image_paths],
            "x_labels": list(self._x_axis.values),
            "y_label": f"{self._y_axis.label or self._y_axis.key}:\n{self._y_axis.values[y_index]}" if self._y_axis.key else "",
            "tile_width": self._tile_width,
            "tile_height": self._tile_height,
            "left_width": left_width,
            "top_height": top_height,
            "margin": self.margin,
        }
        manifest_path = self.staging_dir / f"row-z{z_index:04d}-y{y_index:04d}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        result = self._run_child("stage-row", manifest_path)
        width = int(result.get("width") or 0)
        height = int(result.get("height") or 0)
        if width <= 0 or height <= 0 or not row_path.is_file():
            raise XYZContactSheetError("contact sheet child returned an invalid row")
        if self._plane_width and width != self._plane_width:
            raise XYZContactSheetError("contact sheet row width changed")
        self._plane_width = width
        self._rows[z_index].append({"path": str(row_path), "height": height})
        self.staged_row_count += 1

    def _run_child(self, mode: str, manifest_path: Path) -> dict[str, object]:
        backend_dir = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "-m", "app.cli.xyz_contact_sheet", mode, str(manifest_path)],
            cwd=backend_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise XYZContactSheetError("contact sheet child failed")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise XYZContactSheetError("contact sheet child returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise XYZContactSheetError("contact sheet child returned invalid JSON")
        return payload


def create_xyz_contact_sheet_streaming(
    cells: Sequence[XYZGridCell],
    axes: Sequence[XYZGridAxis],
    output_dir: Path,
    *,
    operation_id: str | None = None,
) -> tuple[str, str]:
    stager = XYZContactSheetStager(
        axes=axes,
        output_dir=output_dir,
        operation_id=operation_id or uuid4().hex,
    )
    try:
        for cell in cells:
            stager.add_cell(cell)
        output = stager.finalize()
        return (str(output) if output else "", stager.warning_code)
    finally:
        stager.close()


def build_xyz_contact_sheet(cells: Sequence[XYZGridCell], axes: Sequence[XYZGridAxis], margin: int = 0) -> Image.Image:
    images = _load_cell_images(cells)
    if not images:
        return Image.new("RGB", (1, 1), "white")

    tile_width, tile_height = map(max, zip(*(image.size for image in images.values()), strict=True))
    normalized_axes = list(axes[:3])
    if not normalized_axes:
        return _plain_grid([images[index] for index in sorted(images)], tile_width, tile_height)

    x_axis = normalized_axes[0]
    y_axis = normalized_axes[1] if len(normalized_axes) > 1 else XYZGridAxis("", "", ("",))
    z_axis = normalized_axes[2] if len(normalized_axes) > 2 else XYZGridAxis("", "", ("",))

    sub_grids: list[Image.Image] = []
    for z_value in z_axis.values:
        sub_grid = _sub_grid(cells, images, x_axis, y_axis, z_axis, z_value, tile_width, tile_height)
        sub_grids.append(_draw_annotations(sub_grid, tile_width, tile_height, _x_labels(x_axis), _y_labels(y_axis), margin))

    sub_tile_width = max(image.width for image in sub_grids)
    sub_tile_height = max(image.height for image in sub_grids)
    result = _plain_grid(sub_grids, sub_tile_width, sub_tile_height, rows=1)
    if z_axis.key:
        result = _draw_annotations(result, sub_tile_width, sub_tile_height, _x_labels(z_axis), [""], margin=0)
    return result


def _load_cell_images(cells: Sequence[XYZGridCell]) -> dict[int, Image.Image]:
    loaded: dict[int, Image.Image] = {}
    for index, cell in enumerate(cells):
        if cell.image_path is None:
            continue
        try:
            with Image.open(cell.image_path) as image:
                loaded[index] = image.convert("RGB").copy()
        except OSError:
            continue
    return loaded


def _plain_grid(images: Sequence[Image.Image], tile_width: int, tile_height: int, rows: int | None = None) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), "white")
    if rows is None:
        rows = max(1, round(math.sqrt(len(images))))
    rows = min(rows, len(images))
    columns = math.ceil(len(images) / rows)
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, image in enumerate(images):
        x = (index % columns) * tile_width + (tile_width - image.width) // 2
        y = (index // columns) * tile_height + (tile_height - image.height) // 2
        grid.paste(image, (x, y))
    return grid


def _sub_grid(
    cells: Sequence[XYZGridCell],
    images: dict[int, Image.Image],
    x_axis: XYZGridAxis,
    y_axis: XYZGridAxis,
    z_axis: XYZGridAxis,
    z_value: str,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    grid = Image.new("RGB", (len(x_axis.values) * tile_width, len(y_axis.values) * tile_height), "white")
    cell_lookup = {_cell_key(cell, x_axis, y_axis, z_axis): index for index, cell in enumerate(cells)}
    for y_index, y_value in enumerate(y_axis.values):
        for x_index, x_value in enumerate(x_axis.values):
            cell_index = cell_lookup.get((x_value, y_value, z_value))
            if cell_index is None or cell_index not in images:
                continue
            image = images[cell_index]
            x = x_index * tile_width + (tile_width - image.width) // 2
            y = y_index * tile_height + (tile_height - image.height) // 2
            grid.paste(image, (x, y))
    return grid


def _cell_key(cell: XYZGridCell, x_axis: XYZGridAxis, y_axis: XYZGridAxis, z_axis: XYZGridAxis) -> tuple[str, str, str]:
    return (
        cell.labels.get(x_axis.key, "") if x_axis.key else "",
        cell.labels.get(y_axis.key, "") if y_axis.key else "",
        cell.labels.get(z_axis.key, "") if z_axis.key else "",
    )


def _x_labels(axis: XYZGridAxis) -> list[str]:
    return [str(value) for value in axis.values]


def _y_labels(axis: XYZGridAxis) -> list[str]:
    if not axis.key:
        return [""]
    label = axis.label or axis.key
    return [f"{label}:\n{value}" for value in axis.values]


def _draw_annotations(
    image: Image.Image,
    tile_width: int,
    tile_height: int,
    horizontal_labels: Sequence[str],
    vertical_labels: Sequence[str],
    margin: int,
) -> Image.Image:
    has_horizontal = any(label.strip() for label in horizontal_labels)
    has_vertical = any(label.strip() for label in vertical_labels)
    font_size = max(12, (tile_width + tile_height) // 25)
    line_spacing = max(4, font_size // 2)
    font = _font(font_size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    wrapped_horizontal = [_wrap_label(probe, label, font, tile_width) for label in horizontal_labels]
    left_width = tile_width * 3 // 4 if has_vertical else 0
    wrapped_vertical = [_wrap_label(probe, label, font, max(1, left_width)) for label in vertical_labels]
    top_height = _max_label_height(probe, wrapped_horizontal, font, line_spacing) + line_spacing * 2 if has_horizontal else 0
    output = Image.new(
        "RGB",
        (
            image.width + left_width + margin * max(0, len(horizontal_labels) - 1),
            image.height + top_height + margin * max(0, len(vertical_labels) - 1),
        ),
        "white",
    )
    draw = ImageDraw.Draw(output)
    rows = max(1, len(vertical_labels))
    columns = max(1, len(horizontal_labels))
    for row in range(rows):
        for column in range(columns):
            source = image.crop((tile_width * column, tile_height * row, tile_width * (column + 1), tile_height * (row + 1)))
            output.paste(source, (left_width + (tile_width + margin) * column, top_height + (tile_height + margin) * row))
    for column, lines in enumerate(wrapped_horizontal):
        x = left_width + (tile_width + margin) * column + tile_width // 2
        _draw_centered_lines(draw, lines, x, top_height // 2, font, line_spacing)
    for row, lines in enumerate(wrapped_vertical):
        y = top_height + (tile_height + margin) * row + tile_height // 2
        _draw_centered_lines(draw, lines, left_width // 2, y, font, line_spacing)
    return output


def _wrap_label(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.ImageFont, allowed_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(label).splitlines() or [""]:
        current = ""
        for word in raw_line.split():
            candidate = f"{current} {word}".strip()
            if not current or draw.textlength(candidate, font=font) <= allowed_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _max_label_height(draw: ImageDraw.ImageDraw, labels: Sequence[Sequence[str]], font: ImageFont.ImageFont, line_spacing: int) -> int:
    heights = [_label_height(draw, lines, font, line_spacing) for lines in labels]
    return max(heights, default=0)


def _label_height(draw: ImageDraw.ImageDraw, lines: Sequence[str], font: ImageFont.ImageFont, line_spacing: int) -> int:
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        heights.append(bbox[3] - bbox[1])
    return sum(heights) + line_spacing * max(0, len(heights) - 1)


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    center_x: int,
    center_y: int,
    font: ImageFont.ImageFont,
    line_spacing: int,
) -> None:
    total_height = _label_height(draw, lines, font, line_spacing)
    y = center_y - total_height / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        height = bbox[3] - bbox[1]
        draw.text((center_x, y), line, fill="black", font=font, anchor="ma")
        y += height + line_spacing


def _font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/meiryo.ttc"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()
