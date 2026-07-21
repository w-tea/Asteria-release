from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from collections.abc import Sequence

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
