from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from app.services import api_errors


@dataclass(frozen=True)
class InpaintRegion:
    rect: tuple[int, int, int, int]
    canvas_size: tuple[int, int]
    padding: int

    @property
    def size(self) -> tuple[int, int]:
        left, top, right, bottom = self.rect
        return right - left, bottom - top


@dataclass(frozen=True)
class InpaintRegionAnalysis:
    source_path: Path
    mask_path: Path
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]
    requested_size: tuple[int, int]
    region: InpaintRegion
    canonical_source_png: bytes
    positive_mask_png: bytes

    def open_canonical_source(self) -> Image.Image:
        return Image.open(BytesIO(self.canonical_source_png))

    def open_positive_mask(self) -> Image.Image:
        return Image.open(BytesIO(self.positive_mask_png))


@dataclass(frozen=True)
class PreparedInpaintRegion:
    region: InpaintRegion
    source_path: Path
    mask_path: Path
    canonical_source_path: Path
    init_crop_path: Path
    positive_mask_crop_path: Path
    comfy_mask_crop_path: Path
    metadata: dict[str, object]

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.region.canvas_size


class InpaintRegionError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _apply_exif_orientation(image: Image.Image, orientation: int) -> Image.Image:
    operation = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    return image.transpose(operation) if operation is not None else image.copy()


def _positive_mask(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        luminance = ImageOps.grayscale(image.convert("RGB"))
        if alpha.getextrema() == (255, 255):
            return luminance
        return Image.composite(luminance, alpha, alpha)
    return ImageOps.grayscale(image.convert("RGB"))


def _aligned_axis_interval(
    start: int,
    end: int,
    *,
    canvas_length: int,
    padding: int,
) -> tuple[int, int]:
    if canvas_length <= 0 or start < 0 or end <= start or end > canvas_length:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
    padded_start = max(0, start - padding)
    padded_end = min(canvas_length, end + padding)
    minimum = min(256, canvas_length)
    target_length = max(padded_end - padded_start, minimum)
    target_length = min(canvas_length, ((target_length + 7) // 8) * 8)
    extra = target_length - (padded_end - padded_start)
    resolved_start = padded_start - extra // 2
    resolved_end = padded_end + (extra - extra // 2)
    if resolved_start < 0:
        resolved_start = 0
        resolved_end = target_length
    elif resolved_end > canvas_length:
        resolved_end = canvas_length
        resolved_start = canvas_length - target_length
    if (
        resolved_start < 0
        or resolved_end > canvas_length
        or resolved_end - resolved_start != target_length
        or target_length % 8 != 0
    ):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
    return resolved_start, resolved_end


def compute_inpaint_region(
    positive_mask: Image.Image,
    *,
    canvas_size: tuple[int, int],
    padding: int,
) -> InpaintRegion:
    if positive_mask.size != canvas_size:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_SIZE_MISMATCH)
    bounds = positive_mask.getbbox()
    if bounds is None:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_EMPTY_MASK)
    left, top, right, bottom = bounds
    resolved_left, resolved_right = _aligned_axis_interval(
        left,
        right,
        canvas_length=canvas_size[0],
        padding=padding,
    )
    resolved_top, resolved_bottom = _aligned_axis_interval(
        top,
        bottom,
        canvas_length=canvas_size[1],
        padding=padding,
    )
    return InpaintRegion(
        rect=(resolved_left, resolved_top, resolved_right, resolved_bottom),
        canvas_size=canvas_size,
        padding=padding,
    )


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def validate_only_masked_inputs(
    source_path: str | Path,
    mask_path: str | Path,
    *,
    mask_blur: int = 0,
    requested_size: tuple[int, int] | None = None,
) -> InpaintRegionAnalysis:
    source = Path(source_path)
    mask = Path(mask_path)
    try:
        with Image.open(source) as source_image:
            source_image.load()
            source_size = source_image.size
            source_orientation = int(source_image.getexif().get(274, 1))
            canonical_source = ImageOps.exif_transpose(source_image).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_SOURCE_UNAVAILABLE) from None
    if not (256 <= source_size[0] <= 4096 and 256 <= source_size[1] <= 4096):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
    try:
        with Image.open(mask) as mask_image:
            mask_image.load()
            raw_mask_size = mask_image.size
            positive_mask = _positive_mask(_apply_exif_orientation(mask_image, source_orientation))
    except (OSError, UnidentifiedImageError, ValueError):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_MASK_UNAVAILABLE) from None
    if raw_mask_size != source_size:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_SIZE_MISMATCH)
    if positive_mask.size != canonical_source.size:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_SIZE_MISMATCH)

    effective_size = (
        canonical_source.width // 8 * 8,
        canonical_source.height // 8 * 8,
    )
    if effective_size[0] < 256 or effective_size[1] < 256:
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
    if canonical_source.size != effective_size:
        canonical_source = canonical_source.resize(effective_size, Image.Resampling.LANCZOS)
        positive_mask = positive_mask.resize(effective_size, Image.Resampling.NEAREST)
    padding = max(32, max(0, int(mask_blur)))
    region = compute_inpaint_region(
        positive_mask,
        canvas_size=effective_size,
        padding=padding,
    )
    requested = requested_size or (0, 0)
    return InpaintRegionAnalysis(
        source_path=source,
        mask_path=mask,
        source_size=source_size,
        canvas_size=effective_size,
        requested_size=(int(requested[0]), int(requested[1])),
        region=region,
        canonical_source_png=_encode_png(canonical_source),
        positive_mask_png=_encode_png(positive_mask),
    )


def prepare_only_masked_region(
    source_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    mask_blur: int = 0,
    requested_size: tuple[int, int] | None = None,
    analysis: InpaintRegionAnalysis | None = None,
) -> PreparedInpaintRegion:
    resolved = analysis or validate_only_masked_inputs(
        source_path,
        mask_path,
        mask_blur=mask_blur,
        requested_size=requested_size,
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex
    canonical_source_path = directory / f"inpaint_only_masked_source_{suffix}.png"
    init_crop_path = directory / f"inpaint_only_masked_init_{suffix}.png"
    positive_mask_crop_path = directory / f"inpaint_only_masked_positive_mask_{suffix}.png"
    comfy_mask_crop_path = directory / f"inpaint_only_masked_comfy_mask_{suffix}.png"
    with resolved.open_canonical_source() as canonical_source, resolved.open_positive_mask() as positive_mask:
        canonical_source = canonical_source.convert("RGB")
        positive_mask = positive_mask.convert("L")
        init_crop = canonical_source.crop(resolved.region.rect)
        positive_crop = positive_mask.crop(resolved.region.rect)
        comfy_mask = Image.new("RGBA", resolved.region.size, (255, 255, 255, 0))
        comfy_mask.putalpha(ImageOps.invert(positive_crop))
        canonical_source.save(canonical_source_path, format="PNG")
        init_crop.save(init_crop_path, format="PNG")
        positive_crop.save(positive_mask_crop_path, format="PNG")
        comfy_mask.save(comfy_mask_crop_path, format="PNG")

    crop_width, crop_height = resolved.region.size
    left, top, _, _ = resolved.region.rect
    metadata: dict[str, object] = {
        "requested_base_size": {
            "width": resolved.requested_size[0],
            "height": resolved.requested_size[1],
        },
        "source_init_size": {
            "width": resolved.source_size[0],
            "height": resolved.source_size[1],
        },
        "effective_base_size": {
            "width": resolved.canvas_size[0],
            "height": resolved.canvas_size[1],
        },
        "base_size_runtime": "source_image_dimensions_aligned_to_vae",
        "inpaint_only_masked_padding": resolved.region.padding,
        "inpaint_crop_rect": {
            "x": left,
            "y": top,
            "width": crop_width,
            "height": crop_height,
        },
        "inpaint_crop_size": {"width": crop_width, "height": crop_height},
        "inpaint_full_size": {
            "width": resolved.canvas_size[0],
            "height": resolved.canvas_size[1],
        },
    }
    return PreparedInpaintRegion(
        region=resolved.region,
        source_path=resolved.source_path,
        mask_path=resolved.mask_path,
        canonical_source_path=canonical_source_path,
        init_crop_path=init_crop_path,
        positive_mask_crop_path=positive_mask_crop_path,
        comfy_mask_crop_path=comfy_mask_crop_path,
        metadata=metadata,
    )


def compose_only_masked_output(
    output_path: str | Path,
    *,
    canonical_source_path: str | Path,
    positive_mask_crop_path: str | Path,
    crop_rect: object,
) -> None:
    if not isinstance(crop_rect, dict):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
    try:
        left = int(crop_rect["x"])
        top = int(crop_rect["y"])
        width = int(crop_rect["width"])
        height = int(crop_rect["height"])
    except (KeyError, TypeError, ValueError):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID) from None
    try:
        with Image.open(canonical_source_path).convert("RGB") as canonical_source, Image.open(
            positive_mask_crop_path
        ).convert("L") as positive_mask, Image.open(output_path).convert("RGB") as generated:
            if (
                left < 0
                or top < 0
                or width <= 0
                or height <= 0
                or left + width > canonical_source.width
                or top + height > canonical_source.height
                or positive_mask.size != (width, height)
            ):
                raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
            if generated.size != (width, height):
                generated = generated.resize((width, height), Image.Resampling.LANCZOS)
            source_crop = canonical_source.crop((left, top, left + width, top + height))
            composited_crop = Image.composite(generated, source_crop, positive_mask)
            final = canonical_source.copy()
            final.paste(composited_crop, (left, top))
            final.save(output_path, format="PNG")
    except InpaintRegionError:
        raise
    except (OSError, UnidentifiedImageError, ValueError):
        raise InpaintRegionError(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID) from None
