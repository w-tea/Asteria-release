from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError

from app.schemas.prompt_layout import SpatialAssetReceipt, SpatialMaskSource
from app.services.image_validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    InvalidImageContentError,
    validate_image_bytes,
)


class SpatialAssetResolutionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResolvedSpatialAsset:
    path: Path
    payload: bytes
    width: int
    height: int
    mime_type: str


@dataclass(frozen=True, slots=True)
class ResolvedSpatialMaskSource:
    asset: ResolvedSpatialAsset
    width: int
    height: int
    binary_mask: bytes
    bounds: tuple[int, int, int, int]


def _receipt_error() -> SpatialAssetResolutionError:
    return SpatialAssetResolutionError("spatial_asset_receipt_mismatch")


def _contains_reparse_component(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    current = path
    while True:
        try:
            details = os.lstat(current)
            attributes = getattr(details, "st_file_attributes", 0)
            junction_check = getattr(os.path, "isjunction", None)
            is_junction = bool(junction_check(current)) if junction_check is not None else False
        except OSError:
            return True
        if (
            stat.S_ISLNK(details.st_mode)
            or is_junction
            or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        ):
            return True
        if current == root:
            return False
        current = current.parent


def resolve_spatial_asset(
    receipt: SpatialAssetReceipt,
    data_dir: str | Path,
) -> ResolvedSpatialAsset:
    root = Path(data_dir)
    relative = PurePosixPath(receipt.relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "control_images"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise _receipt_error()
    try:
        data_root = root.resolve()
        control_root = data_root / "control_images"
        candidate = data_root.joinpath(*relative.parts)
        if _contains_reparse_component(candidate, control_root):
            raise _receipt_error()
        resolved_control_root = control_root.resolve()
        path = candidate.resolve()
    except OSError:
        raise _receipt_error() from None
    if path == resolved_control_root or resolved_control_root not in path.parents:
        raise _receipt_error()
    try:
        if not path.is_file():
            raise _receipt_error()
        with path.open("rb") as source:
            file_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > DEFAULT_MAX_IMAGE_BYTES:
                raise _receipt_error()
            payload = source.read(DEFAULT_MAX_IMAGE_BYTES + 1)
        if root.joinpath(*relative.parts).resolve() != path or not path.is_file():
            raise _receipt_error()
    except OSError:
        raise SpatialAssetResolutionError("spatial_asset_unavailable") from None
    if len(payload) != receipt.bytes or hashlib.sha256(payload).hexdigest() != receipt.sha256:
        raise _receipt_error()
    try:
        validate_image_bytes(payload, path.suffix)
        with Image.open(BytesIO(payload)) as image:
            image.seek(0)
            image.load()
            width, height = image.size
            mime_type = Image.MIME.get(image.format or "", "")
    except (InvalidImageContentError, OSError, UnidentifiedImageError, ValueError):
        raise _receipt_error() from None
    if (
        (width, height) != (receipt.width, receipt.height)
        or mime_type != receipt.mime_type
        or receipt.asset_id != f"control-image:sha256:{receipt.sha256}"
    ):
        raise _receipt_error()
    return ResolvedSpatialAsset(
        path=path,
        payload=payload,
        width=width,
        height=height,
        mime_type=mime_type,
    )


def decode_spatial_asset_rgba(asset: ResolvedSpatialAsset) -> Image.Image:
    try:
        with Image.open(BytesIO(asset.payload)) as opened:
            opened.seek(0)
            opened.load()
            icc_profile = opened.info.get("icc_profile")
            if icc_profile:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                destination_profile = ImageCms.createProfile("sRGB")
                return ImageCms.profileToProfile(
                    opened,
                    source_profile,
                    destination_profile,
                    outputMode="RGBA",
                )
            return opened.convert("RGBA")
    except (ImageCms.PyCMSError, OSError, UnidentifiedImageError, ValueError):
        raise _receipt_error() from None


def extract_spatial_binary_mask(
    rgba: Image.Image,
    *,
    source_rect: tuple[int, int, int, int],
    source_color: str,
) -> tuple[tuple[int, int, int, int], bytes]:
    left, top, right, bottom = source_rect
    expected = np.asarray(
        [int(source_color[index : index + 2], 16) for index in (1, 3, 5)],
        dtype=np.uint8,
    )
    minimum_x = right
    minimum_y = bottom
    maximum_x = left - 1
    maximum_y = top - 1
    for y in range(top, bottom):
        row = np.asarray(rgba.crop((left, y, right, y + 1)), dtype=np.uint8).reshape(-1, 4)
        matches = np.all(row[:, :3] == expected, axis=1) & (row[:, 3] > 0)
        indexes = np.flatnonzero(matches)
        if indexes.size:
            minimum_x = min(minimum_x, left + int(indexes[0]))
            maximum_x = max(maximum_x, left + int(indexes[-1]))
            minimum_y = min(minimum_y, y)
            maximum_y = y
    if maximum_x < minimum_x or maximum_y < minimum_y:
        raise SpatialAssetResolutionError("spatial_mask_color_not_found")
    bounds = (minimum_x, minimum_y, maximum_x + 1, maximum_y + 1)
    compact = bytearray()
    for y in range(bounds[1], bounds[3]):
        row = np.asarray(
            rgba.crop((bounds[0], y, bounds[2], y + 1)),
            dtype=np.uint8,
        ).reshape(-1, 4)
        matches = np.all(row[:, :3] == expected, axis=1) & (row[:, 3] > 0)
        compact.extend(matches.astype(np.uint8).tobytes())
    return bounds, bytes(compact)


def resolve_spatial_mask_source(
    source: SpatialMaskSource,
    data_dir: str | Path,
) -> ResolvedSpatialMaskSource:
    asset = resolve_spatial_asset(source.asset, data_dir)
    rgba = decode_spatial_asset_rgba(asset)
    rect = source.source_rect_px
    bounds, compact_binary = extract_spatial_binary_mask(
        rgba,
        source_rect=(rect.x, rect.y, rect.right, rect.bottom),
        source_color=source.source_color,
    )
    declared = source.source_bounds_px
    declared_bounds = (declared.x, declared.y, declared.right, declared.bottom)
    if bounds != declared_bounds or hashlib.sha256(compact_binary).hexdigest() != source.binary_mask_sha256:
        raise _receipt_error()
    return ResolvedSpatialMaskSource(
        asset=asset,
        width=bounds[2] - bounds[0],
        height=bounds[3] - bounds[1],
        binary_mask=compact_binary,
        bounds=bounds,
    )
