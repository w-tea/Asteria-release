from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from pathlib import Path
import re
from typing import Any

from app.services.xyz_grid import bounded_contact_sheet_size


SUPPORTED_MODEL_FAMILIES = frozenset({"sdxl", "anima"})
SOURCE_CHANGING_TARGETS = frozenset(
    {
        "checkpoint",
        "model",
        "model_a",
        "model_b",
        "model_c",
        "model_slot:checkpoint",
        "diffusion_model",
        "model_slot:diffusion_model",
    }
)
NUMERIC_TARGETS = frozenset({"alpha", "beta"})
PNG_MAX_DIMENSION = (2**31) - 1


class MergeXYZValidationError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class MergeXYZNormalizedAxis:
    target: str
    label: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class MergeXYZPlan:
    axes: tuple[MergeXYZNormalizedAxis, ...]
    total_count: int
    model_family: str


@dataclass(frozen=True)
class MergeXYZCandidate:
    index: int
    candidate_id: str
    values: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MergeXYZResourceEstimate:
    candidate_count: int
    final_width: int
    final_height: int
    raw_rgb_staging_bytes: int
    temporary_bytes_lower_bound: int
    temporary_bytes_upper_bound: int
    source_size_lower_bound_bytes: int
    duration_seconds: float | None
    duration_label: str
    png_representable: bool


@dataclass(frozen=True)
class MergeXYZSourceIdentity:
    role: str
    path: str
    name: str
    size_bytes: int
    sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def normalize_merge_xyz_axes(
    axes: Sequence[Mapping[str, Any] | Any],
    *,
    allowed_targets: Set[str] | set[str] | frozenset[str],
    uses_beta: bool,
    merge_type: str,
    model_family: str,
    validate_values: bool = True,
) -> MergeXYZPlan:
    if merge_type.strip().lower() != "checkpoint":
        raise MergeXYZValidationError("xyz_precheck_checkpoint_merge_required")

    normalized_family = model_family.strip().lower()
    if normalized_family not in SUPPORTED_MODEL_FAMILIES:
        raise MergeXYZValidationError("xyz_precheck_model_family_not_supported")

    if not 1 <= len(axes) <= 3:
        raise MergeXYZValidationError("xyz_precheck_requires_1_to_3_axes")

    allowed = {str(target).strip().lower() for target in allowed_targets}
    normalized_axes: list[MergeXYZNormalizedAxis] = []
    for axis in axes:
        target = str(_axis_value(axis, "target", "")).strip().lower()
        if target not in allowed:
            raise MergeXYZValidationError("xyz_precheck_unknown_axis_target")
        if target in SOURCE_CHANGING_TARGETS:
            raise MergeXYZValidationError("xyz_precheck_source_axis_not_supported")
        if target in {"beta", "mbw_beta"} and not uses_beta:
            raise MergeXYZValidationError("xyz_precheck_noop_axis_target")

        raw_values = _axis_value(axis, "values", ())
        values = tuple(str(value).strip() for value in raw_values if str(value).strip())
        if not values:
            raise MergeXYZValidationError("xyz_precheck_axis_values_required")
        if validate_values:
            _validate_axis_values(target, values)

        label = str(_axis_value(axis, "label", "")).strip() or target
        normalized_axes.append(MergeXYZNormalizedAxis(target=target, label=label, values=values))

    total_count = math.prod(len(axis.values) for axis in normalized_axes)
    return MergeXYZPlan(axes=tuple(normalized_axes), total_count=total_count, model_family=normalized_family)


def iter_merge_xyz_candidates(plan: MergeXYZPlan) -> Iterator[MergeXYZCandidate]:
    value_pools = tuple(axis.values for axis in reversed(plan.axes))
    for index, reversed_values in enumerate(product(*value_pools)):
        values = tuple(reversed(reversed_values))
        yield MergeXYZCandidate(
            index=index,
            candidate_id=f"candidate-{index + 1:04d}",
            values=values,
            assignments=tuple((axis.target, value) for axis, value in zip(plan.axes, values, strict=True)),
        )


def resolve_merge_xyz_sources(recipe: Any, *, requires_source_c: bool) -> tuple[MergeXYZSourceIdentity, ...]:
    by_role = {
        str(getattr(source, "role", "")).strip().upper(): source
        for source in getattr(recipe, "sources", ())
        if str(getattr(source, "role", "")).strip()
    }
    required_roles = ("A", "B", "C") if requires_source_c else ("A", "B")
    identities: list[MergeXYZSourceIdentity] = []
    for role in required_roles:
        source = by_role.get(role)
        raw_path = str(getattr(source, "path", "") if source is not None else "").strip()
        if not raw_path:
            raise MergeXYZValidationError(f"xyz_precheck_source_{role.lower()}_required")
        path = Path(raw_path)
        try:
            logical = path.absolute()
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise OSError("source is not a file")
            if resolved.suffix.lower() != ".safetensors":
                raise MergeXYZValidationError("xyz_precheck_source_must_be_safetensors")
            size_bytes = resolved.stat().st_size
            digest = _file_sha256(resolved)
        except MergeXYZValidationError:
            raise
        except OSError as exc:
            raise MergeXYZValidationError(f"xyz_precheck_source_{role.lower()}_not_readable") from exc
        identities.append(
            MergeXYZSourceIdentity(
                role=role,
                path=str(logical),
                name=logical.name,
                size_bytes=size_bytes,
                sha256=digest,
            )
        )
    return tuple(identities)


def estimate_merge_xyz_resources(
    plan: MergeXYZPlan,
    *,
    tile_width: int,
    tile_height: int,
    margin: int = 0,
    source_sizes: Sequence[int] = (),
    annotation_band: int = 64,
    duration_seconds: float | None = None,
) -> MergeXYZResourceEstimate:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if margin < 0 or annotation_band < 0:
        raise ValueError("margin and annotation band must be non-negative")
    tile_width, tile_height = bounded_contact_sheet_size(tile_width, tile_height)

    counts = [len(axis.values) for axis in plan.axes]
    x_count = counts[0]
    y_count = counts[1] if len(counts) > 1 else 1
    z_count = counts[2] if len(counts) > 2 else 1

    left_label_width = tile_width * 3 // 4 if len(counts) > 1 else 0
    x_label_height = annotation_band
    plane_width = left_label_width + x_count * tile_width + max(0, x_count - 1) * margin
    plane_height = x_label_height + y_count * tile_height + max(0, y_count - 1) * margin
    final_width = plane_width * z_count
    final_height = plane_height + (annotation_band if len(counts) > 2 else 0)
    raw_rgb_staging_bytes = final_width * final_height * 3
    temporary_bytes_lower_bound = raw_rgb_staging_bytes
    temporary_bytes_upper_bound = raw_rgb_staging_bytes * 2 + final_height + plan.total_count * 4096
    source_size_lower_bound_bytes = sum(max(0, int(size)) for size in source_sizes)
    normalized_duration = duration_seconds if duration_seconds is not None and duration_seconds >= 0 else None

    return MergeXYZResourceEstimate(
        candidate_count=plan.total_count,
        final_width=final_width,
        final_height=final_height,
        raw_rgb_staging_bytes=raw_rgb_staging_bytes,
        temporary_bytes_lower_bound=temporary_bytes_lower_bound,
        temporary_bytes_upper_bound=temporary_bytes_upper_bound,
        source_size_lower_bound_bytes=source_size_lower_bound_bytes,
        duration_seconds=normalized_duration,
        duration_label=f"{normalized_duration:.1f}s" if normalized_duration is not None else "unknown",
        png_representable=final_width <= PNG_MAX_DIMENSION and final_height <= PNG_MAX_DIMENSION,
    )


def _axis_value(axis: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(axis, Mapping):
        return axis.get(name, default)
    return getattr(axis, name, default)


def _validate_axis_values(target: str, values: Sequence[str]) -> None:
    if target in NUMERIC_TARGETS:
        try:
            parsed = [float(value) for value in values]
        except ValueError as exc:
            raise MergeXYZValidationError("xyz_precheck_invalid_numeric_value") from exc
        if not all(math.isfinite(value) for value in parsed):
            raise MergeXYZValidationError("xyz_precheck_invalid_numeric_value")
        return

    if target == "seed":
        if any(re.fullmatch(r"[+-]?\d+", value) is None for value in values):
            raise MergeXYZValidationError("xyz_precheck_invalid_numeric_value")
        return

    if target == "alpha_beta":
        for value in values:
            parts = [part for part in re.split(r"[,/:\s]+", value) if part]
            try:
                parsed = [float(part) for part in parts]
            except ValueError as exc:
                raise MergeXYZValidationError("xyz_precheck_alpha_beta_pair_required") from exc
            if len(parsed) != 2 or not all(math.isfinite(part) for part in parsed):
                raise MergeXYZValidationError("xyz_precheck_alpha_beta_pair_required")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
