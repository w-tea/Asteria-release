from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.config import AppConfig
from app.services.atomic_json import write_json_atomic

SD15_BLOCKS = [
    "BASE",
    *[f"IN{x:02d}" for x in range(12)],
    "MID",
    *[f"OUT{x:02d}" for x in range(12)],
]

SDXL_BLOCKS = [
    "BASE",
    *[f"IN{x:02d}" for x in range(17)],
    "M00",
    *[f"OUT{x:02d}" for x in range(17)],
]

ANIMA_BLOCKS = [
    "BASE",
    *[f"B{x:03d}" for x in range(28)],
    *[f"LLM{x:03d}" for x in range(6)],
    "OUT",
]

ANIMA_2_9B_40_BLOCKS = [
    "BASE",
    *[f"B{x:03d}" for x in range(40)],
    *[f"LLM{x:03d}" for x in range(6)],
    "OUT",
]

ANIMA_3_8B_52_BLOCKS = [
    "BASE",
    *[f"B{x:03d}" for x in range(52)],
    *[f"LLM{x:03d}" for x in range(6)],
    "OUT",
]

ANIMA_SCHEMA_IDS = (
    "anima",
    "anima_2_9b_40",
    "anima_3_8b_52",
)

FLUX1_BLOCKS = [
    "CLIP",
    "T5",
    "IN",
    *[f"D{x:03d}" for x in range(19)],
    *[f"S{x:03d}" for x in range(38)],
    "OUT",
]

FLUX2_BLOCKS = [
    "CLIP",
    "T5",
    "IN",
    *[f"D{x:03d}" for x in range(8)],
    *[f"S{x:03d}" for x in range(24)],
    "OUT",
]

FLUX_SCHEMA_IDS = ("flux1", "flux2")
LEGACY_FLUX_SCHEMA_ID = "flux"

BLOCK_SCHEMAS = {
    "sd15_sd2": SD15_BLOCKS,
    "sdxl": SDXL_BLOCKS,
    "anima": ANIMA_BLOCKS,
    "anima_2_9b_40": ANIMA_2_9B_40_BLOCKS,
    "anima_3_8b_52": ANIMA_3_8B_52_BLOCKS,
    "flux1": FLUX1_BLOCKS,
    "flux2": FLUX2_BLOCKS,
    LEGACY_FLUX_SCHEMA_ID: FLUX1_BLOCKS,
}


def is_anima_schema(schema: str) -> bool:
    return schema in ANIMA_SCHEMA_IDS


def merge_schema_family(schema: str) -> str:
    if is_anima_schema(schema):
        return "anima"
    return canonical_merge_schema(schema)


def is_flux_schema(schema: str) -> bool:
    return schema in {*FLUX_SCHEMA_IDS, LEGACY_FLUX_SCHEMA_ID}


def canonical_merge_schema(schema: str) -> str:
    return "flux1" if schema == LEGACY_FLUX_SCHEMA_ID else schema


def migrate_legacy_flux2_weights(weights: list[float]) -> list[float]:
    if len(weights) != len(FLUX1_BLOCKS):
        return weights
    by_block = dict(zip(FLUX1_BLOCKS, weights, strict=True))
    return [by_block[block] for block in FLUX2_BLOCKS]


def migrate_legacy_flux2_weight_text(value: str) -> str:
    try:
        weights = _parse_weights(value)
    except ValueError:
        return value
    if len(weights) != len(FLUX1_BLOCKS):
        return value
    return ",".join(f"{weight:g}" for weight in migrate_legacy_flux2_weights(weights))


class MergeBlockPreset(BaseModel):
    id: str
    name: str
    family: str
    weights: list[float]
    source: str = "builtin"
    description: str = ""
    derived_from: str = ""
    compatible_schemas: list[str] = Field(default_factory=list)


def _parse_weights(value: str) -> list[float]:
    weights: list[float] = []
    for item in value.replace("\n", ",").split(","):
        text = item.strip()
        if not text:
            continue
        weights.append(float(text))
    return weights


def _resample(values: list[float], target_length: int) -> list[float]:
    if not values or target_length <= 0:
        return []
    if len(values) == target_length:
        return values
    if target_length == 1:
        return [values[0]]
    if len(values) == 1:
        return [values[0]] * target_length
    result: list[float] = []
    source_span = len(values) - 1
    target_span = target_length - 1
    for index in range(target_length):
        position = index * source_span / target_span
        low = int(position)
        high = min(low + 1, len(values) - 1)
        ratio = position - low
        result.append(values[low] * (1.0 - ratio) + values[high] * ratio)
    return result


def _sd15_groups(weights: list[float]) -> tuple[float, list[float], float, list[float]]:
    return weights[0], weights[1:13], weights[13], weights[14:26]


def _sdxl_from_sd15(weights: list[float]) -> list[float]:
    base, input_blocks, middle, output_blocks = _sd15_groups(weights)
    return [
        base,
        *_resample(input_blocks, 17),
        middle,
        *_resample(output_blocks, 17),
    ]


def _anima_from_sd15(weights: list[float], *, diffusion_blocks: int = 28) -> list[float]:
    base, input_blocks, middle, output_blocks = _sd15_groups(weights)
    diffusion_curve = [*input_blocks, middle, *output_blocks]
    text_curve = _resample([base, *input_blocks[:3], middle], 6)
    return [
        base,
        *_resample(diffusion_curve, diffusion_blocks),
        *text_curve,
        output_blocks[-1],
    ]


def _flux_from_sd15(weights: list[float], *, double_blocks: int, single_blocks: int) -> list[float]:
    base, input_blocks, middle, output_blocks = _sd15_groups(weights)
    double_curve = [*input_blocks, middle]
    single_curve = [middle, *output_blocks]
    return [
        base,
        base,
        input_blocks[0],
        *_resample(double_curve, double_blocks),
        *_resample(single_curve, single_blocks),
        output_blocks[-1],
    ]


def _json_presets(path: Path, *, strict: bool = False) -> list[MergeBlockPreset]:
    try:
        if not path.exists():
            return []
    except OSError:
        if strict:
            raise
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        if strict:
            raise
        return []
    except json.JSONDecodeError:
        if strict:
            raise
        return []
    items = raw.get("items", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        if strict:
            raise ValueError("merge block preset storage root must be a list")
        return []
    presets: list[MergeBlockPreset] = []
    for item in items:
        if not isinstance(item, dict):
            if strict:
                raise ValueError("merge block preset storage contains an invalid item")
            continue
        try:
            presets.append(MergeBlockPreset.model_validate(item))
        except ValidationError as exc:
            if strict:
                raise ValueError("merge block preset storage contains an invalid item") from exc
            continue
    return presets


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _defaults_root() -> Path:
    return Path(
        os.environ.get(
            "ASTERIA_DEFAULTS_DIR",
            AppConfig().root_dir / "defaults",
        )
    )


def _presets_from_mbwpresets(source_label: str) -> list[MergeBlockPreset]:
    path = _defaults_root() / "merge-presets" / "mbwpresets.txt"
    if not _path_exists(path):
        return []

    presets: list[MergeBlockPreset] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = (row.get("preset_name") or "").strip()
            if not name:
                continue
            weights = _parse_weights(row.get("preset_weights") or "")
            if len(weights) == len(SD15_BLOCKS):
                preset_id = f"sd15_sd2:{name}"
                presets.append(
                    MergeBlockPreset(
                        id=preset_id,
                        name=name,
                        family="sd15_sd2",
                        weights=weights,
                        source=source_label,
                        compatible_schemas=["sd15_sd2"],
                    )
                )
                presets.append(
                    MergeBlockPreset(
                        id=f"sdxl:{name}",
                        name=name,
                        family="sdxl",
                        weights=_sdxl_from_sd15(weights),
                        source=source_label,
                        derived_from=preset_id,
                        description="Projected from the 26-value SD1/2 SuperMerger preset into the SDXL 36-block schema.",
                        compatible_schemas=["sdxl"],
                    )
                )
                for anima_schema in ANIMA_SCHEMA_IDS:
                    schema = BLOCK_SCHEMAS[anima_schema]
                    diffusion_blocks = sum(block.startswith("B") and block != "BASE" for block in schema)
                    label = {
                        "anima": "Anima",
                        "anima_2_9b_40": "Anima 2.9B 40-block",
                        "anima_3_8b_52": "Anima 3.8B 52-block",
                    }[anima_schema]
                    presets.append(
                        MergeBlockPreset(
                            id=f"{anima_schema}:{name}",
                            name=name,
                            family=anima_schema,
                            weights=_anima_from_sd15(weights, diffusion_blocks=diffusion_blocks),
                            source=source_label,
                            derived_from=preset_id,
                            description=f"Projected from the SD1/2 SuperMerger preset into the {label} diffusion and LLM schema.",
                            compatible_schemas=[anima_schema],
                        )
                    )
                for flux_family in FLUX_SCHEMA_IDS:
                    label = "Flux.1" if flux_family == "flux1" else "Flux.2"
                    schema = BLOCK_SCHEMAS[flux_family]
                    presets.append(
                        MergeBlockPreset(
                            id=f"{flux_family}:{name}",
                            name=name,
                            family=flux_family,
                            weights=_flux_from_sd15(
                                weights,
                                double_blocks=sum(block.startswith("D") for block in schema),
                                single_blocks=sum(block.startswith("S") for block in schema),
                            ),
                            source=source_label,
                            derived_from=preset_id,
                            description=f"Projected from the SD1/2 SuperMerger preset into the {label} CLIP, T5, double, and single block schema.",
                            compatible_schemas=[flux_family],
                        )
                    )
    return presets


def builtin_block_presets() -> list[MergeBlockPreset]:
    return _presets_from_mbwpresets("defaults/merge-presets/mbwpresets.txt")


class MergePresetRepository:
    def __init__(self, data_dir: Path, preset_dir: Path | None = None):
        self.data_dir = data_dir
        self.preset_dir = preset_dir or self.data_dir / "config" / "merge-presets"
        self.user_path = self.preset_dir / "user.json"
        self.legacy_user_path = self.data_dir / "merge-presets" / "user.json"

    def ensure_default_presets(self) -> None:
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        generated = _presets_from_mbwpresets("data/config/merge-presets")
        by_family: dict[str, list[MergeBlockPreset]] = {family: [] for family in BLOCK_SCHEMAS}
        for preset in generated:
            by_family.setdefault(preset.family, []).append(preset)
        for family, presets in by_family.items():
            path = self.preset_dir / f"{family}.json"
            if not presets:
                continue
            if _path_exists(path):
                current = _json_presets(path)
                expected_length = len(BLOCK_SCHEMAS[family])
                generated_current = current and all(item.source == "data/config/merge-presets" for item in current)
                schema_matches = current and all(len(item.weights) == expected_length for item in current)
                projection_matches = current and all(
                    item.family == "sd15_sd2" or item.description.startswith("Projected from") for item in current
                )
                if not generated_current or (schema_matches and projection_matches):
                    continue
            presets.sort(key=lambda item: item.name.lower())
            write_json_atomic(path, {"items": [item.model_dump() for item in presets]})

    def list(self) -> list[MergeBlockPreset]:
        try:
            self.ensure_default_presets()
        except OSError:
            pass
        presets: list[MergeBlockPreset] = []
        if _path_exists(self.preset_dir):
            try:
                paths = sorted(self.preset_dir.glob("*.json"))
            except OSError:
                paths = []
            for path in paths:
                presets.extend(_json_presets(path))
        if not _path_exists(self.user_path):
            presets.extend(_json_presets(self.legacy_user_path))
        deduped: dict[str, MergeBlockPreset] = {}
        for preset in presets:
            if preset.family == "flux2" and len(preset.weights) == len(FLUX1_BLOCKS):
                preset = preset.model_copy(update={"weights": migrate_legacy_flux2_weights(preset.weights)})
            deduped[preset.id] = preset
        return list(deduped.values())

    def save_user_preset(self, preset: MergeBlockPreset) -> MergeBlockPreset:
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        presets = [
            item
            for item in [
                *_json_presets(self.legacy_user_path, strict=True),
                *_json_presets(self.user_path, strict=True),
            ]
            if item.id != preset.id
        ]
        presets.append(preset)
        presets.sort(key=lambda item: (item.family, item.name.lower(), item.id))
        write_json_atomic(self.user_path, {"items": [item.model_dump() for item in presets]})
        return preset

    def delete_user_preset(self, preset_id: str) -> bool:
        presets = [
            *_json_presets(self.legacy_user_path, strict=True),
            *_json_presets(self.user_path, strict=True),
        ]
        kept = [preset for preset in presets if preset.id != preset_id]
        if len(kept) == len(presets):
            return False
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.user_path, {"items": [item.model_dump() for item in kept]})
        return True


def user_preset_id(family: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return f"user:{family}:{slug or 'preset'}"
