from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import random
import re
import shutil
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import scipy.ndimage
import torch

from app.services.merge_calculation_modes import (
    CALCULATION_CONTRACT_VERSION,
    CALCULATION_IMPLEMENTATION_REVISION,
    CALCULATION_PARAMETER_SCHEMA_VERSION,
    V11_CALCULATION_MODES,
    CalculationModeError,
    merge_weight_sum_tensor,
    normalize_calculation_parameters,
)
from app.services.model_fingerprints import ModelFingerprintService
from app.services.merge_presets import BLOCK_SCHEMAS, is_flux_schema, migrate_legacy_flux2_weight_text

if TYPE_CHECKING:
    from app.services.merge_runtime_contract import MergeRuntimeRecipeContract

MERGE_MODES = {
    "weight_sum": {
        "label": "Weight sum",
        "formula": "A*(1-alpha)+B*alpha",
        "requires_c": False,
        "uses_beta": False,
    },
    "add_difference": {
        "label": "Add difference",
        "formula": "A+(B-C)*alpha",
        "requires_c": True,
        "uses_beta": False,
    },
    "triple_sum": {
        "label": "Triple sum",
        "formula": "A*(1-alpha-beta)+B*alpha+C*beta",
        "requires_c": True,
        "uses_beta": True,
    },
    "sum_twice": {
        "label": "sum Twice",
        "formula": "(A*(1-alpha)+B*alpha)*(1-beta)+C*beta",
        "requires_c": True,
        "uses_beta": True,
    },
}

CALCULATION_MODES = {
    "normal": "Normal linear merge",
    "cosineA": "Cosine distance weighted toward A",
    "cosineB": "Cosine distance weighted toward B",
    "trainDifference": "Difference extraction used by training-difference workflows",
    "smoothAdd": "Smoothed additive delta",
    "smoothAdd MT": "Smoothed additive delta, multi-thread compatible mode",
    "extract": "Extract strong deltas",
    "tensor": "Tensor slice replacement along the first axis",
    "tensor2": "Tensor slice replacement along the second axis for wide tensors",
    "self": "Scale A tensors except projection/norm tensors",
    "plus random": "Add random noise to A tensors except projection/norm tensors",
    "slerp": "Global spherical interpolation over the flattened tensor",
    "nuslerp": "Vector-wise spherical interpolation with stable linear fallback",
    "nearswap": "Element-wise interpolation weighted toward similar parameters",
}

CALCULATION_MODE_COMPATIBILITY = {
    "weight_sum": [
        "normal",
        "cosineA",
        "cosineB",
        "tensor",
        "tensor2",
        "self",
        "plus random",
        "slerp",
        "nuslerp",
        "nearswap",
    ],
    "add_difference": ["normal", "trainDifference", "smoothAdd", "smoothAdd MT", "extract", "tensor", "tensor2", "plus random"],
    "triple_sum": ["normal", "cosineA", "cosineB", "tensor", "tensor2"],
    "sum_twice": ["normal", "cosineA", "cosineB", "tensor", "tensor2"],
}

MERGE_TYPES = {"checkpoint", "lora", "lora_to_checkpoint", "extract_lora"}
DEBUG_TENSOR_LIMIT = 500
LORA_SVD_CLAMP_QUANTILE = 0.99

def _now() -> str:
    return datetime.now(UTC).isoformat()


class MergeSource(BaseModel):
    role: str
    path: str
    label: str = ""
    weight: float | None = Field(default=None, ge=-10, le=10)
    clip_weight: float | None = Field(default=None, ge=-10, le=10)
    block_weights: str = ""


class MergeRecipe(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = "Untitled merge"
    merge_type: str = "checkpoint"
    model_family: str = "sdxl"
    merge_mode: str = "weight_sum"
    calc_mode: str = "normal"
    block_schema: str = "sdxl"
    sources: list[MergeSource] = Field(default_factory=list)
    alpha: float = Field(default=0.5, ge=-10, le=10)
    beta: float = Field(default=0.0, ge=-10, le=10)
    block_weights: str = ""
    block_weights_alpha: str = ""
    block_weights_beta: str = ""
    block_weight_preset: str = ""
    include_blocks: list[str] = Field(default_factory=list)
    exclude_blocks: list[str] = Field(default_factory=list)
    pinpoint_blocks: list[str] = Field(default_factory=list)
    pinpoint_elements: list[str] = Field(default_factory=list)
    pinpoint_adjust: str = ""
    elemental_rules: str = ""
    adjust_values: dict[str, float] = Field(default_factory=dict)
    calculation_option: float = Field(default=0.3, ge=0, le=20)
    calculation_contract_version: str = CALCULATION_CONTRACT_VERSION
    calculation_parameter_schema_version: int = CALCULATION_PARAMETER_SCHEMA_VERSION
    calculation_parameters: dict[str, object] = Field(default_factory=dict)
    calculation_implementation_revision: str = CALCULATION_IMPLEMENTATION_REVISION
    source_fingerprints: list[dict[str, object]] = Field(default_factory=list)
    dice_mode: str = ""
    dice_seed: int | None = None
    dice_ranges: dict[str, str] = Field(default_factory=dict)
    lora_metadata_mode: str = "merge"
    lora_same_to_strength: bool = False
    prompt_ratio_source: str = ""
    use_cuda: bool = True
    save_model: bool = True
    overwrite: bool = False
    safetensors: bool = True
    fp16: bool = False
    save_metadata: bool = True
    copy_config: bool = False
    prune: bool = False
    reset_clip_ids: bool = False
    use_old_calc_method: bool = False
    debug: bool = False
    bake_in_vae: str = ""
    lora_rank: int | None = Field(default=None, ge=1, le=4096)
    lora_conv_rank: int | None = Field(default=None, ge=1, le=4096)
    lora_source_ratio: float = Field(default=1.0, ge=-10, le=10)
    checkpoint_lora_strength: float = Field(default=1.0, ge=-10, le=10)
    output_name: str = ""
    output_path: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @model_validator(mode="after")
    def migrate_legacy_flux2_vectors(self) -> MergeRecipe:
        if self.block_schema != "flux2":
            return self
        self.block_weights = migrate_legacy_flux2_weight_text(self.block_weights)
        self.block_weights_alpha = migrate_legacy_flux2_weight_text(self.block_weights_alpha)
        self.block_weights_beta = migrate_legacy_flux2_weight_text(self.block_weights_beta)
        self.sources = [
            source.model_copy(update={"block_weights": migrate_legacy_flux2_weight_text(source.block_weights)})
            for source in self.sources
        ]
        self.dice_ranges = {
            key: migrate_legacy_flux2_weight_text(value)
            for key, value in self.dice_ranges.items()
        }
        return self

    @model_validator(mode="after")
    def normalize_v11_calculation_contract(self) -> MergeRecipe:
        mode = self.normalized_calc_mode()
        if mode not in V11_CALCULATION_MODES:
            return self
        if self.merge_type != "checkpoint":
            raise ValueError("V1.1 calculation modes require checkpoint merge type")
        if self.model_family not in {"sdxl", "anima"}:
            raise ValueError("V1.1 calculation modes require SDXL or Anima model family")
        if self.normalized_merge_mode() != "weight_sum":
            raise ValueError("V1.1 calculation modes require weight_sum")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"{mode} alpha must be between 0 and 1")
        if self.calculation_contract_version != CALCULATION_CONTRACT_VERSION:
            raise ValueError("incompatible calculation contract version")
        if (
            self.calculation_parameter_schema_version
            != CALCULATION_PARAMETER_SCHEMA_VERSION
        ):
            raise ValueError("incompatible calculation parameter schema version")
        if (
            self.calculation_implementation_revision
            != CALCULATION_IMPLEMENTATION_REVISION
        ):
            raise ValueError("incompatible calculation implementation revision")
        try:
            self.calculation_parameters = normalize_calculation_parameters(
                mode,
                self.calculation_parameters,
            )
        except CalculationModeError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def normalized_merge_mode(self) -> str:
        if self.calc_mode == "weighted_sum":
            return "weight_sum"
        if self.calc_mode == "add_difference":
            return "add_difference"
        return self.merge_mode

    def normalized_calc_mode(self) -> str:
        if self.calc_mode in {"weighted_sum", "add_difference", "block_weight"}:
            return "normal"
        return self.calc_mode

    def normalized_runtime_contract(self) -> MergeRuntimeRecipeContract:
        from app.services.merge_runtime_contract import normalize_merge_runtime_recipe

        return normalize_merge_runtime_recipe(self)

    def resolved_calculation_parameters(self) -> dict[str, object]:
        mode = self.normalized_calc_mode()
        if mode not in V11_CALCULATION_MODES:
            return dict(self.calculation_parameters)
        try:
            return normalize_calculation_parameters(
                mode,
                self.calculation_parameters,
            )
        except CalculationModeError as exc:
            raise ValueError(str(exc)) from exc


class MergeRun(BaseModel):
    id: str = ""
    operation_id: str = ""
    recipe_id: str
    status: str = "completed"
    output_path: str = ""
    metadata_path: str = ""
    calculation_mode_id: str = ""
    normalized_recipe: dict[str, object] = Field(default_factory=dict)
    source_fingerprints: list[dict[str, object]] = Field(default_factory=list)
    output_content_hash: dict[str, object] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    created_at: str = ""


class MergeAnalysisSource(BaseModel):
    role: str
    path: str
    tensor_count: int
    block_counts: dict[str, int]


class MergeShapeMismatch(BaseModel):
    key: str
    a_shape: list[int]
    b_shape: list[int]


class MergeTensorDifference(BaseModel):
    key: str
    block: str
    mean_abs: float
    max_abs: float


class MergeTensorSimilarity(BaseModel):
    key: str
    block: str
    similarity_percent: float


class MergeBlockSimilarity(BaseModel):
    block: str
    mean_percent: float
    min_percent: float
    attn2_percent: float | None = None


class MergeAttentionSimilarity(BaseModel):
    block: str
    key: str
    similarity_percent: float


class MergeElementalMatch(BaseModel):
    rule: str
    matched_keys: list[str]


class MergeAnalysis(BaseModel):
    sources: list[MergeAnalysisSource]
    common_keys: int
    mergeable_keys: int
    skipped_keys: int
    missing_in_b: list[str]
    shape_mismatches: list[MergeShapeMismatch]
    tensor_differences: list[MergeTensorDifference]
    tensor_similarities: list[MergeTensorSimilarity] = Field(default_factory=list)
    block_similarities: list[MergeBlockSimilarity] = Field(default_factory=list)
    attention_similarities: list[MergeAttentionSimilarity] = Field(default_factory=list)
    block_counts: dict[str, int]
    elemental_matches: list[MergeElementalMatch]


class MergeRecipeRepository:
    def __init__(self, path: Path):
        self.path = path

    def list(self) -> list[MergeRecipe]:
        recipes: list[MergeRecipe] = []
        for item in self._read():
            try:
                recipes.append(MergeRecipe(**item))
            except ValidationError:
                continue
        return recipes

    def get(self, recipe_id: str) -> MergeRecipe | None:
        for recipe in self.list():
            if recipe.id == recipe_id:
                return recipe
        return None

    def save(self, recipe: MergeRecipe) -> MergeRecipe:
        items = self._read(strict=True)
        now = _now()
        payload = recipe.model_dump()
        if not payload.get("id"):
            payload["id"] = uuid4().hex
            payload["created_at"] = now
        if not payload.get("created_at"):
            payload["created_at"] = now
        payload["updated_at"] = now
        replaced = False
        for index, item in enumerate(items):
            if item.get("id") == payload["id"]:
                items[index] = payload
                replaced = True
                break
        if not replaced:
            items.insert(0, payload)
        self._write(items)
        return MergeRecipe(**payload)

    def delete(self, recipe_id: str) -> bool:
        items = self._read(strict=True)
        next_items = [item for item in items if item.get("id") != recipe_id]
        self._write(next_items)
        return len(next_items) != len(items)

    def migrate_model_references(self, candidates: list[object]) -> int:
        from app.services.model_reference_migration import migrate_merge_reference_metadata

        items = self._read(strict=True)
        changed = 0
        migrated_items: list[dict] = []
        for item in items:
            migrated = migrate_merge_reference_metadata(item, candidates)
            if migrated != item:
                changed += 1
            migrated_items.append(migrated)
        if changed:
            self._write(migrated_items)
        return changed

    def _read(self, *, strict: bool = False) -> list[dict]:
        try:
            if not self.path.exists():
                return []
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError:
            if strict:
                raise
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise OSError(f"merge recipe file is not valid JSON: {exc}") from exc
            return []
        if not isinstance(raw, list):
            if strict:
                raise OSError("merge recipe file root must be a list")
            return []
        if strict:
            for item in raw:
                if not isinstance(item, dict):
                    raise OSError("merge recipe file contains an invalid item")
                try:
                    MergeRecipe(**item)
                except ValidationError as exc:
                    raise OSError("merge recipe file contains an invalid item") from exc
        return [item for item in raw if isinstance(item, dict)]

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


class MergeRunRepository:
    def __init__(self, path: Path, limit: int = 200, legacy_path: Path | None = None):
        self.path = path
        self.limit = limit
        self.legacy_path = legacy_path

    def list(self) -> list[MergeRun]:
        runs: list[MergeRun] = []
        for item in self._read():
            try:
                runs.append(MergeRun(**item))
            except ValidationError:
                continue
        return runs

    def save(self, run: MergeRun) -> MergeRun:
        items = [run.model_dump(), *[item for item in self._read(strict=True) if item.get("id") != run.id]]
        self._write(items[: self.limit])
        return run

    def _read(self, *, strict: bool = False) -> list[dict]:
        items = self._read_path(self.path, strict=strict)
        if items or self.legacy_path is None:
            return items
        return self._read_path(self.legacy_path, strict=strict)

    def _read_path(self, path: Path, *, strict: bool = False) -> list[dict]:
        try:
            if not path.exists():
                return []
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            if strict:
                raise
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise OSError(f"merge run history file is not valid JSON: {exc}") from exc
            return []
        if not isinstance(raw, list):
            if strict:
                raise OSError("merge run history file root must be a list")
            return []
        if strict:
            for item in raw:
                if not isinstance(item, dict):
                    raise OSError("merge run history file contains an invalid item")
                try:
                    MergeRun(**item)
                except ValidationError as exc:
                    raise OSError("merge run history file contains an invalid item") from exc
        return [item for item in raw if isinstance(item, dict)]

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


class MergeExecutionError(RuntimeError):
    pass


def _source(recipe: MergeRecipe, role: str) -> Path:
    match = next((source for source in recipe.sources if source.role.upper() == role), None)
    if match is None or not match.path:
        raise MergeExecutionError(f"source {role} is required")
    return _validated_source_path(match.path, role)


def _validated_source_path(value: str, role: str) -> Path:
    path = Path(value)
    if not _path_exists(path) or not _path_is_file(path):
        raise MergeExecutionError(f"source {role} does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise MergeExecutionError(f"source {role} must be a safetensors file: {path}")
    return path


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _lora_to_checkpoint_sources(recipe: MergeRecipe) -> list[tuple[str, MergeSource, Path]]:
    sources: list[tuple[str, MergeSource, Path]] = []
    seen: set[Path] = set()
    for source in recipe.sources:
        role = source.role.upper()
        if role == "A" or not source.path:
            continue
        path = _validated_source_path(source.path, role)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        sources.append((role, source, path))
    if not sources:
        raise MergeExecutionError("source B is required")
    return sources


def _merge_device(recipe: MergeRecipe, logs: list[str]) -> str:
    if recipe.use_cuda:
        if torch.cuda.is_available():
            logs.append("merge device: cuda")
            return "cuda"
        logs.append("merge device: cpu (cuda requested but unavailable)")
        return "cpu"
    logs.append("merge device: cpu")
    return "cpu"


def _load_safetensors(path: Path, role: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    try:
        return load_file(str(path), device=device)
    except Exception as exc:
        raise MergeExecutionError(f"source {role} could not be loaded as safetensors: {path}") from exc


def _canonical_checkpoint_diffusion_key(key: str) -> str:
    prefixes = ("model.diffusion_model.", "diffusion_model.")
    canonical = key
    for prefix in prefixes:
        if canonical.startswith(prefix):
            canonical = canonical[len(prefix) :]
            break
    return canonical


def _checkpoint_key_aliases(key: str) -> tuple[str, ...]:
    canonical = _canonical_checkpoint_diffusion_key(key)
    aliases = (key, canonical, f"model.diffusion_model.{canonical}", f"diffusion_model.{canonical}")
    return tuple(dict.fromkeys(aliases))


def _checkpoint_key_for_alias(tensors: dict[str, torch.Tensor], key: str) -> str | None:
    for candidate in _checkpoint_key_aliases(key):
        if candidate in tensors:
            return candidate
    return None


def _checkpoint_tensor_for_key(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    candidate = _checkpoint_key_for_alias(tensors, key)
    return tensors.get(candidate) if candidate is not None else None


def _flux_family_label(schema: str) -> str:
    return {"flux1": "Flux.1", "flux2": "Flux.2"}.get(schema, schema)


def _checkpoint_common_keys(tensors_a: dict[str, torch.Tensor], tensors_b: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in tensors_a if _checkpoint_tensor_for_key(tensors_b, key) is not None)


def _checkpoint_missing_keys(tensors_a: dict[str, torch.Tensor], tensors_b: dict[str, torch.Tensor]) -> list[str]:
    return sorted(key for key in tensors_a if _checkpoint_tensor_for_key(tensors_b, key) is None)


def _safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception:
        return {}
    return dict(metadata or {})


def _source_reference_snapshots(recipe: MergeRecipe) -> dict[str, dict[str, object]]:
    extra = recipe.model_extra or {}
    raw = extra.get("model_reference_snapshots")
    if not isinstance(raw, list):
        return {}
    snapshots: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").upper()
        if role:
            snapshots[role] = item
    return snapshots


def _source_content_fingerprints(
    recipe: MergeRecipe,
    sources: dict[str, Path],
    fingerprint_service: ModelFingerprintService,
    *,
    refresh_hash: bool = False,
) -> list[dict[str, object]]:
    references = _source_reference_snapshots(recipe)
    fingerprints: list[dict[str, object]] = []
    for role, path in sorted(sources.items()):
        try:
            stat = path.stat()
        except OSError as exc:
            raise MergeExecutionError(
                f"merge source provenance inspection failed: {path}"
            ) from exc
        reference = references.get(role, {})
        fingerprints.append(
            {
                "role": role,
                "algorithm": "sha256",
                "version": 1,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "stable_model_id": str(reference.get("stable_model_id") or ""),
                "resolved_root_id": str(reference.get("root_id") or ""),
                "normalized_relative_path": str(
                    reference.get("normalized_relative_path") or ""
                ),
                "value": fingerprint_service.strong_hash(
                    path,
                    refresh=refresh_hash,
                ),
            }
        )
    return fingerprints


def resolve_merge_source_fingerprints(
    recipe: MergeRecipe,
    fingerprint_service: ModelFingerprintService,
) -> list[dict[str, object]]:
    source_a = _source(recipe, "A")
    source_b = _source(recipe, "B")
    source_c = (
        _source(recipe, "C")
        if MERGE_MODES.get(recipe.normalized_merge_mode(), {}).get("requires_c")
        else None
    )
    sources = {
        "A": source_a,
        "B": source_b,
        **({"C": source_c} if source_c is not None else {}),
    }
    if recipe.merge_type == "lora_to_checkpoint":
        sources.update(
            {
                role: path
                for role, _source_item, path in _lora_to_checkpoint_sources(recipe)
                if role not in sources
            }
        )
    return _source_content_fingerprints(
        recipe,
        sources,
        fingerprint_service,
    )


def _output_content_hash(
    path: Path,
    fingerprint_service: ModelFingerprintService,
) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MergeExecutionError(
            f"merge output provenance inspection failed: {path}"
        ) from exc
    return {
        "algorithm": "sha256",
        "version": 1,
        "file_size": size,
        "value": fingerprint_service.strong_hash(path),
    }


def _normalized_recipe_provenance(
    recipe: MergeRecipe,
    *,
    output_content_hash: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = recipe.model_dump(mode="json")
    payload.update(
        {
            "calculation_mode_id": recipe.normalized_calc_mode(),
            "calculation_parameters": recipe.resolved_calculation_parameters(),
            "dtype_policy": "float32_arithmetic_then_requested_output_dtype",
            "key_policy": {
                "include": list(recipe.include_blocks),
                "exclude": list(recipe.exclude_blocks),
                "missing_or_non_floating": "preserve_source_a",
            },
        }
    )
    if output_content_hash is not None:
        payload["output_content_hash"] = dict(output_content_hash)
    return payload


def _safe_model_name(value: str, fallback: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return name or fallback


def _output_path(recipe: MergeRecipe, output_root: Path, transient: bool = False) -> Path:
    folder = "lora" if recipe.merge_type in {"lora", "extract_lora"} else "checkpoints"
    output_dir = output_root / ("merge-candidates" if transient else "merged") / folder
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MergeExecutionError(f"merge output directory unavailable: {output_dir}") from exc
    name = _safe_model_name(recipe.output_name, f"{recipe.name}_{uuid4().hex[:8]}.safetensors")
    if not name.lower().endswith(".safetensors"):
        name = f"{name}.safetensors"
    candidate = output_dir / name
    if recipe.overwrite or not _path_exists(candidate):
        return candidate
    return output_dir / f"{candidate.stem}_{uuid4().hex[:8]}{candidate.suffix}"


def _float_list(value: str) -> list[float]:
    weights: list[float] = []
    for chunk in value.replace("\n", ",").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            weights.append(float(text))
        except ValueError:
            continue
    return weights


def _block_id_for_key(key: str, schema: str) -> str:
    lowered = key.lower()
    if "llm_adapter.blocks." in lowered:
        tail = lowered.split("llm_adapter.blocks.", 1)[1]
        digits = "".join(ch for ch in tail[:5] if ch.isdigit())
        return f"LLM{int(digits):03d}" if digits else "BASE"
    if ".blocks." in lowered and schema == "anima":
        tail = lowered.split(".blocks.", 1)[1]
        digits = "".join(ch for ch in tail[:5] if ch.isdigit())
        return f"B{int(digits):03d}" if digits else "BASE"
    if lowered.startswith("net.blocks.") and schema == "anima":
        tail = lowered.split("net.blocks.", 1)[1]
        digits = "".join(ch for ch in tail[:5] if ch.isdigit())
        return f"B{int(digits):03d}" if digits else "BASE"
    if "final_layer" in lowered and (schema == "anima" or is_flux_schema(schema)):
        return "OUT"
    if schema == "sd15_sd2" and "cond_stage_model" in lowered:
        return "BASE"
    if schema == "sdxl" and "conditioner.embedders." in lowered:
        return "BASE"
    if is_flux_schema(schema) and "t5" in lowered:
        return "T5"
    if is_flux_schema(schema) and any(
        marker in lowered
        for marker in (
            "guidance_in.",
            "img_in.",
            "time_in.",
            "txt_in.",
            "vector_in.",
        )
    ):
        return "IN"
    if "double_blocks." in lowered or "double_block" in lowered:
        marker = "double_blocks." if "double_blocks." in lowered else "double_block"
        tail = lowered.split(marker, 1)[1]
        digits = "".join(ch for ch in tail[:5] if ch.isdigit())
        return f"D{int(digits):03d}" if digits else "IN"
    if "single_blocks." in lowered or "single_block" in lowered:
        marker = "single_blocks." if "single_blocks." in lowered else "single_block"
        tail = lowered.split(marker, 1)[1]
        digits = "".join(ch for ch in tail[:5] if ch.isdigit())
        return f"S{int(digits):03d}" if digits else "OUT"
    if "text" in lowered or "clip" in lowered:
        return "CLIP"
    if "t5" in lowered:
        return "T5"
    if "vae" in lowered:
        return "VAE"
    if schema in {"sd15_sd2", "sdxl"}:
        if "time_embed" in lowered or "label_emb" in lowered:
            return "IN00"
        if "model.diffusion_model.out." in lowered:
            return "OUT08" if schema == "sdxl" else "OUT11"
    if "input_blocks_" in lowered:
        tail = lowered.split("input_blocks_", 1)[1]
        match = re.match(r"(\d+)", tail)
        index = match.group(1) if match else ""
        return f"IN{int(index):02d}" if index else "BASE"
    if "output_blocks_" in lowered:
        tail = lowered.split("output_blocks_", 1)[1]
        match = re.match(r"(\d+)", tail)
        index = match.group(1) if match else ""
        return f"OUT{int(index):02d}" if index else "OUT"
    if "middle_block_" in lowered:
        return "M00" if schema == "sdxl" else "MID"
    if "input_blocks." in lowered:
        tail = lowered.split("input_blocks.", 1)[1]
        match = re.match(r"(\d+)", tail)
        index = match.group(1) if match else ""
        return f"IN{int(index):02d}" if index else "BASE"
    if "output_blocks." in lowered:
        tail = lowered.split("output_blocks.", 1)[1]
        match = re.match(r"(\d+)", tail)
        index = match.group(1) if match else ""
        return f"OUT{int(index):02d}" if index else "OUT"
    if "middle_block" in lowered or ".mid." in lowered:
        return "M00" if schema == "sdxl" else "MID"
    return "BASE"


ElementalRatio = float | str
ElementalRule = tuple[set[str], bool, list[str], bool, ElementalRatio]


def _expand_block_tokens(value: str, schema: list[str]) -> tuple[set[str], bool]:
    tokens = [part.strip().upper() for part in re.split(r"\s+", value.strip()) if part.strip()]
    invert = bool(tokens and tokens[0] == "NOT")
    if invert:
        tokens = tokens[1:]
    if not tokens or tokens == ["*"]:
        return set(), invert
    selected: set[str] = set()
    for token in tokens:
        if "-" in token:
            start, end = [part.strip().upper() for part in token.split("-", 1)]
            if start in schema and end in schema:
                low, high = sorted((schema.index(start), schema.index(end)))
                selected.update(schema[low : high + 1])
            continue
        selected.add(token)
    return selected, invert


def _extended_sdxl_block_id_for_key(key: str) -> str:
    lowered = key.lower()
    if "conditioner.embedders.0." in lowered:
        return "CLIPL"
    if "conditioner.embedders.1." in lowered:
        return "CLIPG"
    if "time_embed" in lowered:
        return "TIME"
    if "label_emb" in lowered:
        return "LABEL"
    if "first_stage_model" in lowered:
        return "VAE"
    if "model.diffusion_model.out." in lowered:
        return "OUT9"
    input_match = re.search(r"input_blocks[._](\d+)[._](\d+).*?transformer_blocks[._](\d+)", lowered)
    if input_match:
        return f"IN{int(input_match.group(1))}{int(input_match.group(2))}{int(input_match.group(3))}"
    output_match = re.search(r"output_blocks[._](\d+)[._](\d+).*?transformer_blocks[._](\d+)", lowered)
    if output_match:
        return f"OUT{int(output_match.group(1))}{int(output_match.group(2))}{int(output_match.group(3))}"
    middle_match = re.search(r"middle_block[._](\d+).*?transformer_blocks[._](\d+)", lowered)
    if middle_match:
        return f"MID{int(middle_match.group(1))}{int(middle_match.group(2))}"
    return ""


def _elemental_block_ids_for_key(key: str, block_id: str, schema_name: str) -> set[str]:
    ids = {block_id}
    if schema_name == "sdxl":
        extended = _extended_sdxl_block_id_for_key(key)
        if extended:
            ids.add(extended)
    return ids


def _element_tokens(value: str) -> tuple[list[str], bool]:
    tokens = [part.strip().lower() for part in re.split(r"\s+", value.strip()) if part.strip()]
    invert = bool(tokens and tokens[0].upper() == "NOT")
    if invert:
        tokens = tokens[1:]
    return tokens, invert


def _element_matches_key(element: str, lowered_key: str) -> bool:
    if not element:
        return True
    if element not in lowered_key:
        return False
    if re.fullmatch(r"[a-z]{1,4}\d*", element):
        return bool(re.search(rf"(^|[._]){re.escape(element)}(?:\d+)?([._]|$)", lowered_key))
    return True


def _parse_elemental_ratio(value: str) -> ElementalRatio | None:
    text = value.strip()
    if text.upper() in {"R", "U", "X"}:
        return text.upper()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_elemental_rules(value: str, schema: list[str]) -> list[ElementalRule]:
    rules: list[ElementalRule] = []
    for raw_line in value.replace(",", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(":")]
        if len(parts) < 3:
            continue
        block_text, element_text, ratio_text = parts[0], parts[1], parts[2]
        ratio = _parse_elemental_ratio(ratio_text)
        if ratio is None:
            continue
        blocks, invert_blocks = _expand_block_tokens(block_text, schema)
        elements, invert_elements = _element_tokens(element_text)
        rules.append((blocks, invert_blocks, elements, invert_elements, ratio))
    return rules


def _elemental_random_ratio(recipe: MergeRecipe, token: str, block_index: int, tensor_index: int) -> float:
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    lower = _dice_float_values(recipe.dice_ranges.get("lower", ""), len(schema), 0.0)
    upper = _dice_float_values(recipe.dice_ranges.get("upper", ""), len(schema), 1.0)
    randomer = _dice_randomer(recipe)
    random_value = float(randomer[tensor_index + 100])
    block_index = max(0, min(block_index, len(schema) - 1))
    if token == "R":
        value = random_value
    elif token == "U":
        value = -2 * random_value + 1.0
    else:
        value = (lower[block_index] - upper[block_index]) * random_value + upper[block_index]
    return round(value, _dice_precision(recipe))


def _elemental_weight_for_key(recipe: MergeRecipe, key: str, block_id: str, fallback: float, tensor_index: int = 0) -> float:
    result = fallback
    lowered = key.lower()
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    for blocks, invert_blocks, elements, invert_elements, ratio in _parse_elemental_rules(recipe.elemental_rules, schema):
        key_block_ids = _elemental_block_ids_for_key(key, block_id, recipe.block_schema)
        block_matched = True if not blocks else bool(blocks.intersection(key_block_ids))
        if invert_blocks:
            block_matched = not block_matched
        if not block_matched:
            continue
        element_matched = True if not elements else any(_element_matches_key(element, lowered) for element in elements)
        if invert_elements:
            element_matched = not element_matched
        if element_matched:
            if isinstance(ratio, str) and block_id in schema:
                result = _elemental_random_ratio(recipe, ratio, schema.index(block_id), tensor_index)
            else:
                result = float(ratio)
    return result


def _elemental_rule_key(block: str, element: str, ratio: float, negate: bool) -> str:
    element_text = f"NOT {element}" if negate else element
    return f"{block}:{element_text}:{ratio:g}"


def _elemental_rule_label(blocks: set[str], invert_blocks: bool, elements: list[str], invert_elements: bool, ratio: ElementalRatio, schema: list[str]) -> str:
    if not blocks:
        block_text = "*"
    else:
        ordered_blocks = [block for block in schema if block in blocks]
        block_text = " ".join(ordered_blocks)
    if invert_blocks:
        block_text = f"NOT {block_text}"
    element_text = " ".join(elements)
    if invert_elements:
        element_text = f"NOT {element_text}".strip()
    ratio_text = f"{ratio:g}" if isinstance(ratio, float) else ratio
    return f"{block_text}:{element_text}:{ratio_text}"


def _block_weight_for_key(recipe: MergeRecipe, key: str, value: str, fallback: float, tensor_index: int = 0) -> float:
    weights = _float_list(value)
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    block_id = _block_id_for_key(key, recipe.block_schema)
    if len(weights) >= len(schema):
        if block_id in schema:
            return _elemental_weight_for_key(recipe, key, block_id, weights[schema.index(block_id)], tensor_index)
    result = fallback
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        pattern, weight = line.split(":", 1)
        pattern_text = pattern.strip()
        key_block_ids = _elemental_block_ids_for_key(key, block_id, recipe.block_schema)
        if pattern_text and (pattern_text in key or pattern_text.upper() in key_block_ids):
            try:
                result = float(weight.strip())
            except ValueError:
                continue
    return _elemental_weight_for_key(recipe, key, block_id, result, tensor_index)


def _should_merge_key(recipe: MergeRecipe, key: str) -> bool:
    block_id = _block_id_for_key(key, recipe.block_schema)

    def matches_token(token: str) -> bool:
        normalized = token.strip()
        if not normalized:
            return False
        upper = normalized.upper()
        if upper == "PRINT":
            return False
        if upper == block_id.upper():
            return True
        if upper == "ADJUST":
            return key in SUPER_MERGER_FINETUNE_KEYS
        if upper == "VAE":
            return block_id.upper() == "VAE" or "first_stage_model" in key.lower()
        return normalized.lower() in key.lower()

    if recipe.include_blocks and not any(matches_token(token) for token in recipe.include_blocks):
        return False
    if recipe.exclude_blocks and any(matches_token(token) for token in recipe.exclude_blocks):
        return False
    return True


def _is_checkpoint_arithmetic_tensor(key: str, tensor: torch.Tensor) -> bool:
    lowered = key.lower()
    return tensor.is_floating_point() and ("weight" in lowered or "bias" in lowered)


def _source_block_counts(tensors: dict[str, torch.Tensor], schema: str) -> dict[str, int]:
    counts = Counter(_block_id_for_key(key, schema) for key in tensors)
    return dict(sorted(counts.items()))


def _cosine_similarity_percent(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.float().flatten()
    flat_b = b.float().flatten()
    if flat_a.numel() == 0 or flat_b.numel() == 0:
        return 0.0
    if float(torch.linalg.vector_norm(flat_a).item()) == 0.0 or float(torch.linalg.vector_norm(flat_b).item()) == 0.0:
        return 0.0
    value = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).clamp(-1, 1)
    return round(float(value.item() * 100.0), 3)


ATTENTION_SIMILARITY_BLOCKS: tuple[tuple[str, str], ...] = (
    ("input_blocks.1", "input"),
    ("input_blocks.2", "input"),
    ("input_blocks.4", "input"),
    ("input_blocks.5", "input"),
    ("input_blocks.7", "input"),
    ("input_blocks.8", "input"),
    ("middle_block", "middle"),
    ("output_blocks.3", "output"),
    ("output_blocks.4", "output"),
    ("output_blocks.5", "output"),
    ("output_blocks.6", "output"),
    ("output_blocks.7", "output"),
    ("output_blocks.8", "output"),
    ("output_blocks.9", "output"),
    ("output_blocks.10", "output"),
    ("output_blocks.11", "output"),
)


def _attention_prefix(block: str) -> str:
    return f"model.diffusion_model.{block}.1.transformer_blocks.0.attn1"


def _cross_attention_probe(to_q: torch.Tensor, to_k: torch.Tensor, to_v: torch.Tensor, rand_input: torch.Tensor) -> torch.Tensor:
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = torch.nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = torch.nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = torch.nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q.float().cpu()})
    attn_to_k.load_state_dict({"weight": to_k.float().cpu()})
    attn_to_v.load_state_dict({"weight": to_v.float().cpu()})
    q = attn_to_q(rand_input)
    k = attn_to_k(rand_input)
    v = attn_to_v(rand_input)
    attention = torch.nn.functional.softmax(torch.einsum("ij,kj->ik", q, k), dim=-1)
    return torch.einsum("ik,jk->ik", attention, v)


def _attention_similarities(tensors_a: dict[str, torch.Tensor], tensors_b: dict[str, torch.Tensor], schema: str) -> list[MergeAttentionSimilarity]:
    results: list[MergeAttentionSimilarity] = []
    for block, _label in ATTENTION_SIMILARITY_BLOCKS:
        prefix = _attention_prefix(block)
        q_key = f"{prefix}.to_q.weight"
        k_key = f"{prefix}.to_k.weight"
        v_key = f"{prefix}.to_v.weight"
        required = [q_key, k_key, v_key]
        if any(key not in tensors_a or key not in tensors_b for key in required):
            continue
        q_a, k_a, v_a = (tensors_a[q_key], tensors_a[k_key], tensors_a[v_key])
        q_b, k_b, v_b = (tensors_b[q_key], tensors_b[k_key], tensors_b[v_key])
        if any(tensor.ndim != 2 for tensor in (q_a, k_a, v_a, q_b, k_b, v_b)):
            continue
        if q_a.shape != q_b.shape or k_a.shape != k_b.shape or v_a.shape != v_b.shape:
            continue
        if q_a.shape != k_a.shape or q_a.shape != v_a.shape:
            continue
        hidden_dim, embed_dim = q_a.shape
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2244096)
        rand_input = torch.randn((embed_dim, hidden_dim), generator=generator, dtype=torch.float32)
        probe_a = _cross_attention_probe(q_a, k_a, v_a, rand_input)
        probe_b = _cross_attention_probe(q_b, k_b, v_b, rand_input)
        results.append(
            MergeAttentionSimilarity(
                block=_block_id_for_key(q_key, schema),
                key=q_key,
                similarity_percent=_cosine_similarity_percent(probe_a, probe_b),
            )
        )
    return results


def analyze_merge_recipe(recipe: MergeRecipe) -> MergeAnalysis:
    source_a = _source(recipe, "A")
    source_b = _source(recipe, "B")
    tensors_a = _load_safetensors(source_a, "A")
    tensors_b = _load_safetensors(source_b, "B")
    sources = [
        MergeAnalysisSource(role="A", path=str(source_a), tensor_count=len(tensors_a), block_counts=_source_block_counts(tensors_a, recipe.block_schema)),
        MergeAnalysisSource(role="B", path=str(source_b), tensor_count=len(tensors_b), block_counts=_source_block_counts(tensors_b, recipe.block_schema)),
    ]
    if MERGE_MODES.get(recipe.normalized_merge_mode(), {}).get("requires_c"):
        source_c = _source(recipe, "C")
        tensors_c = _load_safetensors(source_c, "C")
        sources.append(MergeAnalysisSource(role="C", path=str(source_c), tensor_count=len(tensors_c), block_counts=_source_block_counts(tensors_c, recipe.block_schema)))

    common_keys = _checkpoint_common_keys(tensors_a, tensors_b) if recipe.merge_type == "checkpoint" else sorted(set(tensors_a).intersection(tensors_b))
    missing_in_b = _checkpoint_missing_keys(tensors_a, tensors_b) if recipe.merge_type == "checkpoint" else sorted(set(tensors_a).difference(tensors_b))
    shape_mismatches: list[MergeShapeMismatch] = []
    tensor_differences: list[MergeTensorDifference] = []
    tensor_similarities: list[MergeTensorSimilarity] = []
    mergeable_keys = 0
    skipped_keys = 0
    block_counts: Counter[str] = Counter()
    block_similarity_scores: dict[str, list[float]] = defaultdict(list)
    attn2_similarity_scores: dict[str, float] = {}
    for key in common_keys:
        block_id = _block_id_for_key(key, recipe.block_schema)
        block_counts[block_id] += 1
        if not _should_merge_key(recipe, key):
            skipped_keys += 1
            continue
        if recipe.merge_type == "checkpoint" and not _is_checkpoint_arithmetic_tensor(key, tensors_a[key]):
            skipped_keys += 1
            continue
        tensor_b = _checkpoint_tensor_for_key(tensors_b, key) if recipe.merge_type == "checkpoint" else tensors_b[key]
        if tensor_b is None:
            skipped_keys += 1
            continue
        if tensors_a[key].shape != tensor_b.shape:
            shape_mismatches.append(MergeShapeMismatch(key=key, a_shape=list(tensors_a[key].shape), b_shape=list(tensor_b.shape)))
            continue
        mergeable_keys += 1
        delta = (tensor_b.float() - tensors_a[key].float()).abs()
        tensor_differences.append(
            MergeTensorDifference(
                key=key,
                block=block_id,
                mean_abs=float(delta.mean().item()),
                max_abs=float(delta.max().item()),
            )
        )
        similarity_percent = _cosine_similarity_percent(tensors_a[key], tensor_b)
        tensor_similarities.append(MergeTensorSimilarity(key=key, block=block_id, similarity_percent=similarity_percent))
        block_similarity_scores[block_id].append(similarity_percent)
        if "attn2.to_out.0.weight" in key:
            attn2_similarity_scores[block_id] = similarity_percent

    block_similarities = [
        MergeBlockSimilarity(
            block=block,
            mean_percent=round(sum(scores) / len(scores), 3),
            min_percent=round(min(scores), 3),
            attn2_percent=attn2_similarity_scores.get(block),
        )
        for block, scores in sorted(block_similarity_scores.items())
        if scores
    ]

    elemental_matches: list[MergeElementalMatch] = []
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    for blocks, invert_blocks, elements, invert_elements, ratio in _parse_elemental_rules(recipe.elemental_rules, schema):
        matched_keys: list[str] = []
        for key in common_keys:
            block_id = _block_id_for_key(key, recipe.block_schema)
            key_block_ids = _elemental_block_ids_for_key(key, block_id, recipe.block_schema)
            block_matched = True if not blocks else bool(blocks.intersection(key_block_ids))
            if invert_blocks:
                block_matched = not block_matched
            if not block_matched:
                continue
            lowered = key.lower()
            element_matched = True if not elements else any(_element_matches_key(element, lowered) for element in elements)
            if invert_elements:
                element_matched = not element_matched
            if element_matched:
                matched_keys.append(key)
        elemental_matches.append(
            MergeElementalMatch(
                rule=_elemental_rule_label(blocks, invert_blocks, elements, invert_elements, ratio, schema),
                matched_keys=matched_keys[:50],
            )
        )

    return MergeAnalysis(
        sources=sources,
        common_keys=len(common_keys),
        mergeable_keys=mergeable_keys,
        skipped_keys=skipped_keys,
        missing_in_b=missing_in_b[:50],
        shape_mismatches=shape_mismatches[:50],
        tensor_differences=sorted(tensor_differences, key=lambda item: item.max_abs, reverse=True)[:50],
        tensor_similarities=sorted(tensor_similarities, key=lambda item: item.similarity_percent)[:50],
        block_similarities=block_similarities,
        attention_similarities=_attention_similarities(tensors_a, tensors_b, recipe.block_schema),
        block_counts=dict(sorted(block_counts.items())),
        elemental_matches=elemental_matches,
    )


def _midpoint_percentile(values: torch.Tensor, percentile: float) -> float:
    sorted_values = torch.sort(values.flatten()).values
    if sorted_values.numel() == 0:
        return 0.0
    if sorted_values.numel() == 1:
        return float(sorted_values[0].item())
    rank = (percentile / 100.0) * (sorted_values.numel() - 1)
    low = int(rank)
    high = min(low + 1, sorted_values.numel() - 1)
    if float(low) == rank:
        return float(sorted_values[low].item())
    return float(((sorted_values[low] + sorted_values[high]) / 2.0).item())


def _cosine_similarity_scores(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    cosine = torch.nn.CosineSimilarity(dim=0)
    a_float = a.float()
    b_float = b.float()
    if mode == "cosineA":
        a_norm = torch.nn.functional.normalize(a_float, p=2, dim=0)
        b_norm = torch.nn.functional.normalize(b_float, p=2, dim=0)
        return cosine(a_norm, b_norm)
    simab = cosine(a_float, b_float)
    magnitude = torch.dot(a_float.flatten(), b_float.flatten()) / (torch.norm(a_float) * torch.norm(b_float))
    return (simab + magnitude) / 2.0


def _cosine_context(tensors_a: dict[str, torch.Tensor], tensors_b: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor | float]:
    scores: list[torch.Tensor] = []
    for key, tensor_a in tensors_a.items():
        if "first_stage_model" in key or "model" not in key or key not in tensors_b:
            continue
        tensor_b = tensors_b[key]
        if tensor_a.shape != tensor_b.shape:
            continue
        scores.append(_cosine_similarity_scores(tensor_a, tensor_b, mode).flatten())
    if not scores:
        return {"min": 0.0, "max": 0.0}
    sims = torch.cat(scores)
    sims = sims[~torch.isnan(sims)]
    if sims.numel() == 0:
        return {"min": 0.0, "max": 0.0}
    low = _midpoint_percentile(sims, 1)
    high = _midpoint_percentile(sims, 99)
    sims = sims[(sims >= low) & (sims <= high)]
    if sims.numel() == 0:
        return {"min": 0.0, "max": 0.0}
    return {"min": float(sims.min().item()), "max": float(sims.max().item())}


def _supermerger_cosine_merge(
    key: str,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    mode: str,
    context: dict[str, torch.Tensor | float],
) -> torch.Tensor:
    if "first_stage_model" in key or "model" not in key:
        return a.float()
    min_similarity = float(context.get("min", 0.0))
    max_similarity = float(context.get("max", 0.0))
    if max_similarity == min_similarity:
        return a.float()
    combined = _cosine_similarity_scores(a, b, mode)
    k = (combined - min_similarity) / (max_similarity - min_similarity)
    k = k - (abs(alpha) if mode == "cosineA" else alpha)
    k = k.clip(min=0, max=1.0)
    return b.float() * (1.0 - k) + a.float() * k


def _smooth_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() < 3:
        return tensor
    flat = tensor.float().flatten()
    padded = torch.nn.functional.pad(flat[None, None, :], (1, 1), mode="replicate")
    smoothed = torch.nn.functional.avg_pool1d(padded, kernel_size=3, stride=1).reshape(tensor.shape)
    return smoothed.to(device=tensor.device)


def _apply_calc_mode(recipe: MergeRecipe, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    mode = recipe.normalized_calc_mode()
    a_float = a.float()
    b_float = b.float()
    c_float = c.float() if c is not None else None
    if mode == "cosineA":
        pass
    elif mode == "cosineB":
        pass
    elif mode in {"smoothAdd", "smoothAdd MT"}:
        pass
    elif mode == "self":
        b_float = a_float
    elif mode == "plus random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(abs(hash((recipe.id, recipe.name))) % (2**31))
        alpha += float(torch.rand(1, generator=generator).item() - 0.5) * 0.1
    return a_float, b_float, c_float, alpha


SELF_SKIP_KEY_PARTS = ("to_out", "proj_out", "norm")


def _supermerger_train_difference(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float) -> torch.Tensor:
    a_float = a.float()
    b_float = b.float()
    c_float = c.float()
    if torch.allclose(b_float, c_float.to(device=b_float.device), rtol=0, atol=0):
        return a_float
    diff = b_float - c_float
    distance_base = diff.abs()
    distance_a = (b_float - a_float).abs()
    sum_distances = distance_base + distance_a
    zero = torch.tensor(0.0, device=a_float.device)
    scale = torch.where(sum_distances != 0, distance_a / sum_distances, zero)
    scale = torch.sign(diff) * scale.abs()
    return a_float + scale * diff.abs() * (alpha * 1.8)


def _supermerger_extract(base: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float, beta: float, gamma: float) -> torch.Tensor:
    base_float = base.float()
    b_delta = b.float() - base_float
    c_delta = c.float() - base_float
    similarity = torch.nn.functional.cosine_similarity(b_delta, c_delta, dim=-1).clamp(-1, 1).unsqueeze(-1)
    selectivity = ((similarity + 1.0) / 2.0) ** gamma
    return base_float + torch.lerp(b_delta, c_delta, alpha) * torch.lerp(selectivity, 1.0 - selectivity, beta)


def _supermerger_smooth_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = (b.float() - c.float()).cpu().numpy()
    filtered = scipy.ndimage.median_filter(diff, size=3)
    filtered = scipy.ndimage.gaussian_filter(filtered, sigma=1)
    return a.float() + alpha * torch.tensor(filtered, device=a.device, dtype=torch.float32)


def _key_uses_self_skip(key: str) -> bool:
    return any(part in key for part in SELF_SKIP_KEY_PARTS)


def _plus_random_noise(recipe: MergeRecipe, key: str, base: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha == 0.0 or _key_uses_self_skip(key):
        return base.float()
    seed = recipe.dice_seed
    if seed is None:
        seed = int(hashlib.sha256(f"{recipe.id}|{recipe.name}|{key}".encode("utf-8")).hexdigest()[:16], 16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    noise = torch.randn(base.shape, generator=generator, dtype=torch.float32, device="cpu").to(base.device)
    return base.float() + noise * alpha


def _replace_tensor_slice(base: torch.Tensor, source: torch.Tensor, alpha: float, beta: float, *, second_axis_for_wide_tensors: bool) -> torch.Tensor:
    if base.ndim == 0:
        return base.clone()
    axis = 0
    if second_axis_for_wide_tensors and base.ndim > 1 and base.shape[1] > 100:
        axis = 1
    axis_length = base.shape[axis]
    start_ratio = beta if alpha + beta <= 1 else alpha + beta - 1
    end_ratio = alpha + beta if alpha + beta <= 1 else beta
    start = max(0, min(axis_length, int(axis_length * start_ratio)))
    end = max(0, min(axis_length, int(axis_length * end_ratio)))
    if start >= end:
        return base.clone()
    result = base.clone() if alpha + beta <= 1 else source.clone()
    replacement = source if alpha + beta <= 1 else base
    selector = [slice(None)] * base.ndim
    selector[axis] = slice(start, end)
    result[tuple(selector)] = replacement[tuple(selector)].clone()
    return result


def _dice_seed(recipe: MergeRecipe) -> int:
    if recipe.dice_seed is not None:
        return recipe.dice_seed
    material = "|".join([recipe.id, recipe.name, recipe.dice_mode, recipe.merge_mode, recipe.output_name])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _dice_randomer(recipe: MergeRecipe) -> np.ndarray:
    return np.random.RandomState(_dice_seed(recipe)).rand(3000)


def _dice_precision(recipe: MergeRecipe) -> int:
    try:
        return max(0, int(float(recipe.dice_ranges.get("round", 3))))
    except (TypeError, ValueError):
        return 3


def _dice_float_values(value: str, count: int, default: float) -> list[float]:
    items = [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]
    if not items:
        return [default] * count
    values: list[float] = []
    for item in items:
        try:
            values.append(float(item))
        except ValueError:
            values.append(default)
    if len(values) == 1:
        values = values * count
    if len(values) < count:
        values.extend([default] * (count - len(values)))
    return values[:count]


def _dice_limiter(value: float, index: int, lower_delta: list[float], upper_delta: list[float]) -> float:
    return max(lower_delta[index], min(upper_delta[index], value))


def _dice_token_value(
    token: str,
    randomer: np.ndarray,
    index: int,
    offset: int,
    lower: list[float],
    upper: list[float],
    lower_delta: list[float],
    upper_delta: list[float],
    precision: int,
) -> float:
    random_value = float(randomer[index + offset])
    if token == "R":
        value = random_value
    elif token == "U":
        value = -2 * random_value + 1.5
    elif token == "D":
        value = upper[index] + lower[index] * (random_value - 0.5)
    else:
        value = (lower[index] - upper[index]) * random_value + upper[index]
    return round(_dice_limiter(value, index, lower_delta, upper_delta), precision)


def _expand_dice_block_weights(recipe: MergeRecipe, value: str, *, offset: int) -> tuple[str, str]:
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    tokens = [item.strip() for item in value.replace("\n", ",").split(",")]
    if not tokens or tokens == [""]:
        return value, ""
    if len(tokens) < len(schema):
        tokens.extend(["0"] * (len(schema) - len(tokens)))
    tokens = tokens[: len(schema)]
    randomer = _dice_randomer(recipe)
    precision = _dice_precision(recipe)
    lower = _dice_float_values(recipe.dice_ranges.get("lower", ""), len(schema), 0.0)
    upper = _dice_float_values(recipe.dice_ranges.get("upper", ""), len(schema), 1.0)
    lower_delta = _dice_float_values(recipe.dice_ranges.get("lower_delta", ""), len(schema), 0.0)
    upper_delta = _dice_float_values(recipe.dice_ranges.get("upper_delta", ""), len(schema), 1.0)
    expanded: list[str] = []
    elemental_rules: list[str] = []
    for index, raw_token in enumerate(tokens):
        token = raw_token.upper()
        if token in {"R", "U", "X", "D"}:
            expanded.append(str(_dice_token_value(token, randomer, index, offset, lower, upper, lower_delta, upper_delta, precision)))
        elif token in {"ER", "EU", "EX"}:
            elemental_token = token.replace("E", "", 1)
            elemental_rules.append(f"{schema[index]}::{elemental_token}")
            expanded.append("0")
        else:
            expanded.append(raw_token)
    return ",".join(expanded), ",".join(elemental_rules)


def _has_dice_tokens(value: str) -> bool:
    return any(item.strip().upper() in {"R", "U", "X", "D", "ER", "EU", "EX"} for item in value.replace("\n", ",").split(","))


def _dice_range(value: str, default: tuple[float, float]) -> tuple[float, float]:
    cleaned = value.strip()
    if not cleaned:
        return default
    for separator in (";", "|", "/", ":"):
        cleaned = cleaned.replace(separator, ",")
    parts = [part for chunk in cleaned.split(",") for part in chunk.split()]
    values = [float(part) for part in parts[:2]]
    if not values:
        return default
    if len(values) == 1:
        return (default[0], values[0])
    low, high = values
    return (low, high) if low <= high else (high, low)


def _dice_value(generator: random.Random, mode: str, bounds: tuple[float, float]) -> float:
    low, high = bounds
    normalized = mode.upper()
    if normalized in {"X", "EX"}:
        return generator.choice([low, high])
    if normalized == "ER":
        return low + (high - low) * (generator.random() ** 2)
    return generator.uniform(low, high)


def _apply_dice(recipe: MergeRecipe) -> MergeRecipe:
    mode = recipe.dice_mode.strip()
    if not mode:
        return recipe
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    block_updates: dict[str, str] = {}
    elemental_additions: list[str] = []
    normalized = mode.upper()
    if mode.lower() == "custom":
        if _has_dice_tokens(recipe.block_weights_alpha):
            block_updates["block_weights_alpha"], alpha_elemental = _expand_dice_block_weights(recipe, recipe.block_weights_alpha, offset=0)
            if alpha_elemental:
                elemental_additions.append(alpha_elemental)
        if _has_dice_tokens(recipe.block_weights_beta):
            block_updates["block_weights_beta"], beta_elemental = _expand_dice_block_weights(recipe, recipe.block_weights_beta, offset=50)
            if beta_elemental:
                elemental_additions.append(beta_elemental)
    elif normalized in {"R", "U", "X", "D", "ER", "EU", "EX"}:
        tokens = ",".join([normalized] * len(schema))
        block_updates["block_weights_alpha"], alpha_elemental = _expand_dice_block_weights(recipe, tokens, offset=0)
        if alpha_elemental:
            elemental_additions.append(alpha_elemental)
    if elemental_additions:
        existing = recipe.elemental_rules.strip()
        addition = ",".join(elemental_additions)
        block_updates["elemental_rules"] = f"{existing},{addition}" if existing else addition
    if block_updates:
        recipe = recipe.model_copy(update=block_updates)
    generator = random.Random(_dice_seed(recipe))
    updates: dict[str, float] = {}
    custom = mode.lower() == "custom"
    uses_beta = MERGE_MODES.get(recipe.normalized_merge_mode(), {}).get("uses_beta", False)
    if "alpha" in recipe.dice_ranges or not custom:
        updates["alpha"] = _dice_value(generator, mode, _dice_range(recipe.dice_ranges.get("alpha", ""), (0.0, 1.0)))
    if "beta" in recipe.dice_ranges or (uses_beta and not custom):
        updates["beta"] = _dice_value(generator, mode, _dice_range(recipe.dice_ranges.get("beta", ""), (0.0, 1.0)))
    if not updates:
        return recipe
    return recipe.model_copy(update=updates)


def expand_dice_recipe(recipe: MergeRecipe) -> MergeRecipe:
    return _apply_dice(recipe.model_copy(deep=True))


def _prompt_lora_ratios(value: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for name, ratio in re.findall(r"<lora:([^:>]+):([-+]?(?:\d+(?:\.\d*)?|\.\d+))>", value, flags=re.IGNORECASE):
        ratios[_sanitize_lora_key(name)] = float(ratio)
    return ratios


def _source_name_candidates(source: MergeSource) -> set[str]:
    path = Path(source.path)
    values = {source.label, path.stem, path.name}
    return {_sanitize_lora_key(value) for value in values if value}


def _apply_prompt_ratio_source(recipe: MergeRecipe) -> MergeRecipe:
    if recipe.merge_type != "lora_to_checkpoint" or not recipe.prompt_ratio_source.strip():
        return recipe
    ratios = _prompt_lora_ratios(recipe.prompt_ratio_source)
    if not ratios:
        return recipe
    lora_sources = [source for source in recipe.sources if source.role.upper() != "A" and source.path]
    if len(lora_sources) != 1:
        return recipe
    candidates = _source_name_candidates(lora_sources[0])
    for name, ratio in ratios.items():
        if name in candidates:
            return recipe.model_copy(update={"checkpoint_lora_strength": ratio})
    return recipe


def _checkpoint_lora_strength_for_source(recipe: MergeRecipe, source: MergeSource) -> float:
    if source.weight is not None:
        return source.weight
    ratios = _prompt_lora_ratios(recipe.prompt_ratio_source)
    if not ratios:
        return recipe.checkpoint_lora_strength
    candidates = _source_name_candidates(source)
    for name, ratio in ratios.items():
        if name in candidates:
            return ratio
    return recipe.checkpoint_lora_strength


def _is_clip_checkpoint_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.startswith("conditioner.embedders.") or lowered.startswith("cond_stage_model.")


def _checkpoint_lora_strength_for_target(recipe: MergeRecipe, source: MergeSource, target_key: str, tensor_index: int = 0) -> float:
    model_strength = _checkpoint_lora_strength_for_source(recipe, source)
    strength = source.clip_weight if source.clip_weight is not None and _is_clip_checkpoint_key(target_key) else model_strength
    if source.block_weights.strip():
        strength = _block_weight_for_key(recipe, target_key, source.block_weights, strength, tensor_index)
    return strength


def _adjust_value(recipe: MergeRecipe, key: str) -> float:
    try:
        return float(recipe.adjust_values.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


SUPER_MERGER_FINETUNE_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]

ADJUST_COLORS_SD15 = [[-1, 1 / 3, 2 / 3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
ADJUST_COLORS_SDXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]


def _supermerger_color_adjustments(values: list[float], is_sdxl: bool) -> list[float]:
    colors = ADJUST_COLORS_SDXL if is_sdxl else ADJUST_COLORS_SD15
    outputs = [[component * values[index] * 0.02 for component in color] for index, color in enumerate(colors)]
    return [sum(items) for items in zip(*outputs, strict=True)]


def _supermerger_adjust_factors(recipe: MergeRecipe) -> list[float | list[float]]:
    detail1 = _adjust_value(recipe, "detail_noise_1")
    detail2 = _adjust_value(recipe, "detail_noise_2")
    detail3 = _adjust_value(recipe, "detail_noise_3")
    contrast = _adjust_value(recipe, "contrast_detail")
    brightness = _adjust_value(recipe, "brightness")
    color1 = _adjust_value(recipe, "color_tone_1")
    color2 = _adjust_value(recipe, "color_tone_2")
    color3 = _adjust_value(recipe, "color_tone_3")
    return [
        1 - detail1 * 0.01,
        1 + detail1 * 0.02,
        1 - detail2 * 0.01,
        1 + detail2 * 0.02,
        1 - detail3 * 0.01,
        [contrast * 0.02] + _supermerger_color_adjustments([brightness, color1, color2, color3], recipe.block_schema == "sdxl"),
    ]


def _apply_adjust_values(recipe: MergeRecipe, key: str, value: torch.Tensor) -> torch.Tensor:
    if not recipe.adjust_values:
        return value
    if key not in SUPER_MERGER_FINETUNE_KEYS:
        return value
    index = SUPER_MERGER_FINETUNE_KEYS.index(key)
    factors = _supermerger_adjust_factors(recipe)
    if index < 5:
        return value * float(factors[index])
    addition = torch.tensor(factors[5], device=value.device, dtype=torch.float32)
    return value + addition


def _adjust_target_tensor_count(recipe: MergeRecipe, tensors: dict[str, torch.Tensor]) -> int:
    if not recipe.adjust_values:
        return 0
    return sum(1 for key in SUPER_MERGER_FINETUNE_KEYS if key in tensors and _should_merge_key(recipe, key))


def _merge_tensor(
    recipe: MergeRecipe,
    key: str,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None,
    cosine_context: dict[str, torch.Tensor | float] | None = None,
    tensor_index: int = 0,
) -> torch.Tensor:
    alpha = _block_weight_for_key(recipe, key, recipe.block_weights_alpha or recipe.block_weights, recipe.alpha, tensor_index)
    beta = _block_weight_for_key(recipe, key, recipe.block_weights_beta, recipe.beta, tensor_index)
    calc_mode = recipe.normalized_calc_mode()
    if calc_mode in V11_CALCULATION_MODES:
        try:
            value = merge_weight_sum_tensor(
                calc_mode,
                a.float(),
                b.float(),
                alpha,
                recipe.resolved_calculation_parameters(),
            )
        except CalculationModeError as exc:
            raise MergeExecutionError(f"{key}: {exc}") from exc
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode == "trainDifference":
        if c is None:
            raise MergeExecutionError(f"source C is missing key for trainDifference: {key}")
        value = _supermerger_train_difference(a, b, c, alpha)
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode == "extract":
        if c is None:
            raise MergeExecutionError(f"source C is missing key for extract: {key}")
        value = _supermerger_extract(a, b, c, alpha, beta, recipe.calculation_option)
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode in {"smoothAdd", "smoothAdd MT"}:
        if c is None:
            raise MergeExecutionError(f"source C is missing key for {calc_mode}: {key}")
        value = _supermerger_smooth_add(a, b, c, alpha)
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode in {"cosineA", "cosineB"}:
        value = _supermerger_cosine_merge(key, a, b, alpha, calc_mode, cosine_context or {})
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode in {"tensor", "tensor2"}:
        value = _replace_tensor_slice(
            a.float(),
            b.float(),
            alpha,
            beta,
            second_axis_for_wide_tensors=calc_mode == "tensor2",
        )
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode == "self":
        value = a.float() if alpha == 0.0 or _key_uses_self_skip(key) else a.float() * alpha
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if calc_mode == "plus random":
        value = _plus_random_noise(recipe, key, a, alpha)
        value = _apply_adjust_values(recipe, key, value)
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    if recipe.use_old_calc_method:
        mode = recipe.normalized_merge_mode()
        if mode == "add_difference":
            if c is None:
                raise MergeExecutionError(f"source C is missing key for add_difference: {key}")
            value = a + (b - c) * alpha
        elif mode == "triple_sum":
            if c is None:
                raise MergeExecutionError(f"source C is missing key for triple_sum: {key}")
            value = a * (1.0 - alpha - beta) + b * alpha + c * beta
        elif mode == "sum_twice":
            if c is None:
                raise MergeExecutionError(f"source C is missing key for sum_twice: {key}")
            value = (a * (1.0 - alpha) + b * alpha) * (1.0 - beta) + c * beta
        else:
            value = a * (1.0 - alpha) + b * alpha
        value = _apply_adjust_values(recipe, key, value.float())
        return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)
    a_float, b_float, c_float, alpha = _apply_calc_mode(recipe, a, b, c, alpha)
    mode = recipe.normalized_merge_mode()
    if mode == "add_difference":
        if c_float is None:
            raise MergeExecutionError(f"source C is missing key for add_difference: {key}")
        value = a_float + (b_float - c_float) * alpha
    elif mode == "triple_sum":
        if c_float is None:
            raise MergeExecutionError(f"source C is missing key for triple_sum: {key}")
        value = a_float * (1.0 - alpha - beta) + b_float * alpha + c_float * beta
    elif mode == "sum_twice":
        if c_float is None:
            raise MergeExecutionError(f"source C is missing key for sum_twice: {key}")
        value = (a_float * (1.0 - alpha) + b_float * alpha) * (1.0 - beta) + c_float * beta
    else:
        value = a_float * (1.0 - alpha) + b_float * alpha
    value = _apply_adjust_values(recipe, key, value)
    return value.to(dtype=torch.float16 if recipe.fp16 else a.dtype)


def _lora_tensor_scale(ratio: float, key: str, same_to_strength: bool) -> float:
    if not same_to_strength:
        return ratio
    scale = abs(ratio) ** 0.5
    if ratio < 0 and "lora_down" in key:
        return -scale
    return scale


def _merge_lora_tensor(recipe: MergeRecipe, key: str, a: torch.Tensor | None, b: torch.Tensor | None, tensor_index: int) -> torch.Tensor:
    ratio = _block_weight_for_key(recipe, key, recipe.block_weights_alpha or recipe.block_weights, recipe.alpha, tensor_index)
    a_scale = _lora_tensor_scale(1.0 - ratio, key, recipe.lora_same_to_strength)
    b_scale = _lora_tensor_scale(ratio, key, recipe.lora_same_to_strength)
    if a is None and b is None:
        raise MergeExecutionError(f"missing LoRA tensors for {key}")
    dtype_source = a if a is not None else b
    assert dtype_source is not None
    value = torch.zeros_like(dtype_source.float())
    if a is not None:
        value = value + a.float() * a_scale
    if b is not None:
        value = value + b.float() * b_scale
    return value.to(dtype=torch.float16 if recipe.fp16 else dtype_source.dtype)


def _lora_source_raw_scale(recipe: MergeRecipe, source: MergeSource, key: str, tensor_index: int) -> float:
    ratio = _block_weight_for_key(recipe, key, recipe.block_weights_alpha or recipe.block_weights, recipe.alpha, tensor_index)
    if source.role.upper() == "A":
        return 1.0 - ratio
    strength = source.weight if source.weight is not None else ratio
    if source.block_weights.strip():
        strength = _block_weight_for_key(recipe, key, source.block_weights, strength, tensor_index)
    return strength


def _merge_lora_tensor_from_sources(
    recipe: MergeRecipe,
    key: str,
    sources: list[tuple[str, MergeSource, dict[str, torch.Tensor]]],
    tensor_index: int,
) -> torch.Tensor:
    tensors = [(role, source, values[key]) for role, source, values in sources if key in values]
    if not tensors:
        raise MergeExecutionError(f"missing LoRA tensors for {key}")
    dtype_source = tensors[0][2]
    value = torch.zeros_like(dtype_source.float())
    for role, source, tensor in tensors:
        if tensor.shape != dtype_source.shape:
            raise MergeExecutionError(f"shape mismatch for {key}: {role}{tuple(tensor.shape)} != {tuple(dtype_source.shape)}")
        scale = _lora_tensor_scale(_lora_source_raw_scale(recipe, source, key, tensor_index), key, recipe.lora_same_to_strength)
        value = value + tensor.float() * scale
    return value.to(dtype=torch.float16 if recipe.fp16 else dtype_source.dtype)


def _lora_module_prefixes(*sources: dict[str, torch.Tensor]) -> list[str]:
    prefixes: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for key in source:
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key.removesuffix(".lora_down.weight")
            if prefix in seen:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes


def _lora_alpha_scale(tensors: dict[str, torch.Tensor], prefix: str, network_dim: int) -> float:
    alpha = tensors.get(f"{prefix}.alpha")
    if alpha is None:
        return 1.0
    return float(alpha.float().reshape(-1)[0].item()) / float(network_dim)


def _lora_module_delta(tensors: dict[str, torch.Tensor], prefix: str) -> torch.Tensor | None:
    down = tensors.get(f"{prefix}.lora_down.weight")
    up = tensors.get(f"{prefix}.lora_up.weight")
    if down is None or up is None:
        return None
    up_float = up.float()
    down_float = down.float()
    network_dim = down.shape[0]
    scale = _lora_alpha_scale(tensors, prefix, network_dim)
    if up.ndim == 2 and down.ndim == 2:
        return (up_float @ down_float) * scale
    if up.ndim == 4 and down.ndim == 4 and up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
        return (up_float[:, :, 0, 0] @ down_float[:, :, 0, 0]).unsqueeze(2).unsqueeze(3) * scale
    if up.ndim == 4 and down.ndim == 4:
        return torch.nn.functional.conv2d(down_float.permute(1, 0, 2, 3), up_float).permute(1, 0, 2, 3) * scale
    return None


def _remake_lora_module_with_svd(
    delta: torch.Tensor,
    rank: int,
    conv_rank: int,
    *,
    clamp_quantile: float | None = LORA_SVD_CLAMP_QUANTILE,
    prefer_truncated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if delta.ndim not in {2, 4}:
        return None
    conv2d = delta.ndim == 4
    kernel_size = None if not conv2d else delta.shape[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = delta.shape[0:2]
    matrix = delta.float()
    if conv2d:
        if conv2d_3x3:
            matrix = matrix.flatten(start_dim=1)
        else:
            matrix = matrix.squeeze()
    if matrix.ndim != 2 or matrix.numel() == 0 or not torch.isfinite(matrix).all():
        return None
    module_rank = conv_rank if conv2d_3x3 else rank
    module_rank = max(1, min(module_rank, in_dim, out_dim))
    use_truncated = prefer_truncated and matrix.numel() >= 4_000_000 and module_rank < min(matrix.shape)
    if use_truncated:
        q = min(min(matrix.shape), max(module_rank + 8, module_rank * 2))
        cuda_devices = [matrix.device] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(0)
            u, s, v = torch.svd_lowrank(matrix, q=q, niter=2)
        order = torch.argsort(s, descending=True)
        u = u[:, order]
        s = s[order]
        vh = v[:, order].mT
    else:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u = u[:, :module_rank] @ torch.diag(s[:module_rank])
    vh = vh[:module_rank, :]
    if clamp_quantile is not None:
        dist = torch.cat([u.flatten(), vh.flatten()])
        high = torch.quantile(dist, clamp_quantile)
        u = u.clamp(-high, high)
        vh = vh.clamp(-high, high)
    if conv2d:
        assert kernel_size is not None
        u = u.reshape(out_dim, module_rank, 1, 1)
        vh = vh.reshape(module_rank, in_dim, kernel_size[0], kernel_size[1])
    return u, vh, torch.tensor(module_rank)


def _merge_lora_tensors_with_svd(
    recipe: MergeRecipe,
    sources: list[tuple[str, MergeSource, dict[str, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], int, int]:
    rank = recipe.lora_rank or 4
    conv_rank = recipe.lora_conv_rank or rank
    merged: dict[str, torch.Tensor] = {}
    skipped = 0
    remade_modules = 0
    source_tensors = [tensors for _role, _source, tensors in sources]
    for tensor_index, prefix in enumerate(_lora_module_prefixes(*source_tensors)):
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"
        alpha_key = f"{prefix}.alpha"
        if not _should_merge_key(recipe, down_key):
            for key in (up_key, down_key, alpha_key):
                source_a = sources[0][2]
                if key in source_a:
                    merged[key] = source_a[key]
            skipped += 1
            continue
        delta: torch.Tensor | None = None
        for _role, source, tensors in sources:
            source_delta = _lora_module_delta(tensors, prefix)
            if source_delta is None:
                continue
            weighted_delta = source_delta * _lora_source_raw_scale(recipe, source, down_key, tensor_index)
            delta = weighted_delta if delta is None else delta + weighted_delta
        if delta is None:
            skipped += 1
            continue
        remade = _remake_lora_module_with_svd(delta, rank, conv_rank)
        if remade is None:
            skipped += 1
            continue
        up, down, alpha = remade
        dtype = torch.float16 if recipe.fp16 else torch.float32
        merged[up_key] = up.to(dtype=dtype).contiguous()
        merged[down_key] = down.to(dtype=dtype).contiguous()
        merged[alpha_key] = alpha
        remade_modules += 1
    if not remade_modules:
        raise MergeExecutionError("no remakable LoRA modules were found")
    return merged, skipped, remade_modules


def _sanitize_lora_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _lora_target_id(prefix: str) -> str:
    for marker in ("lora_unet_", "lora_te1_", "lora_te2_", "lora_te_"):
        if prefix.startswith(marker):
            return _sanitize_lora_key(prefix[len(marker):])
    return _sanitize_lora_key(prefix)


def _lora_prefix_for_key(key: str, schema: str = "sdxl") -> str:
    lowered = key.lower()
    if schema == "anima" and lowered.startswith("net.") and lowered.endswith(".weight"):
        return f"diffusion_model.{key[len('net.'):-len('.weight')]}"
    if schema in {"flux1", "flux2"}:
        canonical = _canonical_checkpoint_diffusion_key(key)
        if canonical.lower().endswith(".weight"):
            return f"diffusion_model.{canonical[:-len('.weight')]}"
    if lowered.startswith("conditioner.embedders.0.") or lowered.startswith("cond_stage_model."):
        marker = "lora_te1"
    elif lowered.startswith("conditioner.embedders.1."):
        marker = "lora_te2"
    else:
        marker = "lora_unet"
    return f"{marker}_{_sanitize_lora_key(key)}"


def _is_lora_extractable_checkpoint_key(key: str, tensor: torch.Tensor, schema: str = "sdxl") -> bool:
    lowered = key.lower()
    if not tensor.is_floating_point() or "weight" not in lowered:
        return False
    if lowered.startswith("first_stage_model.") or ".first_stage_model." in lowered:
        return False
    if schema == "anima":
        return lowered.startswith("net.") and lowered.endswith(".weight")
    if schema in {"flux1", "flux2"}:
        canonical = _canonical_checkpoint_diffusion_key(key).lower()
        flux_roots = (
            "double_blocks.",
            "single_blocks.",
            "double_stream_modulation_",
            "single_stream_modulation.",
            "final_layer.",
            "img_in.",
            "txt_in.",
            "time_in.",
            "vector_in.",
            "guidance_in.",
        )
        return canonical.endswith(".weight") and canonical.startswith(flux_roots)
    return (
        lowered.startswith("model.diffusion_model.")
        or lowered.startswith("conditioner.embedders.")
        or lowered.startswith("cond_stage_model.")
    )


def _kohya_delta(up: torch.Tensor, down: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    up_float = up.float()
    down_float = down.float()
    if target.ndim == 2 and up.ndim == 2 and down.ndim == 2:
        delta = up_float @ down_float
    elif (
        target.ndim == 4
        and up.ndim == 4
        and down.ndim == 4
        and up.shape[2:] == (1, 1)
        and down.shape[2:] == (1, 1)
    ):
        delta = (up_float[:, :, 0, 0] @ down_float[:, :, 0, 0]).reshape(up.shape[0], down.shape[1], 1, 1)
    elif target.ndim == 4 and up.ndim == 4 and down.ndim == 4:
        delta = torch.nn.functional.conv2d(down_float.permute(1, 0, 2, 3), up_float).permute(1, 0, 2, 3)
    else:
        return None
    if delta.shape != target.shape:
        return None
    return delta


def _sdxl_te2_openclip_delta(
    prefix: str,
    up: torch.Tensor,
    down: torch.Tensor,
    checkpoint_tensors: dict[str, torch.Tensor],
) -> tuple[str, torch.Tensor] | None:
    match = re.match(r"^lora_te2_text_model_encoder_layers_(\d+)_(.+)$", prefix)
    if match is None:
        return None
    layer = match.group(1)
    module = match.group(2)
    base = f"conditioner.embedders.1.model.transformer.resblocks.{layer}"
    simple_targets = {
        "mlp_fc1": f"{base}.mlp.c_fc.weight",
        "mlp_fc2": f"{base}.mlp.c_proj.weight",
        "self_attn_out_proj": f"{base}.attn.out_proj.weight",
    }
    target_key = simple_targets.get(module)
    if target_key is not None:
        target = checkpoint_tensors.get(target_key)
        if target is None:
            return None
        delta = _kohya_delta(up, down, target)
        return (target_key, delta) if delta is not None else None
    qkv_offsets = {"self_attn_q_proj": 0, "self_attn_k_proj": 1, "self_attn_v_proj": 2}
    qkv_index = qkv_offsets.get(module)
    if qkv_index is None:
        return None
    target_key = f"{base}.attn.in_proj_weight"
    target = checkpoint_tensors.get(target_key)
    if target is None or target.ndim != 2 or target.shape[0] % 3 != 0:
        return None
    block = target.shape[0] // 3
    start = qkv_index * block
    end = start + block
    delta = _kohya_delta(up, down, target[start:end])
    if delta is None:
        return None
    full_delta = torch.zeros_like(target.float())
    full_delta[start:end] = delta
    return target_key, full_delta


def _svd_lora_pair(
    delta: torch.Tensor,
    rank: int,
    conv_rank: int,
    *,
    clamp_quantile: float | None = LORA_SVD_CLAMP_QUANTILE,
    prefer_truncated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    remade = _remake_lora_module_with_svd(
        delta,
        rank,
        conv_rank,
        clamp_quantile=clamp_quantile,
        prefer_truncated=prefer_truncated,
    )
    if remade is None:
        return None
    up, down, _alpha = remade
    return up, down


def _extract_lora_from_difference(
    base_tensors: dict[str, torch.Tensor],
    target_tensors: dict[str, torch.Tensor],
    recipe: MergeRecipe,
) -> tuple[dict[str, torch.Tensor], int]:
    rank = recipe.lora_rank or 8
    conv_rank = recipe.lora_conv_rank or rank
    extracted: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, base in base_tensors.items():
        if not _is_lora_extractable_checkpoint_key(key, base, recipe.block_schema):
            skipped += 1
            continue
        target = _checkpoint_tensor_for_key(target_tensors, key)
        if target is None or target.shape != base.shape:
            skipped += 1
            continue
        delta = (target.float() - base.float()) * recipe.lora_source_ratio
        if not torch.any(delta != 0).item():
            skipped += 1
            continue
        pair = _svd_lora_pair(
            delta,
            rank,
            conv_rank,
            clamp_quantile=None if recipe.block_schema in {"anima", "flux1", "flux2"} else LORA_SVD_CLAMP_QUANTILE,
            prefer_truncated=recipe.block_schema in {"flux1", "flux2"},
        )
        if pair is None:
            skipped += 1
            continue
        up, down = pair
        prefix = _lora_prefix_for_key(key, recipe.block_schema)
        dtype = torch.float16 if recipe.fp16 else torch.float32
        if recipe.block_schema in {"anima", "flux1", "flux2"}:
            extracted[f"{prefix}.lora_B.weight"] = up.to(dtype=dtype).contiguous()
            extracted[f"{prefix}.lora_A.weight"] = down.to(dtype=dtype).contiguous()
        else:
            extracted[f"{prefix}.lora_up.weight"] = up.to(dtype=dtype).contiguous()
            extracted[f"{prefix}.lora_down.weight"] = down.to(dtype=dtype).contiguous()
        extracted[f"{prefix}.alpha"] = torch.tensor(float(down.shape[0]), dtype=torch.float32)
    if not extracted:
        raise MergeExecutionError("no extractable linear or convolution tensors were found")
    return extracted, skipped


def _extract_flux_lora_from_safetensors_difference(
    base_path: Path,
    target_path: Path,
    recipe: MergeRecipe,
    device: str,
) -> tuple[dict[str, torch.Tensor], int]:
    rank = recipe.lora_rank or 8
    conv_rank = recipe.lora_conv_rank or rank
    extracted: dict[str, torch.Tensor] = {}
    skipped = 0
    try:
        with (
            safe_open(str(base_path), framework="pt", device="cpu") as base_handle,
            safe_open(str(target_path), framework="pt", device="cpu") as target_handle,
        ):
            target_keys = {
                _canonical_checkpoint_diffusion_key(key): key
                for key in target_handle.keys()
            }
            for key in base_handle.keys():
                base = base_handle.get_tensor(key)
                if not _is_lora_extractable_checkpoint_key(key, base, recipe.block_schema):
                    skipped += 1
                    continue
                target_key = target_keys.get(_canonical_checkpoint_diffusion_key(key))
                if target_key is None:
                    skipped += 1
                    continue
                target = target_handle.get_tensor(target_key)
                if target.shape != base.shape:
                    skipped += 1
                    continue
                base_work = base.to(device=device, dtype=torch.float32)
                target_work = target.to(device=device, dtype=torch.float32)
                delta = (target_work - base_work) * recipe.lora_source_ratio
                if not torch.any(delta != 0).item():
                    skipped += 1
                    continue
                pair = _svd_lora_pair(
                    delta,
                    rank,
                    conv_rank,
                    clamp_quantile=None,
                    prefer_truncated=True,
                )
                if pair is None:
                    skipped += 1
                    continue
                up, down = pair
                prefix = _lora_prefix_for_key(key, recipe.block_schema)
                dtype = torch.float16 if recipe.fp16 else torch.float32
                extracted[f"{prefix}.lora_B.weight"] = up.to(device="cpu", dtype=dtype).contiguous()
                extracted[f"{prefix}.lora_A.weight"] = down.to(device="cpu", dtype=dtype).contiguous()
                extracted[f"{prefix}.alpha"] = torch.tensor(float(down.shape[0]), dtype=torch.float32)
                del base_work, target_work, delta, up, down
                if device == "cuda":
                    torch.cuda.empty_cache()
    except MergeExecutionError:
        raise
    except Exception as exc:
        raise MergeExecutionError(
            f"{_flux_family_label(recipe.block_schema)} checkpoint difference could not be streamed from safetensors: "
            f"{base_path}, {target_path}"
        ) from exc
    if not extracted:
        raise MergeExecutionError("no extractable linear or convolution tensors were found")
    return extracted, skipped


def _looks_like_lora_tensors(tensors: dict[str, torch.Tensor]) -> bool:
    return any(".lora_" in key or key.endswith(".alpha") for key in tensors)


def _unsupported_lora_format_reason(tensors: dict[str, torch.Tensor]) -> str | None:
    lowered_keys = [key.lower() for key in tensors]
    if any("hada_" in key for key in lowered_keys):
        return "LyCORIS/Hada"
    if any("lokr_" in key or ".lokr" in key for key in lowered_keys):
        return "LyCORIS/LoKr"
    if any("ia3_" in key or ".ia3" in key for key in lowered_keys):
        return "LyCORIS/IA3"
    return None


def _raise_for_unsupported_lora_format(role: str, tensors: dict[str, torch.Tensor]) -> None:
    reason = _unsupported_lora_format_reason(tensors)
    if reason is None:
        return
    raise MergeExecutionError(
        f"unsupported LoRA format in source {role}: {reason} keys are not supported for Merge Studio tensor merge yet. "
        "LyCORIS/Hada/LoKr-style tensor keys require a dedicated adapter; "
        "use a LoRA/LoCon file with lora_down/lora_up weights"
    )


def _flux1_diffusers_lora_target_spec(prefix: str) -> tuple[str, str | None] | None:
    double_match = re.fullmatch(r"transformer\.transformer_blocks\.(\d+)\.(.+)", prefix)
    if double_match is not None:
        block, module = double_match.groups()
        sliced_modules = {
            "attn.to_q": ("img_attn.qkv", "q"),
            "attn.to_k": ("img_attn.qkv", "k"),
            "attn.to_v": ("img_attn.qkv", "v"),
            "attn.add_q_proj": ("txt_attn.qkv", "q"),
            "attn.add_k_proj": ("txt_attn.qkv", "k"),
            "attn.add_v_proj": ("txt_attn.qkv", "v"),
        }
        sliced = sliced_modules.get(module)
        if sliced is not None:
            target_module, slice_kind = sliced
            return f"double_blocks.{block}.{target_module}.weight", slice_kind
        direct_modules = {
            "attn.to_out.0": "img_attn.proj",
            "attn.to_add_out": "txt_attn.proj",
            "norm1.linear": "img_mod.lin",
            "norm1_context.linear": "txt_mod.lin",
            "ff.net.0.proj": "img_mlp.0",
            "ff.net.2": "img_mlp.2",
            "ff_context.net.0.proj": "txt_mlp.0",
            "ff_context.net.2": "txt_mlp.2",
        }
        target_module = direct_modules.get(module)
        if target_module is not None:
            return f"double_blocks.{block}.{target_module}.weight", None
        return None

    single_match = re.fullmatch(r"transformer\.single_transformer_blocks\.(\d+)\.(.+)", prefix)
    if single_match is None:
        return None
    block, module = single_match.groups()
    if module in {"attn.to_q", "attn.to_k", "attn.to_v"}:
        return f"single_blocks.{block}.linear1.weight", module[-1]
    if module == "proj_mlp":
        return f"single_blocks.{block}.linear1.weight", "mlp"
    direct_modules = {
        "norm.linear": "modulation.lin",
        "proj_out": "linear2",
    }
    target_module = direct_modules.get(module)
    if target_module is None:
        return None
    return f"single_blocks.{block}.{target_module}.weight", None


def _flux_lora_slice_bounds(
    slice_kind: str | None,
    target: torch.Tensor,
    up: torch.Tensor,
) -> tuple[int, int, int] | None:
    if slice_kind is None:
        return None
    if target.ndim != 2 or up.ndim != 2:
        return None
    length = int(up.shape[0])
    if length <= 0:
        return None
    if slice_kind in {"q", "k", "v"}:
        if length != target.shape[1] or target.shape[0] < length * 3:
            return None
        start = {"q": 0, "k": length, "v": length * 2}[slice_kind]
    elif slice_kind == "mlp":
        attention_rows = int(target.shape[0]) - length
        if attention_rows != target.shape[1] * 3:
            return None
        start = int(target.shape[0]) - length
    else:
        return None
    if start < 0 or start + length > target.shape[0]:
        return None
    return 0, start, length


def _checkpoint_lora_module_pairs(
    lora_tensors: dict[str, torch.Tensor],
    recipe: MergeRecipe,
) -> list[tuple[int, str, str, str, tuple[str, str | None] | None]]:
    pairs: list[tuple[int, str, str, str, tuple[str, str | None] | None]] = []
    for tensor_index, key in enumerate(lora_tensors):
        if key.endswith(".lora_down.weight"):
            prefix = key.removesuffix(".lora_down.weight")
            up_key = f"{prefix}.lora_up.weight"
            if up_key in lora_tensors:
                pairs.append((tensor_index, prefix, key, up_key, None))
            continue
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key.removesuffix(".lora_A.weight")
        up_key = f"{prefix}.lora_B.weight"
        if up_key not in lora_tensors:
            continue
        target_spec: tuple[str, str | None] | None = None
        if prefix.startswith("diffusion_model."):
            module = prefix.removeprefix("diffusion_model.")
            if recipe.block_schema == "anima":
                target_spec = (f"net.{module}.weight", None)
            elif recipe.block_schema in {"flux1", "flux2"}:
                target_spec = (f"{module}.weight", None)
        elif recipe.block_schema == "flux1":
            target_spec = _flux1_diffusers_lora_target_spec(prefix)
        if target_spec is None:
            continue
        pairs.append((tensor_index, prefix, key, up_key, target_spec))
    return pairs


def _extract_two_loras(
    source_a: dict[str, torch.Tensor],
    source_b: dict[str, torch.Tensor],
    recipe: MergeRecipe,
) -> tuple[dict[str, torch.Tensor], int]:
    extracted: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, tensor_a in source_a.items():
        if key.endswith(".alpha"):
            extracted[key] = tensor_a
            continue
        tensor_b = source_b.get(key)
        if tensor_b is None or tensor_b.shape != tensor_a.shape:
            skipped += 1
            continue
        value = _supermerger_extract(
            torch.zeros_like(tensor_a),
            tensor_a,
            tensor_b,
            recipe.alpha,
            recipe.beta,
            recipe.calculation_option,
        )
        extracted[key] = value.to(dtype=torch.float16 if recipe.fp16 else tensor_a.dtype)
    for key, tensor_b in source_b.items():
        if key.endswith(".alpha") and key not in extracted:
            extracted[key] = tensor_b
    if not any(".lora_" in key for key in extracted):
        raise MergeExecutionError("no extractable common LoRA tensors were found")
    return extracted, skipped


def _lora_deltas_for_checkpoint(
    lora_tensors: dict[str, torch.Tensor],
    checkpoint_tensors: dict[str, torch.Tensor],
    recipe: MergeRecipe,
    source: MergeSource,
    *,
    include_direct_pairs: bool = True,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    target_lookup = {_sanitize_lora_key(key): key for key in checkpoint_tensors}
    target_items = [(key, _sanitize_lora_key(key)) for key in checkpoint_tensors]
    deltas: dict[str, torch.Tensor] = {}
    applied_lora_keys: set[str] = set()
    for tensor_index, prefix, down_key, up_key, direct_target_spec in _checkpoint_lora_module_pairs(lora_tensors, recipe):
        if direct_target_spec is not None and not include_direct_pairs:
            continue
        down = lora_tensors[down_key]
        up = lora_tensors[up_key]
        matched = None
        if direct_target_spec is not None:
            direct_target_key, slice_kind = direct_target_spec
            resolved_target_key = _checkpoint_key_for_alias(checkpoint_tensors, direct_target_key)
            target = checkpoint_tensors.get(resolved_target_key) if resolved_target_key is not None else None
            bounds = _flux_lora_slice_bounds(slice_kind, target, up) if target is not None else None
            delta_target = None if slice_kind is not None and bounds is None else target
            if delta_target is not None and bounds is not None:
                delta_target = delta_target.narrow(*bounds)
            delta = _kohya_delta(up, down, delta_target) if delta_target is not None else None
            if delta is not None:
                if bounds is not None and target is not None:
                    full_delta = torch.zeros_like(target.float())
                    full_delta.narrow(*bounds).copy_(delta)
                    delta = full_delta
                matched = (resolved_target_key, delta)
        else:
            matched = _sdxl_te2_openclip_delta(prefix, up, down, checkpoint_tensors) if prefix.startswith("lora_te2_") else None
            target_id = _lora_target_id(prefix)
            target_ids = {target_id, f"{target_id}weight"}
            target_candidates = []
            exact_target = target_lookup.get(target_id)
            if exact_target is not None:
                target_candidates.append(exact_target)
            target_candidates.extend(key for key, sanitized in target_items if any(sanitized.endswith(item) for item in target_ids) and key != exact_target)
            if matched is None:
                for target_key in target_candidates:
                    target = checkpoint_tensors[target_key]
                    delta = _kohya_delta(up, down, target)
                    if delta is not None:
                        matched = (target_key, delta)
                        break
            if matched is None:
                matched = _sdxl_te2_openclip_delta(prefix, up, down, checkpoint_tensors)
        if matched is None:
            continue
        target_key, delta = matched
        alpha_key = f"{prefix}.alpha"
        rank = max(1, int(down.shape[0]))
        alpha_tensor = lora_tensors.get(alpha_key)
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None and alpha_tensor.numel() == 1 else float(rank)
        strength = _checkpoint_lora_strength_for_target(recipe, source, target_key, tensor_index)
        scale = strength * alpha / rank
        target_tensor = checkpoint_tensors[target_key]
        deltas[target_key] = deltas.get(target_key, torch.zeros_like(target_tensor.float())) + delta * scale
        applied_lora_keys.update({down_key, up_key})
        if alpha_key in lora_tensors:
            applied_lora_keys.add(alpha_key)
    return deltas, applied_lora_keys


def _apply_flux_lora_source_incrementally(
    lora_tensors: dict[str, torch.Tensor],
    checkpoint_tensors: dict[str, torch.Tensor],
    applied_tensors: dict[str, torch.Tensor],
    recipe: MergeRecipe,
    source: MergeSource,
    compute_device: str,
) -> tuple[set[str], set[str], set[str]]:
    applied_targets: set[str] = set()
    applied_lora_keys: set[str] = set()
    promoted_targets: set[str] = set()
    for tensor_index, prefix, down_key, up_key, direct_target_spec in _checkpoint_lora_module_pairs(
        lora_tensors,
        recipe,
    ):
        if direct_target_spec is None:
            continue
        direct_target_key, slice_kind = direct_target_spec
        target_key = _checkpoint_key_for_alias(checkpoint_tensors, direct_target_key)
        if target_key is None:
            continue
        base_tensor = checkpoint_tensors[target_key]
        down = lora_tensors[down_key].to(device=compute_device)
        up = lora_tensors[up_key].to(device=compute_device)
        current = applied_tensors.get(target_key, base_tensor).to(device=compute_device)
        bounds = _flux_lora_slice_bounds(slice_kind, current, up)
        if slice_kind is not None and bounds is None:
            del current, down, up
            if compute_device == "cuda":
                torch.cuda.empty_cache()
            continue
        delta_target = current.narrow(*bounds) if bounds is not None else current
        delta = _kohya_delta(up, down, delta_target)
        if delta is None:
            continue
        alpha_key = f"{prefix}.alpha"
        rank = max(1, int(down.shape[0]))
        alpha_tensor = lora_tensors.get(alpha_key)
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None and alpha_tensor.numel() == 1 else float(rank)
        strength = _checkpoint_lora_strength_for_target(recipe, source, target_key, tensor_index)
        scaled_delta = delta * (strength * alpha / rank)
        if bounds is None:
            applied, promoted_float8 = _apply_materialized_lora_delta(recipe, current, scaled_delta)
        else:
            applied_slice, promoted_float8 = _apply_materialized_lora_delta(
                recipe,
                current.narrow(*bounds),
                scaled_delta,
            )
            applied = current.to(dtype=_materialized_checkpoint_dtype(recipe, current)).clone()
            applied.narrow(*bounds).copy_(applied_slice)
        applied_tensors[target_key] = applied.detach().cpu()
        applied_targets.add(target_key)
        applied_lora_keys.update({down_key, up_key})
        if alpha_key in lora_tensors:
            applied_lora_keys.add(alpha_key)
        if promoted_float8:
            promoted_targets.add(target_key)
        del current, applied, delta, scaled_delta, down, up
        if compute_device == "cuda":
            torch.cuda.empty_cache()
    return applied_targets, applied_lora_keys, promoted_targets


def _pruned_tensors(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    pruned: dict[str, torch.Tensor] = {}
    removed: list[str] = []
    for key, tensor in tensors.items():
        lowered = key.lower()
        if (
            lowered.startswith("optimizer.")
            or lowered.startswith("state.")
            or lowered.startswith("loss.")
            or ".optimizer." in lowered
            or "model_ema" in lowered
            or "ema_decay" in lowered
        ):
            removed.append(key)
            continue
        pruned[key] = tensor
    return pruned, removed


def _reset_clip_position_ids(tensors: dict[str, torch.Tensor]) -> int:
    reset_count = 0
    for key, tensor in list(tensors.items()):
        lowered = key.lower()
        if "clip" not in lowered or not lowered.endswith("position_ids"):
            continue
        if tensor.ndim == 0:
            continue
        values = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape)
        tensors[key] = values.to(device=tensor.device)
        reset_count += 1
    return reset_count


def _bake_vae_tensors(recipe: MergeRecipe, tensors: dict[str, torch.Tensor]) -> int:
    if not recipe.bake_in_vae.strip():
        return 0
    vae_path = Path(recipe.bake_in_vae)
    if not _path_exists(vae_path) or not _path_is_file(vae_path):
        raise MergeExecutionError(f"bake-in VAE does not exist: {vae_path}")
    if vae_path.suffix.lower() != ".safetensors":
        raise MergeExecutionError(f"bake-in VAE must be a safetensors file: {vae_path}")
    vae_tensors = _load_safetensors(vae_path, "VAE")
    baked = 0
    for key, tensor in vae_tensors.items():
        candidates = []
        if key.startswith("first_stage_model."):
            candidates.append(key)
        if key.startswith("vae."):
            candidates.append(f"first_stage_model.{key.removeprefix('vae.')}")
        candidates.append(f"first_stage_model.{key}")
        for target_key in candidates:
            target = tensors.get(target_key)
            if target is None or target.shape != tensor.shape:
                continue
            tensors[target_key] = tensor.to(dtype=torch.float16 if recipe.fp16 else target.dtype)
            baked += 1
            break
    if baked == 0:
        raise MergeExecutionError(f"bake-in VAE had no matching first_stage_model tensors: {vae_path}")
    return baked


def _debug_tensor_entry(recipe: MergeRecipe, key: str, tensor: torch.Tensor, tensor_index: int, action: str) -> dict[str, object]:
    return {
        "index": tensor_index,
        "key": key,
        "block": _block_id_for_key(key, recipe.block_schema),
        "alpha": _block_weight_for_key(recipe, key, recipe.block_weights_alpha or recipe.block_weights, recipe.alpha, tensor_index),
        "beta": _block_weight_for_key(recipe, key, recipe.block_weights_beta, recipe.beta, tensor_index),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "action": action,
    }


def _output_metadata(recipe: MergeRecipe, sources: dict[str, Path], metadata: dict[str, str]) -> dict[str, str]:
    mode = recipe.lora_metadata_mode.strip().lower()
    if mode == "create":
        mode = "create_new"
    inherited: dict[str, str] = {}
    if mode == "use_first":
        inherited.update(_safetensors_metadata(sources["A"]))
    elif mode == "merge":
        inherited.update(_safetensors_metadata(sources["A"]))
        if recipe.merge_type == "lora_to_checkpoint":
            for role in ("B", "C"):
                source = sources.get(role)
                if source is None:
                    continue
                for key, value in _safetensors_metadata(source).items():
                    inherited[f"asteria.source.{role}.metadata.{key}"] = value
        else:
            for role in ("B", "C"):
                source = sources.get(role)
                if source is not None:
                    inherited.update(_safetensors_metadata(source))
    elif mode == "save_all":
        for role in ("A", "B", "C"):
            source = sources.get(role)
            if source is None:
                continue
            for key, value in _safetensors_metadata(source).items():
                inherited[f"asteria.source.{role}.metadata.{key}"] = value
    return {**inherited, **_kohya_minimum_lora_metadata(recipe, sources), **metadata}


def _kohya_minimum_lora_metadata(recipe: MergeRecipe, sources: dict[str, Path]) -> dict[str, str]:
    if recipe.merge_type != "lora" or (recipe.lora_rank is None and recipe.lora_conv_rank is None):
        return {}
    rank = recipe.lora_rank or 4
    conv_rank = recipe.lora_conv_rank or rank
    output = {
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(rank),
        "ss_network_alpha": str(rank),
        "ss_network_args": json.dumps({"conv_dim": conv_rank, "conv_alpha": conv_rank}),
    }
    for source in sources.values():
        source_metadata = _safetensors_metadata(source)
        for key in ("ss_v2", "ss_base_model_version"):
            if key in source_metadata and key not in output:
                output[key] = source_metadata[key]
    return output


def _copy_source_configs(recipe: MergeRecipe, sources: dict[str, Path], output_path: Path) -> list[dict[str, str]]:
    if not recipe.copy_config:
        return []
    copied: list[dict[str, str]] = []
    suffixes = (".yaml", ".yml", ".json", ".toml", ".txt")
    seen: set[Path] = set()
    for role in ("A", "B", "C"):
        source = sources.get(role)
        if source is None:
            continue
        for suffix in suffixes:
            config_path = source.with_suffix(suffix)
            if config_path in seen or not _path_is_file(config_path):
                continue
            seen.add(config_path)
            if len(copied) == 0:
                destination = output_path.with_suffix(suffix)
            else:
                destination = output_path.with_name(f"{output_path.stem}.{role.lower()}{suffix}")
            try:
                shutil.copy2(config_path, destination)
            except OSError as exc:
                raise MergeExecutionError(f"merge config copy failed: {config_path}") from exc
            copied.append({"role": role, "source": str(config_path), "destination": str(destination)})
    return copied


def _cpu_tensors_for_save(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in tensors.items()}


def _materialized_checkpoint_dtype(recipe: MergeRecipe, tensor: torch.Tensor) -> torch.dtype:
    float8_dtypes = {
        dtype
        for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
        if (dtype := getattr(torch, name, None)) is not None
    }
    if recipe.fp16 or tensor.dtype in float8_dtypes:
        return torch.float16
    return tensor.dtype


def _apply_materialized_lora_delta(
    recipe: MergeRecipe,
    current: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    output_dtype = _materialized_checkpoint_dtype(recipe, current)
    promoted_float8 = output_dtype == torch.float16 and current.dtype != torch.float16 and not recipe.fp16
    if promoted_float8:
        return (current.float() + delta.float()).to(dtype=output_dtype), True
    return current.to(dtype=output_dtype) + delta.to(dtype=output_dtype), False


def execute_merge_recipe(
    recipe: MergeRecipe,
    output_root: Path,
    transient: bool = False,
    *,
    fingerprint_service: ModelFingerprintService | None = None,
) -> MergeRun:
    if recipe.merge_type not in MERGE_TYPES:
        raise MergeExecutionError("invalid merge type")
    fingerprints = fingerprint_service or ModelFingerprintService(
        output_root / ".asteria" / "model-fingerprints.json"
    )
    expected_source_fingerprints = [
        dict(item) for item in recipe.source_fingerprints
    ]
    recipe = _apply_dice(recipe)
    recipe = _apply_prompt_ratio_source(recipe)
    source_a = _source(recipe, "A")
    source_b = _source(recipe, "B")
    source_c = _source(recipe, "C") if MERGE_MODES.get(recipe.normalized_merge_mode(), {}).get("requires_c") else None
    sources = {"A": source_a, "B": source_b, **({"C": source_c} if source_c is not None else {})}
    lora_checkpoint_sources = _lora_to_checkpoint_sources(recipe) if recipe.merge_type == "lora_to_checkpoint" else []
    if recipe.merge_type == "lora_to_checkpoint":
        sources.update({role: path for role, _source_item, path in lora_checkpoint_sources if role not in sources})
    try:
        resolved_source_fingerprints = _source_content_fingerprints(
            recipe,
            sources,
            fingerprints,
            refresh_hash=True,
        )
    except OSError as exc:
        raise MergeExecutionError("merge source provenance hash failed") from exc
    if (
        expected_source_fingerprints
        and resolved_source_fingerprints != expected_source_fingerprints
    ):
        raise MergeExecutionError("merge source changed after enqueue")
    recipe.source_fingerprints = resolved_source_fingerprints
    logs = [f"loading A: {source_a}", f"loading B: {source_b}"]
    merge_device = _merge_device(recipe, logs)
    stream_flux_extract = recipe.merge_type == "extract_lora" and recipe.block_schema in {"flux1", "flux2"}
    stream_flux_materialization = (
        recipe.merge_type == "lora_to_checkpoint" and recipe.block_schema in {"flux1", "flux2"}
    )
    flux_family_label = _flux_family_label(recipe.block_schema)
    if stream_flux_extract:
        logs.append(
            f"{flux_family_label} extraction: streaming checkpoint tensors with truncated SVD for large matrices"
        )
        tensors_a: dict[str, torch.Tensor] = {}
        tensors_b: dict[str, torch.Tensor] = {}
    elif stream_flux_materialization:
        logs.append(
            f"{flux_family_label} materialization: checkpoint stays on CPU and LoRA targets are applied per tensor"
        )
        tensors_a = _load_safetensors(source_a, "A", "cpu")
        tensors_b = _load_safetensors(source_b, "B", "cpu")
    else:
        tensors_a = _load_safetensors(source_a, "A", merge_device)
        tensors_b = _load_safetensors(source_b, "B", merge_device)
    tensors_c = _load_safetensors(source_c, "C", merge_device) if source_c is not None else None
    if tensors_c is not None:
        logs.append(f"loading C: {source_c}")
    source_items_by_role = {source.role.upper(): source for source in recipe.sources if source.path}
    lora_merge_sources: list[tuple[str, MergeSource, dict[str, torch.Tensor]]] = []
    if recipe.merge_type == "lora":
        lora_merge_sources.append(("A", source_items_by_role["A"], tensors_a))
        lora_merge_sources.append(("B", source_items_by_role["B"], tensors_b))
        seen_lora_sources = {"A", "B"}
        if tensors_c is not None and "C" in source_items_by_role:
            lora_merge_sources.append(("C", source_items_by_role["C"], tensors_c))
            seen_lora_sources.add("C")
        for role, source_item, path in _lora_to_checkpoint_sources(recipe):
            if role in seen_lora_sources:
                continue
            logs.append(f"loading {role}: {path}")
            lora_merge_sources.append((role, source_item, _load_safetensors(path, role, merge_device)))
            sources[role] = path
            seen_lora_sources.add(role)
    if recipe.merge_type in {"lora", "extract_lora"} and not stream_flux_extract:
        _raise_for_unsupported_lora_format("A", tensors_a)
        _raise_for_unsupported_lora_format("B", tensors_b)
        if tensors_c is not None:
            _raise_for_unsupported_lora_format("C", tensors_c)
        for role, _source_item, lora_tensors in lora_merge_sources:
            if role in {"A", "B", "C"}:
                continue
            _raise_for_unsupported_lora_format(role, lora_tensors)
    adjust_target_tensors = _adjust_target_tensor_count(recipe, tensors_a)
    if recipe.adjust_values:
        logs.append(f"adjust target tensors: {adjust_target_tensors}")
    merged: dict[str, torch.Tensor] = {}
    lora_applied_tensors: dict[str, torch.Tensor] = {}
    fp8_promoted_lora_tensors: set[str] = set()
    applied_lora_keys: set[str] = set()
    debug_tensors: list[dict[str, object]] = []
    if recipe.debug:
        logs.append("debug tensor trace: enabled")
    if recipe.use_old_calc_method:
        logs.append("old calc compatibility: enabled")
    if recipe.merge_type == "lora_to_checkpoint":
        applied_lora_sources = 0
        for role, source_item, path in lora_checkpoint_sources:
            if role == "B" and path == source_b:
                lora_tensors = tensors_b
            else:
                logs.append(f"loading {role}: {path}")
                lora_tensors = _load_safetensors(path, role, "cpu" if stream_flux_materialization else merge_device)
            _raise_for_unsupported_lora_format(role, lora_tensors)
            if stream_flux_materialization:
                source_targets, source_applied_keys, source_promoted_targets = _apply_flux_lora_source_incrementally(
                    lora_tensors,
                    tensors_a,
                    lora_applied_tensors,
                    recipe,
                    source_item,
                    merge_device,
                )
                legacy_deltas, legacy_applied_keys = _lora_deltas_for_checkpoint(
                    lora_tensors,
                    tensors_a,
                    recipe,
                    source_item,
                    include_direct_pairs=False,
                )
                for key, delta in legacy_deltas.items():
                    current = lora_applied_tensors.get(key, tensors_a[key])
                    applied, promoted_float8 = _apply_materialized_lora_delta(recipe, current, delta)
                    lora_applied_tensors[key] = applied
                    if promoted_float8:
                        source_promoted_targets.add(key)
                if source_targets or legacy_deltas:
                    applied_lora_sources += 1
                applied_lora_keys.update(source_applied_keys)
                applied_lora_keys.update(legacy_applied_keys)
                fp8_promoted_lora_tensors.update(source_promoted_targets)
                del legacy_deltas
            else:
                source_deltas, source_applied_keys = _lora_deltas_for_checkpoint(
                    lora_tensors,
                    tensors_a,
                    recipe,
                    source_item,
                )
                if source_deltas:
                    applied_lora_sources += 1
                for key, delta in source_deltas.items():
                    current = lora_applied_tensors.get(key, tensors_a[key])
                    applied, promoted_float8 = _apply_materialized_lora_delta(recipe, current, delta)
                    lora_applied_tensors[key] = applied
                    if promoted_float8:
                        fp8_promoted_lora_tensors.add(key)
                applied_lora_keys.update(source_applied_keys)
                del source_deltas
        logs.append(f"applied lora sources: {applied_lora_sources}")
        logs.append(f"applied lora deltas: {len(lora_applied_tensors)}")
        logs.append(f"promoted FP8 LoRA targets to FP16: {len(fp8_promoted_lora_tensors)}")
        if not lora_applied_tensors:
            raise MergeExecutionError("no applicable LoRA tensors found for checkpoint")
    skipped = 0
    if recipe.merge_type == "extract_lora":
        if stream_flux_extract:
            merged, skipped = _extract_flux_lora_from_safetensors_difference(
                source_a,
                source_b,
                recipe,
                merge_device,
            )
        elif _looks_like_lora_tensors(tensors_a) and _looks_like_lora_tensors(tensors_b):
            merged, skipped = _extract_two_loras(tensors_a, tensors_b, recipe)
        else:
            merged, skipped = _extract_lora_from_difference(tensors_a, tensors_b, recipe)
        logs.append(f"extracted lora tensors: {len(merged)}")
    elif (
        recipe.merge_type == "lora"
        and recipe.normalized_merge_mode() == "weight_sum"
        and recipe.normalized_calc_mode() == "normal"
        and (recipe.lora_rank is not None or recipe.lora_conv_rank is not None)
    ):
        merged, skipped, remade_modules = _merge_lora_tensors_with_svd(recipe, lora_merge_sources)
        logs.append(f"lora svd remade modules: {remade_modules}")
        logs.append(f"mergeable common tensors: {remade_modules}")
    else:
        mergeable_common_keys = 0
        cosine_context = (
            _cosine_context(tensors_a, tensors_b, recipe.normalized_calc_mode())
            if recipe.normalized_calc_mode() in {"cosineA", "cosineB"}
            else None
        )
        for tensor_index, (key, tensor_a) in enumerate(tensors_a.items()):
            if not _should_merge_key(recipe, key):
                merged[key] = tensor_a
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "skipped"))
                skipped += 1
                continue
            if recipe.merge_type == "lora" and key.endswith(".alpha"):
                merged[key] = tensor_a
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "preserved_alpha"))
                continue
            if recipe.merge_type == "lora" and recipe.normalized_merge_mode() == "weight_sum" and recipe.normalized_calc_mode() == "normal":
                merged[key] = _merge_lora_tensor_from_sources(recipe, key, lora_merge_sources, tensor_index)
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "merged"))
                if key in tensors_b:
                    mergeable_common_keys += 1
                continue
            if recipe.merge_type == "checkpoint" and not _is_checkpoint_arithmetic_tensor(key, tensor_a):
                merged[key] = tensor_a
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "preserved_non_mergeable"))
                skipped += 1
                continue
            if recipe.merge_type == "lora_to_checkpoint":
                applied_tensor = lora_applied_tensors.get(key)
                if applied_tensor is None:
                    merged[key] = tensor_a
                    action = "preserved"
                else:
                    merged[key] = applied_tensor
                    action = "applied_lora_delta"
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, action))
                continue
            tensor_b = _checkpoint_tensor_for_key(tensors_b, key) if recipe.merge_type == "checkpoint" else tensors_b.get(key)
            if tensor_b is None:
                merged[key] = tensor_a
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "missing_in_b"))
                continue
            if tensor_a.shape != tensor_b.shape:
                raise MergeExecutionError(f"shape mismatch for {key}: {tuple(tensor_a.shape)} != {tuple(tensor_b.shape)}")
            tensor_c = None
            if tensors_c is not None:
                if key not in tensors_c:
                    raise MergeExecutionError(f"source C is missing key for {recipe.normalized_merge_mode()}: {key}")
                tensor_c = tensors_c[key]
                if tensor_b.shape != tensor_c.shape:
                    raise MergeExecutionError(f"shape mismatch for {key}: B{tuple(tensor_b.shape)} != C{tuple(tensor_c.shape)}")
            mergeable_common_keys += 1
            merged[key] = _merge_tensor(recipe, key, tensor_a, tensor_b, tensor_c, cosine_context, tensor_index)
            if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                debug_tensors.append(_debug_tensor_entry(recipe, key, tensor_a, tensor_index, "merged"))
        if recipe.merge_type == "checkpoint" and mergeable_common_keys == 0:
            raise MergeExecutionError("no mergeable common checkpoint tensors found between source A and source B")
        if recipe.merge_type == "lora" and recipe.normalized_merge_mode() == "weight_sum" and recipe.normalized_calc_mode() == "normal":
            extra_lora_keys = sorted(set().union(*(set(tensors) for _role, _source_item, tensors in lora_merge_sources[1:])))
            for key in extra_lora_keys:
                if key in merged or not _should_merge_key(recipe, key):
                    continue
                if key.endswith(".alpha"):
                    for _role, _source_item, tensors in lora_merge_sources[1:]:
                        if key in tensors:
                            merged[key] = tensors[key]
                            break
                    continue
                merged[key] = _merge_lora_tensor_from_sources(recipe, key, lora_merge_sources, mergeable_common_keys)
                if recipe.debug and len(debug_tensors) < DEBUG_TENSOR_LIMIT:
                    debug_tensors.append(_debug_tensor_entry(recipe, key, merged[key], mergeable_common_keys, "merged_non_a_only"))
        logs.append(f"mergeable common tensors: {mergeable_common_keys}")
    if recipe.debug:
        logs.append(f"debug tensors captured: {len(debug_tensors)}")
    pruned_keys: list[str] = []
    if recipe.prune:
        merged, pruned_keys = _pruned_tensors(merged)
        logs.append(f"pruned tensors: {len(pruned_keys)}")
    clip_ids_reset = 0
    if recipe.reset_clip_ids:
        clip_ids_reset = _reset_clip_position_ids(merged)
        logs.append(f"reset clip position ids: {clip_ids_reset}")
    vae_baked_tensors = _bake_vae_tensors(recipe, merged)
    if vae_baked_tensors:
        logs.append(f"baked VAE tensors: {vae_baked_tensors}")
    output_path = _output_path(recipe, output_root, transient=transient)
    normalized_recipe = _normalized_recipe_provenance(recipe)
    metadata = {
        "asteria.merge.recipe": json.dumps(
            normalized_recipe,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "asteria.merge.type": recipe.merge_type,
        "asteria.merge.merge_mode": recipe.normalized_merge_mode(),
        "asteria.merge.calc_mode": recipe.calc_mode,
        "asteria.merge.calculation_mode": recipe.normalized_calc_mode(),
        "asteria.merge.calculation_contract_version": recipe.calculation_contract_version,
        "asteria.merge.calculation_parameter_schema_version": str(
            recipe.calculation_parameter_schema_version
        ),
        "asteria.merge.calculation_parameters": json.dumps(
            recipe.calculation_parameters,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "asteria.merge.calculation_implementation_revision": recipe.calculation_implementation_revision,
        "asteria.merge.source_fingerprints": json.dumps(
            recipe.source_fingerprints,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "asteria.merge.block_schema": recipe.block_schema,
        "asteria.merge.alpha": str(recipe.alpha),
        "asteria.merge.beta": str(recipe.beta),
        "asteria.merge.block_weight_preset": recipe.block_weight_preset,
        "asteria.merge.elemental_rules": recipe.elemental_rules,
        "asteria.merge.dice_mode": recipe.dice_mode,
        "asteria.merge.lora_metadata_mode": recipe.lora_metadata_mode,
        "asteria.merge.lora_same_to_strength": "true" if recipe.lora_same_to_strength else "false",
        "asteria.merge.transient": "true" if transient else "false",
        "asteria.merge.pruned_tensors": str(len(pruned_keys)),
        "asteria.merge.reset_clip_position_ids": str(clip_ids_reset),
        "asteria.merge.bake_in_vae": recipe.bake_in_vae,
        "asteria.merge.vae_baked_tensors": str(vae_baked_tensors),
        "asteria.merge.adjust_target_tensors": str(adjust_target_tensors),
        "asteria.merge.use_old_calc_method": "true" if recipe.use_old_calc_method else "false",
        "asteria.merge.debug": "true" if recipe.debug else "false",
        "asteria.merge.cuda_requested": "true" if recipe.use_cuda else "false",
        "asteria.merge.device": merge_device,
    }
    if recipe.merge_type == "lora_to_checkpoint":
        metadata.update(
            {
                "asteria.merge.materialized_checkpoint": "true",
                "asteria.merge.materialized_checkpoint_note": (
                    "This is a materialized checkpoint with LoRA deltas baked into tensors; "
                    "runtime output may be close to but not pixel-identical with dynamic direct LoRA inference."
                ),
                "asteria.merge.fp8_promoted_lora_tensors": str(len(fp8_promoted_lora_tensors)),
                "asteria.merge.materialized_dtype_policy": (
                    "applied_fp8_to_fp16" if fp8_promoted_lora_tensors else "preserve_checkpoint_dtype"
                ),
            }
        )
    elif recipe.merge_type == "extract_lora":
        if recipe.block_schema == "anima" and any(key.endswith(".lora_A.weight") for key in merged):
            extracted_lora_format = "anima_ab"
        elif recipe.block_schema in {"flux1", "flux2"} and any(
            key.endswith(".lora_A.weight") for key in merged
        ):
            extracted_lora_format = f"{recipe.block_schema}_ab"
        else:
            extracted_lora_format = "kohya_down_up"
        metadata["asteria.merge.extracted_lora_format"] = extracted_lora_format
    output_metadata = _output_metadata(recipe, sources, metadata) if recipe.save_metadata else None
    try:
        save_file(_cpu_tensors_for_save(merged), str(output_path), metadata=output_metadata)
    except Exception as exc:
        raise MergeExecutionError(f"merge output save failed: {output_path}") from exc
    try:
        output_content_hash = _output_content_hash(output_path, fingerprints)
    except OSError as exc:
        raise MergeExecutionError("merge output provenance hash failed") from exc
    copied_configs = _copy_source_configs(recipe, sources, output_path)
    normalized_recipe = _normalized_recipe_provenance(
        recipe,
        output_content_hash=output_content_hash,
    )
    if copied_configs:
        logs.append(f"copied configs: {len(copied_configs)}")
    metadata_path = output_path.with_suffix(".asteria.merge.json")
    metadata_path_text = str(metadata_path)
    if recipe.save_metadata:
        payload = {
            "schema": "asteria.merge.v1",
            "transient": transient,
            "recipe": normalized_recipe,
            "output_path": str(output_path),
            "output_content_hash": output_content_hash,
            "logs": [*logs, f"skipped blocks: {skipped}"],
            "pruned_keys": pruned_keys[:200],
            "clip_position_ids_reset": clip_ids_reset,
            "bake_in_vae": recipe.bake_in_vae,
            "vae_baked_tensors": vae_baked_tensors,
            "adjust_target_tensors": adjust_target_tensors,
            "cuda_requested": recipe.use_cuda,
            "merge_device": merge_device,
            "copied_configs": copied_configs,
            "debug_tensors": debug_tensors if recipe.debug else [],
            "created_at": _now(),
        }
        try:
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise MergeExecutionError(f"merge metadata save failed: {metadata_path}") from exc
    else:
        metadata_path_text = ""
        logs.append("metadata skipped")
    return MergeRun(
        id=uuid4().hex,
        recipe_id=recipe.id,
        status="completed",
        output_path=str(output_path),
        metadata_path=metadata_path_text,
        calculation_mode_id=recipe.normalized_calc_mode(),
        normalized_recipe=normalized_recipe,
        source_fingerprints=[
            dict(item) for item in recipe.source_fingerprints
        ],
        output_content_hash=dict(output_content_hash),
        logs=[*logs, f"skipped blocks: {skipped}", f"saved: {output_path}"],
        created_at=_now(),
    )
