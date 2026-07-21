from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
import csv
from io import StringIO
import json
import re
import shutil
import sqlite3
from typing import Any
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Response
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from safetensors import safe_open

from app.core.config import AppConfig
from app.api.generation import _data_dir, _generate, _output_dir_for, _record_xyz_history
from app.services.artifact_lifecycle import artifact_write_operation
from app.services.history import HistoryRepository
from app.services.comfy_admission import (
    ComfyAdmissionError,
    ComfyAdmissionTimeoutError,
    ComfyCleanupPendingError,
    acknowledge_comfy_cleanup,
    comfy_admission,
)
from app.services.merge_recipes import (
    BLOCK_SCHEMAS,
    CALCULATION_MODES,
    CALCULATION_MODE_COMPATIBILITY,
    MERGE_MODES,
    MERGE_TYPES,
    MergeAnalysis,
    MergeExecutionError,
    MergeRecipe,
    MergeRecipeRepository,
    MergeRun,
    MergeRunRepository,
    MergeSource,
    analyze_merge_recipe,
    execute_merge_recipe,
    expand_dice_recipe,
)
from app.services.merge_runtime_contract import (
    MERGE_RUNTIME_CONTRACT_VERSION,
    MergeRuntimeContractError,
    normalize_merge_runtime_recipe,
)
from app.services.merge_xyz import (
    MergeXYZPlan,
    MergeXYZSourceIdentity,
    MergeXYZValidationError,
    estimate_merge_xyz_resources,
    iter_merge_xyz_candidates,
    normalize_merge_xyz_axes,
    resolve_merge_xyz_sources,
)
from app.services.merge_presets import (
    MergeBlockPreset,
    MergePresetRepository,
    canonical_merge_schema,
    migrate_legacy_flux2_weight_text,
    migrate_legacy_flux2_weights,
    user_preset_id,
)
from app.services.model_dependencies import ModelDependencyRepository, ModelDependencyStatus
from app.services.model_storage import resolve_model_root
from app.services.metadata import MetadataWriter
from app.services.xyz_grid import XYZContactSheetStager, XYZGridAxis, XYZGridCell
from app.services.user_settings import UserSettingsRepository, resolve_merge_preset_dir
from app.schemas.generation import Txt2ImgRequest, XYZAxis, XYZCellResult, XYZPlotResponse
from app.services.runtime import ComfyClient, RuntimeErrorResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/merge", tags=["merge"])


MERGE_XYZ_AXIS_TARGETS = (
    "alpha",
    "beta",
    "alpha_beta",
    "seed",
    "checkpoint",
    "model_A",
    "model_B",
    "model_C",
    "merge_mode",
    "calculation_mode",
    "mbw_preset",
    "mbw_alpha",
    "mbw_beta",
    "pinpoint_blocks",
    "pinpoint_adjust",
    "pinpoint_element",
    "pinpoint_element_ext",
    "effective",
    "prompt",
    "elemental",
    "dice",
)
MERGE_XYZ_CANONICAL_TARGETS = {target.lower() for target in MERGE_XYZ_AXIS_TARGETS}


class MergeXYZAxis(XYZAxis):
    target: str
    values: list[str] = Field(default_factory=list)
    label: str = ""


class MergeXYZPrecheckRequest(BaseModel):
    recipe: MergeRecipe
    base: Txt2ImgRequest
    axes: list[MergeXYZAxis]
    contact_sheet: bool = True
    confirmation_fingerprint: str = ""


class MergeXYZEstimateResponse(BaseModel):
    candidate_count: int
    axis_counts: list[int]
    order: str
    final_width: int
    final_height: int
    raw_rgb_staging_bytes: int
    temporary_bytes_lower_bound: int
    temporary_bytes_upper_bound: int
    source_size_lower_bound_bytes: int
    duration_seconds: float | None
    duration_label: str
    png_representable: bool
    warning_severity: str
    requires_confirmation: bool
    confirmation_fingerprint: str


class MergeBlockPresetRequest(BaseModel):
    name: str
    family: str
    weights: list[float]
    description: str = ""


class MergeBlockPresetBundle(BaseModel):
    schema_: str = Field(default="asteria.merge.block-presets.v1", alias="schema")
    items: list[MergeBlockPreset]


class MergeBlockPresetImportRequest(MergeBlockPresetBundle):
    overwrite: bool = False


class MergeBlockPresetImportResponse(BaseModel):
    imported: int
    skipped: list[str]
    items: list[MergeBlockPreset]


class PromotePrecheckCandidateRequest(BaseModel):
    candidate_path: str
    metadata_path: str = ""
    output_name: str = ""


class PromotePrecheckCandidateResponse(BaseModel):
    output_path: str
    metadata_path: str
    promoted_from: str


class ClearMergeCandidateCacheResponse(BaseModel):
    deleted_files: int
    deleted_roots: list[str]
    failed_roots: list[str] = []


class MergeDiceChallengeRequest(BaseModel):
    recipe: MergeRecipe
    count: int = 1
    start_seed: int | None = None
    include_elemental_report: bool = True


class MergeDiceChallengeItem(BaseModel):
    index: int
    dice_seed: int
    merge_recipe_patch: dict[str, Any]
    elemental_matches: list[dict[str, Any]] = []


class MergeDiceChallengeResponse(BaseModel):
    report_path: str
    items: list[MergeDiceChallengeItem]


class MergeMetadataAudit(BaseModel):
    output_exists: bool
    metadata_exists: bool
    recipe_present: bool
    safetensors_metadata_present: bool
    safetensors_recipe_present: bool
    recipe_matches_safetensors: bool | None = None
    missing_reproducibility_fields: list[str] = []
    warnings: list[str] = []


class MergeMetadataReadResponse(BaseModel):
    metadata_path: str
    output_path: str
    schema_: str = Field(default="", alias="schema")
    transient: bool | None = None
    recipe: dict[str, Any] = {}
    logs: list[str] = []
    payload: dict[str, Any] = {}
    safetensors_metadata: dict[str, str] = {}
    audit: MergeMetadataAudit


def _repository() -> MergeRecipeRepository:
    return MergeRecipeRepository(AppConfig().data_dir / "merge-recipes.json")


def _run_repository() -> MergeRunRepository:
    data_dir = AppConfig().data_dir
    return MergeRunRepository(data_dir / "reports" / "merge-runs.json", legacy_path=data_dir / "merge-runs.json")


def _preset_repository(config: AppConfig | None = None) -> MergePresetRepository:
    config = config or AppConfig()
    settings = UserSettingsRepository(config.data_dir / "user-settings.json").load()
    preset_dir = resolve_merge_preset_dir(config.data_dir, settings.storage, config.merge_preset_dir)
    return MergePresetRepository(config.data_dir, preset_dir=preset_dir)


def _model_root(config: AppConfig) -> Path:
    return resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)


def _dependency_repository(config: AppConfig) -> ModelDependencyRepository:
    return ModelDependencyRepository(
        manifest_path=config.model_dependencies_path,
        model_root=_model_root(config),
    )


def _dependency_status_item(status: ModelDependencyStatus, model_root: Path) -> dict[str, Any]:
    try:
        relative_path = status.path.relative_to(model_root).as_posix()
    except ValueError:
        relative_path = status.path.as_posix()
    return {
        "id": status.id,
        "label": status.label,
        "type": status.type,
        "status": status.status,
        "relative_path": relative_path,
        "slot_id": status.slot_id,
        "source_url": status.source_url,
    }


def _dependency_status_items(config: AppConfig, profile: str, *, missing_only: bool = False) -> list[dict[str, Any]]:
    model_root = _model_root(config)
    return [
        _dependency_status_item(status, model_root)
        for status in _dependency_repository(config).list_status()
        if status.profile == profile and (not missing_only or status.status != "installed")
    ]


def _block_schema_label(schema_id: str) -> str:
    return {
        "sd15_sd2": "SD1/SD2",
        "sdxl": "SDXL",
        "anima": "Anima",
        "flux1": "Flux.1",
        "flux2": "Flux.2",
    }.get(schema_id, schema_id)


def _safe_model_name(value: str, fallback: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    return name or fallback


def _unique_path(path: Path) -> Path:
    try:
        exists = path.exists()
    except OSError:
        exists = True
    if not exists:
        return path
    return path.with_name(f"{path.stem}_{uuid4().hex[:8]}{path.suffix}")


def _path_exists(path: Path) -> bool:
    return _path_exists_or_none(path) is True


def _path_exists_or_none(path: Path) -> bool | None:
    try:
        return path.exists()
    except OSError:
        return None


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _resolve_or_self(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _metadata_root(path: Path) -> Path:
    return _resolve_or_self(path)


def _allowed_metadata_roots(config: AppConfig) -> list[Path]:
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    return [_metadata_root(config.data_dir), _metadata_root(config.model_dir), _metadata_root(model_root)]


def _is_under_any(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _output_path_from_metadata_path(metadata_path: Path, payload: dict[str, Any]) -> Path:
    output_path = payload.get("output_path")
    if isinstance(output_path, str) and output_path.strip():
        return Path(output_path).resolve()
    name = metadata_path.name
    if name.endswith(".asteria.merge.json"):
        return (metadata_path.parent / name.removesuffix(".asteria.merge.json")).with_suffix(".safetensors").resolve()
    return metadata_path.with_suffix(".safetensors").resolve()


REPRODUCIBLE_RECIPE_FIELDS = [
    "merge_type",
    "model_family",
    "merge_mode",
    "calc_mode",
    "block_schema",
    "sources",
    "alpha",
    "beta",
    "block_weights",
    "block_weights_alpha",
    "block_weights_beta",
    "block_weight_preset",
    "include_blocks",
    "exclude_blocks",
    "pinpoint_blocks",
    "pinpoint_elements",
    "pinpoint_adjust",
    "elemental_rules",
    "adjust_values",
    "calculation_option",
    "dice_mode",
    "dice_seed",
    "dice_ranges",
    "lora_metadata_mode",
    "lora_same_to_strength",
    "prompt_ratio_source",
    "use_cuda",
    "fp16",
    "save_metadata",
    "copy_config",
    "prune",
    "reset_clip_ids",
    "use_old_calc_method",
    "debug",
    "bake_in_vae",
    "lora_rank",
    "lora_conv_rank",
    "lora_source_ratio",
    "checkpoint_lora_strength",
    "output_name",
]


def _read_safetensors_metadata(output_path: Path) -> dict[str, str]:
    if not _path_exists(output_path) or output_path.suffix.lower() != ".safetensors":
        return {}
    try:
        with safe_open(output_path, framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception:
        return {}


def _merge_metadata_audit(payload: dict[str, Any], output_path: Path, safetensors_metadata: dict[str, str]) -> MergeMetadataAudit:
    recipe = payload.get("recipe")
    recipe_dict = recipe if isinstance(recipe, dict) else {}
    safetensors_recipe: dict[str, Any] | None = None
    safetensors_recipe_raw = safetensors_metadata.get("asteria.merge.recipe")
    if safetensors_recipe_raw:
        try:
            parsed = json.loads(safetensors_recipe_raw)
            if isinstance(parsed, dict):
                safetensors_recipe = parsed
        except json.JSONDecodeError:
            safetensors_recipe = None
    warnings: list[str] = []
    if recipe_dict and not safetensors_recipe:
        warnings.append("safetensors_recipe_metadata_missing")
    output_exists_state = _path_exists_or_none(output_path)
    output_exists = output_exists_state is True
    if output_exists_state is not False and not safetensors_metadata:
        warnings.append("safetensors_metadata_missing_or_unreadable")
    missing_fields = [field for field in REPRODUCIBLE_RECIPE_FIELDS if field not in recipe_dict]
    return MergeMetadataAudit(
        output_exists=output_exists,
        metadata_exists=True,
        recipe_present=bool(recipe_dict),
        safetensors_metadata_present=bool(safetensors_metadata),
        safetensors_recipe_present=safetensors_recipe is not None,
        recipe_matches_safetensors=(recipe_dict == safetensors_recipe) if recipe_dict and safetensors_recipe is not None else None,
        missing_reproducibility_fields=missing_fields,
        warnings=warnings,
    )


def _candidate_roots(config: AppConfig) -> dict[str, list[Path]]:
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    return {
        "checkpoints": [
            _resolve_or_self(config.model_dir / "checkpoints" / "merge-candidates" / "checkpoints"),
            _resolve_or_self(model_root / "merge-candidates" / "checkpoints"),
        ],
        "lora": [
            _resolve_or_self(model_root / "merge-candidates" / "lora"),
        ],
    }


@router.get("/metadata", response_model=MergeMetadataReadResponse)
def read_merge_metadata(path: str) -> MergeMetadataReadResponse:
    config = AppConfig()
    try:
        requested_path = Path(path).resolve()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="merge_metadata_not_found") from exc
    allowed_roots = _allowed_metadata_roots(config)
    if not _is_under_any(requested_path, allowed_roots):
        raise HTTPException(status_code=400, detail="merge_metadata_path_outside_allowed_roots")

    metadata_path = requested_path
    payload: dict[str, Any] = {}
    if requested_path.suffix.lower() == ".safetensors":
        metadata_path = requested_path.with_suffix(".asteria.merge.json")
        output_path = requested_path
    else:
        output_path = requested_path
    try:
        metadata_available = metadata_path.exists() and metadata_path.is_file()
    except OSError:
        metadata_available = False
    if metadata_available:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="merge_metadata_unreadable") from exc
    else:
        raise HTTPException(status_code=404, detail="merge_metadata_not_found")

    if requested_path.suffix.lower() != ".safetensors":
        try:
            output_path = _output_path_from_metadata_path(metadata_path, payload)
        except OSError as exc:
            raise HTTPException(status_code=400, detail="merge_metadata_unreadable") from exc
    if not _is_under_any(output_path, allowed_roots):
        raise HTTPException(status_code=400, detail="merge_output_path_outside_allowed_roots")

    safetensors_metadata = _read_safetensors_metadata(output_path)
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
    logs = payload.get("logs") if isinstance(payload.get("logs"), list) else []
    return MergeMetadataReadResponse(
        **{
            "metadata_path": str(metadata_path),
            "output_path": str(output_path),
            "schema": str(payload.get("schema", "")),
            "transient": payload.get("transient") if isinstance(payload.get("transient"), bool) else None,
            "recipe": recipe,
            "logs": [str(item) for item in logs],
            "payload": payload,
            "safetensors_metadata": safetensors_metadata,
            "audit": _merge_metadata_audit(payload, output_path, safetensors_metadata),
        }
    )


@router.get("/recipes")
def list_merge_recipes() -> dict[str, list[MergeRecipe]]:
    return {"items": _repository().list()}


@router.get("/runs")
def list_merge_runs() -> dict[str, list[MergeRun]]:
    return {"items": _run_repository().list()}


@router.get("/catalog")
def merge_catalog() -> dict:
    config = AppConfig()
    public_block_schemas = {
        key: value
        for key, value in BLOCK_SCHEMAS.items()
        if key != "flux"
    }
    flux2_missing_dependencies = _dependency_status_items(config, "flux2", missing_only=True)
    flux2_blocked = bool(flux2_missing_dependencies)
    return {
        "merge_types": [
            {"id": "checkpoint", "label": "Checkpoint merge"},
            {"id": "lora", "label": "LoRA merge"},
            {"id": "lora_to_checkpoint", "label": "LoRA into checkpoint"},
            {"id": "extract_lora", "label": "Extract LoRA"},
        ],
        "merge_modes": [{"id": key, **value} for key, value in MERGE_MODES.items()],
        "calculation_modes": [{"id": key, "label": key, "description": value} for key, value in CALCULATION_MODES.items()],
        "calculation_mode_compatibility": CALCULATION_MODE_COMPATIBILITY,
        "block_schemas": [{"id": key, "label": _block_schema_label(key), "blocks": value} for key, value in public_block_schemas.items()],
        "block_filter_help": {
            "special_tokens": ["Adjust", "VAE", "print"],
            "substring_examples": ["attn", "mlp", "conv", "norm"],
            "syntax_examples": ["IN00", "OUT00", "IN00-IN04", "NOT OUT00", "attn"],
            "elemental_alias_examples": ["TIME", "LABEL", "CLIPL", "CLIPG", "OUT9", "VAE", "IN410", "NOT attn"],
            "family_examples": {
                "sd15_sd2": ["BASE", "IN00", "MID", "OUT11"],
                "sdxl": ["BASE", "IN00", "M00", "OUT16"],
                "anima": ["BASE", "B000", "LLM000", "OUT"],
                "flux1": ["CLIP", "T5", "IN", "D000", "S000", "OUT"],
                "flux2": ["CLIP", "T5", "IN", "D000", "S000", "OUT"],
            },
        },
        "model_family_capabilities": [
            {
                "id": "sd15_sd2",
                "supports_supermerger_adjust": True,
                "adjust_note": "SuperMerger FINETUNES keys are expected on SD1/SD2-style checkpoints.",
                "verification_status": "schema_verified",
                "verification_note": "SD1/SD2 schema and preset projection are available; current real-image parity focus is SDXL and Anima.",
                "missing_dependencies": [],
            },
            {
                "id": "sdxl",
                "supports_supermerger_adjust": True,
                "adjust_note": "SuperMerger FINETUNES keys were verified on SDXL checkpoints.",
                "verification_status": "image_verified",
                "verification_note": "SDXL merge, LoRA, pre-check XYZ, promoted checkpoints, and R-ESGAN4x comparison images have current artifact evidence.",
                "missing_dependencies": [],
            },
            {
                "id": "anima",
                "supports_supermerger_adjust": False,
                "adjust_note": "Anima checkpoints do not expose the SuperMerger FINETUNES keys used by Adjust.",
                "verification_status": "image_verified",
                "verification_note": "Anima MBW candidate images and promoted checkpoint generation have current artifact evidence.",
                "missing_dependencies": [],
            },
            {
                "id": "flux1",
                "supports_supermerger_adjust": False,
                "adjust_note": "Flux.1 checkpoints do not expose the SuperMerger FINETUNES keys used by Adjust.",
                "verification_status": "schema_verified",
                "verification_note": "Flux.1 block mapping and runtime path are wired; broader two-source image comparison depends on compatible local model coverage.",
                "missing_dependencies": [],
            },
            {
                "id": "flux2",
                "supports_supermerger_adjust": False,
                "adjust_note": "Flux.2 checkpoints use the Flux-style block schema but require a separate generation workflow.",
                "verification_status": "blocked_by_local_assets" if flux2_blocked else "schema_verified",
                "verification_note": (
                    "Flux.2 fixed-seed image verification is blocked until matching local text encoder and VAE assets are placed."
                    if flux2_blocked
                    else "Flux.2 local text encoder and VAE assets are present; fixed-seed image verification is still pending."
                ),
                "missing_dependencies": flux2_missing_dependencies,
            },
        ],
        "block_presets": [
            preset.model_dump()
            for preset in _preset_repository(config).list()
            if preset.family != "flux"
        ],
        "elemental_targets": ["*", "CLIP", "T5", "BASE", "IN", "MID", "M00", "OUT", "D", "S"],
        "adjust_parameters": [
            {"id": "detail_noise_1", "label": "Detail / noise 1"},
            {"id": "detail_noise_2", "label": "Detail / noise 2"},
            {"id": "detail_noise_3", "label": "Detail / noise 3"},
            {"id": "contrast_detail", "label": "Contrast / detail"},
            {"id": "brightness", "label": "Brightness"},
            {"id": "color_tone_1", "label": "Color tone 1"},
            {"id": "color_tone_2", "label": "Color tone 2"},
            {"id": "color_tone_3", "label": "Color tone 3"},
        ],
        "dice_modes": ["R", "U", "X", "ER", "EU", "EX", "Custom"],
        "dice_range_parameters": [
            {"id": "alpha", "label": "Alpha range", "placeholder": "0.1,0.9", "help": "Bounds for global alpha Dice randomization."},
            {"id": "beta", "label": "Beta range", "placeholder": "0,0.4", "help": "Bounds for global beta Dice randomization when the merge mode uses beta."},
            {"id": "lower", "label": "Token lower", "placeholder": "0.2 or 0.2,0.2,...", "help": "Per-block lower values for X/D and EX token expansion."},
            {"id": "upper", "label": "Token upper", "placeholder": "0.6 or 0.6,0.6,...", "help": "Per-block upper values for X/D and EX token expansion."},
            {"id": "lower_delta", "label": "Clamp lower", "placeholder": "0 or 0,0,...", "help": "Per-block minimum clamp for R/U/X/D token expansion."},
            {"id": "upper_delta", "label": "Clamp upper", "placeholder": "1 or 1,1,...", "help": "Per-block maximum clamp for R/U/X/D token expansion."},
            {"id": "round", "label": "Round", "placeholder": "3", "help": "Decimal places for expanded block and Elemental random ratios."},
        ],
        "xyz_axis_targets": list(MERGE_XYZ_AXIS_TARGETS),
    }


@router.post("/block-presets", response_model=MergeBlockPreset)
def save_merge_block_preset(request: MergeBlockPresetRequest) -> MergeBlockPreset:
    family = request.family.strip()
    name = request.name.strip()
    if family not in BLOCK_SCHEMAS:
        raise HTTPException(status_code=400, detail="invalid_block_preset_family")
    if not name:
        raise HTTPException(status_code=400, detail="block_preset_name_required")
    weights = migrate_legacy_flux2_weights(request.weights) if family == "flux2" else request.weights
    if len(weights) != len(BLOCK_SCHEMAS[family]):
        raise HTTPException(status_code=400, detail="block_preset_weight_count_mismatch")
    preset = MergeBlockPreset(
        id=user_preset_id(family, name),
        name=name,
        family=family,
        weights=weights,
        source="user",
        description=request.description.strip(),
        compatible_schemas=[family],
    )
    try:
        return _preset_repository().save_user_preset(preset)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge block preset: {exc}") from exc


@router.get("/block-presets/export", response_model=MergeBlockPresetBundle)
def export_merge_block_presets(source: str = "user", family: str = "") -> MergeBlockPresetBundle:
    presets = [preset for preset in _preset_repository().list() if preset.family != "flux"]
    if source == "user":
        presets = [preset for preset in presets if preset.source == "user" or preset.id.startswith("user:")]
    elif source != "all":
        raise HTTPException(status_code=400, detail="invalid_block_preset_export_source")
    if family:
        canonical_family = canonical_merge_schema(family.strip())
        if canonical_family not in BLOCK_SCHEMAS:
            raise HTTPException(status_code=400, detail="invalid_block_preset_family")
        presets = [preset for preset in presets if canonical_merge_schema(preset.family) == canonical_family]
    presets.sort(key=lambda item: (item.family, item.name.lower(), item.id))
    return MergeBlockPresetBundle(**{"schema": "asteria.merge.block-presets.v1", "items": presets})


@router.post("/block-presets/import", response_model=MergeBlockPresetImportResponse)
def import_merge_block_presets(request: MergeBlockPresetImportRequest) -> MergeBlockPresetImportResponse:
    if request.schema_ != "asteria.merge.block-presets.v1":
        raise HTTPException(status_code=400, detail="block_preset_import_schema_unsupported")
    repository = _preset_repository()
    existing_ids = {preset.id for preset in repository.list()}
    imported: list[MergeBlockPreset] = []
    skipped: list[str] = []
    for item in request.items:
        family = canonical_merge_schema(item.family.strip())
        name = item.name.strip()
        if family not in BLOCK_SCHEMAS or family == "flux":
            raise HTTPException(status_code=400, detail="invalid_block_preset_family")
        if not name:
            raise HTTPException(status_code=400, detail="block_preset_name_required")
        weights = migrate_legacy_flux2_weights(item.weights) if family == "flux2" else item.weights
        if len(weights) != len(BLOCK_SCHEMAS[family]):
            raise HTTPException(status_code=400, detail="block_preset_weight_count_mismatch")
        preset = MergeBlockPreset(
            id=user_preset_id(family, name),
            name=name,
            family=family,
            weights=weights,
            source="user",
            description=item.description.strip(),
            derived_from=item.derived_from,
            compatible_schemas=[family],
        )
        if preset.id in existing_ids and not request.overwrite:
            skipped.append(preset.id)
            continue
        try:
            repository.save_user_preset(preset)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=f"Could not save merge block preset: {exc}") from exc
        existing_ids.add(preset.id)
        imported.append(preset)
    return MergeBlockPresetImportResponse(imported=len(imported), skipped=skipped, items=imported)


@router.delete("/block-presets/{preset_id:path}")
def delete_merge_block_preset(preset_id: str) -> dict[str, bool]:
    try:
        deleted = _preset_repository().delete_user_preset(preset_id)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge block preset: {exc}") from exc
    return {"deleted": deleted}


@router.post("/recipes", response_model=MergeRecipe)
def save_merge_recipe(recipe: MergeRecipe) -> MergeRecipe:
    if recipe.merge_type not in MERGE_TYPES:
        raise HTTPException(status_code=400, detail="invalid_merge_type")
    if recipe.normalized_merge_mode() not in MERGE_MODES:
        raise HTTPException(status_code=400, detail="invalid_merge_mode")
    if recipe.normalized_calc_mode() not in CALCULATION_MODES:
        raise HTTPException(status_code=400, detail="invalid_calc_mode")
    if recipe.block_schema not in BLOCK_SCHEMAS:
        raise HTTPException(status_code=400, detail="invalid_block_schema")
    if recipe.model_family not in BLOCK_SCHEMAS:
        raise HTTPException(status_code=400, detail="invalid_model_family")
    if canonical_merge_schema(recipe.block_schema) != canonical_merge_schema(recipe.model_family):
        raise HTTPException(status_code=400, detail="block_schema_must_match_model_family")
    if not recipe.safetensors:
        raise HTTPException(status_code=400, detail="safetensors_output_required")
    compatible_calc_modes = CALCULATION_MODE_COMPATIBILITY.get(recipe.normalized_merge_mode(), [])
    if compatible_calc_modes and recipe.normalized_calc_mode() not in compatible_calc_modes:
        raise HTTPException(status_code=400, detail="calc_mode_not_allowed_for_merge_mode")
    required_sources = 3 if MERGE_MODES[recipe.normalized_merge_mode()]["requires_c"] else 2
    if len([source for source in recipe.sources if source.path.strip()]) < required_sources:
        if required_sources == 3:
            raise HTTPException(status_code=400, detail="merge_recipe_requires_source_c")
        raise HTTPException(status_code=400, detail="merge_recipe_requires_at_least_two_sources")
    try:
        return _repository().save(recipe)
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge recipe: {exc}") from exc


@router.delete("/recipes/{recipe_id}")
def delete_merge_recipe(recipe_id: str) -> dict[str, bool]:
    try:
        deleted = _repository().delete(recipe_id)
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge recipe: {exc}") from exc
    return {"deleted": deleted}


@router.post("/dice/challenges", response_model=MergeDiceChallengeResponse)
def build_dice_challenges(request: MergeDiceChallengeRequest) -> MergeDiceChallengeResponse:
    count = max(1, min(request.count, 64))
    start_seed = request.start_seed if request.start_seed is not None else request.recipe.dice_seed
    if start_seed is None:
        start_seed = 1
    items: list[MergeDiceChallengeItem] = []
    for index in range(count):
        challenge_recipe = request.recipe.model_copy(deep=True)
        challenge_recipe.dice_seed = int(start_seed) + index
        expanded_recipe = expand_dice_recipe(challenge_recipe)
        elemental_matches: list[dict[str, Any]] = []
        if request.include_elemental_report:
            try:
                elemental_matches = [match.model_dump() for match in analyze_merge_recipe(expanded_recipe).elemental_matches]
            except MergeExecutionError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        items.append(
            MergeDiceChallengeItem(
                index=index,
                dice_seed=challenge_recipe.dice_seed,
                merge_recipe_patch=_candidate_recipe_patch(expanded_recipe),
                elemental_matches=elemental_matches,
            )
        )
    report_dir = AppConfig().data_dir / "reports" / "merge-dice-challenges"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = _unique_path(report_dir / f"{_safe_model_name(request.recipe.name, 'dice-challenge')}.json")
        report_path.write_text(
            json.dumps(
                {
                    "schema": "asteria.merge-dice-challenge.v1",
                    "created_at": datetime.now(UTC).isoformat(),
                    "recipe": request.recipe.model_dump(),
                    "count": count,
                    "start_seed": int(start_seed),
                    "include_elemental_report": request.include_elemental_report,
                    "items": [item.model_dump() for item in items],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge dice challenge report: {exc}") from exc
    return MergeDiceChallengeResponse(report_path=str(report_path), items=items)


@router.post("/analyze")
def analyze_merge(recipe: MergeRecipe) -> dict:
    try:
        return analyze_merge_recipe(recipe).model_dump()
    except MergeExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze/export")
def export_merge_analysis(recipe: MergeRecipe, format: str = "txt") -> Response:
    try:
        analysis = analyze_merge_recipe(recipe)
    except MergeExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    normalized_format = format.strip().lower()
    if normalized_format == "txt":
        return Response(content=_analysis_to_text(recipe, analysis), media_type="text/plain; charset=utf-8")
    if normalized_format == "csv":
        return Response(content=_analysis_to_csv(analysis), media_type="text/csv; charset=utf-8")
    raise HTTPException(status_code=400, detail="analysis export format must be txt or csv")


def _analysis_to_text(recipe: MergeRecipe, analysis: MergeAnalysis) -> str:
    lines = [
        f"Merge Analysis: {recipe.name}",
        f"merge_type: {recipe.merge_type}",
        f"model_family: {recipe.model_family}",
        f"block_schema: {recipe.block_schema}",
        f"common: {analysis.common_keys}",
        f"mergeable: {analysis.mergeable_keys}",
        f"skipped: {analysis.skipped_keys}",
        "",
        "Sources",
    ]
    for source in analysis.sources:
        lines.append(f"- {source.role}: {source.path} tensors={source.tensor_count}")
    lines.extend(["", "Blocks"])
    for block, count in analysis.block_counts.items():
        lines.append(f"- {block}: {count}")
    lines.extend(["", "Elemental matches"])
    if analysis.elemental_matches:
        for match in analysis.elemental_matches:
            lines.append(f"- {match.rule}: {len(match.matched_keys)}")
            for key in match.matched_keys[:10]:
                lines.append(f"  - {key}")
    else:
        lines.append("- none")
    lines.extend(["", "Shape mismatches"])
    if analysis.shape_mismatches:
        for item in analysis.shape_mismatches:
            lines.append(f"- {item.key}: {item.a_shape} vs {item.b_shape}")
    else:
        lines.append("- none")
    lines.extend(["", "Tensor differences"])
    for item in analysis.tensor_differences:
        lines.append(f"- {item.block}: max={item.max_abs} mean={item.mean_abs} key={item.key}")
    lines.extend(["", "Block similarities"])
    for item in analysis.block_similarities:
        lines.append(f"- {item.block}: mean={item.mean_percent} min={item.min_percent} attn2={item.attn2_percent}")
    lines.extend(["", "Attention similarities"])
    for item in analysis.attention_similarities:
        lines.append(f"- {item.block}: {item.similarity_percent} key={item.key}")
    return "\n".join(lines) + "\n"


def _analysis_to_csv(analysis: MergeAnalysis) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["section", "block", "key", "metric", "value"])
    for block, count in analysis.block_counts.items():
        writer.writerow(["block_count", block, "", "count", count])
    for match in analysis.elemental_matches:
        writer.writerow(["elemental", "", match.rule, "matched_count", len(match.matched_keys)])
        for key in match.matched_keys:
            writer.writerow(["elemental_key", "", key, "matched", 1])
    for item in analysis.shape_mismatches:
        writer.writerow(["shape_mismatch", "", item.key, "a_shape", "x".join(map(str, item.a_shape))])
        writer.writerow(["shape_mismatch", "", item.key, "b_shape", "x".join(map(str, item.b_shape))])
    for item in analysis.tensor_differences:
        writer.writerow(["tensor_diff", item.block, item.key, "mean_abs", item.mean_abs])
        writer.writerow(["tensor_diff", item.block, item.key, "max_abs", item.max_abs])
    for item in analysis.tensor_similarities:
        writer.writerow(["tensor_similarity", item.block, item.key, "similarity_percent", item.similarity_percent])
    for item in analysis.block_similarities:
        writer.writerow(["block_similarity", item.block, "", "mean_percent", item.mean_percent])
        writer.writerow(["block_similarity", item.block, "", "min_percent", item.min_percent])
        if item.attn2_percent is not None:
            writer.writerow(["block_similarity", item.block, "", "attn2_percent", item.attn2_percent])
    for item in analysis.attention_similarities:
        writer.writerow(["attention_similarity", item.block, item.key, "similarity_percent", item.similarity_percent])
    return output.getvalue()


@router.post("/recipes/{recipe_id}/run", response_model=MergeRun)
def run_merge_recipe(recipe_id: str) -> MergeRun:
    recipe = _repository().get(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail="merge_recipe_not_found")
    config = AppConfig()
    try:
        run = execute_merge_recipe(
            recipe,
            resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir),
            transient=not recipe.save_model,
        )
    except MergeExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if recipe.save_model:
        recipe.output_path = run.output_path
    try:
        _repository().save(recipe)
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge recipe: {exc}") from exc
    try:
        _run_repository().save(run)
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not save merge run history: {exc}") from exc
    return run


@router.post("/precheck-candidates/promote", response_model=PromotePrecheckCandidateResponse)
def promote_precheck_candidate(request: PromotePrecheckCandidateRequest) -> PromotePrecheckCandidateResponse:
    config = AppConfig()
    try:
        candidate = Path(request.candidate_path).resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="candidate_not_found") from exc
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    candidate_roots = _candidate_roots(config)
    output_folder = next((folder for folder, roots in candidate_roots.items() if any(root in candidate.parents for root in roots)), "")
    if not output_folder:
        raise HTTPException(status_code=400, detail="candidate_path_outside_precheck_cache")
    try:
        candidate_available = candidate.exists() and candidate.is_file()
    except OSError:
        candidate_available = False
    if not candidate_available or candidate.suffix.lower() != ".safetensors":
        raise HTTPException(status_code=400, detail="candidate_not_found")

    try:
        output_root = model_root / "merged" / output_folder
        output_root.mkdir(parents=True, exist_ok=True)
        output_name = _safe_model_name(request.output_name, candidate.name)
        if not output_name.lower().endswith(".safetensors"):
            output_name = f"{output_name}.safetensors"
        output_path = _unique_path(output_root / output_name)
        shutil.copy2(candidate, output_path)

        metadata_source = Path(request.metadata_path).resolve() if request.metadata_path else candidate.with_suffix(".asteria.merge.json").resolve()
        metadata_path = output_path.with_suffix(".asteria.merge.json")
        metadata: dict = {}
        try:
            source_available = metadata_source.exists() and metadata_source.is_file()
        except OSError:
            source_available = False
        if source_available:
            try:
                metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        metadata.update(
            {
                "transient": False,
                "promoted_from": str(candidate),
                "output_path": str(output_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=409, detail=f"Could not promote merge candidate: {exc}") from exc
    return PromotePrecheckCandidateResponse(output_path=str(output_path), metadata_path=str(metadata_path), promoted_from=str(candidate))


@router.delete("/precheck-candidates", response_model=ClearMergeCandidateCacheResponse)
def clear_merge_candidate_cache() -> ClearMergeCandidateCacheResponse:
    deleted_files = 0
    deleted_roots: list[str] = []
    failed_roots: list[str] = []
    seen: set[Path] = set()
    for roots in _candidate_roots(AppConfig()).values():
        for root in roots:
            if root in seen:
                continue
            try:
                root_exists = root.exists()
            except OSError:
                failed_roots.append(str(root))
                seen.add(root)
                continue
            if not root_exists:
                continue
            seen.add(root)
            try:
                file_count = sum(1 for path in root.rglob("*") if path.is_file())
            except OSError:
                failed_roots.append(str(root))
                continue
            try:
                shutil.rmtree(root)
            except OSError:
                failed_roots.append(str(root))
                continue
            deleted_files += file_count
            deleted_roots.append(str(root))
    return ClearMergeCandidateCacheResponse(deleted_files=deleted_files, deleted_roots=deleted_roots, failed_roots=failed_roots)


def _preset_weights(preset_id: str) -> str:
    for preset in _preset_repository().list():
        if preset.id == preset_id:
            return ",".join(str(round(value, 4)).rstrip("0").rstrip(".") if isinstance(value, float) else str(value) for value in preset.weights)
    return ""


def _preset_label(preset_id: str) -> str:
    for preset in _preset_repository().list():
        if preset.id == preset_id:
            return preset.name
    return preset_id


def _set_recipe_source(recipe: MergeRecipe, role: str, path: str) -> None:
    role = role.upper()
    for source in recipe.sources:
        if source.role.upper() == role:
            source.path = path
            return
    recipe.sources.append(MergeSource(role=role, path=path))


def _apply_alpha_beta(recipe: MergeRecipe, value: str) -> None:
    parts = [part for part in re.split(r"[,/:\s]+", value.strip()) if part]
    if not parts:
        raise HTTPException(status_code=400, detail="alpha_beta axis requires at least one numeric value")
    recipe.alpha = float(parts[0])
    if len(parts) > 1:
        recipe.beta = float(parts[1])


def _format_weight(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _weights_for_pinpoint(value: str, schema: list[str], fallback: float) -> list[float]:
    parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    weights: list[float] = []
    for part in parts:
        try:
            weights.append(float(part))
        except ValueError:
            continue
    if not weights:
        return [fallback] * len(schema)
    if len(weights) < len(schema):
        weights.extend([fallback] * (len(schema) - len(weights)))
    return weights[: len(schema)]


def _pinpoint_block_ids(tokens: list[str], schema: list[str]) -> set[str]:
    raw_tokens = [part for token in tokens for part in re.split(r"[\s,]+", token.strip()) if part]
    if not raw_tokens:
        return set()
    invert = raw_tokens[0].upper() == "NOT"
    if invert:
        raw_tokens = raw_tokens[1:]
    selected: set[str] = set()
    for token in raw_tokens:
        token = token.upper()
        if "-" in token:
            start, end = [part.strip().upper() for part in token.split("-", 1)]
            if start in schema and end in schema:
                start_index = schema.index(start)
                end_index = schema.index(end)
                low, high = sorted((start_index, end_index))
                selected.update(schema[low : high + 1])
            continue
        if token in schema:
            selected.add(token)
    return set(schema).difference(selected) if invert else selected


def _apply_pinpoint_blocks(recipe: MergeRecipe, base_recipe: MergeRecipe, target: str) -> None:
    schema = BLOCK_SCHEMAS.get(recipe.block_schema, BLOCK_SCHEMAS["sdxl"])
    selected = _pinpoint_block_ids(recipe.pinpoint_blocks, schema)
    if not selected:
        return
    if target == "alpha":
        source = base_recipe.block_weights_alpha or base_recipe.block_weights
        fallback = base_recipe.alpha
        replacement = recipe.alpha
        field = "block_weights_alpha"
    elif target == "beta":
        source = base_recipe.block_weights_beta
        fallback = base_recipe.beta
        replacement = recipe.beta
        field = "block_weights_beta"
    else:
        return
    weights = _weights_for_pinpoint(source, schema, fallback)
    for block_id in selected:
        weights[schema.index(block_id)] = replacement
    setattr(recipe, field, ",".join(_format_weight(weight) for weight in weights))


PINPOINT_ADJUST_TARGETS = {
    "IN": "detail_noise_1",
    "OUT": "detail_noise_2",
    "OUT2": "detail_noise_3",
    "CONT": "contrast_detail",
    "BRI": "brightness",
    "COL1": "color_tone_1",
    "COL2": "color_tone_2",
    "COL3": "color_tone_3",
}


def _apply_pinpoint_axes(recipe: MergeRecipe, base_recipe: MergeRecipe, axes: list[XYZAxis]) -> None:
    targets = {axis.target.strip().lower() for axis in axes}
    if "pinpoint_blocks" in targets:
        if "alpha" in targets:
            _apply_pinpoint_blocks(recipe, base_recipe, "alpha")
        if "beta" in targets:
            _apply_pinpoint_blocks(recipe, base_recipe, "beta")
    if recipe.pinpoint_elements and "alpha" in targets:
        additions = [f"{element}:{_format_weight(recipe.alpha)}" for element in recipe.pinpoint_elements if element]
        if additions:
            existing = base_recipe.elemental_rules.strip()
            recipe.elemental_rules = f"{existing},{','.join(additions)}" if existing else ",".join(additions)
        recipe.alpha = base_recipe.alpha
    if recipe.pinpoint_adjust and "alpha" in targets:
        adjust_key = PINPOINT_ADJUST_TARGETS.get(recipe.pinpoint_adjust.strip().upper())
        if adjust_key:
            values = dict(base_recipe.adjust_values)
            values[adjust_key] = recipe.alpha
            recipe.adjust_values = values
            recipe.alpha = base_recipe.alpha


def _apply_precheck_axis(recipe: MergeRecipe, target: str, value: str, data_dir) -> None:
    target = target.strip().lower()
    if target == "alpha":
        recipe.alpha = float(value)
    elif target == "beta":
        recipe.beta = float(value)
    elif target == "alpha_beta":
        _apply_alpha_beta(recipe, value)
    elif target in {"model_a", "model_b", "model_c"}:
        _set_recipe_source(recipe, target.rsplit("_", 1)[1], value)
    elif target == "merge_mode":
        recipe.merge_mode = value
    elif target == "calculation_mode":
        recipe.calc_mode = value
    elif target == "mbw_preset":
        recipe.block_weight_preset = value
        weights = _preset_weights(value)
        if weights:
            recipe.block_weights_alpha = weights
    elif target == "mbw_alpha":
        recipe.block_weights_alpha = (
            migrate_legacy_flux2_weight_text(value) if recipe.block_schema == "flux2" else value
        )
    elif target == "mbw_beta":
        recipe.block_weights_beta = (
            migrate_legacy_flux2_weight_text(value) if recipe.block_schema == "flux2" else value
        )
    elif target == "pinpoint_blocks":
        recipe.pinpoint_blocks = [item.strip() for item in re.split(r"[\s,]+", value) if item.strip()]
    elif target == "pinpoint_adjust":
        recipe.pinpoint_adjust = value.strip().upper()
    elif target in {"pinpoint_element", "pinpoint_element_ext", "effective"}:
        recipe.pinpoint_elements = [value.strip()] if value.strip() else []
    elif target == "elemental":
        recipe.elemental_rules = value
    elif target == "dice":
        recipe.dice_mode = value


def _apply_precheck_generation_axis(request: Txt2ImgRequest, target: str, value: str) -> None:
    target = target.strip().lower()
    if target == "seed":
        request.seed = int(value)
    elif target == "prompt":
        request.prompt = value
    elif target in {"checkpoint", "model", "model_slot:checkpoint"}:
        request.model_slot_overrides["checkpoint"] = value
    elif target in {"vae", "model_slot:vae"}:
        request.vae = value
    elif target == "sampler":
        request.sampler = value
    elif target == "scheduler":
        request.scheduler = value
    elif target == "steps":
        request.steps = int(value)
    elif target in {"cfg", "cfg_scale"}:
        request.cfg_scale = float(value)
    elif target == "hires_upscaler":
        request.hires_upscaler = value
    elif target in {"hires_denoise", "hires_denoising_strength"}:
        request.hires_denoising_strength = float(value)


def _validate_merge_xyz_precheck_axes(request: MergeXYZPrecheckRequest) -> None:
    uses_beta = bool(MERGE_MODES.get(request.recipe.normalized_merge_mode(), {}).get("uses_beta", False))
    try:
        plan = normalize_merge_xyz_axes(
            request.axes,
            allowed_targets=MERGE_XYZ_CANONICAL_TARGETS,
            uses_beta=uses_beta,
            merge_type=request.recipe.merge_type,
            model_family=request.recipe.model_family,
        )
    except MergeXYZValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc
    for axis, normalized in zip(request.axes, plan.axes, strict=True):
        axis.target = normalized.target
        axis.label = normalized.label
        axis.values = list(normalized.values)



def _merge_xyz_effective_dimensions(request: MergeXYZPrecheckRequest) -> tuple[int, int]:
    width = request.base.width
    height = request.base.height
    if not request.base.hires_enabled:
        return width, height
    scale = max(1.0, float(request.base.hires_scale))
    return (
        request.base.hires_resize_width or max(1, round(width * scale)),
        request.base.hires_resize_height or max(1, round(height * scale)),
    )


def _merge_xyz_confirmation_fingerprint(
    request: MergeXYZPrecheckRequest,
    source_identities: tuple[MergeXYZSourceIdentity, ...],
) -> str:
    payload = request.model_dump(mode="json", exclude={"confirmation_fingerprint"})
    payload["source_identities"] = [source.metadata() for source in source_identities]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _merge_xyz_estimate_response(
    request: MergeXYZPrecheckRequest,
    plan: MergeXYZPlan,
    source_identities: tuple[MergeXYZSourceIdentity, ...],
) -> MergeXYZEstimateResponse:
    width, height = _merge_xyz_effective_dimensions(request)
    estimate = estimate_merge_xyz_resources(
        plan,
        tile_width=width,
        tile_height=height,
        source_sizes=[source.size_bytes for source in source_identities],
    )
    raw_staging = estimate.raw_rgb_staging_bytes if request.contact_sheet else 0
    temporary_lower = estimate.temporary_bytes_lower_bound if request.contact_sheet else 0
    temporary_upper = estimate.temporary_bytes_upper_bound if request.contact_sheet else 0
    png_representable = estimate.png_representable or not request.contact_sheet
    requires_confirmation = (
        plan.total_count > 64
        or temporary_upper >= 4 * 1024 * 1024 * 1024
        or estimate.final_width > 32768
        or estimate.final_height > 32768
    )
    warning_severity = "blocked" if not png_representable else ("warning" if requires_confirmation else "info")
    return MergeXYZEstimateResponse(
        candidate_count=estimate.candidate_count,
        axis_counts=[len(axis.values) for axis in plan.axes],
        order="x-fastest",
        final_width=estimate.final_width,
        final_height=estimate.final_height,
        raw_rgb_staging_bytes=raw_staging,
        temporary_bytes_lower_bound=temporary_lower,
        temporary_bytes_upper_bound=temporary_upper,
        source_size_lower_bound_bytes=estimate.source_size_lower_bound_bytes,
        duration_seconds=estimate.duration_seconds,
        duration_label=estimate.duration_label,
        png_representable=png_representable,
        warning_severity=warning_severity,
        requires_confirmation=requires_confirmation,
        confirmation_fingerprint=_merge_xyz_confirmation_fingerprint(request, source_identities),
    )


def _candidate_axis(axis: XYZAxis | None, value: str | None, order: str) -> dict[str, str]:
    if axis is None or value is None:
        return {"order": order, "target": "", "label": "", "value": "", "display": ""}
    target = axis.target.strip()
    display = _preset_label(value) if target.lower() == "mbw_preset" else value
    return {
        "order": order,
        "target": target,
        "label": axis.label.strip() or target,
        "value": value,
        "display": display,
    }


def _candidate_recipe_patch(recipe: MergeRecipe) -> dict[str, Any]:
    return {
        "merge_type": recipe.merge_type,
        "merge_mode": recipe.merge_mode,
        "calc_mode": recipe.calc_mode,
        "alpha": recipe.alpha,
        "beta": recipe.beta,
        "block_weight_preset": recipe.block_weight_preset,
        "block_weights_alpha": recipe.block_weights_alpha,
        "block_weights_beta": recipe.block_weights_beta,
        "include_blocks": recipe.include_blocks,
        "exclude_blocks": recipe.exclude_blocks,
        "pinpoint_blocks": recipe.pinpoint_blocks,
        "pinpoint_elements": recipe.pinpoint_elements,
        "elemental_rules": recipe.elemental_rules,
        "calculation_option": recipe.calculation_option,
        "dice_mode": recipe.dice_mode,
        "dice_seed": recipe.dice_seed,
        "dice_ranges": recipe.dice_ranges,
        "adjust_values": recipe.adjust_values,
    }


def _candidate_source_models(recipe: MergeRecipe) -> dict[str, str]:
    return {source.role.upper(): source.path for source in recipe.sources if source.path}


def _candidate_overlay_label(candidate_id: str, axes: dict[str, dict[str, str]]) -> str:
    lines = [candidate_id]
    x_axis = axes.get("x") or {}
    y_axis = axes.get("y") or {}
    z_axis = axes.get("z") or {}
    if x_axis.get("target"):
        lines.append(f"X {x_axis.get('display') or x_axis.get('value')}")
    if y_axis.get("target"):
        lines.append(f"Y {y_axis.get('display') or y_axis.get('value')}")
    if z_axis.get("target"):
        lines.append(f"Z {z_axis.get('display') or z_axis.get('value')}")
    return "\n".join(lines)


def _candidate_image_paths(generation_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in generation_paths:
        path = Path(value)
        if (
            _path_exists(path)
            and _path_is_file(path)
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and ".effective-diff" not in path.stem
        ):
            paths.append(path)
    return paths


def _uses_effective_axis(axes: list[XYZAxis]) -> bool:
    return any(axis.target.strip().lower() == "effective" for axis in axes)


def _effective_diff_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.effective-diff.png")


def _effective_diff_record(base_image_path: Path, image_path: Path) -> dict[str, Any]:
    with Image.open(base_image_path) as base_image, Image.open(image_path) as candidate_image:
        base = base_image.convert("RGB")
        candidate = candidate_image.convert("RGB")
        if candidate.size != base.size:
            candidate = candidate.resize(base.size)
        base_array = np.asarray(base, dtype=np.int16)
        candidate_array = np.asarray(candidate, dtype=np.int16)
        abs_diff = np.abs(candidate_array - base_array).astype(np.uint8)
        thresholded = (abs_diff > 20).astype(np.uint8)
        diff_percent = float(np.count_nonzero(thresholded) * 100 / max(abs_diff.size, 1))
        diff_image = Image.fromarray(np.bitwise_not(abs_diff), mode="RGB")
    diff_path = _effective_diff_path(image_path)
    diff_image.save(diff_path)
    return {
        "diff_percent": round(diff_percent, 6),
        "diff_image_path": str(diff_path),
        "threshold": 20,
    }


def _attach_effective_checks(cells: list[XYZCellResult]) -> None:
    if len(cells) < 1:
        return
    first_images = _candidate_image_paths(cells[0].generation.output_paths)
    if not first_images:
        return
    baseline_path = first_images[0]
    baseline_candidate = cells[0].candidate
    baseline_candidate["effective_check"] = {
        "baseline": True,
        "baseline_candidate_id": baseline_candidate.get("candidate_id", "C-001"),
        "baseline_image_path": str(baseline_path),
        "diff_percent": 0,
        "diff_image_path": "",
        "threshold": 20,
    }
    for cell in cells[1:]:
        candidate_images = _candidate_image_paths(cell.generation.output_paths)
        candidate = cell.candidate
        if not candidate_images:
            candidate["effective_check"] = {
                "baseline": False,
                "baseline_candidate_id": baseline_candidate.get("candidate_id", "C-001"),
                "baseline_image_path": str(baseline_path),
                "diff_percent": None,
                "diff_image_path": "",
                "threshold": 20,
                "error": "candidate_image_not_found",
            }
            continue
        diff_record = _effective_diff_record(baseline_path, candidate_images[0])
        candidate["effective_check"] = {
            "baseline": False,
            "baseline_candidate_id": baseline_candidate.get("candidate_id", "C-001"),
            "baseline_image_path": str(baseline_path),
            **diff_record,
        }
        cell.generation.output_paths = [*cell.generation.output_paths, diff_record["diff_image_path"]]


def _font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/meiryo.ttc"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_candidate_overlay(image_path: Path, label: str) -> None:
    if not label.strip():
        return
    try:
        with Image.open(image_path) as original:
            image = original.convert("RGBA")
            pnginfo = PngInfo()
            if image_path.suffix.lower() == ".png":
                for key, value in original.info.items():
                    if isinstance(value, str):
                        pnginfo.add_text(key, value)
    except OSError:
        return

    draw = ImageDraw.Draw(image)
    font = _font(max(16, image.width // 32))
    lines = label.splitlines()
    spacing = max(4, image.width // 180)
    padding = max(10, image.width // 80)
    margin = max(10, image.width // 90)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    width = max((box[2] - box[0] for box in boxes), default=0) + padding * 2
    height = sum((box[3] - box[1] for box in boxes)) + spacing * max(0, len(lines) - 1) + padding * 2
    x = margin
    y = margin
    draw.rounded_rectangle((x, y, x + width, y + height), radius=max(6, padding // 2), fill=(3, 8, 12, 196), outline=(37, 197, 189, 220), width=2)
    text_y = y + padding
    for line, box in zip(lines, boxes, strict=True):
        draw.text((x + padding, text_y), line, font=font, fill=(238, 252, 250, 255))
        text_y += (box[3] - box[1]) + spacing
    try:
        if image_path.suffix.lower() == ".png":
            image.convert("RGB").save(image_path, pnginfo=pnginfo)
        else:
            image.convert("RGB").save(image_path)
    except OSError:
        return


def _write_candidate_sidecars(generation: XYZCellResult, candidate: dict[str, Any]) -> list[str]:
    sidecars: list[str] = []
    for image_path in _candidate_image_paths(generation.generation.output_paths):
        payload = {
            "schema": "asteria.merge-candidate.v1",
            "written_at": datetime.now(UTC).isoformat(),
            **candidate,
            "image_path": str(image_path),
        }
        sidecar_path = image_path.with_suffix(".asteria.merge-candidate.json")
        sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sidecars.append(str(sidecar_path))
    checkpoint_path_text = str(candidate.get("candidate_checkpoint_path") or "").strip()
    checkpoint_path = Path(checkpoint_path_text) if checkpoint_path_text else None
    if checkpoint_path is not None and _path_exists(checkpoint_path) and _path_is_file(checkpoint_path):
        payload = {
            "schema": "asteria.merge-candidate.v1",
            "written_at": datetime.now(UTC).isoformat(),
            **candidate,
            "image_path": candidate["image_paths"][0] if candidate.get("image_paths") else "",
        }
        sidecar_path = checkpoint_path.with_suffix(".asteria.merge-candidate.json")
        sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        sidecars.append(str(sidecar_path))
    if sidecars:
        generation.generation.output_paths = [*generation.generation.output_paths, *sidecars]
        _update_output_paths_safely(
            HistoryRepository(AppConfig().data_dir / "asteria.db"),
            generation.generation.id,
            generation.generation.output_paths,
        )
    return sidecars


def _update_output_paths_safely(repository: HistoryRepository, generation_id: int, output_paths: list[str]) -> None:
    try:
        repository.update_output_paths(generation_id, output_paths)
    except (OSError, sqlite3.Error, ValueError):
        return


def _update_settings_safely(repository: HistoryRepository, generation_id: int, settings: dict[str, Any]) -> None:
    try:
        repository.update_settings(generation_id, settings)
    except (OSError, sqlite3.Error, ValueError):
        return


def _rewrite_merge_xyz_generation_metadata(cell: XYZCellResult) -> None:
    generation = cell.generation
    sidecars = MetadataWriter().write_generation_sidecars(
        {
            "generation_id": generation.id,
            "status": generation.status,
            "profile": generation.profile,
            "prompt": generation.prompt,
            "expanded_prompt": generation.expanded_prompt,
            "negative_prompt": generation.negative_prompt,
            "settings": generation.settings,
            "model_slots": generation.model_slots,
            "loras": [lora.model_dump() for lora in generation.loras],
            "embeddings": generation.embeddings,
        },
        [str(path) for path in _candidate_image_paths(generation.output_paths)],
    )
    known = set(generation.output_paths)
    generation.output_paths.extend(path for path in sidecars if path not in known)


def _build_candidate_metadata(
    *,
    operation_id: str,
    index: int,
    axes: list[XYZAxis],
    values: tuple[str, ...],
    labels: dict[str, str],
    candidate_recipe: MergeRecipe,
    normalized_recipe: dict[str, Any],
    source_identities: tuple[MergeXYZSourceIdentity, ...],
    generation_snapshot: dict[str, Any],
    generation: XYZCellResult,
) -> dict[str, Any]:
    candidate_id = f"C-{index + 1:03d}"
    x_axis = _candidate_axis(axes[0] if len(axes) > 0 else None, str(values[0]) if len(values) > 0 else None, "x")
    y_axis = _candidate_axis(axes[1] if len(axes) > 1 else None, str(values[1]) if len(values) > 1 else None, "y")
    z_axis = _candidate_axis(axes[2] if len(axes) > 2 else None, str(values[2]) if len(values) > 2 else None, "z")
    axes_payload = {"x": x_axis, "y": y_axis, "z": z_axis}
    overlay_label = _candidate_overlay_label(candidate_id, axes_payload)
    image_paths = [str(path) for path in _candidate_image_paths(generation.generation.output_paths)]
    return {
        "artifact_type": "merge_xyz_candidate",
        "operation_id": operation_id,
        "candidate_id": candidate_id,
        "candidate_index": index + 1,
        "labels": labels,
        "overlay_label": overlay_label,
        "axes": axes_payload,
        "axis_values": [
            {
                "target": axis.target,
                "label": axis.label.strip() or axis.target,
                "value": str(value),
                "display": _preset_label(str(value)) if axis.target.strip().lower() == "mbw_preset" else str(value),
            }
            for axis, value in zip(axes, values, strict=True)
        ],
        "merge_recipe_patch": _candidate_recipe_patch(candidate_recipe),
        "normalized_recipe": normalized_recipe,
        "resolved_mbw": {
            "block_schema": normalized_recipe.get("block_schema"),
            "block_weights": normalized_recipe.get("block_weights"),
            "block_weights_alpha": normalized_recipe.get("block_weights_alpha"),
            "block_weights_beta": normalized_recipe.get("block_weights_beta"),
        },
        "calculation": {
            "mode": normalized_recipe.get("calculation_mode"),
            "option": normalized_recipe.get("calculation_option"),
            "merge_mode": normalized_recipe.get("merge_mode"),
        },
        "source_models": _candidate_source_models(candidate_recipe),
        "source_identities": [source.metadata() for source in source_identities],
        "generation_snapshot": generation_snapshot,
        "seed": generation.generation.settings.get("seed", generation_snapshot.get("seed")),
        "runtime_adapter": generation.generation.settings.get("runtime_adapter", "comfy_workflow"),
        "workflow_mapping_id": f"{generation.generation.profile}_txt2img",
        "candidate_state": "completed",
        "operation_state": "partial",
        "generation_id": generation.generation.id,
        "image_paths": image_paths,
        "rating": 0,
        "favorite": False,
    }


def _merge_runtime_source_payload(
    model_family: str,
    sources: tuple[MergeXYZSourceIdentity, ...],
) -> tuple[str, dict[str, str], dict[str, str] | None]:
    by_role = {source.role: source for source in sources}
    slot = "checkpoint" if model_family == "sdxl" else "diffusion_model"
    source_b = {slot: by_role["B"].path}
    source_c = {slot: by_role["C"].path} if "C" in by_role else None
    return by_role["A"].path, source_b, source_c


@contextmanager
def _merge_xyz_runtime_session(config: AppConfig):
    try:
        lease = comfy_admission.acquire("xyz", timeout_seconds=30.0)
    except ComfyCleanupPendingError as exc:
        raise HTTPException(status_code=409, detail="comfy_cleanup_pending") from exc
    except (ComfyAdmissionTimeoutError, ComfyAdmissionError) as exc:
        raise HTTPException(status_code=409, detail="comfy_admission_busy") from exc
    client = ComfyClient(config.comfy_base_url)
    try:
        yield lease.token
    finally:
        try:
            acknowledge_comfy_cleanup(
                service=comfy_admission,
                lease=lease,
                client=client,
                fence_workflow={
                    "1": {
                        "class_type": "AsteriaCleanupFence",
                        "inputs": {"contract_version": MERGE_RUNTIME_CONTRACT_VERSION},
                    }
                },
                timeout_seconds=60.0,
            )
        except (RuntimeErrorResponse, TimeoutError, OSError) as exc:
            raise HTTPException(status_code=503, detail="comfy_cleanup_pending") from exc
        finally:
            lease.release(defer_cleanup=comfy_admission.cleanup_pending_for(lease.token))


@router.post("/xyz-precheck", response_model=XYZPlotResponse)
@artifact_write_operation
def run_merge_xyz_precheck(request: MergeXYZPrecheckRequest) -> XYZPlotResponse:
    _validate_merge_xyz_precheck_axes(request)
    config = AppConfig()
    data_dir = config.data_dir
    if request.base.profile != request.recipe.model_family:
        raise HTTPException(status_code=400, detail="xyz_precheck_profile_family_mismatch")
    try:
        initial_contract = normalize_merge_runtime_recipe(request.recipe)
        plan = normalize_merge_xyz_axes(
            request.axes,
            allowed_targets=MERGE_XYZ_CANONICAL_TARGETS,
            uses_beta=bool(MERGE_MODES.get(request.recipe.normalized_merge_mode(), {}).get("uses_beta", False)),
            merge_type=request.recipe.merge_type,
            model_family=request.recipe.model_family,
        )
        source_identities = resolve_merge_xyz_sources(
            request.recipe,
            requires_source_c=initial_contract.requires_source_c,
        )
    except (MergeRuntimeContractError, MergeXYZValidationError) as exc:
        detail = exc.detail if isinstance(exc, MergeXYZValidationError) else str(exc)
        raise HTTPException(status_code=400, detail=detail) from exc

    estimate = _merge_xyz_estimate_response(request, plan, source_identities)
    if not estimate.png_representable:
        raise HTTPException(status_code=400, detail="xyz_precheck_contact_sheet_too_large")
    provided_fingerprint = request.confirmation_fingerprint.strip()
    if provided_fingerprint and provided_fingerprint != estimate.confirmation_fingerprint:
        raise HTTPException(status_code=409, detail="xyz_precheck_confirmation_stale")
    if estimate.requires_confirmation and not provided_fingerprint:
        raise HTTPException(status_code=409, detail="xyz_precheck_confirmation_required")

    operation_id = f"merge-xyz-{uuid4().hex}"
    base_snapshot = request.base.model_copy(deep=True)
    source_a_path, source_b, source_c = _merge_runtime_source_payload(request.recipe.model_family, source_identities)
    model_slot = "checkpoint" if request.recipe.model_family == "sdxl" else "diffusion_model"
    cells: list[XYZCellResult] = []
    contact_sheet_warning = ""
    contact_sheet_stager: XYZContactSheetStager | None = None
    if request.contact_sheet:
        try:
            contact_sheet_stager = XYZContactSheetStager(
                axes=[
                    XYZGridAxis(
                        key=axis.label.strip() or axis.target.strip() or f"axis_{index + 1}",
                        label=axis.label.strip() or axis.target.strip() or f"axis_{index + 1}",
                        values=tuple(str(value) for value in axis.values),
                    )
                    for index, axis in enumerate(request.axes)
                ],
                output_dir=_output_dir_for(_data_dir(), request.base.profile),
                operation_id=operation_id,
            )
        except (OSError, ValueError, MemoryError):
            contact_sheet_warning = "contact_sheet_creation_failed"
    try:
        with _merge_xyz_runtime_session(config) as lease_token:
            for planned_candidate in iter_merge_xyz_candidates(plan):
                index = planned_candidate.index
                values = planned_candidate.values
                candidate_recipe = request.recipe.model_copy(deep=True)
                candidate_recipe.id = f"{operation_id}-{index + 1:04d}"
                candidate_recipe.name = f"{request.recipe.name} precheck {index + 1}"
                candidate_recipe.output_name = ""
                candidate_recipe.save_model = False
                labels: dict[str, str] = {}
                for axis, value in zip(request.axes, values, strict=True):
                    value_text = str(value)
                    labels[axis.label.strip() or axis.target] = value_text
                    _apply_precheck_axis(candidate_recipe, axis.target, value_text, data_dir)
                _apply_pinpoint_axes(candidate_recipe, request.recipe, request.axes)
                try:
                    normalized_contract = normalize_merge_runtime_recipe(candidate_recipe)
                except MergeRuntimeContractError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                normalized_recipe = normalized_contract.model_dump(mode="json")

                cell_request = base_snapshot.model_copy(deep=True)
                for axis, value in zip(request.axes, values, strict=True):
                    _apply_precheck_generation_axis(cell_request, axis.target, str(value))
                cell_request.runtime = "comfy_workflow"
                cell_request.model_slot_overrides[model_slot] = source_a_path
                generation_snapshot = cell_request.model_dump(mode="json")
                merge_runtime_context: dict[str, Any] = {
                    "contract_version": MERGE_RUNTIME_CONTRACT_VERSION,
                    "recipe_payload": normalized_contract.model_dump_json(),
                    "source_b": dict(source_b),
                }
                if source_c is not None:
                    merge_runtime_context["source_c"] = dict(source_c)
                merge_xyz_metadata = {
                    "operation_id": operation_id,
                    "candidate_id": f"C-{index + 1:03d}",
                    "candidate_index": index + 1,
                    "candidate_count": plan.total_count,
                    "axes": [
                        {
                            "target": axis.target,
                            "label": axis.label,
                            "value": str(value),
                            "display": _preset_label(str(value)) if axis.target == "mbw_preset" else str(value),
                        }
                        for axis, value in zip(request.axes, values, strict=True)
                    ],
                    "source_identities": [source.metadata() for source in source_identities],
                    "normalized_recipe": normalized_recipe,
                    "resolved_mbw": {
                        "block_schema": normalized_recipe.get("block_schema"),
                        "block_weights": normalized_recipe.get("block_weights"),
                        "block_weights_alpha": normalized_recipe.get("block_weights_alpha"),
                        "block_weights_beta": normalized_recipe.get("block_weights_beta"),
                    },
                    "calculation": {
                        "mode": normalized_recipe.get("calculation_mode"),
                        "option": normalized_recipe.get("calculation_option"),
                        "merge_mode": normalized_recipe.get("merge_mode"),
                    },
                    "generation_snapshot": generation_snapshot,
                    "seed": cell_request.seed,
                    "runtime_adapter": "comfy_workflow",
                    "workflow_mapping_id": f"{cell_request.profile}_txt2img",
                    "candidate_state": "partial",
                    "operation_state": "partial",
                }
                generation = _generate(
                    cell_request,
                    "txt2img",
                    merge_runtime_context=merge_runtime_context,
                    comfy_admission_lease_token=lease_token,
                    merge_xyz_metadata=merge_xyz_metadata,
                )
                cell = XYZCellResult(index=index, labels=labels, generation=generation)
                expanded_recipe = expand_dice_recipe(candidate_recipe)
                candidate = _build_candidate_metadata(
                    operation_id=operation_id,
                    index=index,
                    axes=request.axes,
                    values=values,
                    labels=labels,
                    candidate_recipe=expanded_recipe,
                    normalized_recipe=normalized_recipe,
                    source_identities=source_identities,
                    generation_snapshot=generation_snapshot,
                    generation=cell,
                )
                cell.generation.settings["merge_xyz"] = candidate
                cell.generation.settings["merge_candidate"] = candidate
                cell.candidate = candidate
                cells.append(cell)
                if contact_sheet_stager is not None:
                    image_paths = _candidate_image_paths(cell.generation.output_paths)
                    contact_sheet_stager.add_cell(
                        XYZGridCell(
                            labels=labels,
                            image_path=image_paths[0] if image_paths else None,
                        )
                    )
    except BaseException:
        if contact_sheet_stager is not None:
            contact_sheet_stager.close()
        raise
    contact_sheet_path = ""
    if contact_sheet_stager is not None:
        try:
            contact_sheet = contact_sheet_stager.finalize()
            contact_sheet_path = str(contact_sheet) if contact_sheet is not None else ""
            contact_sheet_warning = contact_sheet_stager.warning_code
        finally:
            contact_sheet_stager.close()
    if _uses_effective_axis(request.axes):
        _attach_effective_checks(cells)
    history = HistoryRepository(AppConfig().data_dir / "asteria.db")
    for cell in cells:
        candidate = cell.candidate
        candidate["operation_state"] = "completed"
        cell.generation.settings["merge_xyz"] = candidate
        for image_path in _candidate_image_paths(cell.generation.output_paths):
            _draw_candidate_overlay(image_path, candidate["overlay_label"])
        candidate["output_paths"] = cell.generation.output_paths
        _write_candidate_sidecars(cell, candidate)
        candidate["output_paths"] = cell.generation.output_paths
        cell.generation.settings["merge_candidate"] = candidate
        _rewrite_merge_xyz_generation_metadata(cell)
        _update_settings_safely(history, cell.generation.id, cell.generation.settings)
        _update_output_paths_safely(history, cell.generation.id, cell.generation.output_paths)
    operation_status = "completed-with-warnings" if contact_sheet_warning else "completed"
    generation_id, output_paths = _record_xyz_history(
        request=type(
            "_MergeXYZRecord",
            (),
            {
                "base": request.base,
                "axes": request.axes,
                "status": operation_status,
                "warning_code": contact_sheet_warning,
            },
        )(),
        cells=cells,
        contact_sheet_path=contact_sheet_path,
    )
    return XYZPlotResponse(
        id=generation_id,
        status=operation_status,
        profile=request.base.profile,
        axes=request.axes,
        cells=cells,
        contact_sheet_path=contact_sheet_path,
        output_paths=output_paths,
    )


@router.post("/xyz-precheck/estimate", response_model=MergeXYZEstimateResponse)
def estimate_merge_xyz_precheck(request: MergeXYZPrecheckRequest) -> MergeXYZEstimateResponse:
    _validate_merge_xyz_precheck_axes(request)
    if request.base.profile != request.recipe.model_family:
        raise HTTPException(status_code=400, detail="xyz_precheck_profile_family_mismatch")
    try:
        initial_contract = normalize_merge_runtime_recipe(request.recipe)
        plan = normalize_merge_xyz_axes(
            request.axes,
            allowed_targets=MERGE_XYZ_CANONICAL_TARGETS,
            uses_beta=bool(MERGE_MODES.get(request.recipe.normalized_merge_mode(), {}).get("uses_beta", False)),
            merge_type=request.recipe.merge_type,
            model_family=request.recipe.model_family,
        )
        source_identities = resolve_merge_xyz_sources(
            request.recipe,
            requires_source_c=initial_contract.requires_source_c,
        )
    except (MergeRuntimeContractError, MergeXYZValidationError) as exc:
        detail = exc.detail if isinstance(exc, MergeXYZValidationError) else str(exc)
        raise HTTPException(status_code=400, detail=detail) from exc
    return _merge_xyz_estimate_response(request, plan, source_identities)
