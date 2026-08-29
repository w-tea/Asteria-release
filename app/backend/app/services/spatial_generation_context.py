from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence
from uuid import uuid4

from PIL import Image, ImageOps

from app.schemas.prompt_layout import PromptLayout
from app.schemas.spatial_prompt_runtime import (
    SpatialModelProfileReceipt,
    SpatialPromptPlan,
    canonical_runtime_sha256,
)
from app.services.spatial_mask_compiler import (
    CompiledSpatialMaskBundle,
    compile_spatial_masks,
)
from app.services.spatial_model_profiles import load_spatial_model_profiles
from app.services.spatial_profile_resolver import (
    SpatialModelEvidence,
    resolve_spatial_model_profile,
)
from app.services.spatial_prompt_contract import SpatialPromptSnapshot
from app.services.spatial_prompt_mapping import ASTERIA_SPATIAL_NODE_CONTRACT_VERSION
from app.services.spatial_prompt_planner import build_spatial_prompt_plan


SpatialStage = Literal["base", "hires", "refiner"]


class SpatialGenerationContextError(ValueError):
    def __init__(
        self,
        code: str,
        *,
        target_ids: Sequence[str] = (),
        stage: SpatialStage | None = None,
        profile_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.target_ids = tuple(target_ids)
        self.stage = stage
        self.profile_ids = tuple(profile_ids)


@dataclass(frozen=True, slots=True)
class SpatialStagePreparation:
    stage: SpatialStage
    layout: PromptLayout
    width: int
    height: int
    evidence: SpatialModelEvidence
    inpaint_mask: Image.Image | None = None
    crop_rect: tuple[int, int, int, int] | None = None
    manual_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedSpatialGenerationContext:
    prompt_plan: SpatialPromptPlan
    runtime_context: dict[str, object]
    history_record: dict[str, object]
    working_directory: Path


def _profile_json(
    receipt: SpatialModelProfileReceipt,
    *,
    profiles_path: Path,
) -> str:
    catalog = load_spatial_model_profiles(profiles_path)
    definition = next(
        (item for item in catalog.profiles if item.profile_id == receipt.profile_id),
        None,
    )
    if definition is None:
        raise SpatialGenerationContextError("spatial_profile_not_found")
    body = {
        "schema_version": "asteria.spatial-node-profile.v1",
        "contract_version": ASTERIA_SPATIAL_NODE_CONTRACT_VERSION,
        "profile_receipt": receipt.model_dump(mode="json"),
        "coefficients": definition.coefficients.model_dump(mode="json"),
    }
    payload = {
        **body,
        "profile_payload_sha256": canonical_runtime_sha256(body),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_target_ids(bundle: CompiledSpatialMaskBundle) -> tuple[str, ...]:
    targets = sorted(bundle.receipt.targets, key=lambda item: item.depth_index)
    if [item.depth_index for item in targets] != list(range(len(targets))):
        raise SpatialGenerationContextError("spatial_mask_depth_order_invalid")
    return tuple(item.target_id for item in targets)


def _materialize_masks(
    bundle: CompiledSpatialMaskBundle,
    *,
    stage: SpatialStage,
    directory: Path,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for index, compiled in enumerate(bundle.masks):
        source = compiled.open_image().convert("L")
        rgba = Image.new("RGBA", source.size, (255, 255, 255, 0))
        rgba.putalpha(ImageOps.invert(source))
        target = directory / f"{stage}-{index:03d}-{compiled.target_id}.png"
        temporary = directory / f".{target.name}.{uuid4().hex}.tmp"
        try:
            rgba.save(temporary, format="PNG")
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        paths[compiled.target_id] = str(target)
    return paths


def prepare_spatial_generation_context(
    snapshot: SpatialPromptSnapshot,
    *,
    profile_id: str,
    generation_seed: int,
    stages: Sequence[SpatialStagePreparation],
    profiles_path: str | Path,
    working_directory: str | Path,
    advanced_styles: Sequence[object] = (),
    hires_prompt: str = "",
    wildcard_dir: str | Path | None = None,
    known_embedding_names: Sequence[str] = (),
    data_dir: str | Path | None = None,
) -> PreparedSpatialGenerationContext:
    if not stages or stages[0].stage != "base":
        raise SpatialGenerationContextError("spatial_base_stage_required")
    stage_ids = tuple(item.stage for item in stages)
    if len(stage_ids) != len(set(stage_ids)) or any(
        stage not in {"base", "hires", "refiner"} for stage in stage_ids
    ):
        raise SpatialGenerationContextError("spatial_stage_set_invalid")
    if "hires" in stage_ids and stage_ids.index("hires") < stage_ids.index("base"):
        raise SpatialGenerationContextError("spatial_stage_set_invalid")
    if "refiner" in stage_ids and stage_ids[-1] != "refiner":
        raise SpatialGenerationContextError("spatial_stage_set_invalid")

    prompt_plan = build_spatial_prompt_plan(
        snapshot,
        profile_id=profile_id,
        generation_seed=generation_seed,
        advanced_styles=advanced_styles,
        hires_prompt=hires_prompt,
        include_hires="hires" in stage_ids,
        include_refiner="refiner" in stage_ids,
        wildcard_dir=Path(wildcard_dir) if wildcard_dir is not None else None,
        known_embedding_names=known_embedding_names,
    )
    profiles_file = Path(profiles_path)
    workdir = Path(working_directory)
    runtime_stages: dict[str, object] = {}
    history_stages: dict[str, object] = {}
    canonical_target_ids: tuple[str, ...] | None = None

    for item in stages:
        compiled = compile_spatial_masks(
            item.layout,
            width=item.width,
            height=item.height,
            stage=item.stage,
            data_dir=data_dir,
            inpaint_mask=item.inpaint_mask,
            crop_rect=item.crop_rect,
        )
        target_ids = _ordered_target_ids(compiled)
        if canonical_target_ids is None:
            canonical_target_ids = target_ids
        elif canonical_target_ids != target_ids:
            raise SpatialGenerationContextError(
                "spatial_stage_target_ids_mismatch",
                target_ids=target_ids,
            )
        receipt = resolve_spatial_model_profile(
            item.evidence,
            stage=item.stage,
            target_ids=target_ids,
            profiles_path=profiles_file,
            manual_profile_id=item.manual_profile_id,
        )
        if receipt.support_status == "selection_required":
            raise SpatialGenerationContextError(
                "spatial_profile_selection_required",
                stage=item.stage,
                profile_ids=(receipt.profile_id,),
            )
        if receipt.support_status != "supported":
            raise SpatialGenerationContextError(
                receipt.reason_code or "spatial_profile_unsupported",
                stage=item.stage,
            )
        profile_json = _profile_json(receipt, profiles_path=profiles_file)
        profile_payload = json.loads(profile_json)
        mask_paths = _materialize_masks(
            compiled,
            stage=item.stage,
            directory=workdir / item.stage,
        )
        runtime_stages[item.stage] = {
            "mask_bundle": compiled.receipt.model_dump(mode="json"),
            "profile_receipt": receipt.model_dump(mode="json"),
            "profile_json": profile_json,
            "mask_paths": mask_paths,
        }
        history_stages[item.stage] = {
            "mask_bundle": compiled.receipt.model_dump(mode="json"),
            "profile_receipt": receipt.model_dump(mode="json"),
            "profile_payload_sha256": profile_payload["profile_payload_sha256"],
        }

    runtime_context: dict[str, object] = {
        "schema_version": "asteria.spatial-workflow-context.v1",
        "prompt_plan": prompt_plan.model_dump(mode="json"),
        "stages": runtime_stages,
    }
    history_body: dict[str, object] = {
        "schema_version": "asteria.spatial-generation-context.v1",
        "prompt_plan": prompt_plan.model_dump(mode="json"),
        "stages": history_stages,
    }
    history_record = {
        **history_body,
        "context_sha256": canonical_runtime_sha256(history_body),
    }
    return PreparedSpatialGenerationContext(
        prompt_plan=prompt_plan,
        runtime_context=runtime_context,
        history_record=history_record,
        working_directory=workdir,
    )


def upload_spatial_runtime_masks(
    runtime_context: Mapping[str, object],
    *,
    upload,
) -> dict[str, object]:
    if runtime_context.get("schema_version") != "asteria.spatial-workflow-context.v1":
        raise SpatialGenerationContextError("spatial_runtime_context_invalid")
    stages = runtime_context.get("stages")
    if not isinstance(stages, Mapping):
        raise SpatialGenerationContextError("spatial_runtime_context_invalid")
    uploaded_context = dict(runtime_context)
    uploaded_stages: dict[str, object] = {}
    for stage, raw_payload in stages.items():
        if not isinstance(stage, str) or not isinstance(raw_payload, Mapping):
            raise SpatialGenerationContextError("spatial_runtime_context_invalid")
        raw_paths = raw_payload.get("mask_paths")
        if not isinstance(raw_paths, Mapping):
            raise SpatialGenerationContextError("spatial_runtime_context_invalid")
        uploaded_paths: dict[str, str] = {}
        for target_id, raw_path in raw_paths.items():
            if not isinstance(target_id, str) or not isinstance(raw_path, str):
                raise SpatialGenerationContextError("spatial_runtime_context_invalid")
            path = Path(raw_path)
            if not path.is_file():
                raise SpatialGenerationContextError("spatial_mask_file_unavailable")
            uploaded_paths[target_id] = str(upload(path))
        uploaded_stages[stage] = {**dict(raw_payload), "mask_paths": uploaded_paths}
    uploaded_context["stages"] = uploaded_stages
    return uploaded_context
