from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.prompt_layout import TARGET_ID_PATTERN


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FRAGMENT_ID_PATTERN = re.compile(
    rf"^(?:base|hires):(?:global|target:{TARGET_ID_PATTERN[1:-1]})$"
)

Stage = Literal["base", "hires", "refiner"]
PromptStage = Literal["base", "hires"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TargetId = Annotated[str, Field(pattern=TARGET_ID_PATTERN)]


class _RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_runtime_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_text_sha256(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _receipt_sha256(model: BaseModel, field: str) -> str:
    return canonical_runtime_sha256(model.model_dump(mode="json", exclude={field}))


def _require_receipt_hash(model: BaseModel, field: str) -> None:
    if getattr(model, field) != _receipt_sha256(model, field):
        raise ValueError(f"{field} does not match the canonical receipt payload")


class SpatialPromptSelectionReceipt(_RuntimeModel):
    occurrence: int = Field(ge=0)
    source_token: str = Field(min_length=1, max_length=4096)
    selected_value: str = Field(max_length=65536)
    selection_kind: Literal["wildcard", "choice"]


class SpatialLoraReceipt(_RuntimeModel):
    name: str = Field(min_length=1, max_length=1024)
    weight: float = Field(ge=-10.0, le=10.0, allow_inf_nan=False)


class SpatialPromptSyntaxSegment(_RuntimeModel):
    text: str
    weight: str = "1"
    start: str | None = None
    end: str | None = None
    assignment: Literal["A", "B"] | None = None


class SpatialPromptSyntaxReceipt(_RuntimeModel):
    schema_version: Literal["asteria.prompt-plan.v1"]
    parser_generation: Literal["asteria-prompt-v1"]
    profile_id: str = Field(min_length=1, max_length=512)
    operation: Literal[
        "identity",
        "token_weight",
        "scheduled_switch",
        "alternating_switch",
        "conditioning_and",
        "conditioning_break",
    ]
    enabled_features: tuple[str, ...] = ()
    segments: tuple[SpatialPromptSyntaxSegment, ...] = Field(min_length=1, max_length=8)
    sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt_hash(self) -> "SpatialPromptSyntaxReceipt":
        _require_receipt_hash(self, "sha256")
        return self


class SpatialPromptFragmentReceipt(_RuntimeModel):
    schema_version: Literal["asteria.spatial-prompt-fragment.v1"]
    fragment_id: str = Field(pattern=FRAGMENT_ID_PATTERN.pattern)
    fragment_seed: int = Field(ge=0, le=(2**64) - 1)
    stage: PromptStage
    kind: Literal["global", "target"]
    target_id: TargetId | None = None
    raw_prompt: str
    prepared_raw_prompt: str
    expanded_prompt: str
    raw_sha256: Sha256
    expanded_sha256: Sha256
    selections: tuple[SpatialPromptSelectionReceipt, ...] = ()
    loras: tuple[SpatialLoraReceipt, ...] = ()
    embeddings: tuple[str, ...] = ()
    syntax_plan: SpatialPromptSyntaxReceipt
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_fragment_identity(self) -> "SpatialPromptFragmentReceipt":
        if self.kind == "global":
            if self.target_id is not None or self.fragment_id != f"{self.stage}:global":
                raise ValueError("global fragment identity does not match its stage")
            return self._validate_integrity()
        if self.target_id is None:
            raise ValueError("target fragment requires a target_id")
        if self.fragment_id != f"{self.stage}:target:{self.target_id}":
            raise ValueError("target fragment identity does not match its target_id")
        return self._validate_integrity()

    def _validate_integrity(self) -> "SpatialPromptFragmentReceipt":
        if self.raw_sha256 != runtime_text_sha256(self.raw_prompt):
            raise ValueError("raw_sha256 does not match raw_prompt")
        if self.expanded_sha256 != runtime_text_sha256(self.expanded_prompt):
            raise ValueError("expanded_sha256 does not match expanded_prompt")
        if [item.occurrence for item in self.selections] != list(range(len(self.selections))):
            raise ValueError("fragment selection occurrences must be contiguous from zero")
        _require_receipt_hash(self, "receipt_sha256")
        return self


class SpatialExcludedTargetReceipt(_RuntimeModel):
    target_id: TargetId
    reason: Literal["user_choice", "inpaint_no_overlap"]


class SpatialStagePromptPlan(_RuntimeModel):
    schema_version: Literal["asteria.spatial-stage-prompt.v1"]
    stage: Stage
    global_fragment: SpatialPromptFragmentReceipt
    target_fragments: tuple[SpatialPromptFragmentReceipt, ...]
    excluded_targets: tuple[SpatialExcludedTargetReceipt, ...] = ()
    combined_expanded_prompt: str
    combined_expanded_sha256: Sha256
    reused_from_stage: Literal["base", "hires"] | None = None
    stage_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_stage_fragments(self) -> "SpatialStagePromptPlan":
        target_ids = [item.target_id for item in self.target_fragments]
        excluded_ids = [item.target_id for item in self.excluded_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target fragment ids must be unique")
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("excluded target ids must be unique")
        if set(target_ids) & set(excluded_ids):
            raise ValueError("a target cannot be active and excluded")
        if any(item.kind != "target" for item in self.target_fragments):
            raise ValueError("target_fragments must contain target fragments")
        if self.global_fragment.kind != "global":
            raise ValueError("global_fragment must contain a global fragment")
        if self.stage == "base":
            if self.reused_from_stage is not None:
                raise ValueError("base stage cannot reuse another stage")
            if self.global_fragment.stage != "base" or any(
                item.stage != "base" for item in self.target_fragments
            ):
                raise ValueError("base stage requires base fragment receipts")
            return self._validate_integrity()
        if self.stage == "hires" and self.reused_from_stage == "base":
            if self.global_fragment.stage != "base" or any(
                item.stage != "base" for item in self.target_fragments
            ):
                raise ValueError("reused hires stage must preserve base receipts")
            return self._validate_integrity()
        if self.stage == "hires" and self.global_fragment.stage != "hires":
            raise ValueError("explicit hires stage requires a hires global receipt")
        if self.stage == "hires" and any(
            item.stage != "base" for item in self.target_fragments
        ):
            raise ValueError("hires target fragments must reuse base receipts")
        if self.stage == "hires":
            return self._validate_integrity()
        if self.reused_from_stage not in {"base", "hires"}:
            raise ValueError("refiner stage must reuse base or hires fragments")
        expected_stage = self.reused_from_stage
        if self.global_fragment.stage != expected_stage or any(
            item.stage != "base" for item in self.target_fragments
        ):
            raise ValueError("refiner fragments do not match the reused stage")
        return self._validate_integrity()

    def _validate_integrity(self) -> "SpatialStagePromptPlan":
        combined = "\n".join(
            item.expanded_prompt
            for item in (self.global_fragment, *self.target_fragments)
            if item.expanded_prompt
        )
        if self.combined_expanded_prompt != combined:
            raise ValueError("combined_expanded_prompt does not match its fragments")
        if self.combined_expanded_sha256 != runtime_text_sha256(combined):
            raise ValueError("combined_expanded_sha256 does not match the combined prompt")
        _require_receipt_hash(self, "stage_receipt_sha256")
        return self


class SpatialPromptPlan(_RuntimeModel):
    schema_version: Literal["asteria.spatial-prompt-plan.v1"]
    planner_revision: str = Field(min_length=1, max_length=256)
    profile_id: str = Field(min_length=1, max_length=512)
    generation_seed: int = Field(ge=0, le=(2**64) - 1)
    expansion_version: str = Field(min_length=1, max_length=256)
    canonical_raw_prompt: str
    canonical_raw_sha256: Sha256
    layout_global_raw_prompt: str
    layout_global_raw_sha256: Sha256
    target_ids: tuple[TargetId, ...]
    base: SpatialStagePromptPlan
    hires: SpatialStagePromptPlan | None = None
    refiner: SpatialStagePromptPlan | None = None
    loras: tuple[SpatialLoraReceipt, ...] = ()
    embeddings: tuple[str, ...] = ()
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan_targets(self) -> "SpatialPromptPlan":
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("prompt plan target ids must be unique")
        if self.base.stage != "base":
            raise ValueError("base plan must use the base stage")
        expected = set(self.target_ids)
        for expected_stage, stage in (
            ("base", self.base),
            ("hires", self.hires),
            ("refiner", self.refiner),
        ):
            if stage is None:
                continue
            if stage.stage != expected_stage:
                raise ValueError(f"{expected_stage} plan must use the {expected_stage} stage")
            actual = {item.target_id for item in stage.target_fragments}
            actual.update(item.target_id for item in stage.excluded_targets)
            if actual != expected:
                raise ValueError("stage target ids do not match the prompt plan")
        if self.canonical_raw_sha256 != runtime_text_sha256(self.canonical_raw_prompt):
            raise ValueError("canonical_raw_sha256 does not match canonical_raw_prompt")
        if self.layout_global_raw_sha256 != runtime_text_sha256(self.layout_global_raw_prompt):
            raise ValueError("layout_global_raw_sha256 does not match layout_global_raw_prompt")
        _require_receipt_hash(self, "plan_sha256")
        return self


class SpatialMaskTargetReceipt(_RuntimeModel):
    target_id: TargetId
    mask_sha256: Sha256
    overlap_weight: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    depth_index: int = Field(ge=0)
    strength_mode: Literal["auto", "manual"]
    strength: int | None = Field(default=None, ge=0, le=100)
    blending_mode: Literal["auto", "manual"]
    blending: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def validate_relative_controls(self) -> "SpatialMaskTargetReceipt":
        if (self.strength_mode == "auto") != (self.strength is None):
            raise ValueError("spatial strength must be null only in auto mode")
        if (self.blending_mode == "auto") != (self.blending is None):
            raise ValueError("spatial blending must be null only in auto mode")
        return self


class SpatialMaskBundle(_RuntimeModel):
    schema_version: Literal["asteria.spatial-mask-bundle.v1"]
    stage: Stage
    width: int = Field(ge=1, le=65536)
    height: int = Field(ge=1, le=65536)
    targets: tuple[SpatialMaskTargetReceipt, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "SpatialMaskBundle":
        _require_unique_target_ids(self.targets)
        _require_receipt_hash(self, "bundle_sha256")
        return self


class SpatialModelProfileReceipt(_RuntimeModel):
    schema_version: Literal["asteria.spatial-model-profile.v1"]
    stage: Stage
    profile_id: str = Field(min_length=1, max_length=512)
    profile_revision: str = Field(min_length=1, max_length=128)
    model_family: str = Field(min_length=1, max_length=256)
    model_slot_id: str = Field(min_length=1, max_length=1024)
    encoder_kind: str = Field(min_length=1, max_length=256)
    verified_structure_id: str = Field(min_length=1, max_length=256)
    layout_fingerprint: Sha256
    structure_fingerprint: Sha256
    support_status: Literal["supported", "selection_required", "unsupported"]
    selection_mode: Literal["auto", "manual"]
    reason_code: str | None = Field(default=None, min_length=1, max_length=256)
    target_ids: tuple[TargetId, ...] = Field(min_length=1)
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "SpatialModelProfileReceipt":
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("profile target ids must be unique")
        _require_receipt_hash(self, "receipt_sha256")
        return self


class SpatialConditioningTarget(_RuntimeModel):
    target_id: TargetId
    target_conditioning_ref: str = Field(min_length=1, max_length=2048)


class SpatialConditioningBundle(_RuntimeModel):
    schema_version: Literal["asteria.spatial-conditioning-bundle.v1"]
    stage: Stage
    encoder_id: str = Field(min_length=1, max_length=512)
    profile_receipt_sha256: Sha256
    global_conditioning_ref: str = Field(min_length=1, max_length=2048)
    negative_conditioning_ref: str = Field(min_length=1, max_length=2048)
    targets: tuple[SpatialConditioningTarget, ...] = Field(min_length=1)
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "SpatialConditioningBundle":
        _require_unique_target_ids(self.targets)
        _require_receipt_hash(self, "bundle_sha256")
        return self


class SpatialExecutionTargetReceipt(_RuntimeModel):
    target_id: TargetId
    prompt_fragment_sha256: Sha256
    mask_sha256: Sha256
    conditioning_ref: str = Field(min_length=1, max_length=2048)
    applied: bool


class SpatialExecutionReceipt(_RuntimeModel):
    schema_version: Literal["asteria.spatial-execution.v1"]
    stage: Stage
    adapter_revision: str = Field(min_length=1, max_length=256)
    prompt_plan_sha256: Sha256
    mask_bundle_sha256: Sha256
    conditioning_bundle_sha256: Sha256
    targets: tuple[SpatialExecutionTargetReceipt, ...] = Field(min_length=1)
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "SpatialExecutionReceipt":
        _require_unique_target_ids(self.targets)
        _require_receipt_hash(self, "receipt_sha256")
        return self


def _require_unique_target_ids(items: tuple[object, ...]) -> None:
    target_ids = [getattr(item, "target_id") for item in items]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("bundle target ids must be unique")
