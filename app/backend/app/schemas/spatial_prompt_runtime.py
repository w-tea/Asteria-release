from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.prompt_input import join_prompt_contract_fragments

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


class SpatialCharacterBindingReceipt(_RuntimeModel):
    target_id: TargetId
    fragment_id: str = Field(pattern=FRAGMENT_ID_PATTERN.pattern)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: Sha256
    separator_before: Literal["", ",\n", "\n"]
    syntax_plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding_identity(self) -> "SpatialCharacterBindingReceipt":
        if self.end <= self.start:
            raise ValueError("character binding end must be greater than start")
        if self.fragment_id != f"base:target:{self.target_id}" and not self.fragment_id.endswith(
            f":target:{self.target_id}"
        ):
            raise ValueError("character binding target_id does not match fragment_id")
        return self


class SpatialScheduledPromptBinding(_RuntimeModel):
    schema_version: Literal["asteria.spatial-scheduled-binding.v1"]
    start_percent: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    end_percent: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    full_text: str
    full_text_sha256: Sha256
    character_bindings: tuple[SpatialCharacterBindingReceipt, ...]
    global_character_ranges: tuple[tuple[int, int], ...]
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_scheduled_binding(self) -> "SpatialScheduledPromptBinding":
        if self.end_percent <= self.start_percent:
            raise ValueError("scheduled binding end must be greater than start")
        _validate_character_ranges(
            full_text=self.full_text,
            full_text_sha256=self.full_text_sha256,
            character_bindings=self.character_bindings,
            global_character_ranges=self.global_character_ranges,
        )
        _require_receipt_hash(self, "binding_sha256")
        return self


class SpatialStagePromptPlan(_RuntimeModel):
    schema_version: Literal[
        "asteria.spatial-stage-prompt.v1",
        "asteria.spatial-stage-prompt.v2",
    ]
    stage: Stage
    global_fragment: SpatialPromptFragmentReceipt
    target_fragments: tuple[SpatialPromptFragmentReceipt, ...]
    excluded_targets: tuple[SpatialExcludedTargetReceipt, ...] = ()
    combined_expanded_prompt: str
    combined_expanded_sha256: Sha256
    reused_from_stage: Literal["base", "hires"] | None = None
    full_text: str | None = Field(default=None, exclude_if=lambda value: value is None)
    full_text_sha256: Sha256 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    character_bindings: tuple[SpatialCharacterBindingReceipt, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    global_character_ranges: tuple[tuple[int, int], ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    fragment_receipts: tuple[SpatialPromptFragmentReceipt, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    scheduled_bindings: tuple[SpatialScheduledPromptBinding, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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
        combined = join_prompt_contract_fragments(
            item.expanded_prompt
            for item in (self.global_fragment, *self.target_fragments)
            if item.expanded_prompt
        )
        if self.combined_expanded_prompt != combined:
            raise ValueError("combined_expanded_prompt does not match its fragments")
        if self.combined_expanded_sha256 != runtime_text_sha256(combined):
            raise ValueError("combined_expanded_sha256 does not match the combined prompt")
        self._validate_guidance_v2()
        _require_receipt_hash(self, "stage_receipt_sha256")
        return self

    def _validate_guidance_v2(self) -> None:
        guidance_fields = (
            self.full_text,
            self.full_text_sha256,
            self.character_bindings,
            self.global_character_ranges,
            self.fragment_receipts,
            self.scheduled_bindings,
        )
        if self.schema_version == "asteria.spatial-stage-prompt.v1":
            if any(item is not None for item in guidance_fields):
                raise ValueError("v1 stage must not contain v2 guidance fields")
            return
        if any(item is None for item in guidance_fields):
            raise ValueError("v2 stage requires all guidance fields")
        assert self.full_text is not None
        assert self.full_text_sha256 is not None
        assert self.character_bindings is not None
        assert self.global_character_ranges is not None
        assert self.fragment_receipts is not None
        assert self.scheduled_bindings is not None
        expected_fragments = (self.global_fragment, *self.target_fragments)
        if self.fragment_receipts != expected_fragments:
            raise ValueError("fragment_receipts do not match stage fragments")
        _validate_character_ranges(
            full_text=self.full_text,
            full_text_sha256=self.full_text_sha256,
            character_bindings=self.character_bindings,
            global_character_ranges=self.global_character_ranges,
        )
        expected_targets = tuple(item.target_id for item in self.target_fragments)
        if tuple(item.target_id for item in self.character_bindings) != expected_targets:
            raise ValueError("character binding target_ids do not match target fragments")
        if self.scheduled_bindings:
            if self.full_text != self.scheduled_bindings[0].full_text:
                raise ValueError("full_text must match the first scheduled binding")
            cursor = 0.0
            for binding in self.scheduled_bindings:
                if binding.start_percent != cursor:
                    raise ValueError("scheduled bindings must be contiguous from zero")
                if tuple(item.target_id for item in binding.character_bindings) != expected_targets:
                    raise ValueError("scheduled binding target_ids do not match target fragments")
                cursor = binding.end_percent
            if cursor != 1.0:
                raise ValueError("scheduled bindings must cover the full sampling interval")
        elif self.full_text != self.combined_expanded_prompt:
            raise ValueError("full_text does not match the expanded stage prompt")


class SpatialPromptPlan(_RuntimeModel):
    schema_version: Literal[
        "asteria.spatial-prompt-plan.v1",
        "asteria.spatial-prompt-plan.v2",
    ]
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
            expected_schema = (
                "asteria.spatial-stage-prompt.v2"
                if self.schema_version == "asteria.spatial-prompt-plan.v2"
                else "asteria.spatial-stage-prompt.v1"
            )
            if stage.schema_version != expected_schema:
                raise ValueError("stage schema does not match the prompt plan schema")
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


class SpatialConditioningBundle(_RuntimeModel):
    schema_version: Literal["asteria.spatial-conditioning-bundle.v2"]
    stage: Stage
    encoder_id: str = Field(min_length=1, max_length=512)
    profile_receipt_sha256: Sha256
    text_encoder_ref: str = Field(min_length=1, max_length=2048)
    full_positive_conditioning_ref: str = Field(min_length=1, max_length=2048)
    negative_conditioning_ref: str = Field(min_length=1, max_length=2048)
    full_text: str
    full_text_sha256: Sha256
    character_bindings: tuple[SpatialCharacterBindingReceipt, ...] = Field(min_length=1)
    scheduled_bindings: tuple[SpatialScheduledPromptBinding, ...]
    required_encoders: tuple[str, ...] = Field(min_length=1)
    adapter_kind: Literal[
        "identity",
        "anima_native_t5_query_v1",
        "anima_progressive_qwen35_t5_query_v1",
    ]
    step_start_percent: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    step_end_percent: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "SpatialConditioningBundle":
        target_ids = [item.target_id for item in self.character_bindings]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("conditioning target ids must be unique")
        if self.full_text_sha256 != runtime_text_sha256(self.full_text):
            raise ValueError("conditioning full text hash does not match")
        if len(self.required_encoders) != len(set(self.required_encoders)):
            raise ValueError("conditioning encoders must be unique")
        if self.step_start_percent >= self.step_end_percent:
            raise ValueError("conditioning step range must increase")
        if self.scheduled_bindings:
            expected_ids = tuple(target_ids)
            cursor = 0.0
            for scheduled in self.scheduled_bindings:
                if abs(scheduled.start_percent - cursor) > 1e-9:
                    raise ValueError("conditioning schedule must be contiguous")
                if tuple(item.target_id for item in scheduled.character_bindings) != expected_ids:
                    raise ValueError("conditioning schedule target ids do not match")
                cursor = scheduled.end_percent
            if abs(cursor - 1.0) > 1e-9:
                raise ValueError("conditioning schedule must cover the full sampling range")
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


def _validate_character_ranges(
    *,
    full_text: str,
    full_text_sha256: str,
    character_bindings: tuple[SpatialCharacterBindingReceipt, ...],
    global_character_ranges: tuple[tuple[int, int], ...],
) -> None:
    if full_text_sha256 != runtime_text_sha256(full_text):
        raise ValueError("full_text_sha256 does not match full_text")
    ranges: list[tuple[int, int, str]] = []
    for start, end in global_character_ranges:
        if start < 0 or end <= start or end > len(full_text):
            raise ValueError("global character range is outside full_text")
        ranges.append((start, end, "global"))
    for binding in character_bindings:
        if binding.end > len(full_text):
            raise ValueError("character binding is outside full_text")
        if runtime_text_sha256(full_text[binding.start : binding.end]) != binding.text_sha256:
            raise ValueError("character binding text_sha256 does not match full_text")
        separator_start = binding.start - len(binding.separator_before)
        if separator_start < 0 or full_text[separator_start : binding.start] != binding.separator_before:
            raise ValueError("character binding separator does not match full_text")
        ranges.append((binding.start, binding.end, binding.target_id))
    ordered = sorted(ranges, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            raise ValueError("character ranges overlap")
