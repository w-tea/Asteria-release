from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from app.schemas.prompt_input import PromptInput, PromptInputV2, TargetPromptContent
from app.schemas.prompt_layout import PromptLayout
from app.services.prompt_input import (
    PromptInputError,
    saved_prompt_input_matches,
    upgrade_prompt_input_to_v2,
    validate_prompt_input_pair,
)


GenerationMode = Literal["txt2img", "img2img", "inpaint"]
SpatialPromptWarningCode = Literal["spatial_prompt_invalid_fallback"]
SpatialPromptRecoveryStatus = Literal["ready", "fallback"]


class SpatialPromptContractError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SpatialPromptSnapshot:
    prompt: str
    prompt_input: PromptInput | None
    prompt_layout: PromptLayout | None
    mode: GenerationMode
    target_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpatialPromptRecovery:
    snapshot: SpatialPromptSnapshot
    status: SpatialPromptRecoveryStatus
    warning_code: SpatialPromptWarningCode | None = None
    legacy_snapshot: dict[str, object] | None = None


def _require_generation_mode(mode: GenerationMode) -> None:
    if mode not in {"txt2img", "img2img", "inpaint"}:
        raise SpatialPromptContractError("prompt_layout_generation_mode_invalid")


def _ordered_prompt_target_ids(prompt_input: PromptInput) -> tuple[str, ...]:
    if not isinstance(prompt_input.root, PromptInputV2) or prompt_input.mode != "sections":
        return ()
    return tuple(
        target.id
        for section in sorted(prompt_input.sections or (), key=lambda item: item.order)
        if isinstance(section.content, TargetPromptContent)
        for target in sorted(section.content.targets, key=lambda item: item.order)
    )


def validate_spatial_prompt_contract(
    prompt: str,
    prompt_input: PromptInput | None,
    prompt_layout: PromptLayout | None,
    mode: GenerationMode,
) -> SpatialPromptSnapshot:
    """Validate the public prompt triplet once and return an immutable snapshot."""
    _require_generation_mode(mode)

    if prompt_input is None:
        if prompt_layout is not None:
            raise SpatialPromptContractError("prompt_layout_requires_v2_targets")
        return SpatialPromptSnapshot(
            prompt=prompt,
            prompt_input=None,
            prompt_layout=None,
            mode=mode,
        )

    try:
        validate_prompt_input_pair(prompt, prompt_input)
    except PromptInputError as exc:
        raise SpatialPromptContractError(exc.code) from exc

    if not isinstance(prompt_input.root, PromptInputV2):
        if prompt_layout is not None:
            raise SpatialPromptContractError("prompt_layout_requires_v2_targets")
        return SpatialPromptSnapshot(
            prompt=prompt,
            prompt_input=prompt_input,
            prompt_layout=None,
            mode=mode,
        )
    if prompt_input.mode == "single":
        if prompt_layout is not None:
            raise SpatialPromptContractError("prompt_layout_requires_v2_targets")
        return SpatialPromptSnapshot(
            prompt=prompt,
            prompt_input=prompt_input,
            prompt_layout=None,
            mode=mode,
        )

    target_ids = _ordered_prompt_target_ids(prompt_input)
    if not target_ids:
        if prompt_layout is not None:
            raise SpatialPromptContractError("prompt_layout_forbidden_without_targets")
        return SpatialPromptSnapshot(
            prompt=prompt,
            prompt_input=prompt_input,
            prompt_layout=None,
            mode=mode,
        )
    if prompt_layout is None:
        raise SpatialPromptContractError("prompt_layout_required")

    layout_target_ids = tuple(target.target_id for target in prompt_layout.targets)
    if len(layout_target_ids) != len(target_ids) or set(layout_target_ids) != set(target_ids):
        raise SpatialPromptContractError("prompt_layout_target_mismatch")
    if not set(prompt_layout.excluded_target_ids).issubset(target_ids):
        raise SpatialPromptContractError("prompt_layout_target_mismatch")
    if prompt_layout.excluded_target_ids and mode != "inpaint":
        raise SpatialPromptContractError("prompt_layout_exclusions_require_inpaint")

    return SpatialPromptSnapshot(
        prompt=prompt,
        prompt_input=prompt_input,
        prompt_layout=prompt_layout,
        mode=mode,
        target_ids=target_ids,
    )


def recover_saved_spatial_prompt_contract(
    prompt: str,
    raw_prompt_input: object,
    raw_prompt_layout: object,
    mode: GenerationMode,
) -> SpatialPromptRecovery:
    """Recover saved spatial data without modifying or partially applying its source."""
    _require_generation_mode(mode)
    if raw_prompt_input is None and raw_prompt_layout is None:
        return SpatialPromptRecovery(
            snapshot=SpatialPromptSnapshot(
                prompt=prompt,
                prompt_input=None,
                prompt_layout=None,
                mode=mode,
            ),
            status="ready",
        )

    try:
        parsed_input = (
            PromptInput.model_validate(raw_prompt_input)
            if raw_prompt_input is not None
            else None
        )
        parsed_layout = (
            PromptLayout.model_validate(raw_prompt_layout)
            if raw_prompt_layout is not None
            else None
        )
        if parsed_input is not None:
            parsed_input = upgrade_prompt_input_to_v2(parsed_input)
        validation_prompt = prompt
        if parsed_input is not None and saved_prompt_input_matches(prompt, parsed_input):
            validation_prompt = parsed_input.canonical_prompt or prompt
        snapshot = validate_spatial_prompt_contract(
            validation_prompt,
            parsed_input,
            parsed_layout,
            mode,
        )
        if validation_prompt != prompt:
            snapshot = SpatialPromptSnapshot(
                prompt=prompt,
                prompt_input=snapshot.prompt_input,
                prompt_layout=snapshot.prompt_layout,
                mode=snapshot.mode,
                target_ids=snapshot.target_ids,
            )
    except (TypeError, ValidationError, SpatialPromptContractError):
        fallback_input = PromptInput.model_validate(
            {
                "schema_version": "asteria.prompt-input.v2",
                "mode": "single",
                "sections": None,
            }
        )
        return SpatialPromptRecovery(
            snapshot=SpatialPromptSnapshot(
                prompt=prompt,
                prompt_input=fallback_input,
                prompt_layout=None,
                mode=mode,
            ),
            status="fallback",
            warning_code="spatial_prompt_invalid_fallback",
            legacy_snapshot={
                "prompt_input": deepcopy(raw_prompt_input),
                "prompt_layout": deepcopy(raw_prompt_layout),
            },
        )
    return SpatialPromptRecovery(snapshot=snapshot, status="ready")
