from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.schemas.prompt_input import (
    GlobalPromptContent,
    PromptInputV2,
    TargetPromptContent,
)
from app.schemas.spatial_prompt_runtime import (
    SpatialExcludedTargetReceipt,
    SpatialLoraReceipt,
    SpatialPromptFragmentReceipt,
    SpatialPromptPlan,
    SpatialPromptSelectionReceipt,
    SpatialPromptSyntaxReceipt,
    SpatialStagePromptPlan,
    canonical_runtime_sha256,
    runtime_text_sha256,
)
from app.services.prompt_pipeline import (
    expand_prompt_fragment,
    extract_embeddings,
    extract_loras,
)
from app.services.prompt_syntax import parse_prompt_syntax
from app.services.spatial_prompt_contract import SpatialPromptSnapshot


SPATIAL_PROMPT_PLANNER_REVISION = "asteria-spatial-planner-v1"
SPATIAL_FRAGMENT_EXPANSION_VERSION = "asteria-fragment-expansion-v1"


class SpatialPromptPlannerError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _sha256_json(payload: object) -> str:
    return canonical_runtime_sha256(payload)


def _sha256_text(value: str) -> str:
    return runtime_text_sha256(value)


def spatial_fragment_seed(
    generation_seed: int,
    fragment_id: str,
    *,
    expansion_version: str = SPATIAL_FRAGMENT_EXPANSION_VERSION,
) -> int:
    if generation_seed < 0 or generation_seed > (2**64) - 1:
        raise SpatialPromptPlannerError("spatial_prompt_seed_invalid")
    if not fragment_id or not expansion_version:
        raise SpatialPromptPlannerError("spatial_prompt_fragment_identity_invalid")
    digest = canonical_runtime_sha256(
        {
            "expansion_version": expansion_version,
            "fragment_id": fragment_id,
            "generation_seed": generation_seed,
        }
    )
    return int.from_bytes(bytes.fromhex(digest)[:8], "big")


def _style_value(style: object, field: str, default: object = "") -> object:
    if isinstance(style, Mapping):
        return style.get(field, default)
    return getattr(style, field, default)


def _apply_style_text(base: str, style_text: object) -> str:
    value = str(style_text or "").strip()
    if not value:
        return base
    if "{prompt}" in value:
        return value.replace("{prompt}", base).strip()
    if not base.strip():
        return value
    return f"{base.rstrip()}, {value}"


def _prepare_base_global(raw_prompt: str, styles: Sequence[object]) -> str:
    prepared = raw_prompt
    for style in sorted(styles, key=lambda item: int(_style_value(item, "order", 0))):
        prepared = _apply_style_text(prepared, _style_value(style, "prompt", ""))
    return prepared


def _syntax_receipt(expanded_prompt: str, profile_id: str) -> SpatialPromptSyntaxReceipt:
    plan = parse_prompt_syntax(expanded_prompt, profile_id=profile_id)
    return SpatialPromptSyntaxReceipt.model_validate({**plan.to_dict(), "sha256": plan.sha256})


def _fragment_receipt(
    *,
    fragment_id: str,
    stage: str,
    kind: str,
    target_id: str | None,
    raw_prompt: str,
    prepared_raw_prompt: str,
    profile_id: str,
    generation_seed: int,
    expansion_version: str,
    wildcard_dir: Path | None,
    known_embedding_names: Sequence[str],
) -> SpatialPromptFragmentReceipt:
    fragment_seed = spatial_fragment_seed(
        generation_seed,
        fragment_id,
        expansion_version=expansion_version,
    )
    expansion = expand_prompt_fragment(
        prepared_raw_prompt,
        seed=fragment_seed,
        wildcard_dir=wildcard_dir,
    )
    syntax = _syntax_receipt(expansion.expanded, profile_id)
    payload: dict[str, Any] = {
        "schema_version": "asteria.spatial-prompt-fragment.v1",
        "fragment_id": fragment_id,
        "fragment_seed": fragment_seed,
        "stage": stage,
        "kind": kind,
        "target_id": target_id,
        "raw_prompt": raw_prompt,
        "prepared_raw_prompt": prepared_raw_prompt,
        "expanded_prompt": expansion.expanded,
        "raw_sha256": _sha256_text(raw_prompt),
        "expanded_sha256": _sha256_text(expansion.expanded),
        "selections": [
            SpatialPromptSelectionReceipt(
                occurrence=item.occurrence,
                source_token=item.source_token,
                selected_value=item.selected_value,
                selection_kind=item.selection_kind,
            ).model_dump(mode="json")
            for item in expansion.selections
        ],
        "loras": [
            SpatialLoraReceipt.model_validate(item).model_dump(mode="json")
            for item in extract_loras(expansion.expanded)
        ],
        "embeddings": extract_embeddings(
            expansion.expanded,
            list(known_embedding_names),
        ),
        "syntax_plan": syntax.model_dump(mode="json"),
    }
    return SpatialPromptFragmentReceipt.model_validate(
        {**payload, "receipt_sha256": _sha256_json(payload)}
    )


def _stage_receipt(
    *,
    stage: str,
    global_fragment: SpatialPromptFragmentReceipt,
    target_fragments: Sequence[SpatialPromptFragmentReceipt],
    excluded_targets: Sequence[SpatialExcludedTargetReceipt],
    reused_from_stage: str | None,
) -> SpatialStagePromptPlan:
    combined = "\n".join(
        item.expanded_prompt
        for item in (global_fragment, *target_fragments)
        if item.expanded_prompt
    )
    payload: dict[str, Any] = {
        "schema_version": "asteria.spatial-stage-prompt.v1",
        "stage": stage,
        "global_fragment": global_fragment.model_dump(mode="json"),
        "target_fragments": [item.model_dump(mode="json") for item in target_fragments],
        "excluded_targets": [item.model_dump(mode="json") for item in excluded_targets],
        "combined_expanded_prompt": combined,
        "combined_expanded_sha256": _sha256_text(combined),
        "reused_from_stage": reused_from_stage,
    }
    return SpatialStagePromptPlan.model_validate(
        {**payload, "stage_receipt_sha256": _sha256_json(payload)}
    )


def _deduplicate_loras(
    fragments: Iterable[SpatialPromptFragmentReceipt],
) -> tuple[SpatialLoraReceipt, ...]:
    result: list[SpatialLoraReceipt] = []
    seen: set[tuple[str, float]] = set()
    for fragment in fragments:
        for item in fragment.loras:
            key = (item.name, item.weight)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return tuple(result)


def _deduplicate_embeddings(
    fragments: Iterable[SpatialPromptFragmentReceipt],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        for item in fragment.embeddings:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return tuple(result)


def build_spatial_prompt_plan(
    snapshot: SpatialPromptSnapshot,
    *,
    profile_id: str,
    generation_seed: int,
    advanced_styles: Sequence[object] = (),
    hires_prompt: str = "",
    include_hires: bool = False,
    include_refiner: bool = False,
    wildcard_dir: Path | None = None,
    known_embedding_names: Sequence[str] = (),
    expansion_version: str = SPATIAL_FRAGMENT_EXPANSION_VERSION,
) -> SpatialPromptPlan:
    prompt_input = snapshot.prompt_input.root if snapshot.prompt_input is not None else None
    layout = snapshot.prompt_layout
    if (
        not isinstance(prompt_input, PromptInputV2)
        or prompt_input.mode != "sections"
        or layout is None
        or layout.mode != "enabled"
    ):
        raise SpatialPromptPlannerError("spatial_prompt_layout_not_enabled")
    if not profile_id.strip():
        raise SpatialPromptPlannerError("spatial_prompt_profile_required")
    if generation_seed < 0 or generation_seed > (2**64) - 1:
        raise SpatialPromptPlannerError("spatial_prompt_seed_invalid")

    global_parts: list[str] = []
    ordered_targets: list[tuple[str, str]] = []
    for section in sorted(prompt_input.sections or (), key=lambda item: item.order):
        if isinstance(section.content, GlobalPromptContent):
            if section.content.text:
                global_parts.append(section.content.text)
            continue
        if isinstance(section.content, TargetPromptContent):
            ordered_targets.extend(
                (target.id, target.text)
                for target in sorted(section.content.targets, key=lambda item: item.order)
            )

    global_raw = "\n".join(global_parts)
    excluded_ids = set(layout.excluded_target_ids)
    excluded_targets = tuple(
        SpatialExcludedTargetReceipt(target_id=target_id, reason="user_choice")
        for target_id, _ in ordered_targets
        if target_id in excluded_ids
    )
    base_global = _fragment_receipt(
        fragment_id="base:global",
        stage="base",
        kind="global",
        target_id=None,
        raw_prompt=global_raw,
        prepared_raw_prompt=_prepare_base_global(global_raw, advanced_styles),
        profile_id=profile_id,
        generation_seed=generation_seed,
        expansion_version=expansion_version,
        wildcard_dir=wildcard_dir,
        known_embedding_names=known_embedding_names,
    )
    base_targets = tuple(
        _fragment_receipt(
            fragment_id=f"base:target:{target_id}",
            stage="base",
            kind="target",
            target_id=target_id,
            raw_prompt=raw_prompt,
            prepared_raw_prompt=raw_prompt,
            profile_id=profile_id,
            generation_seed=generation_seed,
            expansion_version=expansion_version,
            wildcard_dir=wildcard_dir,
            known_embedding_names=known_embedding_names,
        )
        for target_id, raw_prompt in ordered_targets
        if target_id not in excluded_ids
    )
    base = _stage_receipt(
        stage="base",
        global_fragment=base_global,
        target_fragments=base_targets,
        excluded_targets=excluded_targets,
        reused_from_stage=None,
    )

    hires: SpatialStagePromptPlan | None = None
    if include_hires:
        normalized_hires = hires_prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized_hires:
            hires = _stage_receipt(
                stage="hires",
                global_fragment=base_global,
                target_fragments=base_targets,
                excluded_targets=excluded_targets,
                reused_from_stage="base",
            )
        else:
            hires_global = _fragment_receipt(
                fragment_id="hires:global",
                stage="hires",
                kind="global",
                target_id=None,
                raw_prompt=normalized_hires,
                prepared_raw_prompt=normalized_hires,
                profile_id=profile_id,
                generation_seed=generation_seed,
                expansion_version=expansion_version,
                wildcard_dir=wildcard_dir,
                known_embedding_names=known_embedding_names,
            )
            hires = _stage_receipt(
                stage="hires",
                global_fragment=hires_global,
                target_fragments=base_targets,
                excluded_targets=excluded_targets,
                reused_from_stage=None,
            )

    refiner: SpatialStagePromptPlan | None = None
    if include_refiner:
        source = hires or base
        refiner = _stage_receipt(
            stage="refiner",
            global_fragment=source.global_fragment,
            target_fragments=source.target_fragments,
            excluded_targets=source.excluded_targets,
            reused_from_stage="hires" if hires is not None else "base",
        )

    execution_fragments: list[SpatialPromptFragmentReceipt] = [base_global, *base_targets]
    if hires is not None and hires.reused_from_stage is None:
        execution_fragments.append(hires.global_fragment)
    payload: dict[str, Any] = {
        "schema_version": "asteria.spatial-prompt-plan.v1",
        "planner_revision": SPATIAL_PROMPT_PLANNER_REVISION,
        "profile_id": profile_id,
        "generation_seed": generation_seed,
        "expansion_version": expansion_version,
        "canonical_raw_prompt": snapshot.prompt,
        "canonical_raw_sha256": _sha256_text(snapshot.prompt),
        "layout_global_raw_prompt": global_raw,
        "layout_global_raw_sha256": _sha256_text(global_raw),
        "target_ids": [target_id for target_id, _ in ordered_targets],
        "base": base.model_dump(mode="json"),
        "hires": hires.model_dump(mode="json") if hires is not None else None,
        "refiner": refiner.model_dump(mode="json") if refiner is not None else None,
        "loras": [item.model_dump(mode="json") for item in _deduplicate_loras(execution_fragments)],
        "embeddings": list(_deduplicate_embeddings(execution_fragments)),
    }
    return SpatialPromptPlan.model_validate(
        {**payload, "plan_sha256": _sha256_json(payload)}
    )
