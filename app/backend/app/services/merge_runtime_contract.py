from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.merge_recipes import (
    CALCULATION_MODE_COMPATIBILITY,
    MERGE_MODES,
    MergeRecipe,
    expand_dice_recipe,
)


MERGE_RUNTIME_CONTRACT_VERSION = "asteria.merge.runtime.v1"


class MergeRuntimeContractError(ValueError):
    pass


class MergeRuntimeExportControls(BaseModel):
    model_config = ConfigDict(frozen=True)

    prune: bool
    fp16: bool
    save_metadata: bool
    copy_config: bool
    overwrite: bool
    safetensors: bool
    save_model: bool
    debug: bool


class MergeRuntimeRecipeContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    contract_version: Literal["asteria.merge.runtime.v1"] = MERGE_RUNTIME_CONTRACT_VERSION
    recipe_id: str
    recipe_name: str
    model_family: Literal["sdxl", "anima"]
    merge_mode: Literal["weight_sum", "add_difference", "triple_sum", "sum_twice"]
    calculation_mode: str
    calculation_option: float
    block_schema: Literal["sdxl", "anima"]
    alpha: float
    beta: float
    block_weights: str
    block_weights_alpha: str
    block_weights_beta: str
    include_blocks: list[str] = Field(default_factory=list)
    exclude_blocks: list[str] = Field(default_factory=list)
    pinpoint_blocks: list[str] = Field(default_factory=list)
    pinpoint_elements: list[str] = Field(default_factory=list)
    pinpoint_adjust: str = ""
    elemental_rules: str = ""
    adjust_values: dict[str, float] = Field(default_factory=dict)
    random_seed: int | None = None
    dice_ranges: dict[str, str] = Field(default_factory=dict)
    use_old_calc_method: bool = False
    requires_source_c: bool = False
    export_controls: MergeRuntimeExportControls


def normalize_merge_runtime_recipe(recipe: MergeRecipe) -> MergeRuntimeRecipeContract:
    if recipe.merge_type != "checkpoint":
        raise MergeRuntimeContractError("merge_type must be checkpoint for in-memory runtime")
    if recipe.model_family not in {"sdxl", "anima"}:
        raise MergeRuntimeContractError("model_family is not supported by in-memory runtime")
    if recipe.block_schema != recipe.model_family:
        raise MergeRuntimeContractError("block_schema must match model_family")
    if recipe.reset_clip_ids:
        raise MergeRuntimeContractError("reset_clip_ids is not implemented by in-memory runtime")
    if recipe.bake_in_vae.strip():
        raise MergeRuntimeContractError("bake_in_vae is not implemented by in-memory runtime")

    expanded = expand_dice_recipe(recipe)
    merge_mode = expanded.normalized_merge_mode()
    calculation_mode = expanded.normalized_calc_mode()
    if merge_mode not in MERGE_MODES:
        raise MergeRuntimeContractError("merge_mode is not supported by in-memory runtime")
    if calculation_mode not in CALCULATION_MODE_COMPATIBILITY.get(merge_mode, []):
        raise MergeRuntimeContractError("calculation_mode is incompatible with merge_mode")
    requires_source_c = bool(MERGE_MODES[merge_mode]["requires_c"]) or calculation_mode in {
        "trainDifference",
        "smoothAdd",
        "smoothAdd MT",
        "extract",
    }

    return MergeRuntimeRecipeContract(
        recipe_id=expanded.id,
        recipe_name=expanded.name,
        model_family=expanded.model_family,
        merge_mode=merge_mode,
        calculation_mode=calculation_mode,
        calculation_option=expanded.calculation_option,
        block_schema=expanded.block_schema,
        alpha=expanded.alpha,
        beta=expanded.beta,
        block_weights=expanded.block_weights,
        block_weights_alpha=expanded.block_weights_alpha,
        block_weights_beta=expanded.block_weights_beta,
        include_blocks=list(expanded.include_blocks),
        exclude_blocks=list(expanded.exclude_blocks),
        pinpoint_blocks=list(expanded.pinpoint_blocks),
        pinpoint_elements=list(expanded.pinpoint_elements),
        pinpoint_adjust=expanded.pinpoint_adjust,
        elemental_rules=expanded.elemental_rules,
        adjust_values=dict(expanded.adjust_values),
        random_seed=expanded.dice_seed,
        dice_ranges=dict(expanded.dice_ranges),
        use_old_calc_method=expanded.use_old_calc_method,
        requires_source_c=requires_source_c,
        export_controls=MergeRuntimeExportControls(
            prune=expanded.prune,
            fp16=expanded.fp16,
            save_metadata=expanded.save_metadata,
            copy_config=expanded.copy_config,
            overwrite=expanded.overwrite,
            safetensors=expanded.safetensors,
            save_model=expanded.save_model,
            debug=expanded.debug,
        ),
    )
