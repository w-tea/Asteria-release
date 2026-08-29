from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Mapping

from app.schemas.spatial_prompt_runtime import (
    SpatialModelProfileReceipt,
    canonical_runtime_sha256,
)
from app.services.media_model_catalog import SafetensorsHeader
from app.services.spatial_model_profiles import (
    SpatialModelProfileDefinition,
    load_spatial_model_profiles,
)


SpatialProfileStage = Literal["base", "hires", "refiner"]
_ANIMA_SPATIAL_LAYOUT_TENSOR = re.compile(
    r"^net\.(?:"
    r"x_embedder\.proj\.1|"
    r"final_layer\.linear|"
    r"blocks\.\d+\.(?:cross_attn\.k_proj|self_attn\.q_proj|mlp\.layer1)|"
    r"llm_adapter\.blocks\.\d+\.(?:cross_attn\.k_proj|self_attn\.q_proj)"
    r")\.weight$"
)


@dataclass(frozen=True, slots=True)
class SpatialModelEvidence:
    model_slot_id: str
    architecture: str
    variant: str
    encoder_kind: str
    verified_structure_id: str
    layout_fingerprint: str
    structure_fingerprint: str
    diffusion_block_count: int

    def __post_init__(self) -> None:
        if not self.model_slot_id.strip():
            raise ValueError("spatial model slot id is required")
        if not self.architecture.strip() or not self.variant.strip():
            raise ValueError("spatial model architecture evidence is required")
        if not self.encoder_kind.strip():
            raise ValueError("spatial model encoder evidence is required")
        if not self.verified_structure_id.strip():
            raise ValueError("verified spatial structure id is required")
        if len(self.layout_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.layout_fingerprint
        ):
            raise ValueError("spatial layout fingerprint must be a lowercase sha256")
        if len(self.structure_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.structure_fingerprint
        ):
            raise ValueError("spatial structure fingerprint must be a lowercase sha256")
        if self.diffusion_block_count < 1:
            raise ValueError("spatial diffusion block count must be positive")


def spatial_structure_fingerprint(header: SafetensorsHeader) -> str:
    payload = {
        "metadata": {
            key: header.metadata[key]
            for key in sorted(header.metadata)
            if key.startswith("architecture") or key.startswith("model_type")
        },
        "tensors": [
            {
                "name": name,
                "dtype": info.dtype,
                "shape": list(info.shape),
            }
            for name, info in sorted(header.tensors.items())
        ],
    }
    return canonical_runtime_sha256(payload)


def spatial_layout_fingerprint(
    header: SafetensorsHeader,
    *,
    architecture: Literal["sdxl", "anima"],
) -> str:
    def include(name: str) -> bool:
        if architecture == "sdxl":
            return name.startswith("model.diffusion_model.")
        return _ANIMA_SPATIAL_LAYOUT_TENSOR.match(name) is not None

    layout = [
        {"name": name, "shape": list(info.shape)}
        for name, info in sorted(header.tensors.items())
        if include(name)
    ]
    if not layout:
        raise ValueError("spatial model layout is missing")
    return canonical_runtime_sha256(layout)


def sdxl_spatial_model_evidence(
    header: SafetensorsHeader,
    *,
    model_slot_id: str,
    variant: Literal["sdxl", "illustrious"],
) -> SpatialModelEvidence:
    """Build SDXL-family evidence from the resolved header, never its filename."""
    return SpatialModelEvidence(
        model_slot_id=model_slot_id,
        architecture="sdxl",
        variant=variant,
        encoder_kind="clip",
        verified_structure_id="sdxl_unet_v1",
        layout_fingerprint=spatial_layout_fingerprint(
            header,
            architecture="sdxl",
        ),
        structure_fingerprint=spatial_structure_fingerprint(header),
        diffusion_block_count=24,
    )


def _matches(definition: SpatialModelProfileDefinition, evidence: SpatialModelEvidence) -> bool:
    return (
        evidence.architecture in definition.architectures
        and evidence.variant in definition.variants
        and evidence.encoder_kind in definition.encoder_kinds
        and evidence.verified_structure_id in definition.verified_structure_ids
        and evidence.layout_fingerprint in definition.layout_fingerprints
        and evidence.diffusion_block_count in definition.diffusion_block_counts
    )


def _receipt(
    *,
    evidence: SpatialModelEvidence,
    stage: SpatialProfileStage,
    target_ids: tuple[str, ...],
    profile_id: str,
    revision: str,
    model_family: str,
    support_status: str,
    selection_mode: str,
    reason_code: str | None,
) -> SpatialModelProfileReceipt:
    payload = {
        "schema_version": "asteria.spatial-model-profile.v1",
        "stage": stage,
        "profile_id": profile_id,
        "profile_revision": revision,
        "model_family": model_family,
        "model_slot_id": evidence.model_slot_id,
        "encoder_kind": evidence.encoder_kind,
        "verified_structure_id": evidence.verified_structure_id,
        "layout_fingerprint": evidence.layout_fingerprint,
        "structure_fingerprint": evidence.structure_fingerprint,
        "support_status": support_status,
        "selection_mode": selection_mode,
        "reason_code": reason_code,
        "target_ids": list(target_ids),
    }
    return SpatialModelProfileReceipt.model_validate(
        {**payload, "receipt_sha256": canonical_runtime_sha256(payload)}
    )


def resolve_spatial_model_profile(
    evidence: SpatialModelEvidence,
    *,
    stage: SpatialProfileStage,
    target_ids: tuple[str, ...],
    profiles_path: str | Path,
    manual_profile_id: str | None = None,
) -> SpatialModelProfileReceipt:
    catalog = load_spatial_model_profiles(profiles_path)
    by_id = {item.profile_id: item for item in catalog.profiles}
    selection_mode = "manual" if manual_profile_id else "auto"
    if manual_profile_id is not None:
        definition = by_id.get(manual_profile_id)
        if definition is None:
            return _receipt(
                evidence=evidence,
                stage=stage,
                target_ids=target_ids,
                profile_id=manual_profile_id,
                revision="0",
                model_family=evidence.architecture or "unknown",
                support_status="unsupported",
                selection_mode="manual",
                reason_code="spatial_profile_unknown",
            )
        if not definition.manual_selectable or not _matches(definition, evidence):
            return _receipt(
                evidence=evidence,
                stage=stage,
                target_ids=target_ids,
                profile_id=definition.profile_id,
                revision=definition.revision,
                model_family=definition.model_family,
                support_status="unsupported",
                selection_mode="manual",
                reason_code="spatial_profile_evidence_mismatch",
            )
        return _receipt(
            evidence=evidence,
            stage=stage,
            target_ids=target_ids,
            profile_id=definition.profile_id,
            revision=definition.revision,
            model_family=definition.model_family,
            support_status="supported",
            selection_mode="manual",
            reason_code=None,
        )

    matches = [item for item in catalog.profiles if _matches(item, evidence)]
    if len(matches) != 1:
        reason = "spatial_profile_ambiguous" if matches else "spatial_profile_not_found"
        return _receipt(
            evidence=evidence,
            stage=stage,
            target_ids=target_ids,
            profile_id="unresolved",
            revision="0",
            model_family=evidence.architecture or "unknown",
            support_status="unsupported",
            selection_mode=selection_mode,
            reason_code=reason,
        )
    definition = matches[0]
    reason_code = {
        "selection_required": "spatial_profile_manual_selection_required",
        "unsupported": "spatial_profile_unsupported",
    }.get(definition.automatic_status)
    return _receipt(
        evidence=evidence,
        stage=stage,
        target_ids=target_ids,
        profile_id=definition.profile_id,
        revision=definition.revision,
        model_family=definition.model_family,
        support_status=definition.automatic_status,
        selection_mode=selection_mode,
        reason_code=reason_code,
    )


def resolve_spatial_stage_profiles(
    evidence_by_stage: Mapping[SpatialProfileStage, SpatialModelEvidence],
    *,
    target_ids: tuple[str, ...],
    profiles_path: str | Path,
    manual_profiles: Mapping[SpatialProfileStage, str] | None = None,
) -> tuple[SpatialModelProfileReceipt, ...]:
    unknown_stages = set(evidence_by_stage).difference({"base", "hires", "refiner"})
    if unknown_stages:
        raise ValueError("unknown spatial profile stage")
    manual_profiles = manual_profiles or {}
    return tuple(
        resolve_spatial_model_profile(
            evidence_by_stage[stage],
            stage=stage,
            target_ids=target_ids,
            profiles_path=profiles_path,
            manual_profile_id=manual_profiles.get(stage),
        )
        for stage in ("base", "hires", "refiner")
        if stage in evidence_by_stage
    )
