from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.schemas.model_architecture import (
    AnimaArchitectureComponent,
    AnimaArchitectureVariant,
    ArchitectureDetectionStatus,
)
from app.services.media_model_catalog import read_safetensors_header
from app.services.model_architecture_inspector import AnimaArchitectureInspector


class AnimaGenerationBundleError(ValueError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class AnimaGenerationBundleReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant_id: AnimaArchitectureVariant
    diffusion_layout_signature: str
    diffusion_block_count: int
    llm_adapter_block_count: int
    native_text_encoder_layout_signature: str
    qwen35_layout_signature: str | None = None
    progressive_adapter_signature: str | None = None
    progressive_adapter_layout_signature: str | None = None
    adapter_strength: float | None = None
    negative_adapter_strength: float | None = None


def _inspect_component(
    path: Path,
    *,
    expected: AnimaArchitectureComponent,
    unavailable_reason: str,
    incompatible_reason: str,
):
    try:
        descriptor = AnimaArchitectureInspector().inspect(
            read_safetensors_header(path)
        )
    except (OSError, ValueError) as exc:
        raise AnimaGenerationBundleError(unavailable_reason) from exc
    if (
        descriptor.status != ArchitectureDetectionStatus.SUPPORTED
        or descriptor.component != expected
        or not descriptor.key_layout_signature
    ):
        raise AnimaGenerationBundleError(incompatible_reason)
    return descriptor


def prepare_anima_generation_bundle(
    model_slots: dict[str, str],
    *,
    adapter_strength: float = 1.0,
    negative_adapter_strength: float = 0.0,
) -> AnimaGenerationBundleReceipt:
    diffusion_value = str(model_slots.get("diffusion_model") or "").strip()
    if not diffusion_value:
        raise AnimaGenerationBundleError("anima_diffusion_model_required")
    diffusion = _inspect_component(
        Path(diffusion_value),
        expected=AnimaArchitectureComponent.DIFFUSION_MODEL,
        unavailable_reason="anima_diffusion_model_header_unavailable",
        incompatible_reason="anima_diffusion_model_layout_unverified",
    )
    assert diffusion.variant_id is not None
    assert diffusion.key_layout_signature is not None
    assert diffusion.diffusion_block_count is not None
    assert diffusion.llm_adapter_block_count is not None

    native_text_encoder_value = str(model_slots.get("text_encoder") or "").strip()
    if not native_text_encoder_value:
        raise AnimaGenerationBundleError("anima_native_text_encoder_required")
    native_text_encoder = _inspect_component(
        Path(native_text_encoder_value),
        expected=AnimaArchitectureComponent.NATIVE_TEXT_ENCODER,
        unavailable_reason="anima_native_text_encoder_header_unavailable",
        incompatible_reason="anima_native_text_encoder_layout_unverified",
    )
    assert native_text_encoder.key_layout_signature is not None

    qwen35_signature: str | None = None
    adapter_signature: str | None = None
    adapter_layout_signature: str | None = None
    if diffusion.variant_id == AnimaArchitectureVariant.V3_8B_52:
        qwen35_value = str(model_slots.get("qwen35_text_encoder") or "").strip()
        adapter_value = str(model_slots.get("progressive_adapter") or "").strip()
        missing = [
            slot
            for slot, value in (
                ("qwen35_text_encoder", qwen35_value),
                ("progressive_adapter", adapter_value),
            )
            if not value
        ]
        if missing:
            raise AnimaGenerationBundleError(
                "anima_3_8b_components_required:" + ",".join(missing)
            )
        qwen35 = _inspect_component(
            Path(qwen35_value),
            expected=AnimaArchitectureComponent.QWEN35_TEXT_ENCODER,
            unavailable_reason="anima_qwen35_header_unavailable",
            incompatible_reason="anima_qwen35_layout_unverified",
        )
        adapter = _inspect_component(
            Path(adapter_value),
            expected=AnimaArchitectureComponent.PROGRESSIVE_ADAPTER,
            unavailable_reason="anima_progressive_adapter_header_unavailable",
            incompatible_reason="anima_progressive_adapter_layout_unverified",
        )
        qwen35_signature = qwen35.key_layout_signature
        adapter_signature = adapter.progressive_adapter_signature
        adapter_layout_signature = adapter.key_layout_signature
    elif model_slots.get("qwen35_text_encoder") or model_slots.get(
        "progressive_adapter"
    ):
        raise AnimaGenerationBundleError(
            "anima_3_8b_components_require_52_block_model"
        )

    return AnimaGenerationBundleReceipt(
        variant_id=diffusion.variant_id,
        diffusion_layout_signature=diffusion.key_layout_signature,
        diffusion_block_count=diffusion.diffusion_block_count,
        llm_adapter_block_count=diffusion.llm_adapter_block_count,
        native_text_encoder_layout_signature=(
            native_text_encoder.key_layout_signature
        ),
        qwen35_layout_signature=qwen35_signature,
        progressive_adapter_signature=adapter_signature,
        progressive_adapter_layout_signature=adapter_layout_signature,
        adapter_strength=(
            float(adapter_strength)
            if diffusion.variant_id == AnimaArchitectureVariant.V3_8B_52
            else None
        ),
        negative_adapter_strength=(
            float(negative_adapter_strength)
            if diffusion.variant_id == AnimaArchitectureVariant.V3_8B_52
            else None
        ),
    )
