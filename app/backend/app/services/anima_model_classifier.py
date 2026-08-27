from __future__ import annotations

from app.schemas.media_model_catalog import (
    MediaModelClassification,
    MediaModelCompatibility,
    MediaModelComponentRole,
    MediaModelDomain,
)
from app.schemas.model_architecture import (
    AnimaArchitectureComponent,
    AnimaArchitectureVariant,
    ArchitectureDetectionStatus,
)
from app.services.media_model_catalog import SafetensorsHeader
from app.services.model_architecture_inspector import AnimaArchitectureInspector
from app.services.model_library import ModelLibraryItem


_DTYPE_NAMES = {
    "BF16": "bf16",
    "F16": "fp16",
    "F32": "fp32",
    "F64": "fp64",
    "F8_E4M3": "fp8_e4m3",
    "F8_E5M2": "fp8_e5m2",
    "I8": "int8",
}


class AnimaModelClassifier:
    domain = MediaModelDomain.IMAGE

    def __init__(self, inspector: AnimaArchitectureInspector | None = None) -> None:
        self._inspector = inspector or AnimaArchitectureInspector()

    def classify(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader | None,
    ) -> MediaModelClassification:
        if header is None:
            return self._unsupported("anima_model_header_unavailable")
        descriptor = self._inspector.inspect(header)
        if (
            descriptor.status == ArchitectureDetectionStatus.SUPPORTED
            and descriptor.component == AnimaArchitectureComponent.DIFFUSION_MODEL
            and descriptor.variant_id is not None
        ):
            if item.type != "checkpoint":
                return self._unsupported("anima_component_role_not_supported")
            required_roles = (MediaModelComponentRole.NATIVE_TEXT_ENCODER,)
            if descriptor.variant_id == AnimaArchitectureVariant.V3_8B_52:
                required_roles = (
                    MediaModelComponentRole.NATIVE_TEXT_ENCODER,
                    MediaModelComponentRole.QWEN35_TEXT_ENCODER,
                    MediaModelComponentRole.PROGRESSIVE_ADAPTER,
                )
            return MediaModelClassification(
                component_role=MediaModelComponentRole.DIFFUSION_MODEL,
                architecture="anima",
                variant=descriptor.variant_id.value,
                precision=self._precision(header),
                supported_tasks=("text_to_image", "image_to_image"),
                compatible_architecture_ids=("anima",),
                compatible_variant_ids=(descriptor.variant_id.value,),
                required_component_roles=required_roles,
                compatible_profile_ids=("anima",),
                compatibility=MediaModelCompatibility.SUPPORTED,
            )

        component_roles = {
            AnimaArchitectureComponent.NATIVE_TEXT_ENCODER: MediaModelComponentRole.NATIVE_TEXT_ENCODER,
            AnimaArchitectureComponent.QWEN35_TEXT_ENCODER: MediaModelComponentRole.QWEN35_TEXT_ENCODER,
            AnimaArchitectureComponent.PROGRESSIVE_ADAPTER: MediaModelComponentRole.PROGRESSIVE_ADAPTER,
        }
        component_role = component_roles.get(descriptor.component)
        if (
            descriptor.status == ArchitectureDetectionStatus.SUPPORTED
            and component_role is not None
        ):
            if item.type != "text_encoder":
                return self._unsupported("anima_component_role_not_supported")
            return MediaModelClassification(
                component_role=component_role,
                architecture="anima",
                variant=(
                    descriptor.variant_id.value
                    if descriptor.variant_id is not None
                    else None
                ),
                precision=self._precision(header),
                supported_tasks=("text_to_image", "image_to_image"),
                compatible_architecture_ids=("anima",),
                compatible_variant_ids=(
                    (descriptor.variant_id.value,)
                    if descriptor.variant_id is not None
                    else tuple(variant.value for variant in AnimaArchitectureVariant)
                ),
                compatible_profile_ids=("anima",),
                compatibility=MediaModelCompatibility.SUPPORTED,
            )

        compatibility = (
            MediaModelCompatibility.PROVISIONAL_UNSUPPORTED
            if descriptor.status
            == ArchitectureDetectionStatus.PROVISIONAL_UNSUPPORTED
            else MediaModelCompatibility.UNSUPPORTED
        )
        return MediaModelClassification(
            architecture="anima"
            if descriptor.diffusion_block_count is not None
            else "unknown",
            variant=(
                descriptor.variant_id.value
                if descriptor.variant_id is not None
                else None
            ),
            precision=self._precision(header),
            compatible_architecture_ids=("anima",)
            if descriptor.diffusion_block_count is not None
            else (),
            compatible_profile_ids=("anima",)
            if descriptor.diffusion_block_count is not None
            else (),
            compatibility=compatibility,
            reason_code=descriptor.unsupported_reason
            or "anima_component_role_not_supported",
        )

    @staticmethod
    def _precision(header: SafetensorsHeader) -> str | None:
        weights = {
            name: info
            for name, info in header.tensors.items()
            if name.endswith(".weight")
        }
        source = weights or header.tensors
        dtypes = sorted(
            {_DTYPE_NAMES.get(info.dtype, info.dtype.casefold()) for info in source.values()}
        )
        if not dtypes:
            return None
        return dtypes[0] if len(dtypes) == 1 else "mixed"

    @staticmethod
    def _unsupported(reason_code: str) -> MediaModelClassification:
        return MediaModelClassification(
            compatibility=MediaModelCompatibility.UNSUPPORTED,
            reason_code=reason_code,
        )
