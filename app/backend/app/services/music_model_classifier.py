from __future__ import annotations

from app.schemas.media_model_catalog import (
    MediaModelClassification,
    MediaModelCompatibility,
    MediaModelComponentRole,
    MediaModelDomain,
)
from app.services.media_model_catalog import SafetensorsHeader
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
_ARCHITECTURE = "minimax_music3"
_PROFILE_ID = "minimax-music3"
_SUPPORTED_TASKS = ("vocal", "instrumental")


class MusicModelClassifier:
    domain = MediaModelDomain.MUSIC

    def classify(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader | None,
    ) -> MediaModelClassification:
        if header is None:
            return self._unsupported("music_model_header_unavailable")
        if item.type == "checkpoint":
            return self._classify_diffusion(header)
        if item.type == "text_encoder":
            return self._classify_text_encoder(header)
        if item.type == "vae":
            return self._classify_dav(header)
        return self._unsupported("music_component_role_not_supported")

    def _classify_diffusion(
        self,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        quantized_weights = (
            "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight",
            "diffusion_transformer.transformer.layers.35.self_attn.to_qkv.weight",
        )
        if not self._has_shapes(
            header,
            {
                "diffusion_transformer.preprocess_conv.weight": (2304, 2304, 1),
                "diffusion_transformer.transformer.project_out.weight": (128, 2048),
                "cond_layer_logits": (8,),
                quantized_weights[0]: (6144, 2048),
                quantized_weights[1]: (6144, 2048),
            },
        ):
            return self._unsupported("music_diffusion_signature_not_recognized")
        quantization = self._quantization(header, quantized_weights)
        if quantization is False:
            return self._unsupported("music_diffusion_quantization_unrecognized")
        return self._supported(
            role=MediaModelComponentRole.DIFFUSION_MODEL,
            header=header,
            quantization=quantization or None,
            required_component_roles=(
                MediaModelComponentRole.TEXT_ENCODER,
                MediaModelComponentRole.AUDIO_VAE,
            ),
        )

    def _classify_text_encoder(
        self,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        full_weights = (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.35.self_attn.q_proj.weight",
            "model.audio_decoder.layers.0.self_attn.q_proj.weight",
        )
        pruned_weights = (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.35.self_attn.qkv_proj.weight",
            "model.audio_decoder.layers.0.self_attn.qkv_proj.weight",
        )
        full_signature = self._has_shapes(
            header,
            {
                full_weights[0]: (4096, 4096),
                full_weights[1]: (4096, 4096),
                full_weights[2]: (4096, 4096),
                "model.lm_head.weight": (200000, 4096),
            },
        )
        pruned_signature = self._has_shapes(
            header,
            {
                pruned_weights[0]: (6144, 4096),
                pruned_weights[1]: (6144, 4096),
                "model.layers.0.mlp.gate_up_proj.weight": (24576, 4096),
                "model.layers.35.mlp.gate_up_proj.weight": (24576, 4096),
                pruned_weights[2]: (12288, 4096),
                "model.lm_head_pruned.weight": (16385, 4096),
            },
        )
        if not (full_signature or pruned_signature):
            return self._unsupported("music_text_encoder_signature_not_recognized")
        quantization = self._quantization(
            header,
            pruned_weights if pruned_signature else full_weights,
        )
        if quantization is False:
            return self._unsupported("music_text_encoder_quantization_unrecognized")
        return self._supported(
            role=MediaModelComponentRole.TEXT_ENCODER,
            header=header,
            quantization=quantization or None,
        )

    def _classify_dav(
        self,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        if not self._has_shapes(
            header,
            {
                "dec_in_proj.weight": (1024, 64, 1),
                "decoder.model.0.weight_v": (1536, 1024, 7),
                "decoder.model.6.weight_v": (1, 96, 7),
                "decoder.model.6.weight_g": (1, 1, 1),
            },
        ):
            return self._unsupported("music_dav_signature_not_recognized")
        return self._supported(
            role=MediaModelComponentRole.AUDIO_VAE,
            header=header,
            quantization=None,
        )

    @classmethod
    def _supported(
        cls,
        *,
        role: MediaModelComponentRole,
        header: SafetensorsHeader,
        quantization: str | None,
        required_component_roles: tuple[MediaModelComponentRole, ...] = (),
    ) -> MediaModelClassification:
        return MediaModelClassification(
            component_role=role,
            architecture=_ARCHITECTURE,
            precision=cls._precision(header),
            quantization=quantization,
            supported_tasks=_SUPPORTED_TASKS,
            compatible_architecture_ids=(_ARCHITECTURE,),
            required_component_roles=required_component_roles,
            compatible_profile_ids=(_PROFILE_ID,),
            compatibility=MediaModelCompatibility.SUPPORTED,
        )

    @staticmethod
    def _has_shapes(
        header: SafetensorsHeader,
        expected: dict[str, tuple[int, ...]],
    ) -> bool:
        return all(
            name in header.tensors and header.tensors[name].shape == shape
            for name, shape in expected.items()
        )

    @staticmethod
    def _precision(header: SafetensorsHeader) -> str | None:
        # Tokenizer/config byte payloads are stored as U8 tensors in the
        # official text encoders. They are not model-weight precision and must
        # not turn a BF16 encoder into a misleading `mixed` descriptor.
        weight_tensors = {
            name: info
            for name, info in header.tensors.items()
            if name.endswith(".weight")
        }
        source = weight_tensors or header.tensors
        dtypes = sorted(
            {_DTYPE_NAMES.get(info.dtype, info.dtype.casefold()) for info in source.values()}
        )
        if not dtypes:
            return None
        return dtypes[0] if len(dtypes) == 1 else "mixed"

    @staticmethod
    def _quantization(
        header: SafetensorsHeader,
        core_weights: tuple[str, ...],
    ) -> str | bool | None:
        if not any(info.dtype == "I8" for info in header.tensors.values()):
            return None
        if all(
            header.tensors.get(weight) is not None
            and header.tensors[weight].dtype == "I8"
            and header.tensors.get(f"{weight.removesuffix('.weight')}.weight_scale")
            is not None
            and header.tensors[
                f"{weight.removesuffix('.weight')}.weight_scale"
            ].dtype
            == "F32"
            and header.tensors.get(f"{weight.removesuffix('.weight')}.comfy_quant")
            is not None
            and header.tensors[
                f"{weight.removesuffix('.weight')}.comfy_quant"
            ].dtype
            == "U8"
            for weight in core_weights
        ):
            return "int8_convrot"
        return False

    @staticmethod
    def _unsupported(reason_code: str) -> MediaModelClassification:
        return MediaModelClassification(
            compatibility=MediaModelCompatibility.UNSUPPORTED,
            reason_code=reason_code,
        )
