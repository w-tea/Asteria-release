from __future__ import annotations

import json
import re

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


class VideoModelClassifier:
    domain = MediaModelDomain.VIDEO

    def classify(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader | None,
    ) -> MediaModelClassification:
        if header is None:
            return self._unsupported("video_model_header_unavailable")
        if item.type == "checkpoint":
            return self._classify_diffusion(item, header)
        if item.type == "text_encoder":
            return self._classify_text_encoder(header)
        if item.type == "vae":
            return self._classify_vae(item, header)
        if item.type == "lora":
            return self._classify_lora(item, header)
        if item.type == "upscaler":
            return self._classify_upscaler(header)
        return self._unsupported("video_component_role_not_supported")

    def _classify_diffusion(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        keys = set(header.tensors)
        required_keys = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
            "token_refiner.blocks.0.attn.qkv_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
        }
        config = self._metadata_json(header, "config")
        transformer = config.get("transformer") if isinstance(config, dict) else None
        is_h3_config = isinstance(transformer, dict) and transformer.get("image_model") == "minimax_h3"
        is_managed_h3_structure = self._has_shapes(
            header,
            {
                "adaln_t_table": (1025, 8),
                "video_patch_proj.weight": (5376, 96),
                "audio_patch_proj.weight": (5376, 32),
                "condition_proj.weight": (5376, 5120),
                "final_layer.video_out.weight": (96, 5376),
                "final_layer.audio_out.weight": (32, 5376),
                "token_refiner.blocks.0.attn.qkv_proj.weight": (21504, 5376),
                "blocks.0.attn.qkv_proj.weight": (21504, 5376),
            },
        )
        if not required_keys.issubset(keys) or not (
            is_h3_config or is_managed_h3_structure
        ):
            return self._unsupported("video_model_signature_not_recognized")
        variant = self._declared_variant(item, header)
        if variant == "ref2va":
            variant = "ref2va"
            tasks = ("reference_to_video", "sequence")
        elif variant == "fl2va":
            variant = "fl2va"
            tasks = ("text_to_video", "image_to_video", "first_last_to_video", "sequence")
        else:
            return MediaModelClassification(
                component_role=MediaModelComponentRole.DIFFUSION_MODEL,
                architecture="minimax_h3",
                precision=self._precision(header),
                quantization=self._quantization(item.name),
                compatible_architecture_ids=("minimax_h3",),
                compatible_profile_ids=("minimax-h3",),
                compatibility=MediaModelCompatibility.PROVISIONAL_UNSUPPORTED,
                reason_code="minimax_h3_variant_unresolved",
            )
        return MediaModelClassification(
            component_role=MediaModelComponentRole.DIFFUSION_MODEL,
            architecture="minimax_h3",
            variant=variant,
            precision=self._precision(header),
            quantization=self._quantization(item.name),
            supported_tasks=tasks,
            compatible_architecture_ids=("minimax_h3",),
            compatible_variant_ids=(variant,),
            required_component_roles=(
                MediaModelComponentRole.TEXT_ENCODER,
                MediaModelComponentRole.VIDEO_VAE,
                MediaModelComponentRole.AUDIO_VAE,
            ),
            compatible_profile_ids=("minimax-h3",),
            compatibility=MediaModelCompatibility.SUPPORTED,
        )

    def _classify_text_encoder(self, header: SafetensorsHeader) -> MediaModelClassification:
        prefixes = (
            "text_encoders.qwen3vl_32b.transformer.",
            "qwen3vl_32b.transformer.",
        )
        legacy_signature = any(key.startswith(prefixes) for key in header.tensors)
        managed_signature = self._has_shapes(
            header,
            {
                "model.embed_tokens.weight": (151936, 5120),
                "visual.patch_embed.proj.weight": (1152, 3, 2, 16, 16),
                "visual.blocks.0.attn.qkv.weight": (3456, 1152),
                "visual.merger.linear_fc2.weight": (5120, 4608),
            },
        )
        if not (legacy_signature or managed_signature):
            return self._unsupported("video_text_encoder_signature_not_recognized")
        return MediaModelClassification(
            component_role=MediaModelComponentRole.TEXT_ENCODER,
            architecture="minimax_h3",
            precision=self._precision(header),
            compatible_architecture_ids=("minimax_h3",),
            compatible_variant_ids=("fl2va", "ref2va"),
            compatible_profile_ids=("minimax-h3",),
            compatibility=MediaModelCompatibility.SUPPORTED,
            supported_tasks=(
                "text_to_video",
                "image_to_video",
                "first_last_to_video",
                "reference_to_video",
                "sequence",
            ),
        )

    def _classify_vae(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        keys = set(header.tensors)
        video_metadata = self._metadata_json(header, "minimax_h3_video_vae")
        audio_metadata = self._metadata_json(header, "minimax_h3_audio_vae")
        video_evidence = (
            isinstance(video_metadata, dict)
            and isinstance(video_metadata.get("source_config"), dict)
            and video_metadata["source_config"].get("z_channels") == 24
            and {"latents_mean", "decoder.mask_token", "decoder.proj_out.weight"}.issubset(keys)
            and header.tensors["latents_mean"].shape == (24,)
        )
        audio_evidence = (
            isinstance(audio_metadata, dict)
            and audio_metadata.get("sample_rate") == 32000
            and audio_metadata.get("latent_channels") == 32
            and {"latents_mean", "dec_in_proj.weight", "decoder.conv_pre.weight"}.issubset(keys)
            and header.tensors["latents_mean"].shape == (32,)
        )
        role = (
            MediaModelComponentRole.VIDEO_VAE
            if video_evidence
            else MediaModelComponentRole.AUDIO_VAE
            if audio_evidence
            else None
        )
        if role is None:
            return self._unsupported("video_vae_signature_not_recognized")
        return MediaModelClassification(
            component_role=role,
            architecture="minimax_h3",
            precision=self._precision(header),
            compatible_architecture_ids=("minimax_h3",),
            compatible_variant_ids=("fl2va", "ref2va"),
            compatible_profile_ids=("minimax-h3",),
            compatibility=MediaModelCompatibility.SUPPORTED,
            supported_tasks=(
                "text_to_video",
                "image_to_video",
                "first_last_to_video",
                "reference_to_video",
                "sequence",
                "upscale",
            ),
        )

    def _classify_lora(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader,
    ) -> MediaModelClassification:
        architecture = header.metadata.get("modelspec.architecture", "").casefold()
        has_h3_lora_tensors = any(
            "lora_" in key.casefold()
            and "blocks_" in key.casefold()
            and "attn_qkv_proj" in key.casefold()
            for key in header.tensors
        )
        filename_variant = self._lora_filename_variant(item.name)
        if not has_h3_lora_tensors or (
            architecture != "minimax_h3" and filename_variant is None
        ):
            return self._unsupported("video_lora_signature_not_recognized")
        declared_variant = header.metadata.get("modelspec.variant", "").casefold()
        if declared_variant not in {"turbo", "fl2va", "ref2va"}:
            declared_variant = filename_variant or ""
        if declared_variant not in {"turbo", "fl2va", "ref2va"}:
            return MediaModelClassification(
                component_role=MediaModelComponentRole.LORA,
                architecture="minimax_h3",
                precision=self._precision(header),
                compatible_architecture_ids=("minimax_h3",),
                compatible_profile_ids=("minimax-h3",),
                compatibility=MediaModelCompatibility.PROVISIONAL_UNSUPPORTED,
                reason_code="video_lora_variant_unresolved",
            )
        variant = declared_variant
        compatible_variants = ("fl2va",) if variant == "turbo" else (variant,)
        supported_tasks = (
            ("reference_to_video", "sequence")
            if variant == "ref2va"
            else ("text_to_video", "image_to_video", "first_last_to_video", "sequence")
        )
        return MediaModelClassification(
            component_role=MediaModelComponentRole.LORA,
            architecture="minimax_h3",
            variant=variant,
            precision=self._precision(header),
            compatible_architecture_ids=("minimax_h3",),
            compatible_variant_ids=compatible_variants,
            compatible_profile_ids=("minimax-h3",),
            compatibility=MediaModelCompatibility.SUPPORTED,
            supported_tasks=supported_tasks,
        )

    def _classify_upscaler(self, header: SafetensorsHeader) -> MediaModelClassification:
        architecture = header.metadata.get("asteria.architecture", "").casefold()
        component = header.metadata.get("asteria.component", "").casefold()
        has_declared_signature = (
            architecture == "minimax_h3"
            and component == "latent_upscaler"
            and any(key.startswith("latent_upscaler.") for key in header.tensors)
        )
        # Official LBH-123-AI checkpoints do not publish architecture metadata.
        # Identify their bounded SafeTensors structure instead of trusting the
        # filename. These shapes were verified against the published fp16
        # checkpoint at Hugging Face revision 13ccf95d85d120bdbc92c05b1247a6e147bf54bf.
        has_official_3d_signature = self._has_shapes(
            header,
            {
                "conv_in.weight": (512, 24, 3, 3, 3),
                "conv_out.weight": (24, 512, 3, 3, 3),
                "embed.0.weight": (64, 1),
                "in_blocks.0.in_layers.2.weight": (512, 512, 3, 3, 3),
                "in_blocks.1.dwconv.weight": (512, 1, 5, 1, 1),
                "out_blocks.0.in_layers.2.weight": (512, 512, 3, 3, 3),
            },
        ) and all(
            any(key.startswith(f"{side}.{index}.") for key in header.tensors)
            for side in ("in_blocks", "out_blocks")
            for index in range(13)
        )
        if not (has_declared_signature or has_official_3d_signature):
            return self._unsupported("video_latent_upscaler_signature_not_recognized")
        return MediaModelClassification(
            component_role=MediaModelComponentRole.LATENT_UPSCALER,
            architecture="minimax_h3",
            precision=self._precision(header),
            compatible_architecture_ids=("minimax_h3",),
            compatible_variant_ids=("fl2va", "ref2va"),
            required_component_roles=(
                MediaModelComponentRole.VIDEO_VAE,
                MediaModelComponentRole.AUDIO_VAE,
            ),
            compatible_profile_ids=("minimax-h3",),
            compatibility=MediaModelCompatibility.SUPPORTED,
            supported_tasks=("upscale",),
        )

    @staticmethod
    def _precision(header: SafetensorsHeader) -> str | None:
        dtypes = sorted({_DTYPE_NAMES.get(info.dtype, info.dtype.casefold()) for info in header.tensors.values()})
        if not dtypes:
            return None
        return dtypes[0] if len(dtypes) == 1 else "mixed"

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
    def _quantization(filename: str) -> str | None:
        folded = filename.casefold()
        for token in ("int8_convrot", "fp8_scaled", "nvfp4_awq", "int4_convrot"):
            if token in folded:
                return token
        return None

    @staticmethod
    def _metadata_json(header: SafetensorsHeader, key: str) -> object:
        raw = header.metadata.get(key)
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            return {}

    @classmethod
    def _declared_variant(
        cls,
        item: ModelLibraryItem,
        header: SafetensorsHeader,
    ) -> str | None:
        explicit = header.metadata.get("asteria.variant", "").casefold()
        if explicit in {"fl2va", "ref2va"}:
            return explicit
        match = re.search(r"(?:^|_)minimax_h3_(fl2va|ref2va)(?:_|\.)", item.name.casefold())
        return match.group(1) if match else None

    @staticmethod
    def _lora_filename_variant(filename: str) -> str | None:
        folded = filename.casefold()
        if re.search(r"(?:^|_)minimax_h3_(?:fl2v|fl2va)_turbo(?:_|\.)", folded):
            return "turbo"
        if re.search(r"(?:^|_)minimax_h3_fl2va(?:_|\.)", folded):
            return "fl2va"
        if re.search(r"(?:^|_)minimax_h3_ref2va(?:_|\.)", folded):
            return "ref2va"
        return None

    @staticmethod
    def _unsupported(reason_code: str) -> MediaModelClassification:
        return MediaModelClassification(
            compatibility=MediaModelCompatibility.UNSUPPORTED,
            reason_code=reason_code,
        )
