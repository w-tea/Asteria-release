from __future__ import annotations

import hashlib
import json
import re

from app.schemas.model_architecture import (
    AnimaArchitectureComponent,
    AnimaArchitectureDescriptor,
    AnimaArchitectureVariant,
    ArchitectureDetectionStatus,
)
from app.services.media_model_catalog import SafetensorsHeader


_DIFFUSION_BLOCK = re.compile(r"^net\.blocks\.(\d+)\.")
_LLM_ADAPTER_BLOCK = re.compile(r"^net\.llm_adapter\.blocks\.(\d+)\.")
_NORMALIZED_DIFFUSION_BLOCK = re.compile(r"^net\.blocks\.\d+\.")
_NORMALIZED_LLM_ADAPTER_BLOCK = re.compile(
    r"^net\.llm_adapter\.blocks\.\d+\."
)
_VARIANTS: dict[int, tuple[AnimaArchitectureVariant, str]] = {
    28: (AnimaArchitectureVariant.BASE_28, "anima_base_28"),
    40: (AnimaArchitectureVariant.V2_9B_40, "anima_2_9b_40"),
    52: (AnimaArchitectureVariant.V3_8B_52, "anima_3_8b_52"),
}
_PROGRESSIVE_ADAPTER_SIGNATURE = "anima_progressive_qwen35_cross_adapter_v1"
_QWEN35_LAYER = re.compile(r"^layers\.(\d+)\.")
_QWEN35_SELF_ATTENTION = re.compile(r"^layers\.(\d+)\.self_attn\.")
_QWEN35_LINEAR_ATTENTION = re.compile(r"^layers\.(\d+)\.linear_attn\.")
_NATIVE_QWEN_LAYER = re.compile(r"^model\.layers\.(\d+)\.")


class AnimaArchitectureInspector:
    """Classify Anima assets from a bounded SafeTensors header only."""

    def inspect(self, header: SafetensorsHeader) -> AnimaArchitectureDescriptor:
        keys = tuple(sorted(header.tensors))
        if self._is_qwen35_text_encoder(header, keys):
            return AnimaArchitectureDescriptor(
                component=AnimaArchitectureComponent.QWEN35_TEXT_ENCODER,
                variant_id=AnimaArchitectureVariant.V3_8B_52,
                progressive_adapter_required=True,
                progressive_adapter_signature=_PROGRESSIVE_ADAPTER_SIGNATURE,
                text_encoder_requirements=("qwen3_5_4b",),
                key_layout_signature=self._layout_signature(keys, None, None),
                status=ArchitectureDetectionStatus.SUPPORTED,
            )
        if self._is_native_text_encoder(header, keys):
            return AnimaArchitectureDescriptor(
                component=AnimaArchitectureComponent.NATIVE_TEXT_ENCODER,
                text_encoder_requirements=("anima_native_qwen3_0_6b",),
                key_layout_signature=self._layout_signature(keys, None, None),
                status=ArchitectureDetectionStatus.SUPPORTED,
            )
        if self._is_progressive_adapter(header, keys):
            return AnimaArchitectureDescriptor(
                component=AnimaArchitectureComponent.PROGRESSIVE_ADAPTER,
                variant_id=AnimaArchitectureVariant.V3_8B_52,
                progressive_adapter_signature=_PROGRESSIVE_ADAPTER_SIGNATURE,
                text_encoder_requirements=("qwen3_5_4b",),
                key_layout_signature=self._layout_signature(keys, None, None),
                status=ArchitectureDetectionStatus.SUPPORTED,
            )

        diffusion_ordinals = self._ordinals(keys, _DIFFUSION_BLOCK)
        if not diffusion_ordinals:
            return self._unsupported("anima_signature_not_recognized")

        diffusion_count = max(diffusion_ordinals) + 1
        if diffusion_ordinals != set(range(diffusion_count)):
            return self._unsupported(
                "anima_diffusion_blocks_not_contiguous",
                diffusion_block_count=len(diffusion_ordinals),
            )

        if not self._has_complete_diffusion_structure(keys, diffusion_count):
            return self._unsupported(
                "anima_diffusion_structure_incomplete",
                diffusion_block_count=diffusion_count,
            )

        llm_ordinals = self._ordinals(keys, _LLM_ADAPTER_BLOCK)
        llm_count = max(llm_ordinals) + 1 if llm_ordinals else 0
        if llm_ordinals != set(range(llm_count)):
            return self._unsupported(
                "anima_llm_adapter_blocks_not_contiguous",
                diffusion_block_count=diffusion_count,
                llm_adapter_block_count=len(llm_ordinals),
            )
        if llm_count != 6:
            return self._unsupported(
                "anima_llm_adapter_block_count_unverified",
                status=ArchitectureDetectionStatus.PROVISIONAL_UNSUPPORTED,
                diffusion_block_count=diffusion_count,
                llm_adapter_block_count=llm_count,
            )

        variant = _VARIANTS.get(diffusion_count)
        if variant is None:
            return self._unsupported(
                "anima_diffusion_block_count_unverified",
                status=ArchitectureDetectionStatus.PROVISIONAL_UNSUPPORTED,
                diffusion_block_count=diffusion_count,
                llm_adapter_block_count=llm_count,
                key_layout_signature=self._layout_signature(
                    keys, diffusion_count, llm_count
                ),
            )

        variant_id, schema_id = variant
        requires_progressive_adapter = (
            variant_id == AnimaArchitectureVariant.V3_8B_52
        )
        text_encoders = (
            ("anima_native_qwen3_0_6b", "qwen3_5_4b")
            if requires_progressive_adapter
            else ("anima_native_qwen3_0_6b",)
        )
        return AnimaArchitectureDescriptor(
            component=AnimaArchitectureComponent.DIFFUSION_MODEL,
            variant_id=variant_id,
            diffusion_block_count=diffusion_count,
            llm_adapter_block_count=llm_count,
            progressive_adapter_required=requires_progressive_adapter,
            progressive_adapter_signature=(
                _PROGRESSIVE_ADAPTER_SIGNATURE
                if requires_progressive_adapter
                else None
            ),
            text_encoder_requirements=text_encoders,
            key_layout_signature=self._layout_signature(
                keys, diffusion_count, llm_count
            ),
            merge_schema_id=schema_id,
            status=ArchitectureDetectionStatus.SUPPORTED,
        )

    @staticmethod
    def _ordinals(keys: tuple[str, ...], pattern: re.Pattern[str]) -> set[int]:
        return {
            int(match.group(1))
            for key in keys
            if (match := pattern.match(key)) is not None
        }

    @staticmethod
    def _has_complete_diffusion_structure(
        keys: tuple[str, ...], block_count: int
    ) -> bool:
        key_set = set(keys)
        if not {
            "net.final_layer.linear.weight",
            "net.x_embedder.proj.1.weight",
        }.issubset(key_set):
            return False
        return all(
            {
                f"net.blocks.{ordinal}.cross_attn.k_proj.weight",
                f"net.blocks.{ordinal}.self_attn.q_proj.weight",
                f"net.blocks.{ordinal}.mlp.layer1.weight",
            }.issubset(key_set)
            for ordinal in range(block_count)
        )

    @staticmethod
    def _is_progressive_adapter(
        header: SafetensorsHeader, keys: tuple[str, ...]
    ) -> bool:
        declared = header.metadata.get("architecture", "").casefold()
        required_prefixes = (
            "query_norms.",
            "semantic_attentions.",
        )
        return (
            declared == _PROGRESSIVE_ADAPTER_SIGNATURE
            and "layer_mix_logits" in header.tensors
            and all(
                any(key.startswith(prefix) for key in keys)
                for prefix in required_prefixes
            )
            and not any(_DIFFUSION_BLOCK.match(key) for key in keys)
        )

    @staticmethod
    def _is_qwen35_text_encoder(
        header: SafetensorsHeader,
        keys: tuple[str, ...],
    ) -> bool:
        embedding = header.tensors.get("embed_tokens.weight")
        if embedding is None or tuple(embedding.shape) != (248320, 2560):
            return False
        layer_ordinals = AnimaArchitectureInspector._ordinals(keys, _QWEN35_LAYER)
        if layer_ordinals != set(range(32)):
            return False
        self_attention = AnimaArchitectureInspector._ordinals(
            keys, _QWEN35_SELF_ATTENTION
        )
        linear_attention = AnimaArchitectureInspector._ordinals(
            keys, _QWEN35_LINEAR_ATTENTION
        )
        return (
            self_attention == {3, 7, 11, 15, 19, 23, 27, 31}
            and linear_attention == set(range(32)).difference(self_attention)
        )

    @staticmethod
    def _is_native_text_encoder(
        header: SafetensorsHeader,
        keys: tuple[str, ...],
    ) -> bool:
        expected_global = {
            "model.embed_tokens.weight": (151936, 1024),
            "model.norm.weight": (1024,),
        }
        if any(
            name not in header.tensors
            or tuple(header.tensors[name].shape) != shape
            for name, shape in expected_global.items()
        ):
            return False
        layer_ordinals = AnimaArchitectureInspector._ordinals(
            keys,
            _NATIVE_QWEN_LAYER,
        )
        if layer_ordinals != set(range(28)):
            return False
        expected_layer_shapes = {
            "input_layernorm.weight": (1024,),
            "post_attention_layernorm.weight": (1024,),
            "self_attn.q_proj.weight": (2048, 1024),
            "self_attn.k_proj.weight": (1024, 1024),
            "self_attn.v_proj.weight": (1024, 1024),
            "self_attn.o_proj.weight": (1024, 2048),
            "mlp.gate_proj.weight": (3072, 1024),
            "mlp.up_proj.weight": (3072, 1024),
            "mlp.down_proj.weight": (1024, 3072),
        }
        return all(
            (
                tensor := header.tensors.get(f"model.layers.{ordinal}.{suffix}")
            )
            is not None
            and tuple(tensor.shape) == shape
            for ordinal in range(28)
            for suffix, shape in expected_layer_shapes.items()
        )

    @staticmethod
    def _layout_signature(
        keys: tuple[str, ...],
        diffusion_count: int | None,
        llm_count: int | None,
    ) -> str:
        normalized = sorted(
            {
                _NORMALIZED_LLM_ADAPTER_BLOCK.sub(
                    "net.llm_adapter.blocks.*.",
                    _NORMALIZED_DIFFUSION_BLOCK.sub("net.blocks.*.", key),
                )
                for key in keys
            }
        )
        payload = json.dumps(
            {
                "diffusion_block_count": diffusion_count,
                "llm_adapter_block_count": llm_count,
                "normalized_keys": normalized,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _unsupported(
        reason: str,
        *,
        status: ArchitectureDetectionStatus = ArchitectureDetectionStatus.UNSUPPORTED,
        diffusion_block_count: int | None = None,
        llm_adapter_block_count: int | None = None,
        key_layout_signature: str | None = None,
    ) -> AnimaArchitectureDescriptor:
        return AnimaArchitectureDescriptor(
            diffusion_block_count=diffusion_block_count,
            llm_adapter_block_count=llm_adapter_block_count,
            key_layout_signature=key_layout_signature,
            status=status,
            unsupported_reason=reason,
        )
