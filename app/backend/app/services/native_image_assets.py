"""Content-verified assets for the two bounded native image profiles."""
from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig
from app.schemas.media_model_catalog import (
    MediaModelClassification, MediaModelCompatibility, MediaModelComponentRole, MediaModelDomain,
)
from app.services.comfy_native_image_families import NATIVE_IMAGE_DEFAULTS
from app.services.media_model_catalog import SafetensorsHeader, SafetensorsHeaderError, read_safetensors_header
from app.services.model_dependencies import ModelDependency, ModelDependencyRepository
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibraryItem


_SLOTS = {"diffusion_model": "checkpoint", "text_encoder": "text_encoder", "vae": "vae"}
_ROLES = {"diffusion_model": MediaModelComponentRole.DIFFUSION_MODEL,
          "text_encoder": MediaModelComponentRole.TEXT_ENCODER, "vae": MediaModelComponentRole.IMAGE_VAE}
_VARIANTS = {"krea2": "krea2_turbo_fp8_scaled", "moss_image": "moss_image_sft_bf16"}
_COMPONENT_VARIANTS = {
    "moss-image-sft-int8-convrot": "moss_image_sft_int8_convrot",
    "moss-image-text-encoder-w4a8-gptq": "moss_image_qwen35_w4a8_gptq",
}
_COMPONENT_PRECISIONS = {
    "moss-image-sft-int8-convrot": "int8",
    "moss-image-text-encoder-w4a8-gptq": "w4a8",
}


def _structural_profile(item: ModelLibraryItem, header: SafetensorsHeader | None) -> str | None:
    if header is None:
        return None
    tensors = header.tensors
    prefix = ''
    if item.type == 'checkpoint':
        # ComfyUI's diffusion loader also accepts UNet tensors extracted from
        # checkpoints with one of these namespaces (v0.43.0, model_detection).
        counts = {candidate: sum(key.startswith(candidate) for key in tensors)
                  for candidate in ('model.diffusion_model.', 'model.model.', 'net.')}
        dominant = max(counts, key=lambda candidate: counts[candidate])
        selected = dominant if counts[dominant] > 5 else 'model.'
        if any(key.startswith(selected) for key in tensors):
            prefix = selected
    def shape(key: str) -> tuple[int, ...]:
        tensor = tensors.get(prefix + key)
        return tensor.shape if tensor is not None else ()
    # Match the loader's architecture keys and the channels used by this
    # profile. Metadata, filenames, precision and trained values are not identity.
    if item.type == 'checkpoint':
        if (len(shape('txtfusion.projector.weight')) == 2
                and len(shape('first.weight')) == 2 and shape('first.weight')[1] == 64
                and shape('txtfusion.layerwise_blocks.0.prenorm.scale') == (2560,)
                and len(shape('blocks.0.attn.wq.weight')) == 2
                and len(shape('blocks.0.attn.wk.weight')) == 2):
            return 'krea2'
        if (len(shape('shared_block_modulation.weight')) == 2
                and len(shape('layers.0.proj.ffn_up_gate.weight')) == 2
                and len(shape('final_linear_video.weight')) == 2
                and shape('final_linear_video.weight')[0] == 128):
            return 'moss_image'
    elif item.type == 'text_encoder':
        if (len(shape('model.embed_tokens.weight')) == 2 and shape('model.embed_tokens.weight')[1] == 2560
                and len(shape('model.layers.0.self_attn.q_proj.weight')) == 2
                and shape('model.layers.0.self_attn.q_proj.weight')[1] == 2560
                and shape('model.layers.0.self_attn.q_norm.weight') == (128,)):
            return 'krea2'
        if (len(shape('model.language_model.embed_tokens.weight')) == 2
                and shape('model.language_model.embed_tokens.weight')[1] == 4096
                and shape('model.language_model.layers.0.linear_attn.norm.weight') == (128,)):
            return 'moss_image'
    elif item.type == 'vae':
        if (len(shape('decoder.conv1.weight')) == 5 and shape('decoder.conv1.weight')[1] == 16
                and len(shape('encoder.conv1.weight')) == 5 and shape('encoder.conv1.weight')[1] == 3):
            return 'krea2'
        if (len(shape('decoder.conv_in.weight')) == 4 and shape('decoder.conv_in.weight')[1] == 32
                and shape('bn.running_mean') == (128,)):
            return 'moss_image'
    return None


class NativeImageModelClassifier:
    domain = MediaModelDomain.IMAGE

    def __init__(self, dependencies: list[ModelDependency], fingerprints: ModelFingerprintService) -> None:
        self.dependencies = tuple(d for d in dependencies if d.profile in NATIVE_IMAGE_DEFAULTS)
        self.fingerprints = fingerprints

    @classmethod
    def from_config(cls, config: AppConfig) -> NativeImageModelClassifier:
        return cls(ModelDependencyRepository(config.model_dependencies_path, config.model_dir).list(),
                   ModelFingerprintService(config.data_dir / "config/model-fingerprints.json"))

    def matching_dependency(self, item: ModelLibraryItem, header: SafetensorsHeader | None) -> ModelDependency | None:
        if header is None or not header.tensors:
            return None
        candidates = [d for d in self.dependencies
                      if d.type == item.type and d.size_bytes == item.size_bytes and d.checksum_sha256]
        if not candidates:
            return None
        try:
            before = item.path.stat()
            checksum = self.fingerprints.strong_hash(item.path)
            after = item.path.stat()
        except OSError:
            return None
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        ):
            return None
        return next((d for d in candidates if d.checksum_sha256 == checksum), None)

    def classify(self, item: ModelLibraryItem, header: SafetensorsHeader | None) -> MediaModelClassification:
        dependency = self.matching_dependency(item, header)
        if dependency is not None:
            return MediaModelClassification(
                component_role=_ROLES[dependency.slot_id], architecture=dependency.profile,
                variant=_COMPONENT_VARIANTS.get(dependency.id, _VARIANTS[dependency.profile]),
                precision=_COMPONENT_PRECISIONS.get(dependency.id), supported_tasks=("text_to_image",),
                compatible_architecture_ids=(dependency.profile,),
                compatible_profile_ids=(dependency.profile,), compatibility=MediaModelCompatibility.SUPPORTED,
            )
        profile = _structural_profile(item, header)
        if profile is not None:
            slot = next(slot for slot, kind in _SLOTS.items() if kind == item.type)
            return MediaModelClassification(
                component_role=_ROLES[slot], architecture=profile, variant=f'{profile}_native',
                supported_tasks=('text_to_image',), compatible_architecture_ids=(profile,),
                compatible_profile_ids=(profile,), compatibility=MediaModelCompatibility.SUPPORTED,
            )
        family = "unknown"
        if header is not None:
            keys = header.tensors.keys()
            if "txtfusion.projector.weight" in keys:
                family = "krea2"
            elif "shared_block_modulation.weight" in keys and "layers.0.proj.ffn_up_gate.weight" in keys:
                family = "moss_image"
        return MediaModelClassification(
            architecture=family, compatibility=MediaModelCompatibility.PROVISIONAL_UNSUPPORTED,
            reason_code="native_image_component_unverified",
        )


class ImageModelClassifier:
    domain = MediaModelDomain.IMAGE

    def __init__(self) -> None:
        from app.services.anima_model_classifier import AnimaModelClassifier

        self.native = NativeImageModelClassifier.from_config(AppConfig())
        self.anima = AnimaModelClassifier()

    def classify(self, item: ModelLibraryItem, header: SafetensorsHeader | None) -> MediaModelClassification:
        result = self.native.classify(item, header)
        if result.architecture != "unknown":
            return result
        return self.anima.classify(item, header)


def prepare_native_image_bundle(
    profile: str, model_slots: dict[str, str], classifier: NativeImageModelClassifier,
) -> dict:
    components = {}
    classifications = {}
    for slot, kind in _SLOTS.items():
        value = str(model_slots.get(slot) or "").strip()
        if not value:
            raise ValueError(f"native_image_component_unverified:{slot}")
        path = Path(value)
        try:
            before = path.stat()
            header = read_safetensors_header(path)
            item = ModelLibraryItem(kind, path.name, path, Path(path.name), before.st_size, {})
            dependency = classifier.matching_dependency(item, header)
            classification = classifier.classify(item, header)
            checksum = classifier.fingerprints.strong_hash(path)
            after = path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            ):
                raise ValueError(f"native_image_component_changed:{slot}")
        except (OSError, SafetensorsHeaderError):
            classification = None
        if (classification is None or classification.compatibility != MediaModelCompatibility.SUPPORTED
                or profile not in classification.compatible_profile_ids
                or classification.component_role != _ROLES[slot]):
            raise ValueError(f"native_image_component_unverified:{slot}")
        classifications[slot] = classification
        components[slot] = {"dependency_id": dependency.id if dependency else None,
                            "size_bytes": item.size_bytes,
                            "sha256": checksum}
    variant = classifications['diffusion_model'].variant
    return {"profile": profile, "variant": variant, "components": components}
