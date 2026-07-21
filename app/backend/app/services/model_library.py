from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct


MODEL_FILE_EXTENSIONS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".json", ".model", ".txt"}
MODEL_SIDECAR_IMAGE_EXTENSIONS = (".preview.webp", ".preview.png", ".preview.jpg", ".preview.jpeg")

MODEL_TYPE_DIRECTORIES = {
    "checkpoint": ("checkpoints", "diffusion_models", "unet", "merged/checkpoints"),
    "vae": ("vae", "vae_approx"),
    "text_encoder": ("text_encoders", "clip"),
    "tokenizer": ("tokenizers", "tokenizer"),
    "lora": ("loras", "lora", "merged/lora"),
    "embedding": ("embeddings", "textual_inversion"),
    "controlnet": ("controlnet", "controlnets"),
    "detailer": ("bbox", "ultralytics/bbox"),
    "upscaler": ("upscalers", "esrgan", "realesrgan"),
}
HIDDEN_MODEL_PATH_PARTS = {"merge-candidates"}


@dataclass(frozen=True)
class ModelLibraryItem:
    type: str
    name: str
    path: Path
    relative_path: Path
    size_bytes: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class ModelLibrary:
    items: list[ModelLibraryItem]

    def by_type(self, model_type: str) -> list[ModelLibraryItem]:
        if model_type == "diffusion_model":
            model_type = "checkpoint"
        return [item for item in self.items if item.type == model_type]


class ModelLibraryScanner:
    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root

    def scan(self) -> ModelLibrary:
        items: list[ModelLibraryItem] = []
        for model_type, directories in MODEL_TYPE_DIRECTORIES.items():
            for directory in directories:
                base = self.model_root / directory
                if self._path_exists(base):
                    items.extend(self._scan_directory(model_type, base))
        items = self._dedupe_checkpoint_aliases(items)
        items.sort(key=lambda item: (item.type, item.name.lower(), str(item.relative_path).lower()))
        return ModelLibrary(items=items)

    def _dedupe_checkpoint_aliases(self, items: list[ModelLibraryItem]) -> list[ModelLibraryItem]:
        preferred: dict[tuple[str, str], ModelLibraryItem] = {}
        others: list[ModelLibraryItem] = []
        for item in items:
            if item.type != "checkpoint":
                others.append(item)
                continue
            parts = item.relative_path.parts
            family = parts[1].lower() if len(parts) > 2 and parts[0].lower() in {"checkpoints", "diffusion_models", "unet"} else ""
            key = (family, item.name.lower())
            current = preferred.get(key)
            if current is None:
                preferred[key] = item
                continue
            current_is_canonical = current.relative_path.parts[0].lower() == "checkpoints"
            item_is_canonical = item.relative_path.parts[0].lower() == "checkpoints"
            if item_is_canonical and not current_is_canonical:
                preferred[key] = item
        return [*others, *preferred.values()]

    def _scan_directory(self, model_type: str, base: Path) -> list[ModelLibraryItem]:
        items: list[ModelLibraryItem] = []
        try:
            paths = base.rglob("*")
            for path in paths:
                if not self._path_is_file(path) or path.suffix.lower() not in MODEL_FILE_EXTENSIONS:
                    continue
                if any(part.lower() in HIDDEN_MODEL_PATH_PARTS for part in path.relative_to(base).parts[:-1]):
                    continue
                lower_name = path.name.lower()
                if lower_name.endswith(".cm-info.json") or lower_name.endswith(".asteria.merge.json"):
                    continue
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                items.append(
                    ModelLibraryItem(
                        type=model_type,
                        name=path.name,
                        path=path,
                        relative_path=path.relative_to(self.model_root),
                        size_bytes=size_bytes,
                        metadata=self._metadata_for(path),
                    )
                )
        except OSError:
            return items
        return items

    def _metadata_for(self, path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if path.suffix.lower() != ".safetensors":
            return metadata
        try:
            with path.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size <= 0 or header_size > 16 * 1024 * 1024:
                    return self._sidecar_metadata_for(path)
                header = json.loads(handle.read(header_size).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
            return self._sidecar_metadata_for(path)
        header_metadata = header.get("__metadata__", {})
        if isinstance(header_metadata, dict):
            metadata.update({str(key): str(value) for key, value in header_metadata.items() if value is not None})
        metadata.update(self._sidecar_metadata_for(path))
        return metadata

    def _sidecar_metadata_for(self, path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        sidecar = path.with_name(f"{path.stem}.cm-info.json")
        if self._path_exists(sidecar) and self._path_is_file(sidecar):
            metadata["cm_info_path"] = str(sidecar.relative_to(self.model_root))
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                field_map = {
                    "ModelName": "display_name",
                    "BaseModel": "base_model",
                    "ModelType": "model_type",
                    "VersionName": "version_name",
                    "ThumbnailImageUrl": "thumbnail_image_url",
                }
                for source, target in field_map.items():
                    value = data.get(source)
                    if value is not None:
                        metadata[target] = str(value)
                trained_words = data.get("TrainedWords")
                if isinstance(trained_words, list):
                    metadata["trained_words"] = ", ".join(str(item) for item in trained_words if item)
                tags = data.get("Tags")
                if isinstance(tags, list):
                    metadata["tags"] = ", ".join(str(item) for item in tags if item)
        for suffix in MODEL_SIDECAR_IMAGE_EXTENSIONS:
            preview = path.with_name(f"{path.stem}{suffix}")
            if self._path_exists(preview) and self._path_is_file(preview):
                metadata["preview_image_path"] = str(preview.relative_to(self.model_root))
                break
        return metadata

    def _path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def _path_is_file(self, path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False
