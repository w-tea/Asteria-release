from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import stat
import struct


COMFY_WEIGHT_EXTENSIONS = frozenset({".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"})
MODEL_TYPE_EXTENSIONS = {
    "checkpoint": COMFY_WEIGHT_EXTENSIONS,
    "vae": COMFY_WEIGHT_EXTENSIONS,
    "text_encoder": COMFY_WEIGHT_EXTENSIONS,
    "tokenizer": frozenset({".json", ".model", ".txt"}),
    "lora": COMFY_WEIGHT_EXTENSIONS,
    "embedding": COMFY_WEIGHT_EXTENSIONS,
    "controlnet": COMFY_WEIGHT_EXTENSIONS,
    "detailer": frozenset({".pt", ".pth", ".onnx"}),
    "upscaler": COMFY_WEIGHT_EXTENSIONS,
}
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
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class ModelLinkHealth:
    linked_model_count: int = 0
    broken_link_count: int = 0
    unreadable_link_count: int = 0
    loop_link_count: int = 0
    duplicate_target_count: int = 0
    skipped_file_count: int = 0


@dataclass
class _MutableLinkHealth:
    linked_model_count: int = 0
    broken_link_count: int = 0
    unreadable_link_count: int = 0
    loop_link_count: int = 0
    duplicate_target_count: int = 0
    skipped_file_count: int = 0

    def freeze(self) -> ModelLinkHealth:
        return ModelLinkHealth(**self.__dict__)


@dataclass(frozen=True)
class ModelLibraryItem:
    type: str
    name: str
    path: Path
    relative_path: Path
    size_bytes: int
    metadata: dict[str, str]
    source: str = "local"


@dataclass(frozen=True)
class ModelLibrary:
    items: list[ModelLibraryItem]
    link_health: ModelLinkHealth = field(default_factory=ModelLinkHealth)

    def by_type(self, model_type: str) -> list[ModelLibraryItem]:
        if model_type == "diffusion_model":
            model_type = "checkpoint"
        return [item for item in self.items if item.type == model_type]


class ModelLibraryScanner:
    def __init__(self, model_root: Path) -> None:
        self.model_root = Path(os.path.abspath(os.fspath(model_root)))

    def scan(self) -> ModelLibrary:
        items: list[ModelLibraryItem] = []
        health = _MutableLinkHealth()
        for model_type, directories in MODEL_TYPE_DIRECTORIES.items():
            # One physical file may intentionally be registered under multiple
            # model roles. Only aliases within the same role are duplicates.
            seen_directories: set[tuple[object, ...]] = set()
            seen_files: set[tuple[object, ...]] = set()
            for directory in directories:
                base = self.model_root / directory
                if self._lexically_present(base):
                    self._scan_directory(
                        model_type,
                        base,
                        inherited_link=False,
                        ancestors=frozenset(),
                        seen_directories=seen_directories,
                        seen_files=seen_files,
                        health=health,
                        items=items,
                    )
        items = self._dedupe_checkpoint_aliases(items)
        items.sort(key=lambda item: (item.type, item.name.casefold(), str(item.relative_path).casefold()))
        return ModelLibrary(items=items, link_health=health.freeze())

    def _dedupe_checkpoint_aliases(self, items: list[ModelLibraryItem]) -> list[ModelLibraryItem]:
        preferred: dict[tuple[str, ...], ModelLibraryItem] = {}
        others: list[ModelLibraryItem] = []
        aliases = {"checkpoints", "diffusion_models", "unet"}
        for item in items:
            parts = item.relative_path.parts
            if item.type != "checkpoint" or not parts or parts[0].casefold() not in aliases:
                others.append(item)
                continue
            key = tuple(part.casefold() for part in parts[1:])
            current = preferred.get(key)
            if current is None:
                preferred[key] = item
                continue
            current_is_canonical = current.relative_path.parts[0].casefold() == "checkpoints"
            item_is_canonical = parts[0].casefold() == "checkpoints"
            if item_is_canonical and not current_is_canonical:
                preferred[key] = item
        return [*others, *preferred.values()]

    def _scan_directory(
        self,
        model_type: str,
        logical_directory: Path,
        *,
        inherited_link: bool,
        ancestors: frozenset[tuple[object, ...]],
        seen_directories: set[tuple[object, ...]],
        seen_files: set[tuple[object, ...]],
        health: _MutableLinkHealth,
        items: list[ModelLibraryItem],
    ) -> None:
        link_here = self._is_reparse_point(logical_directory, health)
        linked = inherited_link or link_here
        try:
            directory_stat = os.stat(logical_directory, follow_symlinks=True)
        except FileNotFoundError:
            if link_here:
                health.broken_link_count += 1
            return
        except PermissionError:
            if link_here:
                health.unreadable_link_count += 1
            return
        except OSError:
            if link_here:
                health.unreadable_link_count += 1
            return
        if not stat.S_ISDIR(directory_stat.st_mode):
            return
        identity = self._physical_identity(logical_directory, directory_stat)
        if identity in ancestors:
            health.loop_link_count += 1
            return
        if identity in seen_directories:
            health.duplicate_target_count += 1
            return
        seen_directories.add(identity)

        try:
            entries = list(os.scandir(logical_directory))
        except PermissionError:
            if linked:
                health.unreadable_link_count += 1
            return
        except OSError:
            if linked:
                health.unreadable_link_count += 1
            return
        entries.sort(key=lambda entry: (self._entry_is_reparse(entry), entry.name.casefold()))
        child_ancestors = ancestors | {identity}
        for entry in entries:
            logical_path = Path(entry.path)
            reparse = self._entry_is_reparse(entry)
            child_linked = linked or reparse
            try:
                target_stat = logical_path.stat()
            except FileNotFoundError:
                if reparse:
                    health.broken_link_count += 1
                else:
                    health.skipped_file_count += 1
                continue
            except PermissionError:
                if reparse:
                    health.unreadable_link_count += 1
                else:
                    health.skipped_file_count += 1
                continue
            except OSError:
                if reparse:
                    health.unreadable_link_count += 1
                else:
                    health.skipped_file_count += 1
                continue

            if stat.S_ISDIR(target_stat.st_mode):
                self._scan_directory(
                    model_type,
                    logical_path,
                    inherited_link=child_linked,
                    ancestors=child_ancestors,
                    seen_directories=seen_directories,
                    seen_files=seen_files,
                    health=health,
                    items=items,
                )
                continue
            if not stat.S_ISREG(target_stat.st_mode):
                if reparse:
                    health.unreadable_link_count += 1
                continue
            if logical_path.suffix.casefold() not in MODEL_TYPE_EXTENSIONS[model_type]:
                continue
            try:
                relative_path = logical_path.relative_to(self.model_root)
            except ValueError:
                health.skipped_file_count += 1
                continue
            if any(part.casefold() in HIDDEN_MODEL_PATH_PARTS for part in relative_path.parts[:-1]):
                continue
            physical_identity = self._physical_identity(logical_path, target_stat)
            if physical_identity in seen_files:
                health.duplicate_target_count += 1
                continue
            seen_files.add(physical_identity)
            source = "linked" if child_linked else "local"
            if source == "linked":
                health.linked_model_count += 1
            items.append(
                ModelLibraryItem(
                    type=model_type,
                    name=logical_path.name,
                    path=logical_path,
                    relative_path=relative_path,
                    size_bytes=target_stat.st_size,
                    metadata=self._metadata_for(logical_path),
                    source=source,
                )
            )

    def _metadata_for(self, path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if path.suffix.casefold() != ".safetensors":
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
            try:
                metadata["cm_info_path"] = str(sidecar.relative_to(self.model_root))
            except ValueError:
                pass
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
                try:
                    metadata["preview_image_path"] = str(preview.relative_to(self.model_root))
                except ValueError:
                    pass
                break
        return metadata

    @staticmethod
    def _physical_identity(path: Path, details: os.stat_result) -> tuple[object, ...]:
        if details.st_ino:
            return ("inode", details.st_dev, details.st_ino)
        return ("path", os.path.normcase(os.path.realpath(path)))

    @staticmethod
    def _entry_is_reparse(entry: os.DirEntry[str]) -> bool:
        try:
            details = entry.stat(follow_symlinks=False)
        except OSError:
            try:
                return entry.is_symlink()
            except OSError:
                return False
        try:
            is_symlink = entry.is_symlink()
        except OSError:
            is_symlink = False
        return is_symlink or bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)

    @staticmethod
    def _is_reparse_point(path: Path, health: _MutableLinkHealth) -> bool:
        try:
            details = os.lstat(path)
        except FileNotFoundError:
            return False
        except PermissionError:
            health.unreadable_link_count += 1
            return False
        except OSError:
            health.unreadable_link_count += 1
            return False
        return stat.S_ISLNK(details.st_mode) or bool(
            getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        )

    @staticmethod
    def _lexically_present(path: Path) -> bool:
        try:
            if path.exists():
                return True
        except OSError:
            return False
        try:
            details = os.lstat(path)
            return stat.S_ISLNK(details.st_mode) or bool(
                getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            )
        except FileNotFoundError:
            return False
        except OSError:
            return False

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    @staticmethod
    def _path_is_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False
