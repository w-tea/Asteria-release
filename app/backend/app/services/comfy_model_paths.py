from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from app.services.external_model_libraries import ExternalModelRoot
from app.services.model_library import MODEL_TYPE_DIRECTORIES


_CHECKPOINT_DIRECTORIES = MODEL_TYPE_DIRECTORIES["checkpoint"]
_VAE_DIRECTORIES = MODEL_TYPE_DIRECTORIES["vae"]
_TEXT_ENCODER_DIRECTORIES = MODEL_TYPE_DIRECTORIES["text_encoder"]
_TOKENIZER_DIRECTORIES = MODEL_TYPE_DIRECTORIES["tokenizer"]
_DETAILER_BBOX_DIRECTORIES = ("adetailer", "bbox", "ultralytics/bbox")
_DETAILER_SEGM_DIRECTORIES = ("adetailer", "segm", "ultralytics/segm")

# Every directory recognized by ModelLibraryScanner must be registered with
# ComfyUI under the folder key used by its loader (or a dedicated registry key
# for dependency-only assets such as tokenizers).
COMFY_MODEL_PATH_DIRECTORIES: dict[str, tuple[str, ...]] = {
    "checkpoints": _CHECKPOINT_DIRECTORIES,
    "diffusion_models": _CHECKPOINT_DIRECTORIES,
    "unet": _CHECKPOINT_DIRECTORIES,
    "vae": _VAE_DIRECTORIES,
    "vae_approx": ("vae_approx",),
    "text_encoders": _TEXT_ENCODER_DIRECTORIES,
    "clip": _TEXT_ENCODER_DIRECTORIES,
    "tokenizers": _TOKENIZER_DIRECTORIES,
    "tokenizer": _TOKENIZER_DIRECTORIES,
    "loras": MODEL_TYPE_DIRECTORIES["lora"],
    "embeddings": MODEL_TYPE_DIRECTORIES["embedding"],
    "controlnet": MODEL_TYPE_DIRECTORIES["controlnet"],
    # Current ComfyUI loads Anima LLLite through ModelPatchLoader. Keep the
    # existing ControlNet library layout available to that standard loader so
    # users do not have to move already installed models.
    "model_patches": MODEL_TYPE_DIRECTORIES["controlnet"],
    "ultralytics_bbox": _DETAILER_BBOX_DIRECTORIES,
    "ultralytics_segm": _DETAILER_SEGM_DIRECTORIES,
    "upscale_models": MODEL_TYPE_DIRECTORIES["upscaler"],
}


_EXTERNAL_COMFY_KEYS: dict[str, tuple[str, ...]] = {
    "checkpoint": ("checkpoints", "diffusion_models", "unet"),
    "vae": ("vae",),
    "text_encoder": ("text_encoders", "clip"),
    "tokenizer": ("tokenizers", "tokenizer"),
    "lora": ("loras",),
    "embedding": ("embeddings",),
    "controlnet": ("controlnet", "model_patches"),
    "upscaler": ("upscale_models",),
}


def comfy_extra_model_paths_document(
    model_root: Path,
    external_roots: Iterable[ExternalModelRoot] = (),
) -> dict[str, dict[str, str]]:
    document: dict[str, dict[str, str]] = {
        "asteria": {
            "base_path": Path(model_root).as_posix(),
            **{
                folder_name: "\n".join(directories)
                for folder_name, directories in COMFY_MODEL_PATH_DIRECTORIES.items()
            },
        }
    }
    for root in external_roots:
        if not root.enabled or not root.path:
            continue
        entry: dict[str, str] = {"base_path": Path(root.path).as_posix()}
        for model_type, directories in root.mappings.items():
            if model_type == "detailer":
                detailer_paths = _external_detailer_paths(directories)
                for folder_name, paths in detailer_paths.items():
                    if paths:
                        entry[folder_name] = "\n".join(paths)
                continue
            for folder_name in _EXTERNAL_COMFY_KEYS.get(model_type, ()):
                entry[folder_name] = "\n".join(directories)
        if len(entry) > 1:
            document[f"asteria_external_{root.root_id}"] = entry
    return document


def _external_detailer_paths(
    directories: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    bbox: list[str] = []
    segm: list[str] = []
    for directory in directories:
        parts = {part.casefold() for part in Path(directory.replace("\\", "/")).parts}
        if "segm" in parts:
            segm.append(directory)
        elif "bbox" in parts:
            bbox.append(directory)
        else:
            # ADetailer and custom detector folders may contain either model
            # family, so ComfyUI must be able to discover them from both lists.
            bbox.append(directory)
            segm.append(directory)
    return {
        "ultralytics_bbox": tuple(dict.fromkeys(bbox)),
        "ultralytics_segm": tuple(dict.fromkeys(segm)),
    }


def render_comfy_extra_model_paths(
    model_root: Path,
    external_roots: Iterable[ExternalModelRoot] = (),
) -> str:
    lines: list[str] = []
    for section, entry in comfy_extra_model_paths_document(model_root, external_roots).items():
        lines.extend([f"{section}:", f"  base_path: {json.dumps(entry['base_path'])}"])
        for folder_name, value in entry.items():
            if folder_name == "base_path":
                continue
            lines.append(f"  {folder_name}: |-")
            lines.extend(f"    {directory}" for directory in value.splitlines())
    return "\n".join([*lines, ""])


def comfy_model_name(
    value: str,
    model_dir_name: str,
    *,
    model_root: str | os.PathLike[str] | None = None,
    external_roots: Iterable[ExternalModelRoot] = (),
) -> str:
    normalized = value.replace("\\", "/")
    relative = _relative_to_model_root(normalized, model_root)
    aliases = sorted(
        COMFY_MODEL_PATH_DIRECTORIES.get(model_dir_name, (model_dir_name,)),
        key=len,
        reverse=True,
    )
    if not _is_absolute_path(normalized):
        return _strip_leading_alias(normalized, aliases).replace("/", "\\")
    if model_root is not None:
        if relative is None:
            external_name = _relative_to_external_model_root(
                normalized,
                model_dir_name,
                external_roots,
            )
            if external_name is not None:
                return external_name.replace("/", "\\")
            anchored = _relative_to_standard_models_anchor(normalized, aliases)
            if anchored is not None:
                return anchored.replace("/", "\\")
            return normalized.replace("/", "\\")
        stripped = _strip_leading_alias(relative, aliases)
        return stripped.replace("/", "\\")

    # Backward compatibility for relative names and callers that have not
    # supplied a root. Runtime call sites always provide the configured root.
    stripped = _strip_leading_alias(normalized, aliases)
    if stripped != normalized:
        return stripped.replace("/", "\\")
    folded = normalized.casefold()
    for directory in aliases:
        marker = f"/{directory.strip('/')}/"
        index = folded.find(marker.casefold())
        if index >= 0:
            return normalized[index + len(marker) :].replace("/", "\\")
    return normalized.replace("/", "\\")


def _relative_to_external_model_root(
    normalized_value: str,
    model_dir_name: str,
    external_roots: Iterable[ExternalModelRoot],
) -> str | None:
    model_types = tuple(
        model_type
        for model_type, comfy_keys in _EXTERNAL_COMFY_KEYS.items()
        if model_dir_name in comfy_keys
    )
    for root in external_roots:
        if not root.enabled or not root.path:
            continue
        relative = _relative_to_model_root(normalized_value, root.path)
        if relative is None:
            continue
        directories = sorted(
            {
                directory
                for model_type in model_types
                for directory in root.mappings.get(model_type, ())
            },
            key=len,
            reverse=True,
        )
        stripped = _strip_leading_alias(relative, directories)
        if stripped != relative:
            return stripped
    return None


def _relative_to_model_root(
    normalized_value: str,
    model_root: str | os.PathLike[str] | None,
) -> str | None:
    if model_root is None:
        return None
    normalized_root = os.fspath(model_root).replace("\\", "/").rstrip("/")
    if not normalized_root:
        return None
    folded_value = normalized_value.casefold()
    folded_root = normalized_root.casefold()
    if folded_value == folded_root:
        return ""
    prefix = f"{folded_root}/"
    if not folded_value.startswith(prefix):
        return None
    return normalized_value[len(normalized_root) + 1 :]


def _strip_leading_alias(value: str, aliases: list[str]) -> str:
    normalized = value.lstrip("/")
    folded = normalized.casefold()
    for directory in aliases:
        alias = directory.strip("/")
        if folded == alias.casefold():
            return ""
        prefix = f"{alias}/"
        if folded.startswith(prefix.casefold()):
            return normalized[len(prefix) :]
    return normalized


def _relative_to_standard_models_anchor(value: str, aliases: list[str]) -> str | None:
    parts = value.split("/")
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].casefold() != "models":
            continue
        candidate = "/".join(parts[index + 1 :])
        stripped = _strip_leading_alias(candidate, aliases)
        if stripped != candidate:
            return stripped
    return None


def _is_absolute_path(value: str) -> bool:
    return value.startswith("/") or (len(value) >= 3 and value[1] == ":" and value[2] == "/")
