from __future__ import annotations

import json
import os
from pathlib import Path

from app.services.model_library import MODEL_TYPE_DIRECTORIES


_CHECKPOINT_DIRECTORIES = MODEL_TYPE_DIRECTORIES["checkpoint"]
_VAE_DIRECTORIES = MODEL_TYPE_DIRECTORIES["vae"]
_TEXT_ENCODER_DIRECTORIES = MODEL_TYPE_DIRECTORIES["text_encoder"]
_TOKENIZER_DIRECTORIES = MODEL_TYPE_DIRECTORIES["tokenizer"]

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
    "ultralytics_bbox": MODEL_TYPE_DIRECTORIES["detailer"],
    "upscale_models": MODEL_TYPE_DIRECTORIES["upscaler"],
}


def comfy_extra_model_paths_document(model_root: Path) -> dict[str, dict[str, str]]:
    return {
        "asteria": {
            "base_path": Path(model_root).as_posix(),
            **{
                folder_name: "\n".join(directories)
                for folder_name, directories in COMFY_MODEL_PATH_DIRECTORIES.items()
            },
        }
    }


def render_comfy_extra_model_paths(model_root: Path) -> str:
    lines = ["asteria:", f"  base_path: {json.dumps(Path(model_root).as_posix())}"]
    for folder_name, directories in COMFY_MODEL_PATH_DIRECTORIES.items():
        lines.append(f"  {folder_name}: |-")
        lines.extend(f"    {directory}" for directory in directories)
    return "\n".join([*lines, ""])


def comfy_model_name(
    value: str,
    model_dir_name: str,
    *,
    model_root: str | os.PathLike[str] | None = None,
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
