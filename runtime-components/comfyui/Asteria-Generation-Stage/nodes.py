"""Lossless, operation-owned generation handoff; decode only, no resampling or network."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch


CONTRACT = "asteria.generation-stage.v1"
MAX_LATENT_BYTES = 256 * 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024


def _digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _json(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("generation stage metadata is invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("generation stage metadata is invalid")
    return value


def _directory(value: str) -> Path:
    path = Path(value).resolve(strict=True)
    owner = _json(path / "owner.json")
    root = Path(owner["output_root"]).resolve(strict=True)
    if (owner.get("schema_version") != CONTRACT or not re.fullmatch(r"[0-9a-f]{32}", path.name)
            or owner.get("operation_id") != path.name or path.parent != root / ".image-check"):
        raise ValueError("generation stage directory is outside its operation root")
    return path


def _metadata(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (tuple, list)):
        return [_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata(item) for key, item in value.items()}
    raise ValueError("generation stage latent metadata is not serializable")


class AsteriaSaveGenerationStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "staging_dir": ("STRING",)},
                "optional": {"check_samples": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Asteria/internal"

    def save(self, samples, vae, staging_dir, check_samples=None):
        directory = _directory(staging_dir)
        if any(directory.iterdir()) and set(p.name for p in directory.iterdir()) != {"owner.json"}:
            raise ValueError("generation stage was already written")
        tensors, metadata = {}, {}
        byte_count = 0
        for key, value in samples.items():
            if not isinstance(key, str):
                raise ValueError("generation stage metadata key is invalid")
            if torch.is_tensor(value):
                byte_count += value.numel() * value.element_size()
                if byte_count > MAX_LATENT_BYTES:
                    raise ValueError("generation stage latent byte limit exceeded")
                tensors[key] = value.detach().to(device="cpu", copy=True).contiguous()
            else:
                metadata[key] = _metadata(value)
        latent = tensors.get("samples")
        if latent is None or latent.ndim < 4 or not 1 <= latent.shape[0] <= 64:
            raise ValueError("generation stage latent shape is invalid")
        # Preserve exact tensor values and dtype; the standard LoadLatent casts to FP32.
        encoded_metadata = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
        if len(encoded_metadata.encode("utf-8")) > MAX_MANIFEST_BYTES:
            raise ValueError("generation stage metadata byte limit exceeded")
        archive = directory / "latent.safetensors"
        temporary = directory / "latent.safetensors.tmp"
        save_file(tensors, str(temporary), metadata={"contract": CONTRACT, "latent_metadata": encoded_metadata})
        os.replace(temporary, archive)
        images = vae.decode((check_samples if check_samples is not None else samples)["samples"])
        # Image VAEs based on a video decoder (Anima/Wan) retain a singleton frame axis.
        # This boundary judges still images, never separate video frames as batch members.
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 4 or images.shape[0] != latent.shape[0] or images.shape[-1] != 3:
            raise ValueError("generation stage decoded image shape is invalid")
        image_records, total_image_bytes = [], 0
        for index, image in enumerate(images):
            if image.shape[0] * image.shape[1] > 16_000_000:
                raise ValueError("generation stage image pixel limit exceeded")
            pixels = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            path = directory / f"base-{index:04}.png"
            Image.fromarray(pixels).save(path)
            total_image_bytes += path.stat().st_size
            if total_image_bytes > MAX_IMAGE_BYTES:
                raise ValueError("generation stage image byte limit exceeded")
            image_records.append({"index": index, "filename": path.name, "sha256": _digest(path)})
        manifest = {
            "schema_version": CONTRACT, "batch_size": int(latent.shape[0]),
            "noise_indices": metadata.get("batch_index"),
            "check_source": "denoised_preview" if check_samples is not None else "base_samples",
            "latent_sha256": _digest(archive), "latent_bytes": archive.stat().st_size,
            "images": image_records,
        }
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = directory / "stage.json.tmp"
        temporary.write_bytes(body)
        os.replace(temporary, directory / "stage.json")
        return {"ui": {"asteria_generation_stage": [sha256(body).hexdigest()]}, "result": ()}


class AsteriaLoadGenerationStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"staging_dir": ("STRING",), "indices": ("STRING",), "stage_sha256": ("STRING",)}}

    RETURN_TYPES = ("LATENT", "ASTERIA_BATCH_SELECTION")
    FUNCTION = "load"
    CATEGORY = "Asteria/internal"

    def load(self, staging_dir, indices, stage_sha256):
        directory = _directory(staging_dir)
        manifest = _json(directory / "stage.json")
        if manifest.get("schema_version") != CONTRACT or _digest(directory / "stage.json") != stage_sha256:
            raise ValueError("generation stage manifest changed")
        archive = directory / "latent.safetensors"
        if (archive.is_symlink() or archive.stat().st_size != manifest["latent_bytes"]
                or archive.stat().st_size > MAX_LATENT_BYTES + MAX_MANIFEST_BYTES
                or _digest(archive) != manifest["latent_sha256"]):
            raise ValueError("generation stage latent changed")
        selected = json.loads(indices)
        count = manifest["batch_size"]
        if (not isinstance(selected, list) or not selected or len(selected) > count
                or any(type(i) is not int or not 0 <= i < count for i in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("generation stage selection is invalid")
        from safetensors import safe_open
        with safe_open(str(archive), framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
        if metadata.get("contract") != CONTRACT:
            raise ValueError("generation stage latent contract is invalid")
        values = json.loads(metadata["latent_metadata"])
        for key, value in load_file(str(archive), device="cpu").items():
            values[key] = value[selected] if value.ndim > 0 and value.shape[0] == count else value
        if "batch_index" in values or selected != list(range(count)):
            original_indices = values.get("batch_index", list(range(count)))
            values["batch_index"] = [original_indices[i] for i in selected]
        return values, {"count": count, "indices": selected, "noise_indices": manifest.get("noise_indices")}


NODE_CLASS_MAPPINGS = {
    "AsteriaSaveGenerationStage": AsteriaSaveGenerationStage,
    "AsteriaLoadGenerationStage": AsteriaLoadGenerationStage,
}
