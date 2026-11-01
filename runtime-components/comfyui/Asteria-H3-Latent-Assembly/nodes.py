from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
import torch


CONTRACT_VERSION = "asteria.video.sequence.latent-assembly.v1"
MANIFEST_SCHEMA_VERSION = "asteria.video.sequence.latent-assembly-input.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_CLIPS = 512


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: Any, label: str) -> str:
    normalized = str(value or "")
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} is invalid")
    return normalized


def _digest(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{label} is invalid")
    return normalized


def _load_manifest(path_value: str, fps: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(str(path_value)).expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("Asteria latent assembly manifest is invalid")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Asteria latent assembly manifest is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Asteria latent assembly manifest contract is invalid")
    _identifier(payload.get("operation_id"), "assembly operation")
    _identifier(payload.get("sequence_id"), "sequence")
    try:
        active_revision = int(payload.get("active_chain_revision"))
        assembly_revision = int(payload.get("assembly_revision"))
        manifest_fps = float(payload.get("fps"))
    except (TypeError, ValueError) as error:
        raise ValueError("Asteria latent assembly manifest metadata is invalid") from error
    if active_revision < 1 or assembly_revision < 1 or not math.isfinite(manifest_fps):
        raise ValueError("Asteria latent assembly manifest metadata is invalid")
    if abs(manifest_fps - float(fps)) > 1e-9:
        raise ValueError("Asteria latent assembly FPS does not match the manifest")
    raw_roots = payload.get("allowed_roots")
    if not isinstance(raw_roots, list) or not raw_roots or len(raw_roots) > _MAX_CLIPS:
        raise ValueError("Asteria latent assembly roots are invalid")
    allowed_roots = []
    for raw_root in raw_roots:
        root = Path(str(raw_root or "")).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Asteria latent assembly root is invalid")
        allowed_roots.append(root)
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, list) or not 1 <= len(raw_clips) <= _MAX_CLIPS:
        raise ValueError("Asteria latent assembly clip list is invalid")
    clips: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_clips):
        if not isinstance(raw, dict) or int(raw.get("ordinal", -1)) != ordinal:
            raise ValueError("Asteria latent assembly clip order is invalid")
        accepted_revision = int(raw.get("accepted_revision", 0))
        overlap_frames = int(raw.get("overlap_frames", -1))
        if accepted_revision < 1 or overlap_frames < 0 or (ordinal == 0 and overlap_frames != 0):
            raise ValueError("Asteria latent assembly clip revision is invalid")
        segment_id = _identifier(raw.get("segment_id"), "segment")
        media_artifact_id = _identifier(raw.get("media_artifact_id"), "media artifact")
        working_artifact_id = _identifier(raw.get("working_artifact_id"), "working artifact")
        working_operation_id = _identifier(raw.get("working_operation_id"), "working operation")
        media_sha256 = _digest(raw.get("media_sha256"), "media digest")
        working_sha256 = _digest(raw.get("working_sha256"), "working digest")
        source_path = Path(str(raw.get("path") or "")).expanduser().resolve(strict=True)
        if not any(source_path == root or source_path.is_relative_to(root) for root in allowed_roots):
            raise ValueError("Asteria latent assembly path is outside the operation roots")
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("Asteria latent assembly working artifact is invalid")
        if _file_sha256(source_path) != working_sha256:
            raise ValueError("Asteria latent assembly working artifact changed")
        clips.append(
            {
                "ordinal": ordinal,
                "segment_id": segment_id,
                "accepted_revision": accepted_revision,
                "media_artifact_id": media_artifact_id,
                "media_sha256": media_sha256,
                "working_artifact_id": working_artifact_id,
                "working_operation_id": working_operation_id,
                "working_sha256": working_sha256,
                "path": source_path,
                "overlap_frames": overlap_frames,
            }
        )
    return payload, clips


def _load_archive(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        metadata = archive.metadata() or {}
    if metadata.get("format") not in {None, "h3_motion_context_av_v1"}:
        raise ValueError("Asteria latent assembly archive format is invalid")
    values = load_file(str(path), device="cpu")
    if set(("video", "audio")) - set(values):
        raise ValueError("Asteria latent assembly archive streams are missing")
    video = values["video"]
    audio = values["audio"]
    if video.ndim != 5 or audio.ndim != 4 or video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("Asteria latent assembly archive shape is invalid")
    return video, audio


def _decode_video(vae, latent: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latent)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    elif images.ndim != 4:
        raise ValueError("Asteria video VAE returned an invalid shape")
    return images.to(dtype=torch.float32, device="cpu").clamp(0, 1).contiguous()


def _decode_audio(vae, latent: torch.Tensor) -> dict[str, Any]:
    audio = vae.decode(latent)
    if audio.ndim != 3:
        raise ValueError("Asteria audio VAE returned an invalid shape")
    waveform = audio.movedim(-1, 1).to(dtype=torch.float32, device="cpu").contiguous()
    sample_rate = int(
        getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 32000))
    )
    if sample_rate < 1 or waveform.shape[-1] < 1:
        raise ValueError("Asteria audio VAE returned invalid audio")
    return {"waveform": waveform, "sample_rate": sample_rate}


class AsteriaH3UpscaleLoadLatent:
    """Restore an operation-owned H3 AV archive as ComfyUI's nested latent type."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "Asteria/runtime"

    @classmethod
    def IS_CHANGED(cls, latent_path):
        try:
            raw = Path(str(latent_path)).expanduser()
            if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
                return float("nan")
            stat_result = raw.stat()
            return f"{raw.resolve(strict=True)}:{stat_result.st_mtime_ns}:{stat_result.st_size}"
        except (OSError, RuntimeError, ValueError):
            return float("nan")

    @staticmethod
    def load(latent_path):
        raw = Path(str(latent_path)).expanduser()
        if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
            raise ValueError("Asteria H3 upscale latent path is invalid")
        video, audio = _load_archive(raw.resolve(strict=True))
        if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
            raise ValueError("Asteria H3 upscale latent streams are incompatible")
        from comfy.nested_tensor import NestedTensor

        return ({"samples": NestedTensor((video, audio))},)


def _video_blend(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    length = int(previous.shape[0])
    if length == 1:
        alpha = torch.full((1, 1, 1, 1), 0.5, dtype=previous.dtype)
    else:
        alpha = torch.linspace(0.0, 1.0, length, dtype=previous.dtype).view(
            length, 1, 1, 1
        )
    return previous * (1 - alpha) + current * alpha


def _audio_blend(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    length = int(previous.shape[-1])
    theta = torch.linspace(0.0, math.pi / 2, length, dtype=previous.dtype).view(1, 1, length)
    return previous * torch.cos(theta) + current * torch.sin(theta)


class AsteriaH3AcceptedLatentStitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": ""}),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "contract_version": ([CONTRACT_VERSION],),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_count", "report")
    FUNCTION = "stitch"
    CATEGORY = "Asteria/runtime"

    @classmethod
    def IS_CHANGED(cls, manifest_path, video_vae, audio_vae, fps, contract_version):
        del video_vae, audio_vae
        try:
            payload, clips = _load_manifest(manifest_path, fps)
            return (
                contract_version,
                sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                tuple((clip["working_sha256"], clip["overlap_frames"]) for clip in clips),
            )
        except Exception:
            return float("nan")

    @staticmethod
    def stitch(manifest_path, video_vae, audio_vae, fps, contract_version):
        if contract_version != CONTRACT_VERSION:
            raise ValueError("Asteria latent assembly contract is unsupported")
        _payload, clips = _load_manifest(manifest_path, fps)
        image_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        sample_rate: int | None = None
        image_shape: tuple[int, ...] | None = None
        audio_channels: int | None = None
        report: list[str] = []
        for clip in clips:
            video_latent, audio_latent = _load_archive(clip["path"])
            images = _decode_video(video_vae, video_latent)
            del video_latent
            audio = _decode_audio(audio_vae, audio_latent)
            del audio_latent
            current_shape = tuple(images.shape[1:])
            current_channels = int(audio["waveform"].shape[1])
            if image_shape is None:
                image_shape = current_shape
                audio_channels = current_channels
                sample_rate = int(audio["sample_rate"])
            elif (
                current_shape != image_shape
                or current_channels != audio_channels
                or int(audio["sample_rate"]) != sample_rate
            ):
                raise ValueError("Asteria latent assembly decoded streams are incompatible")
            overlap = int(clip["overlap_frames"])
            cross_samples = round(overlap / float(fps) * int(sample_rate or 0))
            if overlap:
                if (
                    not image_parts
                    or not audio_parts
                    or images.shape[0] < overlap
                    or image_parts[-1].shape[0] < overlap
                    or audio["waveform"].shape[-1] < cross_samples
                    or audio_parts[-1].shape[-1] < cross_samples
                    or cross_samples < 1
                ):
                    raise ValueError("Asteria latent assembly overlap is invalid")
                previous_images = image_parts.pop()
                previous_audio = audio_parts.pop()
                if previous_images.shape[0] > overlap:
                    image_parts.append(previous_images[:-overlap])
                image_parts.append(
                    _video_blend(previous_images[-overlap:], images[:overlap])
                )
                if previous_audio.shape[-1] > cross_samples:
                    audio_parts.append(previous_audio[..., :-cross_samples])
                audio_parts.append(
                    _audio_blend(
                        previous_audio[..., -cross_samples:],
                        audio["waveform"][..., :cross_samples],
                    )
                )
                images = images[overlap:]
                audio["waveform"] = audio["waveform"][..., cross_samples:]
            if images.shape[0]:
                image_parts.append(images)
            if audio["waveform"].shape[-1]:
                audio_parts.append(audio["waveform"])
            report.append(
                f"{clip['segment_id']}: accepted_revision={clip['accepted_revision']}, "
                f"overlap_frames={overlap}, working_sha256={clip['working_sha256']}"
            )
            del images, audio
        final_images = torch.cat(image_parts, dim=0).contiguous()
        final_waveform = torch.cat(audio_parts, dim=-1).contiguous()
        expected_samples = round(final_images.shape[0] / float(fps) * int(sample_rate or 0))
        maximum_sync_correction = max(1, round(int(sample_rate or 0) / float(fps)))
        if abs(final_waveform.shape[-1] - expected_samples) > maximum_sync_correction:
            raise ValueError("Asteria latent assembly audio/video duration mismatch")
        if final_waveform.shape[-1] > expected_samples:
            final_waveform = final_waveform[..., :expected_samples]
        elif final_waveform.shape[-1] < expected_samples:
            final_waveform = torch.nn.functional.pad(
                final_waveform,
                (0, expected_samples - final_waveform.shape[-1]),
            )
        report.append(
            f"TOTAL: frames={final_images.shape[0]}, samples={final_waveform.shape[-1]}, "
            f"fps={float(fps):.6f}, sample_rate={int(sample_rate or 0)}"
        )
        return (
            final_images,
            {"waveform": final_waveform, "sample_rate": int(sample_rate or 0)},
            int(final_images.shape[0]),
            "\n".join(report),
        )


NODE_CLASS_MAPPINGS = {
    "AsteriaH3AcceptedLatentStitcher": AsteriaH3AcceptedLatentStitcher,
    "AsteriaH3UpscaleLoadLatent": AsteriaH3UpscaleLoadLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaH3AcceptedLatentStitcher": "Asteria H3 Accepted Latent Stitcher",
    "AsteriaH3UpscaleLoadLatent": "Asteria H3 Upscale Load Latent",
}
