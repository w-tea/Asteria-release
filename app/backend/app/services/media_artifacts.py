from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat

from PIL import Image

from app.schemas.media import MediaArtifact, MediaKind
from app.services.media_probe import probe_media


_MEDIA_DELIVERY_TYPES = {
    ("image", ".png"): "image/png",
    ("image", ".jpg"): "image/jpeg",
    ("image", ".jpeg"): "image/jpeg",
    ("image", ".webp"): "image/webp",
    ("image", ".gif"): "image/gif",
    ("video", ".mp4"): "video/mp4",
    ("video", ".webm"): "video/webm",
    ("audio", ".wav"): "audio/wav",
    ("audio", ".flac"): "audio/flac",
    ("audio", ".mp3"): "audio/mpeg",
    ("audio", ".opus"): "audio/ogg",
    ("poster", ".webp"): "image/webp",
}


class MediaArtifactAccessError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MediaFinalizationResult:
    media_kind: MediaKind
    artifacts: tuple[MediaArtifact, ...]
    output_paths: tuple[str, ...]


def finalize_media_artifact(
    *,
    generation_id: int,
    source_path: Path,
    poster_root: Path,
) -> MediaFinalizationResult:
    source = Path(source_path)
    suffix = source.suffix.lower()
    if suffix not in {".mp4", ".webm", ".wav", ".flac", ".mp3", ".opus"}:
        raise ValueError("media_artifact_unsupported")
    identity = sha256(f"{generation_id}:{source}".encode("utf-8")).hexdigest()[:20]
    primary_kind: MediaKind = "audio" if suffix in {".wav", ".flac", ".mp3", ".opus"} else "video"
    poster_id = f"media:poster:{generation_id}:{identity}" if primary_kind == "video" else None
    poster_path = (
        Path(poster_root) / f"{source.stem}.{identity}.poster.webp"
        if primary_kind == "video"
        else None
    )
    probe = probe_media(source, poster_path=poster_path)
    primary_id = f"media:{primary_kind}:{generation_id}:{identity}"
    primary = MediaArtifact(
        artifact_id=primary_id,
        kind=primary_kind,
        role="primary",
        path=str(source),
        mime_type=probe.mime_type,
        poster_artifact_id=poster_id,
        width=probe.width,
        height=probe.height,
        duration_ms=probe.duration_ms,
        fps_numerator=probe.fps_numerator,
        fps_denominator=probe.fps_denominator,
        codec=probe.codec,
        has_audio=probe.has_audio,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        bit_depth=probe.bit_depth,
        size_bytes=probe.size_bytes,
        sha256=file_sha256(source),
        adapter_metadata={
            "container_format": probe.container_format,
            "rotation_degrees": probe.rotation_degrees,
            "video_time_base": (
                f"{probe.video_time_base_numerator}/{probe.video_time_base_denominator}"
                if probe.video_time_base_numerator is not None
                and probe.video_time_base_denominator is not None
                else None
            ),
            "audio_time_base": (
                f"{probe.audio_time_base_numerator}/{probe.audio_time_base_denominator}"
                if probe.audio_time_base_numerator is not None
                and probe.audio_time_base_denominator is not None
                else None
            ),
        },
    )
    artifacts: list[MediaArtifact] = [primary]
    output_paths = [str(source)]
    if poster_path is not None:
        try:
            with Image.open(poster_path) as poster:
                poster_width, poster_height = poster.size
        except (OSError, ValueError) as error:
            raise ValueError("media_poster_invalid") from error
        poster_artifact = MediaArtifact(
            artifact_id=poster_id or "",
            kind="poster",
            role="poster",
            path=str(poster_path),
            mime_type="image/webp",
            width=poster_width,
            height=poster_height,
            size_bytes=poster_path.stat().st_size,
            sha256=file_sha256(poster_path),
            parent_artifact_ids=(primary_id,),
        )
        artifacts.append(poster_artifact)
        output_paths.append(str(poster_path))
    return MediaFinalizationResult(
        media_kind=primary_kind,
        artifacts=tuple(artifacts),
        output_paths=tuple(output_paths),
    )


def resolve_registered_media_artifact(
    item: dict,
    artifact_id: str,
    *,
    accepted_roots: set[Path],
    allowed_kinds: set[str],
) -> tuple[MediaArtifact, Path]:
    artifact: MediaArtifact | None = None
    for candidate in item.get("artifacts", []):
        try:
            parsed = MediaArtifact.model_validate(candidate)
        except ValueError:
            continue
        if parsed.artifact_id == artifact_id:
            artifact = parsed
            break
    if artifact is None or artifact.kind not in allowed_kinds:
        raise MediaArtifactAccessError("media_artifact_not_found")
    registered_paths = [
        value for value in item.get("output_paths", []) if isinstance(value, str)
    ]
    try:
        raw_source = Path(os.path.abspath(os.fspath(artifact.path)))
        resolved_source = raw_source.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise MediaArtifactAccessError("media_artifact_not_found") from error
    if not resolved_source.is_file() or not _same_registered_path(resolved_source, registered_paths):
        raise MediaArtifactAccessError("media_artifact_not_registered")
    expected_mime = _MEDIA_DELIVERY_TYPES.get((artifact.kind, resolved_source.suffix.lower()))
    if expected_mime is None or artifact.mime_type != expected_mime:
        raise MediaArtifactAccessError("media_artifact_type_mismatch")
    accepted = False
    for root in accepted_roots:
        try:
            raw_root = Path(os.path.abspath(os.fspath(root)))
            resolved_root = raw_root.resolve(strict=True)
            resolved_source.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if _contains_reparse_component(raw_source, raw_root):
            continue
        accepted = True
        break
    if not accepted:
        raise MediaArtifactAccessError("media_artifact_not_registered")
    try:
        actual_size = resolved_source.stat().st_size
    except OSError as error:
        raise MediaArtifactAccessError("media_artifact_not_found") from error
    size_is_unknown_legacy_image = artifact.kind == "image" and artifact.size_bytes == 0
    if not size_is_unknown_legacy_image and artifact.size_bytes != actual_size:
        raise MediaArtifactAccessError("media_artifact_changed")
    return artifact, resolved_source


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _same_registered_path(candidate: Path, registered: list[str]) -> bool:
    candidate_value = os.path.normcase(str(candidate))
    for value in registered:
        try:
            if os.path.normcase(str(Path(value).resolve(strict=False))) == candidate_value:
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def _contains_reparse_component(source: Path, root: Path) -> bool:
    try:
        source.relative_to(root)
    except ValueError:
        return True
    current = source
    while True:
        try:
            details = os.lstat(current)
        except OSError:
            return True
        attributes = getattr(details, "st_file_attributes", 0)
        is_reparse = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        junction_check = getattr(os.path, "isjunction", None)
        try:
            is_junction = bool(junction_check(current)) if junction_check is not None else False
        except OSError:
            return True
        if stat.S_ISLNK(details.st_mode) or is_reparse or is_junction:
            return True
        if current == root:
            return False
        current = current.parent
