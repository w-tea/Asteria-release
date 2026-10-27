from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


MediaKind = Literal["image", "video", "audio"]
MediaArtifactKind = Literal["image", "video", "audio", "sidecar", "poster"]
MediaArtifactRole = Literal[
    "primary",
    "derived",
    "source",
    "poster",
    "preview",
    "metadata",
    "sidecar",
    "original",
]

_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_MIME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _validate_media_path(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("media path must be non-empty and must not contain surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("media path contains an unsafe control character")
    segments = value.replace("\\", "/").split("/")
    if ".." in segments:
        raise ValueError("media path traversal is not allowed")
    return value


class MediaArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    kind: MediaArtifactKind
    role: MediaArtifactRole
    path: str
    mime_type: str
    poster_artifact_id: str | None = None
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    fps_numerator: int | None = Field(default=None, ge=0)
    fps_denominator: int | None = Field(default=None, gt=0)
    codec: str | None = None
    has_audio: bool | None = None
    sample_rate: int | None = Field(default=None, ge=0)
    channels: int | None = Field(default=None, ge=0)
    bit_depth: int | None = Field(default=None, ge=0)
    size_bytes: int = Field(default=0, ge=0)
    sha256: str | None = None
    adapter_metadata: dict[str, Any] = Field(default_factory=dict)
    sequence_id: str | None = None
    segment_id: str | None = None
    accepted_revision: int | None = Field(default=None, ge=0)
    parent_artifact_ids: tuple[str, ...] = ()

    @field_validator("artifact_id", "poster_artifact_id", "sequence_id", "segment_id")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is not None and not _ARTIFACT_ID_PATTERN.fullmatch(value):
            raise ValueError("media identifier is invalid")
        return value

    @field_validator("parent_artifact_ids")
    @classmethod
    def validate_parent_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("parent artifact identifiers must be unique")
        if any(not _ARTIFACT_ID_PATTERN.fullmatch(value) for value in values):
            raise ValueError("parent artifact identifier is invalid")
        return values

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_media_path(value)

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _MIME_PATTERN.fullmatch(normalized):
            raise ValueError("media MIME type is invalid")
        return normalized

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not _SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("media SHA-256 is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_media_semantics(self) -> "MediaArtifact":
        if self.poster_artifact_id is not None and self.kind != "video":
            raise ValueError("poster artifact reference is only valid for a video artifact")
        if (self.fps_numerator is None) != (self.fps_denominator is None):
            raise ValueError("frame-rate numerator and denominator must be provided together")
        audio_metadata = (self.sample_rate, self.channels, self.bit_depth)
        if self.kind == "audio" and self.has_audio is not True:
            raise ValueError("audio artifact must declare has_audio=true")
        if any(value is not None for value in audio_metadata):
            if self.kind not in {"audio", "video"} or self.has_audio is not True:
                raise ValueError("audio metadata requires an audio-bearing artifact")
        return self

    @computed_field(return_type=float | None)
    @property
    def frame_rate(self) -> float | None:
        if self.fps_numerator is None or self.fps_denominator is None:
            return None
        return self.fps_numerator / self.fps_denominator


def image_media_type(path: str) -> str | None:
    suffix = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    return _IMAGE_MIME_TYPES.get(suffix)


def legacy_image_artifacts(generation_id: int, paths: list[str]) -> list[MediaArtifact]:
    artifacts: list[MediaArtifact] = []
    for ordinal, path in enumerate(paths):
        if not isinstance(path, str):
            continue
        mime_type = image_media_type(path)
        if mime_type is None:
            continue
        digest = sha256(f"{generation_id}:{ordinal}:{path}".encode("utf-8")).hexdigest()[:20]
        try:
            artifacts.append(
                MediaArtifact(
                    artifact_id=f"media:legacy:{generation_id}:{digest}",
                    kind="image",
                    role="primary",
                    path=path,
                    mime_type=mime_type,
                    size_bytes=0,
                )
            )
        except ValueError:
            # The original compatibility field remains untouched; only an unsafe
            # path is omitted from the typed projection.
            continue
    return artifacts
