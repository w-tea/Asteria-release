from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.provider_capabilities import (
    VideoContinuationMethod,
    VideoContinuationMethodCapability,
    VideoSequenceCapability,
)
from app.schemas.video import VideoRequest, contains_secret_key
from app.services.video_frame_policy import normalized_h3_frame_count
from app.services.video_runtime_adapters import VideoRuntimeAdapter


VIDEO_SEQUENCE_CONTEXT_SCHEMA_VERSION = "asteria.video.sequence.context.v2"
VIDEO_SEQUENCE_CONTEXT_LEGACY_SCHEMA_VERSION = "asteria.video.sequence.context.v1"
DEFAULT_VIDEO_CONTEXT_FRAMES = 22
DEFAULT_AUDIO_CONTEXT_FRAMES = 24
_DIRECT_CONTEXT_WINDOWS = (DEFAULT_VIDEO_CONTEXT_FRAMES, 5)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class VideoSequenceContextError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class VideoSequenceSourceArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    accepted_revision: int = Field(ge=1)
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0, allow_inf_nan=False)
    frame_count: int = Field(ge=1)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    audio_sample_rate: int | None = Field(default=None, gt=0)
    audio_sample_count: int | None = Field(default=None, ge=1)
    working_artifact_id: str | None = None
    working_operation_id: str | None = None

    @field_validator("artifact_id", "working_artifact_id", "working_operation_id")
    @classmethod
    def validate_artifact_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("video sequence source artifact identifier is invalid")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("video sequence source SHA-256 is invalid")
        return normalized

    @model_validator(mode="after")
    def validate_audio_shape(self) -> "VideoSequenceSourceArtifact":
        if (self.audio_sample_rate is None) != (self.audio_sample_count is None):
            raise ValueError("video sequence source audio rate and count must be recorded together")
        if (self.working_artifact_id is None) != (self.working_operation_id is None):
            raise ValueError("video sequence working artifact identity must be recorded together")
        return self


class VideoSequenceFrameRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "VideoSequenceFrameRange":
        if self.end_frame <= self.start_frame:
            raise ValueError("video sequence frame range must be non-empty")
        return self


class VideoSequenceAudioRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=1)
    sample_rate: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "VideoSequenceAudioRange":
        if self.end_sample <= self.start_sample:
            raise ValueError("video sequence audio range must be non-empty")
        return self


class VideoSequenceContextReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "asteria.video.sequence.context.v1",
        "asteria.video.sequence.context.v2",
    ] = (
        VIDEO_SEQUENCE_CONTEXT_SCHEMA_VERSION
    )
    method: VideoContinuationMethod
    quality_description: str
    source_artifact_id: str
    source_accepted_revision: int = Field(ge=1)
    source_sha256: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    source_fps: float = Field(gt=0, allow_inf_nan=False)
    source_frame_range: VideoSequenceFrameRange
    source_audio_range: VideoSequenceAudioRange | None = None
    source_working_artifact_id: str | None = None
    source_working_operation_id: str | None = None
    video_context_frames: int = Field(default=0, ge=0)
    audio_context_frames: int = Field(default=0, ge=0)
    overlap_frames: int = Field(ge=0)
    overlap_seconds: float = Field(ge=0, allow_inf_nan=False)
    adapter_id: str
    adapter_revision: str
    mapping_id: str
    mapping_version: int = Field(ge=1)
    context_sha256: str

    @field_validator(
        "source_artifact_id",
        "source_working_artifact_id",
        "source_working_operation_id",
        "adapter_id",
        "adapter_revision",
        "mapping_id",
    )
    @classmethod
    def validate_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("video sequence context identifier is invalid")
        return value

    @field_validator("source_sha256", "context_sha256")
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("video sequence context SHA-256 is invalid")
        return normalized

    @field_validator("quality_description")
    @classmethod
    def validate_quality_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("video sequence context quality description is required")
        return normalized

    @model_validator(mode="after")
    def validate_receipt(self) -> "VideoSequenceContextReceipt":
        if self.method == "direct_context":
            if self.overlap_frames < 1:
                raise ValueError("direct video sequence context requires overlap")
            if self.schema_version != VIDEO_SEQUENCE_CONTEXT_SCHEMA_VERSION:
                raise ValueError("direct video sequence context requires the current schema")
            if self.source_working_artifact_id is None or self.source_working_operation_id is None:
                raise ValueError("direct video sequence context requires an exact working artifact")
            if self.video_context_frames not in {5, 22, 39, 56}:
                raise ValueError("direct video sequence context frame window is unsupported")
            if self.overlap_frames != self.video_context_frames:
                raise ValueError("direct video sequence overlap must match its context window")
            if self.audio_context_frames < 1:
                raise ValueError("direct video sequence audio context is required")
        if self.method == "reference_continuation" and (
            self.overlap_frames != 0
            or self.overlap_seconds != 0
            or self.source_working_artifact_id is not None
            or self.source_working_operation_id is not None
            or self.video_context_frames != 0
            or self.audio_context_frames != 0
        ):
            raise ValueError("reference video sequence context cannot claim direct overlap")
        if contains_secret_key(self.model_dump(mode="json")):
            raise ValueError("video sequence context cannot contain secret values")
        exclude = {"context_sha256"}
        if self.schema_version == VIDEO_SEQUENCE_CONTEXT_LEGACY_SCHEMA_VERSION:
            exclude.update(
                {
                    "source_working_artifact_id",
                    "source_working_operation_id",
                    "video_context_frames",
                    "audio_context_frames",
                }
            )
        payload = self.model_dump(mode="json", exclude=exclude)
        if self.context_sha256 != _hash_payload(payload):
            raise ValueError("video sequence context hash does not match the receipt")
        return self


class VideoSequenceContextPreparer:
    def prepare(
        self,
        request: VideoRequest,
        source: VideoSequenceSourceArtifact,
        adapter: VideoRuntimeAdapter,
        *,
        preferred_method: VideoContinuationMethod | None = None,
        capability: VideoSequenceCapability | None = None,
    ) -> VideoSequenceContextReceipt:
        if request.mode != "sequence":
            raise VideoSequenceContextError("video_sequence_request_required")
        if request.provider != adapter.provider_id:
            raise VideoSequenceContextError("video_provider_mismatch")
        if (
            source.width != request.size.width
            or source.height != request.size.height
            or source.fps != request.fps
        ):
            raise VideoSequenceContextError("video_sequence_context_incompatible")
        advertised = capability or adapter.sequence_capabilities()
        direct_context_frames = self._direct_context_frames(request, source)
        if (
            preferred_method == "direct_context"
            and source.working_artifact_id is not None
            and direct_context_frames is None
        ):
            raise VideoSequenceContextError("video_sequence_direct_context_too_short")
        method = self._select_method(
            advertised,
            preferred_method,
            source,
            direct_context_available=direct_context_frames is not None,
        )
        frame_range, overlap_frames = self._frame_range(
            method,
            source,
            direct_context_frames=direct_context_frames,
        )
        audio_context_frames = (
            DEFAULT_AUDIO_CONTEXT_FRAMES if method.method == "direct_context" else 0
        )
        audio_range = self._audio_range(
            frame_range,
            source,
            context_frames=(audio_context_frames or None),
        )
        overlap_seconds = overlap_frames / source.fps if overlap_frames else 0.0
        payload: dict[str, Any] = {
            "schema_version": VIDEO_SEQUENCE_CONTEXT_SCHEMA_VERSION,
            "method": method.method,
            "quality_description": method.quality_description,
            "source_artifact_id": source.artifact_id,
            "source_accepted_revision": source.accepted_revision,
            "source_sha256": source.sha256,
            "source_width": source.width,
            "source_height": source.height,
            "source_fps": source.fps,
            "source_frame_range": frame_range.model_dump(mode="json"),
            "source_audio_range": (
                audio_range.model_dump(mode="json") if audio_range is not None else None
            ),
            "source_working_artifact_id": (
                source.working_artifact_id if method.method == "direct_context" else None
            ),
            "source_working_operation_id": (
                source.working_operation_id if method.method == "direct_context" else None
            ),
            "video_context_frames": (
                overlap_frames if method.method == "direct_context" else 0
            ),
            "audio_context_frames": audio_context_frames,
            "overlap_frames": overlap_frames,
            "overlap_seconds": overlap_seconds,
            "adapter_id": adapter.id,
            "adapter_revision": adapter.revision,
            "mapping_id": method.mapping_id,
            "mapping_version": method.mapping_version,
        }
        payload["context_sha256"] = _hash_payload(payload)
        return VideoSequenceContextReceipt.model_validate(payload)

    def validate_source(
        self,
        receipt: VideoSequenceContextReceipt,
        source: VideoSequenceSourceArtifact,
    ) -> None:
        if (
            receipt.source_artifact_id != source.artifact_id
            or receipt.source_accepted_revision != source.accepted_revision
            or receipt.source_sha256 != source.sha256
            or receipt.source_width != source.width
            or receipt.source_height != source.height
            or receipt.source_fps != source.fps
            or (
                receipt.method == "direct_context"
                and (
                    receipt.source_working_artifact_id != source.working_artifact_id
                    or receipt.source_working_operation_id != source.working_operation_id
                )
            )
            or receipt.source_frame_range.end_frame > source.frame_count
            or (
                receipt.source_audio_range is not None
                and (
                    source.audio_sample_count is None
                    or source.audio_sample_rate != receipt.source_audio_range.sample_rate
                    or receipt.source_audio_range.end_sample > source.audio_sample_count
                )
            )
        ):
            raise VideoSequenceContextError("video_sequence_context_stale")

    @staticmethod
    def reconstruct_inputs(receipt: VideoSequenceContextReceipt) -> dict[str, Any]:
        return {
            "method": receipt.method,
            "source_artifact_id": receipt.source_artifact_id,
            "source_accepted_revision": receipt.source_accepted_revision,
            "source_sha256": receipt.source_sha256,
            "source_width": receipt.source_width,
            "source_height": receipt.source_height,
            "source_fps": receipt.source_fps,
            "source_frame_range": receipt.source_frame_range.model_dump(mode="json"),
            "source_audio_range": (
                receipt.source_audio_range.model_dump(mode="json")
                if receipt.source_audio_range is not None
                else None
            ),
            "source_working_artifact_id": receipt.source_working_artifact_id,
            "source_working_operation_id": receipt.source_working_operation_id,
            "video_context_frames": receipt.video_context_frames,
            "audio_context_frames": receipt.audio_context_frames,
            "overlap_frames": receipt.overlap_frames,
            "overlap_seconds": receipt.overlap_seconds,
            "adapter_id": receipt.adapter_id,
            "adapter_revision": receipt.adapter_revision,
            "mapping_id": receipt.mapping_id,
            "mapping_version": receipt.mapping_version,
            "context_sha256": receipt.context_sha256,
        }

    @staticmethod
    def _select_method(
        capability: VideoSequenceCapability,
        preferred_method: VideoContinuationMethod | None,
        source: VideoSequenceSourceArtifact,
        *,
        direct_context_available: bool,
    ) -> VideoContinuationMethodCapability:
        methods = {item.method: item for item in capability.methods}
        if preferred_method is not None:
            selected = methods.get(preferred_method)
            if selected is None or (
                preferred_method == "direct_context"
                and (
                    source.working_artifact_id is None
                    or not direct_context_available
                )
            ):
                raise VideoSequenceContextError("video_sequence_continuation_unavailable")
            return selected
        for method in ("direct_context", "reference_continuation"):
            selected = methods.get(method)
            if selected is not None and not (
                method == "direct_context"
                and (
                    source.working_artifact_id is None
                    or not direct_context_available
                )
            ):
                return selected
        raise VideoSequenceContextError("video_sequence_continuation_unavailable")

    @staticmethod
    def _frame_range(
        method: VideoContinuationMethodCapability,
        source: VideoSequenceSourceArtifact,
        *,
        direct_context_frames: int | None,
    ) -> tuple[VideoSequenceFrameRange, int]:
        if method.method == "reference_continuation":
            return VideoSequenceFrameRange(start_frame=0, end_frame=source.frame_count), 0
        if direct_context_frames is None:
            raise VideoSequenceContextError("video_sequence_direct_context_too_short")
        return (
            VideoSequenceFrameRange(
                start_frame=source.frame_count - direct_context_frames,
                end_frame=source.frame_count,
            ),
            direct_context_frames,
        )

    @staticmethod
    def _direct_context_frames(
        request: VideoRequest,
        source: VideoSequenceSourceArtifact,
    ) -> int | None:
        _, target_frames = normalized_h3_frame_count(
            request.duration_seconds,
            request.fps,
        )
        for window in _DIRECT_CONTEXT_WINDOWS:
            if window <= source.frame_count and window < target_frames:
                return window
        return None

    @staticmethod
    def _audio_range(
        frame_range: VideoSequenceFrameRange,
        source: VideoSequenceSourceArtifact,
        *,
        context_frames: int | None = None,
    ) -> VideoSequenceAudioRange | None:
        if source.audio_sample_rate is None or source.audio_sample_count is None:
            return None
        start_frame = (
            max(0, source.frame_count - min(context_frames, source.frame_count))
            if context_frames is not None
            else frame_range.start_frame
        )
        start_seconds = start_frame / source.fps
        end_seconds = frame_range.end_frame / source.fps
        start_sample = min(
            source.audio_sample_count - 1,
            round(start_seconds * source.audio_sample_rate),
        )
        end_sample = min(
            source.audio_sample_count,
            max(start_sample + 1, round(end_seconds * source.audio_sample_rate)),
        )
        return VideoSequenceAudioRange(
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=source.audio_sample_rate,
        )


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
