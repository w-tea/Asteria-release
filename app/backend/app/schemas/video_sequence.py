from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.schemas.video import (
    FrozenJsonObject,
    VideoProvider,
    VideoRequest,
    contains_secret_key,
    freeze_json_object,
)


VIDEO_SEQUENCE_SCHEMA_VERSION = "asteria.video.sequence.v1"
VideoSequenceStatus = Literal[
    "draft",
    "generating",
    "review",
    "ready_to_continue",
    "assembling",
    "completed",
    "failed",
    "interrupted",
]
VideoSequenceSegmentStatus = Literal[
    "draft",
    "generating",
    "review",
    "accepted",
    "retryable",
    "needs_regeneration",
    "failed",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


def _validate_identifier(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} is invalid")
    return normalized


class VideoSequenceSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    revision: int = Field(default=1, ge=1)
    ordinal: int = Field(ge=0)
    parent_segment_revision: int | None = Field(default=None, ge=1)
    request_snapshot: VideoRequest
    candidate_artifact_ids: tuple[str, ...] = ()
    candidate_generation_ids: FrozenJsonObject = Field(default_factory=dict, validate_default=True)
    candidate_context_receipts: FrozenJsonObject = Field(default_factory=dict, validate_default=True)
    candidate_working_artifacts: FrozenJsonObject = Field(default_factory=dict, validate_default=True)
    accepted_candidate_id: str | None = None
    accepted_revision: int | None = Field(default=None, ge=1)
    context_receipt: FrozenJsonObject | None = None
    operation_id: str | None = None
    status: VideoSequenceSegmentStatus

    @field_validator("segment_id", "operation_id")
    @classmethod
    def validate_segment_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, label="video sequence segment identifier")

    @field_validator("candidate_artifact_ids")
    @classmethod
    def validate_candidate_artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _validate_identifier(item, label="video sequence candidate artifact identifier")
            for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("video sequence candidate artifact identifiers must be unique")
        return normalized

    @field_validator("candidate_generation_ids")
    @classmethod
    def freeze_candidate_generation_ids(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        for artifact_id, generation_id in value.items():
            _validate_identifier(
                artifact_id,
                label="video sequence candidate generation artifact identifier",
            )
            if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 1:
                raise ValueError("video sequence candidate generation identifier is invalid")
        return freeze_json_object(value)

    @field_validator("accepted_candidate_id")
    @classmethod
    def validate_accepted_candidate_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, label="video sequence accepted candidate identifier")

    @field_validator("context_receipt")
    @classmethod
    def freeze_context_receipt(cls, value: dict | None) -> dict | None:
        if value is None:
            return None
        if contains_secret_key(value):
            raise ValueError("video sequence context receipt cannot contain secret values")
        return freeze_json_object(value)

    @field_validator("candidate_context_receipts")
    @classmethod
    def freeze_candidate_context_receipts(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if contains_secret_key(value):
            raise ValueError("video sequence candidate receipts cannot contain secret values")
        for artifact_id, receipt in value.items():
            _validate_identifier(
                artifact_id,
                label="video sequence candidate receipt artifact identifier",
            )
            if not isinstance(receipt, dict):
                raise ValueError("video sequence candidate receipt must be a JSON object")
        return freeze_json_object(value)

    @field_validator("candidate_working_artifacts")
    @classmethod
    def freeze_candidate_working_artifacts(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if contains_secret_key(value):
            raise ValueError("video sequence working artifact receipts cannot contain secret values")
        for candidate_id, receipt in value.items():
            _validate_identifier(
                candidate_id,
                label="video sequence working artifact candidate identifier",
            )
            if not isinstance(receipt, dict) or set(receipt) != {
                "artifact_id",
                "operation_id",
                "generation_id",
                "kind",
            }:
                raise ValueError("video sequence working artifact receipt is invalid")
            _validate_identifier(
                str(receipt.get("artifact_id") or ""),
                label="video sequence working artifact identifier",
            )
            _validate_identifier(
                str(receipt.get("operation_id") or ""),
                label="video sequence working artifact operation",
            )
            generation_id = receipt.get("generation_id")
            if (
                isinstance(generation_id, bool)
                or not isinstance(generation_id, int)
                or generation_id < 1
                or receipt.get("kind") != "av_latent"
            ):
                raise ValueError("video sequence working artifact receipt is invalid")
        return freeze_json_object(value)

    @model_validator(mode="after")
    def validate_segment_contract(self) -> "VideoSequenceSegment":
        if self.request_snapshot.mode != "sequence":
            raise ValueError("video sequence segment requires a sequence request")
        if self.ordinal == 0 and self.parent_segment_revision is not None:
            raise ValueError("first video sequence segment cannot have a parent revision")
        if self.ordinal > 0 and self.parent_segment_revision is None:
            raise ValueError("non-initial video sequence segment requires a parent revision")
        if (self.accepted_candidate_id is None) != (self.accepted_revision is None):
            raise ValueError("accepted candidate identity and revision must be recorded together")
        if (
            self.accepted_candidate_id is not None
            and self.accepted_candidate_id not in self.candidate_artifact_ids
        ):
            raise ValueError("accepted candidate must be present in the candidate list")
        if not set(self.candidate_context_receipts).issubset(self.candidate_artifact_ids):
            raise ValueError("candidate receipts must belong to the candidate list")
        if not set(self.candidate_generation_ids).issubset(self.candidate_artifact_ids):
            raise ValueError("candidate generation identities must belong to the candidate list")
        if not set(self.candidate_working_artifacts).issubset(self.candidate_artifact_ids):
            raise ValueError("candidate working artifacts must belong to the candidate list")
        for candidate_id, receipt in self.candidate_working_artifacts.items():
            generation_id = self.candidate_generation_ids.get(candidate_id)
            if generation_id is None or receipt.get("generation_id") != generation_id:
                raise ValueError("candidate working artifact generation does not match")
        if self.status == "accepted" and self.accepted_candidate_id is None:
            raise ValueError("accepted candidate identity and revision are required")
        if self.status == "review" and not self.candidate_artifact_ids:
            raise ValueError("review segment requires at least one candidate")
        return self


class VideoSequence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video.sequence.v1"] = VIDEO_SEQUENCE_SCHEMA_VERSION
    sequence_id: str
    provider: VideoProvider
    model_profile: str
    shared_settings_snapshot: FrozenJsonObject
    capability_snapshot: FrozenJsonObject
    active_chain_revision: int = Field(ge=1)
    active_segment_ids: tuple[str, ...] = ()
    assembly_operation_id: str | None = None
    assembly_generation_id: int | None = Field(default=None, ge=1)
    assembly_revision: int = Field(default=0, ge=0)
    assembled_artifact_id: str | None = None
    status: VideoSequenceStatus
    segments: tuple[VideoSequenceSegment, ...]
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("sequence_id", "assembly_operation_id", "assembled_artifact_id")
    @classmethod
    def validate_sequence_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, label="video sequence identifier")

    @field_validator("model_profile")
    @classmethod
    def validate_model_profile(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("video sequence model profile is invalid")
        return normalized

    @field_validator("shared_settings_snapshot", "capability_snapshot")
    @classmethod
    def freeze_snapshot(cls, value: dict) -> dict:
        if contains_secret_key(value):
            raise ValueError("video sequence snapshot cannot contain secret values")
        return freeze_json_object(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("video sequence timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_sequence_contract(self) -> "VideoSequence":
        if not self.segments:
            raise ValueError("video sequence requires at least one segment")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("video sequence segment identifiers must be unique")
        if tuple(segment.ordinal for segment in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("video sequence segment ordinals must be contiguous")
        if len(self.active_segment_ids) != len(set(self.active_segment_ids)):
            raise ValueError("active video sequence segment identifiers must be unique")
        if self.active_segment_ids != segment_ids[: len(self.active_segment_ids)]:
            raise ValueError("active video sequence segments must form an ordered prefix")
        if self.updated_at < self.created_at:
            raise ValueError("video sequence updated timestamp cannot precede creation")
        if self.assembly_generation_id is not None and self.assembly_operation_id is None:
            raise ValueError("video sequence assembly generation requires an operation")
        if self.assembly_operation_id is not None and self.assembly_revision < 1:
            raise ValueError("video sequence assembly revision is required")

        for segment in self.segments:
            if segment.request_snapshot.provider != self.provider:
                raise ValueError("video sequence segment provider does not match the sequence")
            if segment.request_snapshot.model_profile != self.model_profile:
                raise ValueError("video sequence segment model profile does not match the sequence")

        previous: VideoSequenceSegment | None = None
        for active_id in self.active_segment_ids:
            segment = self.segments[segment_ids.index(active_id)]
            if segment.status != "accepted":
                raise ValueError("active video sequence segments must be accepted")
            if previous is not None and segment.parent_segment_revision != previous.accepted_revision:
                raise ValueError("active parent revision does not match the accepted predecessor")
            previous = segment
        return self
