from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


WORKING_ARTIFACT_SCHEMA_VERSION = "asteria.generation-working-artifact.v1"
WORKING_ARTIFACT_MANIFEST_SCHEMA_VERSION = "asteria.generation-working-artifact-manifest.v1"

WorkingArtifactKind = Literal[
    "av_latent",
    "conditioning",
    "noise",
    "sigmas",
    "motion_context",
    "upscale_work_bundle",
]
WorkingArtifactState = Literal[
    "writing",
    "available",
    "in_use",
    "consumed",
    "discarded",
    "superseded",
]
WorkingArtifactRetirement = Literal["consumed", "discarded", "superseded"]

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_working_artifact_token(value: str, label: str) -> str:
    normalized = value.strip()
    if not _TOKEN.fullmatch(normalized) or ".." in normalized:
        raise ValueError(f"{label} is invalid")
    return normalized


class GenerationWorkingArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.generation-working-artifact.v1"] = (
        WORKING_ARTIFACT_SCHEMA_VERSION
    )
    artifact_id: str
    operation_id: str
    generation_id: int = Field(ge=1)
    media_artifact_id: str | None = None
    kind: WorkingArtifactKind
    state: WorkingArtifactState
    relative_path: str | None
    sha256: str
    size_bytes: int = Field(ge=0)
    source_sha256: tuple[str, ...] = ()
    conditioning_sha256: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("artifact_id", "operation_id")
    @classmethod
    def _validate_token(cls, value: str) -> str:
        return validate_working_artifact_token(value, "working artifact token")

    @field_validator("media_artifact_id")
    @classmethod
    def _validate_optional_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_working_artifact_token(value, "media artifact token")

    @field_validator("sha256", "conditioning_sha256")
    @classmethod
    def _validate_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("working artifact hash is invalid")
        return normalized

    @field_validator("source_sha256")
    @classmethod
    def _validate_source_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.lower() for item in value)
        if any(not _SHA256.fullmatch(item) for item in normalized):
            raise ValueError("working artifact source hash is invalid")
        return normalized

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "payload"
            or any(part in {"", ".", ".."} for part in path.parts)
            or len(path.parts) != 2
        ):
            raise ValueError("working artifact relative path is invalid")
        return path.as_posix()

    @model_validator(mode="after")
    def _validate_lifecycle(self) -> "GenerationWorkingArtifactRecord":
        for value in (self.created_at, self.updated_at):
            if value.tzinfo is None:
                raise ValueError("working artifact timestamp requires timezone")
        if self.updated_at < self.created_at:
            raise ValueError("working artifact timestamp order is invalid")
        if self.state in {"writing", "available", "in_use"} and self.relative_path is None:
            raise ValueError("live working artifact requires a relative path")
        return self


class GenerationWorkingArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.generation-working-artifact-manifest.v1"] = (
        WORKING_ARTIFACT_MANIFEST_SCHEMA_VERSION
    )
    operation_id: str
    actual_output_root: str
    created_at: datetime
    updated_at: datetime
    artifacts: tuple[GenerationWorkingArtifactRecord, ...] = ()

    @field_validator("operation_id")
    @classmethod
    def _validate_operation_id(cls, value: str) -> str:
        return validate_working_artifact_token(value, "working artifact operation")

    @model_validator(mode="after")
    def _validate_manifest(self) -> "GenerationWorkingArtifactManifest":
        if not self.actual_output_root.strip():
            raise ValueError("working artifact output root is empty")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("working artifact manifest timestamp requires timezone")
        if self.updated_at < self.created_at:
            raise ValueError("working artifact manifest timestamp order is invalid")
        ids = tuple(item.artifact_id for item in self.artifacts)
        if len(ids) != len(set(ids)):
            raise ValueError("working artifact identifiers must be unique")
        if any(item.operation_id != self.operation_id for item in self.artifacts):
            raise ValueError("working artifact operation ownership mismatch")
        return self


class GenerationWorkingArtifactSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    generation_id: int = Field(ge=1)
    media_artifact_id: str | None = None
    kind: WorkingArtifactKind
    state: WorkingArtifactState
    size_bytes: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @field_validator("artifact_id")
    @classmethod
    def _validate_artifact_id(cls, value: str) -> str:
        return validate_working_artifact_token(value, "working artifact token")

    @field_validator("media_artifact_id")
    @classmethod
    def _validate_media_artifact_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_working_artifact_token(value, "media artifact token")

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "GenerationWorkingArtifactSummary":
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("working artifact summary timestamp requires timezone")
        if self.updated_at < self.created_at:
            raise ValueError("working artifact summary timestamp order is invalid")
        return self


class GenerationWorkingArtifactCleanupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scanned_operations: int = Field(ge=0)
    retained_active_operations: int = Field(ge=0)
    removed_payloads: int = Field(ge=0)
    recovered_stale_artifacts: int = Field(ge=0)
    removed_orphans: int = Field(ge=0)
