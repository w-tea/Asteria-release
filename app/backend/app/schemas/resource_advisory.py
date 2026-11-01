from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RESOURCE_ADVISORY_SCHEMA_VERSION = "asteria.resource.advisory.v1"
ResourceAdvisoryLevel = Literal["normal", "caution", "high_risk", "unknown"]
ResourceEstimateConfidence = Literal["high", "medium", "low", "unknown"]
RelativeExecutionClass = Literal["light", "moderate", "heavy", "extreme", "unknown"]


class ResourceByteCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_capacity(self) -> "ResourceByteCapacity":
        if self.available_bytes > self.total_bytes:
            raise ValueError("available resource bytes cannot exceed total bytes")
        return self


class ResourceObservationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vram: ResourceByteCapacity | None = None
    ram: ResourceByteCapacity | None = None
    temp_free_bytes: int | None = Field(default=None, ge=0)
    output_free_bytes: int | None = Field(default=None, ge=0)


class ResourceWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: Literal["video", "music"]
    provider_id: str = Field(min_length=1, max_length=192, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    adapter_id: str = Field(min_length=1, max_length=192, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    model_profile: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    task_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    batch: int = Field(default=1, gt=0)
    guide_count: int = Field(default=0, ge=0)
    guide_duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    audio: bool = False
    sequence_context: bool = False
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    offload: Literal["auto", "none", "balanced", "maximum", "layer_streaming"] = "auto"

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ResourceWorkload":
        if (self.width is None) != (self.height is None):
            raise ValueError("resource workload width and height must be provided together")
        if self.domain == "video" and (self.width is None or self.duration_seconds is None or self.fps is None):
            raise ValueError("video resource workload requires size, duration, and fps")
        return self


class ResourceEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    peak_vram_bytes: int | None = Field(default=None, ge=0)
    peak_ram_bytes: int | None = Field(default=None, ge=0)
    temporary_disk_bytes: int | None = Field(default=None, ge=0)
    output_bytes: int | None = Field(default=None, ge=0)
    relative_execution_class: RelativeExecutionClass


class ResourceAdvisoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=192, pattern=r"^[a-z0-9][a-z0-9_]*$")
    estimate_bytes: int | None = Field(default=None, ge=0)
    available_bytes: int | None = Field(default=None, ge=0)


class ResourceAdvisory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.resource.advisory.v1"] = RESOURCE_ADVISORY_SCHEMA_VERSION
    applicable: bool
    level: ResourceAdvisoryLevel
    confidence: ResourceEstimateConfidence
    can_run: bool
    acknowledgement_required: bool
    provider_id: str
    adapter_id: str
    model_profile: str
    profile_revision: str | None = None
    workload: ResourceWorkload
    observation: ResourceObservationSnapshot
    estimate: ResourceEstimate
    evidence: tuple[ResourceAdvisoryEvidence, ...]
    mitigation_codes: tuple[str, ...] = ()
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("mitigation_codes")
    @classmethod
    def validate_mitigation_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("resource mitigation codes must be unique")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> "ResourceAdvisory":
        if not self.can_run:
            raise ValueError("resource estimates cannot hard-block execution")
        if self.acknowledgement_required != (self.applicable and self.level == "high_risk"):
            raise ValueError("resource acknowledgement policy is inconsistent")
        return self


class ResourceAdvisoryAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advisory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_anyway: Literal[True]


class ResourceAdvisoryAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: Literal[True] = True
    advisory: ResourceAdvisory
    acknowledgement: ResourceAdvisoryAcknowledgement | None = None
