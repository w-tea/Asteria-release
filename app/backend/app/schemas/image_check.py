from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ImageCheckSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    enabled: bool = False
    method: Literal["native", "llm", "codex"] = "llm"
    provider_id: str = Field(default="", max_length=128, pattern=r"^[A-Za-z0-9_.-]*$")
    save_rejected: bool = False
    codex_model: str = Field(default="gpt-6.1-sol", max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    codex_timeout_seconds: int = Field(default=120, ge=10, le=300)

    @model_validator(mode="after")
    def require_selected_provider(self) -> ImageCheckSettings:
        if self.enabled and self.method == "llm" and not self.provider_id:
            raise ValueError("image_check_provider_required")
        return self


class ImageCheckFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["extra_limb", "missing_limb", "impossible_joint", "body_collapse", "hand_deformation"]
    body_part: Literal["hand", "arm", "leg", "foot", "torso", "head", "whole_body"]
    description: str = Field(min_length=1, max_length=500)
    person_index: int | None = Field(default=None, ge=1, le=64)

    @field_validator("description")
    @classmethod
    def require_visible_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image_check_visible_evidence_required")
        return value


class ImageCheckAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["asteria.image-check.v1"] = "asteria.image-check.v1"
    verdict: Literal["pass", "reject", "unknown"]
    findings: tuple[ImageCheckFinding, ...] = Field(default=(), max_length=32)
    reason_code: str = Field(default="", max_length=96, pattern=r"^(?:[a-z][a-z0-9_]*|)$")

    @model_validator(mode="after")
    def require_consistent_evidence(self) -> ImageCheckAssessment:
        if self.verdict == "reject" and not self.findings:
            raise ValueError("image_check_reject_evidence_required")
        if self.verdict != "reject" and self.findings:
            raise ValueError("image_check_verdict_evidence_conflict")
        if self.verdict == "unknown" and not self.reason_code:
            raise ValueError("image_check_unknown_reason_required")
        if self.verdict != "unknown" and self.reason_code:
            raise ValueError("image_check_verdict_reason_conflict")
        return self


class ImageCheckDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["disabled", "pass", "reject", "unknown"]
    continue_postprocessing: bool = False
    save_rejected_image: bool = False
    continue_repeat: bool = False
    pause: bool = False
    reason_code: str = ""


class ImageCheckTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    generation_id: int = Field(ge=1)
    output_path: str = Field(min_length=1, max_length=4096)
    image_check: ImageCheckSettings

    @model_validator(mode="after")
    def require_available_method(self) -> ImageCheckTestRequest:
        if self.image_check.method == "native":
            raise ValueError("image_check_method_not_supported")
        if self.image_check.method == "llm" and not self.image_check.provider_id:
            raise ValueError("image_check_provider_required")
        return self


class ImageCheckTestResponse(BaseModel):
    generation_id: int
    output_path: str
    assessment: ImageCheckAssessment
    receipt: dict
    source_current: bool
