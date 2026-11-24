"""Repeat controls reuse the ordinary image request, without a second sampler schema."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.action_contracts import ActionInput, GenerationDraftDocument, GenerationStart


class RepeatLimits(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_attempts: int = Field(default=0, ge=0, strict=True)
    max_minutes: int = Field(default=0, ge=0, strict=True)
    pass_target: int = Field(default=0, ge=0, strict=True)


class GenerationRepeatRecord(BaseModel):
    id: str
    status: str
    input_revision: int
    document: GenerationDraftDocument
    limits: RepeatLimits
    current_job_id: str | None = None
    latest_job_id: str | None = None
    attempted: int = 0
    passed: int = 0
    completed_cycles: int = 0
    elapsed_ms: int = 0
    stop_reason: str = ''
    error: Any = None


class RepeatStart(GenerationStart):
    limits: RepeatLimits = Field(default_factory=RepeatLimits)


class RepeatGet(ActionInput):
    repeat_id: str = Field(min_length=1)


class RepeatUpdate(RepeatGet):
    expected_revision: int = Field(ge=1, strict=True)
    document: GenerationDraftDocument


class RepeatStop(RepeatGet):
    stop_current: bool = False


class RepeatSubmit(ActionInput):
    request_id: str = Field(min_length=1)
    document: GenerationDraftDocument
    limits: RepeatLimits = Field(default_factory=RepeatLimits)


class RepeatInputUpdate(ActionInput):
    request_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1, strict=True)
    document: GenerationDraftDocument


class RepeatStopRequest(ActionInput):
    request_id: str = Field(min_length=1)
    stop_current: bool = False
