from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.prompt_assist import (
    PromptAssistConnectionProviderMode,
    PromptAssistLanguage,
    PromptAssistProviderMode,
    PromptAssistProviderTestRequestData,
    PromptAssistProviderTestResult,
    PromptAssistRequestData,
    PromptAssistResult,
    PromptAssistTarget,
    build_prompt_assist,
    test_prompt_assist_provider,
)


router = APIRouter(prefix="/api/prompt-assist", tags=["prompt-assist"])


class PromptAssistRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    profile: str = "sdxl"
    mode: str = "txt2img"
    width: int = Field(default=832, ge=64, le=8192)
    height: int = Field(default=1216, ge=64, le=8192)
    loras: list[str] = Field(default_factory=list)
    embeddings: list[str] = Field(default_factory=list)
    target: PromptAssistTarget = "enhance"
    language: PromptAssistLanguage = "en"
    extra_instruction: str = ""
    provider_mode: PromptAssistProviderMode = "prompt_only"
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


class PromptAssistProviderTestRequest(BaseModel):
    provider_mode: PromptAssistConnectionProviderMode
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


@router.post("", response_model=PromptAssistResult)
def create_prompt_assist(request: PromptAssistRequest) -> PromptAssistResult:
    try:
        return build_prompt_assist(PromptAssistRequestData(**request.model_dump()))
    except ValueError as exc:
        detail = str(exc)
        if detail.startswith("prompt_assist_provider_"):
            raise HTTPException(status_code=422, detail=detail) from exc
        raise


@router.post("/providers/test", response_model=PromptAssistProviderTestResult)
def test_provider_connection(request: PromptAssistProviderTestRequest) -> PromptAssistProviderTestResult:
    try:
        return test_prompt_assist_provider(PromptAssistProviderTestRequestData(**request.model_dump()))
    except ValueError as exc:
        code = str(exc).split(":", 1)[0]
        if not code.startswith("prompt_assist_provider_"):
            raise
        status_code = 502 if code in {
            "prompt_assist_provider_request_failed",
            "prompt_assist_provider_response_invalid",
            "prompt_assist_provider_response_empty",
        } else 422
        raise HTTPException(status_code=status_code, detail={"code": code}) from exc
