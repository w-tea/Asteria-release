from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.prompt_assist import (
    PromptAssistProviderModelsRequestData,
    PromptAssistProviderModelsResult,
    PromptAssistProviderTestRequestData,
    PromptAssistProviderTestResult,
    PromptAssistRequestData,
    PromptAssistResult,
)
from app.services.prompt_assist import (
    build_prompt_assist,
    list_prompt_assist_provider_models,
    test_prompt_assist_provider,
)


router = APIRouter(prefix="/api/prompt-assist", tags=["prompt-assist"])


PromptAssistRequest = PromptAssistRequestData
PromptAssistProviderTestRequest = PromptAssistProviderTestRequestData
PromptAssistProviderModelsRequest = PromptAssistProviderModelsRequestData


def _raise_sanitized_prompt_assist_error(exc: ValueError) -> None:
    code = str(exc).split(":", 1)[0]
    if not code.startswith("prompt_assist_provider_"):
        raise exc
    status_code = 502 if code in {
        "prompt_assist_provider_request_failed",
        "prompt_assist_provider_response_invalid",
        "prompt_assist_provider_response_empty",
    } else 422
    raise HTTPException(status_code=status_code, detail={"code": code}) from exc


@router.post("", response_model=PromptAssistResult)
def create_prompt_assist(request: PromptAssistRequest) -> PromptAssistResult:
    try:
        return build_prompt_assist(request)
    except ValueError as exc:
        _raise_sanitized_prompt_assist_error(exc)


@router.post("/providers/test", response_model=PromptAssistProviderTestResult)
def test_provider_connection(request: PromptAssistProviderTestRequest) -> PromptAssistProviderTestResult:
    try:
        return test_prompt_assist_provider(request)
    except ValueError as exc:
        _raise_sanitized_prompt_assist_error(exc)


@router.post("/providers/models", response_model=PromptAssistProviderModelsResult)
def list_provider_models(request: PromptAssistProviderModelsRequest) -> PromptAssistProviderModelsResult:
    try:
        return list_prompt_assist_provider_models(request)
    except ValueError as exc:
        _raise_sanitized_prompt_assist_error(exc)
