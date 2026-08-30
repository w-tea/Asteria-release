from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Literal, Mapping

from pydantic import BaseModel, Field, model_validator

from app.schemas.prompt_input import (
    GlobalPromptContent,
    PromptInput,
    PromptInputV2,
    PromptSectionId,
    TargetPromptContent,
)
from app.services.prompt_assist_presets import PromptAssistPurposePreset
from app.services.prompt_input import validate_prompt_input_pair


JSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
PromptAssistLanguage = Literal["en", "ja"]
PromptAssistTarget = Literal["enhance", "variations", "negative"]
PromptAssistProviderMode = Literal[
    "prompt_only",
    "openai_compatible",
    "openai_responses",
    "anthropic",
    "gemini",
]
PromptAssistConnectionProviderMode = Literal[
    "openai_compatible",
    "openai_responses",
    "anthropic",
    "gemini",
]


class PromptAssistSignaturePayload(BaseModel):
    prompt: str = ""
    prompt_input: PromptInput | None = None
    active_section_id: PromptSectionId | None = None
    active_section_text: str | None = None
    active_target_id: str | None = None
    active_target_text: str | None = None
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
    purpose_preset: PromptAssistPurposePreset = "standard"
    protected_tokens: list[str] = Field(default_factory=list, max_length=64)
    provider_mode: PromptAssistProviderMode = "prompt_only"
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


def _canonical_signature_value(value: object) -> object:
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("prompt assist signature does not accept non-finite numbers")
        return int(value) if value.is_integer() else value
    return value


def prompt_assist_structure_signature(
    value: PromptAssistSignaturePayload | Mapping[str, object],
) -> str:
    payload = (
        value
        if isinstance(value, PromptAssistSignaturePayload)
        else PromptAssistSignaturePayload.model_validate(dict(value))
    )
    canonical = _canonical_signature_value(payload.model_dump(mode="json"))
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_prompt_assist_section_contract(
    *,
    prompt: str,
    prompt_input: PromptInput | None,
    target: PromptAssistTarget,
    active_section_id: PromptSectionId | None,
    active_section_text: str | None,
    active_target_id: str | None = None,
    active_target_text: str | None = None,
    prompt_structure_revision: int | None = None,
    prompt_structure_signature: str | None = None,
    signature_payload: PromptAssistSignaturePayload | None = None,
) -> None:
    if prompt_input is None:
        if any(
            value is not None
            for value in (
                active_section_id,
                active_section_text,
                active_target_id,
                active_target_text,
                prompt_structure_revision,
                prompt_structure_signature,
            )
        ):
            raise ValueError("active prompt identity requires prompt_input")
        return

    validate_prompt_input_pair(prompt, prompt_input)
    if prompt_input.mode == "single":
        if any(
            value is not None
            for value in (
                active_section_id,
                active_section_text,
                active_target_id,
                active_target_text,
                prompt_structure_revision,
                prompt_structure_signature,
            )
        ):
            raise ValueError("single prompt input must not contain active prompt identity")
        return

    if not isinstance(prompt_input.root, PromptInputV2):
        if active_target_id is not None or active_target_text is not None:
            raise ValueError("V1 prompt input must not contain an active target")
        if prompt_structure_revision is not None or prompt_structure_signature is not None:
            raise ValueError("V1 prompt input must not contain V2 prompt identity")
        requires_active_section = target != "negative"
        if active_section_id is None or active_section_text is None:
            if (
                requires_active_section
                or active_section_id is not None
                or active_section_text is not None
            ):
                raise ValueError("section prompt assist requires a complete active section")
            return
        section = next(
            (item for item in prompt_input.sections or () if item.id == active_section_id),
            None,
        )
        if section is None or section.text != active_section_text:
            raise ValueError("active prompt section does not match prompt_input")
        return

    if prompt_structure_revision is None or prompt_structure_signature is None:
        raise ValueError("V2 prompt assist requires revision and signature")
    if target == "negative":
        if any(
            value is not None
            for value in (
                active_section_id,
                active_section_text,
                active_target_id,
                active_target_text,
            )
        ):
            raise ValueError("negative prompt assist must not contain active prompt identity")
    else:
        if active_section_id is None:
            raise ValueError("V2 prompt assist requires an active section")
        section = next(
            (item for item in prompt_input.sections or () if item.id == active_section_id),
            None,
        )
        if section is None:
            raise ValueError("active prompt section does not exist")
        if isinstance(section.content, GlobalPromptContent):
            if active_section_text != section.content.text:
                raise ValueError("active prompt section does not match prompt_input")
            if active_target_id is not None or active_target_text is not None:
                raise ValueError("global prompt section must not contain an active target")
        elif isinstance(section.content, TargetPromptContent):
            if active_section_text is not None:
                raise ValueError("target prompt section must not contain section text")
            if active_target_id is None or active_target_text is None:
                raise ValueError("target prompt assist requires a complete active target")
            active_target = next(
                (item for item in section.content.targets if item.id == active_target_id),
                None,
            )
            if active_target is None or active_target.text != active_target_text:
                raise ValueError("active prompt target does not match prompt_input")
        else:
            raise ValueError("active prompt section is unsupported")

    if signature_payload is None:
        raise ValueError("V2 prompt assist signature payload is missing")
    expected_signature = prompt_assist_structure_signature(signature_payload)
    if prompt_structure_signature != expected_signature:
        raise ValueError("prompt assist request identity is stale")


class PromptAssistRequestData(PromptAssistSignaturePayload):
    provider_api_key: str = ""
    prompt_structure_revision: int | None = Field(
        default=None,
        ge=0,
        le=JSON_SAFE_INTEGER_MAX,
        strict=True,
    )
    prompt_structure_signature: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_prompt_identity(self) -> "PromptAssistRequestData":
        validate_prompt_assist_section_contract(
            prompt=self.prompt,
            prompt_input=self.prompt_input,
            target=self.target,
            active_section_id=self.active_section_id,
            active_section_text=self.active_section_text,
            active_target_id=self.active_target_id,
            active_target_text=self.active_target_text,
            prompt_structure_revision=self.prompt_structure_revision,
            prompt_structure_signature=self.prompt_structure_signature,
            signature_payload=PromptAssistSignaturePayload.model_validate(
                self.model_dump(
                    exclude={
                        "provider_api_key",
                        "prompt_structure_revision",
                        "prompt_structure_signature",
                    }
                )
            ),
        )
        return self


class PromptAssistResult(BaseModel):
    provider_mode: PromptAssistProviderMode = "prompt_only"
    language: PromptAssistLanguage
    target: PromptAssistTarget
    active_section_id: PromptSectionId | None = None
    active_target_id: str | None = None
    prompt_structure_revision: int | None = None
    prompt_structure_signature: str | None = None
    purpose_preset: PromptAssistPurposePreset = "standard"
    system_prompt: str
    user_prompt: str
    suggested_prompt: str
    suggested_negative_prompt: str
    suggested_tags: list[str] = Field(default_factory=list)
    provider_model: str = ""
    provider_response: str = ""
    warnings: list[str] = Field(default_factory=list)


class PromptAssistProviderTestRequestData(BaseModel):
    provider_mode: PromptAssistConnectionProviderMode
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


class PromptAssistProviderTestResult(BaseModel):
    status: Literal["connected"] = "connected"
    provider_mode: PromptAssistConnectionProviderMode
    provider_model: str = ""
    latency_ms: int


class PromptAssistProviderModelsRequestData(BaseModel):
    provider_mode: PromptAssistConnectionProviderMode
    provider_endpoint: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


class PromptAssistProviderModelsResult(BaseModel):
    models: list[str] = Field(default_factory=list)
