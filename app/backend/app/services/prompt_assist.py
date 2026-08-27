from __future__ import annotations

import json
import os
import re
import time
from typing import Literal
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from app.services.prompt_assist_presets import (
    PromptAssistPurposePreset,
    normalize_protected_tokens,
    prompt_assist_preset_instruction,
    restore_protected_tokens,
)


PromptAssistLanguage = Literal["en", "ja"]
PromptAssistTarget = Literal["enhance", "variations", "negative"]
PromptAssistProviderMode = Literal["prompt_only", "openai_compatible", "openai_responses", "anthropic", "gemini"]
PromptAssistConnectionProviderMode = Literal["openai_compatible", "openai_responses", "anthropic", "gemini"]


class PromptAssistRequestData(BaseModel):
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
    purpose_preset: PromptAssistPurposePreset = "standard"
    protected_tokens: list[str] = Field(default_factory=list, max_length=64)
    provider_mode: PromptAssistProviderMode = "prompt_only"
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0)


class PromptAssistResult(BaseModel):
    provider_mode: PromptAssistProviderMode = "prompt_only"
    language: PromptAssistLanguage
    target: PromptAssistTarget
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


def test_prompt_assist_provider(request: PromptAssistProviderTestRequestData) -> PromptAssistProviderTestResult:
    started_at = time.perf_counter()
    provider_key = _provider_api_key("", request.provider_api_key_env_var)
    _call_prompt_assist_provider(
        mode=request.provider_mode,
        endpoint=request.provider_endpoint,
        model=request.provider_model,
        api_key=provider_key,
        timeout_seconds=request.provider_timeout_seconds,
        system_prompt="Asteria AI provider connection test.",
        user_prompt="Reply with a short acknowledgement.",
    )
    return PromptAssistProviderTestResult(
        provider_mode=request.provider_mode,
        provider_model=request.provider_model.strip(),
        latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
    )


def list_prompt_assist_provider_models(
    request: PromptAssistProviderModelsRequestData,
) -> PromptAssistProviderModelsResult:
    if request.provider_mode not in {"openai_compatible", "openai_responses"}:
        raise ValueError("prompt_assist_provider_models_unsupported")
    endpoint = derive_openai_models_endpoint(request.provider_endpoint)
    provider_key = _provider_api_key("", request.provider_api_key_env_var)
    headers = {"Accept": "application/json"}
    if provider_key:
        headers["Authorization"] = f"Bearer {provider_key}"
    body = _get_provider_models_json(endpoint, headers, request.provider_timeout_seconds)
    return PromptAssistProviderModelsResult(models=_provider_model_ids(body))


def build_prompt_assist(request: PromptAssistRequestData) -> PromptAssistResult:
    if request.provider_mode == "prompt_only" and request.purpose_preset != "standard":
        raise ValueError("prompt_assist_provider_required_for_purpose")
    prompt = _compact(request.prompt)
    negative_prompt = _compact(request.negative_prompt)
    profile = _compact(request.profile).lower() or "sdxl"
    mode = _compact(request.mode).lower() or "txt2img"
    orientation = _orientation(request.width, request.height)
    warnings: list[str] = []
    if not prompt:
        warnings.append("prompt_empty")
    if request.width < 256 or request.height < 256:
        warnings.append("small_canvas")

    suggested_prompt = _suggest_prompt(prompt, profile, orientation, request.target)
    suggested_negative = _suggest_negative_prompt(negative_prompt, request.target)
    tags = _suggest_tags(profile, mode, orientation, request.loras, request.embeddings)
    protected_tokens = normalize_protected_tokens(request.protected_tokens)
    system_prompt = _system_prompt(request.language, request.purpose_preset)
    user_prompt = _user_prompt(
        request=request,
        prompt=prompt,
        negative_prompt=negative_prompt,
        profile=profile,
        mode=mode,
        orientation=orientation,
        suggested_prompt=suggested_prompt,
        suggested_negative=suggested_negative,
        protected_tokens=protected_tokens,
    )
    provider_response = ""
    provider_model = ""
    if request.provider_mode != "prompt_only":
        provider_key = _provider_api_key(request.provider_api_key, request.provider_api_key_env_var)
        provider_response = _call_prompt_assist_provider(
            mode=request.provider_mode,
            endpoint=request.provider_endpoint,
            model=request.provider_model,
            api_key=provider_key,
            timeout_seconds=request.provider_timeout_seconds,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        provider_model = request.provider_model.strip()
        parsed_prompt, parsed_negative, parsed_tags, parsed_warnings = _parse_prompt_provider_response(
            provider_response,
            fallback_prompt=suggested_prompt,
            fallback_negative=suggested_negative,
            fallback_tags=tags,
        )
        suggested_prompt = parsed_prompt
        suggested_negative = parsed_negative
        tags = parsed_tags
        warnings.extend(parsed_warnings)
    suggested_prompt, protected_tokens_restored = restore_protected_tokens(suggested_prompt, protected_tokens)
    if protected_tokens_restored:
        warnings.append("protected_tokens_restored")
    return PromptAssistResult(
        provider_mode=request.provider_mode,
        language=request.language,
        target=request.target,
        purpose_preset=request.purpose_preset,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        suggested_prompt=suggested_prompt,
        suggested_negative_prompt=suggested_negative,
        suggested_tags=tags,
        provider_model=provider_model,
        provider_response=provider_response,
        warnings=warnings,
    )


def _compact(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _append_terms(base: str, terms: list[str]) -> str:
    result = base
    lower = result.lower()
    for term in terms:
        if term.lower() not in lower:
            result = f"{result}, {term}" if result else term
            lower = result.lower()
    return result


def _suggest_prompt(prompt: str, profile: str, orientation: str, target: PromptAssistTarget) -> str:
    if target == "negative":
        return prompt
    base = prompt or "detailed subject"
    terms = ["clean composition", "controlled lighting"]
    if orientation == "portrait":
        terms.append("portrait framing")
    elif orientation == "landscape":
        terms.append("wide composition")
    if profile == "anima":
        terms.extend(["anime illustration", "crisp linework"])
    elif profile.startswith("flux"):
        terms.extend(["natural detail", "balanced texture"])
    else:
        terms.extend(["high detail", "sharp focus"])
    if target == "variations":
        terms.append("distinct alternate mood")
    return _append_terms(base, terms)


def _suggest_negative_prompt(negative_prompt: str, target: PromptAssistTarget) -> str:
    terms = ["low quality", "blurry", "distorted anatomy", "unreadable text"]
    if target == "negative":
        terms.extend(["overprocessed", "jpeg artifacts", "extra limbs"])
    return _append_terms(negative_prompt, terms)


def _suggest_tags(profile: str, mode: str, orientation: str, loras: list[str], embeddings: list[str]) -> list[str]:
    tags = [profile, mode, orientation]
    tags.extend(f"lora:{_compact(name)}" for name in loras if _compact(name))
    tags.extend(f"embedding:{_compact(name)}" for name in embeddings if _compact(name))
    return tags[:24]


def _system_prompt(language: PromptAssistLanguage, purpose_preset: PromptAssistPurposePreset) -> str:
    language_rule = (
        "Explain reasoning in Japanese when needed, but keep generation prompt tokens in English unless the user asks otherwise."
        if language == "ja"
        else "Use concise English and keep generation prompt tokens directly usable in Asteria."
    )
    return (
        "You are assisting Asteria image generation prompt drafting. "
        "Preserve the user's subject intent, do not invent unavailable model assets, and keep the result reproducible. "
        "Return compact JSON with prompt, negative_prompt, and notes. "
        f"Purpose-specific rules: {prompt_assist_preset_instruction(purpose_preset)} "
        f"{language_rule}"
    )


def _user_prompt(
    *,
    request: PromptAssistRequestData,
    prompt: str,
    negative_prompt: str,
    profile: str,
    mode: str,
    orientation: str,
    suggested_prompt: str,
    suggested_negative: str,
    protected_tokens: list[str],
) -> str:
    lines = [
        "Asteria generation context:",
        f"- Profile: {profile}",
        f"- Mode: {mode}",
        f"- Canvas: {request.width}x{request.height} ({orientation})",
        f"- Assist target: {request.target}",
        f"- Current prompt: {prompt or '(empty)'}",
        f"- Current negative prompt: {negative_prompt or '(empty)'}",
        f"- Active LoRAs: {', '.join(request.loras) if request.loras else '(none)'}",
        f"- Active embeddings: {', '.join(request.embeddings) if request.embeddings else '(none)'}",
        f"- Purpose preset: {request.purpose_preset}",
        f"- Protected LoRA trigger tokens: {', '.join(protected_tokens) if protected_tokens else '(none)'}",
        f"- Local draft prompt: {suggested_prompt}",
        f"- Local draft negative prompt: {suggested_negative}",
    ]
    if request.extra_instruction.strip():
        lines.append(f"- Extra instruction: {request.extra_instruction.strip()}")
    lines.extend(
        [
            "Return JSON only:",
            '{"prompt":"...","negative_prompt":"...","notes":["..."]}',
        ]
    )
    return "\n".join(lines)


def _provider_api_key(api_key: str, api_key_env_var: str) -> str:
    direct_value = api_key.strip()
    if direct_value:
        return direct_value
    env_name = api_key_env_var.strip()
    if not env_name:
        return ""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("prompt_assist_provider_api_key_env_var_invalid")
    return os.environ.get(env_name, "").strip()


def _call_prompt_assist_provider(
    *,
    mode: PromptAssistProviderMode,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if mode == "openai_compatible":
        return _call_openai_compatible_provider(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if mode == "openai_responses":
        return _call_openai_responses_provider(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if mode == "anthropic":
        return _call_anthropic_provider(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if mode == "gemini":
        return _call_gemini_provider(
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    return ""


def _call_openai_compatible_provider(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    if not model_value:
        raise ValueError("prompt_assist_provider_model_required")
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    payload = {
        "model": model_value,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    return _post_json_and_extract_text(endpoint_value, headers, payload, timeout_seconds, _openai_compatible_response_text)


def _call_openai_responses_provider(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    api_key_value = api_key.strip()
    if not model_value:
        raise ValueError("prompt_assist_provider_model_required")
    if not api_key_value:
        raise ValueError("prompt_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_value}",
    }
    payload = {
        "model": model_value,
        "instructions": system_prompt,
        "input": user_prompt,
        "temperature": 0.2,
    }
    return _post_json_and_extract_text(endpoint_value, headers, payload, timeout_seconds, _openai_responses_response_text)


def _call_anthropic_provider(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    api_key_value = api_key.strip()
    if not model_value:
        raise ValueError("prompt_assist_provider_model_required")
    if not api_key_value:
        raise ValueError("prompt_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key_value,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model_value,
        "max_tokens": 800,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    return _post_json_and_extract_text(endpoint_value, headers, payload, timeout_seconds, _anthropic_response_text)


def _call_gemini_provider(
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    api_key_value = api_key.strip()
    if not api_key_value:
        raise ValueError("prompt_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key_value,
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    return _post_json_and_extract_text(endpoint_value, headers, payload, timeout_seconds, _gemini_response_text)


def _post_json_and_extract_text(
    endpoint: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: float,
    extractor,
) -> str:
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"prompt_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"prompt_assist_provider_response_invalid: {exc}") from exc
    content = extractor(body)
    if not content.strip():
        raise ValueError("prompt_assist_provider_response_empty")
    return content.strip()


def derive_openai_models_endpoint(endpoint: str) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    parsed_url = urlparse(endpoint_value)
    path = parsed_url.path.rstrip("/")
    suffixes = ("/v1/chat/completions", "/v1/responses", "/v1")
    matched_suffix = next((suffix for suffix in suffixes if path.endswith(suffix)), None)
    if matched_suffix is None:
        raise ValueError("prompt_assist_provider_models_endpoint_unsupported")
    prefix = path[: -len(matched_suffix)]
    models_path = f"{prefix}/v1/models"
    return urlunparse(parsed_url._replace(path=models_path, params="", query="", fragment=""))


def _get_provider_models_json(endpoint: str, headers: dict[str, str], timeout_seconds: float) -> object:
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.get(endpoint, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"prompt_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"prompt_assist_provider_response_invalid: {exc}") from exc


def _provider_model_ids(body: object) -> list[str]:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise ValueError("prompt_assist_provider_response_invalid")
    embedding_types = {"embedding", "embeddings", "text_embedding", "text-embedding"}
    models: set[str] = set()
    for item in body["data"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or len(model_id) > 256:
            continue
        model_type = item.get("type")
        if isinstance(model_type, str) and model_type.strip().casefold() in embedding_types:
            continue
        models.add(model_id)
    return sorted(models)


def _validated_provider_endpoint(endpoint: str) -> str:
    endpoint_value = endpoint.strip()
    if not endpoint_value:
        raise ValueError("prompt_assist_provider_endpoint_required")
    parsed_url = urlparse(endpoint_value)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("prompt_assist_provider_endpoint_invalid")
    return endpoint_value


def _provider_timeout(timeout_seconds: float) -> float:
    return max(1.0, min(300.0, timeout_seconds))


def _openai_compatible_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices_missing")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choice_not_object")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
            if parts:
                return "\n".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("message_content_missing")


def _openai_responses_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = body.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("output_missing")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if isinstance(content_item, dict) and content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                parts.append(content_item["text"])
    if parts:
        return "\n".join(parts)
    raise ValueError("output_text_missing")


def _anthropic_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    content = body.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("content_missing")
    parts = [item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)]
    if parts:
        return "\n".join(parts)
    raise ValueError("content_text_missing")


def _gemini_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates_missing")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("candidate_not_object")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ValueError("candidate_content_missing")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("candidate_parts_missing")
    text_parts = [item["text"] for item in parts if isinstance(item, dict) and isinstance(item.get("text"), str)]
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError("candidate_text_missing")


def _parse_prompt_provider_response(
    response_text: str,
    *,
    fallback_prompt: str,
    fallback_negative: str,
    fallback_tags: list[str],
) -> tuple[str, str, list[str], list[str]]:
    stripped = response_text.strip()
    warnings: list[str] = []
    parsed = _provider_response_json_object(stripped)
    if isinstance(parsed, dict):
        prompt = str(parsed.get("prompt") or parsed.get("positive_prompt") or fallback_prompt).strip() or fallback_prompt
        negative = str(parsed.get("negative_prompt") or parsed.get("negative") or fallback_negative).strip() or fallback_negative
        tags_value = parsed.get("tags", fallback_tags)
        if isinstance(tags_value, str):
            tags = _split_tags(tags_value)
        elif isinstance(tags_value, list):
            tags = [str(item).strip() for item in tags_value if str(item).strip()]
        else:
            tags = fallback_tags
        warning_value = parsed.get("warnings", parsed.get("notes", []))
        if isinstance(warning_value, str) and warning_value.strip():
            warnings.append(warning_value.strip())
        elif isinstance(warning_value, list):
            warnings.extend(str(item).strip() for item in warning_value if str(item).strip())
        return prompt, negative, tags[:24], warnings

    prompt_match = re.search(r"(?im)^\s*(?:prompt|positive_prompt|プロンプト)\s*[:：]\s*(.+)$", stripped)
    negative_match = re.search(r"(?im)^\s*(?:negative_prompt|negative|ネガティブ)\s*[:：]\s*(.+)$", stripped)
    tags_match = re.search(r"(?im)^\s*(?:tags?|タグ)\s*[:：]\s*(.+)$", stripped)
    prompt = prompt_match.group(1).strip() if prompt_match else fallback_prompt
    negative = negative_match.group(1).strip() if negative_match else fallback_negative
    tags = _split_tags(tags_match.group(1)) if tags_match else fallback_tags
    if not prompt_match and stripped:
        warnings.append("provider_response_unstructured")
    return prompt, negative, tags[:24], warnings


def _provider_response_json_object(response_text: str) -> dict | None:
    candidates = [response_text]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", response_text))
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(response_text[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _split_tags(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,、;；]", value) if item.strip()]
