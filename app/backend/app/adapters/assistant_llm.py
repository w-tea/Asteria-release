"""Conversation protocol values; provider HTTP transport is independent of tools."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import httpx

from app.services.prompt_assist import _provider_api_key
from app.services.user_settings import AIProviderSettings


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMReply:
    text: str
    calls: list[ToolCall]
    raw: dict[str, Any] = field(default_factory=dict)
    provider_mode: str = ''


class AssistantLLMError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None, message: str = ''):
        super().__init__(code)
        self.detail = {'code': code, 'status_code': status_code, 'message': message}


def build_request(provider, messages, tools, *, api_key):
    from app.adapters.assistant_llm_protocol import build_request as build
    return build(provider, messages, tools, api_key=api_key)


async def _sse(response):
    data: list[str] = []
    event_name = ''
    async for line in response.aiter_lines():
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif line.startswith('event:'):
            event_name = line[6:].strip()
        elif not line:
            if data:
                raw = '\n'.join(data)
                if raw == '[DONE]':
                    yield {'type': 'transport_done'}
                else:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise AssistantLLMError('assistant_response_invalid')
                    if event_name:
                        value.setdefault('type', event_name)
                    yield value
            data, event_name = [], ''


class AssistantLLM:
    def __init__(self, provider: AIProviderSettings, *, client: httpx.AsyncClient | None = None):
        self.provider, self.client = provider, client

    async def stream(self, messages, tools):
        from app.adapters.assistant_llm_protocol import StreamAccumulator
        key = _provider_api_key('', self.provider.api_key_env_var)
        url, headers, payload = build_request(self.provider, messages, tools, api_key=key)
        accumulator = StreamAccumulator(self.provider.provider_mode)
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.provider.timeout_seconds)
        try:
            async with client.stream('POST', url, headers=headers, json=payload,
                                     timeout=self.provider.timeout_seconds) as response:
                if response.is_error:
                    await response.aread()
                    message = ''
                    try:
                        error = response.json().get('error', {})
                        message = str(error.get('message', '')) if isinstance(error, dict) else str(error)
                    except (ValueError, AttributeError):
                        pass
                    raise AssistantLLMError('assistant_provider_request_failed', status_code=response.status_code, message=message)
                async for item in _sse(response):
                    for text in accumulator.add(item):
                        yield TextDelta(text)
                yield accumulator.reply()
        except AssistantLLMError as exc:
            if key:
                exc.detail['message'] = exc.detail['message'].replace(key, '[redacted]')
            raise
        except httpx.TimeoutException as exc:
            raise AssistantLLMError('assistant_provider_timeout') from exc
        except httpx.HTTPError as exc:
            raise AssistantLLMError('assistant_provider_connection_failed') from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise AssistantLLMError('assistant_response_invalid') from exc
        finally:
            if own_client:
                await client.aclose()
