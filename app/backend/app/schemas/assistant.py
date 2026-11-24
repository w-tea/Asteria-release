"""Durable chat state, separate from generation jobs and external MCP clients."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.action_contracts import MediaSourceTarget


TurnStatus = Literal['running', 'completed', 'cancelled', 'failed', 'interrupted']


class Conversation(BaseModel):
    id: str
    revision: int
    title: str
    provider_id: str
    model: str
    created_at: str
    updated_at: str


class AssistantTurn(BaseModel):
    id: str
    conversation_id: str
    prompt: str
    provider_id: str
    model: str
    status: TurnStatus
    created_at: str
    updated_at: str


class AssistantEvent(BaseModel):
    sequence: int
    kind: str
    payload: dict[str, Any]
    created_at: str


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(min_length=1)
    title: str = ''
    provider_id: str
    model: str


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=1)
    title: str
    provider_id: str
    model: str


class AssistantTurnCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    provider_id: str
    model: str
    attachments: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    media_attachments: list[MediaSourceTarget] = Field(default_factory=list)


class AssistantDraftAttach(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(min_length=1)
    draft_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1, strict=True)
