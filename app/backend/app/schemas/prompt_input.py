from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROMPT_INPUT_SCHEMA_VERSION = "asteria.prompt-input.v1"
PROMPT_EDGE_CHARS = " \t\n"

PromptSectionId = Literal["base", "character", "detail"]
PromptInputMode = Literal["single", "sections"]


def normalize_prompt_contract_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip(
        PROMPT_EDGE_CHARS
    )


class PromptSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: PromptSectionId
    label: str
    text: str = ""
    order: int = Field(ge=0, le=2, strict=True)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = normalize_prompt_contract_text(value)
        if not 1 <= len(normalized) <= 40:
            raise ValueError("prompt section label must contain 1 to 40 characters")
        return normalized


class PromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.prompt-input.v1"]
    mode: PromptInputMode
    sections: tuple[PromptSection, ...] | None = None

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "PromptInput":
        if self.mode == "single":
            if self.sections is not None:
                raise ValueError("single prompt input must not contain sections")
            return self

        sections = self.sections
        if sections is None or len(sections) != 3:
            raise ValueError("section prompt input requires three sections")

        section_ids = [section.id for section in sections]
        if len(set(section_ids)) != 3 or set(section_ids) != {
            "base",
            "character",
            "detail",
        }:
            raise ValueError("section prompt input requires each fixed section")

        orders = [section.order for section in sections]
        if len(set(orders)) != 3 or set(orders) != {0, 1, 2}:
            raise ValueError("section prompt input requires orders zero through two")

        if not any(normalize_prompt_contract_text(section.text) for section in sections):
            raise ValueError("section prompt input requires prompt text")
        return self
