from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from app.schemas.prompt_layout import TARGET_ID_PATTERN


PROMPT_INPUT_SCHEMA_VERSION = "asteria.prompt-input.v1"
PROMPT_INPUT_V2_SCHEMA_VERSION = "asteria.prompt-input.v2"
PROMPT_EDGE_CHARS = " \t\n"

PromptSectionId = Literal["base", "character", "detail"]
PromptInputMode = Literal["single", "sections"]


def normalize_prompt_contract_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip(
        PROMPT_EDGE_CHARS
    )


class PromptSection(BaseModel):
    """The unchanged V1 section contract."""

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


class PromptInputV1(BaseModel):
    """The unchanged V1 prompt-input payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.prompt-input.v1"]
    mode: PromptInputMode
    sections: tuple[PromptSection, ...] | None = None

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "PromptInputV1":
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


class GlobalPromptContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["global"]
    text: str = ""

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return normalize_prompt_contract_text(value)


class PromptTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=TARGET_ID_PATTERN)
    name: str
    text: str
    order: int = Field(ge=0, strict=True)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = normalize_prompt_contract_text(value)
        if not 1 <= len(normalized) <= 40:
            raise ValueError("prompt target name must contain 1 to 40 characters")
        return normalized

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = normalize_prompt_contract_text(value)
        if not normalized:
            raise ValueError("prompt target text must not be empty")
        return normalized


class TargetPromptContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["targets"]
    targets: tuple[PromptTarget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_order(self) -> "TargetPromptContent":
        orders = [target.order for target in self.targets]
        if len(set(orders)) != len(orders) or set(orders) != set(range(len(orders))):
            raise ValueError("prompt target order must be contiguous from zero")
        return self


PromptSectionContent = Annotated[
    GlobalPromptContent | TargetPromptContent,
    Field(discriminator="mode"),
]


class PromptSectionV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: PromptSectionId
    label: str
    order: int = Field(ge=0, le=2, strict=True)
    content: PromptSectionContent

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = normalize_prompt_contract_text(value)
        if not 1 <= len(normalized) <= 40:
            raise ValueError("prompt section label must contain 1 to 40 characters")
        return normalized

    @property
    def text(self) -> str:
        if isinstance(self.content, GlobalPromptContent):
            return self.content.text
        return "\n".join(
            target.text for target in sorted(self.content.targets, key=lambda item: item.order)
        )


class PromptInputV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.prompt-input.v2"]
    mode: PromptInputMode
    sections: tuple[PromptSectionV2, ...] | None = None

    @model_validator(mode="after")
    def validate_mode_contract(self) -> "PromptInputV2":
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
        target_ids = [
            target.id
            for section in sections
            if isinstance(section.content, TargetPromptContent)
            for target in section.content.targets
        ]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("prompt target IDs must be unique across all sections")
        if not any(section.text for section in sections):
            raise ValueError("section prompt input requires prompt text")
        return self


PromptInputVariant = Annotated[
    PromptInputV1 | PromptInputV2,
    Field(discriminator="schema_version"),
]


class PromptInput(RootModel[PromptInputVariant]):
    """Stable public wrapper for the discriminated V1/V2 prompt-input union."""

    model_config = ConfigDict(frozen=True)

    @property
    def schema_version(self) -> str:
        return self.root.schema_version

    @property
    def mode(self) -> PromptInputMode:
        return self.root.mode

    @property
    def sections(self) -> tuple[PromptSection, ...] | tuple[PromptSectionV2, ...] | None:
        return self.root.sections

    @property
    def canonical_prompt(self) -> str | None:
        if self.mode == "single" or self.sections is None:
            return None
        values = (
            normalize_prompt_contract_text(section.text)
            for section in sorted(self.sections, key=lambda item: item.order)
        )
        return "\n".join(value for value in values if value)
