from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import ValidationError

from app.schemas.prompt_input import (
    PromptInput,
    PromptSection,
    normalize_prompt_contract_text,
)


PromptInputWarningCode = Literal[
    "prompt_input_invalid_fallback",
    "prompt_input_mismatch_fallback",
]


class PromptInputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class PromptInputRecovery:
    prompt_input: PromptInput | None
    warning_code: PromptInputWarningCode | None = None


def normalize_prompt_section_text(value: str) -> str:
    return normalize_prompt_contract_text(value)


def combine_prompt_sections(sections: Sequence[PromptSection]) -> str:
    values = (
        normalize_prompt_section_text(section.text)
        for section in sorted(sections, key=lambda section: section.order)
    )
    return "\n".join(value for value in values if value)


def validate_prompt_input_pair(prompt: str, prompt_input: PromptInput) -> PromptInput:
    if (
        prompt_input.mode == "sections"
        and prompt_input.sections is not None
        and combine_prompt_sections(prompt_input.sections) != prompt
    ):
        raise PromptInputError("prompt_input_mismatch")
    return prompt_input


def recover_saved_prompt_input(
    prompt: str,
    raw_prompt_input: object,
) -> PromptInputRecovery:
    if raw_prompt_input is None:
        return PromptInputRecovery(prompt_input=None)

    try:
        prompt_input = PromptInput.model_validate(raw_prompt_input)
    except (TypeError, ValidationError):
        return PromptInputRecovery(
            prompt_input=None,
            warning_code="prompt_input_invalid_fallback",
        )

    try:
        validate_prompt_input_pair(prompt, prompt_input)
    except PromptInputError:
        return PromptInputRecovery(
            prompt_input=None,
            warning_code="prompt_input_mismatch_fallback",
        )
    return PromptInputRecovery(prompt_input=prompt_input)
