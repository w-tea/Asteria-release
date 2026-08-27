from __future__ import annotations

import re
import unicodedata
from typing import Literal


PromptAssistPurposePreset = Literal["standard", "anima", "illustrious", "danbooru_tags"]


_PRESET_INSTRUCTIONS: dict[PromptAssistPurposePreset, str] = {
    "standard": (
        "Preserve the subject and improve missing composition, lighting, and image-quality details. "
        "Do not force a model-specific tag format."
    ),
    "anima": (
        "Use an Anima-compatible prompt. Keep quality, year, artist, character count, character, copyright, "
        "appearance, and composition tags coherent. Do not add score tags mechanically. When natural-language "
        "detail helps, place it after the tags under the exact marker Description:. For multiple characters, "
        "use explicit character names instead of pronouns."
    ),
    "illustrious": (
        "Use an Illustrious-compatible Danbooru tag structure. Organize character count, character, appearance, "
        "action, composition, and lighting clearly. Do not impose a fixed quality-tag set because fine-tuned "
        "Illustrious checkpoints may use different conventions."
    ),
    "danbooru_tags": (
        "Convert the user's intent to concise English comma-separated Danbooru-style tags. Return no prose, "
        "numbering, or headings inside the prompt. Do not invent uncertain canonical tags; keep an ordinary "
        "descriptive tag when uncertain."
    ),
}


def prompt_assist_preset_instruction(preset: PromptAssistPurposePreset) -> str:
    return _PRESET_INSTRUCTIONS[preset]


def normalize_protected_tokens(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not 1 <= len(value) <= 120:
            continue
        if any(character in "\r\n" or unicodedata.category(character) == "Cc" for character in value):
            continue
        identity = value.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(value)
        if len(normalized) == 16:
            break
    return normalized


def restore_protected_tokens(prompt: str, tokens: list[str]) -> tuple[str, bool]:
    result = prompt.strip()
    missing: list[str] = []
    changed = False
    for token in normalize_protected_tokens(tokens):
        pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", flags=re.IGNORECASE)
        match = pattern.search(result)
        if match is None:
            missing.append(token)
            continue
        if match.group(0) != token:
            result = pattern.sub(lambda _match, replacement=token: replacement, result, count=1)
            changed = True
    if missing:
        prefix = ", ".join(missing)
        result = f"{prefix}, {result}" if result else prefix
        changed = True
    return result, changed
