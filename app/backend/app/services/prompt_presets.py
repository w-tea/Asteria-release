from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from uuid import uuid4

from app.schemas.prompt_input import PromptInput
from app.services.prompt_input import (
    PromptInputWarningCode,
    recover_saved_prompt_input,
    validate_prompt_input_pair,
)


@dataclass(frozen=True, slots=True)
class PromptPreset:
    id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: tuple[str, ...] = ()
    prompt_input: PromptInput | None = None
    warning_codes: tuple[PromptInputWarningCode, ...] = ()
    recovered_raw_prompt_input: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class PromptPresetRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> list[PromptPreset]:
        return self._read()

    def add(
        self,
        name: str,
        prompt: str,
        negative_prompt: str = "",
        tags: list[str] | None = None,
        prompt_input: PromptInput | None = None,
    ) -> PromptPreset:
        if prompt_input is not None:
            validate_prompt_input_pair(prompt, prompt_input)
        items = self._read(strict=True)
        item = PromptPreset(
            id=uuid4().hex,
            name=name.strip() or "Untitled preset",
            prompt=prompt,
            negative_prompt=negative_prompt,
            tags=tuple(self._normalize_tags(tags or [])),
            prompt_input=prompt_input,
        )
        items.insert(0, item)
        self._write(items)
        return item

    def delete(self, preset_id: str) -> bool:
        items = self._read(strict=True)
        next_items = [item for item in items if item.id != preset_id]
        if len(next_items) == len(items):
            return False
        self._write(next_items)
        return True

    def _read(self, *, strict: bool = False) -> list[PromptPreset]:
        try:
            if not self.path.exists():
                return []
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError:
            if strict:
                raise
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise OSError(f"prompt preset file is not valid JSON: {exc}") from exc
            return []
        if not isinstance(payload, list):
            if strict:
                raise OSError("prompt preset file root must be a list")
            return []
        items: list[PromptPreset] = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
                if strict:
                    raise OSError("prompt preset file contains an invalid item")
                continue
            prompt = str(item["prompt"])
            recovery = recover_saved_prompt_input(prompt, item.get("prompt_input"))
            items.append(
                PromptPreset(
                    id=str(item["id"]),
                    name=str(item.get("name") or "Untitled preset"),
                    prompt=prompt,
                    negative_prompt=str(item.get("negative_prompt") or ""),
                    tags=tuple(
                        self._normalize_tags(
                            item.get("tags") if isinstance(item.get("tags"), list) else []
                        )
                    ),
                    prompt_input=recovery.prompt_input,
                    warning_codes=(recovery.warning_code,) if recovery.warning_code else (),
                    recovered_raw_prompt_input=(
                        deepcopy(item.get("prompt_input")) if recovery.warning_code else None
                    ),
                )
            )
        return items

    def _write(self, items: list[PromptPreset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for item in items:
            serialized = {
                "id": item.id,
                "name": item.name,
                "prompt": item.prompt,
                "negative_prompt": item.negative_prompt,
                "tags": list(item.tags),
            }
            if item.prompt_input is not None:
                serialized["prompt_input"] = item.prompt_input.model_dump(mode="json")
            elif item.warning_codes and item.recovered_raw_prompt_input is not None:
                serialized["prompt_input"] = deepcopy(item.recovered_raw_prompt_input)
            payload.append(serialized)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        normalized = []
        for tag in tags:
            value = " ".join(str(tag).strip().split())
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:16]
