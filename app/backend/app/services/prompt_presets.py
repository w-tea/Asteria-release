from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from uuid import uuid4

from app.schemas.prompt_input import PromptInput
from app.schemas.prompt_layout import PromptLayout
from app.services.spatial_prompt_contract import (
    SpatialPromptRecoveryStatus,
    recover_saved_spatial_prompt_contract,
    validate_spatial_prompt_contract,
)


@dataclass(frozen=True, slots=True)
class PromptPreset:
    id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: tuple[str, ...] = ()
    prompt_input: PromptInput | None = None
    prompt_layout: PromptLayout | None = None
    spatial_recovery_status: SpatialPromptRecoveryStatus = "ready"
    warning_codes: tuple[str, ...] = ()
    persisted_raw_spatial_contract: dict[str, object] | None = field(
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
        prompt_layout: PromptLayout | None = None,
    ) -> PromptPreset:
        validate_spatial_prompt_contract(
            prompt,
            prompt_input,
            prompt_layout,
            "txt2img",
        )
        items = self._read(strict=True)
        item = PromptPreset(
            id=uuid4().hex,
            name=name.strip() or "Untitled preset",
            prompt=prompt,
            negative_prompt=negative_prompt,
            tags=tuple(self._normalize_tags(tags or [])),
            prompt_input=prompt_input,
            prompt_layout=prompt_layout,
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
            recovery = recover_saved_spatial_prompt_contract(
                prompt,
                item.get("prompt_input"),
                item.get("prompt_layout"),
                "txt2img",
            )
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
                    prompt_input=recovery.snapshot.prompt_input,
                    prompt_layout=recovery.snapshot.prompt_layout,
                    spatial_recovery_status=recovery.status,
                    warning_codes=(
                        (recovery.warning_code,) if recovery.warning_code else ()
                    ),
                    persisted_raw_spatial_contract={
                        key: deepcopy(item[key])
                        for key in ("prompt_input", "prompt_layout")
                        if key in item
                    }
                    or None,
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
            if item.persisted_raw_spatial_contract is not None:
                for key, value in item.persisted_raw_spatial_contract.items():
                    serialized[key] = deepcopy(value)
            else:
                if item.prompt_input is not None:
                    serialized["prompt_input"] = item.prompt_input.model_dump(mode="json")
                if item.prompt_layout is not None:
                    serialized["prompt_layout"] = item.prompt_layout.model_dump(mode="json")
            payload.append(serialized)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        normalized = []
        for tag in tags:
            value = " ".join(str(tag).strip().split())
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:16]
