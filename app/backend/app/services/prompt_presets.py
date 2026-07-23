from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4


@dataclass
class PromptPreset:
    id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    tags: list[str] | None = None


class PromptPresetRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list(self) -> list[dict]:
        return self._read()

    def add(self, name: str, prompt: str, negative_prompt: str = "", tags: list[str] | None = None) -> dict:
        items = self._read(strict=True)
        item = {
            "id": uuid4().hex,
            "name": name.strip() or "Untitled preset",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "tags": self._normalize_tags(tags or []),
        }
        items.insert(0, item)
        self._write(items)
        return item

    def delete(self, preset_id: str) -> bool:
        items = self._read(strict=True)
        next_items = [item for item in items if item.get("id") != preset_id]
        if len(next_items) == len(items):
            return False
        self._write(next_items)
        return True

    def _read(self, *, strict: bool = False) -> list[dict]:
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
        items: list[dict] = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
                if strict:
                    raise OSError("prompt preset file contains an invalid item")
                continue
            items.append(
                {
                    "id": str(item["id"]),
                    "name": str(item.get("name") or "Untitled preset"),
                    "prompt": str(item["prompt"]),
                    "negative_prompt": str(item.get("negative_prompt") or ""),
                    "tags": self._normalize_tags(item.get("tags") if isinstance(item.get("tags"), list) else []),
                }
            )
        return items

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        normalized = []
        for tag in tags:
            value = " ".join(str(tag).strip().split())
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:16]
