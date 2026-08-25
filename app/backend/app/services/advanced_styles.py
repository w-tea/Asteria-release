from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class AdvancedStyle:
    id: str
    group: str
    name: str
    prompt: str
    negative: str
    thumbnail_path: str | None = None


@dataclass(frozen=True)
class AdvancedStylesLibrary:
    groups: list[str]
    styles: list[AdvancedStyle]
    source_path: Path


class AdvancedStylesError(Exception):
    pass


class AdvancedStylesRepository:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir / "advanced_styles"
        self.styles_path = self.data_dir / "styles_v2.json"
        self.backup_path = self.data_dir / "styles_v2.json.bak"
        self.thumbnail_dir = self.data_dir / "thumbnails"

    def list(self) -> AdvancedStylesLibrary:
        data = self._load_styles()
        groups = sorted(data.keys(), key=str.lower)
        styles: list[AdvancedStyle] = []
        for group in groups:
            group_styles = data.get(group, {})
            for name in sorted(group_styles.keys(), key=str.lower):
                style_data = group_styles[name]
                styles.append(
                    AdvancedStyle(
                        id=f"{group}/{name}",
                        group=group,
                        name=name,
                        prompt=str(style_data.get("prompt", "")),
                        negative=str(style_data.get("negative", "")),
                        thumbnail_path=self._thumbnail_relative_path(group, name),
                    )
                )
        return AdvancedStylesLibrary(groups=groups, styles=styles, source_path=self.styles_path)

    def save_style(self, group: str, name: str, prompt: str, negative: str) -> AdvancedStyle:
        group_name = str(group or "default").strip() or "default"
        style_name = str(name or "untitled").strip() or "untitled"
        data = self._load_styles(strict=True)
        data.setdefault(group_name, {})
        data[group_name][style_name] = {
            "prompt": str(prompt or ""),
            "negative": str(negative or ""),
        }
        self._save_styles(data)
        return AdvancedStyle(
            id=f"{group_name}/{style_name}",
            group=group_name,
            name=style_name,
            prompt=str(prompt or ""),
            negative=str(negative or ""),
            thumbnail_path=self._thumbnail_relative_path(group_name, style_name),
        )

    def delete_style(self, group: str, name: str) -> bool:
        data = self._load_styles(strict=True)
        group_styles = data.get(group)
        if not group_styles or name not in group_styles:
            return False
        del group_styles[name]
        if not group_styles and group != "default":
            del data[group]
        self._save_styles(data)
        return True

    def save_thumbnail(self, group: str, name: str, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("Thumbnail image is empty")
        try:
            from io import BytesIO

            path = self._thumbnail_path(group, name, create_dir=True)
            with Image.open(BytesIO(image_bytes)) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((512, 512))
                thumbnail.save(path, format="WEBP", quality=90, method=6)
        except OSError as exc:
            raise AdvancedStylesError(f"Could not save thumbnail: {exc}") from exc
        except Exception as exc:
            raise ValueError("Unsupported thumbnail image") from exc
        return str(path.relative_to(self.data_dir))

    def delete_thumbnail(self, group: str, name: str) -> bool:
        path = self._thumbnail_path(group, name, create_dir=False)
        if not self._path_exists(path) or not self._path_is_file(path):
            return False
        try:
            path.unlink()
        except OSError as exc:
            raise AdvancedStylesError(f"Could not delete thumbnail: {exc}") from exc
        group_dir = path.parent
        try:
            if self._path_exists(group_dir) and not any(group_dir.iterdir()):
                group_dir.rmdir()
        except OSError:
            pass
        return True

    def import_csv(self, csv_text: str, target_group: str, update_existing: bool = True) -> int:
        import csv
        from io import StringIO

        group_name = str(target_group or "default").strip() or "default"
        data = self._load_styles(strict=True)
        data.setdefault(group_name, {})
        reader = csv.reader(StringIO(csv_text))
        count = 0
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in {"name", "style name"}:
                continue
            name = row[0].strip()
            if not name:
                continue
            prompt = row[1] if len(row) > 1 else ""
            negative = row[2] if len(row) > 2 else ""
            if not update_existing and name in data[group_name]:
                continue
            data[group_name][name] = {
                "prompt": str(prompt or ""),
                "negative": str(negative or ""),
            }
            count += 1
        if count:
            self._save_styles(data)
        return count

    def _load_styles(self, *, strict: bool = False) -> dict[str, dict[str, dict[str, str]]]:
        try:
            styles_path_exists = self.styles_path.exists()
        except OSError as exc:
            if strict:
                raise AdvancedStylesError(f"Could not load advanced styles: {exc}") from exc
            styles_path_exists = False
        if not styles_path_exists:
            return {"default": {}}
        try:
            raw = json.loads(self.styles_path.read_text(encoding="utf-8"))
        except OSError as exc:
            if strict:
                raise AdvancedStylesError(f"Could not load advanced styles: {exc}") from exc
            if self._path_exists(self.backup_path):
                try:
                    raw = json.loads(self.backup_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    raw = {}
            else:
                raw = {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if strict:
                raise AdvancedStylesError(f"Could not load advanced styles: {exc}") from exc
            if self._path_exists(self.backup_path):
                try:
                    raw = json.loads(self.backup_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    raw = {}
            else:
                raw = {}
        if strict and not isinstance(raw, dict):
            raise AdvancedStylesError("Could not load advanced styles: invalid storage root")
        if strict and isinstance(raw, dict):
            for styles in raw.values():
                if not isinstance(styles, dict):
                    raise AdvancedStylesError("Could not load advanced styles: invalid style group")
        return self._normalize(raw)

    def _normalize(self, data: Any) -> dict[str, dict[str, dict[str, str]]]:
        if not isinstance(data, dict) or not data:
            return {"default": {}}
        normalized: dict[str, dict[str, dict[str, str]]] = {}
        for group, styles in data.items():
            group_name = str(group or "default")
            normalized[group_name] = {}
            if not isinstance(styles, dict):
                continue
            for style_name, style_data in styles.items():
                name = str(style_name or "untitled")
                if not isinstance(style_data, dict):
                    style_data = {"prompt": str(style_data), "negative": ""}
                normalized[group_name][name] = {
                    "prompt": str(style_data.get("prompt", "") or ""),
                    "negative": str(style_data.get("negative", "") or ""),
                }
        return normalized or {"default": {}}

    def _save_styles(self, data: dict[str, dict[str, dict[str, str]]]) -> None:
        normalized = self._normalize(data)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.styles_path.with_suffix(".json.tmp")
            if self._path_exists(self.styles_path):
                shutil.copy2(self.styles_path, self.backup_path)
            tmp_path.write_text(
                json.dumps(normalized, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self.styles_path)
        except OSError as exc:
            raise AdvancedStylesError(f"Could not save advanced styles: {exc}") from exc

    def _thumbnail_relative_path(self, group: str, name: str) -> str | None:
        path = self._thumbnail_path(group, name, create_dir=False)
        if not self._path_exists(path) or not self._path_is_file(path):
            return None
        return str(path.relative_to(self.data_dir))

    def _thumbnail_path(self, group: str, name: str, create_dir: bool) -> Path:
        group_dir = self.thumbnail_dir / self._sanitize_filename(group)
        if create_dir:
            group_dir.mkdir(parents=True, exist_ok=True)
        return group_dir / f"{self._sanitize_filename(name)}.webp"

    def _sanitize_filename(self, name: str) -> str:
        value = str(name or "untitled").strip()
        value = re.sub(r'[\\/*?:"<>|]', "_", value)
        value = re.sub(r"\s+", " ", value)
        return value[:120] or "untitled"

    def _path_exists(self, path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    def _path_is_file(self, path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False
