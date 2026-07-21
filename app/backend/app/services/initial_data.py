from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from uuid import uuid4


class InitialDataError(Exception):
    pass


@dataclass(frozen=True)
class InitialDataResult:
    created_relative_paths: tuple[str, ...]


class InitialDataMaterializer:
    _FILES = (
        (Path("advanced-styles/styles_v2.json"), Path("advanced_styles/styles_v2.json")),
        (Path("prompt-presets/presets.json"), Path("prompt-presets.json")),
        (Path("generation-presets/presets.json"), Path("presets.json")),
    )

    def __init__(self, defaults_root: Path, data_root: Path) -> None:
        self.defaults_root = defaults_root.resolve()
        self.data_root = data_root.resolve()

    def materialize(self) -> InitialDataResult:
        created: list[str] = []
        for source_relative, target_relative in self._FILES:
            source = (self.defaults_root / source_relative).resolve()
            target = (self.data_root / target_relative).resolve()
            self._require_contained(source, self.defaults_root, "default source")
            self._require_contained(target, self.data_root, "initial data target")
            if target.exists():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise InitialDataError("packaged initial data is invalid") from exc
            self._write_json_once(target, payload)
            created.append(target_relative.as_posix())
        return InitialDataResult(tuple(created))

    def is_materialized(self) -> bool:
        return all((self.data_root / target_relative).is_file() for _, target_relative in self._FILES)

    @staticmethod
    def _require_contained(path: Path, root: Path, label: str) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise InitialDataError(f"{label} must stay within its root") from exc

    @staticmethod
    def _write_json_once(path: Path, payload: object) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                return
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if path.exists():
                return
            os.replace(temporary, path)
        except OSError as exc:
            raise InitialDataError("initial data could not be materialized") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
