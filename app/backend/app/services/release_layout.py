from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


@dataclass(frozen=True)
class ReleaseLayout:
    version: int
    backend_root: Path
    backend_app: Path
    frontend_entry: Path
    python: Path
    node: Path
    comfy_root: Path
    writable_roots: dict[str, Path]

    @property
    def cache_root(self) -> Path:
        return self.writable_roots["temp"] / "cache"

    @property
    def extension_root(self) -> Path:
        return self.writable_roots["data"] / "extensions"


class ReleaseLayoutRepository:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def load(self) -> ReleaseLayout:
        try:
            payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("release layout manifest could not be loaded") from exc
        if not isinstance(payload, dict) or payload.get("version") != 3:
            raise ValueError("release layout manifest version is unsupported")
        package = payload.get("package")
        writable_roots = payload.get("writable_roots")
        if not isinstance(package, dict) or not isinstance(writable_roots, dict):
            raise ValueError("release layout manifest structure is invalid")
        paths = {
            key: self._package_path(package.get(key))
            for key in (
                "backend_root",
                "backend_app",
                "frontend_entry",
                "python",
                "node",
                "comfy_root",
            )
        }
        required_writable_roots = {"data", "models", "logs", "temp"}
        if set(writable_roots) != required_writable_roots:
            raise ValueError("release writable root set is invalid")
        contained_writable_roots: dict[str, Path] = {}
        for key in sorted(required_writable_roots):
            value = writable_roots.get(key)
            if isinstance(value, str) and value.strip().casefold() == "external":
                raise ValueError("release writable roots must stay within the application root")
            try:
                contained_writable_roots[key] = self._package_path(value)
            except ValueError as exc:
                raise ValueError("release writable roots must stay within the application root") from exc
        normalized_roots = {
            root.as_posix().replace("\\", "/").casefold()
            for root in contained_writable_roots.values()
        }
        if len(normalized_roots) != len(contained_writable_roots):
            raise ValueError("release writable roots must be distinct")
        return ReleaseLayout(
            version=3,
            backend_root=paths["backend_root"],
            backend_app=paths["backend_app"],
            frontend_entry=paths["frontend_entry"],
            python=paths["python"],
            node=paths["node"],
            comfy_root=paths["comfy_root"],
            writable_roots=contained_writable_roots,
        )

    @staticmethod
    def _package_path(value: object) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("release package path is missing")
        path = Path(value.strip())
        windows_path = PureWindowsPath(value.strip())
        posix_path = PurePosixPath(value.strip())
        all_parts = (*windows_path.parts, *posix_path.parts)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or posix_path.is_absolute()
            or windows_path.drive
            or ".." in all_parts
        ):
            raise ValueError("release package paths must stay within the application root")
        return path
