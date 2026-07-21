from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import AppConfig
from app.services.user_settings import UserSettingsRepository, resolve_merge_preset_dir, resolve_output_root

router = APIRouter(prefix="/api/filesystem", tags=["filesystem"])


class FilesystemRoot(BaseModel):
    label: str
    path: str
    kind: str


class FilesystemEntry(BaseModel):
    name: str
    path: str
    kind: str
    selectable: bool


class FilesystemBrowseResponse(BaseModel):
    current_path: str | None
    parent_path: str | None
    target: str
    roots: list[FilesystemRoot]
    entries: list[FilesystemEntry]
    error: str | None = None


class FilesystemResolveError(Exception):
    def __init__(self, requested: Path, original: OSError) -> None:
        super().__init__(str(original))
        self.requested = requested
        self.original = original


def _path_text(path: Path, *, strict: bool = False) -> str:
    try:
        return str(path.resolve(strict=strict))
    except OSError:
        return str(path)


def _resolve_requested_path(path: str | None, config: AppConfig) -> Path | None:
    if path is None or not path.strip():
        return None
    requested = Path(path.strip()).expanduser()
    if not requested.is_absolute():
        requested = config.root_dir / requested
    try:
        return requested.resolve(strict=False)
    except OSError as exc:
        raise FilesystemResolveError(requested, exc) from exc


def _configured_roots(config: AppConfig) -> list[FilesystemRoot]:
    settings = UserSettingsRepository(config.data_dir / "user-settings.json").load()
    roots = [
        FilesystemRoot(label="Project", path=_path_text(config.root_dir), kind="project"),
        FilesystemRoot(label="Data", path=_path_text(config.data_dir), kind="data"),
        FilesystemRoot(label="Models", path=_path_text(config.model_dir), kind="models"),
        FilesystemRoot(
            label="Output",
            path=_path_text(resolve_output_root(config.data_dir, settings.output)),
            kind="output",
        ),
        FilesystemRoot(
            label="Merge presets",
            path=_path_text(resolve_merge_preset_dir(config.data_dir, settings.storage, config.merge_preset_dir)),
            kind="merge_presets",
        ),
    ]
    try:
        roots.append(FilesystemRoot(label="Home", path=_path_text(Path.home()), kind="home"))
    except RuntimeError:
        pass
    if os.name == "nt":
        for code in range(ord("A"), ord("Z") + 1):
            drive = Path(f"{chr(code)}:/")
            try:
                exists = drive.exists()
            except OSError:
                exists = False
            if exists:
                roots.append(FilesystemRoot(label=f"{chr(code)} drive", path=str(drive), kind="drive"))
    else:
        roots.append(FilesystemRoot(label="Filesystem root", path="/", kind="root"))

    seen: set[str] = set()
    unique_roots: list[FilesystemRoot] = []
    for root in roots:
        normalized = root.path.rstrip("\\/")
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_roots.append(root)
    return unique_roots


def _matches_extensions(path: Path, extensions: set[str]) -> bool:
    if not extensions:
        return True
    return path.suffix.lower().lstrip(".") in extensions


def _browse_error_response(
    requested: Path,
    target: str,
    roots: list[FilesystemRoot],
    exc: OSError,
) -> FilesystemBrowseResponse:
    if isinstance(exc, PermissionError):
        error = f"Permission denied: {exc}"
    else:
        error = str(exc)
    return FilesystemBrowseResponse(
        current_path=str(requested),
        parent_path=str(requested.parent) if requested.parent != requested else None,
        target=target,
        roots=roots,
        entries=[],
        error=error,
    )


@router.get("/browse", response_model=FilesystemBrowseResponse)
def browse_filesystem(
    path: str | None = None,
    target: str = Query("directory", pattern="^(directory|file)$"),
    extensions: str = "",
) -> FilesystemBrowseResponse:
    config = AppConfig()
    roots = _configured_roots(config)
    try:
        requested = _resolve_requested_path(path, config)
    except FilesystemResolveError as exc:
        return _browse_error_response(exc.requested, target, roots, exc.original)
    normalized_extensions = {item.strip().lower().lstrip(".") for item in extensions.split(",") if item.strip()}

    if requested is None:
        root_entries = [
            FilesystemEntry(name=root.label, path=root.path, kind=root.kind, selectable=target == "directory")
            for root in roots
        ]
        return FilesystemBrowseResponse(
            current_path=None,
            parent_path=None,
            target=target,
            roots=roots,
            entries=root_entries,
        )

    try:
        if requested.is_file():
            requested = requested.parent
        if not requested.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {requested}")
        if not requested.is_dir():
            raise HTTPException(status_code=422, detail=f"Path is not a directory: {requested}")
    except HTTPException:
        raise
    except OSError as exc:
        return _browse_error_response(requested, target, roots, exc)

    entries: list[FilesystemEntry] = []
    try:
        children = sorted(
            requested.iterdir(),
            key=lambda child: (not child.is_dir(), child.name.lower()),
        )
    except OSError as exc:
        return _browse_error_response(requested, target, roots, exc)

    for child in children:
        try:
            child_is_dir = child.is_dir()
        except OSError as exc:
            return _browse_error_response(requested, target, roots, exc)
        if child_is_dir:
            entries.append(FilesystemEntry(name=child.name, path=str(child), kind="directory", selectable=target == "directory"))
            continue
        if target != "file":
            continue
        try:
            child_is_file = child.is_file()
        except OSError as exc:
            return _browse_error_response(requested, target, roots, exc)
        if child_is_file and _matches_extensions(child, normalized_extensions):
            entries.append(FilesystemEntry(name=child.name, path=str(child), kind="file", selectable=True))

    return FilesystemBrowseResponse(
        current_path=str(requested),
        parent_path=str(requested.parent) if requested.parent != requested else None,
        target=target,
        roots=roots,
        entries=entries,
    )
