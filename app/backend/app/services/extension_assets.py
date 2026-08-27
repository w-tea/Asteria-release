from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat

from app.schemas.extensions import (
    ExtensionManifest,
    ExtensionState,
    validate_extension_relative_path,
)
from app.services.extension_manifest import ExtensionManifestError, load_extension_manifest
from app.services.extension_repository import ExtensionRepository, ExtensionRepositoryError


_MIME_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
}


class ExtensionAssetError(RuntimeError):
    pass


def resolve_active_manifest(
    repository: ExtensionRepository,
    extension_id: str,
    state: ExtensionState,
) -> tuple[ExtensionManifest, Path]:
    if state.source.kind == "managed":
        repository.read_version_receipt(extension_id, state.active_version)
        return (
            repository.read_manifest(extension_id, state.active_version),
            repository.managed_version_root(extension_id, state.active_version),
        )
    package_root = Path(state.source.external_reference or "")
    try:
        manifest = load_extension_manifest(
            package_root / "asteria-extension.json",
            asteria_version=repository.asteria_version,
        )
    except ExtensionManifestError as exc:
        raise ExtensionRepositoryError(str(exc), code=exc.code) from exc
    if manifest.id != extension_id or manifest.version != state.active_version:
        raise ExtensionRepositoryError(
            "unmanaged extension manifest does not match state",
            code="extension_manifest_mismatch",
        )
    _assert_contained_regular(package_root, package_root / "asteria-extension.json")
    return manifest, package_root


def resolve_extension_asset(
    repository: ExtensionRepository,
    *,
    extension_id: str,
    relative_path: str,
    startup_result: object,
) -> tuple[Path, str]:
    if bool(getattr(startup_result, "safe_mode", False)):
        raise ExtensionAssetError("extension_asset_not_found")
    state = repository.read_state(extension_id)
    if (
        not state.enabled
        or state.recovery.status != "ready"
        or state.pending_action is not None
    ):
        raise ExtensionAssetError("extension_asset_not_found")
    loaded = next(
        (
            record
            for record in tuple(getattr(startup_result, "records", ()))
            if record.extension_id == extension_id
            and record.status == "loaded"
            and record.version == state.active_version
        ),
        None,
    )
    if loaded is None:
        raise ExtensionAssetError("extension_asset_not_found")
    manifest, package_root = resolve_active_manifest(repository, extension_id, state)
    if manifest.execution_level != "full_access":
        raise ExtensionAssetError("extension_asset_not_found")
    normalized = _safe_relative_path(relative_path)
    suffix = PurePosixPath(normalized).suffix.casefold()
    media_type = _MIME_TYPES.get(suffix)
    if media_type is None:
        raise ExtensionAssetError("extension_asset_not_found")
    frontend_root = package_root / PurePosixPath(manifest.entrypoints.frontend).parent
    candidate = package_root / PurePosixPath(normalized)
    try:
        Path(os.path.abspath(candidate)).relative_to(Path(os.path.abspath(frontend_root)))
    except ValueError as exc:
        raise ExtensionAssetError("extension_asset_not_found") from exc
    try:
        resolved = _assert_contained_regular(package_root, candidate)
    except (OSError, ValueError, ExtensionRepositoryError) as exc:
        raise ExtensionAssetError("extension_asset_not_found") from exc
    return resolved, media_type


def _safe_relative_path(value: str) -> str:
    try:
        return validate_extension_relative_path(value)
    except ValueError as exc:
        raise ExtensionAssetError("extension_asset_not_found") from exc


def _assert_contained_regular(root: Path, candidate: Path) -> Path:
    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        candidate_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ExtensionRepositoryError(
            "extension asset escapes package root",
            code="extension_path_escape",
        ) from exc
    current = candidate_absolute
    while True:
        details = current.lstat()
        if _is_reparse(current, details):
            raise ExtensionRepositoryError(
                "extension asset crosses a reparse point",
                code="extension_reparse_traversal",
            )
        if current == root_absolute:
            break
        current = current.parent
    if not stat.S_ISREG(candidate_absolute.lstat().st_mode):
        raise ExtensionRepositoryError(
            "extension asset is not a regular file",
            code="extension_path_unreadable",
        )
    return candidate_absolute


def _is_reparse(path: Path, details: os.stat_result) -> bool:
    return (
        path.is_symlink()
        or bool(getattr(os.path, "isjunction", lambda _path: False)(path))
        or bool(getattr(details, "st_file_attributes", 0) & 0x400)
    )
