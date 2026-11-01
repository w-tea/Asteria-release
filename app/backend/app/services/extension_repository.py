from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from app.schemas.extensions import (
    ExtensionFileHash,
    ExtensionManifest,
    ExtensionState,
    ExtensionVersionReceipt,
    validate_extension_id,
    validate_semver,
)
from app.services.atomic_json import write_json_atomic
from app.services.extension_manifest import (
    ASTERIA_VERSION,
    ExtensionManifestError,
    load_extension_manifest,
)


_WINDOWS_REPARSE_ATTRIBUTE = 0x400


class ExtensionRepositoryError(ValueError):
    def __init__(self, message: str, *, code: str = "extension_repository_invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ResolvedExtensionEntrypoints:
    backend_file: Path
    backend_callable: str
    frontend_file: Path


class ExtensionRepository:
    def __init__(
        self,
        extension_root: Path,
        *,
        asteria_version: str = ASTERIA_VERSION,
    ) -> None:
        self.extension_root = Path(os.path.abspath(extension_root))
        self.asteria_version = asteria_version

    @property
    def registry_path(self) -> Path:
        return self._contained_path(self.extension_root / "registry.json")

    def state_path(self, extension_id: str) -> Path:
        return self._contained_path(self._extension_directory(extension_id) / "state.json")

    def managed_version_root(self, extension_id: str, version: str) -> Path:
        try:
            validate_semver(version)
        except ValueError as exc:
            raise ExtensionRepositoryError("extension version is invalid", code="extension_version_invalid") from exc
        return self._contained_path(self._extension_directory(extension_id) / "versions" / version)

    def manifest_path(self, extension_id: str, version: str) -> Path:
        return self._contained_path(self.managed_version_root(extension_id, version) / "asteria-extension.json")

    def receipt_path(self, extension_id: str, version: str) -> Path:
        return self._contained_path(self.managed_version_root(extension_id, version) / "receipt.json")

    def write_state(self, extension_id: str, state: ExtensionState) -> Path:
        try:
            state = ExtensionState.model_validate(state)
        except ValidationError as exc:
            raise ExtensionRepositoryError("extension state is invalid", code="extension_state_invalid") from exc
        self._validate_state(extension_id, state)
        path = self.state_path(extension_id)
        try:
            write_json_atomic(path, state.model_dump(mode="json"))
        except OSError as exc:
            raise ExtensionRepositoryError("extension state could not be written", code="extension_state_write_failed") from exc
        return path

    def read_state(self, extension_id: str) -> ExtensionState:
        path = self.state_path(extension_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            state = ExtensionState.model_validate(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise ExtensionRepositoryError("extension state is invalid", code="extension_state_invalid") from exc
        self._validate_state(extension_id, state)
        return state

    def read_manifest(self, extension_id: str, version: str) -> ExtensionManifest:
        path = self.manifest_path(extension_id, version)
        try:
            manifest = load_extension_manifest(path, asteria_version=self.asteria_version)
        except ExtensionManifestError as exc:
            raise ExtensionRepositoryError(str(exc), code=exc.code) from exc
        if manifest.id != extension_id or manifest.version != version:
            raise ExtensionRepositoryError(
                "extension manifest does not match its managed directory",
                code="extension_manifest_mismatch",
            )
        self.resolve_entrypoints(extension_id, version, manifest)
        return manifest

    def resolve_entrypoints(
        self,
        extension_id: str,
        version: str,
        manifest: ExtensionManifest | None = None,
    ) -> ResolvedExtensionEntrypoints:
        version_root = self.managed_version_root(extension_id, version)
        if manifest is None:
            try:
                manifest = load_extension_manifest(
                    self.manifest_path(extension_id, version),
                    asteria_version=self.asteria_version,
                )
            except ExtensionManifestError as exc:
                raise ExtensionRepositoryError(str(exc), code=exc.code) from exc
        module_name, callable_name = manifest.entrypoints.backend.split(":", 1)
        module_relative = PurePosixPath(*module_name.split("."))
        module_file = self._contained_path(version_root / f"{module_relative.as_posix()}.py")
        package_file = self._contained_path(version_root / module_relative / "__init__.py")
        backend_file = module_file if self._regular_file(module_file) else package_file
        frontend_file = self._contained_path(version_root / PurePosixPath(manifest.entrypoints.frontend))
        if not self._regular_file(backend_file) or not self._regular_file(frontend_file):
            raise ExtensionRepositoryError(
                "extension entrypoint file is missing or invalid",
                code="extension_entrypoint_invalid",
            )
        return ResolvedExtensionEntrypoints(
            backend_file=backend_file,
            backend_callable=callable_name,
            frontend_file=frontend_file,
        )

    def write_version_receipt(
        self,
        extension_id: str,
        version: str,
        *,
        package_checksum: str,
        source_trust: str | None = None,
        resolved_commit: str | None = None,
    ) -> str:
        if not _valid_sha256(package_checksum):
            raise ExtensionRepositoryError("package checksum is invalid", code="extension_receipt_invalid")
        self.read_manifest(extension_id, version)
        version_root = self.managed_version_root(extension_id, version)
        files = self._version_file_hashes(version_root)
        try:
            receipt = ExtensionVersionReceipt(
                extension_id=extension_id,
                version=version,
                package_checksum=package_checksum,
                tree_checksum=_tree_checksum(files),
                files=tuple(files),
                source_trust=source_trust,
                resolved_commit=resolved_commit,
            )
        except ValidationError as exc:
            raise ExtensionRepositoryError(
                "extension receipt is invalid",
                code="extension_receipt_invalid",
            ) from exc
        path = self.receipt_path(extension_id, version)
        try:
            write_json_atomic(path, receipt.model_dump(mode="json"))
            return _hash_file(path)
        except OSError as exc:
            raise ExtensionRepositoryError("extension receipt could not be written", code="extension_receipt_invalid") from exc

    def read_version_receipt(self, extension_id: str, version: str) -> ExtensionVersionReceipt:
        path = self.receipt_path(extension_id, version)
        try:
            receipt = ExtensionVersionReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ExtensionRepositoryError("extension receipt is invalid", code="extension_receipt_invalid") from exc
        if receipt.extension_id != extension_id or receipt.version != version:
            raise ExtensionRepositoryError("extension receipt does not match version", code="extension_receipt_invalid")
        actual_files = self._version_file_hashes(self.managed_version_root(extension_id, version))
        if tuple(actual_files) != receipt.files or _tree_checksum(actual_files) != receipt.tree_checksum:
            raise ExtensionRepositoryError("extension receipt tree or file hash is invalid", code="extension_receipt_invalid")
        return receipt

    def receipt_checksum(self, extension_id: str, version: str) -> str:
        path = self.receipt_path(extension_id, version)
        if not self._regular_file(path):
            raise ExtensionRepositoryError("extension receipt is invalid", code="extension_receipt_invalid")
        return _hash_file(path)

    def discover_extension_ids(self) -> tuple[str, ...]:
        self._assert_no_reparse(self.extension_root)
        if not self.extension_root.exists():
            return ()
        try:
            names = [path.name for path in self.extension_root.iterdir() if path.is_dir()]
        except OSError as exc:
            raise ExtensionRepositoryError("extension root could not be scanned", code="extension_root_unreadable") from exc
        return tuple(sorted((name for name in names if not name.startswith(".")), key=lambda value: (value.casefold(), value)))

    def _validate_state(self, extension_id: str, state: ExtensionState) -> None:
        self._validated_extension_id(extension_id)
        if state.recovery.status == "missing_install":
            return
        if state.source.kind == "unmanaged":
            reference = Path(state.source.external_reference or "")
            if not reference.is_dir() or _is_reparse_point(reference):
                raise ExtensionRepositoryError(
                    "unmanaged extension reference is missing or invalid",
                    code="extension_unmanaged_reference_missing",
                )
            return
        self._validate_managed_binding(
            extension_id,
            state.active_version,
            state.active_receipt_checksum,
            label="active",
        )
        if state.previous_version is not None:
            self._validate_managed_binding(
                extension_id,
                state.previous_version,
                state.previous_receipt_checksum,
                label="previous",
            )

    def _validate_managed_binding(
        self,
        extension_id: str,
        version: str,
        expected_receipt_checksum: str | None,
        *,
        label: str,
    ) -> None:
        try:
            self.read_manifest(extension_id, version)
            self.read_version_receipt(extension_id, version)
            actual = self.receipt_checksum(extension_id, version)
        except ExtensionRepositoryError as exc:
            if exc.code in {"extension_manifest_invalid", "extension_entrypoint_invalid"}:
                raise ExtensionRepositoryError(
                    f"managed extension {label} version is missing or invalid",
                    code=f"extension_{label}_version_invalid",
                ) from exc
            raise
        if actual != expected_receipt_checksum:
            raise ExtensionRepositoryError(
                f"managed extension {label} receipt binding is invalid",
                code="extension_receipt_invalid",
            )

    def _version_file_hashes(self, version_root: Path) -> list[ExtensionFileHash]:
        self._assert_no_reparse(version_root)
        if not version_root.is_dir():
            raise ExtensionRepositoryError("extension receipt version is missing", code="extension_receipt_invalid")
        files: list[ExtensionFileHash] = []
        relative_paths: set[str] = set()
        for root, directories, filenames in os.walk(version_root, followlinks=False):
            root_path = Path(root)
            for directory in directories:
                if _is_reparse_point(root_path / directory):
                    raise ExtensionRepositoryError("extension receipt crosses a reparse point", code="extension_reparse_traversal")
            for filename in filenames:
                path = root_path / filename
                if path.name == "receipt.json" and path.parent == version_root:
                    continue
                if not self._regular_file(path):
                    raise ExtensionRepositoryError("extension receipt file is invalid", code="extension_receipt_invalid")
                relative = path.relative_to(version_root).as_posix()
                folded_relative = relative.casefold()
                if folded_relative in relative_paths:
                    raise ExtensionRepositoryError(
                        "extension package has a case-insensitive path collision",
                        code="extension_path_collision",
                    )
                relative_paths.add(folded_relative)
                files.append(
                    ExtensionFileHash(relative_path=relative, size=path.stat().st_size, sha256=_hash_file(path))
                )
        return sorted(files, key=lambda item: item.relative_path.casefold())

    def _regular_file(self, path: Path) -> bool:
        self._assert_no_reparse(path)
        return path.is_file() and not _is_reparse_point(path)

    def _extension_directory(self, extension_id: str) -> Path:
        extension_id = self._validated_extension_id(extension_id)
        self._assert_no_reparse(self.extension_root)
        if self.extension_root.exists():
            try:
                for candidate in self.extension_root.iterdir():
                    if candidate.name.casefold() == extension_id.casefold() and candidate.name != extension_id:
                        raise ExtensionRepositoryError(
                            "extension id has a case-insensitive collision",
                            code="extension_id_collision",
                        )
            except ExtensionRepositoryError:
                raise
            except OSError as exc:
                raise ExtensionRepositoryError("extension root could not be scanned", code="extension_root_unreadable") from exc
        return self._contained_path(self.extension_root / extension_id)

    @staticmethod
    def _validated_extension_id(extension_id: str) -> str:
        try:
            return validate_extension_id(extension_id)
        except (TypeError, ValueError) as exc:
            raise ExtensionRepositoryError("extension id is invalid", code="extension_id_invalid") from exc

    def _contained_path(self, candidate: Path) -> Path:
        absolute = Path(os.path.abspath(candidate))
        try:
            if Path(os.path.commonpath((self.extension_root, absolute))) != self.extension_root:
                raise ValueError
        except (ValueError, OSError) as exc:
            raise ExtensionRepositoryError(
                "extension path must stay within the configured root",
                code="extension_path_escape",
            ) from exc
        self._assert_no_reparse(absolute)
        return absolute

    def _assert_no_reparse(self, candidate: Path) -> None:
        try:
            relative = candidate.relative_to(self.extension_root)
        except ValueError:
            relative = Path()
        current = self.extension_root
        if _is_reparse_point(current):
            raise ExtensionRepositoryError("extension path crosses a reparse point", code="extension_reparse_traversal")
        for part in relative.parts:
            current = current / part
            if _is_reparse_point(current):
                raise ExtensionRepositoryError("extension path crosses a reparse point", code="extension_reparse_traversal")


def _tree_checksum(files: list[ExtensionFileHash]) -> str:
    digest = sha256()
    for item in files:
        digest.update(f"{item.relative_path}\0{item.size}\0{item.sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & _WINDOWS_REPARSE_ATTRIBUTE)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ExtensionRepositoryError("extension path could not be inspected", code="extension_path_unreadable") from exc
