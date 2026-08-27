from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.extensions import (
    ExtensionLastLoad,
    ExtensionRecoveryState,
    ExtensionRegistrySnapshot,
    ExtensionState,
    ExtensionVersionReceipt,
)
from app.services.atomic_json import write_json_atomic
from app.services.extension_registry import ExtensionRegistry, ExtensionRegistryRepository
from app.services.extension_repository import (
    ExtensionRepository,
    ExtensionRepositoryError,
    _is_reparse_point,
)


class ExtensionBackupError(RuntimeError):
    pass


@dataclass(slots=True)
class ExtensionRestoreTransaction:
    staging: Path
    rollback: Path
    live_moved: bool = False
    applied: bool = False
    closed: bool = False


class ExtensionBackupService:
    def __init__(self, extension_root: Path, *, asteria_version: str = "1.2.0") -> None:
        self.extension_root = Path(os.path.abspath(extension_root))
        self.asteria_version = asteria_version
        self._swap_failure_injector: Callable[[str], None] | None = None

    def backup_sources(self) -> list[Path]:
        repository = ExtensionRepository(self.extension_root, asteria_version=self.asteria_version)
        sources: list[Path] = []
        try:
            ExtensionRegistry(repository).scan()
            ExtensionRegistryRepository(repository.registry_path).load()
            sources.append(repository.registry_path)
            for extension_id in repository.discover_extension_ids():
                state_path = repository.state_path(extension_id)
                state = self._read_state_model(state_path)
                sources.append(state_path)
                if state.recovery.status == "missing_install":
                    continue
                if state.source.kind == "unmanaged":
                    reference = Path(state.source.external_reference or "")
                    if not reference.is_dir() or _is_reparse_point(reference):
                        raise ExtensionBackupError("extension_unmanaged_reference_missing")
                    continue
                for version, expected in self._version_bindings(state):
                    repository.read_manifest(extension_id, version)
                    repository.read_version_receipt(extension_id, version)
                    if repository.receipt_checksum(extension_id, version) != expected:
                        raise ExtensionRepositoryError(
                            "extension receipt binding is invalid",
                            code="extension_receipt_invalid",
                        )
                    sources.extend(self._regular_tree_files(repository.managed_version_root(extension_id, version)))
        except (ExtensionRepositoryError, ValueError, OSError) as exc:
            code = getattr(exc, "code", "extension_backup_invalid")
            raise ExtensionBackupError(str(code)) from exc
        return self._unique_sorted(sources)

    def validate_backup_tree(self, backup_extension_root: Path) -> None:
        root = Path(os.path.abspath(backup_extension_root))
        if not root.exists():
            return
        self._assert_safe_tree(root)
        repository = ExtensionRepository(root, asteria_version=self.asteria_version)
        try:
            if not repository.registry_path.is_file():
                raise ExtensionBackupError("extension_backup_registry_missing")
            snapshot = ExtensionRegistryRepository(repository.registry_path).load()
            self._assert_restore_allowlist(root, repository, snapshot)
            for extension_id in repository.discover_extension_ids():
                state = self._read_state_model(repository.state_path(extension_id))
                if state.source.kind != "managed" or state.recovery.status == "missing_install":
                    continue
                for version, expected in self._version_bindings(state):
                    version_root = repository.managed_version_root(extension_id, version)
                    if not version_root.exists():
                        continue
                    self._package_is_valid(repository, extension_id, version, expected)
        except (ExtensionRepositoryError, ValueError, OSError, ValidationError) as exc:
            code = getattr(exc, "code", "extension_backup_invalid")
            raise ExtensionBackupError(str(code)) from exc

    def _assert_restore_allowlist(
        self,
        root: Path,
        repository: ExtensionRepository,
        snapshot: ExtensionRegistrySnapshot,
    ) -> None:
        actual_files, actual_directories = self._relative_tree_members(root)
        expected_files: set[str] = set()
        expected_directories: set[str] = set()
        self._allow_file(repository.registry_path, root, expected_files, expected_directories)

        extension_ids = repository.discover_extension_ids()
        snapshot_by_id = {record.extension_id.casefold(): record for record in snapshot.records}
        if {extension_id.casefold() for extension_id in extension_ids} != set(snapshot_by_id):
            raise ExtensionBackupError("extension_backup_registry_mismatch")

        for extension_id in extension_ids:
            state_path = repository.state_path(extension_id)
            state = self._read_state_model(state_path)
            snapshot_record = snapshot_by_id[extension_id.casefold()]
            expected_status = "ready" if state.recovery.status == "ready" else "recovery"
            if (
                snapshot_record.status != expected_status
                or snapshot_record.active_version != state.active_version
                or snapshot_record.previous_version != state.previous_version
                or snapshot_record.enabled != state.enabled
                or snapshot_record.source_kind != state.source.kind
                or snapshot_record.recovery_status != state.recovery.status
            ):
                raise ExtensionBackupError("extension_backup_registry_mismatch")
            self._allow_file(state_path, root, expected_files, expected_directories)
            if state.source.kind != "managed" or state.recovery.status == "missing_install":
                continue
            for version, _expected in self._version_bindings(state):
                version_root = repository.managed_version_root(extension_id, version)
                self._allow_directory(version_root, root, expected_directories)
                if not version_root.exists():
                    continue
                if not version_root.is_dir():
                    self._allow_file(version_root, root, expected_files, expected_directories)
                    continue
                receipt = self._read_receipt_layout(repository.receipt_path(extension_id, version))
                if receipt is None or receipt.extension_id != extension_id or receipt.version != version:
                    self._allow_quarantined_version(
                        version_root,
                        root,
                        actual_files,
                        actual_directories,
                        expected_files,
                        expected_directories,
                    )
                    continue
                self._allow_file(
                    repository.receipt_path(extension_id, version),
                    root,
                    expected_files,
                    expected_directories,
                )
                for item in receipt.files:
                    self._allow_file(
                        version_root / PurePosixPath(item.relative_path),
                        root,
                        expected_files,
                        expected_directories,
                    )

        if actual_files - expected_files or actual_directories - expected_directories:
            raise ExtensionBackupError("extension_backup_unbound_path")

    @staticmethod
    def _read_receipt_layout(path: Path) -> ExtensionVersionReceipt | None:
        try:
            return ExtensionVersionReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError):
            return None

    @staticmethod
    def _relative_tree_members(root: Path) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()
        for directory, child_directories, filenames in os.walk(root, followlinks=False):
            root_path = Path(directory)
            for name in child_directories:
                directories.add((root_path / name).relative_to(root).as_posix().casefold())
            for name in filenames:
                files.add((root_path / name).relative_to(root).as_posix().casefold())
        return files, directories

    @classmethod
    def _allow_file(
        cls,
        path: Path,
        root: Path,
        files: set[str],
        directories: set[str],
    ) -> None:
        relative = path.relative_to(root)
        files.add(relative.as_posix().casefold())
        cls._allow_parent_directories(relative, directories)

    @staticmethod
    def _allow_parent_directories(relative: Path, directories: set[str]) -> None:
        parent = relative.parent
        while parent != Path("."):
            directories.add(parent.as_posix().casefold())
            parent = parent.parent

    @classmethod
    def _allow_directory(cls, path: Path, root: Path, directories: set[str]) -> None:
        relative = path.relative_to(root)
        directories.add(relative.as_posix().casefold())
        cls._allow_parent_directories(relative, directories)

    @classmethod
    def _allow_quarantined_version(
        cls,
        version_root: Path,
        root: Path,
        actual_files: set[str],
        actual_directories: set[str],
        expected_files: set[str],
        expected_directories: set[str],
    ) -> None:
        relative = version_root.relative_to(root).as_posix().casefold()
        prefix = f"{relative}/"
        expected_files.update(path for path in actual_files if path.startswith(prefix))
        expected_directories.update(
            path for path in actual_directories if path == relative or path.startswith(prefix)
        )
        cls._allow_parent_directories(version_root.relative_to(root), expected_directories)

    def restore_from_backup(self, backup_extension_root: Path) -> None:
        transaction = self.prepare_restore(backup_extension_root)
        if transaction is None:
            return
        try:
            self.apply_restore(transaction)
        except Exception:
            self.rollback_restore(transaction)
            raise
        self.commit_restore(transaction)

    def prepare_restore(
        self,
        backup_extension_root: Path,
    ) -> ExtensionRestoreTransaction | None:
        source_root = Path(os.path.abspath(backup_extension_root))
        if not source_root.exists():
            return None
        self.validate_backup_tree(source_root)
        source_fingerprint = self._tree_fingerprint(source_root)
        parent = self.extension_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        staging = parent / f".extensions.{token}.restore.tmp"
        rollback = parent / f".extensions.{token}.restore.rollback"
        try:
            self._copy_tree(source_root, staging)
            self.validate_backup_tree(staging)
            if self._tree_fingerprint(staging) != source_fingerprint:
                raise ExtensionBackupError("extension_restore_copy_mismatch")
            self._normalize_missing_installs(staging)
            ExtensionRegistry(ExtensionRepository(staging, asteria_version=self.asteria_version)).scan()
            return ExtensionRestoreTransaction(staging=staging, rollback=rollback)
        except Exception as exc:
            if staging.exists():
                self._remove_private_tree(staging)
            if rollback.exists():
                self._remove_private_tree(rollback)
            if isinstance(exc, ExtensionBackupError):
                raise
            raise ExtensionBackupError("extension_restore_failed") from exc

    def apply_restore(self, transaction: ExtensionRestoreTransaction) -> None:
        if transaction.closed or transaction.applied:
            raise ExtensionBackupError("extension_restore_transaction_invalid")
        try:
            if self.extension_root.exists():
                self.extension_root.replace(transaction.rollback)
                transaction.live_moved = True
            if self._swap_failure_injector is not None:
                self._swap_failure_injector("live_moved")
            transaction.staging.replace(self.extension_root)
            transaction.applied = True
        except Exception as exc:
            try:
                self.rollback_restore(transaction)
            except Exception as rollback_exc:
                raise ExtensionBackupError("extension_restore_rollback_failed") from rollback_exc
            if isinstance(exc, ExtensionBackupError):
                raise
            raise ExtensionBackupError("extension_restore_failed") from exc

    def rollback_restore(self, transaction: ExtensionRestoreTransaction) -> None:
        if transaction.closed:
            return
        try:
            if transaction.applied and self.extension_root.exists():
                self._remove_private_tree(self.extension_root, allow_live=True)
            if transaction.live_moved and transaction.rollback.exists():
                transaction.rollback.replace(self.extension_root)
            if transaction.staging.exists():
                self._remove_private_tree(transaction.staging)
            if transaction.rollback.exists():
                self._remove_private_tree(transaction.rollback)
            transaction.closed = True
        except Exception as exc:
            raise ExtensionBackupError("extension_restore_rollback_failed") from exc

    def commit_restore(self, transaction: ExtensionRestoreTransaction) -> None:
        if transaction.closed:
            return
        if not transaction.applied:
            raise ExtensionBackupError("extension_restore_transaction_invalid")
        # The restored live tree is already authoritative. Cleanup is deliberately
        # best-effort so a private rollback deletion failure cannot turn a completed
        # maintenance restore into a false failure after its durable journal commit.
        try:
            if transaction.rollback.exists():
                self._remove_private_tree(transaction.rollback)
            if transaction.staging.exists():
                self._remove_private_tree(transaction.staging)
        except (OSError, ExtensionBackupError):
            pass
        transaction.closed = True

    def _normalize_missing_installs(self, root: Path) -> None:
        repository = ExtensionRepository(root, asteria_version=self.asteria_version)
        for extension_id in repository.discover_extension_ids():
            state_path = repository.state_path(extension_id)
            state = self._read_state_model(state_path)
            changed = False
            active_missing = False
            guidance = "reinstall" if state.source.kind == "managed" else "reconnect"
            if state.source.kind == "managed":
                active_root = repository.managed_version_root(extension_id, state.active_version)
                active_missing = not active_root.is_dir() or not self._package_is_valid(
                    repository,
                    extension_id,
                    state.active_version,
                    state.active_receipt_checksum or "",
                )
                if active_missing and active_root.exists():
                    self._remove_package_path(active_root)
                if not active_missing and state.previous_version is not None:
                    previous_version = state.previous_version
                    previous_root = repository.managed_version_root(extension_id, previous_version)
                    previous_valid = previous_root.is_dir() and self._package_is_valid(
                        repository,
                        extension_id,
                        previous_version,
                        state.previous_receipt_checksum or "",
                    )
                    if not previous_valid:
                        if previous_root.exists():
                            self._remove_package_path(previous_root)
                        updates: dict[str, object] = {
                            "previous_version": None,
                            "previous_receipt_checksum": None,
                        }
                        if state.last_load.version == previous_version:
                            updates.update(
                                {
                                    "last_load": ExtensionLastLoad(status="never", version=None, error_code=None),
                                    "recovery": ExtensionRecoveryState(
                                        status="restart_required",
                                        guidance="restart",
                                    ),
                                }
                            )
                        state = state.model_copy(update=updates)
                        changed = True
            else:
                reference = Path(state.source.external_reference or "")
                active_missing = not reference.is_dir() or _is_reparse_point(reference)
            if active_missing:
                state = state.model_copy(
                    update={
                        "enabled": False,
                        "last_load": ExtensionLastLoad(status="never", version=None, error_code=None),
                        "recovery": ExtensionRecoveryState(status="missing_install", guidance=guidance),
                    }
                )
                changed = True
            if changed:
                write_json_atomic(state_path, state.model_dump(mode="json"))
            repository.read_state(extension_id)

    @staticmethod
    def _package_is_valid(
        repository: ExtensionRepository,
        extension_id: str,
        version: str,
        expected_receipt_checksum: str,
    ) -> bool:
        try:
            repository.read_manifest(extension_id, version)
            repository.read_version_receipt(extension_id, version)
            return repository.receipt_checksum(extension_id, version) == expected_receipt_checksum
        except ExtensionRepositoryError:
            return False

    @staticmethod
    def _read_state_model(path: Path) -> ExtensionState:
        try:
            return ExtensionState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ExtensionBackupError("extension_state_invalid") from exc

    @staticmethod
    def _version_bindings(state: ExtensionState) -> tuple[tuple[str, str], ...]:
        bindings = [(state.active_version, state.active_receipt_checksum)]
        if state.previous_version is not None:
            bindings.append((state.previous_version, state.previous_receipt_checksum))
        return tuple((version, checksum) for version, checksum in bindings if checksum is not None)

    def _regular_tree_files(self, root: Path) -> list[Path]:
        self._assert_safe_tree(root)
        files: list[Path] = []
        for directory, _directories, filenames in os.walk(root, followlinks=False):
            root_path = Path(directory)
            for filename in filenames:
                path = root_path / filename
                if not path.is_file() or _is_reparse_point(path):
                    raise ExtensionBackupError("extension_backup_reparse")
                files.append(path)
        return files

    def _assert_safe_tree(self, root: Path) -> None:
        if not root.is_dir() or _is_reparse_point(root):
            raise ExtensionBackupError("extension_backup_reparse")
        folded: set[str] = set()
        for directory, directories, filenames in os.walk(root, followlinks=False):
            root_path = Path(directory)
            for name in (*directories, *filenames):
                path = root_path / name
                if _is_reparse_point(path):
                    raise ExtensionBackupError("extension_backup_reparse")
                relative = path.relative_to(root).as_posix().casefold()
                if relative in folded:
                    raise ExtensionBackupError("extension_backup_path_collision")
                folded.add(relative)

    def _copy_tree(self, source: Path, destination: Path) -> None:
        self._assert_safe_tree(source)
        shutil.copytree(source, destination, symlinks=False)

    def _tree_fingerprint(self, root: Path) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
        self._assert_safe_tree(root)
        directories: set[str] = set()
        hashes: list[tuple[str, str]] = []
        for directory, child_directories, filenames in os.walk(root, followlinks=False):
            root_path = Path(directory)
            for name in child_directories:
                directories.add((root_path / name).relative_to(root).as_posix().casefold())
            for name in filenames:
                path = root_path / name
                digest = sha256()
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                hashes.append((path.relative_to(root).as_posix().casefold(), digest.hexdigest()))
        return frozenset(directories), tuple(sorted(hashes))

    @staticmethod
    def _remove_package_path(path: Path) -> None:
        if _is_reparse_point(path):
            raise ExtensionBackupError("extension_backup_reparse")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    @staticmethod
    def _unique_sorted(paths: list[Path]) -> list[Path]:
        by_folded: dict[str, Path] = {}
        for path in paths:
            key = str(path).casefold()
            if key in by_folded and by_folded[key] != path:
                raise ExtensionBackupError("extension_backup_path_collision")
            by_folded[key] = path
        return sorted(by_folded.values(), key=lambda path: str(path).casefold())

    def _remove_private_tree(self, path: Path, *, allow_live: bool = False) -> None:
        parent = self.extension_root.parent
        if path.parent != parent or (not allow_live and not path.name.startswith(".extensions.")):
            raise ExtensionBackupError("extension_restore_cleanup_refused")
        shutil.rmtree(path)
