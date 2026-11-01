from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.extensions import (
    ExtensionLastLoad,
    ExtensionManifest,
    ExtensionRecoveryState,
    ExtensionSourceState,
    ExtensionState,
)
from app.services.extension_manifest import ExtensionManifestError, load_extension_manifest
from app.services.extension_package import ExtensionPackageError, extract_extension_zip
from app.services.extension_repository import ExtensionRepository, ExtensionRepositoryError
from app.services.extension_sources import (
    ExtensionSourceDescriptor,
    ExtensionSourceCancelled,
    ExtensionSourceError,
    SourceKind,
    SourceTrust,
    SubprocessGitRunner,
    download_remote_source,
)
from app.services.managed_artifacts import ManagedArtifactCancelled
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)


_EXTENSION_STATE_LOCK = RLock()


class ExtensionInstallError(RuntimeError):
    def __init__(self, message: str, *, code: str = "extension_install_failed") -> None:
        super().__init__(message)
        self.code = code


class ExtensionInstallCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StagedExtension:
    extension_id: str
    version: str
    package_checksum: str
    staging_path: Path
    source: ExtensionSourceDescriptor
    manifest: ExtensionManifest


class ExtensionInstaller:
    """Validates and publishes extension packages without importing or executing them."""

    def __init__(
        self,
        repository: ExtensionRepository,
        *,
        git_runner: SubprocessGitRunner | None = None,
    ) -> None:
        self.repository = repository
        self.staging_root = repository.extension_root / ".staging"
        self.download_root = repository.extension_root / ".downloads"
        self.git_runner = git_runner or SubprocessGitRunner()

    def stage_zip(self, package: Path, source: ExtensionSourceDescriptor) -> StagedExtension:
        if source.kind not in {
            SourceKind.LOCAL_ZIP,
            SourceKind.HTTPS_ZIP,
            SourceKind.GITHUB_RELEASE_ZIP,
            SourceKind.GIT,
        }:
            raise ExtensionInstallError("extension source is not a package", code="extension_source_invalid")
        if source.kind is SourceKind.LOCAL_ZIP and os.path.normcase(os.path.abspath(package)) != os.path.normcase(
            os.path.abspath(source.reference)
        ):
            raise ExtensionInstallError(
                "local package path does not match its source snapshot",
                code="extension_source_mismatch",
            )
        try:
            package_size = package.stat().st_size
            if source.expected_size_bytes is not None and package_size != source.expected_size_bytes:
                raise ExtensionInstallError("package size mismatch", code="extension_size_mismatch")
            actual_hash = _hash_file(package)
            if source.expected_sha256 is not None and actual_hash != source.expected_sha256:
                raise ExtensionInstallError("package checksum mismatch", code="extension_checksum_mismatch")
            if source.expected_sha256 is None and source.trust is not SourceTrust.UNVERIFIED_DIRECT:
                raise ExtensionInstallError("package checksum is required", code="extension_checksum_required")
            staging = _private_work_path(self.repository.extension_root, ".staging", uuid4().hex)
            extract_extension_zip(package, staging)
            manifest = self._validate_package_root(staging)
            return StagedExtension(
                extension_id=manifest.id,
                version=manifest.version,
                package_checksum=actual_hash,
                staging_path=staging,
                source=source,
                manifest=manifest,
            )
        except ExtensionInstallError:
            if "staging" in locals():
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except (OSError, ExtensionPackageError, ExtensionManifestError, ExtensionRepositoryError, ValidationError) as exc:
            if "staging" in locals():
                shutil.rmtree(staging, ignore_errors=True)
            raise ExtensionInstallError(
                _safe_install_message(exc),
                code=getattr(exc, "code", "extension_package_invalid"),
            ) from exc

    def stage_remote(
        self,
        source: ExtensionSourceDescriptor,
        *,
        progress_callback=None,
        cancel_requested=None,
    ) -> StagedExtension:
        if source.kind not in {SourceKind.HTTPS_ZIP, SourceKind.GITHUB_RELEASE_ZIP}:
            raise ExtensionInstallError("extension source is not a remote ZIP", code="extension_source_invalid")
        package = _private_work_path(
            self.repository.extension_root,
            ".downloads",
            f"{uuid4().hex}.zip",
        )
        try:
            actual_hash = download_remote_source(
                source,
                package,
                temp_root=self.download_root / "temp",
                progress_callback=progress_callback,
                cancel_requested=cancel_requested,
            )
            staged_source = source
            if source.expected_sha256 is None:
                staged_source = ExtensionSourceDescriptor(
                    kind=source.kind,
                    reference=source.reference,
                    expected_size_bytes=source.expected_size_bytes,
                    trust=SourceTrust.UNVERIFIED_DIRECT,
                    confirmed_unverified=True,
                )
            staged = self.stage_zip(package, staged_source)
            if staged.package_checksum != actual_hash:
                shutil.rmtree(staged.staging_path, ignore_errors=True)
                raise ExtensionInstallError("downloaded package hash changed", code="extension_checksum_mismatch")
            return staged
        except ManagedArtifactCancelled as exc:
            raise ExtensionInstallCancelled("extension download cancelled") from exc
        except ExtensionSourceError as exc:
            raise ExtensionInstallError(str(exc), code="extension_download_failed") from exc
        finally:
            package.unlink(missing_ok=True)
            package.with_name(f"{package.name}.asteria-managed.json").unlink(missing_ok=True)

    def stage_git(
        self,
        source: ExtensionSourceDescriptor,
        *,
        cancel_requested=None,
    ) -> StagedExtension:
        if source.kind is not SourceKind.GIT:
            raise ExtensionInstallError("extension source is not Git", code="extension_source_invalid")
        archive = _private_work_path(
            self.repository.extension_root,
            ".downloads",
            f"{uuid4().hex}.zip",
        )
        try:
            commit = self.git_runner.export_archive(
                source.reference,
                archive,
                revision=source.resolved_commit,
                cancel_requested=cancel_requested,
            )
            if source.resolved_commit is not None and commit != source.resolved_commit:
                raise ExtensionInstallError(
                    "Git source commit does not match its immutable snapshot",
                    code="extension_git_commit_mismatch",
                )
            resolved = source.with_resolved_commit(commit)
            if resolved.expected_size_bytes is None:
                resolved = ExtensionSourceDescriptor(
                    kind=resolved.kind,
                    reference=resolved.reference,
                    expected_sha256=resolved.expected_sha256,
                    expected_size_bytes=archive.stat().st_size,
                    trust=resolved.trust,
                    confirmed_unverified=resolved.confirmed_unverified,
                    resolved_commit=resolved.resolved_commit,
                )
            return self.stage_zip(archive, resolved)
        except ExtensionSourceCancelled as exc:
            raise ExtensionInstallCancelled("Git source operation cancelled") from exc
        except ExtensionInstallError:
            raise
        except (OSError, ExtensionSourceError) as exc:
            raise ExtensionInstallError("Git source could not be staged", code="extension_git_stage_failed") from exc
        finally:
            archive.unlink(missing_ok=True)

    def install(self, staged: StagedExtension, source: ExtensionSourceDescriptor | None = None) -> ExtensionState:
        source = source or staged.source
        with _EXTENSION_STATE_LOCK:
            state_path = self.repository.state_path(staged.extension_id)
            extension_directory = state_path.parent
            if state_path.exists() or (extension_directory.exists() and any(extension_directory.iterdir())):
                self._discard_staged(staged)
                raise ExtensionInstallError("extension ID already exists", code="extension_id_collision")
            return self._publish_new_version(staged, source, previous=None)

    def update(
        self,
        staged: StagedExtension,
        source: ExtensionSourceDescriptor | None = None,
        *,
        confirm: bool = False,
    ) -> ExtensionState:
        if not confirm:
            self._discard_staged(staged)
            raise ExtensionInstallError(
                "explicit update confirmation is required",
                code="extension_update_confirmation_required",
            )
        with _EXTENSION_STATE_LOCK:
            current = self.repository.read_state(staged.extension_id)
            if current.source.kind != "managed":
                self._discard_staged(staged)
                raise ExtensionInstallError("unmanaged extension cannot update", code="extension_update_unsupported")
            if current.pending_action is not None or current.recovery.status != "ready":
                self._discard_staged(staged)
                raise ExtensionInstallError("extension has a pending lifecycle action", code="extension_action_pending")
            if staged.version == current.active_version:
                self._discard_staged(staged)
                raise ExtensionInstallError("extension version already exists", code="extension_version_collision")
            return self._publish_new_version(staged, source or staged.source, previous=current)

    def install_manual(self, source: ExtensionSourceDescriptor) -> ExtensionState:
        if source.kind is not SourceKind.MANUAL_FOLDER:
            raise ExtensionInstallError("source is not a manual folder", code="extension_source_invalid")
        root = Path(source.reference)
        try:
            _assert_regular_source_tree(root)
            manifest = self._validate_package_root(root)
            with _EXTENSION_STATE_LOCK:
                state_path = self.repository.state_path(manifest.id)
                if state_path.exists() or (state_path.parent.exists() and any(state_path.parent.iterdir())):
                    raise ExtensionInstallError("extension ID already exists", code="extension_id_collision")
                state = ExtensionState(
                    active_version=manifest.version,
                    previous_version=None,
                    active_receipt_checksum=None,
                    previous_receipt_checksum=None,
                    enabled=True,
                    source=ExtensionSourceState(kind="unmanaged", external_reference=str(root.absolute())),
                    last_load=ExtensionLastLoad(status="never"),
                    recovery=ExtensionRecoveryState(status="restart_required", guidance="restart"),
                    pending_action="install",
                )
                self.repository.write_state(manifest.id, state)
                return state
        except ExtensionInstallError:
            raise
        except (OSError, ExtensionManifestError, ExtensionRepositoryError, ValidationError) as exc:
            raise ExtensionInstallError(
                _safe_install_message(exc),
                code=getattr(exc, "code", "extension_manual_invalid"),
            ) from exc

    def rollback(self, extension_id: str, *, confirm: bool = False) -> ExtensionState:
        if not confirm:
            raise ExtensionInstallError(
                "explicit rollback confirmation is required",
                code="extension_rollback_confirmation_required",
            )
        with _EXTENSION_STATE_LOCK:
            return self._rollback_locked(extension_id)

    def _rollback_locked(self, extension_id: str) -> ExtensionState:
        state = self.repository.read_state(extension_id)
        if (
            state.source.kind != "managed"
            or state.previous_version is None
            or state.previous_receipt_checksum is None
            or state.pending_action is not None
            or state.recovery.status not in {"ready", "load_failed"}
        ):
            raise ExtensionInstallError("rollback is unavailable", code="extension_rollback_unavailable")
        updated = _validated_state_update(
            state,
            active_version=state.previous_version,
            previous_version=state.active_version,
            active_receipt_checksum=state.previous_receipt_checksum,
            previous_receipt_checksum=state.active_receipt_checksum,
            recovery=ExtensionRecoveryState(status="restart_required", guidance="restart"),
            pending_action="rollback",
        )
        self.repository.write_state(extension_id, updated)
        return updated

    def uninstall(self, extension_id: str, *, confirm: bool = False) -> ExtensionState:
        if not confirm:
            raise ExtensionInstallError(
                "explicit uninstall confirmation is required",
                code="extension_uninstall_confirmation_required",
            )
        with _EXTENSION_STATE_LOCK:
            state = self.repository.read_state(extension_id)
            if state.pending_action is not None:
                raise ExtensionInstallError("extension has a pending lifecycle action", code="extension_action_pending")
            updated = _validated_state_update(
                state,
                enabled=False,
                recovery=ExtensionRecoveryState(status="restart_required", guidance="restart"),
                pending_action="uninstall",
            )
            self.repository.write_state(extension_id, updated)
            return updated

    def set_enabled(self, extension_id: str, *, enabled: bool) -> ExtensionState:
        with _EXTENSION_STATE_LOCK:
            state = self.repository.read_state(extension_id)
            if state.pending_action is not None:
                raise ExtensionInstallError(
                    "extension has a pending lifecycle action",
                    code="extension_action_pending",
                )
            if enabled and state.recovery.status == "missing_install":
                raise ExtensionInstallError(
                    "missing extension cannot be enabled",
                    code="extension_install_missing",
                )
            if state.enabled == enabled:
                return state
            updated = _validated_state_update(state, enabled=enabled)
            self.repository.write_state(extension_id, updated)
            return updated

    def complete_successful_load(self, extension_id: str, version: str) -> ExtensionState:
        state = self.repository.read_state(extension_id)
        if state.pending_action not in {"install", "update", "rollback"} or state.active_version != version:
            raise ExtensionInstallError("successful load does not match pending action", code="extension_load_finalize_invalid")
        ready = _validated_state_update(
            state,
            last_load=ExtensionLastLoad(status="success", version=version),
            recovery=ExtensionRecoveryState(status="ready", guidance="none"),
            pending_action=None,
        )
        self.repository.write_state(extension_id, ready)
        if ready.source.kind == "managed":
            self._prune_stale_versions(extension_id, ready)
        return ready

    def complete_uninstall(self, extension_id: str) -> None:
        state = self.repository.read_state(extension_id)
        if state.pending_action != "uninstall" or state.enabled:
            raise ExtensionInstallError("uninstall is not pending", code="extension_uninstall_finalize_invalid")
        extension_directory = self.repository.state_path(extension_id).parent
        _assert_owned_directory(self.repository.extension_root, extension_directory)
        shutil.rmtree(extension_directory)

    def _publish_new_version(
        self,
        staged: StagedExtension,
        source: ExtensionSourceDescriptor,
        *,
        previous: ExtensionState | None,
    ) -> ExtensionState:
        destination = self.repository.managed_version_root(staged.extension_id, staged.version)
        if destination.exists() or destination.is_symlink():
            self._discard_staged(staged)
            raise ExtensionInstallError("extension version already exists", code="extension_version_collision")
        moved = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged.staging_path, destination)
            moved = True
            receipt_checksum = self.repository.write_version_receipt(
                staged.extension_id,
                staged.version,
                package_checksum=staged.package_checksum,
                source_trust=source.trust.value if source.trust is not SourceTrust.MANUAL else None,
                resolved_commit=source.resolved_commit,
            )
            state = ExtensionState(
                active_version=staged.version,
                previous_version=previous.active_version if previous is not None else None,
                active_receipt_checksum=receipt_checksum,
                previous_receipt_checksum=previous.active_receipt_checksum if previous is not None else None,
                enabled=previous.enabled if previous is not None else True,
                source=ExtensionSourceState(
                    kind="managed",
                    install_url=source.reference if source.kind in {
                        SourceKind.HTTPS_ZIP,
                        SourceKind.GITHUB_RELEASE_ZIP,
                        SourceKind.GIT,
                    } else None,
                ),
                last_load=previous.last_load if previous is not None else ExtensionLastLoad(status="never"),
                recovery=ExtensionRecoveryState(status="restart_required", guidance="restart"),
                pending_action="update" if previous is not None else "install",
            )
            self.repository.write_state(staged.extension_id, state)
            return state
        except (OSError, ExtensionRepositoryError, ValidationError) as exc:
            if moved:
                shutil.rmtree(destination, ignore_errors=True)
                if previous is None:
                    _remove_empty_install_parents(
                        destination.parent,
                        self.repository.extension_root,
                    )
            else:
                self._discard_staged(staged)
            raise ExtensionInstallError(
                _safe_install_message(exc),
                code=getattr(exc, "code", "extension_update_failed" if previous else "extension_install_failed"),
            ) from exc

    def _validate_package_root(self, root: Path) -> ExtensionManifest:
        manifest = load_extension_manifest(
            root / "asteria-extension.json",
            asteria_version=self.repository.asteria_version,
        )
        module_name, _callable = manifest.entrypoints.backend.split(":", 1)
        module_path = PurePosixPath(*module_name.split("."))
        backend_module = root / f"{module_path.as_posix()}.py"
        backend_package = root / module_path / "__init__.py"
        frontend = root / PurePosixPath(manifest.entrypoints.frontend)
        if not (_regular_file(backend_module) or _regular_file(backend_package)) or not _regular_file(frontend):
            raise ExtensionInstallError(
                "extension entrypoint bundle is missing",
                code="extension_entrypoint_invalid",
            )
        return manifest

    def _prune_stale_versions(self, extension_id: str, state: ExtensionState) -> None:
        versions_root = self.repository.managed_version_root(extension_id, state.active_version).parent
        keep = {state.active_version, state.previous_version}
        if not versions_root.exists():
            return
        for candidate in versions_root.iterdir():
            if candidate.name in keep:
                continue
            _assert_owned_directory(versions_root, candidate)
            shutil.rmtree(candidate)

    @staticmethod
    def _discard_staged(staged: StagedExtension) -> None:
        shutil.rmtree(staged.staging_path, ignore_errors=True)


class _ExtensionOperationExecutor:
    version = "extension-lifecycle-v1"

    def __init__(self, installer: ExtensionInstaller, action: str) -> None:
        self.installer = installer
        self.action = action

    def resource_claims(self, snapshot: dict[str, Any]) -> tuple[str, ...]:
        try:
            source = ExtensionSourceDescriptor.from_snapshot(dict(snapshot["source"]))
        except (ExtensionSourceError, KeyError, TypeError):
            return ("disk_write_heavy",)
        if source.kind in {SourceKind.HTTPS_ZIP, SourceKind.GITHUB_RELEASE_ZIP, SourceKind.GIT}:
            return ("network_download", "disk_write_heavy")
        return ("disk_write_heavy",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        snapshot = dict(context.request_snapshot)
        staged: StagedExtension | None = None
        try:
            context.report_progress(0, 2, "steps", phase="validating")
            if self.action in {"install", "update"}:
                source = ExtensionSourceDescriptor.from_snapshot(dict(snapshot["source"]))
                expected_keys = {"source", "confirm"}
                if source.kind is SourceKind.LOCAL_ZIP:
                    expected_keys.add("package_path")
                optional_keys = {
                    "expected_extension_id",
                    "expected_version",
                    "expected_execution_level",
                    "expected_permissions",
                }
                if (
                    not expected_keys.issubset(snapshot)
                    or not set(snapshot).issubset(expected_keys | optional_keys)
                    or snapshot.get("confirm") is not True
                ):
                    raise ExtensionInstallError(
                        "explicit extension action confirmation is required",
                        code="extension_action_confirmation_required",
                    )
                if source.kind is SourceKind.MANUAL_FOLDER:
                    if self.action != "install":
                        raise ExtensionInstallError("manual extension cannot update", code="extension_update_unsupported")
                    context.raise_if_cancelled()
                    state = self.installer.install_manual(source)
                    extension_id = _manifest_id_from_manual(source)
                    version = state.active_version
                else:
                    if source.kind is SourceKind.LOCAL_ZIP:
                        staged = self.installer.stage_zip(Path(str(snapshot["package_path"])), source)
                    elif source.kind in {SourceKind.HTTPS_ZIP, SourceKind.GITHUB_RELEASE_ZIP}:
                        staged = self.installer.stage_remote(
                            source,
                            progress_callback=lambda _component, current, total: context.report_progress(
                                current,
                                total,
                                "bytes",
                                phase="downloading",
                            ),
                            cancel_requested=lambda: context.cancel_requested,
                        )
                    else:
                        staged = self.installer.stage_git(
                            source,
                            cancel_requested=lambda: context.cancel_requested,
                        )
                    extension_id = staged.extension_id
                    version = staged.version
                    expected_extension_id = snapshot.get("expected_extension_id")
                    expected_version = snapshot.get("expected_version")
                    if (
                        expected_extension_id is not None
                        and expected_extension_id != extension_id
                    ):
                        raise ExtensionInstallError(
                            "extension package ID does not match request",
                            code="extension_source_mismatch",
                        )
                    if expected_version is not None and expected_version != version:
                        raise ExtensionInstallError(
                            "extension package version does not match request",
                            code="extension_source_mismatch",
                        )
                    expected_execution_level = snapshot.get("expected_execution_level")
                    if (
                        expected_execution_level is not None
                        and expected_execution_level != staged.manifest.execution_level
                    ):
                        raise ExtensionInstallError(
                            "extension package execution level does not match request",
                            code="extension_source_mismatch",
                        )
                    expected_permissions = snapshot.get("expected_permissions")
                    if (
                        expected_permissions is not None
                        and expected_permissions != staged.manifest.permissions.model_dump(mode="json")
                    ):
                        raise ExtensionInstallError(
                            "extension package permissions do not match request",
                            code="extension_source_mismatch",
                        )
                    context.raise_if_cancelled()
                    state = (
                        self.installer.install(staged)
                        if self.action == "install"
                        else self.installer.update(staged, confirm=snapshot.get("confirm") is True)
                    )
                    staged = None
            elif self.action == "rollback":
                if set(snapshot) != {"extension_id", "confirm"} or not isinstance(snapshot.get("confirm"), bool):
                    raise ExtensionInstallError("extension request is invalid", code="extension_request_invalid")
                extension_id = _snapshot_extension_id(snapshot)
                state = self.installer.rollback(extension_id, confirm=snapshot.get("confirm") is True)
            elif self.action == "uninstall":
                if set(snapshot) != {"extension_id", "confirm"} or not isinstance(snapshot.get("confirm"), bool):
                    raise ExtensionInstallError("extension request is invalid", code="extension_request_invalid")
                extension_id = _snapshot_extension_id(snapshot)
                state = self.installer.uninstall(extension_id, confirm=snapshot.get("confirm") is True)
            else:
                raise ExtensionInstallError("extension action is unsupported", code="extension_request_invalid")
            context.report_progress(2, 2, "steps", phase="restart_required")
            return OperationExecutionResult(artifact_refs=(f"extension:{extension_id}",))
        except ExtensionInstallCancelled as exc:
            if staged is not None:
                self.installer._discard_staged(staged)
            context.acknowledge_cancel()
            raise OperationCancelled() from exc
        except OperationCancelled:
            if staged is not None:
                self.installer._discard_staged(staged)
            context.acknowledge_cancel()
            raise
        except ExtensionInstallError as exc:
            if staged is not None:
                self.installer._discard_staged(staged)
            retryable = exc.code in {
                "extension_download_failed",
                "extension_git_stage_failed",
                "extension_install_failed",
                "extension_update_failed",
            }
            raise OperationExecutionError(exc.code, retryable=retryable) from exc
        except (KeyError, TypeError, ValueError, OSError, ExtensionSourceError) as exc:
            if staged is not None:
                self.installer._discard_staged(staged)
            raise OperationExecutionError("extension_request_invalid", retryable=False) from exc


def register_extension_operations(coordinator: Any, installer: ExtensionInstaller) -> None:
    for action in ("install", "update", "rollback", "uninstall"):
        coordinator.register_executor(f"extension.{action}", _ExtensionOperationExecutor(installer, action))


def configure_extension_operation_coordinator(
    coordinator: Any,
    *,
    extension_root: Path,
    asteria_version: str = "1.2.0",
) -> None:
    register_extension_operations(
        coordinator,
        ExtensionInstaller(
            ExtensionRepository(extension_root, asteria_version=asteria_version),
        ),
    )


def _validated_state_update(state: ExtensionState, **updates: Any) -> ExtensionState:
    payload = state.model_dump(mode="python")
    for key, value in updates.items():
        payload[key] = value.model_dump(mode="python") if hasattr(value, "model_dump") else value
    return ExtensionState.model_validate(payload)


def _snapshot_extension_id(snapshot: dict[str, Any]) -> str:
    extension_id = snapshot.get("extension_id")
    if not isinstance(extension_id, str) or not extension_id:
        raise ExtensionInstallError("extension ID is missing", code="extension_request_invalid")
    return extension_id


def _manifest_id_from_manual(source: ExtensionSourceDescriptor) -> str:
    try:
        return str(load_extension_manifest(Path(source.reference) / "asteria-extension.json").id)
    except ExtensionManifestError as exc:
        raise ExtensionInstallError("manual extension manifest is invalid", code=exc.code) from exc


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(details.st_mode) and not _is_reparse(path, details)


def _assert_regular_source_tree(root: Path) -> None:
    if not root.is_dir() or _is_reparse(root):
        raise ExtensionInstallError("manual extension folder is missing or unsafe", code="extension_manual_invalid")
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *filenames):
            candidate = current_path / name
            details = candidate.lstat()
            if _is_reparse(candidate, details):
                raise ExtensionInstallError("manual extension crosses a reparse point", code="extension_reparse_traversal")
            if name in filenames and not stat.S_ISREG(details.st_mode):
                raise ExtensionInstallError("manual extension contains a special file", code="extension_manual_invalid")


def _assert_owned_directory(root: Path, candidate: Path) -> None:
    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        contained = candidate_absolute != root_absolute and candidate_absolute.is_relative_to(root_absolute)
    except ValueError:
        contained = False
    if not contained or not candidate_absolute.is_dir() or _is_reparse(candidate_absolute):
        raise ExtensionInstallError("extension cleanup path is unsafe", code="extension_path_escape")


def _private_work_path(extension_root: Path, directory_name: str, filename: str) -> Path:
    root = Path(os.path.abspath(extension_root))
    directory = root / directory_name
    for candidate in (*reversed(root.parents), root, directory):
        if candidate.exists() and _is_reparse(candidate):
            raise ExtensionInstallError(
                "extension staging path crosses a reparse point",
                code="extension_reparse_traversal",
            )
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / filename
    if not result.is_relative_to(root) or result.exists() or result.is_symlink():
        raise ExtensionInstallError("extension staging path is unsafe", code="extension_path_escape")
    return result


def _remove_empty_install_parents(versions_root: Path, extension_root: Path) -> None:
    extension_directory = versions_root.parent
    root = Path(os.path.abspath(extension_root))
    for candidate in (versions_root, extension_directory):
        absolute = Path(os.path.abspath(candidate))
        if not absolute.is_relative_to(root) or absolute == root or _is_reparse(absolute):
            return
        try:
            absolute.rmdir()
        except OSError:
            return


def _is_reparse(path: Path, details: os.stat_result | None = None) -> bool:
    try:
        details = details or path.lstat()
        is_junction = bool(getattr(os.path, "isjunction", lambda _path: False)(path))
        return stat.S_ISLNK(details.st_mode) or is_junction or bool(getattr(details, "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


def _safe_install_message(exc: Exception) -> str:
    code = getattr(exc, "code", "")
    if code == "extension_manifest_incompatible":
        return "extension manifest is not compatible with this Asteria version"
    if code == "extension_entrypoint_invalid":
        return "extension entrypoint bundle is missing"
    return "extension package could not be installed"
