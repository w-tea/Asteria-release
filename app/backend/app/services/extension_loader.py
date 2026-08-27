from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from types import ModuleType
from typing import Callable, Literal

from fastapi import FastAPI

from app.extensions.sdk import ExtensionRegistration, ExtensionSDKContext, ExtensionSDKError
from app.schemas.extensions import (
    ExtensionLastLoad,
    ExtensionManifest,
    ExtensionRecoveryState,
    ExtensionState,
)
from app.services.extension_installer import ExtensionInstallError, ExtensionInstaller
from app.services.extension_manifest import ExtensionManifestError, load_extension_manifest
from app.services.operation_coordinator import ExecutorGroupSnapshot, OperationCoordinator
from app.services.extension_registry import ExtensionRegistry
from app.services.extension_repository import (
    ExtensionRepository,
    ExtensionRepositoryError,
    ResolvedExtensionEntrypoints,
)


ExtensionLoadStatus = Literal["loaded", "failed", "skipped"]
@dataclass(frozen=True, slots=True)
class ExtensionLoadRecord:
    extension_id: str
    version: str | None
    status: ExtensionLoadStatus
    error_code: str | None = None
    module_name: str = ""
    registration: ExtensionRegistration | None = None


class ExtensionLoader:
    def __init__(
        self,
        repository: ExtensionRepository,
        *,
        operation_coordinator: OperationCoordinator | None = None,
    ) -> None:
        self.repository = repository
        self.registry = ExtensionRegistry(repository)
        self.installer = ExtensionInstaller(repository)
        self.operation_coordinator = operation_coordinator

    def load_into(self, app: FastAPI) -> tuple[ExtensionLoadRecord, ...]:
        load_records: list[ExtensionLoadRecord] = []
        registry_records = self.registry.scan()
        for registry_record in registry_records:
            state = registry_record.state
            if registry_record.status == "error" or state is None:
                load_records.append(
                    ExtensionLoadRecord(
                        extension_id=registry_record.extension_id,
                        version=None,
                        status="failed",
                        error_code=registry_record.error_code or "extension_registry_invalid",
                    )
                )
                continue
            if state.pending_action == "uninstall":
                try:
                    self.installer.complete_uninstall(registry_record.extension_id)
                    load_records.append(
                        ExtensionLoadRecord(
                            extension_id=registry_record.extension_id,
                            version=state.active_version,
                            status="skipped",
                            error_code="extension_uninstalled",
                        )
                    )
                except (ExtensionInstallError, OSError):
                    load_records.append(
                        ExtensionLoadRecord(
                            extension_id=registry_record.extension_id,
                            version=state.active_version,
                            status="failed",
                            error_code="extension_uninstall_finalize_failed",
                        )
                    )
                continue
            if not state.enabled or state.recovery.status == "missing_install":
                load_records.append(
                    ExtensionLoadRecord(
                        extension_id=registry_record.extension_id,
                        version=state.active_version,
                        status="skipped",
                        error_code="extension_disabled" if not state.enabled else "extension_install_missing",
                    )
                )
                continue
            load_records.append(
                self._load_one(
                    app,
                    registry_record.extension_id,
                    state,
                    registry_record.manifest,
                )
            )
        try:
            self.registry.scan()
        except (ExtensionRepositoryError, OSError, ValueError):
            pass
        return tuple(load_records)

    def _load_one(
        self,
        app: FastAPI,
        extension_id: str,
        state: ExtensionState,
        managed_manifest: ExtensionManifest | None,
    ) -> ExtensionLoadRecord:
        module_namespace = ""
        try:
            manifest, entrypoints, package_root = self._resolve_extension(
                extension_id,
                state,
                managed_manifest,
            )
            if manifest.execution_level == "standard":
                return self._load_standard(extension_id, state, manifest)
            module, module_namespace = _import_extension_module(
                extension_id,
                state.active_version,
                package_root,
                manifest.entrypoints.backend.split(":", 1)[0],
                entrypoints.backend_file,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self._mark_failed(extension_id, state, "extension_import_failed")
            _remove_modules(module_namespace)
            return ExtensionLoadRecord(
                extension_id=extension_id,
                version=state.active_version,
                status="failed",
                error_code="extension_import_failed",
            )

        try:
            factory = getattr(module, entrypoints.backend_callable, None)
            if not callable(factory):
                raise ExtensionSDKError("extension entrypoint callable is missing")
            context = ExtensionSDKContext(manifest)
            returned = factory(context)
            if returned is not None:
                self._mark_failed(extension_id, state, "extension_return_invalid")
                _remove_modules(module_namespace)
                return ExtensionLoadRecord(
                    extension_id=extension_id,
                    version=state.active_version,
                    status="failed",
                    error_code="extension_return_invalid",
                )
            registration = context.finalize()
            rollback_registration = self._commit_registration(app, registration)
            try:
                self._mark_ready(extension_id, state)
            except BaseException:
                rollback_registration()
                raise
            return ExtensionLoadRecord(
                extension_id=extension_id,
                version=state.active_version,
                status="loaded",
                module_name=module.__name__,
                registration=registration,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except ExtensionSDKError:
            code = "extension_registration_invalid"
        except BaseException:
            code = "extension_registration_failed"
        self._mark_failed(extension_id, state, code)
        _remove_modules(module_namespace)
        return ExtensionLoadRecord(
            extension_id=extension_id,
            version=state.active_version,
            status="failed",
            error_code=code,
        )

    def _load_standard(
        self,
        extension_id: str,
        state: ExtensionState,
        manifest: ExtensionManifest,
    ) -> ExtensionLoadRecord:
        backend_actions = tuple(
            item
            for item in manifest.contributions.actions
            if item.surface in {"backend.api_router", "backend.operation"}
        )
        if backend_actions:
            self._mark_failed(
                extension_id,
                state,
                "extension_standard_contribution_unsupported",
            )
            return ExtensionLoadRecord(
                extension_id=extension_id,
                version=state.active_version,
                status="failed",
                error_code="extension_standard_contribution_unsupported",
            )
        registration = ExtensionRegistration(
            extension_id=manifest.id,
            version=manifest.version,
            routers=(),
            workspaces=manifest.contributions.workspaces,
            panels=manifest.contributions.panels,
            actions=manifest.contributions.actions,
            operations=(),
        )
        try:
            self._mark_ready(extension_id, state)
        except BaseException:
            self._mark_failed(extension_id, state, "extension_registration_failed")
            return ExtensionLoadRecord(
                extension_id=extension_id,
                version=state.active_version,
                status="failed",
                error_code="extension_registration_failed",
            )
        return ExtensionLoadRecord(
            extension_id=extension_id,
            version=state.active_version,
            status="loaded",
            registration=registration,
        )

    def _resolve_extension(
        self,
        extension_id: str,
        state: ExtensionState,
        managed_manifest: ExtensionManifest | None,
    ) -> tuple[ExtensionManifest, ResolvedExtensionEntrypoints, Path]:
        if state.source.kind == "managed":
            manifest = managed_manifest or self.repository.read_manifest(extension_id, state.active_version)
            entrypoints = self.repository.resolve_entrypoints(extension_id, state.active_version, manifest)
            return manifest, entrypoints, self.repository.managed_version_root(extension_id, state.active_version)
        package_root = Path(state.source.external_reference or "")
        manifest = load_extension_manifest(
            package_root / "asteria-extension.json",
            asteria_version=self.repository.asteria_version,
        )
        if manifest.id != extension_id or manifest.version != state.active_version:
            raise ExtensionManifestError(
                "extension manifest does not match unmanaged state",
                code="extension_manifest_mismatch",
            )
        module_name, callable_name = manifest.entrypoints.backend.split(":", 1)
        module_relative = PurePosixPath(*module_name.split("."))
        module_file = package_root / f"{module_relative.as_posix()}.py"
        package_file = package_root / module_relative / "__init__.py"
        backend_file = module_file if _regular_contained_file(package_root, module_file) else package_file
        frontend_file = package_root / PurePosixPath(manifest.entrypoints.frontend)
        if not _regular_contained_file(package_root, backend_file) or not _regular_contained_file(
            package_root,
            frontend_file,
        ):
            raise ExtensionRepositoryError(
                "unmanaged extension entrypoint is missing or invalid",
                code="extension_entrypoint_invalid",
            )
        return (
            manifest,
            ResolvedExtensionEntrypoints(
                backend_file=backend_file,
                backend_callable=callable_name,
                frontend_file=frontend_file,
            ),
            package_root,
        )

    def _commit_registration(
        self,
        app: FastAPI,
        registration: ExtensionRegistration,
    ) -> Callable[[], None]:
        if registration.operations and self.operation_coordinator is None:
            raise ExtensionSDKError("extension operation registrar is unavailable")
        route_start = len(app.router.routes)
        group_snapshot: ExecutorGroupSnapshot | None = None
        group_id = f"extension:{registration.extension_id}"
        route_prefix = f"/api/extensions/{registration.extension_id}"
        try:
            for router in registration.routers:
                app.include_router(router.router, prefix=route_prefix)
            if self.operation_coordinator is not None:
                group_snapshot = self.operation_coordinator.replace_executor_group(
                    group_id,
                    {operation.kind: operation.executor for operation in registration.operations},
                )
        except BaseException:
            del app.router.routes[route_start:]
            if group_snapshot is not None:
                assert self.operation_coordinator is not None
                self.operation_coordinator.restore_executor_group(group_snapshot)
            raise

        def rollback() -> None:
            del app.router.routes[route_start:]
            if group_snapshot is not None:
                assert self.operation_coordinator is not None
                self.operation_coordinator.restore_executor_group(group_snapshot)

        return rollback

    def _mark_ready(self, extension_id: str, state: ExtensionState) -> None:
        if state.pending_action in {"install", "update", "rollback"}:
            self.installer.complete_successful_load(extension_id, state.active_version)
            return
        if state.recovery.status == "ready" and state.last_load.status == "success":
            return
        payload = state.model_dump(mode="python")
        payload.update(
            last_load=ExtensionLastLoad(status="success", version=state.active_version).model_dump(mode="python"),
            recovery=ExtensionRecoveryState(status="ready", guidance="none").model_dump(mode="python"),
            pending_action=None,
        )
        self.repository.write_state(extension_id, ExtensionState.model_validate(payload))

    def _mark_failed(self, extension_id: str, state: ExtensionState, error_code: str) -> None:
        payload = state.model_dump(mode="python")
        payload.update(
            last_load=ExtensionLastLoad(
                status="failed",
                version=state.active_version,
                error_code=error_code,
            ).model_dump(mode="python"),
            recovery=ExtensionRecoveryState(status="load_failed", guidance="safe_mode").model_dump(mode="python"),
            pending_action=None,
        )
        try:
            self.repository.write_state(extension_id, ExtensionState.model_validate(payload))
        except (ExtensionRepositoryError, ValueError, OSError):
            pass


def _import_extension_module(
    extension_id: str,
    version: str,
    package_root: Path,
    declared_module: str,
    entrypoint_file: Path,
) -> tuple[ModuleType, str]:
    normalized_id = re.sub(r"[^A-Za-z0-9_]", "_", extension_id.casefold())
    normalized_version = re.sub(r"[^A-Za-z0-9_]", "_", version.casefold())
    identity = sha256(f"{extension_id}\0{package_root.absolute()}".encode("utf-8")).hexdigest()[:12]
    namespace = f"asteria_user_extensions.{normalized_id}.v{normalized_version}_{identity}"
    _ensure_namespace_package("asteria_user_extensions", ())
    _ensure_namespace_package(f"asteria_user_extensions.{normalized_id}", ())
    _ensure_namespace_package(namespace, (str(package_root),))
    module_parts = declared_module.split(".")
    for index in range(1, len(module_parts)):
        parent_name = f"{namespace}.{'/'.join(module_parts[:index]).replace('/', '.')}"
        parent_path = package_root.joinpath(*module_parts[:index])
        _ensure_namespace_package(parent_name, (str(parent_path),))
    module_name = f"{namespace}.{declared_module}"
    is_package = entrypoint_file.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        entrypoint_file,
        submodule_search_locations=[str(entrypoint_file.parent)] if is_package else None,
    )
    if spec is None or spec.loader is None:
        raise ImportError("extension module specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _remove_modules(namespace)
        raise
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module, namespace


def _ensure_namespace_package(name: str, paths: tuple[str, ...]) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        return
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = list(paths)  # type: ignore[attr-defined]
    sys.modules[name] = module


def _remove_modules(namespace: str) -> None:
    if not namespace:
        return
    for name in tuple(sys.modules):
        if name == namespace or name.startswith(f"{namespace}."):
            sys.modules.pop(name, None)
    parent = namespace.rsplit(".", 1)[0]
    if parent.startswith("asteria_user_extensions.") and not any(
        name.startswith(f"{parent}.") for name in sys.modules
    ):
        sys.modules.pop(parent, None)


def clear_loaded_extension_modules() -> None:
    for name in tuple(sys.modules):
        if name == "asteria_user_extensions" or name.startswith("asteria_user_extensions."):
            sys.modules.pop(name, None)


def _regular_contained_file(root: Path, candidate: Path) -> bool:
    try:
        root_absolute = Path(os.path.abspath(root))
        candidate_absolute = Path(os.path.abspath(candidate))
        candidate_absolute.relative_to(root_absolute)
        current = candidate_absolute
        while True:
            details = current.lstat()
            if _is_reparse(current, details):
                return False
            if current == root_absolute:
                break
            current = current.parent
        return stat.S_ISREG(candidate_absolute.lstat().st_mode)
    except (OSError, ValueError):
        return False


def _is_reparse(path: Path, details: os.stat_result) -> bool:
    return (
        path.is_symlink()
        or bool(getattr(os.path, "isjunction", lambda _path: False)(path))
        or bool(getattr(details, "st_file_attributes", 0) & 0x400)
    )
