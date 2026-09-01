from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, NoReturn

from app.services.model_dependencies import ModelDependencyRepository
from app.services.model_dependencies import ModelDependencyInstaller
from app.services.model_selections import ModelSelectionRepository
from app.services.initial_data import InitialDataMaterializer
from app.services.managed_artifacts import (
    ManagedArtifactPathTooLong,
    ManagedArtifactRollbackIncomplete,
    ManagedArtifactService,
    ManagedArtifactSpec,
)
from app.services.package_environment import PackageEnvironmentManager
from app.services.release_manifest import ReleaseManifestRepository
from app.services.runtime_core import ComfyRuntimeCoreManager
from app.services.runtime_dependencies import RuntimeDependencyRepository
from app.services.runtime_dependencies import RuntimeDependencyInstaller


_STARTER_PACK_ORDER = ("sdxl", "anima")


class ManagedSetupError(Exception):
    """Base class for setup control-flow failures safe to map at the boundary."""


class ManagedSetupConsentRequired(ManagedSetupError):
    pass


class ManagedSetupCancelled(ManagedSetupError):
    def __init__(self, completed_component_ids: tuple[str, ...]) -> None:
        super().__init__("managed setup was cancelled")
        self.completed_component_ids = completed_component_ids


@dataclass(frozen=True)
class ManagedSetupComponent:
    id: str
    label: str
    category: str
    size_bytes: int
    optional: bool = False
    requires_license_confirmation: bool = False


@dataclass(frozen=True)
class ManagedSetupPlan:
    components: tuple[ManagedSetupComponent, ...]
    selected_starter_packs: tuple[str, ...]

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(component.id for component in self.components)

    @property
    def optional_component_ids(self) -> tuple[str, ...]:
        return tuple(component.id for component in self.components if component.optional)

    @property
    def total_size_bytes(self) -> int:
        return sum(component.size_bytes for component in self.components)


@dataclass(frozen=True)
class ManagedSetupEvent:
    code: str
    component_id: str | None = None
    error_code: str | None = None
    message_en: str | None = None
    message_ja: str | None = None

    def to_json(self) -> str:
        payload: dict[str, str] = {"code": self.code}
        if self.component_id is not None:
            payload["component_id"] = self.component_id
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.message_en is not None:
            payload["message_en"] = self.message_en
        if self.message_ja is not None:
            payload["message_ja"] = self.message_ja
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ManagedSetupResult:
    completed_component_ids: tuple[str, ...]
    skipped_component_ids: tuple[str, ...]
    failed_component_ids: tuple[str, ...]
    events: tuple[ManagedSetupEvent, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failed_component_ids


def build_setup_plan(
    app_root: Path,
    *,
    selected_starter_packs: Iterable[str] = (),
) -> ManagedSetupPlan:
    root = app_root.resolve()
    selected = set(selected_starter_packs)
    unknown = selected.difference(_STARTER_PACK_ORDER)
    if unknown:
        raise ValueError("managed setup starter pack is unsupported")
    ordered_starters = tuple(pack for pack in _STARTER_PACK_ORDER if pack in selected)

    release = ReleaseManifestRepository(root / "release-manifest.yaml").load()
    runtime_dependencies = RuntimeDependencyRepository(
        root / "runtime-dependencies.yaml",
        root / "runtime" / "ComfyUI",
    ).list()
    model_dependencies = ModelDependencyRepository(
        root / "model-dependencies.yaml",
        root / "models",
    ).list()

    components: list[ManagedSetupComponent] = []
    for artifact in release.artifacts:
        if artifact.id in {"bootstrap-uv", "python-runtime"}:
            continue
        components.append(
            ManagedSetupComponent(
                id=artifact.id,
                label=artifact.label,
                category="release-artifact",
                size_bytes=artifact.size_bytes,
            )
        )

    components.extend(
        (
            ManagedSetupComponent(
                id="asteria-backend-environment",
                label="Asteria backend packages",
                category="package-environment",
                size_bytes=0,
            ),
            ManagedSetupComponent(
                id="pytorch-cu128",
                label="PyTorch CUDA packages",
                category="package-environment",
                size_bytes=0,
            ),
        )
    )
    for package_set in release.package_sets:
        if package_set.id == "pytorch-cu128":
            continue
        components.append(
            ManagedSetupComponent(
                id=package_set.id,
                label=package_set.id,
                category="package-environment",
                size_bytes=0,
            )
        )
    components.append(
        ManagedSetupComponent(
            id="comfy-python-environment",
            label="ComfyUI Python packages",
            category="package-environment",
            size_bytes=0,
        )
    )

    for dependency in runtime_dependencies:
        if dependency.release_class != "mandatory-feature" or not dependency.auto_install:
            continue
        components.append(
            ManagedSetupComponent(
                id=dependency.id,
                label=dependency.label,
                category="runtime-dependency",
                size_bytes=dependency.size_bytes or 0,
            )
        )

    mandatory_models = [
        dependency
        for dependency in model_dependencies
        if dependency.release_class == "mandatory-feature" and dependency.auto_install
    ]
    selected_starters = [
        dependency
        for dependency in model_dependencies
        if dependency.release_class == "optional-starter" and dependency.starter_pack in selected
    ]
    for dependency in (*mandatory_models, *selected_starters):
        is_selected_starter = dependency.release_class == "optional-starter"
        components.append(
            ManagedSetupComponent(
                id=dependency.id,
                label=dependency.label,
                category="model-dependency",
                size_bytes=dependency.size_bytes or 0,
                optional=is_selected_starter,
                requires_license_confirmation=dependency.requires_license_confirmation,
            )
        )

    components.append(
        ManagedSetupComponent(
            id="initial-data",
            label="Asteria initial data",
            category="initial-data",
            size_bytes=0,
        )
    )

    return ManagedSetupPlan(
        components=tuple(components),
        selected_starter_packs=ordered_starters,
    )


class ManagedSetupEngine:
    def __init__(
        self,
        *,
        is_complete: Callable[[ManagedSetupComponent], bool],
        acquire: Callable[[ManagedSetupComponent], None],
        is_cancelled: Callable[[], bool] | None = None,
        event_sink: Callable[[ManagedSetupEvent], None] | None = None,
    ) -> None:
        self._is_complete = is_complete
        self._acquire = acquire
        self._is_cancelled = is_cancelled or (lambda: False)
        self._event_sink = event_sink

    def _append_event(self, events: list[ManagedSetupEvent], event: ManagedSetupEvent) -> None:
        events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def run(self, plan: ManagedSetupPlan, *, download_confirmed: bool) -> ManagedSetupResult:
        if not download_confirmed:
            raise ManagedSetupConsentRequired("managed setup download consent is required")

        completed: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        events: list[ManagedSetupEvent] = []
        self._append_event(events, ManagedSetupEvent("setup_started"))
        for component in plan.components:
            if self._is_cancelled():
                self._append_event(events, ManagedSetupEvent("setup_cancelled"))
                raise ManagedSetupCancelled(tuple(completed))
            if self._is_complete(component):
                skipped.append(component.id)
                self._append_event(events, ManagedSetupEvent("setup_component_skipped", component.id))
                continue
            self._append_event(events, ManagedSetupEvent("setup_component_started", component.id))
            try:
                self._acquire(component)
            except ManagedArtifactRollbackIncomplete:
                failed.append(component.id)
                self._append_event(
                    events,
                    ManagedSetupEvent(
                        "setup_component_failed",
                        component.id,
                        error_code="setup_rollback_incomplete",
                        message_en=(
                            "The previous managed component backup was kept because Asteria could not restore it. "
                            "Close other programs using Asteria, then run Repair-Asteria.cmd again. "
                            "If the problem continues, keep the logs and contact support."
                        ),
                        message_ja=(
                            "以前の管理対象コンポーネントのバックアップは削除せず保持しています。"
                            "Asteriaを使用しているほかのプログラムを閉じてから、"
                            "Repair-Asteria.cmdをもう一度実行してください。"
                            "解決しない場合は、ログを残したままサポートへ連絡してください。"
                        ),
                    ),
                )
                continue
            except ManagedArtifactPathTooLong:
                failed.append(component.id)
                self._append_event(
                    events,
                    ManagedSetupEvent(
                        "setup_component_failed",
                        component.id,
                        error_code="setup_path_too_long",
                        message_en=(
                            "The installation path is too long for this component. "
                            "Move the Asteria folder to a shorter location such as D:\\Asteria, "
                            "then run Repair-Asteria.cmd."
                        ),
                        message_ja=(
                            "このコンポーネントを展開するには、Asteriaの配置先が長すぎます。"
                            "Asteriaを短いパスへ移動してください（例: D:\\Asteria）。"
                            "移動後、Repair-Asteria.cmdを実行してください。"
                        ),
                    ),
                )
                continue
            except Exception:
                failed.append(component.id)
                self._append_event(events, ManagedSetupEvent("setup_component_failed", component.id))
                continue
            completed.append(component.id)
            self._append_event(events, ManagedSetupEvent("setup_component_completed", component.id))
        self._append_event(events, ManagedSetupEvent("setup_completed" if not failed else "setup_incomplete"))
        return ManagedSetupResult(tuple(completed), tuple(skipped), tuple(failed), tuple(events))


class PortableSetupRuntime:
    def __init__(
        self,
        app_root: Path,
        *,
        mode: Literal["install", "repair"] = "install",
        accepted_starter_licenses: Iterable[str] = (),
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.app_root = app_root.resolve()
        self.mode = mode
        self.release = ReleaseManifestRepository(self.app_root / "release-manifest.yaml").load()
        self.artifact_service = ManagedArtifactService(
            self.app_root / "temp",
            progress_callback=progress_callback,
        )
        self.python_path = self.app_root / "runtime" / "python" / "python.exe"
        self.package_manager = PackageEnvironmentManager(self.app_root)
        self.runtime_repository = RuntimeDependencyRepository(
            self.app_root / "runtime-dependencies.yaml",
            self.app_root / "runtime" / "ComfyUI",
            python_path=self.python_path,
        )
        self.runtime_installer = RuntimeDependencyInstaller(
            self.runtime_repository,
            python_path=self.python_path,
            managed_artifact_service=self.artifact_service,
            constraints_path=self.app_root / "bootstrap" / "locks" / "comfyui-v0.28.0.lock",
        )
        self.model_repository = ModelDependencyRepository(
            self.app_root / "model-dependencies.yaml",
            self.app_root / "models",
        )
        self.model_installer = ModelDependencyInstaller(
            self.model_repository,
            managed_artifact_service=self.artifact_service,
        )
        self.model_selections = ModelSelectionRepository(
            self.app_root / "data" / "model-selections.json"
        )
        self.accepted_starter_licenses = frozenset(accepted_starter_licenses)
        self.initial_data = InitialDataMaterializer(
            self.app_root / "defaults",
            self.app_root / "data",
        )
        self.release_specs = {
            artifact.id: ManagedArtifactSpec(
                manifest_version=self.release.version,
                component_id=artifact.id,
                source_url=artifact.url,
                destination=self.app_root / artifact.relative_path,
                expected_size=artifact.size_bytes,
                checksum_sha256=artifact.checksum_sha256,
                kind="zip",
                strip_single_root=True,
            )
            for artifact in self.release.artifacts
            if artifact.id not in {"bootstrap-uv", "python-runtime"}
        }

    def is_complete(self, component: ManagedSetupComponent) -> bool:
        if component.category == "release-artifact":
            spec = self.release_specs[component.id]
            return self.artifact_service.is_installed(spec, require_post_install=True)
        if component.category == "package-environment":
            environment_id, python_path, lock_path, index_url, no_dependencies = self._package_environment(component.id)
            return self.package_manager.is_installed(
                environment_id,
                python_path,
                lock_path,
                index_url=index_url,
                no_dependencies=no_dependencies,
            )
        if component.category == "runtime-dependency":
            dependency = self.runtime_repository.get(component.id)
            if self.mode == "repair" and self.runtime_repository._requires_post_install(dependency):
                return False
            return self.runtime_repository._status_for(dependency).status == "installed"
        if component.category == "model-dependency":
            return self.model_repository._status_for(self.model_repository.get(component.id)).status == "installed"
        if component.category == "initial-data":
            return self.initial_data.is_materialized()
        return False

    def acquire(self, component: ManagedSetupComponent) -> None:
        if component.category == "release-artifact":
            spec = self.release_specs[component.id]
            if component.id == "comfyui-core":
                artifact = next(item for item in self.release.artifacts if item.id == component.id)
                ComfyRuntimeCoreManager(
                    comfy_dir=spec.destination,
                    python_path=self.python_path,
                    repo_url=artifact.url,
                    ref=artifact.revision,
                    managed_artifact_service=self.artifact_service,
                    managed_artifact_spec=spec,
                    install_python_dependencies=False,
                ).install()
            else:
                self.artifact_service.acquire(spec)
                self.artifact_service.mark_post_install_complete(spec)
            return
        if component.category == "package-environment":
            environment_id, python_path, lock_path, index_url, no_dependencies = self._package_environment(component.id)
            self.package_manager.install(
                environment_id,
                python_path,
                lock_path,
                index_url=index_url,
                no_dependencies=no_dependencies,
            )
            return
        if component.category == "runtime-dependency":
            if self.mode == "repair":
                self.runtime_installer.repair(component.id)
            else:
                self.runtime_installer.install(component.id)
            return
        if component.category == "model-dependency":
            dependency = self.model_repository.get(component.id)
            license_confirmed = (
                not dependency.requires_license_confirmation
                or dependency.starter_pack in self.accepted_starter_licenses
            )
            self.model_installer.install(component.id, license_confirmed=license_confirmed)
            return
        if component.category == "initial-data":
            self.initial_data.materialize()
            return
        raise ValueError("managed setup component category is unsupported")

    def autofill_completed_starter_packs(
        self,
        starter_pack_ids: Iterable[str],
    ) -> None:
        self.model_selections.autofill_complete_starter_packs(
            tuple(starter_pack_ids),
            self.model_repository.list_status(),
        )

    def _package_environment(self, component_id: str) -> tuple[str, Path, Path, str | None, bool]:
        if component_id == "asteria-backend-environment":
            return component_id, self.python_path, self.app_root / "app" / "backend" / "requirements-release.lock", None, False
        if component_id == "comfy-python-environment":
            return component_id, self.python_path, self.app_root / "bootstrap" / "locks" / "comfyui-v0.28.0.lock", None, False
        if component_id == "pytorch-cu128":
            package_set = next(item for item in self.release.package_sets if item.id == component_id)
            return (
                component_id,
                self.python_path,
                self.app_root / "bootstrap" / "locks" / "pytorch-cu128.lock",
                package_set.index_url,
                True,
            )
        raise ValueError("managed package environment is unsupported")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asteria managed setup handoff")
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("install", "repair"), default="install")
    parser.add_argument("--starter-pack", action="append", default=[])
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--accept-license", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _emit(code: str, **values: object) -> None:
    print(
        json.dumps({"code": code, **values}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _fail(code: str, return_code: int) -> NoReturn:
    _emit(code)
    raise SystemExit(return_code)


class _DownloadProgressEmitter:
    def __init__(self) -> None:
        self._last_bucket: dict[str, int] = {}

    def __call__(self, component_id: str, downloaded_bytes: int, total_size_bytes: int) -> None:
        percent = min(100, int(downloaded_bytes * 100 / total_size_bytes))
        bucket = percent // 5
        if downloaded_bytes not in {0, total_size_bytes} and self._last_bucket.get(component_id) == bucket:
            return
        self._last_bucket[component_id] = bucket
        _emit(
            "setup_download_progress",
            component_id=component_id,
            downloaded_bytes=downloaded_bytes,
            total_size_bytes=total_size_bytes,
            percent=percent,
        )


def main() -> int:
    args = _parser().parse_args()
    try:
        plan = build_setup_plan(args.app_root, selected_starter_packs=args.starter_pack)
    except (OSError, ValueError):
        _fail("setup_manifest_invalid", 65)
    if not args.confirm_download:
        _fail("setup_download_consent_required", 66)
    selected_licenses = set(args.accept_license)
    if "anima" in plan.selected_starter_packs and "anima" not in selected_licenses:
        _fail("setup_starter_license_required", 67)
    if args.plan_only:
        _emit(
            "setup_plan_ready",
            component_ids=list(plan.component_ids),
            total_size_bytes=plan.total_size_bytes,
            selected_starter_packs=list(plan.selected_starter_packs),
        )
        return 0

    runtime = PortableSetupRuntime(
        args.app_root,
        mode=args.mode,
        accepted_starter_licenses=selected_licenses,
        progress_callback=_DownloadProgressEmitter(),
    )
    result = ManagedSetupEngine(
        is_complete=runtime.is_complete,
        acquire=runtime.acquire,
        event_sink=lambda event: print(event.to_json(), flush=True),
    ).run(plan, download_confirmed=True)
    runtime.autofill_completed_starter_packs(plan.selected_starter_packs)
    return 0 if result.succeeded else 69


if __name__ == "__main__":
    raise SystemExit(main())
