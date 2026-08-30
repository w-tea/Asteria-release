from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from app.services.model_dependencies import ModelDependencyStatus
from app.services.model_profiles import ModelProfile
from app.services.model_selections import ModelSelection
from app.services.runtime_core import ComfyRuntimeCoreStatus
from app.services.runtime_dependencies import RuntimeDependencyStatus
from app.services.windows_hardware import CudaRuntimeStatus, WindowsHardwareStatus


StartupPhase = Literal["product_paths", "hardware", "runtime_core", "runtime_packages", "model_assets"]
StartupTarget = Literal["app", "runtime", "generation"]
StartupSeverity = Literal["blocker", "warning", "info"]
StartupMode = Literal["development", "packaged"]


@dataclass(frozen=True)
class StartupReadinessRequest:
    target_mode: StartupMode
    profile_id: str
    enabled_feature_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeFeatureRequirement:
    feature_id: str
    dependency_ids: tuple[str, ...]
    mode: Literal["all", "any", "core"] = "all"


@dataclass(frozen=True)
class StartupManifestRequirement:
    id: str
    phase: StartupPhase


DEFAULT_REQUIRED_MANIFESTS = (
    StartupManifestRequirement("runtime_dependencies", "runtime_packages"),
    StartupManifestRequirement("model_dependencies", "model_assets"),
    StartupManifestRequirement("model_profiles", "model_assets"),
    StartupManifestRequirement("model_selections", "model_assets"),
)


@dataclass(frozen=True)
class StartupReadinessPolicy:
    required_manifests: tuple[StartupManifestRequirement, ...] = DEFAULT_REQUIRED_MANIFESTS
    runtime_features: tuple[RuntimeFeatureRequirement, ...] = ()


@dataclass(frozen=True)
class StartupManifestHealth:
    id: str
    phase: StartupPhase
    valid: bool
    error_code: str | None = None


@dataclass(frozen=True)
class StartupPathRequirement:
    id: str
    label_key: str
    path: Path
    kind: Literal["directory", "file"] = "directory"


@dataclass(frozen=True)
class PathProbeResult:
    path: Path
    exists: bool
    is_directory: bool
    is_file: bool
    readable: bool


@dataclass(frozen=True)
class StartupReadinessItem:
    code: str
    summary_key: str
    summary_params: tuple[tuple[str, str], ...]
    phase: StartupPhase
    severity: StartupSeverity
    blocks: tuple[StartupTarget, ...]
    action: str | None
    path: str | None = None
    dependency_id: str | None = None
    profile_id: str | None = None
    feature_ids: tuple[str, ...] = ()
    auto_install: bool | None = None
    requires_license_confirmation: bool = False
    download_size_bytes: int | None = None


@dataclass(frozen=True)
class StartupReadinessSummary:
    app_ready: bool
    runtime_ready: bool
    generation_ready: bool
    blocking_count: int
    warning_count: int
    action_count: int
    required_license_dependency_ids: tuple[str, ...]
    optional_license_dependency_ids: tuple[str, ...]
    required_download_bytes: int
    optional_download_bytes: int


@dataclass(frozen=True)
class StartupReadinessReport:
    request: StartupReadinessRequest
    items: tuple[StartupReadinessItem, ...]
    summary: StartupReadinessSummary


class StartupReadinessService:
    """Project startup readiness without mutating paths, dependencies, or processes."""

    def __init__(
        self,
        *,
        product_paths: Iterable[StartupPathRequirement],
        runtime_python_path: Path,
        core_status_provider: Callable[[], ComfyRuntimeCoreStatus],
        runtime_dependency_status_provider: Callable[[], list[RuntimeDependencyStatus]],
        core_feature_status_provider: Callable[[str], bool],
        model_dependency_status_provider: Callable[[], list[ModelDependencyStatus]],
        profile_provider: Callable[[str], ModelProfile | None],
        selection_provider: Callable[[str], ModelSelection | None],
        manifest_health_provider: Callable[[], list[StartupManifestHealth]],
        policy: StartupReadinessPolicy,
        path_probe: Callable[[Path], PathProbeResult] | None = None,
        hardware_status_provider: Callable[[], WindowsHardwareStatus | None] | None = None,
        cuda_status_provider: Callable[[], CudaRuntimeStatus | None] | None = None,
    ) -> None:
        self.product_paths = tuple(product_paths)
        self.runtime_python_path = runtime_python_path
        self.core_status_provider = core_status_provider
        self.runtime_dependency_status_provider = runtime_dependency_status_provider
        self.core_feature_status_provider = core_feature_status_provider
        self.model_dependency_status_provider = model_dependency_status_provider
        self.profile_provider = profile_provider
        self.selection_provider = selection_provider
        self.manifest_health_provider = manifest_health_provider
        self.policy = policy
        self.path_probe = path_probe or probe_path_read_only
        self.hardware_status_provider = hardware_status_provider or (lambda: None)
        self.cuda_status_provider = cuda_status_provider or (lambda: None)

    def inspect(self, request: StartupReadinessRequest) -> StartupReadinessReport:
        normalized_request = StartupReadinessRequest(
            target_mode=request.target_mode,
            profile_id=request.profile_id.strip(),
            enabled_feature_ids=tuple(
                dict.fromkeys(value.strip() for value in request.enabled_feature_ids if value.strip())
            ),
        )
        items: list[StartupReadinessItem] = []
        items.extend(self._inspect_request(normalized_request))
        items.extend(self._inspect_product_paths())
        items.extend(self._inspect_hardware(normalized_request))
        items.extend(self._inspect_runtime_core())
        manifest_health = self.manifest_health_provider()
        items.extend(self._manifest_items(manifest_health, "runtime_packages"))
        items.extend(self._inspect_runtime_dependencies(normalized_request.enabled_feature_ids))
        items.extend(self._inspect_cuda_runtime(normalized_request))
        items.extend(self._manifest_items(manifest_health, "model_assets"))
        items.extend(self._inspect_model_dependencies(normalized_request.profile_id))
        item_tuple = tuple(items)
        return StartupReadinessReport(
            request=normalized_request,
            items=item_tuple,
            summary=self._summarize(item_tuple),
        )

    def _inspect_hardware(self, request: StartupReadinessRequest) -> list[StartupReadinessItem]:
        if request.target_mode != "packaged":
            return []
        status = self.hardware_status_provider()
        if status is None:
            return []
        params: list[tuple[str, str]] = [("minimum_driver", status.minimum_driver)]
        if status.driver_version is not None:
            params.append(("current_driver", status.driver_version))
        summary_keys = {
            "startup_hardware_ready": "startup.hardware.ready",
            "startup_hardware_os_unsupported": "startup.hardware.osUnsupported",
            "startup_hardware_arch_unsupported": "startup.hardware.archUnsupported",
            "startup_hardware_probe_missing": "startup.hardware.probeMissing",
            "startup_hardware_gpu_missing": "startup.hardware.gpuMissing",
            "startup_hardware_nvidia_required": "startup.hardware.nvidiaRequired",
            "startup_hardware_probe_invalid": "startup.hardware.probeFailed",
            "startup_hardware_probe_failed": "startup.hardware.probeFailed",
            "startup_hardware_probe_timeout": "startup.hardware.probeFailed",
            "startup_hardware_driver_outdated": "startup.hardware.driverOutdated",
            "startup_hardware_manifest_invalid": "startup.hardware.manifestInvalid",
        }
        return [
            self._item(
                code=status.code,
                summary_key=summary_keys.get(status.code, "startup.hardware.probeFailed"),
                summary_params=tuple(params),
                phase="hardware",
                severity="info" if status.supported else "blocker",
                blocks=() if status.supported else ("runtime", "generation"),
                action=None if status.supported else "review_nvidia_requirement",
            )
        ]

    def _inspect_cuda_runtime(self, request: StartupReadinessRequest) -> list[StartupReadinessItem]:
        if request.target_mode != "packaged":
            return []
        status = self.cuda_status_provider()
        if status is None:
            return []
        summary_keys = {
            "startup_cuda_ready": "startup.cuda.ready",
            "startup_cuda_torch_missing": "startup.cuda.torchMissing",
            "startup_cuda_unavailable": "startup.cuda.unavailable",
            "startup_cuda_allocation_failed": "startup.cuda.allocationFailed",
        }
        return [
            self._item(
                code=status.code,
                summary_key=summary_keys.get(status.code, "startup.cuda.unavailable"),
                summary_params=(),
                phase="runtime_packages",
                severity="info" if status.supported else "blocker",
                blocks=() if status.supported else ("runtime", "generation"),
                action=None if status.supported else "repair_cuda_runtime",
            )
        ]

    def _inspect_request(self, request: StartupReadinessRequest) -> list[StartupReadinessItem]:
        if request.target_mode in {"development", "packaged"}:
            return []
        return [
            self._item(
                code="startup_target_mode_unknown",
                summary_key="startup.targetMode.unknown",
                summary_params=(("target_mode", str(request.target_mode)),),
                phase="product_paths",
                severity="blocker",
                blocks=("app", "runtime", "generation"),
                action="select_startup_target_mode",
            )
        ]

    def _inspect_product_paths(self) -> list[StartupReadinessItem]:
        return [self._path_item(requirement) for requirement in self.product_paths]

    def _path_item(self, requirement: StartupPathRequirement) -> StartupReadinessItem:
        result = self.path_probe(requirement.path)
        params = (("label_key", requirement.label_key),)
        common = {"phase": "product_paths", "path": str(requirement.path), "summary_params": params}
        if not result.readable:
            return self._item(
                code="product_path_unreadable",
                summary_key="startup.path.unreadable",
                severity="blocker",
                blocks=("app", "runtime", "generation"),
                action="configure_product_path",
                **common,
            )
        if not result.exists:
            return self._item(
                code="product_path_missing",
                summary_key="startup.path.missing",
                severity="blocker",
                blocks=("app", "runtime", "generation"),
                action="create_product_path",
                **common,
            )
        correct_kind = result.is_directory if requirement.kind == "directory" else result.is_file
        if not correct_kind:
            return self._item(
                code="product_path_invalid_type",
                summary_key="startup.path.invalidType",
                severity="blocker",
                blocks=("app", "runtime", "generation"),
                action="configure_product_path",
                **common,
            )
        return self._item(
            code="product_path_ready",
            summary_key="startup.path.ready",
            severity="info",
            blocks=(),
            action=None,
            **common,
        )

    def _inspect_runtime_core(self) -> list[StartupReadinessItem]:
        core = self.core_status_provider()
        if core.status == "installed":
            core_item = self._item(
                code="runtime_core_ready",
                summary_key="startup.runtime.coreReady",
                phase="runtime_core",
                severity="info",
                blocks=(),
                action=None,
                path=str(core.path),
            )
        elif core.status == "missing":
            core_item = self._item(
                code="runtime_core_missing",
                summary_key="startup.runtime.coreMissing",
                phase="runtime_core",
                severity="blocker",
                blocks=("runtime", "generation"),
                action="install_runtime_core",
                path=str(core.path),
            )
        else:
            core_item = self._item(
                code="runtime_core_unavailable",
                summary_key="startup.runtime.coreUnavailable",
                phase="runtime_core",
                severity="blocker",
                blocks=("runtime", "generation"),
                action="repair_runtime_core",
                path=str(core.path),
            )

        python_probe = self.path_probe(self.runtime_python_path)
        common = {"phase": "runtime_core", "path": str(self.runtime_python_path)}
        if not python_probe.readable:
            python_item = self._item(
                code="runtime_python_unreadable",
                summary_key="startup.runtime.pythonUnreadable",
                severity="blocker",
                blocks=("runtime", "generation"),
                action="configure_runtime_python",
                **common,
            )
        elif not python_probe.exists:
            python_item = self._item(
                code="runtime_python_missing",
                summary_key="startup.runtime.pythonMissing",
                severity="blocker",
                blocks=("runtime", "generation"),
                action="configure_runtime_python",
                **common,
            )
        elif not python_probe.is_file:
            python_item = self._item(
                code="runtime_python_invalid_type",
                summary_key="startup.runtime.pythonInvalidType",
                severity="blocker",
                blocks=("runtime", "generation"),
                action="configure_runtime_python",
                **common,
            )
        else:
            python_item = self._item(
                code="runtime_python_ready",
                summary_key="startup.runtime.pythonReady",
                severity="info",
                blocks=(),
                action=None,
                **common,
            )
        return [core_item, python_item]

    def _manifest_items(
        self,
        health_items: list[StartupManifestHealth],
        phase: StartupPhase,
    ) -> list[StartupReadinessItem]:
        health_by_identity = {(health.id, health.phase): health for health in health_items}
        items: list[StartupReadinessItem] = []
        for requirement in self.policy.required_manifests:
            if requirement.phase != phase:
                continue
            health = health_by_identity.get((requirement.id, requirement.phase))
            if health is None:
                items.append(
                    self._item(
                        code="startup_manifest_missing",
                        summary_key="startup.manifest.missing",
                        summary_params=(("manifest_id", requirement.id),),
                        phase=phase,
                        severity="blocker",
                        blocks=("generation",),
                        action="repair_startup_manifest",
                    )
                )
            elif not health.valid:
                params = [("manifest_id", health.id)]
                if health.error_code:
                    params.append(("error_code", health.error_code))
                items.append(
                    self._item(
                        code="startup_manifest_invalid",
                        summary_key="startup.manifest.invalid",
                        summary_params=tuple(params),
                        phase=phase,
                        severity="blocker",
                        blocks=("generation",),
                        action="repair_startup_manifest",
                    )
                )
        return items

    def _inspect_runtime_dependencies(self, feature_ids: tuple[str, ...]) -> list[StartupReadinessItem]:
        statuses = self.runtime_dependency_status_provider()
        status_by_id = {status.id: status for status in statuses}
        requirement_by_feature = {requirement.feature_id: requirement for requirement in self.policy.runtime_features}
        required_all: set[str] = set()
        synthetic: list[StartupReadinessItem] = []
        for feature_id in feature_ids:
            requirement = requirement_by_feature.get(feature_id)
            if requirement is None:
                synthetic.append(
                    self._item(
                        code="runtime_feature_unknown",
                        summary_key="startup.runtime.featureUnknown",
                        summary_params=(("feature_id", feature_id),),
                        phase="runtime_packages",
                        severity="blocker",
                        blocks=("generation",),
                        action="configure_runtime_features",
                        feature_ids=(feature_id,),
                    )
                )
                continue
            if requirement.mode == "core":
                if not self.core_feature_status_provider(feature_id):
                    synthetic.append(
                        self._item(
                            code="runtime_core_feature_unavailable",
                            summary_key="startup.runtime.featureRequirementUnmet",
                            summary_params=(("feature_id", feature_id),),
                            phase="runtime_packages",
                            severity="blocker",
                            blocks=("generation",),
                            action="repair_runtime_core",
                            feature_ids=(feature_id,),
                        )
                    )
                continue
            if not requirement.dependency_ids:
                synthetic.append(
                    self._item(
                        code="runtime_policy_empty",
                        summary_key="startup.runtime.policyEmpty",
                        summary_params=(("feature_id", feature_id),),
                        phase="runtime_packages",
                        severity="blocker",
                        blocks=("generation",),
                        action="repair_runtime_dependency_manifest",
                        feature_ids=(feature_id,),
                    )
                )
                continue
            missing_policy_ids = [
                dependency_id
                for dependency_id in requirement.dependency_ids
                if dependency_id not in status_by_id
            ]
            synthetic.extend(
                self._missing_policy_dependency(feature_id, dependency_id)
                for dependency_id in missing_policy_ids
            )
            if missing_policy_ids:
                continue
            if requirement.mode == "all":
                required_all.update(requirement.dependency_ids)
            elif not any(
                status_by_id.get(dependency_id) is not None
                and status_by_id[dependency_id].status == "installed"
                for dependency_id in requirement.dependency_ids
            ):
                synthetic.append(
                    self._item(
                        code="runtime_feature_requirement_unmet",
                        summary_key="startup.runtime.featureRequirementUnmet",
                        summary_params=(("feature_id", feature_id),),
                        phase="runtime_packages",
                        severity="blocker",
                        blocks=("generation",),
                        action="install_runtime_feature",
                        feature_ids=(feature_id,),
                    )
                )

        items = list(synthetic)
        for status in statuses:
            blocks: tuple[StartupTarget, ...] = ("generation",) if status.id in required_all and status.status != "installed" else ()
            if status.status == "installed":
                code = "runtime_dependency_ready"
                severity: StartupSeverity = "info"
                action = None
                summary_key = "startup.runtime.dependencyReady"
            elif status.status == "post_install_pending":
                code = "runtime_dependency_post_install_pending"
                severity = "blocker" if blocks else "warning"
                action = "repair_runtime_dependency"
                summary_key = "startup.runtime.dependencyPostInstallPending"
            else:
                code = "runtime_dependency_missing"
                severity = "blocker" if blocks else "warning"
                action = "install_runtime_dependency"
                summary_key = "startup.runtime.dependencyMissing"
            items.append(
                self._item(
                    code=code,
                    summary_key=summary_key,
                    summary_params=(("dependency_id", status.id),),
                    phase="runtime_packages",
                    severity=severity,
                    blocks=blocks,
                    action=action,
                    path=str(status.path),
                    dependency_id=status.id,
                    feature_ids=tuple(status.features),
                    auto_install=status.auto_install,
                )
            )
        return items

    def _missing_policy_dependency(self, feature_id: str, dependency_id: str) -> StartupReadinessItem:
        return self._item(
            code="runtime_policy_dependency_unknown",
            summary_key="startup.runtime.policyDependencyUnknown",
            summary_params=(("feature_id", feature_id), ("dependency_id", dependency_id)),
            phase="runtime_packages",
            severity="blocker",
            blocks=("generation",),
            action="repair_runtime_dependency_manifest",
            dependency_id=dependency_id,
            feature_ids=(feature_id,),
        )

    def _inspect_model_dependencies(self, profile_id: str) -> list[StartupReadinessItem]:
        profile = self.profile_provider(profile_id)
        if profile is None:
            return [
                self._item(
                    code="model_profile_unknown",
                    summary_key="startup.models.profileUnknown",
                    summary_params=(("profile_id", profile_id),),
                    phase="model_assets",
                    severity="blocker",
                    blocks=("generation",),
                    action="select_model_profile",
                    profile_id=profile_id,
                )
            ]

        statuses = self.model_dependency_status_provider()
        selected_ids = set(profile.dependency_ids)
        relevant = [status for status in statuses if status.id in selected_ids or status.profile == "shared"]
        candidates_by_slot = {
            status.slot_id: status
            for status in relevant
            if status.profile == profile_id and status.slot_id
        }
        selection = self.selection_provider(profile_id)
        selected_slots = selection.slots if selection is not None and selection.profile == profile_id else {}
        handled_dependencies: set[str] = set()
        items: list[StartupReadinessItem] = []
        for slot in (slot for slot in profile.model_slots if slot.required):
            candidate = candidates_by_slot.get(slot.id)
            if candidate is not None:
                handled_dependencies.add(candidate.id)
            items.append(
                self._required_slot_item(
                    profile_id=profile_id,
                    slot_id=slot.id,
                    selected_value=selected_slots.get(slot.id, ""),
                    candidate=candidate,
                )
            )

        for status in relevant:
            if status.id in handled_dependencies:
                continue
            items.append(self._dependency_item(status, profile_id=profile_id, required=False))
        return items

    def _required_slot_item(
        self,
        *,
        profile_id: str,
        slot_id: str,
        selected_value: str,
        candidate: ModelDependencyStatus | None,
    ) -> StartupReadinessItem:
        selected_path = Path(selected_value) if selected_value else None
        if selected_path is not None:
            selected_probe = self.path_probe(selected_path)
            if (
                candidate is not None
                and normalized_path_identity(selected_path) == normalized_path_identity(candidate.path)
                and candidate.status != "installed"
            ):
                return self._dependency_item(candidate, profile_id=profile_id, required=True)
            if selected_probe.exists and selected_probe.is_file and selected_probe.readable:
                return self._item(
                    code="model_slot_ready",
                    summary_key="startup.models.slotReady",
                    summary_params=(("slot_id", slot_id),),
                    phase="model_assets",
                    severity="info",
                    blocks=(),
                    action=None,
                    path=str(selected_path),
                    dependency_id=candidate.id if candidate is not None else None,
                    profile_id=profile_id,
                )
            return self._item(
                code="model_slot_unreadable" if not selected_probe.readable else "model_slot_missing",
                summary_key=(
                    "startup.models.slotUnreadable" if not selected_probe.readable else "startup.models.slotMissing"
                ),
                summary_params=(("slot_id", slot_id),),
                phase="model_assets",
                severity="blocker",
                blocks=("generation",),
                action="select_model_asset",
                path=str(selected_path),
                dependency_id=candidate.id if candidate is not None else None,
                profile_id=profile_id,
            )
        if candidate is None:
            return self._item(
                code="model_required_slot_unmanaged",
                summary_key="startup.models.requiredSlotUnmanaged",
                summary_params=(("slot_id", slot_id),),
                phase="model_assets",
                severity="blocker",
                blocks=("generation",),
                action="configure_model_dependency",
                profile_id=profile_id,
            )
        if candidate.status == "installed":
            return self._item(
                code="model_slot_unselected",
                summary_key="startup.models.slotUnselected",
                summary_params=(("slot_id", slot_id),),
                phase="model_assets",
                severity="blocker",
                blocks=("generation",),
                action="select_model_asset",
                path=str(candidate.path),
                dependency_id=candidate.id,
                profile_id=profile_id,
            )
        return self._dependency_item(candidate, profile_id=profile_id, required=True)

    def _dependency_item(
        self,
        status: ModelDependencyStatus,
        *,
        profile_id: str,
        required: bool,
    ) -> StartupReadinessItem:
        blocks: tuple[StartupTarget, ...] = ("generation",) if required and status.status != "installed" else ()
        severity: StartupSeverity = "blocker" if blocks else "warning"
        if status.status == "installed":
            code = "model_dependency_ready"
            severity = "info"
            action = None
            summary_key = "startup.models.dependencyReady"
        elif status.status == "missing":
            code = "model_dependency_missing"
            summary_key = "startup.models.dependencyMissing"
            if status.provider == "manual" or not status.auto_install:
                action = "manual_place_model_dependency"
            elif status.requires_license_confirmation:
                action = "confirm_model_license_then_install"
            else:
                action = "install_model_dependency"
        else:
            code = f"model_dependency_{status.status}"
            summary_key = "startup.models.dependencyNeedsRepair"
            action = "repair_model_dependency"
        return self._item(
            code=code,
            summary_key=summary_key,
            summary_params=(("dependency_id", status.id),),
            phase="model_assets",
            severity=severity,
            blocks=blocks,
            action=action,
            path=str(status.path),
            dependency_id=status.id,
            profile_id=profile_id,
            auto_install=status.auto_install,
            requires_license_confirmation=status.requires_license_confirmation and status.status != "installed",
            download_size_bytes=(
                status.size_bytes
                if status.status != "installed" and status.auto_install and status.provider != "manual"
                else None
            ),
        )

    def _summarize(self, items: tuple[StartupReadinessItem, ...]) -> StartupReadinessSummary:
        blocked_targets = {target for item in items for target in item.blocks}
        required_license_ids = tuple(
            item.dependency_id
            for item in items
            if item.requires_license_confirmation and item.dependency_id is not None and bool(item.blocks)
        )
        optional_license_ids = tuple(
            item.dependency_id
            for item in items
            if item.requires_license_confirmation and item.dependency_id is not None and not item.blocks
        )
        return StartupReadinessSummary(
            app_ready="app" not in blocked_targets,
            runtime_ready="app" not in blocked_targets and "runtime" not in blocked_targets,
            generation_ready=not blocked_targets,
            blocking_count=sum(bool(item.blocks) for item in items),
            warning_count=sum(item.severity == "warning" for item in items),
            action_count=sum(item.action is not None for item in items),
            required_license_dependency_ids=tuple(dict.fromkeys(required_license_ids)),
            optional_license_dependency_ids=tuple(dict.fromkeys(optional_license_ids)),
            required_download_bytes=sum(
                item.download_size_bytes or 0 for item in items if item.blocks
            ),
            optional_download_bytes=sum(
                item.download_size_bytes or 0 for item in items if not item.blocks
            ),
        )

    def _item(
        self,
        *,
        code: str,
        summary_key: str,
        phase: StartupPhase,
        severity: StartupSeverity,
        blocks: tuple[StartupTarget, ...],
        action: str | None,
        summary_params: tuple[tuple[str, str], ...] = (),
        path: str | None = None,
        dependency_id: str | None = None,
        profile_id: str | None = None,
        feature_ids: tuple[str, ...] = (),
        auto_install: bool | None = None,
        requires_license_confirmation: bool = False,
        download_size_bytes: int | None = None,
    ) -> StartupReadinessItem:
        return StartupReadinessItem(
            code=code,
            summary_key=summary_key,
            summary_params=summary_params,
            phase=phase,
            severity=severity,
            blocks=blocks,
            action=action,
            path=path,
            dependency_id=dependency_id,
            profile_id=profile_id,
            feature_ids=feature_ids,
            auto_install=auto_install,
            requires_license_confirmation=requires_license_confirmation,
            download_size_bytes=download_size_bytes,
        )


def probe_path_read_only(path: Path) -> PathProbeResult:
    """Inspect existence, type, and readability without a write probe."""

    try:
        exists = path.exists()
        if not exists:
            return PathProbeResult(path, False, False, False, True)
        return PathProbeResult(
            path=path,
            exists=True,
            is_directory=path.is_dir(),
            is_file=path.is_file(),
            readable=os.access(path, os.R_OK),
        )
    except OSError:
        return PathProbeResult(path, False, False, False, False)


def normalized_path_identity(path: Path) -> str:
    """Return a side-effect-free, Windows-aware identity for path comparison."""

    try:
        absolute = path.resolve(strict=False)
    except (OSError, RuntimeError):
        absolute = Path(os.path.abspath(os.path.normpath(str(path))))
    return os.path.normcase(os.path.normpath(str(absolute)))
