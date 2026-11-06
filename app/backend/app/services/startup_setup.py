from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from collections.abc import Callable, Mapping
from typing import Literal
from urllib.parse import urlparse

from app.services.model_dependencies import ModelDependencyStatus
from app.services.startup_readiness import StartupPhase, StartupReadinessItem, StartupReadinessReport


StartupSetupStatus = Literal["pending", "manual", "optional", "completed"]

EXECUTABLE_OPERATIONS = frozenset(
    {
        "install_runtime_core",
        "repair_runtime_core",
        "install_runtime_dependency",
        "repair_runtime_dependency",
        "install_model_dependency",
        "confirm_model_license_then_install",
        "repair_model_dependency",
    }
)


@dataclass(frozen=True)
class StartupSetupStep:
    id: str
    phase: StartupPhase
    code: str
    operation: str | None
    dependency_id: str | None
    status: StartupSetupStatus
    required: bool
    executable: bool
    requires_license_confirmation: bool
    requires_download_confirmation: bool
    download_size_bytes: int | None


@dataclass(frozen=True)
class StartupStarterPack:
    id: str
    status: Literal["available", "partial", "installed"]
    selected_by_default: bool
    requires_license_confirmation: bool
    download_size_bytes: int
    source_urls: tuple[str, ...]
    license_url: str | None


@dataclass(frozen=True)
class StartupSetupPlan:
    plan_id: str
    target_mode: str
    profile_id: str
    enabled_feature_ids: tuple[str, ...]
    steps: tuple[StartupSetupStep, ...]
    required_license_dependency_ids: tuple[str, ...]
    optional_license_dependency_ids: tuple[str, ...]
    required_download_bytes: int
    optional_download_bytes: int
    has_unknown_required_download: bool
    has_unknown_optional_download: bool
    starter_packs: tuple[StartupStarterPack, ...]
    starter_pack_download_bytes: int


@dataclass(frozen=True)
class StartupSetupExecution:
    step_id: str
    status: Literal["completed"] = "completed"
    operation_id: str | None = None


class StartupSetupStepError(Exception):
    pass


class StartupSetupConsentError(Exception):
    pass


class StartupStarterPackSelectionError(Exception):
    pass


class StartupSetupPlanner:
    def __init__(self, *, starter_packs: tuple[StartupStarterPack, ...] = ()) -> None:
        self.starter_packs = starter_packs

    def plan(self, report: StartupReadinessReport) -> StartupSetupPlan:
        steps = tuple(self._step(report, item, index) for index, item in enumerate(report.items))
        identity = {
            "request": {
                "target_mode": report.request.target_mode,
                "profile_id": report.request.profile_id,
                "enabled_feature_ids": report.request.enabled_feature_ids,
            },
            "steps": [step.id for step in steps],
            "starter_packs": [
                {
                    "id": pack.id,
                    "status": pack.status,
                    "download_size_bytes": pack.download_size_bytes,
                    "selected_by_default": pack.selected_by_default,
                    "requires_license_confirmation": pack.requires_license_confirmation,
                    "source_urls": pack.source_urls,
                    "license_url": pack.license_url,
                }
                for pack in self.starter_packs
            ],
        }
        return StartupSetupPlan(
            plan_id=self._digest(identity),
            target_mode=report.request.target_mode,
            profile_id=report.request.profile_id,
            enabled_feature_ids=report.request.enabled_feature_ids,
            steps=steps,
            required_license_dependency_ids=report.summary.required_license_dependency_ids,
            optional_license_dependency_ids=report.summary.optional_license_dependency_ids,
            required_download_bytes=report.summary.required_download_bytes,
            optional_download_bytes=report.summary.optional_download_bytes,
            has_unknown_required_download=any(
                step.status == "pending"
                and step.requires_download_confirmation
                and step.download_size_bytes is None
                for step in steps
            ),
            has_unknown_optional_download=any(
                step.status == "optional"
                and step.requires_download_confirmation
                and step.download_size_bytes is None
                for step in steps
            ),
            starter_packs=self.starter_packs,
            starter_pack_download_bytes=sum(pack.download_size_bytes for pack in self.starter_packs),
        )

    def _step(
        self,
        report: StartupReadinessReport,
        item: StartupReadinessItem,
        index: int,
    ) -> StartupSetupStep:
        executable = item.action in EXECUTABLE_OPERATIONS
        required = bool(item.blocks)
        if item.action is None:
            status: StartupSetupStatus = "completed"
        elif not executable:
            status = "manual"
        elif required:
            status = "pending"
        else:
            status = "optional"
        identity = {
            "request": {
                "target_mode": report.request.target_mode,
                "profile_id": report.request.profile_id,
                "enabled_feature_ids": report.request.enabled_feature_ids,
            },
            "index": index,
            "phase": item.phase,
            "code": item.code,
            "operation": item.action,
            "dependency_id": item.dependency_id,
        }
        return StartupSetupStep(
            id=self._digest(identity),
            phase=item.phase,
            code=item.code,
            operation=item.action,
            dependency_id=item.dependency_id,
            status=status,
            required=required,
            executable=executable,
            requires_license_confirmation=item.requires_license_confirmation,
            requires_download_confirmation=executable and item.action is not None,
            download_size_bytes=item.download_size_bytes,
        )

    @staticmethod
    def _digest(value: object) -> str:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()[:20]


def build_starter_pack_options(
    statuses: tuple[ModelDependencyStatus, ...] | list[ModelDependencyStatus],
) -> tuple[StartupStarterPack, ...]:
    grouped: dict[str, list[ModelDependencyStatus]] = {}
    for status in statuses:
        if status.release_class == "optional-starter" and status.starter_pack:
            grouped.setdefault(status.starter_pack, []).append(status)
    packs: list[StartupStarterPack] = []
    for pack_id in ("sdxl", "anima"):
        assets = grouped.get(pack_id, [])
        if not assets:
            continue
        installed_count = sum(asset.status == "installed" for asset in assets)
        pack_status: Literal["available", "partial", "installed"]
        if installed_count == len(assets):
            pack_status = "installed"
        elif installed_count:
            pack_status = "partial"
        else:
            pack_status = "available"
        remaining = [asset for asset in assets if asset.status != "installed"]
        source_urls = tuple(
            dict.fromkeys(
                url
                for asset in assets
                for url in (asset.source_provenance_url or asset.source_url,)
                if _public_https_url(url)
            )
        )
        license_urls = tuple(
            dict.fromkeys(
                asset.license_url
                for asset in assets
                if asset.license_url and _public_https_url(asset.license_url)
            )
        )
        packs.append(
            StartupStarterPack(
                id=pack_id,
                status=pack_status,
                selected_by_default=all(asset.selected_by_default for asset in assets),
                requires_license_confirmation=any(
                    asset.requires_license_confirmation for asset in remaining
                ),
                download_size_bytes=sum(asset.size_bytes or 0 for asset in remaining),
                source_urls=source_urls,
                license_url=license_urls[0] if license_urls else None,
            )
        )
    return tuple(packs)


def _public_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


class StartupSetupExecutor:
    def __init__(
        self,
        *,
        handlers: Mapping[str, Callable[[StartupSetupStep], str | None]],
    ) -> None:
        self.handlers = dict(handlers)

    def execute(
        self,
        plan: StartupSetupPlan,
        step_id: str,
        *,
        download_confirmed: bool = False,
        accepted_license_dependency_ids: tuple[str, ...] = (),
    ) -> StartupSetupExecution:
        step = next((candidate for candidate in plan.steps if candidate.id == step_id), None)
        if step is None:
            raise StartupSetupStepError("startup setup step was not found")
        if not step.executable or step.operation is None:
            raise StartupSetupStepError("startup setup step is not executable")
        if step.requires_download_confirmation and not download_confirmed:
            raise StartupSetupConsentError("startup setup download confirmation is required")
        accepted = set(accepted_license_dependency_ids)
        if (
            step.requires_license_confirmation
            and (step.dependency_id is None or step.dependency_id not in accepted)
        ):
            raise StartupSetupConsentError("startup setup license confirmation is required")
        handler = self.handlers.get(step.operation)
        if handler is None:
            raise StartupSetupStepError("startup setup operation is unavailable")
        operation_id = handler(step)
        return StartupSetupExecution(
            step_id=step.id,
            status="completed" if operation_id is None else "completed",
            operation_id=operation_id,
        )


def validate_starter_pack_selection(
    packs: tuple[StartupStarterPack, ...],
    selected_pack_ids: tuple[str, ...],
    *,
    download_confirmed: bool,
    accepted_license_pack_ids: tuple[str, ...],
) -> tuple[StartupStarterPack, ...]:
    if len(set(selected_pack_ids)) != len(selected_pack_ids):
        raise StartupStarterPackSelectionError("starter pack selection contains duplicates")
    by_id = {pack.id: pack for pack in packs}
    if any(pack_id not in by_id for pack_id in selected_pack_ids):
        raise StartupStarterPackSelectionError("starter pack selection is unsupported")
    selected = tuple(by_id[pack_id] for pack_id in selected_pack_ids)
    pending = tuple(pack for pack in selected if pack.status != "installed")
    if pending and not download_confirmed:
        raise StartupSetupConsentError("starter pack download confirmation is required")
    accepted = set(accepted_license_pack_ids)
    if any(pack.requires_license_confirmation and pack.id not in accepted for pack in pending):
        raise StartupSetupConsentError("starter pack license confirmation is required")
    return pending
