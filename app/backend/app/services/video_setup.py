from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.services.atomic_json import write_json_atomic
from app.services.managed_artifacts import ManagedArtifactService, ManagedArtifactSpec
from app.services.managed_downloads import managed_download_snapshot
from app.services.video_model_profiles import FeaturePack, VideoModelProfile


class VideoSetupError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Coordinator(Protocol):
    def enqueue(self, **kwargs): ...


class SetupLicenseState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    version: str
    url: str
    accepted: bool
    accepted_at: str | None = None


class SetupArtifactState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    state: Literal["missing", "partial", "installed"]
    size_bytes: int
    remaining_bytes: int


class SetupDiskState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["sufficient", "insufficient", "unknown"]
    required_bytes: int
    available_bytes: int | None
    safety_margin_bytes: int


class VideoSetupPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    feature_pack: FeaturePack
    state: Literal["ready", "setup_required", "disk_space_insufficient"]
    runtime_minimum_version: str
    runtime_managed_version: str
    license: SetupLicenseState
    artifacts: tuple[SetupArtifactState, ...]
    disk: SetupDiskState
    reason_codes: tuple[str, ...]


class VideoSetupStartResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    feature_pack: FeaturePack
    operation_ids: tuple[str, ...]
    restart_required: bool
    state: Literal["queued", "ready"]


class VideoSetupService:
    def __init__(
        self,
        *,
        profile: VideoModelProfile,
        runtime_root: Path,
        data_dir: Path,
        runtime_spec: ManagedArtifactSpec,
        free_space: Callable[[Path], int] | None = None,
        clock: Callable[[], datetime] | None = None,
        safety_margin_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        self.profile = profile
        self.runtime_root = runtime_root.absolute()
        self.data_dir = data_dir.absolute()
        self.runtime_spec = runtime_spec
        self.free_space = free_space or (lambda path: shutil.disk_usage(path).free)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.safety_margin_bytes = safety_margin_bytes
        self.license_receipt_path = (
            self.data_dir / "licenses" / f"{self.profile.license.id}.json"
        )
        self._managed = ManagedArtifactService(self.data_dir / "temp")

    def plan(self, feature_pack: FeaturePack) -> VideoSetupPlan:
        specs = self._specs(feature_pack)
        artifacts = tuple(self._artifact_state(spec) for spec in specs)
        required = sum(item.remaining_bytes for item in artifacts)
        disk = self._disk_state(required)
        license_state = self._license_state()
        reasons: list[str] = []
        if not license_state.accepted:
            reasons.append("video_setup_license_confirmation_required")
        if artifacts[0].state != "installed":
            reasons.append("video_local_runtime_required")
        asset_by_id = {asset.id: asset for asset in self.profile.assets}
        missing_models = [
            item
            for item in artifacts[1:]
            if item.state != "installed" and item.id in asset_by_id
        ]
        if any(asset_by_id[item.id].pack == "shared" for item in missing_models):
            reasons.append("video_local_shared_models_required")
        if any(asset_by_id[item.id].pack != "shared" for item in missing_models):
            reasons.append("video_local_task_model_required")
        if disk.state == "insufficient":
            reasons.append("video_setup_disk_space_insufficient")
        if disk.state == "unknown":
            reasons.append("video_setup_disk_space_unknown")
        if not reasons:
            state: Literal["ready", "setup_required", "disk_space_insufficient"] = "ready"
        elif disk.state == "insufficient":
            state = "disk_space_insufficient"
        else:
            state = "setup_required"
        return VideoSetupPlan(
            profile_id=self.profile.profile_id,
            feature_pack=feature_pack,
            state=state,
            runtime_minimum_version=self.profile.runtime.minimum_version,
            runtime_managed_version=self.profile.runtime.managed_version,
            license=license_state,
            artifacts=artifacts,
            disk=disk,
            reason_codes=tuple(reasons),
        )

    def start(
        self,
        feature_pack: FeaturePack,
        *,
        license_version: str,
        agree_and_download: bool,
        coordinator: _Coordinator,
        idempotency_key: str | None = None,
    ) -> VideoSetupStartResult:
        if not agree_and_download:
            raise VideoSetupError("video_setup_license_confirmation_required")
        if license_version != self.profile.license.version:
            raise VideoSetupError("video_setup_license_version_changed")
        plan = self.plan(feature_pack)
        if plan.disk.state == "insufficient":
            raise VideoSetupError("video_setup_disk_space_insufficient")
        if plan.disk.state == "unknown":
            raise VideoSetupError("video_setup_disk_space_unknown")

        write_json_atomic(
            self.license_receipt_path,
            {
                "license_version": self.profile.license.version,
                "accepted_at": _utc_text(self.clock()),
                "confirmation": "agree_and_download",
            },
        )
        operation_ids: list[str] = []
        runtime_queued = False
        pending = list(zip(self._specs(feature_pack), plan.artifacts, strict=True))
        # The runtime archive atomically replaces runtime/ComfyUI. Model files must
        # therefore be scheduled only after that stage completes and Asteria restarts.
        if pending and pending[0][1].state != "installed":
            pending = pending[:1]
        else:
            pending = pending[1:]
        for spec, status in pending:
            if status.state == "installed":
                continue
            key = f"{idempotency_key}:{spec.component_id}" if idempotency_key else None
            operation = coordinator.enqueue(
                kind="managed.download",
                source_actor="local",
                request_snapshot=managed_download_snapshot(
                    spec,
                    requires_runtime_stop=spec.component_id == self.runtime_spec.component_id,
                ),
                idempotency_key=key,
                progress_total=spec.expected_size,
                progress_unit="bytes",
            )
            operation_ids.append(str(operation.operation_id))
            runtime_queued = runtime_queued or spec.component_id == self.runtime_spec.component_id
        return VideoSetupStartResult(
            profile_id=self.profile.profile_id,
            feature_pack=feature_pack,
            operation_ids=tuple(operation_ids),
            restart_required=runtime_queued,
            state="queued" if operation_ids else "ready",
        )

    def _specs(self, feature_pack: FeaturePack) -> tuple[ManagedArtifactSpec, ...]:
        if feature_pack not in {"base", "reference", "all"}:
            raise VideoSetupError("video_setup_feature_pack_invalid")
        return (
            self.runtime_spec,
            *self.profile.managed_specs_for_pack(feature_pack, self.runtime_root),
        )

    def _artifact_state(self, spec: ManagedArtifactSpec) -> SetupArtifactState:
        if self._managed.is_installed(spec):
            return SetupArtifactState(
                id=spec.component_id,
                state="installed",
                size_bytes=spec.expected_size or spec.size_limit,
                remaining_bytes=0,
            )
        part_path = self._managed.part_path_for(spec)
        if self._managed.has_valid_partial(spec):
            try:
                partial_size = part_path.stat().st_size
            except OSError:
                partial_size = 0
            total = spec.expected_size or spec.size_limit
            return SetupArtifactState(
                id=spec.component_id,
                state="partial",
                size_bytes=total,
                remaining_bytes=max(0, total - partial_size),
            )
        resume_path = self._managed.resume_receipt_path_for(spec)
        if part_path.exists() or resume_path.exists():
            self._managed.discard_partial(spec)
        size = spec.expected_size or spec.size_limit
        return SetupArtifactState(
            id=spec.component_id,
            state="missing",
            size_bytes=size,
            remaining_bytes=size,
        )

    def _disk_state(self, required_bytes: int) -> SetupDiskState:
        if required_bytes == 0:
            return SetupDiskState(
                state="sufficient",
                required_bytes=0,
                available_bytes=None,
                safety_margin_bytes=self.safety_margin_bytes,
            )
        try:
            anchor = self.runtime_root
            while not anchor.exists() and anchor.parent != anchor:
                anchor = anchor.parent
            available = int(self.free_space(anchor))
        except (OSError, ValueError):
            return SetupDiskState(
                state="unknown",
                required_bytes=required_bytes,
                available_bytes=None,
                safety_margin_bytes=self.safety_margin_bytes,
            )
        state: Literal["sufficient", "insufficient", "unknown"] = (
            "sufficient"
            if available >= required_bytes + self.safety_margin_bytes
            else "insufficient"
        )
        return SetupDiskState(
            state=state,
            required_bytes=required_bytes,
            available_bytes=available,
            safety_margin_bytes=self.safety_margin_bytes,
        )

    def _license_state(self) -> SetupLicenseState:
        accepted_at: str | None = None
        accepted = False
        try:
            import json

            payload = json.loads(self.license_receipt_path.read_text(encoding="utf-8"))
            accepted = (
                isinstance(payload, dict)
                and set(payload) == {"license_version", "accepted_at", "confirmation"}
                and payload.get("license_version") == self.profile.license.version
                and payload.get("confirmation") == "agree_and_download"
                and isinstance(payload.get("accepted_at"), str)
            )
            accepted_at = payload.get("accepted_at") if accepted else None
        except (OSError, ValueError):
            pass
        return SetupLicenseState(
            id=self.profile.license.id,
            name=self.profile.license.name,
            version=self.profile.license.version,
            url=self.profile.license.url,
            accepted=accepted,
            accepted_at=accepted_at,
        )


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
