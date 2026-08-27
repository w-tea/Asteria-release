from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.atomic_json import write_json_atomic
from app.services.managed_artifacts import (
    ManagedArtifactError,
    ManagedArtifactService,
    ManagedArtifactSpec,
)
from app.services.managed_downloads import managed_download_snapshot


MusicExecutionProfile = Literal["full", "offload", "layer_streaming"]
_VERSION = re.compile(r"__version__\s*=\s*['\"](\d+\.\d+\.\d+)['\"]")


class MusicSetupError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Coordinator(Protocol):
    def enqueue(self, **kwargs): ...


class MusicRuntimeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: Literal["comfyui-core"]
    minimum_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    managed_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    minimum_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    python_version: Literal["3.12.13"]
    pytorch_package_set: Literal["pytorch-cu128"]
    cuda_required: Literal[True]


class MusicRepositoryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    original_url: str
    verified_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_urls(self) -> "MusicRepositoryProfile":
        _require_https(self.url, "music repository")
        _require_https(self.original_url, "original music repository")
        return self


class MusicLicenseProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["minimax-music3-community"]
    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    url: str

    @model_validator(mode="after")
    def validate_url(self) -> "MusicLicenseProfile":
        _require_https(self.url, "music license")
        if f"/blob/{self.revision}/LICENSE" not in self.url:
            raise ValueError("music license URL must bind its immutable revision")
        return self


class MusicOfficialApiProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    advertised: bool
    model_id: Literal["music-3.0"]
    availability: Literal["legacy_existing_paid_users_only"]
    implemented: Literal[False]
    url: str
    policy_url: str
    verified_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_urls(self) -> "MusicOfficialApiProfile":
        _require_https(self.url, "music API")
        _require_https(self.policy_url, "music API policy")
        return self


class MusicModelAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$")
    label: str = Field(min_length=1)
    component_role: Literal["diffusion_model", "text_encoder", "audio_vae"]
    relative_path: str
    source_url: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    provenance_url: str
    size_bytes: int = Field(gt=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source_and_path(self) -> "MusicModelAsset":
        _safe_relative_path(self.relative_path, "music model path")
        _require_https(self.source_url, "music model source")
        _require_https(self.provenance_url, "music model provenance")
        if f"/resolve/{self.revision}/" not in self.source_url:
            raise ValueError("music model source must use an immutable revision")
        if f"/blob/{self.revision}/" not in self.provenance_url:
            raise ValueError("music model provenance must use an immutable revision")
        if not self.relative_path.casefold().endswith(".safetensors"):
            raise ValueError("music model path must target a safetensors file")
        return self

    def managed_spec(self, runtime_root: Path) -> ManagedArtifactSpec:
        return ManagedArtifactSpec(
            manifest_version=1,
            component_id=self.id,
            source_url=self.source_url,
            destination=runtime_root / Path(self.relative_path),
            expected_size=self.size_bytes,
            checksum_sha256=self.checksum_sha256,
            kind="file",
            require_https=True,
            max_size=self.size_bytes,
        )


class MusicExecutionProfileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: MusicExecutionProfile
    label: str = Field(min_length=1)
    runtime_mode: Literal["resident", "cpu_offload", "layer_streaming"]
    vram_advisory_gb: int = Field(gt=0)
    recommended_bundle_id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$")


class MusicModelBundleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$")
    label: str = Field(min_length=1)
    asset_ids: tuple[str, ...]


class MusicModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.music.model-profile.v1"]
    profile_id: Literal["minimax-music3-local"]
    version: Literal[1]
    provider: Literal["local_music3"]
    attribution: Literal["MiniMax-Music3"]
    runtime: MusicRuntimeRequirement
    model_repository: MusicRepositoryProfile
    license: MusicLicenseProfile
    official_api: MusicOfficialApiProfile
    execution_profiles: tuple[MusicExecutionProfileRecord, ...]
    model_bundles: tuple[MusicModelBundleRecord, ...]
    assets: tuple[MusicModelAsset, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> "MusicModelProfile":
        assets = {asset.id: asset for asset in self.assets}
        if len(assets) != len(self.assets):
            raise ValueError("music profile has duplicate asset ids")
        paths = [asset.relative_path.replace("\\", "/").casefold() for asset in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("music profile has duplicate asset paths")
        profiles = {profile.id: profile for profile in self.execution_profiles}
        if set(profiles) != {"full", "offload", "layer_streaming"}:
            raise ValueError("music execution profile coverage is incomplete")
        bundles = {bundle.id: bundle for bundle in self.model_bundles}
        if len(bundles) != len(self.model_bundles) or not bundles:
            raise ValueError("music model bundles must be non-empty and unique")
        for bundle in bundles.values():
            if len(bundle.asset_ids) != 3 or len(set(bundle.asset_ids)) != 3:
                raise ValueError("music model bundle must select three unique assets")
            if any(asset_id not in assets for asset_id in bundle.asset_ids):
                raise ValueError("music model bundle references an unknown asset")
            if {assets[asset_id].component_role for asset_id in bundle.asset_ids} != {
                "diffusion_model",
                "text_encoder",
                "audio_vae",
            }:
                raise ValueError("music model bundle must cover every component role")
        if any(profile.recommended_bundle_id not in bundles for profile in profiles.values()):
            raise ValueError("music execution profile references an unknown recommended bundle")
        if any(asset.revision != self.model_repository.revision for asset in self.assets):
            raise ValueError("music model revisions must match the repository revision")
        return self

    def execution_profile(
        self,
        execution_profile: MusicExecutionProfile,
    ) -> MusicExecutionProfileRecord:
        selected = next(
            (item for item in self.execution_profiles if item.id == execution_profile),
            None,
        )
        if selected is None:
            raise MusicSetupError("music_setup_execution_profile_invalid")
        return selected

    def model_bundle(self, bundle_id: str) -> MusicModelBundleRecord:
        selected = next((item for item in self.model_bundles if item.id == bundle_id), None)
        if selected is None:
            raise MusicSetupError("music_setup_model_bundle_invalid")
        return selected

    def managed_specs(
        self,
        execution_profile: MusicExecutionProfile,
        runtime_root: Path,
    ) -> tuple[ManagedArtifactSpec, ...]:
        selected = self.execution_profile(execution_profile)
        return self.managed_specs_for_bundle(selected.recommended_bundle_id, runtime_root)

    def managed_specs_for_bundle(
        self,
        bundle_id: str,
        runtime_root: Path,
    ) -> tuple[ManagedArtifactSpec, ...]:
        selected = self.model_bundle(bundle_id)
        assets = {asset.id: asset for asset in self.assets}
        return tuple(assets[asset_id].managed_spec(runtime_root) for asset_id in selected.asset_ids)


class MusicModelProfileRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> MusicModelProfile:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return MusicModelProfile.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"music model profile is invalid: {exc}") from exc


class MusicSetupLicenseState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    version: str
    revision: str
    url: str
    attribution: str
    accepted: bool
    accepted_at: str | None = None


class MusicSetupRuntimeState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["missing", "current", "incompatible"]
    minimum_version: str
    managed_version: str
    minimum_commit: str
    detected_version: str | None = None


class MusicSetupArtifactState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    state: Literal["missing", "partial", "installed", "corrupt"]
    size_bytes: int
    remaining_bytes: int


class MusicSetupDiskState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["sufficient", "insufficient", "unknown"]
    required_bytes: int
    available_bytes: int | None
    safety_margin_bytes: int


class MusicSetupPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    execution_profile: MusicExecutionProfile
    selected_execution_profile: MusicExecutionProfile
    attribution: str
    state: Literal["ready", "setup_required", "disk_space_insufficient"]
    runtime: MusicSetupRuntimeState
    license: MusicSetupLicenseState
    artifacts: tuple[MusicSetupArtifactState, ...]
    disk: MusicSetupDiskState
    download_size_bytes: int
    reason_codes: tuple[str, ...]


class MusicSetupStartResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    execution_profile: MusicExecutionProfile
    operation_ids: tuple[str, ...]
    restart_required: bool
    state: Literal["queued", "ready"]


class MusicLocalSetupService:
    def __init__(
        self,
        *,
        profile: MusicModelProfile,
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
        self.license_receipt_path = self.data_dir / "licenses" / f"{profile.license.id}.json"
        self.selection_path = self.data_dir / "music" / "minimax-music3-selection.json"
        self._managed = ManagedArtifactService(self.data_dir / "temp")

    def plan(self, execution_profile: MusicExecutionProfile) -> MusicSetupPlan:
        specs = self.profile.managed_specs(execution_profile, self.runtime_root)
        runtime = self._runtime_state()
        artifacts = tuple(self._artifact_state(spec) for spec in specs)
        runtime_remaining = 0 if runtime.state == "current" else self.runtime_spec.expected_size or self.runtime_spec.size_limit
        required = runtime_remaining + sum(item.remaining_bytes for item in artifacts)
        disk = self._disk_state(required)
        license_state = self._license_state()
        reasons: list[str] = []
        if not license_state.accepted:
            reasons.append("music_setup_license_confirmation_required")
        if runtime.state == "missing":
            reasons.append("music_local_runtime_required")
        elif runtime.state == "incompatible":
            reasons.append("music_local_runtime_incompatible")
        if any(item.state != "installed" for item in artifacts):
            reasons.append("music_local_model_required")
        if any(item.state == "corrupt" for item in artifacts):
            reasons.append("music_local_model_repair_required")
        if disk.state == "insufficient":
            reasons.append("music_setup_disk_space_insufficient")
        elif disk.state == "unknown":
            reasons.append("music_setup_disk_space_unknown")
        state: Literal["ready", "setup_required", "disk_space_insufficient"]
        if not reasons:
            state = "ready"
        elif disk.state == "insufficient":
            state = "disk_space_insufficient"
        else:
            state = "setup_required"
        return MusicSetupPlan(
            profile_id=self.profile.profile_id,
            execution_profile=execution_profile,
            selected_execution_profile=self.selected_execution_profile(),
            attribution=self.profile.attribution,
            state=state,
            runtime=runtime,
            license=license_state,
            artifacts=artifacts,
            disk=disk,
            download_size_bytes=required,
            reason_codes=tuple(reasons),
        )

    def start(
        self,
        execution_profile: MusicExecutionProfile,
        *,
        license_version: str,
        agree_and_download: bool,
        coordinator: _Coordinator,
        idempotency_key: str | None = None,
        repair: bool = False,
    ) -> MusicSetupStartResult:
        if not agree_and_download:
            raise MusicSetupError("music_setup_license_confirmation_required")
        if license_version != self.profile.license.version:
            raise MusicSetupError("music_setup_license_version_changed")
        plan = self.plan(execution_profile)
        model_pairs: tuple[tuple[ManagedArtifactSpec, MusicSetupArtifactState], ...] = ()
        integrity_failures: set[str] = set()
        runtime_integrity_failed = (
            plan.runtime.state == "current"
            and not self._managed.verify_installed_contents(self.runtime_spec)
        )
        if plan.runtime.state == "current" and not runtime_integrity_failed:
            specs = self.profile.managed_specs(execution_profile, self.runtime_root)
            model_pairs = tuple(zip(specs, plan.artifacts, strict=True))
            for spec, status in model_pairs:
                if status.state == "installed" and not self._managed.verify_installed_contents(spec):
                    integrity_failures.add(spec.component_id)
        disk = plan.disk
        if repair:
            repair_required = 0
            if plan.runtime.state != "current" or runtime_integrity_failed:
                repair_required += self.runtime_spec.expected_size or self.runtime_spec.size_limit
            else:
                for spec, status in model_pairs:
                    if status.state != "installed" or spec.component_id in integrity_failures:
                        repair_required += spec.expected_size or spec.size_limit
            disk = self._disk_state(repair_required)
        elif runtime_integrity_failed:
            disk = self._disk_state(self.runtime_spec.expected_size or self.runtime_spec.size_limit)
        elif integrity_failures:
            disk = self._disk_state(
                plan.disk.required_bytes
                + sum(
                    spec.expected_size or spec.size_limit
                    for spec, _status in model_pairs
                    if spec.component_id in integrity_failures
                )
            )
        if disk.state == "insufficient":
            raise MusicSetupError("music_setup_disk_space_insufficient")
        if disk.state == "unknown":
            raise MusicSetupError("music_setup_disk_space_unknown")
        self._write_license_receipt()
        self._write_execution_profile(execution_profile)

        operation_ids: list[str] = []
        if plan.runtime.state != "current" or runtime_integrity_failed:
            if runtime_integrity_failed:
                self._managed.invalidate_install_receipt(self.runtime_spec)
                if repair:
                    self._managed.discard_partial(self.runtime_spec)
            pending = ((self.runtime_spec, None),)
        else:
            pending = model_pairs
        for spec, status in pending:
            if repair:
                if (
                    status is not None
                    and status.state == "installed"
                    and spec.component_id not in integrity_failures
                ):
                    continue
                self._managed.invalidate_install_receipt(spec)
                self._managed.discard_partial(spec)
            elif status is not None and status.state == "installed":
                if spec.component_id not in integrity_failures:
                    continue
                self._managed.invalidate_install_receipt(spec)
            key = f"{idempotency_key}:{spec.component_id}" if idempotency_key else None
            snapshot = managed_download_snapshot(
                spec,
                requires_runtime_stop=spec.component_id == self.runtime_spec.component_id,
            )
            snapshot["setup_owner"] = f"music:{self.profile.profile_id}"
            snapshot["setup_execution_profile"] = execution_profile
            operation = coordinator.enqueue(
                kind="managed.download",
                source_actor="local",
                request_snapshot=snapshot,
                idempotency_key=key,
                progress_total=spec.expected_size,
                progress_unit="bytes",
            )
            operation_ids.append(str(operation.operation_id))
        return MusicSetupStartResult(
            profile_id=self.profile.profile_id,
            execution_profile=execution_profile,
            operation_ids=tuple(operation_ids),
            restart_required=(plan.runtime.state != "current" or runtime_integrity_failed) and bool(operation_ids),
            state="queued" if operation_ids else "ready",
        )

    def component_ids(self) -> frozenset[str]:
        return frozenset({self.runtime_spec.component_id, *(asset.id for asset in self.profile.assets)})

    def selected_execution_profile(self) -> MusicExecutionProfile:
        try:
            payload = json.loads(self.selection_path.read_text(encoding="utf-8"))
            value = payload.get("execution_profile")
            if not any(item.id == value for item in self.profile.execution_profiles):
                raise ValueError("music execution profile is invalid")
            return value
        except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return "layer_streaming"

    def select_execution_profile(
        self,
        execution_profile: MusicExecutionProfile,
    ) -> MusicSetupPlan:
        plan = self.plan(execution_profile)
        if plan.state != "ready":
            raise MusicSetupError("music_local_setup_required")
        self._write_execution_profile(execution_profile)
        return plan.model_copy(
            update={"selected_execution_profile": execution_profile}
        )

    def _write_execution_profile(self, execution_profile: MusicExecutionProfile) -> None:
        if not any(item.id == execution_profile for item in self.profile.execution_profiles):
            raise MusicSetupError("music_setup_execution_profile_invalid")
        write_json_atomic(
            self.selection_path,
            {
                "schema_version": "asteria.music.execution-selection.v1",
                "profile_id": self.profile.profile_id,
                "execution_profile": execution_profile,
                "selected_at": _utc_text(self.clock()),
            },
        )

    def _write_license_receipt(self) -> None:
        write_json_atomic(
            self.license_receipt_path,
            {
                "license_version": self.profile.license.version,
                "license_url": self.profile.license.url,
                "license_revision": self.profile.license.revision,
                "attribution": self.profile.attribution,
                "accepted_at": _utc_text(self.clock()),
                "confirmation": "agree_and_download",
            },
        )

    def _runtime_state(self) -> MusicSetupRuntimeState:
        try:
            managed_current = self._managed.is_installed(self.runtime_spec)
        except ManagedArtifactError:
            managed_current = False
        if managed_current:
            return MusicSetupRuntimeState(
                state="current",
                minimum_version=self.profile.runtime.minimum_version,
                managed_version=self.profile.runtime.managed_version,
                minimum_commit=self.profile.runtime.minimum_commit,
                detected_version=self.profile.runtime.managed_version,
            )
        if not self.runtime_root.exists():
            state: Literal["missing", "current", "incompatible"] = "missing"
            detected = None
        else:
            detected = self._detect_runtime_version()
            # A plain version string cannot prove that a fork contains the required
            # MiniMax-Music3 commit. Only the pinned managed receipt is accepted.
            state = "incompatible"
        return MusicSetupRuntimeState(
            state=state,
            minimum_version=self.profile.runtime.minimum_version,
            managed_version=self.profile.runtime.managed_version,
            minimum_commit=self.profile.runtime.minimum_commit,
            detected_version=detected,
        )

    def _detect_runtime_version(self) -> str | None:
        path = self.runtime_root / "comfy" / "comfy_version.py"
        try:
            match = _VERSION.search(path.read_text(encoding="utf-8"))
        except OSError:
            return None
        return match.group(1) if match is not None else None

    def _artifact_state(self, spec: ManagedArtifactSpec) -> MusicSetupArtifactState:
        size = spec.expected_size or spec.size_limit
        try:
            if self._managed.is_installed(spec):
                return MusicSetupArtifactState(id=spec.component_id, state="installed", size_bytes=size, remaining_bytes=0)
            partial_size = self._managed.valid_partial_size(spec)
        except ManagedArtifactError:
            return MusicSetupArtifactState(id=spec.component_id, state="corrupt", size_bytes=size, remaining_bytes=size)
        if partial_size > 0:
            return MusicSetupArtifactState(
                id=spec.component_id,
                state="partial",
                size_bytes=size,
                remaining_bytes=max(0, size - partial_size),
            )
        part = self._managed.part_path_for(spec)
        resume = self._managed.resume_receipt_path_for(spec)
        if part.exists() or resume.exists():
            return MusicSetupArtifactState(id=spec.component_id, state="corrupt", size_bytes=size, remaining_bytes=size)
        receipt = self._managed.receipt_path_for(spec)
        if spec.destination.exists() or receipt.exists():
            return MusicSetupArtifactState(id=spec.component_id, state="corrupt", size_bytes=size, remaining_bytes=size)
        return MusicSetupArtifactState(id=spec.component_id, state="missing", size_bytes=size, remaining_bytes=size)

    def _disk_state(self, required_bytes: int) -> MusicSetupDiskState:
        if required_bytes == 0:
            return MusicSetupDiskState(
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
            return MusicSetupDiskState(
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
        return MusicSetupDiskState(
            state=state,
            required_bytes=required_bytes,
            available_bytes=available,
            safety_margin_bytes=self.safety_margin_bytes,
        )

    def _license_state(self) -> MusicSetupLicenseState:
        expected = {
            "license_version": self.profile.license.version,
            "license_url": self.profile.license.url,
            "license_revision": self.profile.license.revision,
            "attribution": self.profile.attribution,
            "confirmation": "agree_and_download",
        }
        accepted = False
        accepted_at: str | None = None
        try:
            payload = json.loads(self.license_receipt_path.read_text(encoding="utf-8"))
            accepted = (
                isinstance(payload, dict)
                and set(payload) == {*expected, "accepted_at"}
                and all(payload.get(key) == value for key, value in expected.items())
                and isinstance(payload.get("accepted_at"), str)
            )
            accepted_at = payload.get("accepted_at") if accepted else None
        except (OSError, ValueError):
            pass
        return MusicSetupLicenseState(
            id=self.profile.license.id,
            name=self.profile.license.name,
            version=self.profile.license.version,
            revision=self.profile.license.revision,
            url=self.profile.license.url,
            attribution=self.profile.attribution,
            accepted=accepted,
            accepted_at=accepted_at,
        )


def _require_https(value: str, label: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme.casefold() != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{label} must use HTTPS without credentials")


def _safe_relative_path(value: str, label: str) -> Path:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or ".." in windows.parts
        or ".." in posix.parts
        or any(_unsafe_windows_part(part) for part in posix.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return Path(*posix.parts)


def _unsafe_windows_part(part: str) -> bool:
    if part in {"", "."} or ":" in part or part.endswith((" ", ".")):
        return True
    stem = part.split(".", 1)[0].casefold()
    return stem in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
