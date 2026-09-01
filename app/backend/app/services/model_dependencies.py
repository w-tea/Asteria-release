from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import httpx
import yaml

from app.services.managed_artifacts import (
    ManagedArtifactError,
    ManagedArtifactPathTooLong,
    ManagedArtifactRollbackIncomplete,
    ManagedArtifactService,
    ManagedArtifactSpec,
)
from app.services.model_fingerprints import ModelFingerprintService


class ModelDependencyError(Exception):
    pass


class LicenseConfirmationRequired(ModelDependencyError):
    pass


class ModelDependencyNotFound(ModelDependencyError):
    pass


@dataclass(frozen=True)
class ModelDependency:
    id: str
    label: str
    profile: str
    type: str
    provider: str
    url: str
    relative_path: Path
    slot_id: str | None
    size_bytes: int | None
    checksum_sha256: str | None
    license_id: str | None
    requires_license_confirmation: bool
    auto_install: bool
    release_class: str = "development"
    source_revision: str | None = None
    starter_pack: str | None = None
    selected_by_default: bool = False
    license_url: str | None = None
    repair_policy: str | None = None
    source_verified_at: str | None = None
    source_provenance_url: str | None = None


@dataclass(frozen=True)
class ModelDependencyStatus:
    id: str
    label: str
    profile: str
    type: str
    provider: str
    status: str
    path: Path
    slot_id: str | None
    source_url: str
    size_bytes: int | None
    local_size_bytes: int | None
    checksum_sha256: str | None
    local_checksum_sha256: str | None
    license_id: str | None
    requires_license_confirmation: bool
    auto_install: bool
    release_class: str = "development"
    source_revision: str | None = None
    starter_pack: str | None = None
    selected_by_default: bool = False
    license_url: str | None = None
    repair_policy: str | None = None
    source_verified_at: str | None = None
    source_provenance_url: str | None = None


class ModelDependencyRepository:
    def __init__(
        self,
        manifest_path: Path,
        model_root: Path,
        fingerprint_service: ModelFingerprintService | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.model_root = model_root
        self.fingerprint_service = fingerprint_service

    def list(self) -> list[ModelDependency]:
        try:
            manifest_exists = self.manifest_path.exists()
        except OSError:
            manifest_exists = False
        if not manifest_exists:
            return []

        try:
            raw = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return []
        if not isinstance(raw, dict):
            return []
        dependencies = raw.get("dependencies", [])
        if not isinstance(dependencies, list):
            return []
        parsed: list[ModelDependency] = []
        for item in dependencies:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(self._parse_dependency(item))
            except (KeyError, TypeError, ValueError):
                continue
        return parsed

    def list_status(self, dependency_ids: set[str] | None = None) -> list[ModelDependencyStatus]:
        dependencies = self.list()
        if dependency_ids is not None:
            dependencies = [
                dependency for dependency in dependencies if dependency.id in dependency_ids
            ]
        return [self._status_for(dependency) for dependency in dependencies]

    def get(self, dependency_id: str) -> ModelDependency:
        for dependency in self.list():
            if dependency.id == dependency_id:
                return dependency
        raise ModelDependencyNotFound(dependency_id)

    def path_for(self, dependency: ModelDependency) -> Path:
        return self.model_root / dependency.relative_path

    def _status_for(self, dependency: ModelDependency) -> ModelDependencyStatus:
        path = self.path_for(dependency)
        local_size, readable = self._local_size(path)
        local_checksum: str | None = None
        if readable and local_size is not None and dependency.checksum_sha256:
            local_checksum, checksum_readable = self._local_checksum(path)
            readable = readable and checksum_readable
        status = "installed" if local_size is not None else "missing"
        if not readable:
            status = "unreadable"
        elif local_size is not None and dependency.size_bytes is not None and local_size != dependency.size_bytes:
            status = "size_mismatch"
        elif local_size is not None and dependency.checksum_sha256 and local_checksum != dependency.checksum_sha256:
            status = "checksum_mismatch"

        return ModelDependencyStatus(
            id=dependency.id,
            label=dependency.label,
            profile=dependency.profile,
            type=dependency.type,
            provider=dependency.provider,
            status=status,
            path=path,
            slot_id=dependency.slot_id,
            source_url=dependency.url,
            size_bytes=dependency.size_bytes,
            local_size_bytes=local_size,
            checksum_sha256=dependency.checksum_sha256,
            local_checksum_sha256=local_checksum,
            license_id=dependency.license_id,
            requires_license_confirmation=dependency.requires_license_confirmation,
            auto_install=dependency.auto_install,
            release_class=dependency.release_class,
            source_revision=dependency.source_revision,
            starter_pack=dependency.starter_pack,
            selected_by_default=dependency.selected_by_default,
            license_url=dependency.license_url,
            repair_policy=dependency.repair_policy,
            source_verified_at=dependency.source_verified_at,
            source_provenance_url=dependency.source_provenance_url,
        )

    def _local_size(self, path: Path) -> tuple[int | None, bool]:
        try:
            if not path.exists() or not path.is_file():
                return None, True
            return path.stat().st_size, True
        except OSError:
            return None, False

    def _local_checksum(self, path: Path) -> tuple[str | None, bool]:
        try:
            if self.fingerprint_service is not None:
                return self.fingerprint_service.strong_hash(path), True
            return self._sha256(path), True
        except OSError:
            return None, False

    def _parse_dependency(self, item: dict) -> ModelDependency:
        source = item["source"]
        target = item["target"]
        license_info = item.get("license") or {}
        if not isinstance(source, dict) or not isinstance(target, dict) or not isinstance(license_info, dict):
            raise ValueError("model dependency nested sections must be objects")
        required_values = [
            item["id"],
            item["label"],
            item["profile"],
            item["type"],
            source["provider"],
            source["url"],
            target["relative_path"],
        ]
        if not all(isinstance(value, str) for value in required_values):
            raise ValueError("model dependency required fields must be strings")
        slot_id = target.get("slot_id")
        if slot_id is not None and not isinstance(slot_id, str):
            raise ValueError("model dependency slot_id must be a string")
        size_bytes = item.get("size_bytes")
        if size_bytes is not None and not isinstance(size_bytes, int):
            raise ValueError("model dependency size_bytes must be an integer")
        checksum_sha256 = item.get("checksum_sha256")
        if checksum_sha256 is not None and not isinstance(checksum_sha256, str):
            raise ValueError("model dependency checksum_sha256 must be a string")
        license_id = license_info.get("id")
        if license_id is not None and not isinstance(license_id, str):
            raise ValueError("model dependency license id must be a string")
        requires_confirmation = license_info.get("requires_confirmation", False)
        if not isinstance(requires_confirmation, bool):
            raise ValueError("model dependency license confirmation must be a boolean")
        auto_install = item.get("auto_install", False)
        if not isinstance(auto_install, bool):
            raise ValueError("model dependency auto_install must be a boolean")
        release_class = item.get("release_class", "development")
        if release_class not in {
            "development",
            "mandatory-feature",
            "optional-starter",
            "user-owned",
        }:
            raise ValueError("model dependency release_class is invalid")
        source_revision = source.get("revision")
        if source_revision is not None and not isinstance(source_revision, str):
            raise ValueError("model dependency source revision must be a string")
        starter_pack = item.get("starter_pack")
        if starter_pack is not None and not isinstance(starter_pack, str):
            raise ValueError("model dependency starter_pack must be a string")
        selected_by_default = item.get("selected_by_default", False)
        if not isinstance(selected_by_default, bool):
            raise ValueError("model dependency selected_by_default must be a boolean")
        license_url = license_info.get("url")
        if license_url is not None and not isinstance(license_url, str):
            raise ValueError("model dependency license url must be a string")
        repair_policy = item.get("repair_policy")
        if repair_policy is not None and not isinstance(repair_policy, str):
            raise ValueError("model dependency repair_policy must be a string")
        source_verified_at = source.get("verified_at")
        source_provenance_url = source.get("provenance_url")
        if source_verified_at is not None and not isinstance(source_verified_at, str):
            raise ValueError("model dependency verified_at must be a string")
        if source_provenance_url is not None and not isinstance(source_provenance_url, str):
            raise ValueError("model dependency provenance_url must be a string")
        return ModelDependency(
            id=item["id"],
            label=item["label"],
            profile=item["profile"],
            type=item["type"],
            provider=source["provider"],
            url=source["url"],
            relative_path=self._parse_relative_path(target["relative_path"]),
            slot_id=slot_id,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            license_id=license_id,
            requires_license_confirmation=requires_confirmation,
            auto_install=auto_install,
            release_class=release_class,
            source_revision=source_revision,
            starter_pack=starter_pack,
            selected_by_default=selected_by_default,
            license_url=license_url,
            repair_policy=repair_policy,
            source_verified_at=source_verified_at,
            source_provenance_url=source_provenance_url,
        )

    def _parse_relative_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("model dependency target must stay within model root")
        return path

    def _sha256(self, path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class ModelDependencyInstaller:
    def __init__(
        self,
        repository: ModelDependencyRepository,
        stream_factory: Callable[[str], Iterable[bytes]] | None = None,
        managed_artifact_service: ManagedArtifactService | None = None,
    ) -> None:
        self.repository = repository
        self.stream_factory = stream_factory or self._http_stream
        self.managed_artifact_service = managed_artifact_service

    def install(self, dependency_id: str, license_confirmed: bool) -> ModelDependencyStatus:
        dependency = self.repository.get(dependency_id)
        current = self.repository._status_for(dependency)
        if current.status == "installed":
            return current
        if dependency.provider == "manual" or dependency.url.startswith("manual://"):
            raise ModelDependencyError(f"Dependency requires manual placement: {dependency_id}")
        if dependency.requires_license_confirmation and not license_confirmed:
            raise LicenseConfirmationRequired(dependency_id)
        if dependency.release_class == "user-owned":
            raise ModelDependencyError(f"Dependency requires user-owned model placement: {dependency_id}")
        if dependency.release_class in {"mandatory-feature", "optional-starter"}:
            return self._install_managed(dependency)

        target = self.repository.path_for(dependency)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.exists() and not target.is_file():
                raise ModelDependencyError(f"Dependency target path is not a file: {target}")
        except OSError as exc:
            raise ModelDependencyError(f"Could not inspect model dependency target: {target}") from exc

        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, dir=target.parent) as tmp:
                temp_path = Path(tmp.name)
                for chunk in self.stream_factory(dependency.url):
                    if chunk:
                        tmp.write(chunk)
            shutil.move(str(temp_path), target)
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_exists = temp_path.exists()
                except OSError:
                    temp_exists = False
                if temp_exists:
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
            if isinstance(exc, ModelDependencyError):
                raise
            message = str(exc) or f"Failed to install model dependency: {dependency.id}"
            raise ModelDependencyError(message) from exc
        return self.repository._status_for(dependency)

    def _install_managed(self, dependency: ModelDependency) -> ModelDependencyStatus:
        if dependency.size_bytes is None or dependency.checksum_sha256 is None:
            raise ModelDependencyError(f"Managed dependency metadata is incomplete: {dependency.id}")
        service = self.managed_artifact_service or ManagedArtifactService(
            self.repository.model_root.parent / "temp"
        )
        spec = ManagedArtifactSpec(
            manifest_version=1,
            component_id=dependency.id,
            source_url=dependency.url,
            destination=self.repository.path_for(dependency),
            expected_size=dependency.size_bytes,
            checksum_sha256=dependency.checksum_sha256,
            kind="file",
        )
        try:
            service.acquire(spec)
            service.mark_post_install_complete(spec)
        except (ManagedArtifactPathTooLong, ManagedArtifactRollbackIncomplete):
            raise
        except (ManagedArtifactError, OSError) as exc:
            message = str(exc) or f"Failed to install model dependency: {dependency.id}"
            raise ModelDependencyError(message) from exc
        return self.repository._status_for(dependency)

    def _http_stream(self, url: str) -> Iterable[bytes]:
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            yield from response.iter_bytes()


@dataclass(frozen=True)
class ModelBootstrapResult:
    installed: list[ModelDependencyStatus]
    skipped: list[ModelDependencyStatus]
    failed: dict[str, str]


@dataclass(frozen=True)
class ModelBootstrapProgress:
    current_dependency_id: str | None
    completed_count: int
    total_count: int


class ModelBootstrapper:
    def __init__(
        self,
        repository: ModelDependencyRepository,
        installer: ModelDependencyInstaller | None = None,
        progress_callback: Callable[[ModelBootstrapProgress], None] | None = None,
    ) -> None:
        self.repository = repository
        self.installer = installer or ModelDependencyInstaller(repository)
        self.progress_callback = progress_callback

    def bootstrap(self, profile: str | None = None) -> ModelBootstrapResult:
        installed: list[ModelDependencyStatus] = []
        skipped: list[ModelDependencyStatus] = []
        failed: dict[str, str] = {}

        dependencies = self.repository.list()
        if profile is not None:
            dependencies = [
                dependency
                for dependency in dependencies
                if dependency.profile == profile or dependency.profile == "shared"
            ]

        total_count = len(dependencies)
        completed_count = 0
        self._publish_progress(None, completed_count, total_count)
        for dependency in dependencies:
            self._publish_progress(dependency.id, completed_count, total_count)
            status = self.repository._status_for(dependency)
            if status.status == "installed" or not dependency.auto_install:
                skipped.append(status)
                completed_count += 1
                self._publish_progress(None, completed_count, total_count)
                continue
            try:
                installed.append(self.installer.install(dependency.id, license_confirmed=True))
            except Exception as exc:  # noqa: BLE001 - keep bootstrapping independent files.
                failed[dependency.id] = str(exc)
            completed_count += 1
            self._publish_progress(None, completed_count, total_count)

        return ModelBootstrapResult(installed=installed, skipped=skipped, failed=failed)

    def _publish_progress(
        self,
        current_dependency_id: str | None,
        completed_count: int,
        total_count: int,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            ModelBootstrapProgress(
                current_dependency_id=current_dependency_id,
                completed_count=completed_count,
                total_count=total_count,
            )
        )
