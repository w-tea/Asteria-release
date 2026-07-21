from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess

import yaml

from app.services.managed_artifacts import (
    ManagedArtifactError,
    ManagedArtifactService,
    ManagedArtifactSpec,
)


class RuntimeDependencyError(Exception):
    pass


class RuntimeDependencyNotFound(RuntimeDependencyError):
    pass


@dataclass(frozen=True)
class RuntimeDependency:
    id: str
    label: str
    type: str
    provider: str
    url: str
    ref: str | None
    relative_path: Path
    features: list[str]
    requirements_file: str | None
    install_script: str | None
    pip_packages: list[str]
    auto_install: bool
    release_class: str = "development"
    size_bytes: int | None = None
    checksum_sha256: str | None = None
    license_id: str | None = None
    license_url: str | None = None
    repair_policy: str | None = None
    source_verified_at: str | None = None
    source_provenance_url: str | None = None


@dataclass(frozen=True)
class RuntimeDependencyStatus:
    id: str
    label: str
    type: str
    provider: str
    status: str
    path: Path
    source_url: str
    ref: str | None
    features: list[str]
    requirements_file: str | None
    install_script: str | None
    pip_packages: list[str]
    auto_install: bool
    release_class: str = "development"
    size_bytes: int | None = None
    checksum_sha256: str | None = None
    license_id: str | None = None
    license_url: str | None = None
    repair_policy: str | None = None
    source_verified_at: str | None = None
    source_provenance_url: str | None = None


class RuntimeDependencyRepository:
    def __init__(self, manifest_path: Path, runtime_root: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.runtime_root = runtime_root.resolve()

    def list(self) -> list[RuntimeDependency]:
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
        parsed: list[RuntimeDependency] = []
        for item in dependencies:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(self._parse_dependency(item))
            except (KeyError, TypeError, ValueError):
                continue
        return parsed

    def list_status(self) -> list[RuntimeDependencyStatus]:
        return [self._status_for(dependency) for dependency in self.list()]

    def get(self, dependency_id: str) -> RuntimeDependency:
        for dependency in self.list():
            if dependency.id == dependency_id:
                return dependency
        raise RuntimeDependencyNotFound(dependency_id)

    def path_for(self, dependency: RuntimeDependency) -> Path:
        return (self.runtime_root / dependency.relative_path).resolve()

    def _status_for(self, dependency: RuntimeDependency) -> RuntimeDependencyStatus:
        path = self.path_for(dependency)
        status = "missing"
        if self._path_is_dir_safely(path):
            status = "installed"
            if self._requires_post_install(dependency) and not self._path_is_file_safely(
                self.post_install_marker_for(dependency)
            ):
                status = "post_install_pending"
        return RuntimeDependencyStatus(
            id=dependency.id,
            label=dependency.label,
            type=dependency.type,
            provider=dependency.provider,
            status=status,
            path=path,
            source_url=dependency.url,
            ref=dependency.ref,
            features=dependency.features,
            requirements_file=dependency.requirements_file,
            install_script=dependency.install_script,
            pip_packages=dependency.pip_packages,
            auto_install=dependency.auto_install,
            release_class=dependency.release_class,
            size_bytes=dependency.size_bytes,
            checksum_sha256=dependency.checksum_sha256,
            license_id=dependency.license_id,
            license_url=dependency.license_url,
            repair_policy=dependency.repair_policy,
            source_verified_at=dependency.source_verified_at,
            source_provenance_url=dependency.source_provenance_url,
        )

    def post_install_marker_for(self, dependency: RuntimeDependency) -> Path:
        return self.path_for(dependency) / ".asteria-postinstall-ok"

    def _path_is_dir_safely(self, path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    def _path_is_file_safely(self, path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False

    def _requires_post_install(self, dependency: RuntimeDependency) -> bool:
        return bool(dependency.requirements_file or dependency.install_script or dependency.pip_packages)

    def _parse_dependency(self, item: dict) -> RuntimeDependency:
        source = item["source"]
        target = item["target"]
        post_install = item.get("post_install") or {}
        if not isinstance(source, dict) or not isinstance(target, dict) or not isinstance(post_install, dict):
            raise ValueError("runtime dependency nested sections must be objects")
        required_values = [
            item["id"],
            item["label"],
            item["type"],
            source["provider"],
            source["url"],
            target["relative_path"],
        ]
        if not all(isinstance(value, str) for value in required_values):
            raise ValueError("runtime dependency required fields must be strings")
        ref = source.get("ref")
        if ref is not None and not isinstance(ref, str):
            raise ValueError("runtime dependency source ref must be a string")
        features = item.get("features", [])
        if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
            raise ValueError("runtime dependency features must be a list of strings")
        requirements_file = post_install.get("requirements_file")
        if requirements_file is not None and not isinstance(requirements_file, str):
            raise ValueError("runtime dependency requirements_file must be a string")
        install_script = post_install.get("install_script")
        if install_script is not None and not isinstance(install_script, str):
            raise ValueError("runtime dependency install_script must be a string")
        pip_packages = post_install.get("pip_packages", [])
        if not isinstance(pip_packages, list) or not all(isinstance(package, str) for package in pip_packages):
            raise ValueError("runtime dependency pip_packages must be a list of strings")
        auto_install = item.get("auto_install", False)
        if not isinstance(auto_install, bool):
            raise ValueError("runtime dependency auto_install must be a boolean")
        release_class = item.get("release_class", "development")
        if release_class not in {"development", "mandatory-feature", "user-owned"}:
            raise ValueError("runtime dependency release_class is invalid")
        size_bytes = item.get("size_bytes")
        if size_bytes is not None and (not isinstance(size_bytes, int) or isinstance(size_bytes, bool)):
            raise ValueError("runtime dependency size_bytes must be an integer")
        checksum_sha256 = item.get("checksum_sha256")
        if checksum_sha256 is not None and not isinstance(checksum_sha256, str):
            raise ValueError("runtime dependency checksum_sha256 must be a string")
        license_info = item.get("license") or {}
        if not isinstance(license_info, dict):
            raise ValueError("runtime dependency license must be an object")
        license_id = license_info.get("id")
        license_url = license_info.get("url")
        if license_id is not None and not isinstance(license_id, str):
            raise ValueError("runtime dependency license id must be a string")
        if license_url is not None and not isinstance(license_url, str):
            raise ValueError("runtime dependency license url must be a string")
        repair_policy = item.get("repair_policy")
        if repair_policy is not None and not isinstance(repair_policy, str):
            raise ValueError("runtime dependency repair_policy must be a string")
        source_verified_at = source.get("verified_at")
        source_provenance_url = source.get("provenance_url")
        if source_verified_at is not None and not isinstance(source_verified_at, str):
            raise ValueError("runtime dependency verified_at must be a string")
        if source_provenance_url is not None and not isinstance(source_provenance_url, str):
            raise ValueError("runtime dependency provenance_url must be a string")
        return RuntimeDependency(
            id=item["id"],
            label=item["label"],
            type=item["type"],
            provider=source["provider"],
            url=source["url"],
            ref=ref,
            relative_path=self._parse_relative_path(target["relative_path"]),
            features=features,
            requirements_file=requirements_file,
            install_script=install_script,
            pip_packages=pip_packages,
            auto_install=auto_install,
            release_class=release_class,
            size_bytes=size_bytes,
            checksum_sha256=checksum_sha256,
            license_id=license_id,
            license_url=license_url,
            repair_policy=repair_policy,
            source_verified_at=source_verified_at,
            source_provenance_url=source_provenance_url,
        )

    def _parse_relative_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime dependency target must stay within runtime root")
        return path


class RuntimeDependencyInstaller:
    def __init__(
        self,
        repository: RuntimeDependencyRepository,
        runner: Callable[[list[str], Path], subprocess.CompletedProcess] | None = None,
        python_path: Path | None = None,
        managed_artifact_service: ManagedArtifactService | None = None,
        constraints_path: Path | None = None,
    ) -> None:
        self.repository = repository
        self.runner = runner or self._run
        self.python_path = python_path
        self.managed_artifact_service = managed_artifact_service
        self.constraints_path = constraints_path

    def install(self, dependency_id: str) -> RuntimeDependencyStatus:
        dependency = self.repository.get(dependency_id)
        if dependency.provider == "archive":
            return self._install_managed(dependency)
        current = self.repository._status_for(dependency)
        if current.status == "installed":
            return current
        if dependency.provider != "git":
            raise RuntimeDependencyError(f"Unsupported runtime dependency provider: {dependency.provider}")

        target = self.repository.path_for(dependency)
        if not self._target_exists(target, dependency.id):
            self._prepare_target_parent(target, dependency.id)
            command = ["git", "clone", dependency.url, str(target)]
            if dependency.ref:
                command = ["git", "clone", "--branch", dependency.ref, "--depth", "1", dependency.url, str(target)]
            result = self._run_command(command, target.parent, dependency.id)
            if result.returncode != 0:
                stderr = result.stderr if isinstance(result.stderr, str) else ""
                raise RuntimeDependencyError(stderr.strip() or f"Failed to install runtime dependency: {dependency.id}")
        self._run_post_install(dependency, target)
        return self.repository._status_for(dependency)

    def update(self, dependency_id: str) -> RuntimeDependencyStatus:
        dependency = self.repository.get(dependency_id)
        if dependency.provider == "archive":
            return self._install_managed(dependency)
        if dependency.provider != "git":
            raise RuntimeDependencyError(f"Unsupported runtime dependency provider: {dependency.provider}")
        target = self.repository.path_for(dependency)
        if not self._target_exists(target, dependency.id):
            return self.install(dependency_id)
        result = self._run_command(["git", "-C", str(target), "pull", "--ff-only"], target, dependency.id)
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            raise RuntimeDependencyError(stderr.strip() or f"Failed to update runtime dependency: {dependency.id}")
        self._run_post_install(dependency, target)
        return self.repository._status_for(dependency)

    def repair(self, dependency_id: str) -> RuntimeDependencyStatus:
        dependency = self.repository.get(dependency_id)
        if dependency.provider == "archive":
            return self._install_managed(dependency, force_post_install=True)
        target = self.repository.path_for(dependency)
        if not self._target_exists(target, dependency.id):
            return self.install(dependency_id)
        self._run_post_install(dependency, target)
        return self.repository._status_for(dependency)

    def _install_managed(
        self,
        dependency: RuntimeDependency,
        *,
        force_post_install: bool = False,
    ) -> RuntimeDependencyStatus:
        if dependency.size_bytes is None or dependency.checksum_sha256 is None:
            raise RuntimeDependencyError(f"Managed runtime metadata is incomplete: {dependency.id}")
        service = self.managed_artifact_service or ManagedArtifactService(
            self.repository.runtime_root.parent.parent / "temp"
        )
        spec = ManagedArtifactSpec(
            manifest_version=1,
            component_id=dependency.id,
            source_url=dependency.url,
            destination=self.repository.path_for(dependency),
            expected_size=dependency.size_bytes,
            checksum_sha256=dependency.checksum_sha256,
            kind="zip",
            strip_single_root=True,
        )
        try:
            service.acquire(spec)
            target = self.repository.path_for(dependency)
            current = self.repository._status_for(dependency)
            if current.status == "installed" and not force_post_install:
                service.mark_post_install_complete(spec)
                return current
            self._run_post_install(dependency, target)
            status = self.repository._status_for(dependency)
            if status.status == "installed":
                service.mark_post_install_complete(spec)
            return status
        except (ManagedArtifactError, OSError) as exc:
            message = str(exc) or f"Failed to install runtime dependency: {dependency.id}"
            raise RuntimeDependencyError(message) from exc

    def _run(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def _target_exists(self, target: Path, dependency_id: str) -> bool:
        try:
            exists = target.exists()
        except OSError as exc:
            message = str(exc) or f"Failed to inspect runtime dependency target: {dependency_id}"
            raise RuntimeDependencyError(message) from exc
        if not exists:
            return False
        try:
            is_dir = target.is_dir()
        except OSError as exc:
            message = str(exc) or f"Failed to inspect runtime dependency target: {dependency_id}"
            raise RuntimeDependencyError(message) from exc
        if not is_dir:
            raise RuntimeDependencyError(f"Runtime dependency target is not a directory: {target}")
        return True

    def _prepare_target_parent(self, target: Path, dependency_id: str) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            message = str(exc) or f"Failed to prepare runtime dependency target directory: {dependency_id}"
            raise RuntimeDependencyError(message) from exc

    def _run_post_install(self, dependency: RuntimeDependency, target: Path) -> None:
        python_path = self.python_path
        if python_path is None:
            return
        if not self.repository._requires_post_install(dependency):
            return
        if dependency.requirements_file:
            requirements = target / dependency.requirements_file
            if self._post_install_file_exists(requirements, dependency.id):
                command = [str(python_path), "-m", "pip", "install"]
                if self.constraints_path is not None:
                    command.extend(("-c", str(self.constraints_path)))
                command.extend(("-r", str(requirements)))
                self._run_checked(command, target, dependency.id)
        if dependency.pip_packages:
            command = [str(python_path), "-m", "pip", "install"]
            if self.constraints_path is not None:
                command.extend(("-c", str(self.constraints_path)))
            command.extend(dependency.pip_packages)
            self._run_checked(command, target, dependency.id)
        if dependency.install_script:
            script = target / dependency.install_script
            if self._post_install_file_exists(script, dependency.id):
                self._run_checked([str(python_path), str(script)], target, dependency.id)
        marker = self.repository.post_install_marker_for(dependency)
        try:
            marker.write_text("ok\n", encoding="utf-8")
        except OSError as exc:
            message = str(exc) or f"Failed to write runtime dependency post-install marker: {dependency.id}"
            raise RuntimeDependencyError(message) from exc

    def _post_install_file_exists(self, path: Path, dependency_id: str) -> bool:
        try:
            exists = path.exists()
        except OSError as exc:
            message = str(exc) or f"Failed to inspect runtime dependency post-install file: {dependency_id}"
            raise RuntimeDependencyError(message) from exc
        if not exists:
            return False
        try:
            return path.is_file()
        except OSError as exc:
            message = str(exc) or f"Failed to inspect runtime dependency post-install file: {dependency_id}"
            raise RuntimeDependencyError(message) from exc

    def _run_checked(self, command: list[str], cwd: Path, dependency_id: str) -> None:
        result = self._run_command(command, cwd, dependency_id)
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            raise RuntimeDependencyError(stderr.strip() or f"Failed post-install for runtime dependency: {dependency_id}")

    def _run_command(self, command: list[str], cwd: Path, dependency_id: str) -> subprocess.CompletedProcess:
        try:
            return self.runner(command, cwd)
        except OSError as exc:
            message = str(exc) or f"Failed to run runtime dependency command: {dependency_id}"
            raise RuntimeDependencyError(message) from exc
