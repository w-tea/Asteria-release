from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess

from app.services.managed_artifacts import (
    ManagedArtifactError,
    ManagedArtifactPathTooLong,
    ManagedArtifactRollbackIncomplete,
    ManagedArtifactService,
    ManagedArtifactSpec,
)


class ComfyRuntimeCoreError(Exception):
    pass


@dataclass(frozen=True)
class ComfyRuntimeCoreStatus:
    status: str
    path: Path
    repo_url: str
    ref: str
    message: str = ""


class ComfyRuntimeCoreManager:
    def __init__(
        self,
        comfy_dir: Path,
        python_path: Path,
        repo_url: str,
        ref: str = "",
        runner: Callable[[list[str], Path], subprocess.CompletedProcess] | None = None,
        managed_artifact_service: ManagedArtifactService | None = None,
        managed_artifact_spec: ManagedArtifactSpec | None = None,
        install_python_dependencies: bool = True,
    ) -> None:
        self.comfy_dir = comfy_dir
        self.python_path = python_path
        self.repo_url = repo_url
        self.ref = ref
        self.runner = runner or self._run
        self.managed_artifact_service = managed_artifact_service
        self.managed_artifact_spec = managed_artifact_spec
        self.install_python_dependencies = install_python_dependencies

    def status(self) -> ComfyRuntimeCoreStatus:
        try:
            if not self._path_exists(self.comfy_dir):
                return self._status("missing", "ComfyUI directory is missing")
            if not self.comfy_dir.is_dir():
                return self._status("unavailable", "ComfyUI runtime path is not a directory")
            if not self._path_exists(self.comfy_dir / ".git"):
                return self._status("installed", "ComfyUI directory exists but is not a git checkout")
        except OSError:
            return self._status("unavailable", "ComfyUI directory status could not be checked")
        return self._status("installed", "ComfyUI runtime is installed")

    def install(self) -> ComfyRuntimeCoreStatus:
        if self.managed_artifact_spec is not None:
            return self._install_managed()
        if not self._runtime_dir_exists():
            try:
                self.comfy_dir.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ComfyRuntimeCoreError(f"Could not prepare ComfyUI runtime directory: {exc}") from exc
            command = ["git", "clone", self.repo_url, str(self.comfy_dir)]
            if self.ref:
                command = ["git", "clone", "--branch", self.ref, "--depth", "1", self.repo_url, str(self.comfy_dir)]
            self._run_checked(command, self.comfy_dir.parent, "Failed to install ComfyUI runtime")
        self._repair_python_dependencies()
        return self.status()

    def update(self) -> ComfyRuntimeCoreStatus:
        if self.managed_artifact_spec is not None:
            return self._install_managed()
        if not self._runtime_dir_exists():
            return self.install()
        if self._path_exists_or_error(self.comfy_dir / ".git", "Could not inspect ComfyUI runtime directory"):
            self._run_checked(["git", "-C", str(self.comfy_dir), "pull", "--ff-only"], self.comfy_dir, "Failed to update ComfyUI runtime")
        self._repair_python_dependencies()
        return self.status()

    def repair(self) -> ComfyRuntimeCoreStatus:
        if self.managed_artifact_spec is not None:
            return self._install_managed()
        if not self._runtime_dir_exists():
            return self.install()
        self._repair_python_dependencies()
        return self.status()

    def _install_managed(self) -> ComfyRuntimeCoreStatus:
        spec = self.managed_artifact_spec
        if spec is None:
            raise ComfyRuntimeCoreError("Managed ComfyUI artifact is not configured")
        if spec.destination.resolve() != self.comfy_dir.resolve():
            raise ComfyRuntimeCoreError("Managed ComfyUI artifact target does not match runtime path")
        service = self.managed_artifact_service or ManagedArtifactService(
            self.comfy_dir.parent.parent / "temp"
        )
        try:
            service.acquire(spec)
            if self.install_python_dependencies:
                self._repair_python_dependencies()
            service.mark_post_install_complete(spec)
        except (ManagedArtifactPathTooLong, ManagedArtifactRollbackIncomplete):
            raise
        except (ManagedArtifactError, OSError) as exc:
            message = str(exc) or "Failed to install managed ComfyUI runtime"
            raise ComfyRuntimeCoreError(message) from exc
        return self.status()

    def _status(self, status: str, message: str) -> ComfyRuntimeCoreStatus:
        return ComfyRuntimeCoreStatus(
            status=status,
            path=self.comfy_dir,
            repo_url=self.repo_url,
            ref=self.ref,
            message=message,
        )

    def _repair_python_dependencies(self) -> None:
        requirements = self.comfy_dir / "requirements.txt"
        if self._path_is_file_or_error(requirements, "Could not inspect ComfyUI Python dependencies"):
            self._run_checked(
                [str(self.python_path), "-m", "pip", "install", "-r", str(requirements)],
                self.comfy_dir,
                "Failed to repair ComfyUI Python dependencies",
            )

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

    def _run_checked(self, command: list[str], cwd: Path, fallback_message: str) -> None:
        try:
            result = self.runner(command, cwd)
        except OSError as exc:
            raise ComfyRuntimeCoreError(f"{fallback_message}: {exc}") from exc
        if result.returncode != 0:
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            raise ComfyRuntimeCoreError(stderr.strip() or fallback_message)

    def _runtime_dir_exists(self) -> bool:
        if not self._path_exists_or_error(self.comfy_dir, "Could not inspect ComfyUI runtime directory"):
            return False
        if not self._path_is_dir_or_error(self.comfy_dir, "Could not inspect ComfyUI runtime directory"):
            raise ComfyRuntimeCoreError(f"ComfyUI runtime path is not a directory: {self.comfy_dir}")
        return True

    def _path_exists_or_error(self, path: Path, message: str) -> bool:
        try:
            return self._path_exists(path)
        except OSError as exc:
            raise ComfyRuntimeCoreError(f"{message}: {exc}") from exc

    def _path_exists(self, path: Path) -> bool:
        return path.exists()

    def _path_is_dir_or_error(self, path: Path, message: str) -> bool:
        try:
            return path.is_dir()
        except OSError as exc:
            raise ComfyRuntimeCoreError(f"{message}: {exc}") from exc

    def _path_is_file_or_error(self, path: Path, message: str) -> bool:
        if not self._path_exists_or_error(path, message):
            return False
        try:
            return path.is_file()
        except OSError as exc:
            raise ComfyRuntimeCoreError(f"{message}: {exc}") from exc
