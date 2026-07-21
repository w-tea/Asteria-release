from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4


_ENVIRONMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_EXACT_REQUIREMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s<>=!*~]+$")


class PackageEnvironmentError(Exception):
    pass


@dataclass(frozen=True)
class PackageEnvironmentResult:
    status: str
    receipt_path: Path
    lock_sha256: str


class PackageEnvironmentManager:
    def __init__(
        self,
        app_root: Path,
        *,
        runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.app_root = app_root.resolve()
        self.runner = runner or self._run

    def receipt_path(self, environment_id: str) -> Path:
        self._validate_environment_id(environment_id)
        return self.app_root / "runtime" / ".asteria-packages" / f"{environment_id}.json"

    def install(
        self,
        environment_id: str,
        python_path: Path,
        lock_path: Path,
        *,
        index_url: str | None = None,
        no_dependencies: bool = False,
    ) -> PackageEnvironmentResult:
        self._validate_environment_id(environment_id)
        uv = self._contained_file(self.app_root / "bootstrap" / "uv" / "uv.exe", "uv runtime")
        python = self._contained_file(python_path, "Python runtime")
        lock = self._contained_file(lock_path, "package lock")
        lock_bytes = lock.read_bytes()
        self._validate_lock(lock_bytes)
        lock_hash = sha256(lock_bytes).hexdigest()
        receipt_path = self.receipt_path(environment_id)
        expected = {
            "version": 1,
            "environment_id": environment_id,
            "uv": uv.relative_to(self.app_root).as_posix(),
            "python": python.relative_to(self.app_root).as_posix(),
            "lock": lock.relative_to(self.app_root).as_posix(),
            "lock_sha256": lock_hash,
            "index_url": index_url,
            "no_dependencies": no_dependencies,
            "state": "complete",
        }
        if self._read_receipt(receipt_path) == expected:
            return PackageEnvironmentResult("already-installed", receipt_path, lock_hash)

        command = [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--no-python-downloads",
            "--no-config",
            "--color",
            "never",
            "--link-mode",
            "copy",
        ]
        if index_url is not None:
            if not index_url.startswith("https://"):
                raise PackageEnvironmentError("package index must use HTTPS")
            command.extend(("--index-url", index_url))
        if no_dependencies:
            command.append("--no-deps")
        command.extend(("-r", str(lock)))
        try:
            result = self.runner(command, self.app_root)
        except OSError as exc:
            raise PackageEnvironmentError("contained package installation failed") from exc
        if result.returncode != 0:
            raise PackageEnvironmentError("contained package installation failed")
        self._write_receipt(receipt_path, expected)
        return PackageEnvironmentResult("installed", receipt_path, lock_hash)

    def is_installed(
        self,
        environment_id: str,
        python_path: Path,
        lock_path: Path,
        *,
        index_url: str | None = None,
        no_dependencies: bool = False,
    ) -> bool:
        try:
            self._validate_environment_id(environment_id)
            uv = self._contained_file(self.app_root / "bootstrap" / "uv" / "uv.exe", "uv runtime")
            python = self._contained_file(python_path, "Python runtime")
            lock = self._contained_file(lock_path, "package lock")
            lock_bytes = lock.read_bytes()
            self._validate_lock(lock_bytes)
        except (OSError, PackageEnvironmentError):
            return False
        expected = {
            "version": 1,
            "environment_id": environment_id,
            "uv": uv.relative_to(self.app_root).as_posix(),
            "python": python.relative_to(self.app_root).as_posix(),
            "lock": lock.relative_to(self.app_root).as_posix(),
            "lock_sha256": sha256(lock_bytes).hexdigest(),
            "index_url": index_url,
            "no_dependencies": no_dependencies,
            "state": "complete",
        }
        return self._read_receipt(self.receipt_path(environment_id)) == expected

    def _contained_file(self, path: Path, label: str) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.app_root)
        except ValueError as exc:
            raise PackageEnvironmentError(f"{label} must be contained in the application root") from exc
        if not resolved.is_file():
            raise PackageEnvironmentError(f"{label} is missing")
        return resolved

    @staticmethod
    def _validate_lock(payload: bytes) -> None:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise PackageEnvironmentError("package lock must be UTF-8") from exc
        requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if not requirements or any(_EXACT_REQUIREMENT_RE.fullmatch(line) is None for line in requirements):
            raise PackageEnvironmentError("package lock must contain only exact requirements")

    @staticmethod
    def _validate_environment_id(environment_id: str) -> None:
        if _ENVIRONMENT_ID_RE.fullmatch(environment_id) is None:
            raise PackageEnvironmentError("package environment id is invalid")

    @staticmethod
    def _read_receipt(path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_receipt(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
