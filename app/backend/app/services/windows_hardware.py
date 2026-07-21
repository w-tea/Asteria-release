from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import platform
import re
import shutil
import subprocess
from typing import Protocol


_DRIVER_RE = re.compile(r"^\d+(?:\.\d+)*$")
_SUPPORTED_MACHINES = {"amd64", "x86_64"}


@dataclass(frozen=True)
class HardwareCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class WindowsHardwareStatus:
    supported: bool
    code: str
    minimum_driver: str
    driver_version: str | None = None
    device_name: str | None = None


@dataclass(frozen=True)
class CudaRuntimeStatus:
    supported: bool
    code: str
    device_name: str | None = None


class _TorchModule(Protocol):
    cuda: object

    def empty(self, size: int, *, device: str) -> object: ...


def _run_command(arguments: list[str], timeout: float) -> HardwareCommandResult:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        creationflags=creation_flags,
    )
    return HardwareCommandResult(completed.returncode, completed.stdout, completed.stderr)


def _driver_tuple(value: str) -> tuple[int, ...] | None:
    normalized = value.strip()
    if _DRIVER_RE.fullmatch(normalized) is None:
        return None
    return tuple(int(part) for part in normalized.split("."))


def _driver_at_least(current: str, minimum: str) -> bool:
    current_parts = _driver_tuple(current)
    minimum_parts = _driver_tuple(minimum)
    if current_parts is None or minimum_parts is None:
        return False
    width = max(len(current_parts), len(minimum_parts))
    return current_parts + (0,) * (width - len(current_parts)) >= minimum_parts + (0,) * (width - len(minimum_parts))


class WindowsHardwareProbe:
    def __init__(
        self,
        *,
        minimum_driver: str,
        system_provider: Callable[[], str] = platform.system,
        machine_provider: Callable[[], str] = platform.machine,
        executable_provider: Callable[[str], str | None] = shutil.which,
        command_runner: Callable[[list[str], float], HardwareCommandResult] = _run_command,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.minimum_driver = minimum_driver
        self.system_provider = system_provider
        self.machine_provider = machine_provider
        self.executable_provider = executable_provider
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds

    def inspect(self) -> WindowsHardwareStatus:
        if self.system_provider().casefold() != "windows":
            return self._status("startup_hardware_os_unsupported")
        if self.machine_provider().casefold() not in _SUPPORTED_MACHINES:
            return self._status("startup_hardware_arch_unsupported")
        executable = self.executable_provider("nvidia-smi")
        if not executable:
            return self._status("startup_hardware_probe_missing")
        try:
            result = self.command_runner(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return self._status("startup_hardware_probe_timeout")
        except (OSError, RuntimeError, ValueError):
            return self._status("startup_hardware_probe_failed")
        if result.returncode != 0:
            return self._status("startup_hardware_probe_failed")

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return self._status("startup_hardware_gpu_missing")
        parsed: list[tuple[str, str]] = []
        for line in lines:
            fields = [field.strip() for field in line.rsplit(",", 1)]
            if len(fields) != 2 or not fields[0] or _driver_tuple(fields[1]) is None:
                return self._status("startup_hardware_probe_invalid")
            parsed.append((fields[0], fields[1]))
        nvidia = [(name, driver) for name, driver in parsed if "nvidia" in name.casefold()]
        if not nvidia:
            return self._status("startup_hardware_nvidia_required")
        name, driver = min(nvidia, key=lambda item: _driver_tuple(item[1]) or ())
        if not _driver_at_least(driver, self.minimum_driver):
            return self._status(
                "startup_hardware_driver_outdated",
                driver_version=driver,
                device_name=name,
            )
        return self._status(
            "startup_hardware_ready",
            supported=True,
            driver_version=driver,
            device_name=name,
        )

    def _status(
        self,
        code: str,
        *,
        supported: bool = False,
        driver_version: str | None = None,
        device_name: str | None = None,
    ) -> WindowsHardwareStatus:
        return WindowsHardwareStatus(
            supported=supported,
            code=code,
            minimum_driver=self.minimum_driver,
            driver_version=driver_version,
            device_name=device_name,
        )


def _import_torch() -> _TorchModule:
    import torch

    return torch


def probe_torch_cuda(*, import_torch: Callable[[], _TorchModule] = _import_torch) -> CudaRuntimeStatus:
    try:
        torch = import_torch()
    except (ImportError, OSError, RuntimeError):
        return CudaRuntimeStatus(False, "startup_cuda_torch_missing")
    try:
        cuda = torch.cuda
        if not cuda.is_available():
            return CudaRuntimeStatus(False, "startup_cuda_unavailable")
        device_name = str(cuda.get_device_name(0)).strip() or None
        tensor = torch.empty(1, device="cuda")
        cuda.synchronize()
        del tensor
    except (AttributeError, OSError, RuntimeError, ValueError):
        return CudaRuntimeStatus(False, "startup_cuda_allocation_failed")
    return CudaRuntimeStatus(True, "startup_cuda_ready", device_name=device_name)
