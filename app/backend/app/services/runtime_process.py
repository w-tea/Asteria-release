from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import psutil
from pydantic import BaseModel

from app.core.config import AppConfig
from app.services.comfy_model_paths import render_comfy_extra_model_paths
from app.services.runtime import ComfyClient


class RuntimeStatus(BaseModel):
    running: bool
    healthy: bool
    pid: int | None
    base_url: str
    device: str | None = None
    message: str = ""


class RuntimeProcessError(RuntimeError):
    pass


class RuntimeProcessManager:
    def __init__(self, config: AppConfig, client: ComfyClient | None = None) -> None:
        self.config = config
        self.client = client or ComfyClient(config.comfy_base_url)

    def status(self) -> RuntimeStatus:
        pid = self._read_pid()
        health = self.client.system_stats()
        healthy = health is not None
        port_pid = self._pid_from_base_url() if healthy else None
        if port_pid:
            pid = port_pid
        running = self._pid_is_running(pid) if pid else False
        device = self._device_name(health)
        return RuntimeStatus(
            running=running or healthy,
            healthy=healthy,
            pid=pid if running else None,
            base_url=self.config.comfy_base_url,
            device=device,
            message="ready" if healthy else "not reachable",
        )

    def start(self) -> RuntimeStatus:
        current = self.status()
        if current.healthy:
            return current
        try:
            if self.config.comfy_dir.exists() and not self.config.comfy_dir.is_dir():
                raise RuntimeProcessError(f"ComfyUI runtime path is not a directory: {self.config.comfy_dir}")
            self.config.comfy_pid_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_extra_model_paths_config()
        except RuntimeProcessError:
            raise
        except OSError as exc:
            raise RuntimeProcessError(f"Could not prepare ComfyUI runtime files: {exc}") from exc
        stdout_path = self.config.comfy_dir / "comfy.out.log"
        stderr_path = self.config.comfy_dir / "comfy.err.log"
        try:
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    [str(self.config.comfy_python_path), "main.py", *self.config.comfy_args],
                    cwd=self.config.comfy_dir,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=self._creation_flags(),
                )
        except OSError as exc:
            raise RuntimeProcessError(f"Could not start ComfyUI: {exc}") from exc
        try:
            self.config.comfy_pid_path.write_text(str(process.pid), encoding="utf-8")
        except OSError:
            pass
        status = self.status()
        if not status.healthy:
            status.message = f"starting pid {process.pid}"
            status.pid = process.pid
            status.running = True
        return status

    def stop(self) -> RuntimeStatus:
        pid = self._pid_from_base_url() or self._read_pid()
        if pid and self._pid_is_running(pid):
            if pid != os.getpid():
                self._terminate_pid(pid)
        try:
            if self.config.comfy_pid_path.exists():
                self.config.comfy_pid_path.unlink()
        except OSError:
            pass
        return self.status()

    def _read_pid(self) -> int | None:
        try:
            return int(self.config.comfy_pid_path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    def _pid_is_running(self, pid: int) -> bool:
        try:
            return psutil.pid_exists(pid)
        except (psutil.Error, OSError):
            return False

    def _pid_from_base_url(self) -> int | None:
        port = urlparse(self.config.comfy_base_url).port
        if port is None:
            return None
        if os.name == "nt":
            netstat_pid = self._pid_from_netstat(port)
            if netstat_pid:
                return netstat_pid
        try:
            connections = psutil.net_connections(kind="tcp")
        except (psutil.Error, OSError):
            return self._pid_from_netstat(port)
        for connection in connections:
            if connection.status == psutil.CONN_LISTEN and self._connection_port(connection) == port:
                if connection.pid:
                    return connection.pid
        return self._pid_from_netstat(port)

    def _connection_port(self, connection) -> int | None:
        laddr = getattr(connection, "laddr", None)
        if not laddr:
            return None
        port = getattr(laddr, "port", None)
        if isinstance(port, int):
            return port
        if isinstance(laddr, tuple) and len(laddr) >= 2 and isinstance(laddr[1], int):
            return laddr[1]
        return None

    def _pid_from_netstat(self, port: int) -> int | None:
        if os.name != "nt":
            return None
        try:
            result = subprocess.run(["netstat", "-ano"], capture_output=True, text=False, check=False)
        except OSError:
            return None
        stdout = result.stdout or b""
        if isinstance(stdout, bytes):
            encoding = "mbcs" if sys.platform == "win32" else "utf-8"
            stdout = stdout.decode(encoding, errors="replace")
        marker = f":{port}"
        for line in stdout.splitlines():
            if marker not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
        return None

    def _device_name(self, health: dict | None) -> str | None:
        if not health:
            return None
        devices = health.get("devices", [])
        if not devices:
            return None
        name = devices[0].get("name")
        return name if isinstance(name, str) else None

    def _terminate_pid(self, pid: int) -> None:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            return

    def _write_extra_model_paths_config(self) -> None:
        path = self.config.comfy_dir / "extra_model_paths.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_comfy_extra_model_paths(self.config.model_dir), encoding="utf-8")

    def _creation_flags(self) -> int:
        if sys.platform != "win32":
            return 0
        return subprocess.CREATE_NO_WINDOW
