"""Core-owned start/stop and pairing controls for the full-GUI LAN listener."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import ipaddress
import logging
import os
import ssl
import subprocess
import sys
import threading
import time

import httpx
import psutil

from app.core.config import AppConfig
from app.services.lan_access import LanAccessRepository, LanListenerSettings
from app.services.lan_runtime import core_identity, local_addresses, validate_lan_host
from app.services.release_layout import launcher_backend_root
from app.services.runtime_ownership import RuntimeOwnershipError


logger = logging.getLogger(__name__)


class LanManagerError(ValueError):
    pass


class LanManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.access = LanAccessRepository(config.data_dir / 'asteria.db')
        self._child: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._closed = False
        self.error: str | None = None

    def is_running(self) -> bool:
        return self._child is not None and self._child.poll() is None

    def status(self) -> dict:
        selected = self.access.listener_settings()
        addresses = []
        for host in local_addresses():
            try:
                ip = ipaddress.ip_address(host)
                if not ip.is_loopback:
                    validate_lan_host(host, assigned={host})
                    addresses.append(host)
            except ValueError:
                continue
        running = self.is_running()
        addresses.sort(key=lambda host: (ipaddress.ip_address(host).is_link_local,
                                        ipaddress.ip_address(host).version, host))
        return {'enabled': self.access.enabled(), 'running': running,
                'settings': selected.model_dump(), 'addresses': addresses,
                'url': selected.origin if running else None, 'error': self.error,
                'devices': [asdict(device) for device in self.access.devices()]}

    def _spawn(self, settings: LanListenerSettings):
        command = [sys.executable, '-B', '-m', 'app.cli.lan_gateway', '--host', settings.host,
                   '--port', str(settings.port)]
        if settings.certificate_path:
            command += ['--cert', settings.certificate_path, '--key', settings.private_key_path]
        env = {**os.environ, 'ASTERIA_ROOT_DIR': str(self.config.root_dir),
               'ASTERIA_DATA_DIR': str(self.config.data_dir),
               'ASTERIA_COMFY_PID_PATH': str(self.config.comfy_pid_path)}
        log_dir = self.config.data_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'lan-{time.time_ns()}.log').open('xb') as log:
            return subprocess.Popen(command, cwd=launcher_backend_root(self.config.root_dir, self.config.target_mode), env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def _wait_ready(self, settings: LanListenerSettings):
        # No credential is sent. A live unauthenticated entry must reject access.
        # For user-supplied local TLS, this probe checks listener availability;
        # browsers still perform their own certificate trust validation.
        with httpx.Client(trust_env=False, timeout=1, verify=False) as client:
            deadline = time.monotonic() + 12
            while self.is_running() and time.monotonic() < deadline:
                try:
                    response = client.get(settings.origin + '/api/health')
                    if response.status_code == 401 and response.json().get('detail', {}).get('code') == 'lan_device_required':
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(.1)
        raise LanManagerError('lan_start_failed')

    def start(self, settings: LanListenerSettings) -> dict:
        with self._lock:
            if self._closed:
                raise LanManagerError('lan_core_unavailable')
            try:
                validate_lan_host(settings.host, assigned=local_addresses())
            except ValueError as exc:
                raise LanManagerError('lan_interface_unavailable') from exc
            try:
                core_identity(self.config)
            except (RuntimeOwnershipError, RuntimeError, KeyError, ValueError) as exc:
                raise LanManagerError('lan_core_unavailable') from exc
            if settings.certificate_path:
                try:
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    # The hidden launcher cannot interactively ask for a key
                    # password. Validate the same certificate/key pair first.
                    context.load_cert_chain(settings.certificate_path, settings.private_key_path, password='')
                except (OSError, ValueError) as exc:
                    raise LanManagerError('lan_certificate_unavailable') from exc
            if self.is_running() and self.access.listener_settings() == settings:
                return self.status()
            self._stop()
            self.access.save_listener_settings(settings)
            self.access.set_enabled(True)
            try:
                self._child = self._spawn(settings)
                self._wait_ready(settings)
                self.error = None
                return self.status()
            except (OSError, subprocess.SubprocessError, LanManagerError) as exc:
                self._stop()
                self.error = 'lan_start_failed'
                raise LanManagerError(self.error) from exc

    def _stop(self):
        self.access.set_enabled(False)
        if self._child is not None:
            try:
                self._child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._child.terminate()  # Only this Popen child, never a port/name search.
                self._child.wait(timeout=5)
            self._child = None

    def stop(self) -> dict:
        with self._lock:
            self._stop()
            self.error = None
            return self.status()

    def close(self):
        with self._lock:
            wanted = self.access.enabled()
            self._closed = True
            self._stop()
            # Preserve the user's choice for a later official Asteria startup.
            self.access.set_enabled(wanted)

    async def restore_after_launcher(self):
        if not self.access.enabled():
            return
        deadline = time.monotonic() + 60
        while not self._closed and time.monotonic() < deadline:
            try:
                owner = core_identity(self.config)
                ancestors = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
                if owner['pid'] in ancestors:
                    break
            except (RuntimeOwnershipError, RuntimeError, ValueError, psutil.Error):
                pass
            await asyncio.sleep(.25)
        else:
            if not self._closed:
                self.error = 'lan_core_unavailable'
                logger.warning('LAN connection was not restored: %s', self.error)
            return
        try:
            await asyncio.to_thread(self.start, self.access.listener_settings())
        except LanManagerError as exc:
            self.error = str(exc)
            logger.warning('LAN connection was not restored: %s', self.error)

    def create_pairing(self) -> dict:
        with self._lock:
            if not self.is_running():
                raise LanManagerError('lan_not_running')
            invitation = self.access.create_pairing()
            return {'url': self.access.listener_settings().origin + '/_asteria/lan/connect#token=' + invitation.token,
                    'expires_at': invitation.expires_at}

    def revoke(self, device_id: str):
        self.access.revoke(device_id)
