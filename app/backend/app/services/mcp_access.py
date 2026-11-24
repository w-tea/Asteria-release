"""Opt-in local MCP credential, protected by the current Windows account."""
from __future__ import annotations

from pathlib import Path
import secrets
from threading import RLock

from pydantic import BaseModel

from app.services.windows_secrets import WindowsSecretStore, _dpapi_protect, _dpapi_unprotect


class LocalMcpStatus(BaseModel):
    enabled: bool


_access_lock = RLock()


class LocalMcpAccess:
    def __init__(self, secret_root: Path):
        self.path = secret_root / 'mcp-local.dpapi'

    def token(self) -> str | None:
        with _access_lock:
            try:
                encrypted = self.path.read_bytes()
            except FileNotFoundError:
                return None
            return _dpapi_unprotect(encrypted).decode('ascii')

    def status(self) -> LocalMcpStatus:
        return LocalMcpStatus(enabled=self.token() is not None)

    def set_enabled(self, enabled: bool) -> LocalMcpStatus:
        with _access_lock:
            if not enabled:
                self.path.unlink(missing_ok=True)
            elif not self.path.exists():
                # Enabling twice keeps existing connections; disable/re-enable revokes them.
                value = secrets.token_urlsafe(32).encode('ascii')
                WindowsSecretStore._write_atomic(self.path, _dpapi_protect(value))
            return self.status()
