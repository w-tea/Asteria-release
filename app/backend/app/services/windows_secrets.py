from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import tempfile
from typing import Callable

from app.schemas.discord import DiscordSecretStatus


class SecretUnavailableError(RuntimeError):
    pass


class SecretDecryptionError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob_from_bytes(value: bytes) -> tuple[_DataBlob, object]:
    buffer = ctypes.create_string_buffer(value)
    blob = _DataBlob(
        cbData=len(value),
        pbData=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
    )
    return blob, buffer


def _dpapi_protect(value: bytes) -> bytes:
    if os.name != "nt":
        raise SecretUnavailableError("windows_dpapi_unavailable")
    input_blob, input_buffer = _blob_from_bytes(value)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    success = crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "Asteria Discord token",
        None,
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    )
    _ = input_buffer
    if not success:
        raise SecretUnavailableError("secret_encryption_failed")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _dpapi_unprotect(value: bytes) -> bytes:
    if os.name != "nt":
        raise SecretUnavailableError("windows_dpapi_unavailable")
    input_blob, input_buffer = _blob_from_bytes(value)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    success = crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x01,
        ctypes.byref(output_blob),
    )
    _ = input_buffer
    if not success:
        raise SecretDecryptionError("secret_decryption_failed")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


class WindowsSecretStore:
    def __init__(
        self,
        secret_root: Path,
        *,
        protector: Callable[[bytes], bytes] | None = None,
        unprotector: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self.secret_root = secret_root
        self._protect = protector or _dpapi_protect
        self._unprotect = unprotector or _dpapi_unprotect

    @property
    def discord_token_path(self) -> Path:
        return self.secret_root / "discord-token.dpapi"

    def status(self) -> DiscordSecretStatus:
        try:
            configured = self.discord_token_path.is_file()
        except OSError:
            configured = False
        return DiscordSecretStatus(configured=configured)

    def save_discord_token(self, token: str) -> DiscordSecretStatus:
        normalized = token.strip()
        if not normalized or len(normalized) > 4096:
            raise ValueError("discord_token_invalid")
        encrypted = self._protect(normalized.encode("utf-8"))
        if not encrypted or normalized.encode("utf-8") in encrypted:
            raise SecretUnavailableError("secret_encryption_failed")
        self._write_atomic(self.discord_token_path, encrypted)
        return self.status()

    def load_discord_token(self) -> str | None:
        path = self.discord_token_path
        try:
            if not path.is_file():
                return None
            encrypted = path.read_bytes()
        except OSError as exc:
            raise SecretDecryptionError("secret_read_failed") from exc
        try:
            value = self._unprotect(encrypted).decode("utf-8")
        except SecretDecryptionError:
            raise
        except (UnicodeDecodeError, ValueError, OSError) as exc:
            raise SecretDecryptionError("secret_decryption_failed") from exc
        if not value:
            raise SecretDecryptionError("secret_decryption_failed")
        return value

    def delete_discord_token(self) -> DiscordSecretStatus:
        try:
            self.discord_token_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SecretUnavailableError("secret_delete_failed") from exc
        return self.status()

    @staticmethod
    def _write_atomic(path: Path, value: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary.write(value)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise SecretUnavailableError("secret_save_failed") from exc
