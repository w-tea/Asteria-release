from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


_KEY_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SEMANTIC_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")


class ReleaseSignatureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    key_id: str
    product_version: str
    canonical_sha256: str


def canonical_manifest_bytes(payload: Any) -> bytes:
    normalized = _canonical_value(payload)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def verify_signed_manifest(
    payload: dict[str, Any],
    envelope: dict[str, Any],
    *,
    trusted_public_keys: dict[str, bytes],
    current_data_application_version: str | None = None,
    coordinated_data_restore: bool = False,
) -> VerifiedManifest:
    if envelope.get("format_version") != 1 or envelope.get("algorithm") != "ed25519":
        raise ReleaseSignatureError("signature_format_unsupported")
    key_id = envelope.get("key_id")
    if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
        raise ReleaseSignatureError("unknown_signing_key")
    public_bytes = trusted_public_keys.get(key_id)
    if public_bytes is None:
        raise ReleaseSignatureError("unknown_signing_key")
    signature_text = envelope.get("signature")
    if not isinstance(signature_text, str):
        raise ReleaseSignatureError("signature_invalid")
    try:
        signature = base64.b64decode(signature_text, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
        canonical = canonical_manifest_bytes(payload)
        public_key.verify(signature, canonical)
    except (ValueError, binascii.Error, InvalidSignature) as exc:
        raise ReleaseSignatureError("signature_invalid") from exc

    product = payload.get("product")
    if not isinstance(product, dict) or product.get("id") != "asteria":
        raise ReleaseSignatureError("manifest_product_invalid")
    product_version = product.get("version")
    if not isinstance(product_version, str) or _parse_version(product_version) is None:
        raise ReleaseSignatureError("manifest_product_version_invalid")
    _validate_manifest_paths(payload)
    if (
        current_data_application_version is not None
        and not coordinated_data_restore
        and _version_is_older(product_version, current_data_application_version)
    ):
        raise ReleaseSignatureError("application_downgrade_refused")
    return VerifiedManifest(
        key_id=key_id,
        product_version=product_version,
        canonical_sha256=sha256(canonical).hexdigest(),
    )


def verify_artifact_hashes(payload: dict[str, Any], artifact_root: Path) -> None:
    root = artifact_root.absolute()
    files = payload.get("files")
    if not isinstance(files, list):
        raise ReleaseSignatureError("manifest_files_invalid")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseSignatureError("manifest_files_invalid")
        relative = _safe_relative_path(item.get("relative_path"))
        key = relative.as_posix().casefold()
        if key in seen:
            raise ReleaseSignatureError("manifest_files_invalid")
        seen.add(key)
        expected_size = item.get("size")
        expected_hash = item.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        ):
            raise ReleaseSignatureError("manifest_files_invalid")
        path = root / relative
        if not path.is_file() or _is_link(path):
            raise ReleaseSignatureError("artifact_missing")
        if path.stat().st_size != expected_size or _hash_file(path) != expected_hash:
            raise ReleaseSignatureError("artifact_hash_mismatch")


def load_trusted_public_keys(path: Path) -> dict[str, bytes]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseSignatureError("trusted_keys_invalid") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ReleaseSignatureError("trusted_keys_invalid")
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ReleaseSignatureError("trusted_keys_invalid")
    keys: dict[str, bytes] = {}
    for item in raw_keys:
        if not isinstance(item, dict):
            raise ReleaseSignatureError("trusted_keys_invalid")
        key_id = item.get("key_id")
        public_key = item.get("public_key")
        if (
            not isinstance(key_id, str)
            or not _KEY_ID.fullmatch(key_id)
            or key_id in keys
            or not isinstance(public_key, str)
        ):
            raise ReleaseSignatureError("trusted_keys_invalid")
        try:
            decoded = base64.b64decode(public_key, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ReleaseSignatureError("trusted_keys_invalid") from exc
        if len(decoded) != 32:
            raise ReleaseSignatureError("trusted_keys_invalid")
        keys[key_id] = decoded
    return keys


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ReleaseSignatureError("canonical_value_invalid")
        return {key: _canonical_value(item) for key, item in value.items()}
    raise ReleaseSignatureError("canonical_value_invalid")


def _validate_manifest_paths(payload: dict[str, Any]) -> None:
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise ReleaseSignatureError("manifest_files_invalid")
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseSignatureError("manifest_files_invalid")
        _safe_relative_path(item.get("relative_path"))


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ReleaseSignatureError("manifest_path_invalid")
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if (
        not normalized
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ReleaseSignatureError("manifest_path_invalid")
    return posix


def _parse_version(value: str) -> tuple[int, int, int] | None:
    match = _SEMANTIC_VERSION.fullmatch(value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _version_is_older(candidate: str, current: str) -> bool:
    candidate_version = _parse_version(candidate)
    current_version = _parse_version(current)
    if candidate_version is None or current_version is None:
        raise ReleaseSignatureError("manifest_product_version_invalid")
    return candidate_version < current_version


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    try:
        is_junction = getattr(os.path, "isjunction", None)
        return path.is_symlink() or bool(is_junction is not None and is_junction(path))
    except OSError as exc:
        raise ReleaseSignatureError("artifact_path_unreadable") from exc
