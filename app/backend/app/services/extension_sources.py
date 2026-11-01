from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from hashlib import sha256
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import time
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4

from app.services.managed_artifacts import (
    ManagedArtifactCancelled,
    ManagedArtifactError,
    ManagedArtifactService,
    ManagedArtifactSpec,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class ExtensionSourceError(ValueError):
    pass


class ExtensionSourceCancelled(RuntimeError):
    pass


class SourceKind(StrEnum):
    LOCAL_ZIP = "local_zip"
    HTTPS_ZIP = "https_zip"
    GITHUB_RELEASE_ZIP = "github_release_zip"
    GIT = "git"
    MANUAL_FOLDER = "manual_folder"


class SourceTrust(StrEnum):
    CATALOG_VERIFIED = "catalog_verified"
    USER_CHECKSUM = "user_checksum"
    UNVERIFIED_DIRECT = "unverified_direct"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ExtensionSourceDescriptor:
    kind: SourceKind
    reference: str
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    trust: SourceTrust = SourceTrust.UNVERIFIED_DIRECT
    confirmed_unverified: bool = False
    resolved_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.reference:
            raise ExtensionSourceError("extension source reference is missing")
        if self.expected_sha256 is not None and _SHA256_RE.fullmatch(self.expected_sha256) is None:
            raise ExtensionSourceError("expected checksum is invalid")
        if self.expected_size_bytes is not None and self.expected_size_bytes <= 0:
            raise ExtensionSourceError("expected size must be positive")
        if self.trust is SourceTrust.CATALOG_VERIFIED:
            if self.expected_sha256 is None or self.expected_size_bytes is None:
                raise ExtensionSourceError("catalog source requires checksum and size")
        elif self.trust is SourceTrust.USER_CHECKSUM:
            if self.expected_sha256 is None:
                raise ExtensionSourceError("user-checksum source requires checksum")
        elif self.trust is SourceTrust.UNVERIFIED_DIRECT:
            if self.expected_sha256 is not None:
                raise ExtensionSourceError("unverified source cannot claim an expected checksum")
            if not self.confirmed_unverified:
                raise ExtensionSourceError("explicit unverified direct confirmation is required")
        elif self.trust is SourceTrust.MANUAL:
            if self.kind is not SourceKind.MANUAL_FOLDER or self.expected_sha256 is not None:
                raise ExtensionSourceError("manual trust is only valid for a folder reference")
        if self.kind in {SourceKind.HTTPS_ZIP, SourceKind.GITHUB_RELEASE_ZIP, SourceKind.GIT}:
            _validate_https_url(self.reference)
        if self.resolved_commit is not None and (
            self.kind is not SourceKind.GIT or _COMMIT_RE.fullmatch(self.resolved_commit) is None
        ):
            raise ExtensionSourceError("resolved Git commit is invalid")

    def with_resolved_commit(self, commit: str) -> ExtensionSourceDescriptor:
        normalized = commit.strip().lower()
        if _COMMIT_RE.fullmatch(normalized) is None:
            raise ExtensionSourceError("resolved Git commit is invalid")
        return replace(self, resolved_commit=normalized)

    def to_snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["trust"] = self.trust.value
        return payload

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> ExtensionSourceDescriptor:
        try:
            expected_keys = {
                "kind",
                "reference",
                "expected_sha256",
                "expected_size_bytes",
                "trust",
                "confirmed_unverified",
                "resolved_commit",
            }
            if set(payload) != expected_keys or not isinstance(payload.get("confirmed_unverified"), bool):
                raise ValueError
            size_value = payload.get("expected_size_bytes")
            if isinstance(size_value, bool):
                raise ValueError
            return cls(
                kind=SourceKind(str(payload["kind"])),
                reference=str(payload["reference"]),
                expected_sha256=str(payload["expected_sha256"]) if payload.get("expected_sha256") is not None else None,
                expected_size_bytes=int(payload["expected_size_bytes"]) if payload.get("expected_size_bytes") is not None else None,
                trust=SourceTrust(str(payload["trust"])),
                confirmed_unverified=payload.get("confirmed_unverified") is True,
                resolved_commit=str(payload["resolved_commit"]) if payload.get("resolved_commit") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtensionSourceError("extension source snapshot is invalid") from exc


def describe_extension_source(
    reference: str | Path,
    *,
    expected_sha256: str | None = None,
    catalog_expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    trust: SourceTrust | str | None = None,
    confirmed_unverified: bool = False,
) -> ExtensionSourceDescriptor:
    reference_text = str(reference)
    parsed = urlparse(reference_text)
    windows_path = PureWindowsPath(reference_text)
    if isinstance(reference, Path) or not parsed.scheme or windows_path.is_absolute():
        path = Path(reference).absolute()
        if path.is_dir():
            if any(value is not None for value in (expected_sha256, catalog_expected_sha256, expected_size_bytes)):
                raise ExtensionSourceError("manual folder cannot include package integrity fields")
            if trust is not None and SourceTrust(trust) is not SourceTrust.MANUAL:
                raise ExtensionSourceError("manual folder requires manual trust")
            return ExtensionSourceDescriptor(
                kind=SourceKind.MANUAL_FOLDER,
                reference=str(path),
                trust=SourceTrust.MANUAL,
            )
        if not path.is_file() or path.suffix.casefold() != ".zip":
            raise ExtensionSourceError("local extension source must be an existing ZIP or folder")
        if _is_reparse(path):
            raise ExtensionSourceError("local extension ZIP cannot be a filesystem link or reparse point")
        expected_size_bytes = expected_size_bytes if expected_size_bytes is not None else path.stat().st_size
        kind = SourceKind.LOCAL_ZIP
    else:
        _validate_https_url(reference_text)
        path_part = parsed.path.casefold()
        path_segments = tuple(segment for segment in parsed.path.split("/") if segment)
        is_github_repository = (
            parsed.hostname is not None
            and parsed.hostname.casefold() == "github.com"
            and len(path_segments) == 2
            and "/releases/" not in path_part
        )
        kind = (
            SourceKind.GIT
            if path_part.endswith(".git") or is_github_repository
            else SourceKind.GITHUB_RELEASE_ZIP
            if parsed.hostname is not None and parsed.hostname.casefold() == "github.com" and "/releases/" in path_part
            else SourceKind.HTTPS_ZIP
        )
    if catalog_expected_sha256 is not None:
        if _SHA256_RE.fullmatch(catalog_expected_sha256) is None:
            raise ExtensionSourceError("catalog checksum is invalid")
        if expected_sha256 != catalog_expected_sha256:
            raise ExtensionSourceError("catalog and request checksum disagree")
    selected_trust = SourceTrust(
        trust
        or (SourceTrust.CATALOG_VERIFIED if catalog_expected_sha256 else SourceTrust.USER_CHECKSUM if expected_sha256 else SourceTrust.UNVERIFIED_DIRECT)
    )
    return ExtensionSourceDescriptor(
        kind=kind,
        reference=reference_text if parsed.scheme else str(Path(reference).absolute()),
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        trust=selected_trust,
        confirmed_unverified=confirmed_unverified,
    )


def download_remote_source(
    source: ExtensionSourceDescriptor,
    destination: Path,
    *,
    temp_root: Path,
    opener: Callable[..., Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> str:
    if source.kind not in {SourceKind.HTTPS_ZIP, SourceKind.GITHUB_RELEASE_ZIP}:
        raise ExtensionSourceError("source is not a remote ZIP")
    component_id = f"extension-{sha256(source.reference.encode('utf-8')).hexdigest()[:24]}"
    spec = ManagedArtifactSpec(
        manifest_version=1,
        component_id=component_id,
        source_url=source.reference,
        destination=destination,
        expected_size=source.expected_size_bytes,
        checksum_sha256=source.expected_sha256,
        kind="file",
        allow_unverified=source.trust is SourceTrust.UNVERIFIED_DIRECT,
        require_https=True,
        max_size=512 * 1024 * 1024,
    )
    service = ManagedArtifactService(
        temp_root,
        opener=opener,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
    )
    try:
        result = service.acquire(spec)
        return result.checksum_sha256
    except ManagedArtifactCancelled:
        raise
    except (ManagedArtifactError, OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        service.receipt_path_for(spec).unlink(missing_ok=True)
        message = str(exc).casefold()
        if "https" in message or "credential" in message:
            reason = "remote redirect must remain HTTPS without credentials"
        elif "checksum" in message:
            reason = "remote source checksum mismatch"
        elif "size" in message:
            reason = "remote source size mismatch"
        else:
            reason = "remote source download failed"
        raise ExtensionSourceError(reason) from exc


class SubprocessGitRunner:
    def __init__(
        self,
        *,
        run: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Git runner bounds must be positive")
        self._run = run or subprocess.run
        self._injected_run = run is not None
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds

    def export_archive(
        self,
        url: str,
        destination: Path,
        *,
        revision: str | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> str:
        _validate_https_url(url)
        if revision is not None and _COMMIT_RE.fullmatch(revision) is None:
            raise ExtensionSourceError("requested Git commit is invalid")
        requested_revision = revision
        cancel_requested = cancel_requested or (lambda: False)
        checkout = destination.parent / f".git-source-{uuid4().hex}"
        try:
            self._execute(
                ("git", "clone", "--filter=blob:none", "--no-checkout", "--", url, str(checkout)),
                cancel_requested=cancel_requested,
            )
            revision = self._execute(
                (
                    "git",
                    "-C",
                    str(checkout),
                    "rev-parse",
                    f"{revision}^{{commit}}" if revision is not None else "HEAD",
                ),
                cancel_requested=cancel_requested,
            ).stdout.strip().lower()
            if _COMMIT_RE.fullmatch(revision) is None:
                raise ExtensionSourceError("Git source returned an invalid immutable commit")
            if requested_revision is not None and revision != requested_revision:
                raise ExtensionSourceError("Git source commit does not match immutable request")
            self._execute(
                ("git", "-C", str(checkout), "archive", "--format=zip", f"--output={destination}", revision),
                cancel_requested=cancel_requested,
            )
            return revision
        except ExtensionSourceCancelled:
            raise
        except ExtensionSourceError:
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExtensionSourceError("Git source could not be exported") from exc
        finally:
            shutil.rmtree(checkout, ignore_errors=True)

    def _execute(
        self,
        command: tuple[str, ...],
        *,
        cancel_requested: Callable[[], bool],
    ) -> Any:
        if cancel_requested():
            raise ExtensionSourceCancelled("Git source operation cancelled")
        if self._injected_run:
            result = self._run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if cancel_requested():
                raise ExtensionSourceCancelled("Git source operation cancelled")
            return result
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if cancel_requested():
                _terminate_process(process)
                raise ExtensionSourceCancelled("Git source operation cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                raise ExtensionSourceError("Git source operation timed out")
            try:
                stdout, stderr = process.communicate(timeout=min(self.poll_seconds, remaining))
            except subprocess.TimeoutExpired:
                continue
            if process.returncode != 0:
                raise ExtensionSourceError("Git source command failed")
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _validate_https_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ExtensionSourceError("source URL must use HTTPS without credentials")


def _is_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
        is_junction = bool(getattr(os.path, "isjunction", lambda _path: False)(path))
        return path.is_symlink() or is_junction or bool(getattr(details, "st_file_attributes", 0) & 0x400)
    except OSError as exc:
        raise ExtensionSourceError("local extension source could not be inspected") from exc


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        process.terminate()
        process.communicate(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
