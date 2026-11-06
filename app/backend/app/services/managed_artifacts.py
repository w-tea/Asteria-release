from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
import errno
from hashlib import sha256
from http.client import HTTPException, IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
from tempfile import mkdtemp
import time
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from uuid import uuid4
import zipfile

from app.services.managed_payload_guard import copy_preserved_file, hold_payload_read


_COMPONENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


def _validate_https_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("managed artifact URL must use HTTPS without credentials")


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_https_url(str(newurl))
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _strict_https_open(request: Request, *, timeout: float):
    return build_opener(_HTTPSOnlyRedirectHandler()).open(request, timeout=timeout)


class ManagedArtifactError(Exception):
    pass


class ManagedArtifactCancelled(ManagedArtifactError):
    pass


class ManagedArtifactPathTooLong(ManagedArtifactError):
    error_code = "managed_artifact_path_too_long"


class ManagedArtifactRollbackIncomplete(ManagedArtifactError):
    error_code = "managed_artifact_rollback_incomplete"


@dataclass(frozen=True)
class ManagedArtifactSpec:
    manifest_version: int
    component_id: str
    source_url: str
    destination: Path
    expected_size: int | None
    checksum_sha256: str | None
    kind: str
    strip_single_root: bool = False
    allow_unverified: bool = False
    require_https: bool = False
    max_size: int | None = None

    def __post_init__(self) -> None:
        if self.manifest_version <= 0:
            raise ValueError("managed artifact manifest version must be positive")
        if _COMPONENT_ID_RE.fullmatch(self.component_id) is None:
            raise ValueError("managed artifact component id is invalid")
        if not self.source_url:
            raise ValueError("managed artifact source URL is missing")
        if self.expected_size is not None and self.expected_size <= 0:
            raise ValueError("managed artifact size must be positive")
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("managed artifact bounded maximum must be positive")
        if self.max_size is not None and self.expected_size is not None and self.expected_size > self.max_size:
            raise ValueError("managed artifact size exceeds bounded maximum")
        if self.checksum_sha256 is not None and _SHA256_RE.fullmatch(self.checksum_sha256) is None:
            raise ValueError("managed artifact checksum must be lowercase SHA-256")
        if self.checksum_sha256 is None and not self.allow_unverified:
            raise ValueError("managed artifact checksum is required")
        if self.allow_unverified and self.checksum_sha256 is not None:
            raise ValueError("unverified managed artifact cannot include an expected checksum")
        if self.require_https:
            _validate_https_url(self.source_url)
        if self.kind not in {"file", "zip"}:
            raise ValueError("managed artifact kind is unsupported")

    @property
    def size_limit(self) -> int:
        return self.max_size or self.expected_size or 512 * 1024 * 1024


@dataclass(frozen=True)
class ManagedArtifactResult:
    status: str
    destination: Path
    receipt_path: Path
    downloaded_bytes: int
    checksum_sha256: str = ""


class ManagedArtifactService:
    def __init__(
        self,
        temp_root: Path,
        *,
        timeout_seconds: float = 60.0,
        opener: Callable[..., BinaryIO] | None = None,
        clock: Callable[[], float] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        before_publish: Callable[[], object] | None = None,
        resume_ttl_seconds: float = 7 * 24 * 60 * 60,
    ) -> None:
        self.temp_root = self._absolute_without_resolving(temp_root)
        self.timeout_seconds = timeout_seconds
        self.opener = opener
        self.clock = clock or time.time
        self.progress_callback = progress_callback
        self.cancel_requested = cancel_requested or (lambda: False)
        self.before_publish = before_publish
        self.resume_ttl_seconds = resume_ttl_seconds

    def acquire(self, spec: ManagedArtifactSpec, *, verify_contents: bool = False) -> ManagedArtifactResult:
        destination = self._absolute_without_resolving(spec.destination)
        self._validate_managed_write_path(destination)
        self._validate_destination_type(spec, destination)
        if spec.kind == "zip":
            self._recover_pending_zip_publish(spec, destination)
            self._validate_destination_type(spec, destination)
        receipt_path = self.receipt_path_for(spec)
        if self._installed_receipt_matches(spec, destination, receipt_path) and (
            not verify_contents or self.verify_installed_contents(spec)
        ):
            return ManagedArtifactResult(
                status="already-installed",
                destination=destination,
                receipt_path=receipt_path,
                downloaded_bytes=0,
                checksum_sha256=spec.checksum_sha256 or "",
            )

        repaired = self._path_exists(destination)
        part_path = self.part_path_for(spec)
        resume_path = self.resume_receipt_path_for(spec)
        effective_spec = spec
        try:
            downloaded_bytes = self._download(spec, part_path, resume_path)
            actual_hash = self._validate_download(spec, part_path)
            if spec.checksum_sha256 is None or spec.expected_size is None:
                effective_spec = replace(
                    spec,
                    expected_size=downloaded_bytes,
                    checksum_sha256=spec.checksum_sha256 or actual_hash,
                    allow_unverified=False,
                )
            self._validate_managed_write_path(destination)
            receipt = self._receipt(effective_spec)
            if self.before_publish is not None:
                try:
                    self.before_publish()
                except Exception as exc:
                    raise ManagedArtifactError(
                        f"managed artifact pre-publish preparation failed: {exc}"
                    ) from exc
            if spec.kind == "zip":
                self._publish_zip(effective_spec, part_path, destination, receipt)
            else:
                self._publish_file(part_path, destination)
                self._validate_managed_write_path(receipt_path)
                self._write_json_atomic(receipt_path, receipt)
        except ManagedArtifactError:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise ManagedArtifactError(f"managed artifact publish failed: {exc}") from exc
        finally:
            if self._installed_receipt_matches(effective_spec, destination, receipt_path):
                self._remove_file_safely(part_path)
                self._remove_file_safely(resume_path)

        return ManagedArtifactResult(
            status="repaired" if repaired else "installed",
            destination=destination,
            receipt_path=receipt_path,
            downloaded_bytes=downloaded_bytes,
            checksum_sha256=effective_spec.checksum_sha256 or "",
        )

    def part_path_for(self, spec: ManagedArtifactSpec) -> Path:
        return self.temp_root / "managed-artifacts" / f"{spec.component_id}.part"

    def resume_receipt_path_for(self, spec: ManagedArtifactSpec) -> Path:
        return self.temp_root / "managed-artifacts" / f"{spec.component_id}.resume.json"

    def rollback_receipt_path_for(self, spec: ManagedArtifactSpec) -> Path:
        return self.temp_root / "managed-artifacts" / f"{spec.component_id}.rollback.json"

    def receipt_path_for(self, spec: ManagedArtifactSpec) -> Path:
        destination = self._absolute_without_resolving(spec.destination)
        if spec.kind == "zip":
            return destination / ".asteria-managed.json"
        return destination.with_name(f"{destination.name}.asteria-managed.json")

    def discard_partial(self, spec: ManagedArtifactSpec) -> None:
        self._remove_file_safely(self.part_path_for(spec))
        self._remove_file_safely(self.resume_receipt_path_for(spec))

    def has_valid_partial(self, spec: ManagedArtifactSpec) -> bool:
        return self._resume_offset(
            spec,
            self.part_path_for(spec),
            self.resume_receipt_path_for(spec),
        ) > 0

    def valid_partial_size(self, spec: ManagedArtifactSpec) -> int:
        """Inspect resumable state without deleting it."""
        part_path = self.part_path_for(spec)
        resume_path = self.resume_receipt_path_for(spec)
        self._validate_managed_write_path(part_path)
        self._validate_managed_write_path(resume_path)
        if not self._path_is_file(part_path):
            return 0
        size = self._file_size(part_path)
        try:
            receipt_age = max(0.0, self.clock() - resume_path.stat().st_mtime)
        except OSError:
            return 0
        if (
            size <= 0
            or size >= spec.size_limit
            or receipt_age > self.resume_ttl_seconds
            or self._read_json(resume_path) != self._resume_receipt(spec)
        ):
            return 0
        return size

    def is_installed(self, spec: ManagedArtifactSpec, *, require_post_install: bool = False) -> bool:
        destination = self._absolute_without_resolving(spec.destination)
        self._validate_managed_write_path(destination)
        receipt_path = self.receipt_path_for(spec)
        if not self._installed_receipt_matches(spec, destination, receipt_path):
            return False
        if not require_post_install:
            return True
        receipt = self._read_json(receipt_path)
        expected = self._completed_receipt(spec)
        return isinstance(receipt, dict) and all(
            receipt.get(key) == value for key, value in expected.items()
        )

    def verify_installed_contents(self, spec: ManagedArtifactSpec) -> bool:
        """Deep-check a managed artifact against its installed-content receipt."""
        if not self.is_installed(spec):
            return False
        if spec.kind == "zip":
            receipt = self._read_json(self.receipt_path_for(spec))
            if not isinstance(receipt, dict) or not isinstance(receipt.get("installed_tree"), list):
                return False
            try:
                expected_tree = receipt["installed_tree"]
                if not expected_tree or not all(isinstance(item, dict) for item in expected_tree):
                    return False
                root = self._absolute_without_resolving(spec.destination)
                seen: set[str] = set()
                for item in expected_tree:
                    kind = item.get("kind")
                    relative = item.get("path")
                    if kind not in {"file", "directory"} or not isinstance(relative, str):
                        return False
                    parts = PurePosixPath(relative).parts
                    windows_path = PureWindowsPath(relative)
                    if (
                        not parts
                        or relative.startswith(("/", "//"))
                        or windows_path.is_absolute()
                        or windows_path.drive
                        or any(part in {"", ".", ".."} or self._unsafe_windows_part(part) for part in parts)
                    ):
                        return False
                    key = relative.casefold()
                    if key in seen:
                        return False
                    seen.add(key)
                    target = root.joinpath(*parts)
                    self._validate_managed_write_path(target)
                    if kind == "directory":
                        if set(item) != {"kind", "path"} or not self._path_is_dir(target):
                            return False
                        continue
                    size_bytes = item.get("size_bytes")
                    checksum = item.get("sha256")
                    if (
                        set(item) != {"kind", "path", "size_bytes", "sha256"}
                        or not isinstance(size_bytes, int)
                        or size_bytes < 0
                        or not isinstance(checksum, str)
                        or _SHA256_RE.fullmatch(checksum) is None
                        or not self._path_is_file(target)
                        or self._file_size(target) != size_bytes
                        or self._sha256(target) != checksum
                    ):
                        return False
                return True
            except (ManagedArtifactError, OSError):
                return False
        try:
            return self._sha256(self._absolute_without_resolving(spec.destination)) == spec.checksum_sha256
        except OSError:
            return False

    def invalidate_install_receipt(self, spec: ManagedArtifactSpec) -> None:
        receipt_path = self.receipt_path_for(spec)
        self._validate_managed_write_path(receipt_path)
        self._remove_file_safely(receipt_path)

    def mark_post_install_complete(self, spec: ManagedArtifactSpec) -> None:
        self._validate_managed_write_path(self._absolute_without_resolving(spec.destination))
        receipt_path = self.receipt_path_for(spec)
        receipt = self._read_json(receipt_path)
        expected = self._receipt(spec)
        comparable = dict(receipt) if isinstance(receipt, dict) else {}
        comparable["post_install_state"] = "pending"
        if not all(comparable.get(key) == value for key, value in expected.items()):
            raise ManagedArtifactError("managed artifact receipt identity is invalid")
        completed = dict(receipt) if isinstance(receipt, dict) else self._receipt(spec)
        completed["post_install_state"] = "complete"
        self._validate_managed_write_path(receipt_path)
        self._write_json_atomic(receipt_path, completed)

    def _download(self, spec: ManagedArtifactSpec, part_path: Path, resume_path: Path) -> int:
        self._validate_managed_write_path(part_path)
        self._validate_managed_write_path(resume_path)
        try:
            part_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ManagedArtifactError(f"managed artifact download directory failed: {exc}") from exc

        offset = self._resume_offset(spec, part_path, resume_path)
        self._write_json_atomic(resume_path, self._resume_receipt(spec))
        headers = {"User-Agent": "Asteria/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(spec.source_url, headers=headers)
        try:
            opener = self.opener or (_strict_https_open if spec.require_https else urlopen)
            response = opener(request, timeout=self.timeout_seconds)
            with response:
                if spec.require_https:
                    final_url = response.geturl() if hasattr(response, "geturl") else spec.source_url
                    try:
                        _validate_https_url(str(final_url))
                    except ValueError as exc:
                        raise ManagedArtifactError(
                            "managed artifact redirected URL must remain HTTPS without credentials"
                        ) from exc
                status = getattr(response, "status", None) or response.getcode()
                append = offset > 0 and status == 206 and self._valid_content_range(response, offset)
                if status not in {200, 206}:
                    raise ManagedArtifactError(f"managed artifact download returned HTTP {status}")
                mode = "ab" if append else "wb"
                written = offset if append else 0
                received = 0
                self._report_progress(spec, written)
                with part_path.open(mode) as output:
                    while True:
                        if self.cancel_requested():
                            raise ManagedArtifactCancelled("managed artifact download cancelled")
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
                        received += len(chunk)
                        self._report_progress(spec, written)
                        if written > spec.size_limit:
                            raise ManagedArtifactError("managed artifact size mismatch")
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and received != int(content_length):
                    raise ManagedArtifactError("managed artifact download interrupted")
                if spec.expected_size is not None and written < spec.expected_size:
                    raise ManagedArtifactError("managed artifact download interrupted")
                if written <= 0:
                    raise ManagedArtifactError("managed artifact download is empty")
                return written
        except ManagedArtifactError:
            if self._file_size(part_path) > spec.size_limit:
                self._remove_file_safely(part_path)
                self._remove_file_safely(resume_path)
            raise
        except (HTTPError, URLError, OSError, HTTPException, IncompleteRead, ValueError) as exc:
            raise ManagedArtifactError(f"managed artifact download failed: {exc}") from exc

    def _report_progress(self, spec: ManagedArtifactSpec, downloaded_bytes: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                spec.component_id,
                downloaded_bytes,
                spec.expected_size or spec.size_limit,
            )

    def _resume_offset(self, spec: ManagedArtifactSpec, part_path: Path, resume_path: Path) -> int:
        size = self.valid_partial_size(spec)
        if size <= 0:
            self._remove_file_safely(part_path)
            self._remove_file_safely(resume_path)
            return 0
        return size

    def _validate_destination_type(self, spec: ManagedArtifactSpec, destination: Path) -> None:
        try:
            if destination.is_symlink():
                raise ManagedArtifactError("managed artifact destination cannot be a symlink")
            if not destination.exists():
                return
            if spec.kind == "zip" and not destination.is_dir():
                raise ManagedArtifactError("managed artifact archive destination must be a directory")
            if spec.kind == "file" and not destination.is_file():
                raise ManagedArtifactError("managed artifact file destination must be a file")
        except ManagedArtifactError:
            raise
        except OSError as exc:
            if self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(f"managed artifact destination check failed: {exc}") from exc

    @staticmethod
    def _absolute_without_resolving(path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    def _validate_managed_write_path(self, path: Path) -> None:
        candidate = self._absolute_without_resolving(path)
        for component in reversed((candidate, *candidate.parents)):
            try:
                details = os.lstat(component)
            except FileNotFoundError:
                continue
            except OSError as exc:
                if self._is_path_too_long_error(exc):
                    raise ManagedArtifactPathTooLong(
                        "managed artifact installation path is too long"
                    ) from exc
                raise ManagedArtifactError("managed artifact destination could not be inspected safely") from exc
            attributes = getattr(details, "st_file_attributes", 0)
            is_reparse = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            is_junction = False
            junction_check = getattr(os.path, "isjunction", None)
            if junction_check is not None:
                try:
                    is_junction = bool(junction_check(component))
                except OSError as exc:
                    if self._is_path_too_long_error(exc):
                        raise ManagedArtifactPathTooLong(
                            "managed artifact installation path is too long"
                        ) from exc
                    raise ManagedArtifactError("managed artifact destination could not be inspected safely") from exc
            if stat.S_ISLNK(details.st_mode) or is_junction or is_reparse:
                raise ManagedArtifactError(
                    "managed artifact destination uses a filesystem link or reparse point"
                )

    def _validate_download(self, spec: ManagedArtifactSpec, part_path: Path) -> str:
        actual_size = self._file_size(part_path)
        if actual_size <= 0 or actual_size > spec.size_limit or (
            spec.expected_size is not None and actual_size != spec.expected_size
        ):
            self._remove_file_safely(part_path)
            self._remove_file_safely(self.resume_receipt_path_for(spec))
            raise ManagedArtifactError("managed artifact size mismatch")
        try:
            actual_hash = self._sha256(part_path)
        except OSError as exc:
            raise ManagedArtifactError(f"managed artifact checksum failed: {exc}") from exc
        if spec.checksum_sha256 is not None and actual_hash != spec.checksum_sha256:
            self._remove_file_safely(part_path)
            self._remove_file_safely(self.resume_receipt_path_for(spec))
            raise ManagedArtifactError("managed artifact checksum mismatch")
        return actual_hash

    def _publish_file(self, part_path: Path, destination: Path) -> None:
        self._validate_managed_write_path(destination)
        self._require_same_volume(part_path, destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._validate_managed_write_path(destination)
            os.replace(part_path, destination)
        except OSError as exc:
            if self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(f"managed artifact atomic publish failed: {exc}") from exc

    def _publish_zip(
        self,
        spec: ManagedArtifactSpec,
        part_path: Path,
        destination: Path,
        receipt: dict[str, object],
    ) -> None:
        self._validate_managed_write_path(destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(f"managed artifact archive destination failed: {exc}") from exc
        publish_root = self._create_publish_workspace()
        stage = publish_root / "s"
        backup = publish_root / "b"
        rollback_receipt_path = self.rollback_receipt_path_for(spec)
        preserve_publish_root = False
        payload_locks = ExitStack()
        verify_payload: Callable[[], None] | None = None
        dispose_payload: Callable[[], None] | None = None
        disposal_started = False
        self._require_same_volume(part_path, stage)
        try:
            self._validate_managed_write_path(destination)
            self._validate_managed_write_path(stage)
            self._validate_managed_write_path(backup)
            self._extract_zip(part_path, stage, strip_single_root=spec.strip_single_root)
            receipt["installed_tree"] = self._installed_tree(stage)
            self._write_json_atomic(stage / ".asteria-managed.json", receipt)
            self._write_json_atomic(
                rollback_receipt_path,
                self._rollback_receipt(spec, publish_root.name),
            )
            self._validate_managed_write_path(destination)
            if self._path_exists(destination):
                self._validate_managed_write_path(destination)
                self._validate_managed_write_path(backup)
                os.replace(destination, backup)
            if spec.component_id == "comfyui-core" and self._path_exists(backup):
                # Park the old namespace before copying. Writes to its public
                # paths cannot silently succeed against an obsolete copy.
                verify_payload, dispose_payload = self._preserve_comfy_payload(
                    backup, stage, receipt["installed_tree"], protection=payload_locks,
                )
            self._validate_managed_write_path(destination)
            self._validate_managed_write_path(stage)
            if verify_payload is not None:
                verify_payload()
                if self._path_exists(destination):
                    raise ManagedArtifactError("ComfyUI destination changed during the update")
                pending = self._rollback_receipt(spec, publish_root.name)
                pending["preservation_pending"] = True
                self._write_json_atomic(rollback_receipt_path, pending)
            os.replace(stage, destination)
            if verify_payload is not None:
                try:
                    verify_payload()
                except (ManagedArtifactError, OSError):
                    # Do not let the next acquire treat the installed receipt
                    # as permission to delete a newer, conflicting backup.
                    preserve_publish_root = True
                    raise ManagedArtifactRollbackIncomplete(
                        "ComfyUI payload changed during publication; previous component backup was preserved"
                    )
            if dispose_payload is not None:
                disposing = self._rollback_receipt(spec, publish_root.name)
                disposing["preservation_disposal"] = True
                self._write_json_atomic(rollback_receipt_path, disposing)
                disposal_started = True
                dispose_payload()
            else:
                self._remove_path_safely(backup)
            self._remove_file_safely(rollback_receipt_path)
        except (ManagedArtifactError, OSError, zipfile.BadZipFile) as exc:
            # Directory handles must be closed before moving the parked tree
            # back to its public name on Windows. No backup is deleted first.
            payload_locks.close()
            self._remove_path_safely(stage)
            if disposal_started and self._path_exists(backup):
                preserve_publish_root = True
                raise ManagedArtifactRollbackIncomplete(
                    "ComfyUI preserved data needs recovery; update backup was retained"
                ) from exc
            if self._path_exists(backup) and not self._restore_zip_backup(backup, destination):
                preserve_publish_root = True
                raise ManagedArtifactRollbackIncomplete(
                    "managed artifact rollback is incomplete; previous component backup was preserved"
                ) from exc
            self._remove_file_safely(rollback_receipt_path)
            self._remove_path_safely(backup)
            if isinstance(exc, ManagedArtifactError):
                self._remove_file_safely(part_path)
                self._remove_file_safely(self.resume_receipt_path_for(spec))
                raise
            if isinstance(exc, OSError) and self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(f"managed artifact archive publish failed: {exc}") from exc
        finally:
            payload_locks.close()
            if not preserve_publish_root and not self._path_exists(backup):
                if spec.component_id == "comfyui-core":
                    # Never recursively collect an unobserved late entry.
                    self._remove_empty_publish_workspace(publish_root)
                else:
                    self._remove_path_safely(publish_root)

    def _preserve_comfy_payload(
        self, destination: Path, stage: Path, new_tree: list[dict[str, object]],
        *, protection: ExitStack,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        """Stage non-core payload before atomic replacement, outside core ownership."""
        def inventory() -> dict:
            rows, pending = {}, [destination]
            while pending:
                current = pending.pop()
                self._validate_managed_write_path(current)
                details = current.stat(follow_symlinks=False)
                relative = current.relative_to(destination).as_posix()
                rows[relative] = (details.st_ino, details.st_mode, details.st_size, details.st_mtime_ns)
                if stat.S_ISDIR(details.st_mode):
                    pending.extend(current.iterdir())
            return rows

        baseline = inventory()
        pins = []
        pins_by_path = {}
        for relative, details in baseline.items():
            pin = protection.enter_context(hold_payload_read(
                destination / relative, directory=stat.S_ISDIR(details[1]),
            ))
            pins.append((relative, details, pin))
            pins_by_path[relative] = pin

        def verify() -> None:
            if inventory() != baseline:
                raise ManagedArtifactError("preserved ComfyUI payload changed during the update")

        def dispose() -> None:
            # Delete opened original objects, never a subsequently replaced
            # pathname. Directory disposition is nonrecursive and fails if an
            # unobserved child has arrived, retaining it with the receipt.
            files = [row for row in pins if not stat.S_ISDIR(row[1][1])]
            directories = sorted((row for row in pins if stat.S_ISDIR(row[1][1])),
                                 key=lambda row: len(PurePosixPath(row[0]).parts), reverse=True)
            for _relative, _details, pin in [*files, *directories]:
                pin.delete_original()

        verify()
        previous = None
        receipt_pin = pins_by_path.get(".asteria-managed.json")
        if receipt_pin is not None:
            try:
                with receipt_pin.reader() as source_receipt:
                    previous = json.load(source_receipt)
            except (OSError, ValueError):
                pass
        old_files: set[str] = set()
        old_directories: set[str] = set()
        old_tree = previous.get("installed_tree") if isinstance(previous, dict) else None
        if isinstance(previous, dict) and previous.get("component_id") == "comfyui-core" and isinstance(old_tree, list):
            old_files = {
                item["path"].casefold()
                for item in old_tree
                if isinstance(item, dict) and item.get("kind") == "file" and isinstance(item.get("path"), str)
            }
            old_directories = {
                item["path"].casefold()
                for item in old_tree
                if isinstance(item, dict) and item.get("kind") == "directory" and isinstance(item.get("path"), str)
            }
        new_entries = {str(item["path"]).casefold(): item for item in new_tree}

        def visit(source: Path) -> None:
            self._validate_managed_write_path(source)
            relative = source.relative_to(destination).as_posix()
            key = relative.casefold()
            existing = new_entries.get(key)
            target = stage / (str(existing["path"]) if existing else relative)
            self._validate_managed_write_path(target)
            details = source.stat(follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                if existing and existing["kind"] != "directory":
                    raise ManagedArtifactError("preserved ComfyUI payload conflicts with the core archive")
                children = sorted(source.iterdir(), key=lambda path: path.name.casefold())
                if not children and key not in old_directories:
                    target.mkdir(parents=True, exist_ok=True)
                for child in children:
                    visit(child)
                return
            if not stat.S_ISREG(details.st_mode):
                raise ManagedArtifactError("preserved ComfyUI payload contains an unsupported entry")
            if key in old_files:
                return
            if existing:
                if existing["kind"] != "file" or pins_by_path[relative].sha256() != existing["sha256"]:
                    raise ManagedArtifactError("preserved ComfyUI payload conflicts with the core archive")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_preserved_file(source, target, pins_by_path[relative])
            self._validate_managed_write_path(source)
            after = source.stat(follow_symlinks=False)
            if ((details.st_ino, details.st_size, details.st_mtime_ns)
                    != (after.st_ino, after.st_size, after.st_mtime_ns)
                    or pins_by_path[relative].sha256() != self._sha256(target)):
                raise ManagedArtifactError("preserved ComfyUI payload changed during the update")

        for source in sorted(destination.iterdir(), key=lambda path: path.name.casefold()):
            if source.name.casefold() != ".asteria-managed.json":
                visit(source)
        verify()
        return verify, dispose

    def _restore_zip_backup(self, backup: Path, destination: Path) -> bool:
        if self._path_exists(destination):
            return False
        try:
            self._validate_managed_write_path(backup)
            self._validate_managed_write_path(destination)
            os.replace(backup, destination)
        except (ManagedArtifactError, OSError):
            return False
        return True

    def _recover_pending_zip_publish(self, spec: ManagedArtifactSpec, destination: Path) -> None:
        rollback_receipt_path = self.rollback_receipt_path_for(spec)
        if not self._path_is_file(rollback_receipt_path):
            return
        self._validate_managed_write_path(rollback_receipt_path)
        state = self._read_json(rollback_receipt_path)
        workspace_name = state.get("workspace") if isinstance(state, dict) else None
        preservation_phase = None
        if isinstance(workspace_name, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", workspace_name):
            for phase in ("preservation_pending", "preservation_disposal"):
                if state == {**self._rollback_receipt(spec, workspace_name), phase: True}:
                    preservation_phase = phase
                    state = self._rollback_receipt(spec, workspace_name)
        if (
            not isinstance(workspace_name, str)
            or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", workspace_name) is None
            or state != self._rollback_receipt(spec, workspace_name)
        ):
            raise ManagedArtifactRollbackIncomplete(
                "managed artifact rollback state is invalid; previous component backup was preserved"
            )
        publish_root = self.temp_root / workspace_name
        if publish_root.parent != self.temp_root:
            raise ManagedArtifactRollbackIncomplete(
                "managed artifact rollback state is invalid; previous component backup was preserved"
            )
        self._validate_managed_write_path(publish_root)
        backup = publish_root / "b"
        if not self._path_exists(backup):
            if spec.component_id == "comfyui-core":
                self._remove_empty_publish_workspace(publish_root)
            else:
                self._remove_path_safely(publish_root)
            self._remove_file_safely(rollback_receipt_path)
            return
        if self._path_exists(destination):
            if spec.component_id == "comfyui-core":
                raise ManagedArtifactRollbackIncomplete(
                    "ComfyUI preserved data needs recovery; update backup was retained"
                )
            if self._installed_receipt_matches(
                spec,
                destination,
                destination / ".asteria-managed.json",
            ):
                self._remove_path_safely(publish_root)
                self._remove_file_safely(rollback_receipt_path)
                return
            raise ManagedArtifactRollbackIncomplete(
                "managed artifact rollback is incomplete; previous component backup was preserved"
            )
        if preservation_phase == "preservation_disposal":
            raise ManagedArtifactRollbackIncomplete(
                "ComfyUI preserved data needs recovery; update backup was retained"
            )
        if not self._restore_zip_backup(backup, destination):
            raise ManagedArtifactRollbackIncomplete(
                "managed artifact rollback is incomplete; previous component backup was preserved"
            )
        self._remove_path_safely(publish_root)
        self._remove_file_safely(rollback_receipt_path)

    def _create_publish_workspace(self) -> Path:
        self._validate_managed_write_path(self.temp_root)
        try:
            self.temp_root.mkdir(parents=True, exist_ok=True)
            self._validate_managed_write_path(self.temp_root)
            workspace = Path(mkdtemp(prefix="", dir=self.temp_root))
            self._validate_managed_write_path(workspace)
            return workspace
        except OSError as exc:
            if self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(
                f"managed artifact publish workspace failed: {exc}"
            ) from exc

    def _extract_zip(self, archive_path: Path, stage: Path, *, strip_single_root: bool) -> None:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                entries = self._validated_zip_entries(archive, strip_single_root=strip_single_root)
                stage.mkdir(parents=True, exist_ok=False)
                resolved_stage = stage.resolve()
                for info, parts in entries:
                    if not parts:
                        continue
                    target = stage.joinpath(*parts)
                    resolved_target = target.resolve()
                    if not resolved_target.is_relative_to(resolved_stage):
                        raise ManagedArtifactError("managed artifact archive escaped staging root")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=_CHUNK_SIZE)
        except ManagedArtifactError:
            raise
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            if isinstance(exc, OSError) and self._is_path_too_long_error(exc):
                raise ManagedArtifactPathTooLong(
                    "managed artifact installation path is too long"
                ) from exc
            raise ManagedArtifactError(f"managed artifact archive extraction failed: {exc}") from exc

    @staticmethod
    def _is_path_too_long_error(exc: OSError) -> bool:
        return exc.errno == errno.ENAMETOOLONG or getattr(exc, "winerror", None) == 206

    def _validated_zip_entries(
        self,
        archive: zipfile.ZipFile,
        *,
        strip_single_root: bool,
    ) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
        validated: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
        roots: set[str] = set()
        seen: set[str] = set()
        for info in archive.infolist():
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ManagedArtifactError("managed artifact archive contains a symlink")
            normalized = info.filename.replace("\\", "/")
            windows_path = PureWindowsPath(normalized)
            posix_path = PurePosixPath(normalized)
            parts = tuple(part for part in posix_path.parts if part not in {"", ".", "/"})
            if (
                normalized.startswith(("/", "//"))
                or windows_path.is_absolute()
                or windows_path.drive
                or posix_path.is_absolute()
                or ".." in parts
                or any(self._unsafe_windows_part(part) for part in parts)
                or not parts
            ):
                raise ManagedArtifactError("managed artifact archive path is unsafe")
            roots.add(parts[0].casefold())
            output_parts = parts[1:] if strip_single_root else parts
            if output_parts:
                key = "/".join(part.casefold() for part in output_parts)
                if key in seen:
                    raise ManagedArtifactError("managed artifact archive has duplicate paths")
                seen.add(key)
            validated.append((info, output_parts))
        if not validated:
            raise ManagedArtifactError("managed artifact archive is empty")
        if strip_single_root and len(roots) != 1:
            raise ManagedArtifactError("managed artifact archive root is invalid")
        if strip_single_root and not any(parts for _info, parts in validated):
            raise ManagedArtifactError("managed artifact archive root has no content")
        return validated

    @staticmethod
    def _unsafe_windows_part(part: str) -> bool:
        if ":" in part or part.endswith((" ", ".")):
            return True
        stem = part.split(".", 1)[0].casefold()
        return stem in {
            "con",
            "prn",
            "aux",
            "nul",
            *(f"com{index}" for index in range(1, 10)),
            *(f"lpt{index}" for index in range(1, 10)),
        }

    def _installed_receipt_matches(self, spec: ManagedArtifactSpec, destination: Path, receipt_path: Path) -> bool:
        if spec.checksum_sha256 is None or spec.expected_size is None:
            return False
        if spec.kind == "zip":
            if not self._path_is_dir(destination):
                return False
        elif not self._path_is_file(destination) or self._file_size(destination) != spec.expected_size:
            return False
        receipt = self._read_json(receipt_path)
        if not isinstance(receipt, dict):
            return False
        expected = self._receipt(spec)
        return all(receipt.get(key) == value for key, value in expected.items()) or all(
            receipt.get(key) == value for key, value in self._completed_receipt(spec).items()
        )

    def _installed_tree(self, root: Path) -> list[dict[str, object]]:
        self._validate_managed_write_path(root)
        records: list[dict[str, object]] = []
        seen: set[str] = set()

        def visit(directory: Path) -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
            except OSError as exc:
                raise ManagedArtifactError("managed artifact installed tree could not be inspected") from exc
            for entry in entries:
                path = Path(entry.path)
                if directory == root and entry.name == ".asteria-managed.json":
                    continue
                self._validate_managed_write_path(path)
                relative = path.relative_to(root).as_posix()
                key = relative.casefold()
                if key in seen:
                    raise ManagedArtifactError("managed artifact installed tree has duplicate paths")
                seen.add(key)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        records.append({"kind": "directory", "path": relative})
                        visit(path)
                    elif entry.is_file(follow_symlinks=False):
                        records.append(
                            {
                                "kind": "file",
                                "path": relative,
                                "size_bytes": entry.stat(follow_symlinks=False).st_size,
                                "sha256": self._sha256(path),
                            }
                        )
                    else:
                        raise ManagedArtifactError("managed artifact installed tree contains an unsupported entry")
                except OSError as exc:
                    raise ManagedArtifactError("managed artifact installed tree could not be inspected") from exc

        visit(root)
        return records

    def _receipt(self, spec: ManagedArtifactSpec) -> dict[str, object]:
        if spec.checksum_sha256 is None or spec.expected_size is None:
            raise ManagedArtifactError("managed artifact receipt requires verified size and checksum")
        return {
            "manifest_version": spec.manifest_version,
            "component_id": spec.component_id,
            "bytes": spec.expected_size,
            "sha256": spec.checksum_sha256,
            "post_install_state": "pending",
        }

    def _completed_receipt(self, spec: ManagedArtifactSpec) -> dict[str, object]:
        receipt = self._receipt(spec)
        receipt["post_install_state"] = "complete"
        return receipt

    def _resume_receipt(self, spec: ManagedArtifactSpec) -> dict[str, object]:
        receipt: dict[str, object] = {
            "manifest_version": spec.manifest_version,
            "component_id": spec.component_id,
            "source_url": spec.source_url,
            "expected_size": spec.expected_size,
            "checksum_sha256": spec.checksum_sha256,
        }
        if spec.allow_unverified:
            receipt["allow_unverified"] = True
        if spec.require_https:
            receipt["require_https"] = True
        if spec.expected_size is None:
            receipt["max_size"] = spec.size_limit
        return receipt

    @staticmethod
    def _rollback_receipt(spec: ManagedArtifactSpec, workspace_name: str) -> dict[str, object]:
        return {
            "state_version": 1,
            "component_id": spec.component_id,
            "manifest_version": spec.manifest_version,
            "expected_size": spec.expected_size,
            "checksum_sha256": spec.checksum_sha256,
            "workspace": workspace_name,
        }

    @staticmethod
    def _valid_content_range(response: BinaryIO, offset: int) -> bool:
        header = response.headers.get("Content-Range", "")
        return header.startswith(f"bytes {offset}-")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_json_atomic(self, path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            self._validate_managed_write_path(path)
            self._validate_managed_write_path(temporary)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._validate_managed_write_path(path)
            self._validate_managed_write_path(temporary)
            temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            self._validate_managed_write_path(path)
            self._validate_managed_write_path(temporary)
            os.replace(temporary, path)
        except ManagedArtifactError:
            self._remove_file_safely(temporary)
            raise
        except OSError as exc:
            self._remove_file_safely(temporary)
            raise ManagedArtifactError(f"managed artifact receipt write failed: {exc}") from exc

    @staticmethod
    def _require_same_volume(source: Path, target: Path) -> None:
        source_drive = source.resolve().drive.casefold()
        target_drive = target.resolve().drive.casefold()
        if source_drive != target_drive:
            raise ManagedArtifactError("managed artifact atomic publish requires one volume")

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            return path.exists()
        except OSError:
            return False

    @staticmethod
    def _path_is_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            return False

    @staticmethod
    def _path_is_dir(path: Path) -> bool:
        try:
            return path.is_dir()
        except OSError:
            return False

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return -1

    def _remove_file_safely(self, path: Path) -> None:
        try:
            self._validate_managed_write_path(path)
            if path.is_file() or path.is_symlink():
                self._validate_managed_write_path(path)
                path.unlink()
        except (ManagedArtifactError, OSError):
            pass

    def _remove_empty_publish_workspace(self, path: Path) -> None:
        self._validate_managed_write_path(path)
        try:
            path.rmdir()
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST} and getattr(exc, "winerror", None) != 145:
                raise
            # A pre-publication interruption can leave a complete stage but
            # no backup. Retain it, including unknown entries, and let the
            # next acquisition use a new workspace instead of blocking it.

    def _remove_path_safely(self, path: Path) -> None:
        try:
            self._validate_managed_write_path(path)
            if path.is_symlink() or path.is_file():
                self._validate_managed_write_path(path)
                path.unlink()
            elif path.is_dir():
                self._validate_managed_write_path(path)
                shutil.rmtree(path)
        except (ManagedArtifactError, OSError):
            pass
