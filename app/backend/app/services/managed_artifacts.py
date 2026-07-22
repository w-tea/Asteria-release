from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException, IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import time
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile


_COMPONENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


class ManagedArtifactError(Exception):
    pass


@dataclass(frozen=True)
class ManagedArtifactSpec:
    manifest_version: int
    component_id: str
    source_url: str
    destination: Path
    expected_size: int
    checksum_sha256: str
    kind: str
    strip_single_root: bool = False

    def __post_init__(self) -> None:
        if self.manifest_version <= 0:
            raise ValueError("managed artifact manifest version must be positive")
        if _COMPONENT_ID_RE.fullmatch(self.component_id) is None:
            raise ValueError("managed artifact component id is invalid")
        if not self.source_url:
            raise ValueError("managed artifact source URL is missing")
        if self.expected_size <= 0:
            raise ValueError("managed artifact size must be positive")
        if _SHA256_RE.fullmatch(self.checksum_sha256) is None:
            raise ValueError("managed artifact checksum must be lowercase SHA-256")
        if self.kind not in {"file", "zip"}:
            raise ValueError("managed artifact kind is unsupported")


@dataclass(frozen=True)
class ManagedArtifactResult:
    status: str
    destination: Path
    receipt_path: Path
    downloaded_bytes: int


class ManagedArtifactService:
    def __init__(
        self,
        temp_root: Path,
        *,
        timeout_seconds: float = 60.0,
        opener: Callable[..., BinaryIO] | None = None,
        clock: Callable[[], float] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.temp_root = self._absolute_without_resolving(temp_root)
        self.timeout_seconds = timeout_seconds
        self.opener = opener or urlopen
        self.clock = clock or time.time
        self.progress_callback = progress_callback

    def acquire(self, spec: ManagedArtifactSpec) -> ManagedArtifactResult:
        destination = self._absolute_without_resolving(spec.destination)
        self._validate_managed_write_path(destination)
        self._validate_destination_type(spec, destination)
        receipt_path = self.receipt_path_for(spec)
        if self._installed_receipt_matches(spec, destination, receipt_path):
            return ManagedArtifactResult(
                status="already-installed",
                destination=destination,
                receipt_path=receipt_path,
                downloaded_bytes=0,
            )

        repaired = self._path_exists(destination)
        part_path = self.part_path_for(spec)
        resume_path = self.resume_receipt_path_for(spec)
        try:
            downloaded_bytes = self._download(spec, part_path, resume_path)
            self._validate_download(spec, part_path)
            self._validate_managed_write_path(destination)
            receipt = self._receipt(spec)
            if spec.kind == "zip":
                self._publish_zip(spec, part_path, destination, receipt)
            else:
                self._publish_file(part_path, destination)
                self._validate_managed_write_path(receipt_path)
                self._write_json_atomic(receipt_path, receipt)
        except ManagedArtifactError:
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            raise ManagedArtifactError(f"managed artifact publish failed: {exc}") from exc
        finally:
            if self._installed_receipt_matches(spec, destination, receipt_path):
                self._remove_file_safely(part_path)
                self._remove_file_safely(resume_path)

        return ManagedArtifactResult(
            status="repaired" if repaired else "installed",
            destination=destination,
            receipt_path=receipt_path,
            downloaded_bytes=downloaded_bytes,
        )

    def part_path_for(self, spec: ManagedArtifactSpec) -> Path:
        return self.temp_root / "managed-artifacts" / f"{spec.component_id}.part"

    def resume_receipt_path_for(self, spec: ManagedArtifactSpec) -> Path:
        return self.temp_root / "managed-artifacts" / f"{spec.component_id}.resume.json"

    def receipt_path_for(self, spec: ManagedArtifactSpec) -> Path:
        destination = self._absolute_without_resolving(spec.destination)
        if spec.kind == "zip":
            return destination / ".asteria-managed.json"
        return destination.with_name(f"{destination.name}.asteria-managed.json")

    def is_installed(self, spec: ManagedArtifactSpec, *, require_post_install: bool = False) -> bool:
        destination = self._absolute_without_resolving(spec.destination)
        self._validate_managed_write_path(destination)
        receipt_path = self.receipt_path_for(spec)
        if not self._installed_receipt_matches(spec, destination, receipt_path):
            return False
        if not require_post_install:
            return True
        return self._read_json(receipt_path) == self._completed_receipt(spec)

    def mark_post_install_complete(self, spec: ManagedArtifactSpec) -> None:
        self._validate_managed_write_path(self._absolute_without_resolving(spec.destination))
        receipt_path = self.receipt_path_for(spec)
        receipt = self._read_json(receipt_path)
        if receipt != self._receipt(spec):
            expected = self._receipt(spec)
            comparable = dict(receipt) if isinstance(receipt, dict) else {}
            comparable["post_install_state"] = "pending"
            if comparable != expected:
                raise ManagedArtifactError("managed artifact receipt identity is invalid")
        completed = self._receipt(spec)
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
            response = self.opener(request, timeout=self.timeout_seconds)
            with response:
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
                        chunk = response.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        written += len(chunk)
                        received += len(chunk)
                        self._report_progress(spec, written)
                        if written > spec.expected_size:
                            raise ManagedArtifactError("managed artifact size mismatch")
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit() and received != int(content_length):
                    raise ManagedArtifactError("managed artifact download interrupted")
                if written < spec.expected_size:
                    raise ManagedArtifactError("managed artifact download interrupted")
                return written
        except ManagedArtifactError:
            if self._file_size(part_path) > spec.expected_size:
                self._remove_file_safely(part_path)
                self._remove_file_safely(resume_path)
            raise
        except (HTTPError, URLError, OSError, HTTPException, IncompleteRead) as exc:
            raise ManagedArtifactError(f"managed artifact download failed: {exc}") from exc

    def _report_progress(self, spec: ManagedArtifactSpec, downloaded_bytes: int) -> None:
        if self.progress_callback is not None:
            self.progress_callback(spec.component_id, downloaded_bytes, spec.expected_size)

    def _resume_offset(self, spec: ManagedArtifactSpec, part_path: Path, resume_path: Path) -> int:
        if not self._path_is_file(part_path):
            self._remove_file_safely(part_path)
            self._remove_file_safely(resume_path)
            return 0
        size = self._file_size(part_path)
        if size <= 0 or size >= spec.expected_size or self._read_json(resume_path) != self._resume_receipt(spec):
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
                raise ManagedArtifactError("managed artifact destination could not be inspected safely") from exc
            attributes = getattr(details, "st_file_attributes", 0)
            is_reparse = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            is_junction = False
            junction_check = getattr(os.path, "isjunction", None)
            if junction_check is not None:
                try:
                    is_junction = bool(junction_check(component))
                except OSError as exc:
                    raise ManagedArtifactError("managed artifact destination could not be inspected safely") from exc
            if stat.S_ISLNK(details.st_mode) or is_junction or is_reparse:
                raise ManagedArtifactError(
                    "managed artifact destination uses a filesystem link or reparse point"
                )

    def _validate_download(self, spec: ManagedArtifactSpec, part_path: Path) -> None:
        if self._file_size(part_path) != spec.expected_size:
            self._remove_file_safely(part_path)
            self._remove_file_safely(self.resume_receipt_path_for(spec))
            raise ManagedArtifactError("managed artifact size mismatch")
        try:
            actual_hash = self._sha256(part_path)
        except OSError as exc:
            raise ManagedArtifactError(f"managed artifact checksum failed: {exc}") from exc
        if actual_hash != spec.checksum_sha256:
            self._remove_file_safely(part_path)
            self._remove_file_safely(self.resume_receipt_path_for(spec))
            raise ManagedArtifactError("managed artifact checksum mismatch")

    def _publish_file(self, part_path: Path, destination: Path) -> None:
        self._validate_managed_write_path(destination)
        self._require_same_volume(part_path, destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._validate_managed_write_path(destination)
            os.replace(part_path, destination)
        except OSError as exc:
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
            raise ManagedArtifactError(f"managed artifact archive destination failed: {exc}") from exc
        stage = destination.parent / f".{destination.name}.asteria-stage-{uuid4().hex}"
        backup = destination.parent / f".{destination.name}.asteria-backup-{uuid4().hex}"
        self._require_same_volume(part_path, stage)
        try:
            self._validate_managed_write_path(destination)
            self._validate_managed_write_path(stage)
            self._validate_managed_write_path(backup)
            self._extract_zip(part_path, stage, strip_single_root=spec.strip_single_root)
            self._write_json_atomic(stage / ".asteria-managed.json", receipt)
            self._validate_managed_write_path(destination)
            had_destination = self._path_exists(destination)
            if had_destination:
                self._validate_managed_write_path(destination)
                self._validate_managed_write_path(backup)
                os.replace(destination, backup)
            try:
                self._validate_managed_write_path(destination)
                self._validate_managed_write_path(stage)
                os.replace(stage, destination)
            except OSError:
                if had_destination and self._path_exists(backup) and not self._path_exists(destination):
                    self._validate_managed_write_path(backup)
                    self._validate_managed_write_path(destination)
                    os.replace(backup, destination)
                raise
            self._remove_path_safely(backup)
        except ManagedArtifactError:
            self._remove_path_safely(stage)
            self._remove_path_safely(backup)
            self._remove_file_safely(part_path)
            self._remove_file_safely(self.resume_receipt_path_for(spec))
            raise
        except (OSError, zipfile.BadZipFile) as exc:
            self._remove_path_safely(stage)
            if self._path_exists(backup) and not self._path_exists(destination):
                try:
                    self._validate_managed_write_path(backup)
                    self._validate_managed_write_path(destination)
                    os.replace(backup, destination)
                except (ManagedArtifactError, OSError):
                    pass
            self._remove_path_safely(backup)
            raise ManagedArtifactError(f"managed artifact archive publish failed: {exc}") from exc

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
            raise ManagedArtifactError(f"managed artifact archive extraction failed: {exc}") from exc

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
        if spec.kind == "zip":
            if not self._path_is_dir(destination):
                return False
        elif not self._path_is_file(destination) or self._file_size(destination) != spec.expected_size:
            return False
        receipt = self._read_json(receipt_path)
        return receipt == self._receipt(spec) or receipt == self._completed_receipt(spec)

    def _receipt(self, spec: ManagedArtifactSpec) -> dict[str, object]:
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
        return {
            "manifest_version": spec.manifest_version,
            "component_id": spec.component_id,
            "source_url": spec.source_url,
            "expected_size": spec.expected_size,
            "checksum_sha256": spec.checksum_sha256,
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
