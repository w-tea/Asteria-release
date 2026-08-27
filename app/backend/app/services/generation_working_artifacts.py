from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.generation_working_artifact import (
    GenerationWorkingArtifactCleanupResult,
    GenerationWorkingArtifactManifest,
    GenerationWorkingArtifactRecord,
    GenerationWorkingArtifactSummary,
    WorkingArtifactKind,
    WorkingArtifactRetirement,
    validate_working_artifact_token,
)
from app.schemas.operations import ACTIVE_OPERATION_STATUSES
from app.services.atomic_json import write_json_atomic


WORKING_ARTIFACT_RELATIVE_ROOT = Path(".asteria") / "working-artifacts"
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,12}$")


class GenerationWorkingArtifactError(RuntimeError):
    pass


class GenerationWorkingArtifactRegistry:
    def __init__(
        self,
        output_root: Path,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.root = self.output_root / WORKING_ARTIFACT_RELATIVE_ROOT
        self._now = now or (lambda: datetime.now(timezone.utc))

    def register_path(
        self,
        *,
        operation_id: str,
        generation_id: int,
        media_artifact_id: str | None,
        kind: WorkingArtifactKind,
        source_path: Path,
        source_sha256: tuple[str, ...] = (),
        conditioning_sha256: str | None = None,
    ) -> GenerationWorkingArtifactRecord:
        operation_id = self._operation_id(operation_id)
        source = Path(source_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise GenerationWorkingArtifactError("working_artifact_source_invalid")
        operation_dir = self._operation_dir(operation_id, create=True)
        payload_dir = operation_dir / "payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        self._require_contained(payload_dir, operation_dir)
        artifact_uuid = uuid4().hex
        suffix = source.suffix if _SAFE_SUFFIX.fullmatch(source.suffix) else ".bin"
        relative_path = Path("payload") / f"{artifact_uuid}{suffix.lower()}"
        destination = operation_dir / relative_path
        temporary = payload_dir / f".{artifact_uuid}.tmp"
        digest = sha256()
        size_bytes = 0
        try:
            with source.open("rb") as input_file, temporary.open("xb") as output_file:
                while chunk := input_file.read(1024 * 1024):
                    output_file.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
            temporary.replace(destination)
            manifest = self._load_manifest(operation_id, allow_missing=True)
            timestamp = self._timestamp()
            record = GenerationWorkingArtifactRecord(
                artifact_id=f"working:{kind}:{artifact_uuid}",
                operation_id=operation_id,
                generation_id=generation_id,
                media_artifact_id=media_artifact_id,
                kind=kind,
                state="available",
                relative_path=relative_path.as_posix(),
                sha256=digest.hexdigest(),
                size_bytes=size_bytes,
                source_sha256=source_sha256,
                conditioning_sha256=conditioning_sha256,
                created_at=timestamp,
                updated_at=timestamp,
            )
            if any(item.artifact_id == record.artifact_id for item in manifest.artifacts):
                raise GenerationWorkingArtifactError("working_artifact_identifier_conflict")
            # Retire the runtime staging copy before publishing the manifest.
            # A crash between these steps leaves only an unlisted private payload,
            # which startup cleanup can safely reclaim.
            source.unlink()
            self._save_manifest(
                manifest.model_copy(
                    update={
                        "updated_at": timestamp,
                        "artifacts": (*manifest.artifacts, record),
                    }
                )
            )
            return record
        except GenerationWorkingArtifactError:
            self._unlink(temporary)
            self._unlink(destination)
            raise
        except (OSError, ValidationError, ValueError) as error:
            self._unlink(temporary)
            self._unlink(destination)
            raise GenerationWorkingArtifactError("working_artifact_registration_failed") from error

    def list_public(self, operation_id: str) -> tuple[GenerationWorkingArtifactSummary, ...]:
        manifest = self._load_manifest(self._operation_id(operation_id), allow_missing=True)
        return tuple(
            GenerationWorkingArtifactSummary(
                artifact_id=item.artifact_id,
                generation_id=item.generation_id,
                media_artifact_id=item.media_artifact_id,
                kind=item.kind,
                state=item.state,
                size_bytes=item.size_bytes,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in manifest.artifacts
        )

    def resolve_for_adapter(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        expected_kind: WorkingArtifactKind,
        allow_in_use: bool = False,
    ) -> Path:
        operation_id = self._operation_id(operation_id)
        manifest = self._load_manifest(operation_id)
        record = self._find(manifest, artifact_id)
        allowed_states = {"available", "in_use"} if allow_in_use else {"available"}
        if record.kind != expected_kind or record.state not in allowed_states:
            raise GenerationWorkingArtifactError("working_artifact_unavailable")
        path = self._record_path(operation_id, record)
        if not path.is_file() or path.is_symlink():
            raise GenerationWorkingArtifactError("working_artifact_missing")
        if _contains_reparse_component(path, self.output_root):
            raise GenerationWorkingArtifactError("working_artifact_path_escape")
        actual_hash, actual_size = self._file_identity(path)
        if actual_hash != record.sha256 or actual_size != record.size_bytes:
            raise GenerationWorkingArtifactError("working_artifact_changed")
        return path

    def begin_use(
        self,
        operation_id: str,
        artifact_id: str,
    ) -> GenerationWorkingArtifactRecord:
        return self._transition(operation_id, artifact_id, from_states={"available"}, to_state="in_use")

    def finish_use(
        self,
        operation_id: str,
        artifact_id: str,
        *,
        outcome: Literal["available", "consumed"],
        operation_status: str,
    ) -> GenerationWorkingArtifactRecord:
        del operation_status
        return self._transition(operation_id, artifact_id, from_states={"in_use"}, to_state=outcome)

    def finish_use_many(
        self,
        operation_id: str,
        artifact_ids: tuple[str, ...],
        *,
        outcome: Literal["available", "consumed"],
        operation_status: str,
    ) -> tuple[GenerationWorkingArtifactRecord, ...]:
        """Finish one composed use without publishing a partially transitioned set."""
        del operation_status
        operation_id = self._operation_id(operation_id)
        if not artifact_ids or len(artifact_ids) != len(set(artifact_ids)):
            raise GenerationWorkingArtifactError("working_artifact_identifier_conflict")
        manifest = self._load_manifest(operation_id)
        selected = {artifact_id: self._find(manifest, artifact_id) for artifact_id in artifact_ids}
        if any(record.state != "in_use" for record in selected.values()):
            raise GenerationWorkingArtifactError("working_artifact_state_conflict")
        timestamp = self._timestamp()
        updated_by_id = {
            artifact_id: record.model_copy(
                update={"state": outcome, "updated_at": timestamp}
            )
            for artifact_id, record in selected.items()
        }
        self._save_manifest(
            manifest.model_copy(
                update={
                    "updated_at": timestamp,
                    "artifacts": tuple(
                        updated_by_id.get(record.artifact_id, record)
                        for record in manifest.artifacts
                    ),
                }
            )
        )
        return tuple(updated_by_id[artifact_id] for artifact_id in artifact_ids)

    def retire(
        self,
        operation_id: str,
        artifact_id: str,
        *,
        outcome: WorkingArtifactRetirement,
        operation_status: str,
    ) -> GenerationWorkingArtifactRecord:
        del operation_status
        return self._transition(
            operation_id,
            artifact_id,
            from_states={"available"},
            to_state=outcome,
        )

    def cleanup(
        self,
        *,
        operation_status: Callable[[str], str | None],
        stale_after: timedelta = timedelta(hours=6),
    ) -> GenerationWorkingArtifactCleanupResult:
        scanned = 0
        retained_active = 0
        removed = 0
        recovered = 0
        removed_orphans = 0
        try:
            operation_dirs = tuple(path for path in self.root.iterdir() if path.is_dir())
        except FileNotFoundError:
            operation_dirs = ()
        except OSError as error:
            raise GenerationWorkingArtifactError("working_artifact_cleanup_failed") from error
        for operation_dir in operation_dirs:
            if operation_dir.is_symlink() or _contains_reparse_component(
                operation_dir,
                self.output_root,
            ):
                continue
            manifest_path = operation_dir / "manifest.json"
            try:
                manifest = GenerationWorkingArtifactManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                expected_dir = self._operation_dir(manifest.operation_id)
                if expected_dir != operation_dir.resolve():
                    continue
            except (OSError, ValidationError, ValueError):
                continue
            scanned += 1
            status = operation_status(manifest.operation_id)
            if status in ACTIVE_OPERATION_STATUSES:
                retained_active += 1
                continue
            timestamp = self._timestamp()
            changed = False
            updated_records: list[GenerationWorkingArtifactRecord] = []
            referenced: set[Path] = set()
            for record in manifest.artifacts:
                updated = record
                if record.relative_path is not None:
                    try:
                        path = self._record_path(manifest.operation_id, record)
                    except GenerationWorkingArtifactError:
                        updated_records.append(record)
                        continue
                    referenced.add(path)
                    if record.state in {"consumed", "discarded", "superseded"}:
                        if self._unlink(path):
                            removed += 1
                        updated = record.model_copy(
                            update={"relative_path": None, "updated_at": timestamp}
                        )
                        changed = True
                    elif (
                        record.state in {"writing", "in_use"}
                        and timestamp - record.updated_at >= stale_after
                    ):
                        if self._unlink(path):
                            removed += 1
                        updated = record.model_copy(
                            update={
                                "state": "discarded",
                                "relative_path": None,
                                "updated_at": timestamp,
                            }
                        )
                        recovered += 1
                        changed = True
                updated_records.append(updated)
            payload_dir = operation_dir / "payload"
            try:
                for candidate in payload_dir.iterdir():
                    resolved = candidate.resolve()
                    if (
                        candidate.is_file()
                        and not candidate.is_symlink()
                        and resolved not in referenced
                        and timestamp
                        - datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
                        >= stale_after
                        and self._unlink(candidate)
                    ):
                        removed_orphans += 1
            except FileNotFoundError:
                pass
            except OSError:
                pass
            if changed:
                self._save_manifest(
                    manifest.model_copy(
                        update={"updated_at": timestamp, "artifacts": tuple(updated_records)}
                    )
                )
        return GenerationWorkingArtifactCleanupResult(
            scanned_operations=scanned,
            retained_active_operations=retained_active,
            removed_payloads=removed,
            recovered_stale_artifacts=recovered,
            removed_orphans=removed_orphans,
        )

    def _transition(
        self,
        operation_id: str,
        artifact_id: str,
        *,
        from_states: set[str],
        to_state: str,
    ) -> GenerationWorkingArtifactRecord:
        operation_id = self._operation_id(operation_id)
        manifest = self._load_manifest(operation_id)
        current = self._find(manifest, artifact_id)
        if current.state not in from_states:
            raise GenerationWorkingArtifactError("working_artifact_state_conflict")
        timestamp = self._timestamp()
        updated = current.model_copy(update={"state": to_state, "updated_at": timestamp})
        artifacts = tuple(updated if item.artifact_id == artifact_id else item for item in manifest.artifacts)
        self._save_manifest(
            manifest.model_copy(update={"updated_at": timestamp, "artifacts": artifacts})
        )
        return updated

    def _load_manifest(
        self,
        operation_id: str,
        *,
        allow_missing: bool = False,
    ) -> GenerationWorkingArtifactManifest:
        operation_dir = self._operation_dir(operation_id, create=allow_missing)
        path = operation_dir / "manifest.json"
        if not path.exists():
            if not allow_missing:
                raise GenerationWorkingArtifactError("working_artifact_manifest_missing")
            timestamp = self._timestamp()
            return GenerationWorkingArtifactManifest(
                operation_id=operation_id,
                actual_output_root=str(self.output_root),
                created_at=timestamp,
                updated_at=timestamp,
                artifacts=(),
            )
        try:
            manifest = GenerationWorkingArtifactManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValidationError as error:
            if "relative path" in str(error).lower():
                raise GenerationWorkingArtifactError("working_artifact_path_escape") from error
            raise GenerationWorkingArtifactError("working_artifact_manifest_invalid") from error
        except (OSError, ValueError) as error:
            raise GenerationWorkingArtifactError("working_artifact_manifest_invalid") from error
        if manifest.operation_id != operation_id or Path(manifest.actual_output_root) != self.output_root:
            raise GenerationWorkingArtifactError("working_artifact_ownership_mismatch")
        return manifest

    def _save_manifest(self, manifest: GenerationWorkingArtifactManifest) -> None:
        operation_dir = self._operation_dir(manifest.operation_id, create=True)
        try:
            write_json_atomic(
                operation_dir / "manifest.json",
                manifest.model_dump(mode="json"),
            )
        except OSError as error:
            raise GenerationWorkingArtifactError("working_artifact_manifest_write_failed") from error

    def _operation_dir(self, operation_id: str, *, create: bool = False) -> Path:
        digest = sha256(operation_id.encode("utf-8")).hexdigest()[:32]
        path = self.root / f"op-{digest}"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        self._require_contained(resolved, self.output_root)
        if _contains_reparse_component(path, self.output_root):
            raise GenerationWorkingArtifactError("working_artifact_path_escape")
        return resolved

    def _record_path(
        self,
        operation_id: str,
        record: GenerationWorkingArtifactRecord,
    ) -> Path:
        if record.relative_path is None:
            raise GenerationWorkingArtifactError("working_artifact_unavailable")
        operation_dir = self._operation_dir(operation_id)
        candidate = operation_dir / Path(record.relative_path)
        try:
            return self._require_contained(candidate.resolve(), operation_dir)
        except (OSError, RuntimeError) as error:
            raise GenerationWorkingArtifactError("working_artifact_path_escape") from error

    @staticmethod
    def _require_contained(path: Path, root: Path) -> Path:
        if path == root or path.is_relative_to(root):
            return path
        raise GenerationWorkingArtifactError("working_artifact_path_escape")

    @staticmethod
    def _find(
        manifest: GenerationWorkingArtifactManifest,
        artifact_id: str,
    ) -> GenerationWorkingArtifactRecord:
        matches = tuple(item for item in manifest.artifacts if item.artifact_id == artifact_id)
        if len(matches) != 1:
            raise GenerationWorkingArtifactError("working_artifact_not_found")
        return matches[0]

    @staticmethod
    def _file_identity(path: Path) -> tuple[str, int]:
        digest = sha256()
        size = 0
        try:
            with path.open("rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as error:
            raise GenerationWorkingArtifactError("working_artifact_unreadable") from error
        return digest.hexdigest(), size

    @staticmethod
    def _unlink(path: Path) -> bool:
        try:
            existed = path.is_file() and not path.is_symlink()
            if existed:
                path.unlink()
            return existed
        except OSError:
            return False

    @staticmethod
    def _operation_id(value: str) -> str:
        try:
            return validate_working_artifact_token(value, "working artifact operation")
        except ValueError as error:
            raise GenerationWorkingArtifactError("working_artifact_operation_invalid") from error

    def _timestamp(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise GenerationWorkingArtifactError("working_artifact_clock_invalid")
        return value.astimezone(timezone.utc)


def _contains_reparse_component(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    current = path
    while True:
        try:
            details = os.lstat(current)
        except OSError:
            return True
        attributes = getattr(details, "st_file_attributes", 0)
        is_reparse = bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        junction_check = getattr(os.path, "isjunction", None)
        try:
            is_junction = (
                bool(junction_check(current)) if junction_check is not None else False
            )
        except OSError:
            return True
        if stat.S_ISLNK(details.st_mode) or is_reparse or is_junction:
            return True
        if current == root:
            return False
        current = current.parent
