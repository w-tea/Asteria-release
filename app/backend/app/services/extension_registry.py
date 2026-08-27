from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.schemas.extensions import (
    ExtensionRegistryRecord,
    ExtensionRegistrySnapshot,
    ExtensionRegistrySnapshotRecord,
    validate_extension_id,
)
from app.services.atomic_json import write_json_atomic
from app.services.extension_repository import ExtensionRepository, ExtensionRepositoryError


_SAFE_ERROR_CODES = frozenset(
    {
        "extension_active_version_invalid",
        "extension_entrypoint_invalid",
        "extension_id_collision",
        "extension_id_invalid",
        "extension_manifest_incompatible",
        "extension_manifest_invalid",
        "extension_manifest_mismatch",
        "extension_path_escape",
        "extension_path_unreadable",
        "extension_previous_version_invalid",
        "extension_receipt_invalid",
        "extension_reparse_traversal",
        "extension_root_unreadable",
        "extension_state_invalid",
        "extension_unmanaged_reference_missing",
    }
)


class ExtensionRegistryRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, records: tuple[ExtensionRegistryRecord, ...]) -> ExtensionRegistrySnapshot:
        snapshot_records: list[ExtensionRegistrySnapshotRecord] = []
        for record in records:
            if record.status == "error":
                snapshot_records.append(
                    ExtensionRegistrySnapshotRecord(
                        extension_id=record.extension_id,
                        status="error",
                        error_code=record.error_code,
                    )
                )
                continue
            assert record.state is not None
            snapshot_records.append(
                ExtensionRegistrySnapshotRecord(
                    extension_id=record.extension_id,
                    status=record.status,
                    active_version=record.state.active_version,
                    previous_version=record.state.previous_version,
                    enabled=record.state.enabled,
                    source_kind=record.state.source.kind,
                    recovery_status=record.state.recovery.status,
                    manifest_version=record.manifest.version if record.manifest is not None else None,
                    error_code=None,
                )
            )
        snapshot = ExtensionRegistrySnapshot(records=tuple(snapshot_records))
        write_json_atomic(self.path, snapshot.model_dump(mode="json"))
        return snapshot

    def load(self) -> ExtensionRegistrySnapshot:
        try:
            return ExtensionRegistrySnapshot.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError) as exc:
            raise ValueError("extension registry snapshot is invalid") from exc


class ExtensionRegistry:
    def __init__(
        self,
        repository: ExtensionRepository,
        *,
        snapshot_repository: ExtensionRegistryRepository | None = None,
    ) -> None:
        self.repository = repository
        self.snapshot_repository = snapshot_repository or ExtensionRegistryRepository(repository.registry_path)

    def scan(self) -> tuple[ExtensionRegistryRecord, ...]:
        extension_ids = self.repository.discover_extension_ids()
        folded_counts: dict[str, int] = {}
        for extension_id in extension_ids:
            folded = extension_id.casefold()
            folded_counts[folded] = folded_counts.get(folded, 0) + 1

        reserved_ids: set[str] = set()
        for extension_id in extension_ids:
            if folded_counts[extension_id.casefold()] != 1:
                continue
            try:
                reserved_ids.add(validate_extension_id(extension_id).casefold())
            except (TypeError, ValueError):
                continue
        used_ids: set[str] = set()
        placeholder_index = 0

        def next_placeholder() -> str:
            nonlocal placeholder_index
            while True:
                placeholder_index += 1
                candidate = (
                    "invalid-extension"
                    if placeholder_index == 1
                    else f"invalid-extension-{placeholder_index}"
                )
                folded = candidate.casefold()
                if folded not in reserved_ids and folded not in used_ids:
                    used_ids.add(folded)
                    return candidate

        records: list[ExtensionRegistryRecord] = []
        for extension_id in extension_ids:
            if folded_counts[extension_id.casefold()] > 1:
                records.append(self._error(next_placeholder(), "extension_id_collision"))
                continue
            try:
                state = self.repository.read_state(extension_id)
                if state.recovery.status == "missing_install":
                    manifest = None
                elif state.source.kind == "managed":
                    manifest = self.repository.read_manifest(extension_id, state.active_version)
                else:
                    manifest = None
                records.append(
                    ExtensionRegistryRecord(
                        extension_id=extension_id,
                        status="ready" if state.recovery.status == "ready" else "recovery",
                        state=state,
                        manifest=manifest,
                        error_code=None,
                    )
                )
                used_ids.add(extension_id.casefold())
            except ExtensionRepositoryError as exc:
                code = exc.code if exc.code in _SAFE_ERROR_CODES else "extension_state_invalid"
                try:
                    public_id = validate_extension_id(extension_id)
                    if public_id.casefold() in used_ids:
                        public_id = next_placeholder()
                    else:
                        used_ids.add(public_id.casefold())
                except (TypeError, ValueError):
                    public_id = next_placeholder()
                records.append(self._error(public_id, code))
        result = tuple(sorted(records, key=lambda record: record.extension_id.casefold()))
        self.snapshot_repository.write(result)
        return result

    @staticmethod
    def _error(public_id: str, code: str) -> ExtensionRegistryRecord:
        return ExtensionRegistryRecord(
            extension_id=public_id,
            status="error",
            state=None,
            manifest=None,
            error_code=code,  # type: ignore[arg-type]
        )
