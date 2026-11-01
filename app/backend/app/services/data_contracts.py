from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any


_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_WINDOWS_REPARSE_ATTRIBUTE = 0x400


class StorageClass(str, Enum):
    DURABLE = "durable"
    RECONSTRUCTIBLE_CACHE = "reconstructible_cache"
    RECOVERABLE_STAGING = "recoverable_staging"
    SECRET = "secret"


class PathKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True, slots=True)
class DataContractEntry:
    id: str
    owner: str
    storage_class: StorageClass
    relative_root: PurePosixPath
    path_kind: PathKind
    record_kinds: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DataContractEntry:
        entry_id = _required_identifier(payload.get("id"), "entry id")
        owner = _required_identifier(payload.get("owner"), "entry owner")
        relative_root = _relative_root(payload.get("relative_root"))
        record_values = payload.get("record_kinds")
        if not isinstance(record_values, list) or not record_values:
            raise ValueError(f"data contract entry {entry_id} must declare record kinds")
        record_kinds = tuple(_required_identifier(value, "record kind") for value in record_values)
        if len(set(record_kinds)) != len(record_kinds):
            raise ValueError(f"data contract entry {entry_id} has duplicate record kind")
        try:
            storage_class = StorageClass(payload.get("storage_class"))
        except ValueError as exc:
            raise ValueError(f"data contract entry {entry_id} has invalid storage class") from exc
        try:
            path_kind = PathKind(payload.get("path_kind"))
        except ValueError as exc:
            raise ValueError(f"data contract entry {entry_id} has invalid path kind") from exc
        return cls(
            id=entry_id,
            owner=owner,
            storage_class=storage_class,
            relative_root=relative_root,
            path_kind=path_kind,
            record_kinds=record_kinds,
        )


@dataclass(frozen=True, slots=True)
class DataContractManifest:
    contract_id: str
    source_version: str
    entries: tuple[DataContractEntry, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DataContractManifest:
        contract_id = _required_text(payload.get("contract_id"), "data contract id")
        source_version = _required_text(payload.get("source_version"), "data contract source version")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("data contract must declare entries")
        entries = tuple(
            DataContractEntry.from_mapping(entry)
            for entry in raw_entries
            if isinstance(entry, Mapping)
        )
        if len(entries) != len(raw_entries):
            raise ValueError("data contract entries must be objects")
        manifest = cls(contract_id=contract_id, source_version=source_version, entries=entries)
        return validate_data_contract(manifest)

    def records_by_kind(self) -> dict[str, DataContractEntry]:
        return {
            record_kind: entry
            for entry in self.entries
            for record_kind in entry.record_kinds
        }


@dataclass(frozen=True, slots=True)
class OwnedDataPath:
    entry: DataContractEntry
    path: Path


def load_data_contract(path: Path) -> DataContractManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"data contract could not be read: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("data contract must be an object")
    return DataContractManifest.from_mapping(payload)


def validate_data_contract(manifest: DataContractManifest) -> DataContractManifest:
    ids: set[str] = set()
    records: set[str] = set()
    for entry in manifest.entries:
        if entry.id in ids:
            raise ValueError(f"duplicate data contract entry id: {entry.id}")
        ids.add(entry.id)
        for record_kind in entry.record_kinds:
            if record_kind in records:
                raise ValueError(f"duplicate record kind: {record_kind}")
            records.add(record_kind)

    for index, entry in enumerate(manifest.entries):
        for other in manifest.entries[index + 1 :]:
            if _paths_overlap(entry.relative_root, other.relative_root):
                raise ValueError(
                    "overlapping mutation roots: "
                    f"{entry.relative_root.as_posix()} and {other.relative_root.as_posix()}"
                )
    return manifest


def validate_owned_paths(manifest: DataContractManifest, data_root: Path) -> tuple[OwnedDataPath, ...]:
    root = data_root.absolute()
    if _is_reparse_point(root):
        raise ValueError("data contract root cannot be a reparse point")

    owned_paths: list[OwnedDataPath] = []
    for entry in manifest.entries:
        candidate = root
        for part in entry.relative_root.parts:
            candidate = candidate / part
            if _is_reparse_point(candidate):
                raise ValueError(f"data contract path crosses a reparse point: {entry.id}")
        owned_paths.append(OwnedDataPath(entry=entry, path=candidate))
    return tuple(owned_paths)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _required_identifier(value: object, label: str) -> str:
    text = _required_text(value, label)
    if not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ValueError(f"{label} is invalid")
    return text


def _relative_root(value: object) -> PurePosixPath:
    text = _required_text(value, "data contract relative root").replace("\\", "/")
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or text in {".", ""}
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError("data contract relative root must stay within the data root")
    return posix_path


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    left_parts = tuple(part.casefold() for part in left.parts)
    right_parts = tuple(part.casefold() for part in right.parts)
    shorter = min(len(left_parts), len(right_parts))
    return left_parts[:shorter] == right_parts[:shorter]


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & _WINDOWS_REPARSE_ATTRIBUTE)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"data contract path could not be inspected: {path.name}") from exc


def current_data_contract() -> DataContractManifest:
    return _CURRENT_DATA_CONTRACT


def _entry(
    entry_id: str,
    owner: str,
    storage_class: StorageClass,
    relative_root: str,
    path_kind: PathKind,
    *record_kinds: str,
) -> DataContractEntry:
    return DataContractEntry(
        id=entry_id,
        owner=owner,
        storage_class=storage_class,
        relative_root=PurePosixPath(relative_root),
        path_kind=path_kind,
        record_kinds=tuple(record_kinds),
    )


_CURRENT_DATA_CONTRACT = validate_data_contract(
    DataContractManifest(
        contract_id="asteria.data.v1.2.0",
        source_version="1.2.0",
        entries=(
            _entry("history", "history", StorageClass.DURABLE, "asteria.db", PathKind.FILE, "generation_history"),
            _entry("user_settings", "settings", StorageClass.DURABLE, "user-settings.json", PathKind.FILE, "user_settings", "library_saved_filters"),
            _entry("presets", "presets", StorageClass.DURABLE, "presets.json", PathKind.FILE, "generation_presets"),
            _entry("prompt_presets", "prompt_presets", StorageClass.DURABLE, "prompt-presets.json", PathKind.FILE, "prompt_presets"),
            _entry("external_integrations", "external_integrations", StorageClass.DURABLE, "external-integrations.json", PathKind.FILE, "external_integration_settings"),
            _entry("model_storage_settings", "model_storage", StorageClass.DURABLE, "model-storage-settings.json", PathKind.FILE, "model_storage_settings"),
            _entry("generation_inputs", "generation_inputs", StorageClass.DURABLE, "init_images", PathKind.DIRECTORY, "generation_input_images"),
            _entry("generation_masks", "generation_inputs", StorageClass.DURABLE, "mask_images", PathKind.DIRECTORY, "generation_mask_images"),
            _entry("control_inputs", "generation_inputs", StorageClass.DURABLE, "control_images", PathKind.DIRECTORY, "control_input_images"),
            _entry("outputs", "generation_artifacts", StorageClass.DURABLE, "outputs", PathKind.DIRECTORY, "generated_artifacts", "generation_sidecars"),
            _entry("advanced_styles", "advanced_styles", StorageClass.DURABLE, "advanced_styles", PathKind.DIRECTORY, "advanced_styles", "advanced_style_thumbnails"),
            _entry("image_editor_drafts", "image_editor", StorageClass.DURABLE, "editor_drafts", PathKind.DIRECTORY, "image_edit_draft", "image_edit_mask"),
            _entry("lora_training", "lora_training", StorageClass.DURABLE, "lora-training", PathKind.DIRECTORY, "lora_datasets", "lora_runs", "copy_machine_sessions", "copy_machine_runs", "lora_recipe_reports"),
            _entry(
                "extensions",
                "extensions",
                StorageClass.DURABLE,
                "extensions",
                PathKind.DIRECTORY,
                "managed_extension_active_package",
                "managed_extension_previous_package",
                "extension_manifests",
                "extension_package_hashes",
                "extension_version_receipts",
                "extension_file_hashes",
                "extension_state",
                "extension_registry",
                "unmanaged_extension_reference_metadata",
            ),
            _entry("merge_recipes", "merge_studio", StorageClass.DURABLE, "merge-recipes.json", PathKind.FILE, "merge_recipes"),
            _entry("reports", "reports", StorageClass.DURABLE, "reports", PathKind.DIRECTORY, "merge_runs", "merge_reports", "copy_machine_reports"),
            _entry("discord_secrets", "discord", StorageClass.SECRET, "secrets", PathKind.DIRECTORY, "discord_bot_token"),
            _entry("cache", "cache", StorageClass.RECONSTRUCTIBLE_CACHE, "cache", PathKind.DIRECTORY, "library_thumbnails", "model_scan_cache"),
            _entry("managed_download_staging", "managed_downloads", StorageClass.RECOVERABLE_STAGING, "downloads", PathKind.DIRECTORY, "download_partials", "download_resume_receipts"),
        ),
    )
)
