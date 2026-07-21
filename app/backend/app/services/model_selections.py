from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading

from app.services.atomic_json import write_json_atomic
from app.services.model_dependencies import ModelDependencyStatus


_selection_locks_guard = threading.Lock()
_selection_locks: dict[str, threading.RLock] = {}


def _selection_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(path))
    with _selection_locks_guard:
        return _selection_locks.setdefault(key, threading.RLock())


@dataclass(frozen=True)
class ModelSelection:
    profile: str
    slots: dict[str, str]


class ModelSelectionRepository:
    def __init__(self, selections_path: Path) -> None:
        self.selections_path = selections_path

    def get(self, profile: str) -> ModelSelection:
        raw = self._read()
        if profile == "flux1" and profile not in raw and "flux" in raw:
            return ModelSelection(profile=profile, slots=self._slots(raw.get("flux", {})))
        return ModelSelection(profile=profile, slots=self._slots(raw.get(profile, {})))

    def save(self, selection: ModelSelection) -> ModelSelection:
        with _selection_lock(self.selections_path):
            raw = self._read(strict=True)
            raw[selection.profile] = selection.slots
            write_json_atomic(self.selections_path, raw)
        return selection

    def autofill(
        self,
        profile: str,
        dependencies: list[ModelDependencyStatus],
        allowed_slot_ids: set[str] | None = None,
    ) -> ModelSelection:
        with _selection_lock(self.selections_path):
            return self._autofill_locked(profile, dependencies, allowed_slot_ids)

    def autofill_complete_starter_packs(
        self,
        starter_pack_ids: Iterable[str],
        dependencies: list[ModelDependencyStatus],
    ) -> tuple[ModelSelection, ...]:
        selected_pack_ids = tuple(dict.fromkeys(starter_pack_ids))
        candidate_pack_ids = list(selected_pack_ids)
        for dependency in dependencies:
            if dependency.starter_pack and dependency.starter_pack not in candidate_pack_ids:
                candidate_pack_ids.append(dependency.starter_pack)
        selections: list[ModelSelection] = []
        for starter_pack_id in candidate_pack_ids:
            pack_dependencies = [
                dependency
                for dependency in dependencies
                if dependency.starter_pack == starter_pack_id
            ]
            if not pack_dependencies or any(
                dependency.status != "installed" for dependency in pack_dependencies
            ):
                continue
            profiles = {dependency.profile for dependency in pack_dependencies}
            if len(profiles) != 1:
                continue
            profile = profiles.pop()
            if (
                starter_pack_id not in selected_pack_ids
                and not self._selection_needs_portable_rebase(profile, pack_dependencies)
            ):
                continue
            selections.append(self.autofill(profile, pack_dependencies))
        return tuple(selections)

    def _selection_needs_portable_rebase(
        self,
        profile: str,
        dependencies: list[ModelDependencyStatus],
    ) -> bool:
        slots = self.get(profile).slots
        for dependency in dependencies:
            if not dependency.slot_id:
                continue
            selected_path = slots.get(dependency.slot_id, "")
            if not selected_path or _path_is_file_safely(Path(selected_path)):
                continue
            selected_relative = _portable_model_relative_parts(Path(selected_path))
            if (
                selected_relative is not None
                and selected_relative == _portable_model_relative_parts(dependency.path)
            ):
                return True
        return False

    def _autofill_locked(
        self,
        profile: str,
        dependencies: list[ModelDependencyStatus],
        allowed_slot_ids: set[str] | None,
    ) -> ModelSelection:
        selection = self.get(profile)
        slots = dict(selection.slots)
        changed = False

        for slot_id, value in list(slots.items()):
            if allowed_slot_ids is not None and slot_id not in allowed_slot_ids:
                slots.pop(slot_id, None)
                changed = True
                continue
            if value and not _path_is_file_safely(Path(value)):
                slots[slot_id] = ""
                changed = True

        for dependency in dependencies:
            if dependency.profile != profile or dependency.status != "installed" or not dependency.slot_id:
                continue
            current = slots.get(dependency.slot_id, "")
            if current and _path_is_file_safely(Path(current)):
                continue
            slots[dependency.slot_id] = str(dependency.path)
            changed = True

        if changed:
            return self.save(ModelSelection(profile=profile, slots=slots))
        return ModelSelection(profile=profile, slots=slots)

    def _read(self, *, strict: bool = False) -> dict[str, dict[str, str]]:
        try:
            if not self.selections_path.exists():
                return {}
            raw = json.loads(self.selections_path.read_text(encoding="utf-8"))
        except OSError:
            if strict:
                raise
            return {}
        except json.JSONDecodeError as exc:
            if strict:
                raise OSError(f"model selections file is not valid JSON: {exc}") from exc
            return {}
        if not isinstance(raw, dict):
            if strict:
                raise OSError("model selections file root must be an object")
            return {}
        if strict:
            for slots in raw.values():
                if not isinstance(slots, dict):
                    raise OSError("model selections file contains an invalid profile")
                if any(not isinstance(path, str) for path in slots.values()):
                    raise OSError("model selections file contains an invalid slot path")
        return {str(profile): self._slots(slots) for profile, slots in raw.items()}

    def _slots(self, value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(slot_id): str(path) for slot_id, path in value.items() if isinstance(path, str)}


def _path_is_file_safely(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _portable_model_relative_parts(path: Path) -> tuple[str, ...] | None:
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "models":
            return tuple(part.casefold() for part in parts[index + 1 :])
    return None
