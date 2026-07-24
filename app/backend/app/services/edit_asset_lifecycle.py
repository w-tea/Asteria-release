from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Callable

from pydantic import ValidationError

from app.schemas.image_edit import ImageEditLineage
from app.services.library_thumbnails import LibraryThumbnailService
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.user_settings import UserSettings, resolve_output_root


EDIT_ASSET_DIRECTORIES = ("init_images", "mask_images", "control_images", "editor_drafts")
EDIT_ASSET_PATTERN = re.compile(
    r"^[0-9a-f]{32}(?:_[^\\/]+|\.(?:json|png|jpe?g|webp))$",
    re.IGNORECASE,
)
LINEAGE_PATTERN = re.compile(
    r"^[0-9a-f]{32}_[^\\/]+\.(?:png|jpe?g|webp)\.asteria\.edit\.json$",
    re.IGNORECASE,
)
THUMBNAIL_PATTERN = re.compile(r"^[0-9a-f]{64}\.webp$", re.IGNORECASE)
DEFAULT_ORPHAN_GRACE = timedelta(days=7)
MAX_REFERENCE_TRAVERSAL_DEPTH = 64
MAX_REFERENCE_TRAVERSAL_ENTRIES = 100_000


@dataclass(frozen=True)
class AssetMeasure:
    count: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class EditAssetGroup:
    paths: tuple[Path, ...]
    referenced: bool
    old_enough: bool
    healthy: bool

    @property
    def measure(self) -> AssetMeasure:
        sizes = []
        for path in self.paths:
            try:
                sizes.append(int(path.stat().st_size))
            except OSError:
                sizes.append(0)
        return AssetMeasure(count=len(self.paths), bytes=sum(sizes))


@dataclass(frozen=True)
class EditAssetInventory:
    editing_temp: AssetMeasure
    safe_reclaimable: AssetMeasure
    thumbnail_cache: AssetMeasure
    protected: AssetMeasure
    reference_health: str
    groups: tuple[EditAssetGroup, ...]


@dataclass(frozen=True)
class EditAssetCleanupResult:
    removed: AssetMeasure
    skipped: AssetMeasure
    failed: AssetMeasure
    reference_health: str


def _safe_resolve(path: Path, *, strict: bool) -> Path | None:
    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError, ValueError):
        return None


def _has_reparse_component(path: Path, root: Path) -> bool:
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
        is_reparse = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        junction_check = getattr(os.path, "isjunction", None)
        try:
            is_junction = bool(junction_check(current)) if junction_check is not None else False
        except OSError:
            return True
        if stat.S_ISLNK(details.st_mode) or is_reparse or is_junction:
            return True
        if current == root:
            return False
        current = current.parent


def _owned_path(path: Path, root: Path, pattern: re.Pattern[str]) -> bool:
    if not pattern.fullmatch(path.name):
        return False
    raw_path = Path(os.path.abspath(os.fspath(path)))
    raw_root = Path(os.path.abspath(os.fspath(root)))
    resolved_path = _safe_resolve(raw_path, strict=True)
    resolved_root = _safe_resolve(raw_root, strict=True)
    if resolved_path is None or resolved_root is None or not resolved_path.is_file():
        return False
    if resolved_path.parent != resolved_root or _has_reparse_component(raw_path, raw_root):
        return False
    return True


def is_owned_edit_asset(path: Path, root: Path) -> bool:
    return _owned_path(path, root, EDIT_ASSET_PATTERN) or _owned_path(path, root, LINEAGE_PATTERN)


def is_owned_thumbnail_cache(path: Path, root: Path) -> bool:
    return _owned_path(path, root, THUMBNAIL_PATTERN)


class EditAssetLifecycleService:
    def __init__(self, data_dir: Path, *, grace: timedelta = DEFAULT_ORPHAN_GRACE) -> None:
        self.data_dir = data_dir
        self.grace = grace
        self.thumbnail_root = data_dir / "cache" / "library-thumbnails"

    def inventory(self) -> EditAssetInventory:
        references, healthy = self._persisted_references()
        groups = tuple(self._groups(references, healthy))
        editing_temp = self._sum(group.measure for group in groups if not group.referenced)
        protected = self._sum(group.measure for group in groups if group.referenced)
        safe = self._sum(
            group.measure
            for group in groups
            if healthy and group.healthy and not group.referenced and group.old_enough
        )
        thumbnail_entries = LibraryThumbnailService(self.thumbnail_root).inventory()
        thumbnail = AssetMeasure(
            count=len(thumbnail_entries),
            bytes=sum(entry.size for entry in thumbnail_entries),
        )
        return EditAssetInventory(
            editing_temp=editing_temp,
            safe_reclaimable=safe,
            thumbnail_cache=thumbnail,
            protected=protected,
            reference_health="healthy" if healthy else "uncertain",
            groups=groups,
        )

    def cleanup(self) -> EditAssetCleanupResult | None:
        cleanup_lease = reference_mutation_coordinator.try_acquire_cleanup()
        if cleanup_lease is None:
            return None
        with cleanup_lease:
            inventory = self.inventory()
            removed = AssetMeasure()
            skipped = AssetMeasure()
            failed = AssetMeasure()
            for group in inventory.groups:
                if (
                    inventory.reference_health != "healthy"
                    or not group.healthy
                    or group.referenced
                    or not group.old_enough
                ):
                    skipped = self._plus(skipped, group.measure)
                    continue
                current_references, current_healthy = self._persisted_references()
                current_group = self._revalidate_group(group, current_references, current_healthy)
                if not current_group:
                    skipped = self._plus(skipped, group.measure)
                    continue
                group_removed = AssetMeasure()
                group_failed = AssetMeasure()
                for path in current_group.paths:
                    try:
                        size = int(path.stat().st_size)
                        path.unlink()
                        group_removed = self._plus(group_removed, AssetMeasure(1, size))
                    except OSError:
                        group_failed = self._plus(group_failed, AssetMeasure(1, 0))
                removed = self._plus(removed, group_removed)
                failed = self._plus(failed, group_failed)
            LibraryThumbnailService(self.thumbnail_root).cleanup()
            return EditAssetCleanupResult(
                removed=removed,
                skipped=skipped,
                failed=failed,
                reference_health=inventory.reference_health,
            )

    def _revalidate_group(
        self,
        group: EditAssetGroup,
        references: set[str],
        healthy: bool,
    ) -> EditAssetGroup | None:
        if not healthy:
            return None
        for path in group.paths:
            root = self._root_for(path)
            if root is None or not is_owned_edit_asset(path, root):
                return None
            if self._path_key(path) in references:
                return None
        return group

    def _groups(self, references: set[str], references_healthy: bool) -> list[EditAssetGroup]:
        now = datetime.now(UTC).timestamp()
        groups: list[EditAssetGroup] = []
        for directory_name in EDIT_ASSET_DIRECTORIES:
            root = self.data_dir / directory_name
            try:
                candidates = [path for path in root.iterdir() if is_owned_edit_asset(path, root)]
            except OSError:
                continue
            used: set[Path] = set()
            for path in candidates:
                if path in used:
                    continue
                members = [path]
                if not path.name.endswith(".asteria.edit.json"):
                    lineage = path.with_suffix(f"{path.suffix}.asteria.edit.json")
                    if lineage in candidates:
                        members.append(lineage)
                else:
                    base = Path(str(path)[: -len(".asteria.edit.json")])
                    if base in candidates:
                        members.insert(0, base)
                used.update(members)
                healthy = all(self._lineage_healthy(member) for member in members)
                referenced = any(self._path_key(member) in references for member in members)
                try:
                    old_enough = all((now - member.stat().st_mtime) >= self.grace.total_seconds() for member in members)
                except OSError:
                    old_enough = False
                    healthy = False
                groups.append(
                    EditAssetGroup(
                        paths=tuple(members),
                        referenced=referenced,
                        old_enough=old_enough,
                        healthy=healthy and references_healthy,
                    )
                )
        return groups

    def _lineage_healthy(self, path: Path) -> bool:
        if not path.name.endswith(".asteria.edit.json"):
            return True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict)

    def _persisted_references(self) -> tuple[set[str], bool]:
        references: set[str] = set()
        healthy = self._history_references(references)
        json_sources, sources_healthy = self._known_json_sources()
        healthy = healthy and sources_healthy
        for path in json_sources:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                healthy = False
                continue
            self._collect_strings(payload, references)
        lineage_dependencies, lineage_healthy = self._lineage_dependencies()
        healthy = healthy and lineage_healthy
        if lineage_healthy:
            healthy = self._expand_lineage_references(
                references,
                lineage_dependencies,
            ) and healthy
        return references, healthy

    def _lineage_dependencies(self) -> tuple[dict[str, set[str]], bool]:
        dependencies: dict[str, set[str]] = {}
        healthy = True
        for directory_name in EDIT_ASSET_DIRECTORIES:
            root = self.data_dir / directory_name
            manifests, manifests_healthy = self._enumerate_json_sources(
                root,
                recursive=False,
                predicate=lambda path: path.name.lower().endswith(".asteria.edit.json"),
            )
            healthy = healthy and manifests_healthy
            for manifest_path in manifests:
                if not is_owned_edit_asset(manifest_path, root):
                    healthy = False
                    continue
                try:
                    lineage = ImageEditLineage.model_validate_json(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValidationError):
                    healthy = False
                    continue
                try:
                    paired_image = Path(
                        str(manifest_path)[: -len(".asteria.edit.json")]
                    )
                    current_key = self._path_key(Path(lineage.current.path))
                    paired_image_key = self._path_key(paired_image)
                    if current_key != paired_image_key:
                        healthy = False
                        continue
                    upstream = {
                        self._path_key(Path(lineage.root.path)),
                        *(
                            self._path_key(Path(operation.source.path))
                            for operation in lineage.operations
                        ),
                        *(
                            self._path_key(Path(operation.output.path))
                            for operation in lineage.operations[:-1]
                        ),
                    }
                    upstream.discard(current_key)
                    keys = {
                        current_key,
                        self._path_key(manifest_path),
                        paired_image_key,
                    }
                except (OSError, RuntimeError, TypeError, ValueError):
                    healthy = False
                    continue
                for key in keys:
                    dependencies.setdefault(key, set()).update(upstream)
        return dependencies, healthy

    @staticmethod
    def _expand_lineage_references(
        references: set[str],
        dependencies: dict[str, set[str]],
    ) -> bool:
        pending = list(references)
        visited: set[str] = set()
        traversed = 0
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            visited.add(key)
            traversed += 1
            if traversed > MAX_REFERENCE_TRAVERSAL_ENTRIES:
                return False
            for dependency in dependencies.get(key, ()):
                if dependency not in references:
                    references.add(dependency)
                    pending.append(dependency)
        return True

    def _history_references(self, references: set[str]) -> bool:
        db_path = self.data_dir / "asteria.db"
        if not db_path.exists():
            return True
        try:
            with sqlite3.connect(db_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    """
                    select settings_json, model_slots_json, loras_json, embeddings_json, output_paths_json
                    from generations
                    """
                ).fetchall()
            for row in rows:
                for column in row.keys():
                    payload = json.loads(row[column])
                    self._collect_strings(payload, references)
            return True
        except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError):
            return False

    def _known_json_sources(self) -> tuple[set[Path], bool]:
        sources: set[Path] = set()
        healthy = True
        root_sources, root_healthy = self._enumerate_json_sources(
            self.data_dir,
            recursive=False,
            predicate=lambda path: path.name.lower().endswith(".json"),
        )
        sources.update(root_sources)
        healthy = healthy and root_healthy
        for root in (
            self.data_dir / "config",
            self.data_dir / "lora-training",
            self.data_dir / "editor_drafts",
        ):
            nested_sources, nested_healthy = self._enumerate_json_sources(
                root,
                recursive=True,
                predicate=lambda path: path.name.lower().endswith(".json"),
            )
            sources.update(nested_sources)
            healthy = healthy and nested_healthy

        settings = UserSettings()
        settings_path = self.data_dir / "user-settings.json"
        if settings_path in sources:
            try:
                payload = json.loads(settings_path.read_text(encoding="utf-8"))
                settings = UserSettings.model_validate(payload)
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValidationError):
                healthy = False
        output_roots = {self.data_dir / "outputs": True}
        try:
            configured_root = resolve_output_root(self.data_dir, settings.output)
            explicitly_configured = bool(settings.output.root.strip())
            output_roots[configured_root] = (
                output_roots.get(configured_root, True) and not explicitly_configured
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            healthy = False
        for root, missing_is_healthy in output_roots.items():
            sidecars, sidecars_healthy = self._enumerate_json_sources(
                root,
                recursive=True,
                predicate=lambda path: path.name.lower().endswith(
                    (".asteria.json", ".asteria.merge-candidate.json")
                ),
                missing_is_healthy=missing_is_healthy,
            )
            sources.update(sidecars)
            healthy = healthy and sidecars_healthy
        sources = {
            path
            for path in sources
            if not path.name.lower().endswith(".asteria.edit.json")
        }
        return sources, healthy

    def _enumerate_json_sources(
        self,
        root: Path,
        *,
        recursive: bool,
        predicate: Callable[[Path], bool],
        missing_is_healthy: bool = True,
    ) -> tuple[set[Path], bool]:
        try:
            root_details = os.lstat(root)
        except FileNotFoundError:
            return set(), missing_is_healthy
        except (OSError, TypeError, ValueError):
            return set(), False
        try:
            if self._details_are_reparse(root, root_details) or not stat.S_ISDIR(root_details.st_mode):
                return set(), False
        except (OSError, TypeError, ValueError):
            return set(), False
        root_identity = self._directory_identity(root_details)
        try:
            current = Path(os.path.abspath(os.fspath(root)))
        except (TypeError, ValueError):
            return set(), False
        current = current.parent
        while True:
            boundary = self._reparse_boundary(current)
            if boundary is not False:
                return set(), False
            parent = current.parent
            if parent == current:
                break
            current = parent

        sources: set[Path] = set()
        healthy = True
        visited = 0
        pending = [(root, 0, root_identity)]
        while pending:
            directory, depth, expected_identity = pending.pop()
            try:
                current_details = os.lstat(directory)
                if (
                    self._details_are_reparse(directory, current_details)
                    or not stat.S_ISDIR(current_details.st_mode)
                    or self._directory_identity(current_details) != expected_identity
                ):
                    healthy = False
                    continue
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > MAX_REFERENCE_TRAVERSAL_ENTRIES:
                            return sources, False
                        path = Path(entry.path)
                        try:
                            if entry.is_symlink():
                                healthy = False
                                continue
                            details = os.lstat(path)
                            if self._details_are_reparse(path, details):
                                healthy = False
                                continue
                            is_file = stat.S_ISREG(details.st_mode)
                            is_dir = stat.S_ISDIR(details.st_mode)
                        except (OSError, TypeError, ValueError):
                            healthy = False
                            continue
                        if is_file:
                            try:
                                if predicate(path):
                                    sources.add(path)
                            except (AttributeError, TypeError):
                                healthy = False
                            continue
                        if not is_dir or not recursive:
                            continue
                        if depth >= MAX_REFERENCE_TRAVERSAL_DEPTH:
                            healthy = False
                            continue
                        pending.append((path, depth + 1, self._directory_identity(details)))
            except FileNotFoundError:
                healthy = False
            except (OSError, TypeError, ValueError):
                healthy = False
        return sources, healthy

    def _directory_identity(self, details: os.stat_result) -> tuple[int, int, int]:
        return int(details.st_dev), int(details.st_ino), int(details.st_ctime_ns)

    def _details_are_reparse(self, path: Path, details: os.stat_result) -> bool:
        attributes = getattr(details, "st_file_attributes", 0)
        if stat.S_ISLNK(details.st_mode):
            return True
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            return True
        junction_check = getattr(os.path, "isjunction", None)
        return bool(junction_check(path)) if junction_check is not None else False

    def _reparse_boundary(self, path: Path) -> bool | None:
        try:
            details = os.lstat(path)
            return self._details_are_reparse(path, details)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError):
            return None

    def _collect_strings(self, value: Any, references: set[str]) -> None:
        if isinstance(value, str):
            if value:
                references.add(self._path_key(Path(value)))
            return
        if isinstance(value, dict):
            for nested in value.values():
                self._collect_strings(nested, references)
            return
        if isinstance(value, list):
            for nested in value:
                self._collect_strings(nested, references)

    def _path_key(self, path: Path) -> str:
        resolved = _safe_resolve(path, strict=False)
        value = resolved if resolved is not None else Path(os.path.abspath(os.fspath(path)))
        return os.path.normcase(str(value))

    def _root_for(self, path: Path) -> Path | None:
        for name in EDIT_ASSET_DIRECTORIES:
            root = self.data_dir / name
            try:
                if path.parent.resolve(strict=True) == root.resolve(strict=True):
                    return root
            except (OSError, RuntimeError):
                continue
        return None

    def _sum(self, values: Any) -> AssetMeasure:
        total = AssetMeasure()
        for value in values:
            total = self._plus(total, value)
        return total

    def _plus(self, left: AssetMeasure, right: AssetMeasure) -> AssetMeasure:
        return AssetMeasure(count=left.count + right.count, bytes=left.bytes + right.bytes)
