from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from threading import Lock, RLock
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


THUMBNAIL_EDGE = 384
THUMBNAIL_ENCODER_REVISION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class LibraryThumbnailError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ThumbnailCacheEntry:
    path: Path
    size: int
    accessed_at_ns: int


_registry_lock = RLock()
_key_locks: dict[str, Lock] = {}
_active_keys: set[str] = set()


class LibraryThumbnailService:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def thumbnail_for(self, item: dict, accepted_roots: set[Path]) -> Path:
        source = self._source_for(item)
        resolved_source = self._validated_source(source, accepted_roots)
        try:
            details = resolved_source.stat()
        except OSError as error:
            raise LibraryThumbnailError("library_thumbnail_source_missing") from error
        identity = json.dumps(
            {
                "path": os.path.normcase(str(resolved_source)),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "edge": THUMBNAIL_EDGE,
                "encoder": THUMBNAIL_ENCODER_REVISION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = sha256(identity.encode("utf-8")).hexdigest()
        target = self.cache_root / f"{key}.webp"
        key_lock = self._lock_for(key)
        with key_lock:
            with _registry_lock:
                _active_keys.add(key)
            try:
                if self._usable_cache_file(target):
                    self._touch(target)
                    return target
                self._encode_atomic(resolved_source, target)
                return target
            finally:
                with _registry_lock:
                    _active_keys.discard(key)

    def inventory(self) -> list[ThumbnailCacheEntry]:
        try:
            candidates = list(self.cache_root.glob("*.webp"))
        except OSError:
            return []
        entries: list[ThumbnailCacheEntry] = []
        for path in candidates:
            if not self._is_owned_cache_path(path):
                continue
            try:
                details = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            entries.append(
                ThumbnailCacheEntry(
                    path=path,
                    size=int(details.st_size),
                    accessed_at_ns=int(details.st_mtime_ns),
                )
            )
        return entries

    def cleanup(self, *, max_entries: int = 2000, max_bytes: int = 512 * 1024 * 1024) -> tuple[int, int]:
        entries = sorted(self.inventory(), key=lambda item: item.accessed_at_ns, reverse=True)
        retained_count = 0
        retained_bytes = 0
        removed_count = 0
        removed_bytes = 0
        with _registry_lock:
            active = set(_active_keys)
        for entry in entries:
            key = entry.path.stem
            must_keep = key in active
            if must_keep:
                continue
            within_bounds = retained_count < max_entries and retained_bytes + entry.size <= max_bytes
            if within_bounds:
                retained_count += 1
                retained_bytes += entry.size
                continue
            try:
                entry.path.unlink()
            except OSError:
                continue
            removed_count += 1
            removed_bytes += entry.size
        return removed_count, removed_bytes

    def _source_for(self, item: dict) -> Path:
        output_paths = [
            Path(value)
            for value in item.get("output_paths", [])
            if isinstance(value, str) and Path(value).suffix.lower() in IMAGE_SUFFIXES
        ]
        primary_outputs = item.get("settings", {}).get("primary_outputs")
        if isinstance(primary_outputs, list) and primary_outputs:
            for output in primary_outputs:
                if not isinstance(output, dict):
                    continue
                value = output.get("path")
                if not isinstance(value, str) or Path(value).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                candidate = Path(value)
                if not self._same_registered_path(candidate, output_paths):
                    raise LibraryThumbnailError("library_thumbnail_source_not_registered")
                return candidate
            raise LibraryThumbnailError("library_thumbnail_source_missing")
        if output_paths:
            return output_paths[0]
        raise LibraryThumbnailError("library_thumbnail_source_missing")

    def _validated_source(self, source: Path, accepted_roots: set[Path]) -> Path:
        try:
            raw_source = Path(os.path.abspath(os.fspath(source)))
            resolved_source = raw_source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise LibraryThumbnailError("library_thumbnail_source_missing") from error
        if not resolved_source.is_file() or resolved_source.suffix.lower() not in IMAGE_SUFFIXES:
            raise LibraryThumbnailError("library_thumbnail_source_missing")
        accepted = False
        for root in accepted_roots:
            try:
                raw_root = Path(os.path.abspath(os.fspath(root)))
                resolved_root = raw_root.resolve(strict=True)
                resolved_source.relative_to(resolved_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if self._contains_reparse_component(raw_source, raw_root):
                continue
            accepted = True
            break
        if not accepted:
            raise LibraryThumbnailError("library_thumbnail_source_not_registered")
        return resolved_source

    def _contains_reparse_component(self, source: Path, root: Path) -> bool:
        try:
            source.relative_to(root)
        except ValueError:
            return True
        current = source
        while True:
            try:
                details = os.lstat(current)
            except OSError:
                return True
            attributes = getattr(details, "st_file_attributes", 0)
            is_reparse = bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            is_junction = False
            junction_check = getattr(os.path, "isjunction", None)
            if junction_check is not None:
                try:
                    is_junction = bool(junction_check(current))
                except OSError:
                    return True
            if stat.S_ISLNK(details.st_mode) or is_reparse or is_junction:
                return True
            if current == root:
                return False
            current = current.parent

    def _same_registered_path(self, candidate: Path, registered: list[Path]) -> bool:
        try:
            candidate_value = os.path.normcase(str(candidate.resolve(strict=False)))
        except (OSError, RuntimeError):
            return False
        for path in registered:
            try:
                if os.path.normcase(str(path.resolve(strict=False))) == candidate_value:
                    return True
            except (OSError, RuntimeError):
                continue
        return False

    def _encode_atomic(self, source: Path, target: Path) -> None:
        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            if not self._cache_root_is_safe():
                raise LibraryThumbnailError("library_thumbnail_cache_unavailable")
            temporary = self.cache_root / f"{target.stem}.{uuid4().hex}.tmp"
            with Image.open(source) as opened:
                opened.load()
                thumbnail = opened.convert("RGB")
                thumbnail.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                thumbnail.save(buffer, format="WEBP", quality=82, method=6)
            temporary.write_bytes(buffer.getvalue())
            os.replace(temporary, target)
        except LibraryThumbnailError:
            raise
        except (OSError, UnidentifiedImageError, ValueError) as error:
            try:
                if "temporary" in locals() and temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
            raise LibraryThumbnailError("library_thumbnail_encode_failed") from error

    def _cache_root_is_safe(self) -> bool:
        try:
            root = self.cache_root.resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return not self._contains_reparse_component(root, root)

    def _usable_cache_file(self, target: Path) -> bool:
        if not self._is_owned_cache_path(target):
            return False
        try:
            return target.is_file() and target.stat().st_size > 0
        except OSError:
            return False

    def _is_owned_cache_path(self, path: Path) -> bool:
        if path.suffix.lower() != ".webp" or len(path.stem) != 64:
            return False
        if any(character not in "0123456789abcdef" for character in path.stem):
            return False
        try:
            return path.parent.resolve(strict=True) == self.cache_root.resolve(strict=True)
        except (OSError, RuntimeError):
            return False

    def _touch(self, path: Path) -> None:
        try:
            os.utime(path, None)
        except OSError:
            pass

    def _lock_for(self, key: str) -> Lock:
        with _registry_lock:
            lock = _key_locks.get(key)
            if lock is None:
                lock = Lock()
                _key_locks[key] = lock
            return lock
