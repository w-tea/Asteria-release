from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import threading


@dataclass(frozen=True)
class ModelFingerprint:
    algorithm: str
    version: int
    file_size: int
    mtime_ns: int
    root_id: str
    model_type: str
    relative_path: str
    value: str


class ModelFingerprintService:
    """Versioned fast identities plus an identity-verified strong-hash cache."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._lock = threading.RLock()

    def fast_fingerprint(
        self,
        path: Path,
        *,
        root_id: str,
        model_type: str,
        relative_path: str,
    ) -> ModelFingerprint:
        details = path.stat()
        normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
        if path.suffix.casefold() == ".safetensors":
            algorithm = "safetensors-header+size+mtime-v1"
            header = self._safetensors_header(path)
        else:
            algorithm = "file-identity+size+mtime-v1"
            header = b""
        identity = self._file_identity(path, details)
        digest = hashlib.sha256()
        digest.update(algorithm.encode("ascii"))
        digest.update(b"\0")
        digest.update(root_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model_type.encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalized.casefold().encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(identity).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(details.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(details.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(header)
        return ModelFingerprint(
            algorithm=algorithm,
            version=1,
            file_size=details.st_size,
            mtime_ns=details.st_mtime_ns,
            root_id=root_id,
            model_type=model_type,
            relative_path=normalized,
            value=digest.hexdigest(),
        )

    def strong_hash(self, path: Path, *, refresh: bool = False) -> str:
        details = path.stat()
        identity = repr(self._file_identity(path, details))
        cache_key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if not refresh:
            with self._lock:
                cache = self._load_cache()
                cached = cache.get(cache_key)
                if (
                    isinstance(cached, dict)
                    and cached.get("identity") == identity
                    and cached.get("size") == details.st_size
                    and cached.get("mtime_ns") == details.st_mtime_ns
                    and isinstance(cached.get("sha256"), str)
                ):
                    return cached["sha256"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._lock:
            cache = self._load_cache()
            cache[cache_key] = {
                "identity": identity,
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "sha256": value,
            }
            self._save_cache(cache)
        return value

    @staticmethod
    def stable_model_id(*, root_id: str, model_type: str, relative_path: str) -> str:
        normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix().casefold()
        return "model:" + hashlib.sha256(
            f"{root_id}\0{model_type}\0{normalized}".encode("utf-8")
        ).hexdigest()[:32]

    @staticmethod
    def _safetensors_header(path: Path) -> bytes:
        try:
            with path.open("rb") as handle:
                raw_size = handle.read(8)
                if len(raw_size) != 8:
                    return b""
                size = int.from_bytes(raw_size, "little")
                if size <= 0 or size > 16 * 1024 * 1024:
                    return b""
                return handle.read(size)
        except OSError:
            return b""

    @staticmethod
    def _file_identity(path: Path, details: os.stat_result) -> tuple[object, ...]:
        if details.st_ino:
            return ("inode", details.st_dev, details.st_ino)
        return ("realpath", os.path.normcase(os.path.realpath(path)))

    def _load_cache(self) -> dict[str, dict[str, object]]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_cache(self, cache: dict[str, dict[str, object]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{self.cache_path.name}.",
            suffix=".tmp",
            dir=self.cache_path.parent,
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(cache, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.cache_path)
        finally:
            temp.unlink(missing_ok=True)
