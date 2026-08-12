from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import threading
import time
from typing import Callable, Literal
from uuid import uuid4

from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import MODEL_TYPE_EXTENSIONS


ROOT_LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    "asteria_comfy": {
        "checkpoint": ("checkpoints", "diffusion_models", "unet", "merged/checkpoints"),
        "vae": ("vae", "vae_approx"),
        "text_encoder": ("text_encoders", "clip"),
        "tokenizer": ("tokenizers", "tokenizer"),
        "lora": ("loras", "lora", "merged/lora"),
        "embedding": ("embeddings",),
        "controlnet": ("controlnet", "controlnets"),
        "detailer": ("adetailer", "bbox", "ultralytics/bbox", "segm", "ultralytics/segm"),
        "upscaler": ("upscalers", "esrgan", "realesrgan"),
    },
    "forge_a1111": {
        "checkpoint": ("models/Stable-diffusion",),
        "vae": ("models/VAE",),
        "text_encoder": ("models/text_encoder",),
        "tokenizer": ("models/tokenizer",),
        "lora": ("models/Lora", "models/LyCORIS"),
        "embedding": ("embeddings",),
        "controlnet": ("models/ControlNet",),
        "detailer": (
            "models/adetailer",
            "models/bbox",
            "models/ultralytics/bbox",
            "models/segm",
            "models/ultralytics/segm",
        ),
        "upscaler": ("models/ESRGAN", "models/RealESRGAN", "models/SwinIR"),
    },
    "shared_models": {
        "checkpoint": ("StableDiffusion", "Stable-diffusion"),
        "vae": ("VAE", "vae"),
        "text_encoder": ("text_encoder", "text_encoders", "clip"),
        "tokenizer": ("tokenizer", "tokenizers"),
        "lora": ("Lora", "LyCORIS"),
        "embedding": ("Embeddings",),
        "controlnet": ("ControlNet",),
        "detailer": ("adetailer", "bbox", "ultralytics/bbox", "segm", "ultralytics/segm"),
        "upscaler": ("ESRGAN", "RealESRGAN", "SwinIR"),
    },
}


class ExternalModelLibraryError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalModelRoot:
    root_id: str
    display_name: str
    path: str | None
    layout: Literal["asteria_comfy", "forge_a1111", "shared_models", "custom"]
    mappings: dict[str, tuple[str, ...]]
    priority: int
    enabled: bool
    revision: int
    created_at: str
    updated_at: str

    @property
    def path_hint(self) -> str:
        if not self.path:
            return "Unbound"
        path = Path(self.path)
        return path.name or path.anchor or "External"


@dataclass(frozen=True)
class ExternalModelItem:
    stable_model_id: str
    root_id: str
    model_type: str
    logical_name: str
    normalized_relative_path: str
    execution_path: Path
    size_bytes: int
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str


@dataclass(frozen=True)
class ExternalModelCollision:
    model_type: str
    logical_name: str
    winner_root_id: str
    shadowed_root_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExternalModelScanResult:
    items: tuple[ExternalModelItem, ...]
    collisions: tuple[ExternalModelCollision, ...]
    root_statuses: dict[str, str]
    truncated: bool


@dataclass(frozen=True)
class ExternalModelRootPreview:
    path: str
    path_hint: str
    suggested_name: str
    detected_layout: Literal["asteria_comfy", "forge_a1111", "shared_models"] | None
    counts: dict[str, int]
    total_items: int
    truncated: bool


class ExternalModelLibraryRepository:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def list(self) -> list[ExternalModelRoot]:
        with self._lock:
            data = self._read()
        roots: list[ExternalModelRoot] = []
        for value in data.get("roots", []):
            if not isinstance(value, dict):
                continue
            mappings = {
                str(key): tuple(str(item) for item in entries)
                for key, entries in value.get("mappings", {}).items()
                if isinstance(entries, list)
            }
            roots.append(
                ExternalModelRoot(
                    root_id=str(value["root_id"]),
                    display_name=str(value["display_name"]),
                    path=str(value["path"]) if value.get("path") else None,
                    layout=str(value["layout"]),
                    mappings=mappings,
                    priority=int(value.get("priority", 0)),
                    enabled=bool(value.get("enabled", True)),
                    revision=int(value.get("revision", 1)),
                    created_at=str(value["created_at"]),
                    updated_at=str(value["updated_at"]),
                )
            )
        return sorted(roots, key=lambda root: (-root.priority, root.display_name.casefold(), root.root_id))

    def save_all(self, roots: list[ExternalModelRoot]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "roots": [
                {
                    **asdict(root),
                    "mappings": {key: list(value) for key, value in root.mappings.items()},
                }
                for root in roots
            ],
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temp = Path(temp_name)
            try:
                with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, self.path)
            finally:
                temp.unlink(missing_ok=True)

    def _read(self) -> dict[str, object]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": self.SCHEMA_VERSION, "roots": []}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalModelLibraryError("registry_unreadable") from exc
        if not isinstance(value, dict) or value.get("schema_version") != self.SCHEMA_VERSION:
            raise ExternalModelLibraryError("registry_schema_unsupported")
        return value


class ExternalModelLibraryService:
    def __init__(
        self,
        repository: ExternalModelLibraryRepository,
        *,
        protected_roots: tuple[Path, ...],
        fingerprint_service: ModelFingerprintService | None = None,
        active_root_ids: Callable[[], set[str]] | None = None,
    ) -> None:
        self.repository = repository
        self.protected_roots = tuple(self._canonical(root, strict=False) for root in protected_roots)
        self.fingerprints = fingerprint_service or ModelFingerprintService(
            repository.path.with_name("model-fingerprints.json")
        )
        self.active_root_ids = active_root_ids or (lambda: set())

    def list_roots(self) -> list[ExternalModelRoot]:
        return self.repository.list()

    def add_root(
        self,
        *,
        display_name: str,
        path: Path,
        layout: str,
        mappings: dict[str, list[str] | tuple[str, ...]] | None = None,
        priority: int = 0,
        root_id: str | None = None,
    ) -> ExternalModelRoot:
        roots = self.repository.list()
        canonical = self._validate_new_binding(path, roots)
        normalized_mappings = self._normalize_mappings(layout, mappings)
        now = datetime.now(UTC).isoformat()
        root = ExternalModelRoot(
            root_id=root_id or uuid4().hex,
            display_name=self._display_name(display_name),
            path=str(canonical),
            layout=layout,
            mappings=normalized_mappings,
            priority=int(priority),
            enabled=True,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        if any(existing.root_id == root.root_id for existing in roots):
            raise ExternalModelLibraryError("root_id_conflict")
        self.repository.save_all([*roots, root])
        return root

    def preview_root(
        self,
        path: Path,
        *,
        exclude_root_id: str | None = None,
        max_items: int = 10_000,
        timeout_seconds: float = 10.0,
    ) -> ExternalModelRootPreview:
        roots = [
            root
            for root in self.repository.list()
            if root.root_id != exclude_root_id
        ]
        canonical = self._validate_new_binding(path, roots)
        scores: dict[str, int] = {}
        for layout, mappings in ROOT_LAYOUTS.items():
            scores[layout] = sum(
                1
                for directories in mappings.values()
                for directory in directories
                if canonical.joinpath(*PurePosixPath(directory).parts).is_dir()
            )
        highest = max(scores.values(), default=0)
        winners = [layout for layout, score in scores.items() if score == highest and score > 0]
        detected_layout = winners[0] if len(winners) == 1 else None
        counts: dict[str, int] = {}
        truncated = False
        if detected_layout:
            started = time.monotonic()
            total = 0
            for model_type, directories in ROOT_LAYOUTS[detected_layout].items():
                extensions = MODEL_TYPE_EXTENSIONS.get(model_type)
                if not extensions:
                    continue
                seen: set[str] = set()
                for directory in directories:
                    self._check_scan_boundary(started, timeout_seconds, None)
                    base = canonical.joinpath(*PurePosixPath(directory).parts)
                    if not base.is_dir():
                        continue
                    try:
                        entries = base.rglob("*")
                        for item in entries:
                            self._check_scan_boundary(started, timeout_seconds, None)
                            if not item.is_file() or item.suffix.casefold() not in extensions:
                                continue
                            identity = os.path.normcase(str(item.resolve(strict=True)))
                            if identity in seen:
                                continue
                            seen.add(identity)
                            total += 1
                            if total >= max_items:
                                truncated = True
                                break
                    except OSError:
                        continue
                    if truncated:
                        break
                if seen:
                    counts[model_type] = len(seen)
                if truncated:
                    break
        return ExternalModelRootPreview(
            path=str(canonical),
            path_hint=canonical.name or canonical.anchor or "External",
            suggested_name=canonical.name or "External models",
            detected_layout=detected_layout,
            counts=counts,
            total_items=sum(counts.values()),
            truncated=truncated,
        )

    def update_root(
        self,
        root_id: str,
        *,
        display_name: str | None = None,
        enabled: bool | None = None,
        priority: int | None = None,
        layout: str | None = None,
        mappings: dict[str, list[str] | tuple[str, ...]] | None = None,
    ) -> ExternalModelRoot:
        roots = self.repository.list()
        current = self._require_root(roots, root_id)
        if enabled is False and current.enabled:
            self._ensure_idle(root_id)
        next_layout = layout or current.layout
        next_mappings = (
            self._normalize_mappings(next_layout, mappings)
            if layout is not None or mappings is not None
            else current.mappings
        )
        updated = replace(
            current,
            display_name=self._display_name(display_name) if display_name is not None else current.display_name,
            enabled=current.enabled if enabled is None else enabled,
            priority=current.priority if priority is None else int(priority),
            layout=next_layout,
            mappings=next_mappings,
            revision=current.revision + 1,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.repository.save_all([updated if root.root_id == root_id else root for root in roots])
        return updated

    def rebind_root(self, root_id: str, path: Path) -> ExternalModelRoot:
        self._ensure_idle(root_id)
        roots = self.repository.list()
        current = self._require_root(roots, root_id)
        canonical = self._validate_new_binding(path, [root for root in roots if root.root_id != root_id])
        updated = replace(
            current,
            path=str(canonical),
            revision=current.revision + 1,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.repository.save_all([updated if root.root_id == root_id else root for root in roots])
        return updated

    def remove_root(self, root_id: str) -> None:
        self._ensure_idle(root_id)
        roots = self.repository.list()
        self._require_root(roots, root_id)
        self.repository.save_all([root for root in roots if root.root_id != root_id])

    def create_owned_link(
        self,
        root_id: str,
        *,
        model_type: str,
        local_models_root: Path,
        link_name: str,
    ) -> Path:
        root = self._require_root(self.repository.list(), root_id)
        if not root.path or not root.enabled:
            raise ExternalModelLibraryError("root_offline")
        safe_name = self._link_name(link_name)
        external_base = self._canonical(Path(root.path), strict=True)
        directories = root.mappings.get(model_type, ())
        local_directories = ROOT_LAYOUTS["asteria_comfy"].get(model_type, ())
        if not directories or not local_directories:
            raise ExternalModelLibraryError("model_type_not_mapped")
        target = self._canonical(
            external_base.joinpath(*PurePosixPath(directories[0]).parts),
            strict=True,
        )
        if not self._contains(external_base, target):
            raise ExternalModelLibraryError("link_target_escaped")
        local_root = self._canonical(local_models_root, strict=False)
        parent = local_root.joinpath(*PurePosixPath(local_directories[0]).parts)
        parent.mkdir(parents=True, exist_ok=True)
        link = parent / safe_name
        if not self._contains(local_root, link) or os.path.lexists(link):
            raise ExternalModelLibraryError("link_object_conflict")
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError as exc:
            raise ExternalModelLibraryError("link_creation_failed") from exc
        receipt = self._load_link_receipts()
        receipt[self._link_receipt_key(root_id, model_type, safe_name)] = {
            "root_id": root_id,
            "model_type": model_type,
            "link_name": safe_name,
            "link_path": str(link),
            "target_path": str(target),
        }
        try:
            self._save_link_receipts(receipt)
        except Exception:
            link.unlink(missing_ok=True)
            raise
        return link

    def remove_owned_link(
        self,
        root_id: str,
        *,
        model_type: str,
        local_models_root: Path,
        link_name: str,
    ) -> None:
        self._ensure_idle(root_id)
        safe_name = self._link_name(link_name)
        key = self._link_receipt_key(root_id, model_type, safe_name)
        receipts = self._load_link_receipts()
        receipt = receipts.get(key)
        if not isinstance(receipt, dict):
            raise ExternalModelLibraryError("owned_link_receipt_missing")
        local_root = self._canonical(local_models_root, strict=False)
        link = Path(str(receipt.get("link_path", "")))
        if not self._contains(local_root, self._canonical(link.parent, strict=False)):
            raise ExternalModelLibraryError("owned_link_identity_changed")
        if not os.path.lexists(link) or not link.is_symlink():
            raise ExternalModelLibraryError("owned_link_identity_changed")
        try:
            actual_target = self._canonical(link, strict=True)
            expected_target = self._canonical(Path(str(receipt.get("target_path", ""))), strict=True)
        except ExternalModelLibraryError as exc:
            raise ExternalModelLibraryError("owned_link_identity_changed") from exc
        if actual_target != expected_target:
            raise ExternalModelLibraryError("owned_link_identity_changed")
        link.unlink()
        receipts.pop(key, None)
        self._save_link_receipts(receipts)

    def export_registry(self, *, include_machine_paths: bool = False) -> dict[str, object]:
        return {
            "schema_version": 1,
            "roots": [
                {
                    "root_id": root.root_id,
                    "display_name": root.display_name,
                    "layout": root.layout,
                    "mappings": {key: list(value) for key, value in root.mappings.items()},
                    "priority": root.priority,
                    "enabled": root.enabled,
                    "bound": False,
                    **({"path": root.path, "bound": root.path is not None} if include_machine_paths else {}),
                }
                for root in self.repository.list()
            ],
        }

    def import_registry(self, payload: dict[str, object]) -> list[ExternalModelRoot]:
        imported: list[ExternalModelRoot] = []
        existing_roots = self.repository.list()
        now = datetime.now(UTC).isoformat()
        for value in payload.get("roots", []):
            if not isinstance(value, dict):
                continue
            layout = str(value.get("layout", "custom"))
            path: str | None = None
            raw_path = value.get("path")
            if value.get("bound") is True and isinstance(raw_path, str) and raw_path.strip():
                canonical = self._validate_new_binding(
                    Path(raw_path),
                    [*existing_roots, *imported],
                )
                path = str(canonical)
            imported.append(
                ExternalModelRoot(
                    root_id=str(value.get("root_id") or uuid4().hex),
                    display_name=self._display_name(str(value.get("display_name", "External models"))),
                    path=path,
                    layout=layout,
                    mappings=self._normalize_mappings(layout, value.get("mappings")),
                    priority=int(value.get("priority", 0)),
                    enabled=bool(value.get("enabled", True)),
                    revision=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        existing = {root.root_id: root for root in existing_roots}
        for root in imported:
            existing.setdefault(root.root_id, root)
        self.repository.save_all(list(existing.values()))
        return imported

    def scan(
        self,
        *,
        max_items: int = 10_000,
        timeout_seconds: float = 30.0,
        cancel_event: threading.Event | None = None,
    ) -> ExternalModelScanResult:
        started = time.monotonic()
        statuses: dict[str, str] = {}
        candidates: dict[tuple[str, str], list[ExternalModelItem]] = {}
        for root in self.repository.list():
            self._check_scan_boundary(started, timeout_seconds, cancel_event)
            if not root.enabled:
                statuses[root.root_id] = "disabled"
                continue
            if not root.path:
                statuses[root.root_id] = "offline"
                continue
            try:
                bound = self._canonical(Path(root.path), strict=True)
            except ExternalModelLibraryError:
                statuses[root.root_id] = "offline"
                continue
            statuses[root.root_id] = "online"
            for model_type, directories in root.mappings.items():
                extensions = MODEL_TYPE_EXTENSIONS.get(model_type)
                if not extensions:
                    continue
                seen: set[tuple[int, int] | str] = set()
                for directory in directories:
                    self._check_scan_boundary(started, timeout_seconds, cancel_event)
                    base = bound.joinpath(*PurePosixPath(directory).parts)
                    try:
                        entries = base.rglob("*")
                    except OSError:
                        statuses[root.root_id] = "offline"
                        continue
                    try:
                        for path in entries:
                            self._check_scan_boundary(started, timeout_seconds, cancel_event)
                            try:
                                if not path.is_file() or path.suffix.casefold() not in extensions:
                                    continue
                                details = path.stat()
                                resolved = path.resolve(strict=True)
                                resolved.relative_to(bound)
                                relative = path.relative_to(base).as_posix()
                            except (OSError, ValueError):
                                continue
                            identity: tuple[int, int] | str = (
                                (details.st_dev, details.st_ino)
                                if details.st_ino
                                else os.path.normcase(str(resolved))
                            )
                            if identity in seen:
                                continue
                            seen.add(identity)
                            fingerprint = self.fingerprints.fast_fingerprint(
                                resolved,
                                root_id=root.root_id,
                                model_type=model_type,
                                relative_path=relative,
                            )
                            item = ExternalModelItem(
                                stable_model_id=self.fingerprints.stable_model_id(
                                    root_id=root.root_id,
                                    model_type=model_type,
                                    relative_path=relative,
                                ),
                                root_id=root.root_id,
                                model_type=model_type,
                                logical_name=relative,
                                normalized_relative_path=relative,
                                execution_path=resolved,
                                size_bytes=details.st_size,
                                fingerprint_algorithm=fingerprint.algorithm,
                                fingerprint_version=fingerprint.version,
                                fingerprint_value=fingerprint.value,
                            )
                            candidates.setdefault((model_type, relative.casefold()), []).append(item)
                    except OSError:
                        statuses[root.root_id] = "offline"

        ordered_roots = {root.root_id: index for index, root in enumerate(self.repository.list())}
        winners: list[ExternalModelItem] = []
        collisions: list[ExternalModelCollision] = []
        for key in sorted(candidates):
            values = sorted(candidates[key], key=lambda item: ordered_roots[item.root_id])
            winners.append(values[0])
            if len(values) > 1:
                collisions.append(
                    ExternalModelCollision(
                        model_type=values[0].model_type,
                        logical_name=values[0].logical_name,
                        winner_root_id=values[0].root_id,
                        shadowed_root_ids=tuple(item.root_id for item in values[1:]),
                    )
                )
                for item in values:
                    statuses[item.root_id] = "conflict"
        winners.sort(key=lambda item: (item.model_type, item.logical_name.casefold(), item.root_id))
        return ExternalModelScanResult(
            items=tuple(winners[:max_items]),
            collisions=tuple(collisions),
            root_statuses=statuses,
            truncated=len(winners) > max_items,
        )

    def _validate_new_binding(self, path: Path, roots: list[ExternalModelRoot]) -> Path:
        canonical = self._canonical(path, strict=True)
        if canonical == Path(canonical.anchor):
            raise ExternalModelLibraryError("filesystem_root_forbidden")
        for protected in self.protected_roots:
            if self._contains(protected, canonical):
                raise ExternalModelLibraryError("protected_root")
        for root in roots:
            if not root.path:
                continue
            existing = self._canonical(Path(root.path), strict=False)
            if self._contains(existing, canonical) or self._contains(canonical, existing):
                raise ExternalModelLibraryError("nested_root")
        return canonical

    @staticmethod
    def _canonical(path: Path, *, strict: bool) -> Path:
        try:
            resolved = path.expanduser().resolve(strict=strict)
        except (OSError, RuntimeError) as exc:
            raise ExternalModelLibraryError("root_unavailable") from exc
        if strict and not resolved.is_dir():
            raise ExternalModelLibraryError("root_not_directory")
        return resolved

    @staticmethod
    def _contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def _display_name(value: str | None) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ExternalModelLibraryError("display_name_required")
        return normalized[:120]

    @staticmethod
    def _normalize_mappings(
        layout: str,
        mappings: object,
    ) -> dict[str, tuple[str, ...]]:
        if layout not in {"asteria_comfy", "forge_a1111", "shared_models", "custom"}:
            raise ExternalModelLibraryError("layout_invalid")
        source = ROOT_LAYOUTS.get(layout) if mappings is None else mappings
        if not isinstance(source, dict):
            raise ExternalModelLibraryError("mappings_required")
        normalized: dict[str, tuple[str, ...]] = {}
        for model_type, values in source.items():
            if model_type not in MODEL_TYPE_EXTENSIONS or not isinstance(values, (list, tuple)):
                continue
            directories: list[str] = []
            for value in values:
                relative = PurePosixPath(str(value).replace("\\", "/"))
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise ExternalModelLibraryError("mapping_path_invalid")
                directories.append(relative.as_posix())
            if directories:
                normalized[str(model_type)] = tuple(dict.fromkeys(directories))
        if not normalized:
            raise ExternalModelLibraryError("mappings_required")
        return normalized

    @staticmethod
    def _require_root(roots: list[ExternalModelRoot], root_id: str) -> ExternalModelRoot:
        try:
            return next(root for root in roots if root.root_id == root_id)
        except StopIteration as exc:
            raise ExternalModelLibraryError("root_not_found") from exc

    def _ensure_idle(self, root_id: str) -> None:
        if root_id in self.active_root_ids():
            raise ExternalModelLibraryError("root_in_use")

    @property
    def _link_receipts_path(self) -> Path:
        return self.repository.path.with_name("external-model-links.json")

    @staticmethod
    def _link_name(value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or normalized in {".", ".."}
            or "/" in normalized
            or "\\" in normalized
            or len(normalized) > 120
        ):
            raise ExternalModelLibraryError("link_name_invalid")
        return normalized

    @staticmethod
    def _link_receipt_key(root_id: str, model_type: str, link_name: str) -> str:
        return f"{root_id}:{model_type}:{link_name.casefold()}"

    def _load_link_receipts(self) -> dict[str, object]:
        try:
            value = json.loads(self._link_receipts_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalModelLibraryError("owned_link_receipts_unreadable") from exc
        return value if isinstance(value, dict) else {}

    def _save_link_receipts(self, receipts: dict[str, object]) -> None:
        path = self._link_receipts_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(receipts, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _check_scan_boundary(
        started: float,
        timeout_seconds: float,
        cancel_event: threading.Event | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise ExternalModelLibraryError("scan_cancelled")
        if time.monotonic() - started > timeout_seconds:
            raise ExternalModelLibraryError("scan_timeout")
