from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path

from app.core.config import AppConfig
from app.services.external_model_libraries import (
    ExternalModelLibraryError,
    ExternalModelLibraryRepository,
    ExternalModelLibraryService,
)
from app.services.model_library import ModelLibrary, ModelLibraryScanner
from app.services.model_library import ModelLibraryItem
from app.services.model_fingerprints import ModelFingerprintService


_cache_lock = threading.RLock()
_libraries: dict[str, ModelLibrary] = {}


def _cache_key(model_root: Path) -> str:
    return os.path.normcase(os.path.abspath(str(model_root)))


def get_model_library(model_root: Path) -> ModelLibrary:
    key = _cache_key(model_root)
    with _cache_lock:
        library = _libraries.get(key)
        if library is None:
            library = _scan_model_libraries(model_root)
            _libraries[key] = library
        return library


def refresh_model_library(model_root: Path) -> ModelLibrary:
    library = _scan_model_libraries(model_root)
    with _cache_lock:
        _libraries[_cache_key(model_root)] = library
    return library


def clear_model_library_cache() -> None:
    with _cache_lock:
        _libraries.clear()


def _scan_model_libraries(model_root: Path) -> ModelLibrary:
    local = ModelLibraryScanner(model_root).scan()
    config = AppConfig()
    fingerprints = ModelFingerprintService(config.data_dir / "config" / "model-fingerprints.json")
    local_items: list[ModelLibraryItem] = []
    for item in local.items:
        relative_path = item.relative_path.as_posix()
        try:
            fingerprint = fingerprints.fast_fingerprint(
                item.path,
                root_id="asteria-local",
                model_type=item.type,
                relative_path=relative_path,
            )
        except OSError:
            local_items.append(item)
            continue
        local_items.append(
            replace(
                item,
                root_id="asteria-local",
                stable_model_id=ModelFingerprintService.stable_model_id(
                    root_id="asteria-local",
                    model_type=item.type,
                    relative_path=relative_path,
                ),
                fingerprint_algorithm=fingerprint.algorithm,
                fingerprint_version=fingerprint.version,
                fingerprint_value=fingerprint.value,
            )
        )
    service = ExternalModelLibraryService(
        ExternalModelLibraryRepository(config.data_dir / "config" / "external-model-libraries.json"),
        protected_roots=(config.root_dir, config.data_dir),
    )
    try:
        external = service.scan()
    except ExternalModelLibraryError:
        return ModelLibrary(items=local_items, link_health=local.link_health)
    external_items = [
        ModelLibraryItem(
            type=item.model_type,
            name=Path(item.logical_name).name,
            path=item.execution_path,
            relative_path=Path("external") / item.root_id / item.model_type / Path(item.normalized_relative_path),
            size_bytes=item.size_bytes,
            metadata={
                "external_root_id": item.root_id,
                "external_relative_path": item.normalized_relative_path,
            },
            source="external",
            root_id=item.root_id,
            stable_model_id=item.stable_model_id,
            fingerprint_algorithm=item.fingerprint_algorithm,
            fingerprint_version=item.fingerprint_version,
            fingerprint_value=item.fingerprint_value,
        )
        for item in external.items
    ]
    items = [*local_items, *external_items]
    items.sort(key=lambda item: (item.type, item.name.casefold(), str(item.relative_path).casefold()))
    return ModelLibrary(items=items, link_health=local.link_health)
