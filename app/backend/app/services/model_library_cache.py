from __future__ import annotations

import os
import threading
from pathlib import Path

from app.services.model_library import ModelLibrary, ModelLibraryScanner


_cache_lock = threading.RLock()
_libraries: dict[str, ModelLibrary] = {}


def _cache_key(model_root: Path) -> str:
    return os.path.normcase(os.path.abspath(str(model_root)))


def get_model_library(model_root: Path) -> ModelLibrary:
    key = _cache_key(model_root)
    with _cache_lock:
        library = _libraries.get(key)
        if library is None:
            library = ModelLibraryScanner(model_root).scan()
            _libraries[key] = library
        return library


def refresh_model_library(model_root: Path) -> ModelLibrary:
    library = ModelLibraryScanner(model_root).scan()
    with _cache_lock:
        _libraries[_cache_key(model_root)] = library
    return library


def clear_model_library_cache() -> None:
    with _cache_lock:
        _libraries.clear()
