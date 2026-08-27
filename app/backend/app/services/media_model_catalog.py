from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Protocol, Sequence

from app.schemas.media_model_catalog import (
    MediaModelClassification,
    MediaModelDescriptor,
    MediaModelDomain,
)
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibrary, ModelLibraryItem


MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


class SafetensorsHeaderError(ValueError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class SafetensorsTensorInfo:
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SafetensorsHeader:
    metadata: dict[str, str]
    tensors: dict[str, SafetensorsTensorInfo]


class ModelAssetClassifier(Protocol):
    @property
    def domain(self) -> MediaModelDomain: ...

    def classify(
        self,
        item: ModelLibraryItem,
        header: SafetensorsHeader | None,
    ) -> MediaModelClassification: ...


def read_safetensors_header(path: Path) -> SafetensorsHeader:
    if path.suffix.casefold() != ".safetensors":
        raise SafetensorsHeaderError("unsupported_model_container")
    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise SafetensorsHeaderError("safetensors_header_truncated")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0:
                raise SafetensorsHeaderError("safetensors_header_invalid")
            if header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise SafetensorsHeaderError("safetensors_header_too_large")
            raw_header = handle.read(header_size)
    except SafetensorsHeaderError:
        raise
    except OSError as exc:
        raise SafetensorsHeaderError("safetensors_header_unreadable") from exc
    if len(raw_header) != header_size:
        raise SafetensorsHeaderError("safetensors_header_truncated")
    try:
        parsed = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SafetensorsHeaderError("safetensors_header_invalid") from exc
    if not isinstance(parsed, dict):
        raise SafetensorsHeaderError("safetensors_header_invalid")

    raw_metadata = parsed.get("__metadata__", {})
    metadata = (
        {str(key): str(value) for key, value in raw_metadata.items() if value is not None}
        if isinstance(raw_metadata, dict)
        else {}
    )
    tensors: dict[str, SafetensorsTensorInfo] = {}
    for name, raw_info in parsed.items():
        if name == "__metadata__" or not isinstance(raw_info, dict):
            continue
        dtype = raw_info.get("dtype")
        shape = raw_info.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            continue
        if not all(isinstance(value, int) and value >= 0 for value in shape):
            continue
        tensors[str(name)] = SafetensorsTensorInfo(dtype=dtype, shape=tuple(shape))
    return SafetensorsHeader(metadata=metadata, tensors=tensors)


class MediaModelCatalogService:
    def __init__(self, classifiers: Sequence[ModelAssetClassifier] | None = None) -> None:
        if classifiers is None:
            from app.services.anima_model_classifier import AnimaModelClassifier
            from app.services.music_model_classifier import MusicModelClassifier
            from app.services.video_model_classifier import VideoModelClassifier

            classifiers = (
                VideoModelClassifier(),
                MusicModelClassifier(),
                AnimaModelClassifier(),
            )
        self._classifiers = tuple(classifiers)

    def classify_library(
        self,
        library: ModelLibrary,
        *,
        domain: MediaModelDomain | str,
    ) -> list[MediaModelDescriptor]:
        normalized_domain = MediaModelDomain(domain)
        classifier = next(
            (candidate for candidate in self._classifiers if candidate.domain == normalized_domain),
            None,
        )
        if classifier is None:
            return []

        descriptors: list[MediaModelDescriptor] = []
        for item in library.items:
            header: SafetensorsHeader | None = None
            header_error: str | None = None
            try:
                header = read_safetensors_header(item.path)
            except SafetensorsHeaderError as exc:
                header_error = exc.reason_code
            classification = classifier.classify(item, header)
            if header_error and classification.reason_code is None:
                classification = classification.model_copy(
                    update={"reason_code": header_error}
                )
            stable_asset_id = item.stable_model_id or ModelFingerprintService.stable_model_id(
                root_id=item.root_id,
                model_type=item.type,
                relative_path=item.relative_path.as_posix(),
            )
            descriptors.append(
                MediaModelDescriptor(
                    **classification.model_dump(),
                    stable_asset_id=stable_asset_id,
                    domain=normalized_domain,
                    display_name=item.name,
                    size_bytes=item.size_bytes,
                    source=item.source,
                    root_id=item.root_id,
                    fingerprint_algorithm=item.fingerprint_algorithm,
                    fingerprint_version=item.fingerprint_version,
                    fingerprint_value=item.fingerprint_value,
                )
            )
        descriptors.sort(
            key=lambda item: (
                item.display_name.casefold(),
                item.stable_asset_id,
            )
        )
        return descriptors
