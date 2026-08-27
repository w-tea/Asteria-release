from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Callable, Mapping, Protocol

from app.core.config import AppConfig
from app.schemas.media_model_catalog import (
    MediaModelCompatibility,
    MediaModelComponentRole,
    MediaModelDescriptor,
)
from app.schemas.video import VideoRequest
from app.schemas.video_upscale import VideoUpscaleModelReceipt
from app.services.comfy_model_paths import comfy_model_name
from app.services.media_model_catalog import MediaModelCatalogService
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibrary, ModelLibraryItem, ModelLibraryScanner
from app.services.model_library_cache import get_model_library, refresh_model_library
from app.services.model_storage import resolve_model_root
from app.services.release_manifest import ReleaseManifestRepository
from app.services.video_model_profiles import VideoModelProfileRepository
from app.services.video_packet import VideoModelPreflightError


VIDEO_RUNTIME_MODEL_ROOT_ID = "minimax-h3-runtime"

_ROLE_TO_MODEL_TYPE = {
    "diffusion_model": "checkpoint",
    "text_encoder": "text_encoder",
    "video_vae": "vae",
    "audio_vae": "vae",
    "lora": "lora",
    "latent_upscaler": "upscaler",
}
_ROLE_TO_COMFY_FOLDER = {
    "diffusion_model": "diffusion_models",
    "text_encoder": "text_encoders",
    "video_vae": "vae",
    "audio_vae": "vae",
    "lora": "loras",
    "latent_upscaler": "latent_upscale_models",
}
_MODE_TO_TASK = {
    "text": "text_to_video",
    "image": "image_to_video",
    "first_last": "first_last_to_video",
    "reference": "reference_to_video",
    "sequence": "sequence",
}


class VideoCatalogService(Protocol):
    def classify_library(
        self,
        library: ModelLibrary,
        *,
        domain: str,
    ) -> list[MediaModelDescriptor]: ...


@dataclass(frozen=True, slots=True)
class ResolvedVideoAsset:
    role: str
    stable_asset_id: str
    loader_name: str
    path: Path
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str


@dataclass(frozen=True, slots=True)
class ResolvedVideoLora(ResolvedVideoAsset):
    weight: float


@dataclass(frozen=True, slots=True)
class ResolvedVideoModelSelection:
    components: Mapping[str, ResolvedVideoAsset]
    loras: tuple[ResolvedVideoLora, ...]


class VideoModelResolver:
    """Resolve immutable request receipts to current registered Comfy assets.

    Frontend-provided compatibility claims are deliberately ignored here. The
    current server-side catalog and fingerprint are the execution authority.
    """

    def __init__(
        self,
        *,
        library_provider: Callable[[], ModelLibrary],
        model_roots: Mapping[str, Path],
        catalog_service: VideoCatalogService | None = None,
        fingerprint_service: ModelFingerprintService | None = None,
    ) -> None:
        self._library_provider = library_provider
        self._model_roots = {
            str(root_id): Path(path).absolute()
            for root_id, path in model_roots.items()
        }
        self._catalog = catalog_service or MediaModelCatalogService()
        self._fingerprints = fingerprint_service

    def validate(self, request: VideoRequest) -> None:
        self.resolve(request)

    def resolve(self, request: VideoRequest) -> ResolvedVideoModelSelection:
        if request.schema_version != "asteria.video.request.v2":
            raise VideoModelPreflightError("video_model_selection_required")
        selection = request.model_selection
        if selection is None:
            raise VideoModelPreflightError("video_model_selection_required")
        library = self._library_provider()
        items = self._items_by_stable_id(library)
        descriptors = self._descriptors_by_stable_id(library)
        task = _MODE_TO_TASK[request.mode]
        components: dict[str, ResolvedVideoAsset] = {}
        for receipt in selection.components:
            role = str(receipt.role)
            components[role] = self._resolve_asset(
                library=library,
                items=items,
                descriptors=descriptors,
                stable_asset_id=receipt.stable_asset_id,
                expected_role=role,
                expected_profile=request.model_profile,
                expected_architecture=selection.architecture,
                expected_variant=selection.variant,
                expected_task=task,
                expected_fingerprint=(
                    receipt.fingerprint_algorithm,
                    receipt.fingerprint_version,
                    receipt.fingerprint_value,
                ),
            )
        resolved_loras: list[ResolvedVideoLora] = []
        for receipt in request.loras:
            asset = self._resolve_asset(
                    library=library,
                    items=items,
                    descriptors=descriptors,
                    stable_asset_id=receipt.stable_asset_id,
                    expected_role="lora",
                    expected_profile=request.model_profile,
                    expected_architecture=selection.architecture,
                    expected_variant=selection.variant,
                    expected_task=task,
                    expected_fingerprint=(
                        receipt.fingerprint_algorithm,
                        receipt.fingerprint_version,
                        receipt.fingerprint_value,
                    ),
                )
            resolved_loras.append(
                ResolvedVideoLora(
                    role=asset.role,
                    stable_asset_id=asset.stable_asset_id,
                    loader_name=asset.loader_name,
                    path=asset.path,
                    fingerprint_algorithm=asset.fingerprint_algorithm,
                    fingerprint_version=asset.fingerprint_version,
                    fingerprint_value=asset.fingerprint_value,
                    weight=receipt.weight,
                )
            )
        return ResolvedVideoModelSelection(
            components=components,
            loras=tuple(resolved_loras),
        )

    def resolve_upscaler(
        self,
        receipt: VideoUpscaleModelReceipt,
        *,
        profile_id: str,
        variant: str,
    ) -> ResolvedVideoAsset:
        library = self._library_provider()
        return self._resolve_asset(
            library=library,
            items=self._items_by_stable_id(library),
            descriptors=self._descriptors_by_stable_id(library),
            stable_asset_id=receipt.stable_asset_id,
            expected_role="latent_upscaler",
            expected_profile=profile_id,
            expected_architecture="minimax_h3",
            expected_variant=variant,
            expected_task="upscale",
            expected_fingerprint=(
                receipt.fingerprint_algorithm,
                receipt.fingerprint_version,
                receipt.fingerprint_value,
            ),
        )

    def _resolve_asset(
        self,
        *,
        library: ModelLibrary,
        items: Mapping[str, ModelLibraryItem],
        descriptors: Mapping[str, MediaModelDescriptor],
        stable_asset_id: str,
        expected_role: str,
        expected_profile: str,
        expected_architecture: str,
        expected_variant: str,
        expected_task: str,
        expected_fingerprint: tuple[str, int, str],
    ) -> ResolvedVideoAsset:
        item = items.get(stable_asset_id)
        descriptor = descriptors.get(stable_asset_id)
        if item is None or descriptor is None:
            raise VideoModelPreflightError("video_model_asset_not_found")
        manual_diffusion_variant = (
            expected_role == "diffusion_model"
            and descriptor.compatibility == MediaModelCompatibility.PROVISIONAL_UNSUPPORTED
            and descriptor.reason_code == "minimax_h3_variant_unresolved"
            and descriptor.component_role == MediaModelComponentRole.DIFFUSION_MODEL
            and descriptor.architecture == "minimax_h3"
        )
        if (
            descriptor.compatibility != MediaModelCompatibility.SUPPORTED
            and not manual_diffusion_variant
        ):
            raise VideoModelPreflightError("video_model_asset_unsupported")
        if str(descriptor.component_role) != expected_role:
            raise VideoModelPreflightError("video_model_role_mismatch")
        if item.type != _ROLE_TO_MODEL_TYPE[expected_role]:
            raise VideoModelPreflightError("video_model_role_mismatch")
        if (
            descriptor.architecture != expected_architecture
            or expected_architecture not in descriptor.compatible_architecture_ids
        ):
            raise VideoModelPreflightError("video_model_architecture_incompatible")
        if expected_profile not in descriptor.compatible_profile_ids:
            raise VideoModelPreflightError("video_model_profile_incompatible")
        if not manual_diffusion_variant and expected_variant not in descriptor.compatible_variant_ids:
            raise VideoModelPreflightError("video_model_variant_incompatible")
        if not manual_diffusion_variant and expected_task not in descriptor.supported_tasks:
            raise VideoModelPreflightError("video_model_task_incompatible")
        current = self._current_fingerprint(item)
        if current != expected_fingerprint:
            raise VideoModelPreflightError("video_model_fingerprint_changed")
        if descriptor.fingerprint_algorithm is not None and (
            descriptor.fingerprint_algorithm,
            descriptor.fingerprint_version,
            descriptor.fingerprint_value,
        ) != current:
            raise VideoModelPreflightError("video_model_fingerprint_changed")
        loader_name = self._loader_name(item, expected_role)
        if self._loader_name_is_ambiguous(library, item, expected_role, loader_name):
            raise VideoModelPreflightError("video_model_loader_name_ambiguous")
        return ResolvedVideoAsset(
            role=expected_role,
            stable_asset_id=stable_asset_id,
            loader_name=loader_name,
            path=item.path,
            fingerprint_algorithm=current[0],
            fingerprint_version=current[1],
            fingerprint_value=current[2],
        )

    def _current_fingerprint(self, item: ModelLibraryItem) -> tuple[str, int, str]:
        try:
            if not item.path.is_file():
                raise OSError("registered model is unavailable")
            if self._fingerprints is not None:
                fingerprint = self._fingerprints.fast_fingerprint(
                    item.path,
                    root_id=item.root_id,
                    model_type=item.type,
                    relative_path=self._fingerprint_relative_path(item),
                )
                return fingerprint.algorithm, fingerprint.version, fingerprint.value
        except OSError as error:
            raise VideoModelPreflightError("video_model_asset_not_found") from error
        if (
            item.fingerprint_algorithm is None
            or item.fingerprint_version is None
            or item.fingerprint_value is None
        ):
            raise VideoModelPreflightError("video_model_fingerprint_unavailable")
        return (
            item.fingerprint_algorithm,
            item.fingerprint_version,
            item.fingerprint_value,
        )

    @staticmethod
    def _fingerprint_relative_path(item: ModelLibraryItem) -> str:
        external = item.metadata.get("external_relative_path")
        return external if external else item.relative_path.as_posix()

    def _loader_name(self, item: ModelLibraryItem, role: str) -> str:
        external = item.metadata.get("external_relative_path")
        if external:
            value = external.replace("/", "\\")
        else:
            root = self._model_roots.get(item.root_id)
            source = str(item.path) if root is not None else item.relative_path.as_posix()
            value = comfy_model_name(
                source,
                _ROLE_TO_COMFY_FOLDER[role],
                model_root=root,
            )
        if not value or os.path.isabs(value) or (len(value) > 1 and value[1] == ":"):
            raise VideoModelPreflightError("video_model_loader_name_invalid")
        return value

    def _loader_name_is_ambiguous(
        self,
        library: ModelLibrary,
        selected: ModelLibraryItem,
        role: str,
        loader_name: str,
    ) -> bool:
        matches: set[str] = set()
        for candidate in library.items:
            if candidate.type != selected.type:
                continue
            try:
                candidate_name = self._loader_name(candidate, role)
            except VideoModelPreflightError:
                continue
            if candidate_name.casefold() == loader_name.casefold():
                matches.add(candidate.stable_model_id or str(candidate.path))
        return len(matches) != 1

    def _items_by_stable_id(self, library: ModelLibrary) -> dict[str, ModelLibraryItem]:
        values: dict[str, ModelLibraryItem] = {}
        duplicates: set[str] = set()
        for item in library.items:
            if not item.stable_model_id:
                continue
            if item.stable_model_id in values:
                duplicates.add(item.stable_model_id)
            values[item.stable_model_id] = item
        for duplicate in duplicates:
            values.pop(duplicate, None)
        return values

    def _descriptors_by_stable_id(
        self,
        library: ModelLibrary,
    ) -> dict[str, MediaModelDescriptor]:
        values: dict[str, MediaModelDescriptor] = {}
        duplicates: set[str] = set()
        for descriptor in self._catalog.classify_library(library, domain="video"):
            if descriptor.stable_asset_id in values:
                duplicates.add(descriptor.stable_asset_id)
            values[descriptor.stable_asset_id] = descriptor
        for duplicate in duplicates:
            values.pop(duplicate, None)
        return values


def registered_video_model_library(
    config: AppConfig,
    *,
    refresh: bool = False,
) -> ModelLibrary:
    common_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    common = (
        refresh_model_library(common_root, config=config)
        if refresh
        else get_model_library(common_root, config=config)
    )
    runtime_root = managed_video_runtime_root(config)
    runtime_models = runtime_root / "models"
    runtime = ModelLibraryScanner(runtime_models).scan()
    fingerprints = ModelFingerprintService(
        config.data_dir / "config" / "model-fingerprints.json"
    )
    runtime_items: list[ModelLibraryItem] = []
    for item in runtime.items:
        relative_path = item.relative_path.as_posix()
        try:
            fingerprint = fingerprints.fast_fingerprint(
                item.path,
                root_id=VIDEO_RUNTIME_MODEL_ROOT_ID,
                model_type=item.type,
                relative_path=relative_path,
            )
        except OSError:
            continue
        runtime_items.append(
            replace(
                item,
                root_id=VIDEO_RUNTIME_MODEL_ROOT_ID,
                source="managed",
                stable_model_id=ModelFingerprintService.stable_model_id(
                    root_id=VIDEO_RUNTIME_MODEL_ROOT_ID,
                    model_type=item.type,
                    relative_path=relative_path,
                ),
                fingerprint_algorithm=fingerprint.algorithm,
                fingerprint_version=fingerprint.version,
                fingerprint_value=fingerprint.value,
            )
        )
    items = [*common.items, *runtime_items]
    items.sort(
        key=lambda item: (
            item.type,
            item.name.casefold(),
            item.root_id,
            item.relative_path.as_posix().casefold(),
        )
    )
    return ModelLibrary(items=items, link_health=common.link_health)


def managed_video_runtime_root(config: AppConfig) -> Path:
    profile = VideoModelProfileRepository(
        config.root_dir / "profiles" / "video" / "minimax-h3-local.v1.json",
        workflow_root=config.root_dir / "workflows" / "video" / "minimax-h3",
    ).load()
    manifest = ReleaseManifestRepository(config.release_manifest_path).load()
    runtime = next(
        (
            artifact
            for artifact in manifest.artifacts
            if artifact.id == profile.runtime.component_id
        ),
        None,
    )
    if runtime is None or runtime.version != profile.runtime.managed_version:
        raise VideoModelPreflightError("video_setup_runtime_manifest_mismatch")
    return (config.root_dir / runtime.relative_path).absolute()


def production_video_model_resolver(config: AppConfig) -> VideoModelResolver:
    common_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    runtime_root = managed_video_runtime_root(config) / "models"
    return VideoModelResolver(
        library_provider=lambda: registered_video_model_library(config),
        model_roots={
            "asteria-local": common_root,
            VIDEO_RUNTIME_MODEL_ROOT_ID: runtime_root,
        },
        fingerprint_service=ModelFingerprintService(
            config.data_dir / "config" / "model-fingerprints.json"
        ),
    )
