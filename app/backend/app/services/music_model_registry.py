from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Protocol

from app.core.config import AppConfig
from app.schemas.media_model_catalog import MediaModelComponentRole, MediaModelDescriptor
from app.schemas.music import MusicRequest
from app.services.comfy_model_paths import comfy_model_name
from app.services.media_model_catalog import MediaModelCatalogService
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library import ModelLibrary, ModelLibraryItem, ModelLibraryScanner
from app.services.model_library_cache import get_model_library, refresh_model_library
from app.services.model_storage import resolve_model_root
from app.services.music_local_adapter import MusicWorkflowModelAsset, MusicWorkflowModelBundle
from app.services.music_local_setup import MusicModelProfile, MusicModelProfileRepository
from app.services.music_model_bundle_resolver import (
    MusicModelBundlePreflightError,
    MusicModelBundleResolver,
    MusicModelBundleSelection,
    MusicModelComponentReceipt,
)
from app.services.release_manifest import ReleaseManifestRepository


MUSIC_RUNTIME_MODEL_ROOT_ID = "minimax-music3-runtime"
_ROLE_TO_MODEL_TYPE = {
    MediaModelComponentRole.DIFFUSION_MODEL: "checkpoint",
    MediaModelComponentRole.TEXT_ENCODER: "text_encoder",
    MediaModelComponentRole.AUDIO_VAE: "vae",
}
_ROLE_TO_COMFY_FOLDER = {
    MediaModelComponentRole.DIFFUSION_MODEL: "diffusion_models",
    MediaModelComponentRole.TEXT_ENCODER: "text_encoders",
    MediaModelComponentRole.AUDIO_VAE: "vae",
}


class MusicCatalogService(Protocol):
    def classify_library(
        self,
        library: ModelLibrary,
        *,
        domain: str,
    ) -> list[MediaModelDescriptor]: ...


@dataclass(frozen=True, slots=True)
class MusicModelBundleCandidate:
    bundle_id: str
    label: str
    profile_id: str
    architecture: str
    managed: bool
    components: tuple[MusicModelComponentReceipt, ...]
    compatible: bool
    reason_code: str | None = None


class MusicModelRegistry:
    """Queue-time authority for Music3 catalog receipts and Comfy loader names."""

    def __init__(
        self,
        *,
        library_provider: Callable[[], ModelLibrary],
        model_roots: Mapping[str, Path],
        profile: MusicModelProfile,
        runtime_root: Path,
        catalog_service: MusicCatalogService | None = None,
        fingerprint_service: ModelFingerprintService | None = None,
    ) -> None:
        self._library_provider = library_provider
        self._model_roots = {
            str(root_id): Path(root).absolute()
            for root_id, root in model_roots.items()
        }
        self.profile = profile
        self.runtime_root = Path(runtime_root).absolute()
        self._catalog = catalog_service or MediaModelCatalogService()
        self._fingerprints = fingerprint_service
        self._resolver = MusicModelBundleResolver()

    def resolve(self, request: MusicRequest) -> MusicWorkflowModelBundle:
        if request.schema_version != "asteria.music.request.v2":
            raise MusicModelBundlePreflightError("music_model_selection_required")
        if request.model_selection is None or request.sampling is None:
            raise MusicModelBundlePreflightError("music_model_selection_required")

        selection = request.model_selection
        receipts = tuple(
            MusicModelComponentReceipt(
                role=MediaModelComponentRole(component.role),
                stable_asset_id=component.stable_asset_id,
                fingerprint_algorithm=component.fingerprint_algorithm,
                fingerprint_version=component.fingerprint_version,
                fingerprint_value=component.fingerprint_value,
            )
            for component in selection.components
        )
        library = self._with_current_fingerprints(
            self._library_provider(),
            {receipt.stable_asset_id for receipt in receipts},
        )
        descriptors = self._catalog.classify_library(library, domain="music")
        resolved = self._resolver.resolve(
            library=library,
            descriptors=descriptors,
            selection=MusicModelBundleSelection(
                architecture=selection.architecture,
                profile_id=selection.profile_id,
                task=request.mode,
                execution_profile=request.sampling.execution_profile,
                components=receipts,
            ),
        )
        self._validate_managed_identity(
            selection.bundle_id,
            receipts,
            library=library,
            descriptors=descriptors,
        )

        items = self._unique_items_by_stable_id(library)
        workflow_components: list[MusicWorkflowModelAsset] = []
        for component in resolved.components:
            item = items.get(component.stable_asset_id)
            if item is None:
                raise MusicModelBundlePreflightError("music_model_asset_not_found")
            loader_name = self._loader_name(item, component.role)
            if self._loader_name_is_ambiguous(library, item, component.role, loader_name):
                raise MusicModelBundlePreflightError("music_model_loader_name_ambiguous")
            workflow_components.append(
                MusicWorkflowModelAsset(
                    role=component.role,
                    stable_asset_id=component.stable_asset_id,
                    loader_name=loader_name,
                    fingerprint_algorithm=component.fingerprint_algorithm,
                    fingerprint_version=component.fingerprint_version,
                    fingerprint_value=component.fingerprint_value,
                )
            )
        return MusicWorkflowModelBundle(
            bundle_id=selection.bundle_id,
            architecture=selection.architecture,
            execution_profile=request.sampling.execution_profile,
            model_revision=self._model_revision(selection.bundle_id, workflow_components),
            components=tuple(workflow_components),
        )

    def managed_bundle_candidates(self) -> tuple[MusicModelBundleCandidate, ...]:
        initial_library = self._library_provider()
        expected_ids = {
            self._managed_stable_id(asset.id)
            for asset in self.profile.assets
        }
        library = self._with_current_fingerprints(initial_library, expected_ids)
        descriptors = self._catalog.classify_library(library, domain="music")
        items = self._unique_items_by_stable_id(library)
        assets = {asset.id: asset for asset in self.profile.assets}
        candidates: list[MusicModelBundleCandidate] = []
        for bundle in self.profile.model_bundles:
            receipts: list[MusicModelComponentReceipt] = []
            for asset_id in bundle.asset_ids:
                profile_asset = assets[asset_id]
                stable_id = self._managed_stable_id(asset_id)
                item = items.get(stable_id)
                if (
                    item is None
                    or item.fingerprint_algorithm is None
                    or item.fingerprint_version is None
                    or item.fingerprint_value is None
                ):
                    continue
                receipts.append(
                    MusicModelComponentReceipt(
                        role=MediaModelComponentRole(profile_asset.component_role),
                        stable_asset_id=stable_id,
                        fingerprint_algorithm=item.fingerprint_algorithm,
                        fingerprint_version=item.fingerprint_version,
                        fingerprint_value=item.fingerprint_value,
                    )
                )
            compatible = False
            reason_code: str | None = None
            if len(receipts) == 3:
                try:
                    self._resolver.resolve(
                        library=library,
                        descriptors=descriptors,
                        selection=MusicModelBundleSelection(
                            architecture="minimax_music3",
                            profile_id="minimax-music3",
                            task="vocal",
                            execution_profile="automatic",
                            components=tuple(receipts),
                        ),
                    )
                    compatible = True
                except MusicModelBundlePreflightError as error:
                    reason_code = error.code
            else:
                reason_code = "music_model_bundle_incomplete"
            candidates.append(
                MusicModelBundleCandidate(
                    bundle_id=bundle.id,
                    label=bundle.label,
                    profile_id="minimax-music3",
                    architecture="minimax_music3",
                    managed=True,
                    components=tuple(receipts),
                    compatible=compatible,
                    reason_code=reason_code,
                )
            )
        return tuple(candidates)

    def _validate_managed_identity(
        self,
        bundle_id: str,
        receipts: tuple[MusicModelComponentReceipt, ...],
        *,
        library: ModelLibrary,
        descriptors: Iterable[MediaModelDescriptor],
    ) -> None:
        managed = {bundle.id: bundle for bundle in self.profile.model_bundles}
        if bundle_id not in managed:
            return
        assets = {asset.id: asset for asset in self.profile.assets}
        expected: list[MusicModelComponentReceipt] = []
        items = self._unique_items_by_stable_id(library)
        for asset_id in managed[bundle_id].asset_ids:
            stable_id = self._managed_stable_id(asset_id)
            item = items.get(stable_id)
            if (
                item is None
                or item.fingerprint_algorithm is None
                or item.fingerprint_version is None
                or item.fingerprint_value is None
            ):
                raise MusicModelBundlePreflightError("music_model_bundle_incomplete")
            expected.append(
                MusicModelComponentReceipt(
                    role=MediaModelComponentRole(assets[asset_id].component_role),
                    stable_asset_id=stable_id,
                    fingerprint_algorithm=item.fingerprint_algorithm,
                    fingerprint_version=item.fingerprint_version,
                    fingerprint_value=item.fingerprint_value,
                )
            )
        expected_by_role = {item.role: item for item in expected}
        actual_by_role = {item.role: item for item in receipts}
        if expected_by_role != actual_by_role:
            raise MusicModelBundlePreflightError("music_model_managed_bundle_mismatch")
        self._resolver.resolve(
            library=library,
            descriptors=descriptors,
            selection=MusicModelBundleSelection(
                architecture="minimax_music3",
                profile_id="minimax-music3",
                task="vocal",
                execution_profile="automatic",
                components=tuple(expected),
            ),
        )

    def _managed_stable_id(self, asset_id: str) -> str:
        assets = {asset.id: asset for asset in self.profile.assets}
        try:
            asset = assets[asset_id]
            relative = PurePosixPath(asset.relative_path.replace("\\", "/"))
            if not relative.parts or relative.parts[0].casefold() != "models":
                raise ValueError("managed music model path is invalid")
            model_relative = PurePosixPath(*relative.parts[1:]).as_posix()
            model_type = _ROLE_TO_MODEL_TYPE[MediaModelComponentRole(asset.component_role)]
        except (KeyError, ValueError) as error:
            raise MusicModelBundlePreflightError("music_model_profile_invalid") from error
        return ModelFingerprintService.stable_model_id(
            root_id=MUSIC_RUNTIME_MODEL_ROOT_ID,
            model_type=model_type,
            relative_path=model_relative,
        )

    def _with_current_fingerprints(
        self,
        library: ModelLibrary,
        selected_ids: set[str],
    ) -> ModelLibrary:
        if self._fingerprints is None:
            return library
        items: list[ModelLibraryItem] = []
        for item in library.items:
            if item.stable_model_id not in selected_ids:
                items.append(item)
                continue
            try:
                fingerprint = self._fingerprints.fast_fingerprint(
                    item.path,
                    root_id=item.root_id,
                    model_type=item.type,
                    relative_path=self._fingerprint_relative_path(item),
                )
            except OSError as error:
                raise MusicModelBundlePreflightError("music_model_asset_not_found") from error
            items.append(
                replace(
                    item,
                    size_bytes=fingerprint.file_size,
                    fingerprint_algorithm=fingerprint.algorithm,
                    fingerprint_version=fingerprint.version,
                    fingerprint_value=fingerprint.value,
                )
            )
        return ModelLibrary(items=items, link_health=library.link_health)

    @staticmethod
    def _fingerprint_relative_path(item: ModelLibraryItem) -> str:
        return item.metadata.get("external_relative_path") or item.relative_path.as_posix()

    def _loader_name(
        self,
        item: ModelLibraryItem,
        role: MediaModelComponentRole,
    ) -> str:
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
            raise MusicModelBundlePreflightError("music_model_loader_name_invalid")
        return value

    def _loader_name_is_ambiguous(
        self,
        library: ModelLibrary,
        selected: ModelLibraryItem,
        role: MediaModelComponentRole,
        loader_name: str,
    ) -> bool:
        matches: set[str] = set()
        for candidate in library.items:
            if candidate.type != selected.type:
                continue
            try:
                candidate_name = self._loader_name(candidate, role)
            except MusicModelBundlePreflightError:
                continue
            if candidate_name.casefold() == loader_name.casefold():
                matches.add(candidate.stable_model_id or str(candidate.path))
        return len(matches) != 1

    @staticmethod
    def _unique_items_by_stable_id(library: ModelLibrary) -> dict[str, ModelLibraryItem]:
        values: dict[str, ModelLibraryItem] = {}
        duplicates: set[str] = set()
        for item in library.items:
            if not item.stable_model_id:
                continue
            if item.stable_model_id in values:
                duplicates.add(item.stable_model_id)
            values[item.stable_model_id] = item
        for stable_id in duplicates:
            values.pop(stable_id, None)
        return values

    def _model_revision(
        self,
        bundle_id: str,
        components: list[MusicWorkflowModelAsset],
    ) -> str:
        if any(bundle.id == bundle_id for bundle in self.profile.model_bundles):
            return self.profile.model_repository.revision
        digest = hashlib.sha256()
        for component in components:
            digest.update(component.role.value.encode("ascii"))
            digest.update(b"\0")
            digest.update(component.stable_asset_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(component.fingerprint_algorithm.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(component.fingerprint_version).encode("ascii"))
            digest.update(b"\0")
            digest.update(component.fingerprint_value.encode("ascii"))
            digest.update(b"\0")
        return f"music-bundle-v1:{digest.hexdigest()}"


def managed_music_runtime_root(config: AppConfig) -> Path:
    profile = MusicModelProfileRepository(
        config.root_dir / "profiles" / "music" / "minimax-music3-local.v1.json"
    ).load()
    manifest = ReleaseManifestRepository(config.release_manifest_path).load()
    runtime = next(
        (artifact for artifact in manifest.artifacts if artifact.id == profile.runtime.component_id),
        None,
    )
    if runtime is None or runtime.version != profile.runtime.managed_version:
        raise MusicModelBundlePreflightError("music_setup_runtime_manifest_mismatch")
    return (config.root_dir / runtime.relative_path).absolute()


def registered_music_model_library(
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
    runtime_models = managed_music_runtime_root(config) / "models"
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
                root_id=MUSIC_RUNTIME_MODEL_ROOT_ID,
                model_type=item.type,
                relative_path=relative_path,
            )
        except OSError:
            continue
        runtime_items.append(
            replace(
                item,
                root_id=MUSIC_RUNTIME_MODEL_ROOT_ID,
                source="managed",
                stable_model_id=ModelFingerprintService.stable_model_id(
                    root_id=MUSIC_RUNTIME_MODEL_ROOT_ID,
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


def production_music_model_registry(config: AppConfig) -> MusicModelRegistry:
    common_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    runtime_root = managed_music_runtime_root(config)
    profile = MusicModelProfileRepository(
        config.root_dir / "profiles" / "music" / "minimax-music3-local.v1.json"
    ).load()
    return MusicModelRegistry(
        library_provider=lambda: registered_music_model_library(config),
        model_roots={
            "asteria-local": common_root,
            MUSIC_RUNTIME_MODEL_ROOT_ID: runtime_root / "models",
        },
        profile=profile,
        runtime_root=runtime_root,
        fingerprint_service=ModelFingerprintService(
            config.data_dir / "config" / "model-fingerprints.json"
        ),
    )
