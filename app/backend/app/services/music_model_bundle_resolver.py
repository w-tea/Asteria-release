from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.schemas.media_model_catalog import (
    MediaModelCompatibility,
    MediaModelComponentRole,
    MediaModelDescriptor,
    MediaModelDomain,
)
from app.services.model_library import ModelLibrary, ModelLibraryItem


_BUNDLE_ROLES = (
    MediaModelComponentRole.DIFFUSION_MODEL,
    MediaModelComponentRole.TEXT_ENCODER,
    MediaModelComponentRole.AUDIO_VAE,
)
_ROLE_TO_MODEL_TYPE = {
    MediaModelComponentRole.DIFFUSION_MODEL: "checkpoint",
    MediaModelComponentRole.TEXT_ENCODER: "text_encoder",
    MediaModelComponentRole.AUDIO_VAE: "vae",
}
_EXECUTION_PROFILES = frozenset({"automatic"})


class MusicModelBundlePreflightError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MusicModelComponentReceipt:
    role: MediaModelComponentRole
    stable_asset_id: str
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str


@dataclass(frozen=True, slots=True)
class MusicModelBundleSelection:
    architecture: str
    profile_id: str
    task: str
    execution_profile: str
    components: tuple[MusicModelComponentReceipt, ...]
    managed_bundle_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedMusicModelBundle:
    """An explicit official component mapping supplied by the caller.

    The resolver deliberately has no built-in file-name lookup. A caller may
    provide one of these declarations for managed official assets, but custom
    selections must always carry their own complete component receipts.
    """

    bundle_id: str
    architecture: str
    profile_id: str
    supported_tasks: tuple[str, ...]
    components: tuple[MusicModelComponentReceipt, ...]


@dataclass(frozen=True, slots=True)
class ResolvedMusicModelAsset:
    role: MediaModelComponentRole
    stable_asset_id: str
    path: Path
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str
    architecture: str
    precision: str | None
    quantization: str | None


@dataclass(frozen=True, slots=True)
class ResolvedMusicModelBundle:
    architecture: str
    profile_id: str
    task: str
    execution_profile: str
    managed_bundle_id: str | None
    components: tuple[ResolvedMusicModelAsset, ...]


class MusicModelBundleResolver:
    """Resolve immutable Music3 bundle receipts against the current catalog.

    This service intentionally has no runtime, API, setup, or filename policy.
    It only binds a request-owned component triple to registered, fingerprinted
    model items and their server-side descriptors.
    """

    def resolve(
        self,
        *,
        library: ModelLibrary,
        descriptors: Iterable[MediaModelDescriptor],
        selection: MusicModelBundleSelection,
        managed_bundles: Iterable[ManagedMusicModelBundle] = (),
    ) -> ResolvedMusicModelBundle:
        if selection.execution_profile not in _EXECUTION_PROFILES:
            raise MusicModelBundlePreflightError("music_execution_profile_invalid")

        receipts, managed_bundle_id = self._receipts_for_selection(
            selection=selection,
            managed_bundles=managed_bundles,
        )
        self._validate_receipts(receipts)
        items, duplicate_item_ids = self._items_by_stable_id(library)
        catalog, duplicate_descriptor_ids = self._descriptors_by_stable_id(descriptors)

        components = tuple(
            self._resolve_component(
                receipt=receipt,
                selection=selection,
                items=items,
                descriptors=catalog,
                duplicate_item_ids=duplicate_item_ids,
                duplicate_descriptor_ids=duplicate_descriptor_ids,
            )
            for receipt in sorted(receipts, key=lambda receipt: _BUNDLE_ROLES.index(receipt.role))
        )
        return ResolvedMusicModelBundle(
            architecture=selection.architecture,
            profile_id=selection.profile_id,
            task=selection.task,
            execution_profile=selection.execution_profile,
            managed_bundle_id=managed_bundle_id,
            components=components,
        )

    @staticmethod
    def _receipts_for_selection(
        *,
        selection: MusicModelBundleSelection,
        managed_bundles: Iterable[ManagedMusicModelBundle],
    ) -> tuple[tuple[MusicModelComponentReceipt, ...], str | None]:
        if selection.managed_bundle_id is None:
            return selection.components, None
        if selection.components:
            raise MusicModelBundlePreflightError(
                "music_model_managed_bundle_conflicting_receipts"
            )
        matches = [
            bundle
            for bundle in managed_bundles
            if bundle.bundle_id == selection.managed_bundle_id
        ]
        if len(matches) != 1:
            raise MusicModelBundlePreflightError(
                "music_model_managed_bundle_not_declared"
            )
        bundle = matches[0]
        if bundle.architecture != selection.architecture:
            raise MusicModelBundlePreflightError("music_model_architecture_incompatible")
        if bundle.profile_id != selection.profile_id:
            raise MusicModelBundlePreflightError("music_model_profile_incompatible")
        if selection.task not in bundle.supported_tasks:
            raise MusicModelBundlePreflightError("music_model_task_incompatible")
        return bundle.components, bundle.bundle_id

    @staticmethod
    def _validate_receipts(
        receipts: tuple[MusicModelComponentReceipt, ...],
    ) -> None:
        roles: set[MediaModelComponentRole] = set()
        asset_ids: set[str] = set()
        for receipt in receipts:
            if receipt.role not in _BUNDLE_ROLES:
                raise MusicModelBundlePreflightError(
                    "music_model_component_role_unsupported"
                )
            if receipt.role in roles:
                raise MusicModelBundlePreflightError("music_model_bundle_duplicate_role")
            if not receipt.stable_asset_id:
                raise MusicModelBundlePreflightError("music_model_stable_id_missing")
            if receipt.stable_asset_id in asset_ids:
                raise MusicModelBundlePreflightError("music_model_bundle_duplicate_asset")
            if (
                not receipt.fingerprint_algorithm
                or receipt.fingerprint_version < 1
                or not receipt.fingerprint_value
            ):
                raise MusicModelBundlePreflightError(
                    "music_model_fingerprint_unavailable"
                )
            roles.add(receipt.role)
            asset_ids.add(receipt.stable_asset_id)
        if roles != set(_BUNDLE_ROLES):
            raise MusicModelBundlePreflightError("music_model_bundle_incomplete")

    def _resolve_component(
        self,
        *,
        receipt: MusicModelComponentReceipt,
        selection: MusicModelBundleSelection,
        items: dict[str, ModelLibraryItem],
        descriptors: dict[str, MediaModelDescriptor],
        duplicate_item_ids: set[str],
        duplicate_descriptor_ids: set[str],
    ) -> ResolvedMusicModelAsset:
        stable_asset_id = receipt.stable_asset_id
        if (
            stable_asset_id in duplicate_item_ids
            or stable_asset_id in duplicate_descriptor_ids
        ):
            raise MusicModelBundlePreflightError("music_model_duplicate_stable_id")
        item = items.get(stable_asset_id)
        descriptor = descriptors.get(stable_asset_id)
        if item is None or descriptor is None:
            if descriptor is not None and item is None:
                raise MusicModelBundlePreflightError("music_model_stable_id_missing")
            raise MusicModelBundlePreflightError("music_model_asset_not_found")
        if not item.path.is_file():
            raise MusicModelBundlePreflightError("music_model_asset_not_found")
        if (
            descriptor.domain != MediaModelDomain.MUSIC
            or descriptor.compatibility != MediaModelCompatibility.SUPPORTED
        ):
            raise MusicModelBundlePreflightError("music_model_asset_unsupported")
        if descriptor.component_role != receipt.role:
            raise MusicModelBundlePreflightError("music_model_role_mismatch")
        if item.type != _ROLE_TO_MODEL_TYPE[receipt.role]:
            raise MusicModelBundlePreflightError("music_model_role_mismatch")
        if (
            descriptor.architecture != selection.architecture
            or selection.architecture not in descriptor.compatible_architecture_ids
        ):
            raise MusicModelBundlePreflightError("music_model_architecture_incompatible")
        if selection.profile_id not in descriptor.compatible_profile_ids:
            raise MusicModelBundlePreflightError("music_model_profile_incompatible")
        if selection.task not in descriptor.supported_tasks:
            raise MusicModelBundlePreflightError("music_model_task_incompatible")
        fingerprint = self._current_fingerprint(item)
        expected = (
            receipt.fingerprint_algorithm,
            receipt.fingerprint_version,
            receipt.fingerprint_value,
        )
        if fingerprint != expected:
            raise MusicModelBundlePreflightError("music_model_fingerprint_changed")
        descriptor_fingerprint = (
            descriptor.fingerprint_algorithm,
            descriptor.fingerprint_version,
            descriptor.fingerprint_value,
        )
        if any(value is None for value in descriptor_fingerprint):
            raise MusicModelBundlePreflightError("music_model_fingerprint_unavailable")
        if descriptor_fingerprint != fingerprint:
            raise MusicModelBundlePreflightError("music_model_fingerprint_changed")
        return ResolvedMusicModelAsset(
            role=receipt.role,
            stable_asset_id=stable_asset_id,
            path=item.path,
            fingerprint_algorithm=fingerprint[0],
            fingerprint_version=fingerprint[1],
            fingerprint_value=fingerprint[2],
            architecture=descriptor.architecture,
            precision=descriptor.precision,
            quantization=descriptor.quantization,
        )

    @staticmethod
    def _current_fingerprint(item: ModelLibraryItem) -> tuple[str, int, str]:
        if (
            item.fingerprint_algorithm is None
            or item.fingerprint_version is None
            or item.fingerprint_value is None
        ):
            raise MusicModelBundlePreflightError("music_model_fingerprint_unavailable")
        return (
            item.fingerprint_algorithm,
            item.fingerprint_version,
            item.fingerprint_value,
        )

    @staticmethod
    def _items_by_stable_id(
        library: ModelLibrary,
    ) -> tuple[dict[str, ModelLibraryItem], set[str]]:
        values: dict[str, ModelLibraryItem] = {}
        duplicates: set[str] = set()
        for item in library.items:
            if not item.stable_model_id:
                continue
            if item.stable_model_id in values:
                duplicates.add(item.stable_model_id)
            values[item.stable_model_id] = item
        for stable_asset_id in duplicates:
            values.pop(stable_asset_id, None)
        return values, duplicates

    @staticmethod
    def _descriptors_by_stable_id(
        descriptors: Iterable[MediaModelDescriptor],
    ) -> tuple[dict[str, MediaModelDescriptor], set[str]]:
        values: dict[str, MediaModelDescriptor] = {}
        duplicates: set[str] = set()
        for descriptor in descriptors:
            if descriptor.stable_asset_id in values:
                duplicates.add(descriptor.stable_asset_id)
            values[descriptor.stable_asset_id] = descriptor
        for stable_asset_id in duplicates:
            values.pop(stable_asset_id, None)
        return values, duplicates
