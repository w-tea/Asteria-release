"""Apply changes to the existing Library record and its companion metadata."""
from __future__ import annotations

from collections.abc import Iterable

from app.schemas.generation import LibraryCurationFields
from app.services.history import HistoryRepository
from app.services.metadata import MetadataWriter


def sync_library_curation_sidecars(items: Iterable[dict]) -> None:
    writer = MetadataWriter()
    for item in items:
        writer.update_curation_sidecars(item)


def update_library_curation(
    history: HistoryRepository, generation_id: int, patch: LibraryCurationFields, *, update_folder: bool,
) -> dict | None:
    """Share the native update and its sidecar behavior with common actions."""
    item = history.update_curation(generation_id, rating=patch.rating, favorite=patch.favorite,
                                   tags=patch.tags, notes=patch.notes, folder_id=patch.folder_id,
                                   update_folder=update_folder)
    if item is not None:
        sync_library_curation_sidecars([item])
    return item
