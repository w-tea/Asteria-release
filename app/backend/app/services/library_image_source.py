"""Resolve the exact Library image before handing it to an external adapter."""
from pathlib import Path

from app.schemas.action_contracts import MediaSourceTarget
from app.services.history import HistoryRepository
from app.services.media_artifacts import MediaArtifactAccessError, file_sha256, resolve_registered_media_artifact


def resolve_library_integration_image(data_dir: Path, target: MediaSourceTarget) -> Path:
    from app.api.generation import _known_output_roots

    items = HistoryRepository.get_many_existing(data_dir / 'asteria.db', [target.generation_id])
    if not items:
        raise MediaArtifactAccessError('generation_not_found')
    item = items[0]
    artifact, path = resolve_registered_media_artifact(
        item, target.artifact_id, accepted_roots=_known_output_roots(data_dir, [item]),
        allowed_kinds={'image'},
    )
    if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise MediaArtifactAccessError('invalid_output_image')
    if artifact.sha256 is not None:
        try:
            actual_hash = file_sha256(path)
        except OSError as exc:
            raise MediaArtifactAccessError('output_image_unreadable') from exc
        if actual_hash != artifact.sha256:
            raise MediaArtifactAccessError('media_artifact_changed')
    return path
