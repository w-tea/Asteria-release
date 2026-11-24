"""Register user-selected input bytes as ordinary Library media, not as a job."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import sqlite3
from tempfile import mkdtemp

from fastapi import HTTPException
from PIL import Image

from app.schemas.action_contracts import MediaSourceRecord, MediaSourceTarget
from app.schemas.media import MediaArtifact, MediaKind, image_media_type
from app.services.action_receipts import create_action_receipt_table, read_action_receipt, save_action_receipt
from app.services.history import HistoryRepository
from app.services.image_validation import SUPPORTED_IMAGE_FORMATS, validate_image_file
from app.services.media_artifacts import finalize_media_artifact, resolve_registered_media_artifact, MediaArtifactAccessError
from app.services.operation_repository import canonical_operation_json
from app.services.user_settings import resolve_current_output_root


_ACTION = 'media.sources.import'
MEDIA_IMPORT_SCHEMA = 'asteria.media-source.v1'


@dataclass(slots=True)
class MediaSourceUpload:
    root: Path
    directory: Path
    filename: str
    kind: MediaKind
    suffix: str
    preserve: bool = False

    @classmethod
    def create(cls, data_dir: Path, raw_name: str) -> MediaSourceUpload:
        filename = PurePosixPath(raw_name.replace('\\', '/')).name
        if not filename or any(ord(character) < 32 or ord(character) == 127 for character in filename):
            raise HTTPException(422, {'code': 'assistant_media_invalid'})
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix in SUPPORTED_IMAGE_FORMATS:
            kind: MediaKind = 'image'
        elif suffix in {'.mp4', '.webm'}:
            kind = 'video'
        elif suffix in {'.wav', '.flac', '.mp3', '.opus'}:
            kind = 'audio'
        else:
            raise HTTPException(422, {'code': 'assistant_media_invalid'})
        root = resolve_current_output_root(data_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root, Path(mkdtemp(prefix='import-', dir=root)), filename, kind, suffix)

    @property
    def path(self) -> Path:
        return self.directory / ('source' + self.suffix)

    def cleanup(self) -> None:
        if not self.preserve:
            # The instance owns one unique newly created directory; never
            # recursively delete files or follow directory links here.
            for path in self.directory.iterdir():
                if path == self.path or path.name.startswith('source.') and path.name.endswith('.poster.webp'):
                    path.unlink(missing_ok=True)
            self.directory.rmdir()

    def register(self, data_dir: Path, request_id: str, digest: str) -> tuple[MediaSourceRecord, bool]:
        history = HistoryRepository(data_dir / 'asteria.db')
        arguments = canonical_operation_json({'filename': self.filename, 'sha256': digest})
        generation_id = None
        with self.path.open('rb+') as file:
            os.fsync(file.fileno())
        # Validate and build the poster before holding a database write lock.
        if self.kind == 'image':
            validate_image_file(self.path)
            with Image.open(self.path) as image:
                width, height = image.size
            finalized = None
        else:
            finalized = finalize_media_artifact(generation_id=None, source_path=self.path, poster_root=self.directory,
                                               local_upload=True, source_sha256=digest)
        try:
            with history.publication() as connection:
                create_action_receipt_table(connection)
                receipt = read_action_receipt(connection, request_id, _ACTION, arguments)
                if receipt is not None:
                    return MediaSourceRecord.model_validate(receipt), True
                settings = {'schema_version': MEDIA_IMPORT_SCHEMA, 'output_root': str(self.root),
                            'original_filename': self.filename}
                generation_id = history.add(status='imported', profile='imported', prompt=self.filename,
                                            expanded_prompt='', negative_prompt='', settings=settings, model_slots={},
                                            loras=[], embeddings=[], output_paths=[], media_kind=self.kind)
                if finalized is None:
                    primary = MediaArtifact(artifact_id=f'media:image:{generation_id}:import', path=str(self.path),
                                            kind='image', role='primary', mime_type=image_media_type(self.path.name),
                                            width=width, height=height, sha256=digest, size_bytes=self.path.stat().st_size)
                    artifacts, paths = [primary], [str(self.path)]
                else:
                    artifacts, paths = list(finalized.artifacts), list(finalized.output_paths)
                    primary = artifacts[0]
                history.update_media_artifacts(generation_id, media_kind=self.kind, artifacts=artifacts, output_paths=paths)
                result = _record(generation_id, primary, self.filename)
                save_action_receipt(connection, request_id, _ACTION, arguments, result.model_dump_json())
            self.preserve = True
        except BaseException:
            if generation_id is not None:
                # An interruption can arrive just after SQLite commits.
                # If the referenced file cannot be ruled out, keep it.
                try:
                    self.preserve = str(self.path) in (history.get(generation_id) or {}).get('output_paths', [])
                except (sqlite3.Error, OSError, ValueError):
                    self.preserve = True
            raise
        return result, False


def _record(generation_id: int, artifact: MediaArtifact, filename: str) -> MediaSourceRecord:
    return MediaSourceRecord(generation_id=generation_id, artifact_id=artifact.artifact_id,
                             filename=filename, kind=artifact.kind, mime_type=artifact.mime_type, sha256=artifact.sha256,
                             url=f'/api/generations/history/{generation_id}/artifacts/{artifact.artifact_id}',
                             poster_artifact_id=artifact.poster_artifact_id)


def saved_media_filename(item: dict, artifact: MediaArtifact, path: Path) -> str:
    """Use the displayed import name without changing the verified storage path."""
    settings = item.get('settings', {})
    if settings.get('schema_version') == MEDIA_IMPORT_SCHEMA and artifact.role == 'primary':
        name = settings.get('original_filename')
        if (isinstance(name, str) and name == PurePosixPath(name.replace('\\', '/')).name
                and PurePosixPath(name).suffix.lower() == path.suffix.lower()
                and not any(ord(character) < 32 or ord(character) == 127 for character in name)):
            return name
    return path.name


def get_media_source(history: HistoryRepository, data_dir: Path, target: MediaSourceTarget) -> MediaSourceRecord:
    from app.services.video_local_adapter import HistoryVideoSourceResolver
    from app.core.config import AppConfig

    item = history.get(target.generation_id)
    if item is not None:
        roots = HistoryVideoSourceResolver(history, AppConfig(data_dir=data_dir)).accepted_roots(item)
        try:
            artifact, path = resolve_registered_media_artifact(
                item, target.artifact_id, accepted_roots=roots, allowed_kinds={'image', 'video', 'audio'},
            )
        except MediaArtifactAccessError:
            pass
        else:
            return _record(target.generation_id, artifact, saved_media_filename(item, artifact, path))
    raise HTTPException(404, {'code': 'assistant_media_not_found'})
