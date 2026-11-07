"""Recover interrupted file publication before the application starts serving."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import stat

from app.services.atomic_json import write_json_atomic
from app.services.history import HistoryRepository


PENDING_DIRECTORY = '.image-check-pending'
PUBLICATION_SCHEMA = 'asteria.image-check-publication.v2'
logger = logging.getLogger(__name__)


def file_identity(path: Path) -> list[int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_ino == 0:
        raise ValueError('image_publication_file_identity_unavailable')
    return [info.st_dev, info.st_ino]


def owns_file(path: Path, identities: list[list[int]]) -> bool:
    try:
        return not path.is_symlink() and file_identity(path) in identities
    except FileNotFoundError:
        return False


def move_without_replacement(source: Path, destination: Path) -> None:
    # Windows rename is atomic and refuses an existing destination. POSIX rename
    # overwrites, so use an exclusive hard link there (the paths share a root).
    if os.name == 'nt':
        source.rename(destination)
    else:
        os.link(source, destination)
        source.unlink()


def recover_image_publications(history: HistoryRepository) -> list[dict]:
    results = []
    for record in history.list_image_publications():
        publication_id = record['publication_id']
        try:
            result = _recover_publication_files(record)
        except (OSError, ValueError) as exc:
            # Keep the SQLite row and journal for a later startup, for example
            # after the previous output drive is available again.
            logger.warning('Deferred image publication recovery for %s: %s', publication_id, exc)
            results.append({'publication_id': publication_id, 'status': 'deferred'})
            continue
        history.forget_image_publication(publication_id)
        results.append(result)
    return results


def _recover_publication_files(record: dict) -> dict:
    publication_id = record['publication_id']
    root = Path(record['output_root'])
    directory = Path(record['directory'])
    if (root.resolve() != root or directory.is_symlink() or directory.resolve() != directory
            or directory.parent != root / PENDING_DIRECTORY or directory.name != publication_id):
        raise ValueError('image_publication_recovery_path_invalid')
    if not root.is_dir():
        raise ValueError('image_publication_recovery_root_unavailable')
    if record['committed'] or record['cleanup_ready']:
        # This marker committed in the same SQLite transaction as the final
        # history references. Never roll back those files.
        if directory.exists():
            shutil.rmtree(directory)
        return {'publication_id': publication_id, 'status': 'committed' if record['committed'] else 'rolled_back'}
    journal_path = directory / 'publication.json'
    if journal_path.is_symlink():
        raise ValueError('image_publication_recovery_journal_invalid')
    journal = json.loads(journal_path.read_text(encoding='utf-8'))
    if (not isinstance(journal, dict) or journal.get('schema_version') != PUBLICATION_SCHEMA
            or journal.get('output_root') != str(root) or journal.get('publication_id') != publication_id
            or not isinstance(journal.get('moves'), list)):
        raise ValueError('image_publication_recovery_journal_invalid')
    quarantine = directory / 'quarantine'
    if quarantine.resolve() != quarantine:
        raise ValueError('image_publication_recovery_path_invalid')
    moves = []
    for move in journal['moves']:
        if (not isinstance(move, dict) or not isinstance(move.get('destination'), str)
                or not isinstance(move.get('identities'), list)):
            raise ValueError('image_publication_recovery_journal_invalid')
        path = Path(move['destination'])
        if (not path.is_relative_to(root) or path.is_relative_to(root / PENDING_DIRECTORY)
                or path.resolve() != path):
            raise ValueError('image_publication_recovery_destination_invalid')
        moves.append((path, move['identities']))
    quarantined = sorted(str(path) for path in quarantine.iterdir()) if quarantine.exists() else []
    for index, (path, identities) in enumerate(moves):
        # File identity also covers a crash between the rename and journal write.
        if owns_file(path, identities):
            quarantine.mkdir(exist_ok=True)
            saved = quarantine / f'{index}-{path.name}'
            move_without_replacement(path, saved)
            quarantined.append(str(saved))
    journal.update(phase='interrupted', quarantined=quarantined)
    write_json_atomic(journal_path, journal)
    return {'publication_id': publication_id, 'status': 'quarantined', 'paths': quarantined}
