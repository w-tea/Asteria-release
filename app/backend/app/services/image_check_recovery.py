"""Recover interrupted file publication before the application starts serving."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat

from app.services.atomic_json import write_json_atomic
from app.services.history import HistoryRepository


PENDING_DIRECTORY = '.image-check-pending'
PUBLICATION_SCHEMA = 'asteria.image-check-publication.v2'


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
        root = Path(record['output_root'])
        directory = Path(record['directory'])
        if (root.resolve() != root or directory.is_symlink() or directory.resolve() != directory
                or directory.parent != root / PENDING_DIRECTORY
                or directory.name != record['publication_id']):
            raise ValueError('image_publication_recovery_path_invalid')
        if record['committed'] or record['cleanup_ready']:
            # This marker committed in the same SQLite transaction as the final
            # history references. Never roll its files back, even if the process
            # died before the in-memory session was marked committed.
            if directory.exists():
                shutil.rmtree(directory)
            history.forget_image_publication(record['publication_id'])
            results.append({'publication_id': record['publication_id'],
                            'status': 'committed' if record['committed'] else 'rolled_back'})
            continue
        journal_path = directory / 'publication.json'
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        if (journal.get('schema_version') != PUBLICATION_SCHEMA
                or journal.get('output_root') != str(root)
                or journal.get('publication_id') != record['publication_id']):
            raise ValueError('image_publication_recovery_journal_invalid')
        quarantine = directory / 'quarantine'
        quarantined = sorted(str(path) for path in quarantine.iterdir()) if quarantine.exists() else []
        for index, move in enumerate(journal['moves']):
            path = Path(move['destination'])
            if (not path.is_relative_to(root) or path.is_relative_to(root / PENDING_DIRECTORY)
                    or path.resolve() != path):
                raise ValueError('image_publication_recovery_destination_invalid')
            # Identity, not an optimistic "moved" flag, covers a crash between
            # the actual rename and its following journal write.
            if owns_file(path, move['identities']):
                quarantine.mkdir(exist_ok=True)
                saved = quarantine / f'{index}-{path.name}'
                move_without_replacement(path, saved)
                quarantined.append(str(saved))
        journal.update(phase='interrupted', quarantined=quarantined)
        write_json_atomic(journal_path, journal)
        history.forget_image_publication(record['publication_id'])
        results.append({'publication_id': record['publication_id'], 'status': 'quarantined', 'paths': quarantined})
    return results
