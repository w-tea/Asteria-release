from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable


PREVIEW_REGISTRY_SQL = """
create table if not exists operation_previews (
    operation_id text not null,
    preview_path text not null,
    output_root text not null,
    ordinal integer not null,
    registered_at text not null,
    terminal_at text,
    is_current integer not null,
    primary key(operation_id, preview_path)
)
"""


def create_preview_registry_table(connection: sqlite3.Connection) -> None:
    connection.execute(PREVIEW_REGISTRY_SQL)
    connection.execute(
        "create index if not exists operation_previews_terminal_idx "
        "on operation_previews(terminal_at, registered_at)"
    )


@dataclass(frozen=True)
class PreviewCleanupResult:
    deleted_preview_files: int
    skipped_preview_files: int
    reclaimed_preview_bytes: int


class PreviewRegistry:
    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.clock = clock or (lambda: datetime.now(UTC))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            create_preview_registry_table(connection)

    def register(
        self,
        operation_id: str,
        preview_path: Path,
        *,
        output_root: Path,
        current: bool = True,
    ) -> None:
        preview = self._canonical_preview(preview_path, output_root, require_file=True)
        root = output_root.resolve(strict=True)
        now = self.clock().isoformat()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("begin immediate")
            if current:
                connection.execute(
                    "update operation_previews set is_current = 0 where operation_id = ?",
                    (operation_id,),
                )
            ordinal = connection.execute(
                "select coalesce(max(ordinal), -1) + 1 from operation_previews where operation_id = ?",
                (operation_id,),
            ).fetchone()[0]
            connection.execute(
                """
                insert into operation_previews(
                    operation_id, preview_path, output_root, ordinal,
                    registered_at, terminal_at, is_current
                ) values (?, ?, ?, ?, ?, null, ?)
                on conflict(operation_id, preview_path) do update set
                    ordinal = excluded.ordinal,
                    registered_at = excluded.registered_at,
                    is_current = excluded.is_current
                """,
                (
                    operation_id,
                    str(preview),
                    str(root),
                    ordinal,
                    now,
                    int(current),
                ),
            )
            stale_rows = connection.execute(
                """
                select preview_path from operation_previews
                where operation_id = ? and is_current = 0 and terminal_at is null
                order by ordinal desc limit -1 offset 32
                """,
                (operation_id,),
            ).fetchall()
            connection.executemany(
                "delete from operation_previews where operation_id = ? and preview_path = ?",
                ((operation_id, row[0]) for row in stale_rows),
            )

    def mark_terminal(self, operation_id: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                update operation_previews set terminal_at = ?
                where operation_id = ? and terminal_at is null
                """,
                (self.clock().isoformat(), operation_id),
            )

    def operation_paths(self, operation_id: str) -> tuple[str, ...]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                select preview_path from operation_previews
                where operation_id = ? order by ordinal desc
                """,
                (operation_id,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def cleanup(self, *, terminal_ttl: timedelta = timedelta(hours=24)) -> PreviewCleanupResult:
        cutoff = self.clock() - terminal_ttl
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                select operation_id, preview_path, output_root
                from operation_previews
                where terminal_at is not null and terminal_at <= ?
                order by terminal_at asc
                """,
                (cutoff.isoformat(),),
            ).fetchall()
        deleted = skipped = reclaimed = 0
        for operation_id, path_text, root_text in rows:
            path = Path(path_text)
            root = Path(root_text)
            try:
                validated = self._canonical_preview(path, root, require_file=True)
                with sqlite3.connect(self.db_path) as connection:
                    active_reference = connection.execute(
                        """
                        select 1 from operation_previews
                        where preview_path = ? and terminal_at is null limit 1
                        """,
                        (path_text,),
                    ).fetchone()
                if active_reference is not None:
                    skipped += 1
                    continue
                size = validated.stat().st_size
                validated.unlink()
            except (OSError, RuntimeError, ValueError):
                skipped += 1
                continue
            with sqlite3.connect(self.db_path) as connection:
                cursor = connection.execute(
                    """
                    delete from operation_previews
                    where operation_id = ? and preview_path = ? and terminal_at is not null
                    """,
                    (operation_id, path_text),
                )
            if cursor.rowcount:
                deleted += 1
                reclaimed += size
            else:
                skipped += 1
        return PreviewCleanupResult(deleted, skipped, reclaimed)

    @staticmethod
    def _canonical_preview(path: Path, output_root: Path, *, require_file: bool) -> Path:
        raw_root = Path(os.path.abspath(output_root))
        raw_preview_root = raw_root / "_previews"
        raw_path = Path(os.path.abspath(path))
        if raw_path.parent != raw_preview_root:
            raise ValueError("preview_path_not_owned")
        root = raw_root.resolve(strict=True)
        preview_root = raw_preview_root.resolve(strict=True)
        preview = raw_path.resolve(strict=require_file)
        if preview_root.parent != root or preview.parent != preview_root:
            raise ValueError("preview_path_not_owned")
        for candidate in (raw_root, raw_preview_root, raw_path):
            if not candidate.exists() and candidate == raw_path and not require_file:
                continue
            details = os.lstat(candidate)
            attributes = getattr(details, "st_file_attributes", 0)
            if stat.S_ISLNK(details.st_mode) or attributes & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
            ):
                raise ValueError("preview_reparse_not_allowed")
        if require_file and not preview.is_file():
            raise ValueError("preview_path_not_file")
        return preview
