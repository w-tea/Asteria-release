from __future__ import annotations

from pathlib import Path
import sqlite3

from app.services.operation_repository import create_operation_tables


class OperationArtifactIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        with sqlite3.connect(db_path) as connection:
            create_operation_tables(connection)

    def referenced(self, artifact_ref: str) -> bool:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "select 1 from operation_artifact_refs where artifact_ref = ? limit 1",
                (artifact_ref,),
            ).fetchone()
        return row is not None

    def operation_refs(self, operation_id: str) -> tuple[str, ...]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                select artifact_ref from operation_artifact_refs
                where operation_id = ? order by artifact_ref
                """,
                (operation_id,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def referencing_operations(self, artifact_ref: str) -> tuple[str, ...]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                select operation_id from operation_artifact_refs
                where artifact_ref = ? order by operation_id
                """,
                (artifact_ref,),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def media_artifact_operations(self, artifact_id: str) -> tuple[str, ...]:
        if not artifact_id.startswith("media:"):
            raise ValueError("media_artifact_id_invalid")
        return self.referencing_operations(artifact_id)
