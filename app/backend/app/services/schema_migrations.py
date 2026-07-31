from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3

from app.services.atomic_json import write_json_atomic
from app.services.schema_versions import CURRENT_SCHEMA_VERSIONS, SchemaVersions
from app.services.user_settings import UserSettings


FailureInjector = Callable[[str, int, int], None]
_SOURCE_SETTINGS_VERSION = {
    "1.0.1": 0,
    "1.0.2": 1,
    "1.0.3": 1,
}


class SchemaMigrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SchemaMigrationReport:
    from_source_version: str
    versions: SchemaVersions
    applied_steps: tuple[str, ...]


def migrate_data_schemas(
    data_dir: Path,
    *,
    source_version: str,
    failure_injector: FailureInjector | None = None,
) -> SchemaMigrationReport:
    root = data_dir.absolute()
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "user-settings.json"
    settings_existed = settings_path.exists()
    settings_snapshot = settings_path.read_bytes() if settings_existed else b""
    settings_steps = migrate_settings_schema(
        settings_path,
        source_version=source_version,
        failure_injector=failure_injector,
    )
    try:
        database_steps = migrate_history_database_path(
            root / "asteria.db",
            failure_injector=failure_injector,
        )
    except Exception:
        _restore_file_snapshot(
            settings_path,
            existed=settings_existed,
            content=settings_snapshot,
        )
        raise
    return SchemaMigrationReport(
        from_source_version=source_version,
        versions=CURRENT_SCHEMA_VERSIONS,
        applied_steps=tuple(sorted({*settings_steps, *database_steps})),
    )


def migrate_settings_schema(
    path: Path,
    *,
    source_version: str,
    failure_injector: FailureInjector | None = None,
) -> tuple[str, ...]:
    existed = path.exists()
    if existed:
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaMigrationError("settings migration failed: unreadable settings") from exc
        if not isinstance(raw_payload, dict):
            raise SchemaMigrationError("settings migration failed: settings root must be an object")
        payload = dict(raw_payload)
    else:
        payload = {}

    version = _settings_version(payload, source_version=source_version, existed=existed)
    if version > CURRENT_SCHEMA_VERSIONS.settings:
        raise SchemaMigrationError("newer settings schema is not supported")

    applied: list[str] = []
    try:
        while version < CURRENT_SCHEMA_VERSIONS.settings:
            next_version = version + 1
            if version == 0:
                payload.setdefault("library", {"saved_filters": []})
                if not isinstance(payload["library"], dict):
                    payload["library"] = {"saved_filters": []}
                payload["library"].setdefault("saved_filters", [])
            elif version == 1:
                payload["schema_version"] = 2
            elif version == 2:
                payload.setdefault("appearance", {"theme": "system"})
            else:
                raise SchemaMigrationError(f"settings migration path is missing at version {version}")
            if failure_injector is not None:
                failure_injector("settings", version, next_version)
            payload["schema_version"] = next_version
            applied.append(f"settings:{version}->{next_version}")
            version = next_version
        normalized = UserSettings.model_validate(payload).model_dump(mode="json")
        write_json_atomic(path, normalized)
    except SchemaMigrationError:
        raise
    except Exception as exc:
        raise SchemaMigrationError("settings migration failed") from exc
    return tuple(applied)


def migrate_history_database_path(
    path: Path,
    *,
    failure_injector: FailureInjector | None = None,
) -> tuple[str, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(path) as connection:
            return migrate_history_database_schema(connection, failure_injector=failure_injector)
    except SchemaMigrationError:
        raise
    except sqlite3.Error as exc:
        raise SchemaMigrationError("database migration failed") from exc


def migrate_history_database_schema(
    connection: sqlite3.Connection,
    *,
    failure_injector: FailureInjector | None = None,
) -> tuple[str, ...]:
    owns_transaction = not connection.in_transaction
    savepoint = "asteria_schema_migration"
    try:
        if owns_transaction:
            connection.execute("begin immediate")
        else:
            connection.execute(f"savepoint {savepoint}")
        version = _database_version(connection)
        if version > CURRENT_SCHEMA_VERSIONS.database:
            raise SchemaMigrationError("newer database schema is not supported")

        applied: list[str] = []
        while version < CURRENT_SCHEMA_VERSIONS.database:
            next_version = version + 1
            if version == 0:
                _create_schema_tables(connection)
            elif version == 1:
                connection.execute(
                    "create index if not exists idx_generations_created_at on generations(created_at)"
                ) if _table_exists(connection, "generations") else None
            elif version == 2:
                from app.services.operation_repository import create_operation_tables

                create_operation_tables(connection)
            elif version == 3:
                from app.services.managed_downloads import create_managed_download_tables
                from app.services.operation_repository import create_operation_tables
                from app.services.preview_registry import create_preview_registry_table

                create_operation_tables(connection)
                create_preview_registry_table(connection)
                create_managed_download_tables(connection)
            elif version == 4:
                connection.execute(
                    """
                    create table if not exists history_tag_index (
                        generation_id integer not null,
                        kind text not null,
                        normalized_key text not null,
                        value text not null,
                        ordinal integer not null,
                        primary key(generation_id, kind, normalized_key)
                    )
                    """
                )
                connection.execute(
                    "create index if not exists history_tag_catalog_idx "
                    "on history_tag_index(kind, normalized_key, generation_id desc)"
                )
            elif version == 5:
                from app.services.discord_overrides import create_discord_override_tables
                from app.services.discord_profiles import create_discord_profile_tables
                from app.services.discord_receipts import create_discord_receipt_tables

                create_discord_profile_tables(connection)
                create_discord_receipt_tables(connection)
                create_discord_override_tables(connection)
            else:
                raise SchemaMigrationError(f"database migration path is missing at version {version}")
            if failure_injector is not None:
                failure_injector("database", version, next_version)
            _create_schema_tables(connection)
            timestamp = datetime.now(UTC).isoformat()
            connection.execute(
                """
                insert into asteria_schema_metadata(domain, version, updated_at)
                values ('database', ?, ?)
                on conflict(domain) do update set version = excluded.version, updated_at = excluded.updated_at
                """,
                (next_version, timestamp),
            )
            connection.execute(
                """
                insert into asteria_migration_journal(
                    domain, from_version, to_version, status, started_at, completed_at, error_code
                ) values ('database', ?, ?, 'completed', ?, ?, '')
                """,
                (version, next_version, timestamp, timestamp),
            )
            applied.append(f"database:{version}->{next_version}")
            version = next_version

        if owns_transaction:
            connection.commit()
        else:
            connection.execute(f"release savepoint {savepoint}")
        return tuple(applied)
    except Exception as exc:
        if owns_transaction:
            connection.rollback()
        else:
            connection.execute(f"rollback to savepoint {savepoint}")
            connection.execute(f"release savepoint {savepoint}")
        if isinstance(exc, SchemaMigrationError):
            raise
        raise SchemaMigrationError("database migration failed") from exc


def _settings_version(payload: dict, *, source_version: str, existed: bool) -> int:
    value = payload.get("schema_version")
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaMigrationError("settings migration failed: invalid schema version")
        return value
    if not existed and source_version == "clean":
        return CURRENT_SCHEMA_VERSIONS.settings
    try:
        return _SOURCE_SETTINGS_VERSION[source_version]
    except KeyError as exc:
        raise SchemaMigrationError(f"unsupported source version: {source_version}") from exc


def _database_version(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, "asteria_schema_metadata"):
        return 0
    row = connection.execute(
        "select version from asteria_schema_metadata where domain = 'database'"
    ).fetchone()
    if row is None:
        return 0
    value = row[0]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaMigrationError("database migration failed: invalid schema version")
    return value


def _create_schema_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        create table if not exists asteria_schema_metadata (
            domain text primary key,
            version integer not null,
            updated_at text not null
        )
        """
    )
    connection.execute(
        """
        create table if not exists asteria_migration_journal (
            id integer primary key autoincrement,
            domain text not null,
            from_version integer not null,
            to_version integer not null,
            status text not null,
            started_at text not null,
            completed_at text not null,
            error_code text not null default ''
        )
        """
    )


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _restore_file_snapshot(path: Path, *, existed: bool, content: bytes) -> None:
    try:
        if not existed:
            path.unlink(missing_ok=True)
            return
        temporary = path.with_name(f".{path.name}.schema-rollback.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    except OSError as exc:
        raise SchemaMigrationError("settings rollback failed after database migration failure") from exc
