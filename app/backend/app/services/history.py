from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
import base64
import json
from pathlib import Path
import sqlite3
import unicodedata

from app.schemas.media import MediaArtifact, MediaKind, legacy_image_artifacts
from app.schemas.generation import PromptSyntaxGenerationRecord
from app.services.generation_metadata import metadata_tags
from app.services.schema_migrations import migrate_history_database_schema
from app.services.video_sequence_history import is_hidden_sequence_child


MAX_USER_TAGS = 160
HISTORY_PAGE_SCAN_BATCH = 128
HISTORY_TAG_CATALOG_LIMIT = 1_000
_active_publication: ContextVar[tuple[Path, sqlite3.Connection] | None] = ContextVar("history_publication", default=None)


@dataclass(frozen=True)
class HistoryPage:
    items: list[dict]
    next_cursor: str | None
    has_more: bool


def _history_cursor_payload(cursor: str) -> tuple[int, int | None]:
    try:
        padded = cursor + ("=" * (-len(cursor) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        scanned_id = int(payload["scan"])
        pending_id = payload.get("pending")
        if pending_id is not None:
            pending_id = int(pending_id)
        if payload.get("v") != 1 or scanned_id <= 0 or (pending_id is not None and pending_id != scanned_id):
            raise ValueError
        return scanned_id, pending_id
    except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid_history_cursor") from error


def encode_history_cursor(scanned_id: int, *, pending_id: int | None = None) -> str:
    payload = {"v": 1, "scan": int(scanned_id)}
    if pending_id is not None:
        payload["pending"] = int(pending_id)
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return encoded.rstrip("=")


def decode_history_cursor(cursor: str) -> int:
    return _history_cursor_payload(cursor)[0]


class HistoryTagLimitExceededError(ValueError):
    pass


class HistoryBulkDeleteConflictError(ValueError):
    def __init__(self, missing_ids: list[int]):
        self.missing_ids = missing_ids
        super().__init__("history_bulk_delete_conflict")


GENERATIONS_TABLE_SQL = """
create table if not exists generations (
    id integer primary key autoincrement,
    status text not null,
    profile text not null,
    prompt text not null,
    expanded_prompt text not null,
    negative_prompt text not null,
    settings_json text not null,
    model_slots_json text not null,
    loras_json text not null,
    embeddings_json text not null default '[]',
    output_paths_json text not null,
    media_kind text not null default 'image',
    artifacts_json text not null default '[]',
    rating integer not null default 0,
    favorite integer not null default 0,
    tags_json text not null default '[]',
    generated_tags_json text not null default '[]',
    notes text not null default '',
    folder_id integer,
    created_at text not null default current_timestamp
)
"""

HISTORY_TAG_INDEX_SQL = """
create table if not exists history_tag_index (
    generation_id integer not null,
    kind text not null,
    normalized_key text not null,
    value text not null,
    ordinal integer not null,
    primary key(generation_id, kind, normalized_key)
)
"""

IMAGE_PUBLICATIONS_SQL = """
create table if not exists image_check_publications (
    publication_id text primary key,
    output_root text not null,
    directory text not null,
    generation_id integer,
    committed integer not null default 0,
    cleanup_ready integer not null default 0
)
"""


@contextmanager
def _history_connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Commit or roll back the transaction, then release the Windows file handle."""
    active = _active_publication.get()
    if active is not None and active[0] == db_path.resolve():
        yield active[1]
        return
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            yield connection


class HistoryRepository:
    def begin_image_publication(self, publication_id: str, output_root: Path, directory: Path) -> None:
        with _history_connection(self.db_path) as con:
            con.execute('insert into image_check_publications(publication_id, output_root, directory) values (?, ?, ?)',
                        (publication_id, str(output_root), str(directory)))

    def complete_image_publication(self, publication_id: str, generation_id: int) -> None:
        active = _active_publication.get()
        if active is None or active[0] != self.db_path.resolve():
            raise RuntimeError('image_publication_requires_history_transaction')
        active[1].execute('update image_check_publications set committed = 1, generation_id = ? where publication_id = ?',
                          (generation_id, publication_id))

    def list_image_publications(self) -> list[dict]:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            return [dict(row) for row in con.execute('select * from image_check_publications')]

    def image_publication_committed(self, publication_id: str) -> bool:
        with _history_connection(self.db_path) as con:
            row = con.execute('select committed from image_check_publications where publication_id = ?',
                              (publication_id,)).fetchone()
            return row is not None and row[0] == 1

    def prepare_image_publication_cleanup(self, publication_id: str) -> None:
        with _history_connection(self.db_path) as con:
            con.execute('update image_check_publications set cleanup_ready = 1 where publication_id = ?',
                        (publication_id,))

    def forget_image_publication(self, publication_id: str) -> None:
        with _history_connection(self.db_path) as con:
            con.execute('delete from image_check_publications where publication_id = ?', (publication_id,))

    @contextmanager
    def publication(self) -> Iterator[None]:
        """Publish final references together; callers perform no remote work here."""
        if _active_publication.get() is not None:
            raise RuntimeError("history_publication_already_active")
        with _history_connection(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            token = _active_publication.set((self.db_path.resolve(), connection))
            try:
                yield
            finally:
                _active_publication.reset(token)

    @staticmethod
    def registered_output_source(db_path: Path, generation_id: int) -> dict | None:
        """Read an existing output registration without initializing or migrating history."""
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("select id, media_kind, settings_json, output_paths_json from generations where id = ?",
                (generation_id,)).fetchone()
        if row is None:
            return None
        settings, paths = json.loads(row["settings_json"]), json.loads(row["output_paths_json"])
        if not isinstance(settings, dict) or not isinstance(paths, list):
            raise ValueError("invalid_output_registration")
        return {"id": row["id"], "media_kind": row["media_kind"], "settings": settings, "output_paths": paths}

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _history_connection(db_path) as con:
            self._drop_legacy_table(con)
            con.execute(GENERATIONS_TABLE_SQL)
            con.execute(HISTORY_TAG_INDEX_SQL)
            con.execute(IMAGE_PUBLICATIONS_SQL)
            if 'cleanup_ready' not in {row[1] for row in con.execute('pragma table_info(image_check_publications)')}:
                con.execute('alter table image_check_publications add column cleanup_ready integer not null default 0')
            con.execute(
                "create index if not exists history_tag_catalog_idx "
                "on history_tag_index(kind, normalized_key, generation_id desc)"
            )
            con.execute(
                """
                create table if not exists history_folders (
                    id integer primary key,
                    name text not null unique,
                    created_at text not null default current_timestamp
                )
                """
            )
            self._backfill_generated_tags(con)
            self._backfill_tag_index(con)
            migrate_history_database_schema(con)

    def recover_interrupted_media_generations(self) -> list[int]:
        """Recover only inflight history owned by restart-interrupted media jobs."""
        media_kinds = {'video.generate': 'video', 'video.upscale': 'video',
                       'video.sequence.clip': 'video', 'video.sequence.assemble': 'video',
                       'music.generate': 'audio'}
        active_statuses = {'queued', 'preparing', 'running', 'cancelling'}
        with _history_connection(self.db_path) as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute("select 1 from sqlite_master where type = 'table' and name = 'operations'").fetchone() is None:
                return []
            rows = con.execute(
                "select kind, status, error_code, request_snapshot_json from operations "
                "where kind in ('video.generate', 'video.upscale', 'video.sequence.clip', "
                "'video.sequence.assemble', 'music.generate') "
                "and status in ('queued', 'preparing', 'running', 'cancelling', 'interrupted')"
            ).fetchall()
            interrupted: set[tuple[int, str]] = set()
            active: set[tuple[int, str]] = set()
            for kind, status, error_code, raw_snapshot in rows:
                if status not in active_statuses and error_code != 'operation_interrupted_by_restart':
                    continue
                try:
                    snapshot = json.loads(raw_snapshot)
                except (TypeError, ValueError):
                    continue
                if not isinstance(snapshot, dict):
                    continue
                candidates = [snapshot]
                batch = snapshot.get('batch')
                if isinstance(batch, dict) and isinstance(batch.get('candidates'), list):
                    candidates.extend(item for item in batch['candidates'] if isinstance(item, dict))
                target = active if status in active_statuses else interrupted
                for candidate in candidates:
                    identity = candidate.get('generation_id')
                    if type(identity) is int and identity > 0:
                        target.add((identity, media_kinds[kind]))
            recovered = []
            for identity, media_kind in sorted(interrupted - active):
                updated = con.execute(
                    "update generations set status = 'interrupted' "
                    "where id = ? and media_kind = ? and status in ('queued', 'running')",
                    (identity, media_kind),
                )
                if updated.rowcount:
                    recovered.append(identity)
            return recovered

    def recover_interrupted_upscale_inputs(self) -> list[int]:
        """Release verified borrowed inputs, never unfinished generated payloads."""
        from app.services.generation_working_artifacts import (
            GenerationWorkingArtifactError,
            GenerationWorkingArtifactRegistry,
        )
        from app.services.media_artifacts import file_sha256

        with _history_connection(self.db_path) as con:
            if con.execute("select 1 from sqlite_master where type='table' and name='operations'").fetchone() is None:
                return []
            rows = con.execute(
                "select status, error_code, request_snapshot_json from operations "
                "where kind='video.upscale' and status in ('queued','preparing','running','cancelling','interrupted')"
            ).fetchall()
        active_statuses = {'queued', 'preparing', 'running', 'cancelling'}
        active: set[tuple[str, str, str]] = set()
        candidates = []
        histories = {}
        for status, error_code, raw in rows:
            try:
                snapshot = json.loads(raw)
                request = snapshot['request']
                identity, source_id = snapshot['generation_id'], request['source_generation_id']
                if type(identity) is not int or type(source_id) is not int:
                    continue
                for generation_id in (identity, source_id):
                    if generation_id not in histories:
                        histories[generation_id] = self.get(generation_id)
                child, source = histories[identity], histories[source_id]
                if (not child or not source or child['media_kind'] != 'video'
                        or source['media_kind'] != 'video' or source['status'] != 'completed'):
                    continue
                settings, original = child['settings'], source['settings']
                original_root = Path(original['output_root']).expanduser()
                if not original_root.is_absolute():
                    continue
                root = str(original_root.resolve(strict=True))
                owner = snapshot['bundle_operation_id']
                expected = {
                    snapshot['bundle_artifact_id']: ('upscale_work_bundle', snapshot['bundle_sha256']),
                    snapshot['latent_artifact_id']: ('av_latent', snapshot['latent_sha256']),
                }
                if len(expected) != 2 or not all(isinstance(key, str) for key in expected):
                    continue
                if (original['upscale_work_bundle_operation_id'] != owner
                        or original['upscale_work_bundle_id'] != snapshot['bundle_artifact_id']):
                    continue
                keys = {(root, owner, key) for key in expected}
                if status in active_statuses:
                    active.update(keys)
                    continue
                if (error_code != 'operation_interrupted_by_restart'
                        or child['status'] != 'interrupted'
                        or settings['request_snapshot'] != request
                        or settings['source_generation_id'] != source_id
                        or settings['source_bundle_artifact_id'] != snapshot['bundle_artifact_id']
                        or settings['source_latent_artifact_id'] != snapshot['latent_artifact_id']
                        or settings['source_bundle_sha256'] != snapshot['bundle_sha256']
                        or settings['source_latent_sha256'] != snapshot['latent_sha256']):
                    continue
                candidates.append((identity, source_id, request['source_artifact_id'], root, owner, expected, keys))
            except (KeyError, TypeError, ValueError, OSError, RuntimeError):
                continue
        recovered = set()
        for identity, source_id, media_id, root, owner, expected, keys in candidates:
            if keys & active:
                continue
            try:
                registry = GenerationWorkingArtifactRegistry(Path(root))
                records = {record.artifact_id: record for record in registry.list_public(owner)}
                selected = [records[key] for key in expected]
                if any(record.generation_id != source_id or record.media_artifact_id != media_id
                       or record.kind != expected[record.artifact_id][0]
                       or record.state not in {'available', 'in_use'} for record in selected):
                    continue
                borrowed = tuple(record.artifact_id for record in selected if record.state == 'in_use')
                if not borrowed:
                    continue
                for key, (kind, checksum) in expected.items():
                    payload = registry.resolve_for_adapter(operation_id=owner, artifact_id=key,
                        expected_kind=kind, allow_in_use=True)
                    if file_sha256(payload) != checksum:
                        raise GenerationWorkingArtifactError('working_artifact_changed')
                registry.finish_use_many(owner, borrowed, outcome='available', operation_status='interrupted')
                recovered.add(identity)
            except (GenerationWorkingArtifactError, KeyError, ValueError, OSError):
                continue
        return sorted(recovered)

    def add(
        self,
        *,
        status: str,
        profile: str,
        prompt: str,
        expanded_prompt: str,
        negative_prompt: str,
        settings: dict,
        model_slots: dict[str, str],
        loras: list[dict],
        embeddings: list[str],
        output_paths: list[str],
        media_kind: MediaKind = "image",
        artifacts: list[dict] | list[MediaArtifact] | None = None,
        tags: list[str] | None = None,
        generated_tags: list[str] | None = None,
    ) -> int:
        self._validate_prompt_syntax_settings(settings)
        normalized_tags = self._normalize_tags(tags or [])
        normalized_generated_tags = self._normalize_tags(generated_tags or [])
        self._validate_media_kind(media_kind)
        serialized_artifacts = self._serialize_artifacts(
            media_kind=media_kind,
            artifacts=artifacts or [],
        )
        with _history_connection(self.db_path) as con:
            cur = con.execute(
                """
                insert into generations(
                    status,
                    profile,
                    prompt,
                    expanded_prompt,
                    negative_prompt,
                    settings_json,
                    model_slots_json,
                    loras_json,
                    embeddings_json,
                    output_paths_json,
                    media_kind,
                    artifacts_json,
                    tags_json,
                    generated_tags_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    status,
                    profile,
                    prompt,
                    expanded_prompt,
                    negative_prompt,
                    json.dumps(settings, ensure_ascii=False),
                    json.dumps(model_slots, ensure_ascii=False),
                    json.dumps(loras, ensure_ascii=False),
                    json.dumps(embeddings, ensure_ascii=False),
                    json.dumps(output_paths, ensure_ascii=False),
                    media_kind,
                    json.dumps(serialized_artifacts, ensure_ascii=False),
                    json.dumps(normalized_tags, ensure_ascii=False),
                    json.dumps(normalized_generated_tags, ensure_ascii=False),
                ),
            )
            generation_id = int(cur.lastrowid)
            if artifacts is None and media_kind == "image":
                serialized_artifacts = self._legacy_artifact_payloads(generation_id, output_paths)
                con.execute(
                    "update generations set artifacts_json = ? where id = ?",
                    (json.dumps(serialized_artifacts, ensure_ascii=False), generation_id),
                )
            self._replace_tag_index(
                con,
                generation_id,
                tags=normalized_tags,
                generated_tags=normalized_generated_tags,
            )
            return generation_id

    def tag_catalog(self) -> dict[str, list[str]]:
        with _history_connection(self.db_path) as con:
            rows = con.execute(
                """
                with latest as (
                    select kind, value, generation_id, ordinal,
                           row_number() over (
                               partition by kind, normalized_key
                               order by generation_id desc
                           ) as identity_rank
                    from history_tag_index
                ),
                bounded as (
                    select kind, value, generation_id, ordinal,
                           row_number() over (
                               partition by kind
                               order by generation_id desc, ordinal
                           ) as catalog_rank
                    from latest
                    where identity_rank = 1
                )
                select kind, value, generation_id
                from bounded
                where catalog_rank <= ?
                order by kind, generation_id desc, ordinal
                """,
                (HISTORY_TAG_CATALOG_LIMIT,),
            ).fetchall()
        return {
            "tags": [value for kind, value, _ in rows if kind == "user"],
            "generated_tags": [value for kind, value, _ in rows if kind == "generated"],
        }

    def migrate_model_references(self, candidates: list[object]) -> int:
        from app.services.model_reference_migration import migrate_generation_reference_metadata

        changed = 0
        with _history_connection(self.db_path) as con:
            rows = con.execute(
                "select id, settings_json, model_slots_json, loras_json from generations"
            ).fetchall()
            for generation_id, settings_json, model_slots_json, loras_json in rows:
                settings = self._json_dict(settings_json)
                payload = {
                    **settings,
                    "model_slots": self._json_dict(model_slots_json),
                    "loras": [
                        item for item in self._json_list(loras_json) if isinstance(item, dict)
                    ],
                }
                migrated = migrate_generation_reference_metadata(payload, candidates)
                for key in ("model_slots", "loras"):
                    if key in settings:
                        migrated[key] = settings[key]
                    else:
                        migrated.pop(key, None)
                if migrated == settings:
                    continue
                con.execute(
                    "update generations set settings_json = ? where id = ?",
                    (json.dumps(migrated, ensure_ascii=False), generation_id),
                )
                changed += 1
        return changed

    def list_recent(self, limit: int = 50, folder: str = "all") -> list[dict]:
        where, params = self._folder_where(folder)
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                select g.*, f.name as folder_name
                from generations g
                left join history_folders f on f.id = g.folder_id
                {where}
                order by id desc
                limit ?
                """,
                (*params, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def page(
        self,
        *,
        limit: int = 72,
        cursor: str = "",
        query: str = "",
        profile: str = "",
        favorite: bool = False,
        tag: str = "",
        folder: str = "all",
        media_kind: str = "all",
        condition_verdict: str = "",
    ) -> HistoryPage:
        if limit < 1 or limit > 100:
            raise ValueError("invalid_history_limit")
        query = query.strip().lower()
        tag = tag.strip().lower()
        self._validate_media_filter(media_kind)
        scanned_id: int | None = None
        pending_id: int | None = None
        if cursor:
            scanned_id, pending_id = _history_cursor_payload(cursor)

        folder_where, folder_params = self._folder_where(folder)
        clauses = [folder_where.removeprefix("where ")] if folder_where else []
        params = list(folder_params)
        if condition_verdict:
            from app.services.image_conditions import condition_filter_clause
            condition_clause, condition_params = condition_filter_clause(self.db_path, condition_verdict)
            clauses.append(condition_clause)
            params.extend(condition_params)
        if profile:
            clauses.append("g.profile = ?")
            params.append(profile)
        if favorite:
            clauses.append("g.favorite != 0")
        if media_kind != "all":
            clauses.append("g.media_kind = ?")
            params.append(media_kind)
        base_where = f"where {' and '.join(clauses)}" if clauses else ""

        items: list[dict] = []
        next_pending_id: int | None = None
        scan_before = scanned_id
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            if pending_id is not None:
                pending = self._page_row(con, pending_id, base_where, params)
                if pending is not None:
                    pending_item = self._row_to_dict(pending)
                    if self._matches_page_filters(pending_item, query=query, tag=tag):
                        items.append(pending_item)
                scan_before = pending_id

            exhausted = False
            while not exhausted and next_pending_id is None:
                cursor_clause = "g.id < ?" if scan_before is not None else ""
                where = self._combine_where(base_where, cursor_clause)
                page_params = [*params]
                if scan_before is not None:
                    page_params.append(scan_before)
                rows = con.execute(
                    f"""
                    select g.*, f.name as folder_name
                    from generations g
                    left join history_folders f on f.id = g.folder_id
                    {where}
                    order by g.id desc
                    limit ?
                    """,
                    (*page_params, HISTORY_PAGE_SCAN_BATCH),
                ).fetchall()
                if not rows:
                    exhausted = True
                    break
                for row in rows:
                    scan_before = int(row["id"])
                    item = self._row_to_dict(row)
                    if not self._matches_page_filters(item, query=query, tag=tag):
                        continue
                    if len(items) < limit:
                        items.append(item)
                    else:
                        next_pending_id = int(row["id"])
                        break
                if len(rows) < HISTORY_PAGE_SCAN_BATCH:
                    exhausted = True

        if next_pending_id is None:
            return HistoryPage(items=items, next_cursor=None, has_more=False)
        return HistoryPage(
            items=items,
            next_cursor=encode_history_cursor(next_pending_id, pending_id=next_pending_id),
            has_more=True,
        )

    def tag_page(self, *, before_id: int | None = None, limit: int = 128) -> tuple[list[dict], int | None]:
        if limit < 1:
            return [], None
        where = "where id < ?" if before_id is not None else ""
        params = (before_id, limit) if before_id is not None else (limit,)
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"""
                select id, tags_json, generated_tags_json
                from generations
                {where}
                order by id desc
                limit ?
                """,
                params,
            ).fetchall()
        items = [
            {
                "id": int(row["id"]),
                "tags": self._normalize_tags(self._json_list(row["tags_json"])),
                "generated_tags": self._normalize_tags(self._json_list(row["generated_tags_json"])),
            }
            for row in rows
        ]
        next_before = int(rows[-1]["id"]) if len(rows) == limit else None
        return items, next_before

    def list_all(self) -> list[dict]:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                select g.*, f.name as folder_name
                from generations g
                left join history_folders f on f.id = g.folder_id
                order by id desc
                """
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search(
        self,
        *,
        limit: int = 50,
        query: str = "",
        profile: str = "",
        favorite: bool = False,
        tag: str = "",
        folder: str = "all",
        media_kind: str = "all",
    ) -> list[dict]:
        query = query.strip().lower()
        tag = tag.strip().lower()
        self._validate_media_filter(media_kind)
        if limit <= 0:
            return []
        folder_where, folder_params = self._folder_where(folder)
        clauses = [folder_where.removeprefix("where ")] if folder_where else []
        params = list(folder_params)
        if profile:
            clauses.append("g.profile = ?")
            params.append(profile)
        if media_kind != "all":
            clauses.append("g.media_kind = ?")
            params.append(media_kind)
        where = f"where {' and '.join(clauses)}" if clauses else ""
        sql = f"""
            select g.*, f.name as folder_name
            from generations g
            left join history_folders f on f.id = g.folder_id
            {where}
            order by g.id desc
        """
        if not query and not tag and not favorite:
            sql += " limit ?"
            params.append(limit)

        filtered = []
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            for row in con.execute(sql, params):
                item = self._row_to_dict(row)
                if favorite and not item["favorite"]:
                    continue
                tags = [str(value).lower() for value in item["tags"]]
                generated_tags = [str(value).lower() for value in item["generated_tags"]]
                searchable_tags = [*tags, *generated_tags]
                if tag and tag not in searchable_tags:
                    continue
                haystack = " ".join(
                    [
                        item["prompt"],
                        item["expanded_prompt"],
                        item["negative_prompt"],
                        item["profile"],
                        item["notes"],
                        " ".join(tags),
                        " ".join(generated_tags),
                        json.dumps(item["settings"], ensure_ascii=False),
                        json.dumps(item["model_slots"], ensure_ascii=False),
                    ]
                ).lower()
                if query and query not in haystack:
                    continue
                filtered.append(item)
                if len(filtered) >= limit:
                    break
        return filtered

    def _page_row(
        self,
        con: sqlite3.Connection,
        generation_id: int,
        base_where: str,
        params: list[object],
    ) -> sqlite3.Row | None:
        id_clause = "g.id = ?"
        where = self._combine_where(base_where, id_clause)
        return con.execute(
            f"""
            select g.*, f.name as folder_name
            from generations g
            left join history_folders f on f.id = g.folder_id
            {where}
            """,
            (*params, generation_id),
        ).fetchone()

    def _combine_where(self, base_where: str, clause: str) -> str:
        if not clause:
            return base_where
        if base_where:
            return f"{base_where} and {clause}"
        return f"where {clause}"

    def _matches_page_filters(self, item: dict, *, query: str, tag: str) -> bool:
        if is_hidden_sequence_child(item):
            return False
        tags = [str(value).lower() for value in item["tags"]]
        generated_tags = [str(value).lower() for value in item["generated_tags"]]
        if tag and tag not in [*tags, *generated_tags]:
            return False
        if not query:
            return True
        haystack = " ".join(
            [
                item["prompt"],
                item["expanded_prompt"],
                item["negative_prompt"],
                item["profile"],
                item["notes"],
                " ".join(tags),
                " ".join(generated_tags),
                json.dumps(item["settings"], ensure_ascii=False),
                json.dumps(item["model_slots"], ensure_ascii=False),
            ]
        ).lower()
        return query in haystack

    def get(self, generation_id: int) -> dict | None:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                select g.*, f.name as folder_name
                from generations g
                left join history_folders f on f.id = g.folder_id
                where g.id = ?
                """,
                (generation_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_folders(self) -> list[dict]:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                select f.id, f.name, f.created_at, count(g.id) as item_count
                from history_folders f
                left join generations g on g.folder_id = f.id
                group by f.id
                order by lower(f.name)
                """
            ).fetchall()
            unclassified_count = con.execute(
                "select count(*) from generations where folder_id is null"
            ).fetchone()[0]
            total_count = con.execute("select count(*) from generations").fetchone()[0]
        folders = [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "created_at": row["created_at"],
                "item_count": int(row["item_count"]),
            }
            for row in rows
        ]
        return [
            {"id": None, "name": "All", "system_key": "all", "created_at": "", "item_count": int(total_count)},
            {"id": None, "name": "Unclassified", "system_key": "unclassified", "created_at": "", "item_count": int(unclassified_count)},
            *folders,
        ]

    def referenced_output_paths(self) -> set[str]:
        with _history_connection(self.db_path) as con:
            rows = con.execute("select output_paths_json from generations").fetchall()
        paths: set[str] = set()
        for (raw_paths,) in rows:
            try:
                values = json.loads(raw_paths)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str):
                    paths.add(value)
        return paths

    def create_folder(self, name: str) -> dict:
        normalized = " ".join(name.strip().split())
        if not normalized:
            raise ValueError("folder_name_required")
        with _history_connection(self.db_path) as con:
            cur = con.execute(
                "insert into history_folders(name) values (?) on conflict(name) do update set name = excluded.name returning id",
                (normalized,),
            )
            folder_id = int(cur.fetchone()[0])
        return self.get_folder(folder_id) or {"id": folder_id, "name": normalized, "created_at": "", "item_count": 0}

    def get_folder(self, folder_id: int) -> dict | None:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                select f.id, f.name, f.created_at, count(g.id) as item_count
                from history_folders f
                left join generations g on g.folder_id = f.id
                where f.id = ?
                group by f.id
                """,
                (folder_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "created_at": row["created_at"],
            "item_count": int(row["item_count"]),
        }

    def update_output_paths(self, generation_id: int, output_paths: list[str]) -> None:
        with _history_connection(self.db_path) as con:
            row = con.execute(
                "select media_kind from generations where id = ?", (generation_id,)
            ).fetchone()
            artifacts_json = (
                json.dumps(self._legacy_artifact_payloads(generation_id, output_paths), ensure_ascii=False)
                if row is not None and row[0] == "image"
                else None
            )
            con.execute(
                """
                update generations
                set output_paths_json = ?,
                    artifacts_json = coalesce(?, artifacts_json)
                where id = ?
                """,
                (json.dumps(output_paths, ensure_ascii=False), artifacts_json, generation_id),
            )

    def update_media_artifacts(
        self,
        generation_id: int,
        *,
        media_kind: MediaKind,
        artifacts: list[dict] | list[MediaArtifact],
        output_paths: list[str],
    ) -> None:
        self._validate_media_kind(media_kind)
        serialized_artifacts = self._serialize_artifacts(
            media_kind=media_kind,
            artifacts=artifacts,
        )
        registered_paths = set(output_paths)
        if any(artifact["path"] not in registered_paths for artifact in serialized_artifacts):
            raise ValueError("media artifact path must be registered in output paths")
        with _history_connection(self.db_path) as con:
            cur = con.execute(
                """
                update generations
                set media_kind = ?, artifacts_json = ?, output_paths_json = ?
                where id = ?
                """,
                (
                    media_kind,
                    json.dumps(serialized_artifacts, ensure_ascii=False),
                    json.dumps(output_paths, ensure_ascii=False),
                    generation_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("generation_not_found")

    def complete_media_generation(
        self,
        generation_id: int,
        *,
        status: str,
        settings: dict,
        media_kind: MediaKind,
        artifacts: list[dict] | list[MediaArtifact],
        output_paths: list[str],
    ) -> None:
        self._validate_media_kind(media_kind)
        serialized_artifacts = self._serialize_artifacts(
            media_kind=media_kind,
            artifacts=artifacts,
        )
        registered_paths = set(output_paths)
        if any(artifact["path"] not in registered_paths for artifact in serialized_artifacts):
            raise ValueError("media artifact path must be registered in output paths")
        with _history_connection(self.db_path) as con:
            cur = con.execute(
                """
                update generations
                set status = ?, settings_json = ?, media_kind = ?,
                    artifacts_json = ?, output_paths_json = ?
                where id = ?
                """,
                (
                    status,
                    json.dumps(settings, ensure_ascii=False),
                    media_kind,
                    json.dumps(serialized_artifacts, ensure_ascii=False),
                    json.dumps(output_paths, ensure_ascii=False),
                    generation_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("generation_not_found")

    def update_settings(self, generation_id: int, settings: dict) -> None:
        self._validate_prompt_syntax_settings(settings)
        with _history_connection(self.db_path) as con:
            con.execute(
                """
                update generations
                set settings_json = ?
                where id = ?
                """,
                (json.dumps(settings, ensure_ascii=False), generation_id),
            )

    @staticmethod
    def _validate_prompt_syntax_settings(settings: dict) -> None:
        prompt_syntax = settings.get("prompt_syntax")
        if prompt_syntax is not None:
            PromptSyntaxGenerationRecord.model_validate(prompt_syntax)

    def update_generation(
        self,
        generation_id: int,
        *,
        status: str,
        settings: dict,
        output_paths: list[str],
    ) -> None:
        settings_json = json.dumps(settings, ensure_ascii=False)
        output_paths_json = json.dumps(output_paths, ensure_ascii=False)
        with _history_connection(self.db_path) as con:
            row = con.execute(
                "select media_kind from generations where id = ?", (generation_id,)
            ).fetchone()
            artifacts_json = (
                json.dumps(self._legacy_artifact_payloads(generation_id, output_paths), ensure_ascii=False)
                if row is not None and row[0] == "image"
                else None
            )
            cur = con.execute(
                """
                update generations
                set status = ?, settings_json = ?, output_paths_json = ?,
                    artifacts_json = coalesce(?, artifacts_json)
                where id = ?
                """,
                (status, settings_json, output_paths_json, artifacts_json, generation_id),
            )
            if cur.rowcount != 1:
                raise ValueError("generation_not_found")

    def mutate_generation_state(
        self,
        generation_id: int,
        mutation: Callable[[dict], tuple[dict, list[str]]],
    ) -> dict:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            con.execute("begin immediate")
            item = self._get_with_connection(con, generation_id)
            if item is None:
                raise ValueError("generation_not_found")
            settings, output_paths = mutation(item)
            artifacts_json = (
                json.dumps(self._legacy_artifact_payloads(generation_id, output_paths), ensure_ascii=False)
                if item["media_kind"] == "image"
                else None
            )
            cur = con.execute(
                """
                update generations
                set settings_json = ?, output_paths_json = ?,
                    artifacts_json = coalesce(?, artifacts_json)
                where id = ?
                """,
                (
                    json.dumps(settings, ensure_ascii=False),
                    json.dumps(output_paths, ensure_ascii=False),
                    artifacts_json,
                    generation_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError("generation_not_found")
            updated = self._get_with_connection(con, generation_id)
            if updated is None:
                raise ValueError("generation_not_found")
            return updated

    def append_external_event(self, generation_id: int, event: dict) -> dict | None:
        def append_event(item: dict) -> tuple[dict, list[str]]:
            settings = dict(item["settings"])
            current_events = settings.get("external_events", [])
            events = (
                [entry for entry in current_events if isinstance(entry, dict)]
                if isinstance(current_events, list)
                else []
            )
            event_payload = dict(event)
            event_payload.setdefault("created_at", datetime.now(UTC).isoformat())
            settings["external_events"] = [*events, event_payload]
            return settings, list(item["output_paths"])

        try:
            return self.mutate_generation_state(generation_id, append_event)
        except ValueError as exc:
            if str(exc) == "generation_not_found":
                return None
            raise

    def delete(self, generation_id: int) -> bool:
        with _history_connection(self.db_path) as con:
            con.execute("delete from history_tag_index where generation_id = ?", (generation_id,))
            cur = con.execute("delete from generations where id = ?", (generation_id,))
            return cur.rowcount > 0

    def bulk_delete(self, generation_ids: list[int]) -> list[dict]:
        unique_ids = list(dict.fromkeys(int(value) for value in generation_ids))
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            targets = [
                (generation_id, self._get_with_connection(con, generation_id))
                for generation_id in unique_ids
            ]
            missing_ids = [generation_id for generation_id, item in targets if item is None]
            if missing_ids:
                raise HistoryBulkDeleteConflictError(missing_ids)
            placeholders = ", ".join("?" for _ in unique_ids)
            con.execute(
                f"delete from history_tag_index where generation_id in ({placeholders})",
                unique_ids,
            )
            cur = con.execute(
                f"delete from generations where id in ({placeholders})",
                unique_ids,
            )
            if cur.rowcount != len(unique_ids):
                raise HistoryBulkDeleteConflictError(unique_ids)
            return [item for _, item in targets if item is not None]

    def update_curation(
        self,
        generation_id: int,
        *,
        rating: int | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        folder_id: int | None = None,
        update_folder: bool = False,
    ) -> dict | None:
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            if not con.in_transaction:
                con.execute('BEGIN IMMEDIATE')
            item = self._get_with_connection(con, generation_id)
            if item is None:
                return None
            values = self._next_curation_values(
                item,
                rating=rating,
                favorite=favorite,
                tags=tags,
                notes=notes,
                folder_id=folder_id,
                update_folder=update_folder,
            )
            self._validate_folder_with_connection(con, values[4])
            self._update_curation_with_connection(con, generation_id, values)
            if tags is not None:
                self._replace_tag_index(
                    con,
                    generation_id,
                    tags=values[2],
                    generated_tags=item["generated_tags"],
                )
            return self._get_with_connection(con, generation_id)

    def bulk_update_curation(
        self,
        generation_ids: list[int],
        *,
        rating: int | None = None,
        favorite: bool | None = None,
        tags_add: list[str] | None = None,
        tags_remove: list[str] | None = None,
        folder_id: int | None = None,
        update_folder: bool = False,
    ) -> list[dict]:
        normalized_add = self._normalize_bulk_tags(tags_add or [])
        normalized_remove = self._normalize_bulk_tags(tags_remove or [])
        unique_ids = list(dict.fromkeys(int(value) for value in generation_ids))
        with _history_connection(self.db_path) as con:
            con.row_factory = sqlite3.Row
            if not con.in_transaction:
                con.execute('BEGIN IMMEDIATE')
            targets = [
                (generation_id, item)
                for generation_id in unique_ids
                if (item := self._get_with_connection(con, generation_id)) is not None
            ]
            updates = [
                (
                    generation_id,
                    self._next_curation_values(
                        item,
                        rating=rating,
                        favorite=favorite,
                        folder_id=folder_id,
                        update_folder=update_folder,
                        tags_add=normalized_add,
                        tags_remove=normalized_remove,
                    ),
                )
                for generation_id, item in targets
            ]
            for _, values in updates:
                self._validate_folder_with_connection(con, values[4])
            for generation_id, values in updates:
                self._update_curation_with_connection(con, generation_id, values)
                if normalized_add or normalized_remove:
                    item = next(item for item_id, item in targets if item_id == generation_id)
                    self._replace_tag_index(
                        con,
                        generation_id,
                        tags=values[2],
                        generated_tags=item["generated_tags"],
                    )
            return [
                item
                for generation_id, _ in updates
                if (item := self._get_with_connection(con, generation_id)) is not None
            ]

    def _get_with_connection(self, con: sqlite3.Connection, generation_id: int) -> dict | None:
        row = con.execute(
            """
            select g.*, f.name as folder_name
            from generations g
            left join history_folders f on f.id = g.folder_id
            where g.id = ?
            """,
            (generation_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def _next_curation_values(
        self,
        item: dict,
        *,
        rating: int | None = None,
        favorite: bool | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        folder_id: int | None = None,
        update_folder: bool = False,
        tags_add: list[str] | None = None,
        tags_remove: list[str] | None = None,
    ) -> tuple[int, bool, list[str], str, int | None]:
        next_tags = item["tags"] if tags is None else self._normalize_tags(tags)
        if tags_add or tags_remove:
            next_tags = self._apply_bulk_tags(next_tags, tags_add or [], tags_remove or [])
        return (
            item["rating"] if rating is None else max(0, min(5, int(rating))),
            item["favorite"] if favorite is None else bool(favorite),
            next_tags,
            item["notes"] if notes is None else notes,
            item["folder_id"] if not update_folder else folder_id,
        )

    def _update_curation_with_connection(
        self,
        con: sqlite3.Connection,
        generation_id: int,
        values: tuple[int, bool, list[str], str, int | None],
    ) -> None:
        rating, favorite, tags, notes, folder_id = values
        con.execute(
            """
            update generations
            set rating = ?, favorite = ?, tags_json = ?, notes = ?, folder_id = ?
            where id = ?
            """,
            (
                rating,
                1 if favorite else 0,
                json.dumps(tags, ensure_ascii=False),
                notes,
                folder_id,
                generation_id,
            ),
        )

    def _validate_folder_with_connection(self, con: sqlite3.Connection, folder_id: int | None) -> None:
        if folder_id is None:
            return
        row = con.execute("select 1 from history_folders where id = ?", (int(folder_id),)).fetchone()
        if row is None:
            raise ValueError("folder_not_found")

    def _normalize_bulk_tags(self, tags: list[str]) -> list[str]:
        normalized = []
        seen = set()
        for tag in tags:
            value = " ".join(str(tag).strip().split())
            key = self._bulk_tag_key(value)
            if value and key not in seen:
                normalized.append(value)
                seen.add(key)
        return normalized

    def _apply_bulk_tags(self, current: list[str], tags_add: list[str], tags_remove: list[str]) -> list[str]:
        remove_keys = {self._bulk_tag_key(tag) for tag in tags_remove}
        updated = [tag for tag in current if self._bulk_tag_key(tag) not in remove_keys]
        existing_keys = {self._bulk_tag_key(tag) for tag in updated}
        for tag in tags_add:
            key = self._bulk_tag_key(tag)
            if key in remove_keys or key in existing_keys:
                continue
            updated.append(tag)
            existing_keys.add(key)
        if len(updated) > MAX_USER_TAGS:
            raise HistoryTagLimitExceededError("history_tag_limit_exceeded")
        return updated

    def _bulk_tag_key(self, tag: str) -> str:
        return unicodedata.normalize("NFKC", tag).upper()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        media_kind = row["media_kind"] if row["media_kind"] in {"image", "video", "audio"} else "image"
        output_paths = self._json_list(row["output_paths_json"])
        artifacts = self._deserialize_artifacts(row["artifacts_json"])
        if media_kind == "image" and not artifacts:
            artifacts = self._legacy_artifact_payloads(int(row["id"]), output_paths)
        return {
            "id": row["id"],
            "status": row["status"],
            "profile": row["profile"],
            "prompt": row["prompt"],
            "expanded_prompt": row["expanded_prompt"],
            "negative_prompt": row["negative_prompt"],
            "settings": self._json_dict(row["settings_json"]),
            "model_slots": self._json_dict(row["model_slots_json"]),
            "loras": self._json_list(row["loras_json"]),
            "embeddings": self._json_list(row["embeddings_json"]),
            "output_paths": output_paths,
            "media_kind": media_kind,
            "artifacts": artifacts,
            "rating": self._bounded_int(row["rating"], default=0, minimum=0, maximum=5),
            "favorite": self._bool_int(row["favorite"]),
            "tags": self._normalize_tags(self._json_list(row["tags_json"])),
            "generated_tags": self._normalize_tags(self._json_list(row["generated_tags_json"])),
            "notes": row["notes"],
            "folder_id": self._optional_int(row["folder_id"]),
            "folder_name": row["folder_name"] if "folder_name" in row.keys() else None,
            "created_at": row["created_at"],
        }

    def _json_dict(self, value: str) -> dict:
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _json_list(self, value: str) -> list:
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def _validate_media_kind(self, media_kind: str) -> None:
        if media_kind not in {"image", "video", "audio"}:
            raise ValueError("invalid_media_kind")

    def _validate_media_filter(self, media_kind: str) -> None:
        if media_kind not in {"all", "image", "video", "audio"}:
            raise ValueError("invalid_media_kind")

    def _serialize_artifacts(
        self,
        *,
        media_kind: str,
        artifacts: list[dict] | list[MediaArtifact],
    ) -> list[dict]:
        serialized: list[dict] = []
        parsed_artifacts: list[MediaArtifact] = []
        for value in artifacts:
            artifact = MediaArtifact.model_validate(value)
            if artifact.role == "primary" and artifact.kind != media_kind:
                raise ValueError("primary artifact kind must match history media kind")
            parsed_artifacts.append(artifact)
            serialized.append(artifact.model_dump(mode="json", exclude={"frame_rate"}))
        artifact_by_id = {artifact.artifact_id: artifact for artifact in parsed_artifacts}
        if len(artifact_by_id) != len(parsed_artifacts):
            raise ValueError("media artifact identifiers must be unique")
        for artifact in parsed_artifacts:
            if artifact.poster_artifact_id is None:
                continue
            poster = artifact_by_id.get(artifact.poster_artifact_id)
            if poster is None or poster.kind != "poster" or poster.role != "poster":
                raise ValueError("video poster reference must resolve to a poster artifact")
        return serialized

    def _deserialize_artifacts(self, value: str) -> list[dict]:
        payload = self._json_list(value)
        artifacts: list[dict] = []
        for candidate in payload:
            try:
                artifact = MediaArtifact.model_validate(candidate)
            except ValueError:
                continue
            artifacts.append(artifact.model_dump(mode="json", exclude={"frame_rate"}))
        return artifacts

    def _legacy_artifact_payloads(self, generation_id: int, output_paths: list[str]) -> list[dict]:
        return [
            artifact.model_dump(mode="json", exclude={"frame_rate"})
            for artifact in legacy_image_artifacts(generation_id, output_paths)
        ]

    def _bounded_int(self, value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    def _optional_int(self, value: object) -> int | None:
        if value in {"", None}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bool_int(self, value: object) -> bool:
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return False

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        normalized = []
        for tag in tags:
            value = " ".join(str(tag).strip().split())
            if value and value not in normalized:
                normalized.append(value)
        return normalized[:160]

    def _drop_legacy_table(self, con: sqlite3.Connection) -> None:
        columns = con.execute("pragma table_info(generations)").fetchall()
        if not columns:
            return
        column_names = {column[1] for column in columns}
        required = {"settings_json", "model_slots_json", "output_paths_json", "created_at"}
        if not required.issubset(column_names):
            con.execute("drop table generations")
            return
        if "embeddings_json" not in column_names:
            con.execute("alter table generations add column embeddings_json text not null default '[]'")
        if "rating" not in column_names:
            con.execute("alter table generations add column rating integer not null default 0")
        if "favorite" not in column_names:
            con.execute("alter table generations add column favorite integer not null default 0")
        if "tags_json" not in column_names:
            con.execute("alter table generations add column tags_json text not null default '[]'")
        if "generated_tags_json" not in column_names:
            con.execute("alter table generations add column generated_tags_json text not null default '[]'")
        if "notes" not in column_names:
            con.execute("alter table generations add column notes text not null default ''")
        if "folder_id" not in column_names:
            con.execute("alter table generations add column folder_id integer")
        self._migrate_generations_to_autoincrement(con)

    def _migrate_generations_to_autoincrement(self, con: sqlite3.Connection) -> None:
        table = con.execute("select sql from sqlite_master where type = 'table' and name = 'generations'").fetchone()
        if table is None or "autoincrement" in str(table[0]).lower():
            return
        con.execute("alter table generations rename to generations_legacy_no_autoincrement")
        con.execute(GENERATIONS_TABLE_SQL)
        columns = [
            "id",
            "status",
            "profile",
            "prompt",
            "expanded_prompt",
            "negative_prompt",
            "settings_json",
            "model_slots_json",
            "loras_json",
            "embeddings_json",
            "output_paths_json",
            "rating",
            "favorite",
            "tags_json",
            "generated_tags_json",
            "notes",
            "folder_id",
            "created_at",
        ]
        legacy_columns = {
            str(row[1])
            for row in con.execute(
                "pragma table_info(generations_legacy_no_autoincrement)"
            ).fetchall()
        }
        for optional_column in ("media_kind", "artifacts_json"):
            if optional_column in legacy_columns:
                columns.append(optional_column)
        column_sql = ", ".join(columns)
        con.execute(
            f"""
            insert into generations({column_sql})
            select {column_sql}
            from generations_legacy_no_autoincrement
            order by id
            """
        )
        con.execute("drop table generations_legacy_no_autoincrement")

    def _backfill_generated_tags(self, con: sqlite3.Connection) -> None:
        original_row_factory = con.row_factory
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                """
                select id, profile, expanded_prompt, prompt, model_slots_json, embeddings_json, loras_json, generated_tags_json
                from generations
                """
            ).fetchall()
            for row in rows:
                try:
                    existing = json.loads(row["generated_tags_json"] or "[]")
                except json.JSONDecodeError:
                    existing = []
                if existing:
                    continue
                try:
                    model_slots = json.loads(row["model_slots_json"] or "{}")
                    embeddings = json.loads(row["embeddings_json"] or "[]")
                    loras = json.loads(row["loras_json"] or "[]")
                except json.JSONDecodeError:
                    continue
                generated_tags = metadata_tags(
                    profile=str(row["profile"] or "unknown"),
                    prompt=str(row["expanded_prompt"] or row["prompt"] or ""),
                    model_slots=model_slots if isinstance(model_slots, dict) else {},
                    embeddings=embeddings if isinstance(embeddings, list) else [],
                    loras=loras if isinstance(loras, list) else [],
                )
                con.execute(
                    "update generations set generated_tags_json = ? where id = ?",
                    (json.dumps(self._normalize_tags(generated_tags), ensure_ascii=False), row["id"]),
                )
        finally:
            con.row_factory = original_row_factory

    def _backfill_tag_index(self, con: sqlite3.Connection) -> None:
        indexed = con.execute("select 1 from history_tag_index limit 1").fetchone()
        if indexed is not None:
            return
        rows = con.execute(
            "select id, tags_json, generated_tags_json from generations order by id"
        ).fetchall()
        for generation_id, tags_json, generated_tags_json in rows:
            self._replace_tag_index(
                con,
                int(generation_id),
                tags=self._normalize_tags(self._json_list(tags_json)),
                generated_tags=self._normalize_tags(self._json_list(generated_tags_json)),
            )

    def _replace_tag_index(
        self,
        con: sqlite3.Connection,
        generation_id: int,
        *,
        tags: list[str],
        generated_tags: list[str],
    ) -> None:
        con.execute("delete from history_tag_index where generation_id = ?", (generation_id,))
        values: list[tuple[int, str, str, str, int]] = []
        for kind, entries in (("user", tags), ("generated", generated_tags)):
            seen: set[str] = set()
            for ordinal, value in enumerate(entries):
                key = unicodedata.normalize("NFKC", value).casefold()
                if value and key not in seen:
                    values.append((generation_id, kind, key, value, ordinal))
                    seen.add(key)
        con.executemany(
            """
            insert into history_tag_index(generation_id, kind, normalized_key, value, ordinal)
            values (?, ?, ?, ?, ?)
            """,
            values,
        )

    def _folder_where(self, folder: str) -> tuple[str, tuple]:
        folder = str(folder or "all").strip().lower()
        if folder in {"", "all"}:
            return "", ()
        if folder in {"unclassified", "uncategorized"}:
            return "where g.folder_id is null", ()
        if folder.startswith("folder:"):
            folder = folder.split(":", 1)[1]
        try:
            folder_id = int(folder)
        except ValueError:
            return "", ()
        return "where g.folder_id = ?", (folder_id,)
