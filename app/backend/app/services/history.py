from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import unicodedata

from app.services.generation_metadata import metadata_tags


MAX_USER_TAGS = 160


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
    rating integer not null default 0,
    favorite integer not null default 0,
    tags_json text not null default '[]',
    generated_tags_json text not null default '[]',
    notes text not null default '',
    folder_id integer,
    created_at text not null default current_timestamp
)
"""


class HistoryRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as con:
            self._drop_legacy_table(con)
            con.execute(GENERATIONS_TABLE_SQL)
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
        tags: list[str] | None = None,
        generated_tags: list[str] | None = None,
    ) -> int:
        normalized_tags = self._normalize_tags(tags or [])
        normalized_generated_tags = self._normalize_tags(generated_tags or [])
        with sqlite3.connect(self.db_path) as con:
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
                    tags_json,
                    generated_tags_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(normalized_tags, ensure_ascii=False),
                    json.dumps(normalized_generated_tags, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def list_recent(self, limit: int = 50, folder: str = "all") -> list[dict]:
        where, params = self._folder_where(folder)
        with sqlite3.connect(self.db_path) as con:
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

    def list_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
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
    ) -> list[dict]:
        query = query.strip().lower()
        tag = tag.strip().lower()
        if limit <= 0:
            return []
        folder_where, folder_params = self._folder_where(folder)
        clauses = [folder_where.removeprefix("where ")] if folder_where else []
        params = list(folder_params)
        if profile:
            clauses.append("g.profile = ?")
            params.append(profile)
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
        with sqlite3.connect(self.db_path) as con:
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

    def get(self, generation_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as con:
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
        with sqlite3.connect(self.db_path) as con:
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
        with sqlite3.connect(self.db_path) as con:
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
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "insert into history_folders(name) values (?) on conflict(name) do update set name = excluded.name returning id",
                (normalized,),
            )
            folder_id = int(cur.fetchone()[0])
        return self.get_folder(folder_id) or {"id": folder_id, "name": normalized, "created_at": "", "item_count": 0}

    def get_folder(self, folder_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as con:
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
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                update generations
                set output_paths_json = ?
                where id = ?
                """,
                (json.dumps(output_paths, ensure_ascii=False), generation_id),
            )

    def update_settings(self, generation_id: int, settings: dict) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                update generations
                set settings_json = ?
                where id = ?
                """,
                (json.dumps(settings, ensure_ascii=False), generation_id),
            )

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
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                """
                update generations
                set status = ?, settings_json = ?, output_paths_json = ?
                where id = ?
                """,
                (status, settings_json, output_paths_json, generation_id),
            )
            if cur.rowcount != 1:
                raise ValueError("generation_not_found")

    def mutate_generation_state(
        self,
        generation_id: int,
        mutation: Callable[[dict], tuple[dict, list[str]]],
    ) -> dict:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            con.execute("begin immediate")
            item = self._get_with_connection(con, generation_id)
            if item is None:
                raise ValueError("generation_not_found")
            settings, output_paths = mutation(item)
            cur = con.execute(
                """
                update generations
                set settings_json = ?, output_paths_json = ?
                where id = ?
                """,
                (
                    json.dumps(settings, ensure_ascii=False),
                    json.dumps(output_paths, ensure_ascii=False),
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
        with sqlite3.connect(self.db_path) as con:
            cur = con.execute("delete from generations where id = ?", (generation_id,))
            return cur.rowcount > 0

    def bulk_delete(self, generation_ids: list[int]) -> list[dict]:
        unique_ids = list(dict.fromkeys(int(value) for value in generation_ids))
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            targets = [
                (generation_id, self._get_with_connection(con, generation_id))
                for generation_id in unique_ids
            ]
            missing_ids = [generation_id for generation_id, item in targets if item is None]
            if missing_ids:
                raise HistoryBulkDeleteConflictError(missing_ids)
            placeholders = ", ".join("?" for _ in unique_ids)
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
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
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
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
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
            "output_paths": self._json_list(row["output_paths_json"]),
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
