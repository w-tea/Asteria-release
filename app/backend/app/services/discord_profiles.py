from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from app.schemas.discord import (
    DiscordProfileStatus,
    DiscordPublishedProfile,
    DiscordPublishedProfileSummary,
    DiscordPublishRequest,
)
from app.services.discord_overrides import (
    DiscordOverrideValidationError,
    validate_discord_overrides,
)


class DiscordProfileConflictError(RuntimeError):
    pass


class DiscordProfileValidationError(ValueError):
    pass


class DiscordProfileNotPublishedError(LookupError):
    pass


_PROFILE_TABLE_SQL = """
create table if not exists discord_profile_revisions (
    revision integer primary key,
    published_at text not null,
    profile_name text not null,
    preset_id text,
    request_json text not null,
    model_references_json text not null,
    summary_json text not null
)
"""

_POINTER_TABLE_SQL = """
create table if not exists discord_profile_pointer (
    singleton_id integer primary key check(singleton_id = 1),
    pointer_revision integer not null,
    active_revision integer,
    updated_at text not null
)
"""


def create_discord_profile_tables(connection: sqlite3.Connection) -> None:
    connection.execute(_PROFILE_TABLE_SQL)
    connection.execute(_POINTER_TABLE_SQL)
    connection.execute(
        """
        insert or ignore into discord_profile_pointer(
            singleton_id, pointer_revision, active_revision, updated_at
        ) values (1, 0, null, ?)
        """,
        (datetime.now(UTC).isoformat(),),
    )


class DiscordProfileRepository:
    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path
        self.clock = clock or (lambda: datetime.now(UTC))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            create_discord_profile_tables(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def status(self) -> DiscordProfileStatus:
        with self._connect() as connection:
            pointer = connection.execute(
                "select pointer_revision, active_revision from discord_profile_pointer where singleton_id = 1"
            ).fetchone()
            row = None
            if pointer["active_revision"] is not None:
                row = connection.execute(
                    "select * from discord_profile_revisions where revision = ?",
                    (pointer["active_revision"],),
                ).fetchone()
        profile = self._row_to_profile(row) if row is not None else None
        return DiscordProfileStatus(
            pointer_revision=int(pointer["pointer_revision"]),
            active_revision=profile.revision if profile else None,
            published=profile is not None,
            profile=profile,
        )

    def publish(
        self,
        *,
        expected_revision: int,
        profile_name: str,
        preset_id: str | None,
        request: dict[str, Any],
        model_references: dict[str, str],
        summary: dict[str, Any],
    ) -> DiscordProfileStatus:
        with self._connect() as connection:
            connection.execute("begin immediate")
            pointer = connection.execute(
                "select pointer_revision from discord_profile_pointer where singleton_id = 1"
            ).fetchone()
            current_revision = int(pointer["pointer_revision"])
            if current_revision != expected_revision:
                raise DiscordProfileConflictError("discord_publish_revision_conflict")
            next_revision = current_revision + 1
            published_at = self.clock()
            if published_at.tzinfo is None:
                raise ValueError("discord profile clock must include timezone")
            connection.execute(
                """
                insert into discord_profile_revisions(
                    revision, published_at, profile_name, preset_id,
                    request_json, model_references_json, summary_json
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_revision,
                    published_at.isoformat(),
                    profile_name,
                    preset_id,
                    self._json(request),
                    self._json(model_references),
                    self._json(summary),
                ),
            )
            connection.execute(
                """
                update discord_profile_pointer
                set pointer_revision = ?, active_revision = ?, updated_at = ?
                where singleton_id = 1
                """,
                (next_revision, next_revision, published_at.isoformat()),
            )
        return self.status()

    def unpublish(self, *, expected_revision: int) -> DiscordProfileStatus:
        with self._connect() as connection:
            connection.execute("begin immediate")
            pointer = connection.execute(
                "select pointer_revision from discord_profile_pointer where singleton_id = 1"
            ).fetchone()
            current_revision = int(pointer["pointer_revision"])
            if current_revision != expected_revision:
                raise DiscordProfileConflictError("discord_publish_revision_conflict")
            next_revision = current_revision + 1
            connection.execute(
                """
                update discord_profile_pointer
                set pointer_revision = ?, active_revision = null, updated_at = ?
                where singleton_id = 1
                """,
                (next_revision, self.clock().isoformat()),
            )
        return self.status()

    def get_revision(self, revision: int) -> DiscordPublishedProfile:
        with self._connect() as connection:
            row = connection.execute(
                "select * from discord_profile_revisions where revision = ?",
                (revision,),
            ).fetchone()
        if row is None:
            raise DiscordProfileNotPublishedError("discord_profile_revision_not_found")
        return self._row_to_profile(row)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> DiscordPublishedProfile:
        return DiscordPublishedProfile.model_validate(
            {
                "revision": row["revision"],
                "published_at": row["published_at"],
                "profile_name": row["profile_name"],
                "preset_id": row["preset_id"],
                "request": json.loads(row["request_json"]),
                "model_references": json.loads(row["model_references_json"]),
                "summary": json.loads(row["summary_json"]),
            }
        )


class DiscordProfileService:
    def __init__(
        self,
        repository: DiscordProfileRepository,
        *,
        model_exists: Callable[[str], bool],
        preset_exists: Callable[[str], bool],
    ) -> None:
        self.repository = repository
        self.model_exists = model_exists
        self.preset_exists = preset_exists

    def get_status(self) -> DiscordProfileStatus:
        return self.repository.status()

    def publish(self, request: DiscordPublishRequest) -> DiscordProfileStatus:
        if request.preset_id and not self.preset_exists(request.preset_id):
            raise DiscordProfileValidationError("discord_preset_unresolved")
        for stable_model_id in request.model_references.values():
            if not self.model_exists(stable_model_id):
                raise DiscordProfileValidationError("discord_model_unresolved")
        selected_slots = {
            slot_id
            for slot_id, value in request.request.model_slot_overrides.items()
            if str(value).strip()
        }
        if selected_slots - set(request.model_references):
            raise DiscordProfileValidationError("discord_model_reference_missing")
        try:
            validate_discord_overrides(
                {
                    "steps": request.request.steps,
                    "width": request.request.width,
                    "height": request.request.height,
                    "batch_size": request.request.batch_size,
                    "batch_count": request.request.batch_count,
                    "hires_scale": request.request.hires_scale,
                }
            )
        except DiscordOverrideValidationError as exc:
            raise DiscordProfileValidationError(str(exc)) from exc
        summary = DiscordPublishedProfileSummary(
            profile=request.request.profile,
            width=request.request.width,
            height=request.request.height,
            steps=request.request.steps,
            batch_size=request.request.batch_size,
            batch_count=request.request.batch_count,
            hires_enabled=request.request.hires_enabled,
            detailer_enabled=any(unit.enabled for unit in request.request.detailer_units),
            topaz_enabled=request.request.topaz_enabled,
        )
        pathless_request = request.request.model_copy(
            update={"model_slot_overrides": {}}
        )
        return self.repository.publish(
            expected_revision=request.expected_revision,
            profile_name=request.profile_name,
            preset_id=request.preset_id,
            request=pathless_request.model_dump(mode="json"),
            model_references=dict(request.model_references),
            summary=summary.model_dump(mode="json"),
        )

    def unpublish(self, *, expected_revision: int) -> DiscordProfileStatus:
        return self.repository.unpublish(expected_revision=expected_revision)

    def require_published(self) -> DiscordPublishedProfile:
        status = self.repository.status()
        if status.profile is None:
            raise DiscordProfileNotPublishedError("discord_profile_not_published")
        return status.profile
