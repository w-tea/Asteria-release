from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.discord import DiscordPublishedProfile
from app.schemas.generation import Txt2ImgRequest


class DiscordOverrideValidationError(ValueError):
    pass


DiscordOverrideScopeType = Literal["guild", "channel", "user"]


class DiscordOverrideScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: DiscordOverrideScopeType
    guild_id: str = ""
    channel_id: str = ""
    user_id: str = ""

    @model_validator(mode="after")
    def _validate_scope_identity(self) -> "DiscordOverrideScope":
        values = {
            "guild_id": self.guild_id.strip(),
            "channel_id": self.channel_id.strip(),
            "user_id": self.user_id.strip(),
        }
        object.__setattr__(self, "guild_id", values["guild_id"])
        object.__setattr__(self, "channel_id", values["channel_id"])
        object.__setattr__(self, "user_id", values["user_id"])
        required = {
            "guild": ("guild_id",),
            "channel": ("channel_id",),
            "user": ("user_id",),
        }[self.scope]
        if any(not values[field] for field in required):
            raise ValueError("discord override scope identity is incomplete")
        return self


class DiscordSessionOverride(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: DiscordOverrideScope
    fields: dict[str, Any]
    updated_at: datetime
    expires_at: datetime


class DiscordGenerationComposition(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: Txt2ImgRequest
    snapshot: dict[str, Any]


DISCORD_OVERRIDES_TABLE_SQL = """
create table if not exists discord_session_overrides (
    scope text not null,
    guild_id text not null,
    channel_id text not null,
    user_id text not null,
    fields_json text not null,
    updated_at text not null,
    expires_at text not null,
    primary key(scope, guild_id, channel_id, user_id)
)
"""


def create_discord_override_tables(connection: sqlite3.Connection) -> None:
    connection.execute(DISCORD_OVERRIDES_TABLE_SQL)
    connection.execute(
        "create index if not exists discord_overrides_expiry_idx "
        "on discord_session_overrides(expires_at)"
    )


_ALLOWED_OVERRIDE_FIELDS = frozenset(
    {
        "steps",
        "cfg_scale",
        "width",
        "height",
        "batch_size",
        "batch_count",
        "seed",
        "sampler",
        "scheduler",
        "hires_enabled",
        "hires_scale",
        "hires_steps",
        "topaz_enabled",
    }
)


def validate_discord_overrides(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise DiscordOverrideValidationError("discord_override_empty")
    unknown = set(value) - _ALLOWED_OVERRIDE_FIELDS
    if unknown:
        raise DiscordOverrideValidationError("discord_override_field_not_allowed")
    normalized = dict(value)
    try:
        if "steps" in normalized:
            normalized["steps"] = int(normalized["steps"])
            if not 1 <= normalized["steps"] <= 80:
                raise DiscordOverrideValidationError("discord_steps_out_of_range")
        if "cfg_scale" in normalized:
            normalized["cfg_scale"] = float(normalized["cfg_scale"])
            if not 1 <= normalized["cfg_scale"] <= 20:
                raise DiscordOverrideValidationError("discord_cfg_out_of_range")
        for field in ("width", "height"):
            if field in normalized:
                normalized[field] = int(normalized[field])
                if not 256 <= normalized[field] <= 2048 or normalized[field] % 64:
                    raise DiscordOverrideValidationError(f"discord_{field}_out_of_range")
        for field in ("batch_size", "batch_count"):
            if field in normalized:
                normalized[field] = int(normalized[field])
                if not 1 <= normalized[field] <= 4:
                    raise DiscordOverrideValidationError(f"discord_{field}_out_of_range")
        if "seed" in normalized:
            normalized["seed"] = int(normalized["seed"])
            if not -1 <= normalized["seed"] <= 2**63 - 1:
                raise DiscordOverrideValidationError("discord_seed_out_of_range")
        if "hires_scale" in normalized:
            normalized["hires_scale"] = float(normalized["hires_scale"])
            if not 1 <= normalized["hires_scale"] <= 4:
                raise DiscordOverrideValidationError("discord_hires_scale_out_of_range")
        if "hires_steps" in normalized:
            normalized["hires_steps"] = int(normalized["hires_steps"])
            if not 0 <= normalized["hires_steps"] <= 80:
                raise DiscordOverrideValidationError("discord_hires_steps_out_of_range")
        for field in ("hires_enabled", "topaz_enabled"):
            if field in normalized and not isinstance(normalized[field], bool):
                raise DiscordOverrideValidationError(f"discord_{field}_invalid")
        for field in ("sampler", "scheduler"):
            if field in normalized:
                normalized[field] = str(normalized[field]).strip()
                if not normalized[field] or len(normalized[field]) > 80:
                    raise DiscordOverrideValidationError(f"discord_{field}_invalid")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, DiscordOverrideValidationError):
            raise
        raise DiscordOverrideValidationError("discord_override_value_invalid") from exc
    return normalized


class DiscordOverrideRepository:
    MIN_TTL = timedelta(minutes=5)
    MAX_TTL = timedelta(hours=24)

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
            create_discord_override_tables(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def set(
        self,
        scope: DiscordOverrideScope,
        fields: dict[str, Any],
        *,
        ttl: timedelta,
    ) -> DiscordSessionOverride:
        if ttl < self.MIN_TTL or ttl > self.MAX_TTL:
            raise DiscordOverrideValidationError("discord_override_ttl_invalid")
        normalized = validate_discord_overrides(fields)
        now = self.clock()
        expires_at = now + ttl
        with self._connect() as connection:
            connection.execute(
                """
                insert into discord_session_overrides(
                    scope, guild_id, channel_id, user_id,
                    fields_json, updated_at, expires_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                on conflict(scope, guild_id, channel_id, user_id)
                do update set
                    fields_json = excluded.fields_json,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    scope.scope,
                    scope.guild_id,
                    scope.channel_id,
                    scope.user_id,
                    json.dumps(
                        normalized,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return DiscordSessionOverride(
            scope=scope,
            fields=normalized,
            updated_at=now,
            expires_at=expires_at,
        )

    def show(self, scope: DiscordOverrideScope) -> DiscordSessionOverride | None:
        with self._connect() as connection:
            connection.execute(
                "delete from discord_session_overrides where expires_at <= ?",
                (self.clock().isoformat(),),
            )
            row = connection.execute(
                """
                select * from discord_session_overrides
                where scope = ? and guild_id = ? and channel_id = ? and user_id = ?
                """,
                (scope.scope, scope.guild_id, scope.channel_id, scope.user_id),
            ).fetchone()
        return self._row_to_override(row) if row is not None else None

    def clear(self, scope: DiscordOverrideScope) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                delete from discord_session_overrides
                where scope = ? and guild_id = ? and channel_id = ? and user_id = ?
                """,
                (scope.scope, scope.guild_id, scope.channel_id, scope.user_id),
            )
        return cursor.rowcount == 1

    def resolve(
        self,
        *,
        guild_id: str | None,
        channel_id: str | None,
        user_id: str,
    ) -> dict[str, Any]:
        guild = str(guild_id or "").strip()
        channel = str(channel_id or "").strip()
        user = str(user_id or "").strip()
        scopes: list[DiscordOverrideScope] = []
        if guild:
            scopes.append(
                DiscordOverrideScope(
                    scope="guild",
                    guild_id=guild,
                    channel_id="",
                    user_id="",
                )
            )
        if channel:
            scopes.append(
                DiscordOverrideScope(
                    scope="channel",
                    guild_id=guild,
                    channel_id=channel,
                    user_id="",
                )
            )
        if user:
            scopes.append(
                DiscordOverrideScope(
                    scope="user",
                    guild_id=guild,
                    channel_id=channel,
                    user_id=user,
                )
            )
        composed: dict[str, Any] = {}
        for scope in scopes:
            item = self.show(scope)
            if item is not None:
                composed.update(item.fields)
        return composed

    @staticmethod
    def _row_to_override(row: sqlite3.Row) -> DiscordSessionOverride:
        return DiscordSessionOverride(
            scope=DiscordOverrideScope(
                scope=row["scope"],
                guild_id=row["guild_id"],
                channel_id=row["channel_id"],
                user_id=row["user_id"],
            ),
            fields=json.loads(row["fields_json"]),
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )


def compose_discord_generation_request(
    *,
    published: DiscordPublishedProfile,
    preset: dict[str, Any] | None,
    session_overrides: dict[str, Any],
    command_overrides: dict[str, Any],
    prompt: str,
    negative_prompt: str | None,
    resolved_model_slots: dict[str, str],
) -> DiscordGenerationComposition:
    payload = published.request.model_dump(mode="json")
    preset_id: str | None = None
    if preset is not None:
        preset_id = str(preset.get("id") or "").strip() or None
        raw_preset = preset.get("data")
        if isinstance(raw_preset, dict):
            model_fields = set(Txt2ImgRequest.model_fields)
            payload.update(
                {
                    key: value
                    for key, value in raw_preset.items()
                    if key in model_fields
                }
            )
    if session_overrides:
        payload.update(validate_discord_overrides(session_overrides))
    if command_overrides:
        payload.update(validate_discord_overrides(command_overrides))
    final_prompt = prompt.strip()
    if not final_prompt:
        raise DiscordOverrideValidationError("discord_prompt_required")
    payload["prompt"] = final_prompt
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt.strip()
    # Resolution is performed here as an acceptance gate, but execution paths
    # stay out of the immutable Discord operation snapshot. The generation
    # executor resolves the stored stable IDs again immediately before use.
    if published.model_references and not resolved_model_slots:
        raise DiscordOverrideValidationError("discord_model_unresolved")
    payload["model_slot_overrides"] = {}
    request = Txt2ImgRequest.model_validate(payload)
    snapshot = {
        "published_revision": published.revision,
        "published_at": published.published_at.isoformat(),
        "published_profile_name": published.profile_name,
        "preset_id": preset_id,
        "model_references": dict(published.model_references),
        "session_override_fields": sorted(session_overrides),
        "command_override_fields": sorted(command_overrides),
        "composition_order": [
            "published_base",
            "preset",
            "session_override",
            "command_override",
            "command_prompt",
        ],
    }
    return DiscordGenerationComposition(request=request, snapshot=snapshot)
