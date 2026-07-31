from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.operations import OperationRecord


class DiscordReceiptIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1, max_length=32)
    application_id: str = Field(min_length=1, max_length=64)
    interaction_id: str = Field(min_length=1, max_length=64)

    @field_validator("provider", "application_id", "interaction_id", mode="before")
    @classmethod
    def _normalize_identity(cls, value: object) -> str:
        return str(value or "").strip()

    @property
    def operation_idempotency_key(self) -> str:
        return f"{self.provider}:{self.application_id}:{self.interaction_id}"


class DiscordReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    identity: DiscordReceiptIdentity
    operation_id: str
    requester_id: str
    guild_id: str | None = None
    channel_id: str | None = None
    claimed_at: datetime
    delivery_status: str = "pending"
    delivery_error_code: str | None = None


class DiscordReceiptCleanup(BaseModel):
    model_config = ConfigDict(frozen=True)

    deleted: int = Field(ge=0)
    retained: int = Field(ge=0)


class DiscordOperationClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: OperationRecord
    created: bool


DISCORD_RECEIPTS_TABLE_SQL = """
create table if not exists discord_interaction_receipts (
    provider text not null,
    application_id text not null,
    interaction_id text not null,
    operation_id text not null,
    requester_id text not null,
    guild_id text,
    channel_id text,
    claimed_at text not null,
    delivery_status text not null default 'pending',
    delivery_error_code text,
    primary key(provider, application_id, interaction_id)
)
"""


def create_discord_receipt_tables(connection: sqlite3.Connection) -> None:
    connection.execute(DISCORD_RECEIPTS_TABLE_SQL)
    connection.execute(
        "create index if not exists discord_receipts_claimed_idx "
        "on discord_interaction_receipts(claimed_at desc)"
    )
    connection.execute(
        "create index if not exists discord_receipts_operation_idx "
        "on discord_interaction_receipts(operation_id)"
    )


class DiscordReceiptRepository:
    DEFAULT_MAX_AGE = timedelta(days=30)
    DEFAULT_MAX_ROWS = 20_000

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
            create_discord_receipt_tables(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def get(self, identity: DiscordReceiptIdentity) -> DiscordReceipt:
        with self._connect() as connection:
            row = connection.execute(
                """
                select * from discord_interaction_receipts
                where provider = ? and application_id = ? and interaction_id = ?
                """,
                (identity.provider, identity.application_id, identity.interaction_id),
            ).fetchone()
        if row is None:
            raise LookupError("discord_receipt_not_found")
        return self._row_to_receipt(row)

    def get_by_operation(self, operation_id: str) -> DiscordReceipt:
        with self._connect() as connection:
            row = connection.execute(
                "select * from discord_interaction_receipts where operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise LookupError("discord_receipt_not_found")
        return self._row_to_receipt(row)

    def list(self, *, limit: int = 200) -> list[DiscordReceipt]:
        if limit < 1 or limit > 1000:
            raise ValueError("discord_receipt_limit_invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """
                select * from discord_interaction_receipts
                order by claimed_at desc, rowid desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_receipt(row) for row in rows]

    def update_delivery(
        self,
        identity: DiscordReceiptIdentity,
        *,
        status: str,
        error_code: str | None = None,
    ) -> DiscordReceipt:
        if status not in {"pending", "delivered", "delivery_failed"}:
            raise ValueError("discord_delivery_status_invalid")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                update discord_interaction_receipts
                set delivery_status = ?, delivery_error_code = ?
                where provider = ? and application_id = ? and interaction_id = ?
                """,
                (
                    status,
                    error_code,
                    identity.provider,
                    identity.application_id,
                    identity.interaction_id,
                ),
            )
        if cursor.rowcount != 1:
            raise LookupError("discord_receipt_not_found")
        return self.get(identity)

    def cleanup(
        self,
        *,
        max_age: timedelta = DEFAULT_MAX_AGE,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> DiscordReceiptCleanup:
        if max_age <= timedelta(0) or max_rows < 1 or max_rows > self.DEFAULT_MAX_ROWS:
            raise ValueError("discord_receipt_retention_invalid")
        cutoff = self.clock() - max_age
        with self._connect() as connection:
            connection.execute("begin immediate")
            expired = connection.execute(
                """
                delete from discord_interaction_receipts
                where claimed_at < ?
                """,
                (cutoff.isoformat(),),
            ).rowcount
            overflow = connection.execute(
                """
                delete from discord_interaction_receipts
                where rowid in (
                    select rowid from discord_interaction_receipts
                    order by claimed_at desc, rowid desc
                    limit -1 offset ?
                )
                """,
                (max_rows,),
            ).rowcount
            retained = int(
                connection.execute(
                    "select count(*) from discord_interaction_receipts"
                ).fetchone()[0]
            )
        return DiscordReceiptCleanup(
            deleted=max(0, expired) + max(0, overflow),
            retained=retained,
        )

    @staticmethod
    def _row_to_receipt(row: sqlite3.Row) -> DiscordReceipt:
        return DiscordReceipt(
            identity=DiscordReceiptIdentity(
                provider=row["provider"],
                application_id=row["application_id"],
                interaction_id=row["interaction_id"],
            ),
            operation_id=row["operation_id"],
            requester_id=row["requester_id"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            claimed_at=row["claimed_at"],
            delivery_status=row["delivery_status"],
            delivery_error_code=row["delivery_error_code"],
        )
