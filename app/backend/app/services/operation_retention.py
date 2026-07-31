from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from app.schemas.operations import TERMINAL_OPERATION_STATUSES
from app.services.operation_repository import OperationRepository


@dataclass(frozen=True)
class OperationRetentionResult:
    removed_operations: int
    remaining_active_operations: int
    remaining_terminal_operations: int


class OperationRetentionService:
    def __init__(
        self,
        repository: OperationRepository,
        *,
        terminal_ttl: timedelta = timedelta(hours=24),
        terminal_max_count: int = 200,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if terminal_ttl.total_seconds() < 0 or terminal_max_count < 0:
            raise ValueError("operation_retention_policy_invalid")
        self.repository = repository
        self.terminal_ttl = terminal_ttl
        self.terminal_max_count = terminal_max_count
        self.clock = clock or (lambda: datetime.now(UTC))

    def cleanup(self) -> OperationRetentionResult:
        now = self.clock()
        active = self.repository.list(
            statuses=("queued", "preparing", "running", "cancelling"),
            limit=1000,
            include_dismissed=True,
        )
        terminal = self.repository.list(
            statuses=TERMINAL_OPERATION_STATUSES,
            limit=1000,
            include_dismissed=True,
        )
        newest_first = sorted(
            terminal,
            key=lambda item: (item.completed_at or item.updated_at, item.operation_id),
            reverse=True,
        )
        removable: list[str] = []
        for index, operation in enumerate(newest_first):
            completed_at = operation.completed_at or operation.updated_at
            expired = now - completed_at >= self.terminal_ttl
            over_count = index >= self.terminal_max_count
            if operation.dismissed_at is not None or expired or over_count:
                removable.append(operation.operation_id)
        removed = self.repository.delete_terminal_metadata(removable)
        return OperationRetentionResult(
            removed_operations=removed,
            remaining_active_operations=len(active),
            remaining_terminal_operations=max(0, len(terminal) - removed),
        )
