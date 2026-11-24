"""Retry receipts written inside the transaction that performs the core action."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequestIdentity:
    request_id: str
    action: str
    arguments: dict[str, Any]


class ActionRequestConflict(ValueError):
    def __init__(self):
        super().__init__('action_request_conflict')


def create_action_receipt_table(connection: sqlite3.Connection) -> None:
    connection.execute('''create table if not exists action_receipts (
        request_id text primary key, action text not null,
        arguments_json text not null, result_json text not null
    )''')


def read_action_receipt(connection: sqlite3.Connection, request_id: str,
                        action: str, arguments_json: str) -> dict[str, Any] | None:
    row = connection.execute(
        'select action, arguments_json, result_json from action_receipts where request_id=?',
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    if row[0] != action or row[1] != arguments_json:
        raise ActionRequestConflict()
    return json.loads(row[2])


def save_action_receipt(connection: sqlite3.Connection, request_id: str,
                        action: str, arguments_json: str, result_json: str) -> None:
    connection.execute('insert into action_receipts values (?, ?, ?, ?)',
                       (request_id, action, arguments_json, result_json))
