"""Opt-in LAN device access; independent of conversations and AI tool scope."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.reference_mutation import reference_mutation_coordinator


class LanAccessError(ValueError):
    pass


class LanListenerSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    host: str = ''
    port: int = Field(default=3001, ge=1, le=65535)
    certificate_path: str | None = None
    private_key_path: str | None = None

    @model_validator(mode='after')
    def paired_tls_paths(self):
        if bool(self.certificate_path) != bool(self.private_key_path):
            raise ValueError('lan_tls_pair_required')
        return self

    @property
    def origin(self) -> str:
        host = f'[{self.host}]' if ':' in self.host else self.host
        return f'{"https" if self.certificate_path else "http"}://{host}:{self.port}'


@dataclass(frozen=True)
class LanDevice:
    id: str
    name: str
    created_at: float


@dataclass(frozen=True)
class LanPairing:
    token: str
    expires_at: float


@dataclass(frozen=True)
class LanCredential:
    device: LanDevice
    token: str


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


class LanAccessRepository:
    def __init__(self, db_path: Path, *, clock: Callable[[], float] = time.time):
        self.db_path, self.clock = db_path, clock
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.execute('create table if not exists lan_access (id integer primary key check(id=1), enabled integer not null)')
            con.execute('insert or ignore into lan_access values (1, 0)')
            con.execute('''create table if not exists lan_pairings (
                token_hash text primary key, expires_at real not null)''')
            con.execute('''create table if not exists lan_devices (
                id text primary key, name text not null, created_at real not null,
                token_hash text not null unique)''')
            con.execute('create table if not exists lan_listener (id integer primary key check(id=1), settings_json text not null)')

    def listener_settings(self) -> LanListenerSettings:
        with closing(self._connect()) as con:
            row = con.execute('select settings_json from lan_listener where id=1').fetchone()
            return LanListenerSettings.model_validate_json(row[0]) if row else LanListenerSettings()

    def save_listener_settings(self, settings: LanListenerSettings) -> None:
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('insert into lan_listener values (1, ?) on conflict(id) do update set settings_json=excluded.settings_json',
                        (settings.model_dump_json(),))

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _enabled(con) -> bool:
        return bool(con.execute('select enabled from lan_access where id=1').fetchone()[0])

    @staticmethod
    def _device(row) -> LanDevice:
        return LanDevice(row['id'], row['name'], row['created_at'])

    def enabled(self) -> bool:
        with closing(self._connect()) as con:
            return self._enabled(con)

    def set_enabled(self, enabled: bool) -> None:
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('update lan_access set enabled=? where id=1', (int(enabled),))
            if not enabled:
                con.execute('delete from lan_pairings')

    def create_pairing(self) -> LanPairing:
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('begin immediate')
            if not self._enabled(con):
                raise LanAccessError('lan_disabled')
            now = self.clock()
            invitation = LanPairing(secrets.token_urlsafe(32), now + 24 * 60 * 60)
            con.execute('delete from lan_pairings where expires_at<=?', (now,))
            con.execute('insert into lan_pairings values (?, ?)', (_digest(invitation.token), invitation.expires_at))
            return invitation

    def register(self, token: str, name: str) -> LanCredential:
        name = name.strip()
        if not name:
            raise LanAccessError('lan_device_name_required')
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('begin immediate')
            if not self._enabled(con):
                raise LanAccessError('lan_disabled')
            consumed = con.execute('delete from lan_pairings where token_hash=? and expires_at>?',
                                   (_digest(token), self.clock()))
            if consumed.rowcount != 1:
                raise LanAccessError('lan_pairing_invalid')
            credential = LanCredential(LanDevice(str(uuid4()), name, self.clock()), secrets.token_urlsafe(32))
            device = credential.device
            con.execute('insert into lan_devices values (?, ?, ?, ?)',
                        (device.id, device.name, device.created_at, _digest(credential.token)))
            return credential

    def authenticate(self, token: str | None) -> LanDevice | None:
        if not token:
            return None
        with closing(self._connect()) as con:
            row = con.execute('''select d.* from lan_devices d, lan_access a
                                 where a.id=1 and a.enabled=1 and d.token_hash=?''', (_digest(token),)).fetchone()
            return self._device(row) if row else None

    def devices(self) -> list[LanDevice]:
        with closing(self._connect()) as con:
            return [self._device(row) for row in con.execute('select * from lan_devices order by created_at, id')]

    def revoke(self, device_id: str) -> None:
        with reference_mutation_coordinator.acquire_publisher(), closing(self._connect()) as con, con:
            con.execute('delete from lan_devices where id=?', (device_id,))
