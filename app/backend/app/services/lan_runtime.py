"""LAN listener lifecycle bound to the current, exact Asteria backend process."""
from __future__ import annotations

import asyncio
from datetime import datetime
import ipaddress
import os
from pathlib import Path
import socket

import psutil
import uvicorn

from app.core.config import AppConfig
from app.mcp.bridge_ownership import registered_connection
from app.services.lan_access import LanAccessRepository
from app.services.lan_gateway import create_lan_gateway
from app.services.runtime_ownership import launcher_state_path, read_launcher_state


def local_addresses() -> set[str]:
    return {address.address for addresses in psutil.net_if_addrs().values() for address in addresses
            if address.family in {socket.AF_INET, socket.AF_INET6}}


def validate_lan_host(host: str, *, assigned: set[str] | None = None) -> str:
    address = ipaddress.ip_address(host)
    if address.is_unspecified or address.is_multicast or address.is_global or host not in (assigned if assigned is not None else local_addresses()):
        raise ValueError('lan_interface_unavailable')
    return host


def core_process_alive(identity: dict) -> bool:
    try:
        process = psutil.Process(identity['pid'])
        return (process.is_running() and abs(process.create_time() - identity['started_at']) < .001
                and os.path.normcase(process.exe()) == os.path.normcase(identity['executable']))
    except (psutil.Error, OSError):
        return False


def core_identity(config: AppConfig) -> dict:
    records = [row for row in read_launcher_state(config)['processes'] if row.get('name') == 'backend']
    if len(records) != 1:
        raise RuntimeError('lan_core_unavailable')
    record = records[0]
    identity = {'pid': record['pid'], 'executable': record['executable_path'],
                'started_at': datetime.fromisoformat(record['started_at_utc'].replace('Z', '+00:00')).timestamp()}
    if not core_process_alive(identity):
        raise RuntimeError('lan_core_unavailable')
    return identity


async def serve_gateway(config: AppConfig, *, host: str, port: int, cert: Path | None = None, key: Path | None = None) -> None:
    host = validate_lan_host(host)
    if not 1 <= port <= 65535 or bool(cert) != bool(key):
        raise ValueError('lan_listener_configuration_invalid')
    identity = core_identity(config)
    access = LanAccessRepository(config.data_dir / 'asteria.db')
    if not access.enabled():
        raise ValueError('lan_disabled')
    bracketed = f'[{host}]' if ':' in host else host
    origin = f'{"https" if cert else "http"}://{bracketed}:{port}'
    server = uvicorn.Server(uvicorn.Config(create_lan_gateway(access, public_origin=origin), host=host, port=port,
        ssl_certfile=str(cert) if cert else None, ssl_keyfile=str(key) if key else None,
        access_log=False, log_level='warning', timeout_graceful_shutdown=5))

    async def watch_owner():
        try:
            while core_process_alive(identity) and await asyncio.to_thread(access.enabled):
                await asyncio.sleep(.5)
        finally:
            # Only this server closes; accepted core jobs are not cancelled.
            server.should_exit = True

    with registered_connection(config.root_dir, launcher_state_path(config), kind='lan-gateway'):
        owner = asyncio.create_task(watch_owner())
        try:
            await server.serve()
        finally:
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
