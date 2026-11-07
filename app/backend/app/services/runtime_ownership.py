from __future__ import annotations

from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import psutil

from app.core.config import AppConfig
from app.services.atomic_json import write_json_atomic
from app.services.runtime_recovery import RuntimeOwnership


class RuntimeOwnershipError(RuntimeError):
    pass


def same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve(strict=False))) == os.path.normcase(str(Path(right).resolve(strict=False)))


def launcher_state_path(config: AppConfig) -> Path:
    # The launcher sets COMFY_PID_PATH beside its state, including a custom TempRoot.
    return config.comfy_pid_path.with_name('launcher-state.json')


def read_launcher_state(config: AppConfig, *, allow_missing: bool = False) -> dict[str, Any]:
    path = launcher_state_path(config)
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if (not isinstance(payload, dict) or payload.get('version') != 2
                or not same_path(payload['app_root'], config.root_dir)
                or not isinstance(payload.get('processes'), list)
                or not all(isinstance(item, dict) for item in payload['processes'])):
            raise ValueError('invalid launcher identity')
        return payload
    except FileNotFoundError:
        if allow_missing:
            return {'version':2, 'app_root':str(config.root_dir.resolve(strict=False)),
                    'created_at_utc':datetime.now(UTC).isoformat(), 'processes':[]}
        raise RuntimeOwnershipError('runtime_ownership_unverified') from None
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeOwnershipError('runtime_ownership_unverified') from exc


def ownership_from_record(config: AppConfig, record: dict[str, Any]) -> RuntimeOwnership:
    try:
        pid = record['pid']
        started = datetime.fromisoformat(str(record['started_at_utc']).replace('Z','+00:00'))
        executable = record['executable_path']
        if (type(pid) is not int or pid <= 0 or started.tzinfo is None
                or not isinstance(executable,str) or not Path(executable).is_absolute()
                or not math.isfinite(started.timestamp())):
            raise ValueError('invalid runtime identity')
        return RuntimeOwnership(pid, str(Path(executable).resolve(strict=False)),
                                started.timestamp(), str(config.comfy_dir.resolve(strict=False)))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeOwnershipError('runtime_ownership_unverified') from exc


def load_runtime_ownership(config: AppConfig) -> RuntimeOwnership | None:
    payload = read_launcher_state(config)
    records = [item for item in payload['processes'] if item.get('name') == 'comfyui']
    if len(records) > 1:
        raise RuntimeOwnershipError('runtime_ownership_unverified')
    return ownership_from_record(config, records[0]) if records else None


def process_matches(process: psutil.Process, ownership: RuntimeOwnership) -> bool:
    return (process.pid == ownership.pid and same_path(process.exe(),ownership.executable)
            and abs(process.create_time()-ownership.start_time) < .001
            and same_path(process.cwd(),ownership.root) and process.is_running())


def terminate_process_tree(process: psutil.Process) -> bool:
    """Keep the same psutil objects so their PID-reuse checks remain effective."""
    try:
        targets = [*reversed(process.children(recursive=True)), process]
        for target in targets:
            try:
                if target.is_running():
                    target.terminate()
            except psutil.NoSuchProcess:
                continue
        deadline = time.monotonic()+10
        remaining = []
        for target in targets:
            try:
                target.wait(timeout=max(0,deadline-time.monotonic()))
            except psutil.NoSuchProcess:
                continue
            except psutil.TimeoutExpired:
                remaining.append(target)
        deadline = time.monotonic()+5
        for target in remaining:
            try:
                if target.is_running():
                    target.kill()
                target.wait(timeout=max(0,deadline-time.monotonic()))
            except psutil.NoSuchProcess:
                continue
        return all(not target.is_running() for target in targets)
    except (psutil.Error, OSError):
        return False


def record_started_runtime(config: AppConfig, process: psutil.Process) -> None:
    pid = process.pid
    if type(pid) is not int or pid <= 0:
        raise RuntimeOwnershipError('runtime_ownership_unverified')
    try:
        payload = read_launcher_state(config, allow_missing=True)
        if os.name == 'nt':
            command = (f"$ErrorActionPreference='Stop'; $p=Get-Process -Id {pid}; "
                "[ordered]@{name='comfyui';pid=$p.Id;executable_path=$p.Path;"
                "started_at_utc=$p.StartTime.ToUniversalTime().ToString('o');optional=$false}|ConvertTo-Json -Compress")
            result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                capture_output=True,text=True,encoding='utf-8-sig',timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode or result.stderr.strip():
                raise RuntimeOwnershipError('runtime_ownership_unverified')
            record = json.loads(result.stdout)
        else:
            record = {'name':'comfyui','pid':pid,'executable_path':process.exe(),
                      'started_at_utc':datetime.fromtimestamp(process.create_time(),UTC).isoformat(),'optional':False}
        if record.get('name') != 'comfyui' or not process_matches(process,ownership_from_record(config,record)):
            raise RuntimeOwnershipError('runtime_ownership_unverified')
        records = [item for item in payload['processes'] if item.get('name') != 'comfyui']
        payload['processes'] = [*records,record]
        write_json_atomic(launcher_state_path(config),payload)
    except (psutil.Error, OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise RuntimeOwnershipError('runtime_ownership_unverified') from exc


def remove_runtime_ownership(config: AppConfig, expected: RuntimeOwnership) -> None:
    payload = read_launcher_state(config)
    records = [item for item in payload['processes'] if item.get('name') == 'comfyui']
    if len(records) != 1 or ownership_from_record(config,records[0]) != expected:
        raise RuntimeOwnershipError('runtime_ownership_unverified')
    payload['processes'] = [item for item in payload['processes'] if item.get('name') != 'comfyui']
    try:
        write_json_atomic(launcher_state_path(config),payload)
    except OSError as exc:
        raise RuntimeOwnershipError('runtime_stop_failed') from exc
