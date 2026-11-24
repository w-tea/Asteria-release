"""Register only this relay with the existing Windows launcher's stop path."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess


@contextmanager
def registered_bridge(app_root: Path, state_path: Path):
    with registered_connection(app_root, state_path, kind='mcp-bridge') as path:
        yield path


@contextmanager
def registered_connection(app_root: Path, state_path: Path, *, kind: str):
    directory = {'mcp-bridge': 'mcp-bridges', 'lan-gateway': 'lan-gateways'}.get(kind)
    if directory is None:
        raise ValueError('Unknown connection process kind.')
    script = app_root / 'scripts/windows/Register-AsteriaMcpBridge.ps1'
    if not script.is_file() or not state_path.is_file():
        raise RuntimeError('Start Asteria before connecting.')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', str(script), '-AppRoot', str(app_root), '-StatePath', str(state_path), '-ProcessId', str(os.getpid()), '-Kind', kind],
        capture_output=True, text=True, encoding='utf-8-sig', timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Asteria could not register the connection. Start Asteria and reconnect.')
    path = Path(json.loads(result.stdout)['path'])
    if path.parent.resolve() != (state_path.parent / directory).resolve():
        raise RuntimeError('Unexpected connection ownership record location.')
    try:
        yield path
    finally:
        # A launcher stop may already have removed this exact, unique record.
        path.unlink(missing_ok=True)
