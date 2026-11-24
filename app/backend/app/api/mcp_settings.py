"""Local settings for the opt-in MCP connection; no token in API responses."""
import json
import os
import sys

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from app.services.mcp_access import LocalMcpAccess, LocalMcpStatus
from app.services.release_layout import launcher_backend_root
from app.services.windows_secrets import SecretDecryptionError, SecretUnavailableError


router = APIRouter(prefix='/api/mcp', tags=['mcp'])


class LocalMcpConnection(BaseModel):
    codex_toml: str
    stdio_json: str


@router.get('/connection', response_model=LocalMcpConnection)
def get_mcp_connection(request: Request, response: Response):
    config = request.app.state.config
    launcher_state = os.environ.get('ASTERIA_LAUNCHER_STATE_PATH') or str(
        config.root_dir / 'temp' / 'launcher' / 'launcher-state.json'
    )
    server = request.scope.get('server')
    port = server[1] if server and server[1] else 8000
    command = sys.executable
    # The client's working directory must not shadow the installed app package.
    args = ['-B', '-P', '-m', 'app.mcp.stdio_bridge', '--endpoint', f'http://127.0.0.1:{port}/mcp/',
            '--secret-dir', str(config.secret_dir), '--app-root', str(config.root_dir),
            '--launcher-state', launcher_state]
    python_path = str(launcher_backend_root(config.root_dir, config.target_mode))
    entry = {'command': command, 'args': args, 'env': {'PYTHONPATH': python_path}}
    response.headers['Cache-Control'] = 'no-store'
    return LocalMcpConnection(
        codex_toml='\n'.join((
            '[mcp_servers.asteria]',
            f'command = {json.dumps(command, ensure_ascii=False)}',
            f'args = {json.dumps(args, ensure_ascii=False)}',
            'env = { PYTHONPATH = ' + json.dumps(python_path, ensure_ascii=False) + ' }',
            '',
        )),
        stdio_json=json.dumps({'mcpServers': {'asteria': entry}}, indent=2, ensure_ascii=False) + '\n',
    )


@router.get('/settings', response_model=LocalMcpStatus)
def get_mcp_settings(request: Request):
    try:
        return LocalMcpAccess(request.app.state.config.secret_dir).status()
    except (OSError, UnicodeError, SecretDecryptionError, SecretUnavailableError) as exc:
        raise HTTPException(503, detail={'code': 'mcp_credential_unavailable'}) from exc


@router.put('/settings', response_model=LocalMcpStatus)
def set_mcp_settings(settings: LocalMcpStatus, request: Request):
    try:
        return LocalMcpAccess(request.app.state.config.secret_dir).set_enabled(settings.enabled)
    except (OSError, UnicodeError, SecretDecryptionError, SecretUnavailableError) as exc:
        raise HTTPException(503, detail={'code': 'mcp_credential_unavailable'}) from exc
