"""A stdio client host's lightweight relay to an already running Asteria.

No core dispatcher, application import, database or model runtime lives here.
The parent MCP client owns this process and closing its stdin ends the relay.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

import httpx2
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent

from app.services.mcp_access import LocalMcpAccess
from app.mcp.bridge_ownership import registered_bridge


def _transport_failure(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return all(_transport_failure(child) for child in error.exceptions)
    return isinstance(error, (httpx2.HTTPError, OSError))


def local_endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path.rstrip('/') != '/mcp'):
        raise ValueError('This local bridge connects only to the running Asteria loopback /mcp endpoint.')
    return value.rstrip('/') + '/'


def create_bridge_server(endpoint: str, access: LocalMcpAccess) -> Server:
    endpoint = local_endpoint(endpoint)
    schemas = {}

    async def forward(method: str, *args):
        token = access.token()
        if token is None:
            raise MCPError(-32000, 'Local MCP is disabled. Enable the MCP connection in Asteria.')
        # No automatic tool retry. The caller decides whether to resend the
        # same request_id; transport loss must not create a fresh generation.
        try:
            async with httpx2.AsyncClient(
                headers={'Authorization': f'Bearer {token}'},
                timeout=httpx2.Timeout(30, read=300), trust_env=False,
            ) as http:
                async with Client(streamable_http_client(endpoint, http_client=http)) as core:
                    return await getattr(core, method)(*args)
        except Exception as exc:
            # SDK task groups can wrap a refused socket in ExceptionGroup.
            # Convert only transport failures, not unrelated programming errors.
            if not _transport_failure(exc):
                raise
            raise MCPError(-32000, 'Asteria is unavailable. Start Asteria and check that local MCP is enabled.') from None

    async def list_tools(_context, _params):
        result = await forward('list_tools')
        schemas.clear()
        schemas.update({tool.name: tool.input_schema for tool in result.tools})
        return result

    async def call_tool(_context, params):
        try:
            return await forward('call_tool', params.name, params.arguments)
        except MCPError as exc:
            data = {'code': 'mcp_connection_failed', 'message': str(exc)}
            return CallToolResult(is_error=True, structured_content=data,
                                  content=[TextContent(type='text', text=json.dumps(data))])

    return Server('Asteria local bridge', version='1.2.3', on_list_tools=list_tools,
                  on_call_tool=call_tool, get_tool_input_schema=schemas.get)


async def serve(endpoint: str, secret_dir: Path) -> None:
    server = create_bridge_server(endpoint, LocalMcpAccess(secret_dir))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description='Connect an MCP client to an already running Asteria.')
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000/mcp/')
    parser.add_argument('--secret-dir', type=Path, required=True)
    parser.add_argument('--app-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--launcher-state', type=Path)
    args = parser.parse_args()
    try:
        state_path = args.launcher_state or args.app_root / 'temp/launcher/launcher-state.json'
        with registered_bridge(args.app_root, state_path):
            asyncio.run(serve(args.endpoint, args.secret_dir))
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
