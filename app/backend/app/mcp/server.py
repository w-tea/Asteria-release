"""Official MCP SDK adapter over the single core action catalog."""
from __future__ import annotations

from copy import deepcopy
import json
import secrets

import anyio
from fastapi import FastAPI, HTTPException
from mcp.server import Server
from mcp.types import CallToolResult, ImageContent, ListToolsResult, TextContent, Tool, ToolAnnotations
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.applications import Starlette
from starlette.routing import Mount

from app.schemas.action_contracts import ActionCall
from app.services.action_dispatch import ActionDispatcher, ActionError
from app.services.mcp_access import LocalMcpAccess
from app.services.windows_secrets import SecretDecryptionError, SecretUnavailableError


def action_tool(item) -> Tool:
    # Keep the exact core model under arguments; promote $defs to the document
    # root so its existing absolute JSON-Schema references still resolve.
    arguments = deepcopy(item.input_schema)
    definitions = arguments.pop('$defs', {})
    schema = ActionCall.model_json_schema()
    schema['properties']['arguments'] = arguments
    schema['$defs'] = definitions
    if not item.read_only:
        schema['required'] = ['arguments', 'request_id']
    return Tool(name=item.name, title=item.title, description=item.description,
                input_schema=schema,
                annotations=ToolAnnotations(read_only_hint=item.read_only))


def tool_result(data: dict, *, error: bool = False) -> CallToolResult:
    # Move bytes to the protocol's image block, not a local URL or a huge text
    # string. The metadata and original hash remain in the structured result.
    image = data.get('data', {}).get('image') if not error else None
    content = []
    if image is not None:
        data = {**data, 'data': {key: value for key, value in data['data'].items() if key != 'image'}}
        content.append(ImageContent(type='image', data=image['data_base64'], mime_type=image['mime_type']))
    return CallToolResult(content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False)), *content],
                          structured_content=data, is_error=error)


def create_action_server(dispatcher: ActionDispatcher) -> Server:
    tools = [action_tool(item) for item in dispatcher.catalog()]
    schemas = {tool.name: tool.input_schema for tool in tools}

    async def list_tools(_context, _params):
        return ListToolsResult(tools=tools)

    async def call_tool(_context, params):
        try:
            call = ActionCall.model_validate(params.arguments or {})
            # Existing core services are synchronous. Never block the ASGI loop
            # with SQLite/model-catalog IO, and do not cancel an accepted job.
            result = await anyio.to_thread.run_sync(dispatcher.invoke, params.name, call)
            return tool_result(result.model_dump(mode='json'))
        except ActionError as exc:
            return tool_result({'status_code': exc.status_code, 'detail': exc.detail}, error=True)
        except HTTPException as exc:
            return tool_result({'status_code': exc.status_code, 'detail': exc.detail}, error=True)
        except ValidationError as exc:
            errors = json.loads(exc.json(include_url=False))
            return tool_result({'status_code': 422, 'detail': errors}, error=True)

    return Server('Asteria', version='1.2.3', on_list_tools=list_tools, on_call_tool=call_tool,
                  get_tool_input_schema=schemas.get,
                  instructions='Asteria must already be running. Read installed models and generation fields '
                  'before preparing a draft. New requests are independent jobs. Reuse request_id only for '
                  'redelivery of the same operation and arguments. Tool errors are Asteria core errors, '
                  'not successful generations. Image input uses the authenticated binary upload endpoint '
                  '/mcp/uploads/images (filename, role, request_id query parameters), never base64 tool arguments. '
                  'Use its returned asset_id/path for image reading or generation drafts. '
                  'For Video Studio inputs, POST the file bytes to /mcp/uploads/media (filename and request_id '
                  'query parameters); put the returned artifact_id in the appropriate VideoRequest field. '
                  'Training and model merging are not AI operations.')


class LocalMcpAuthMiddleware:
    """Credential gate only for the loopback MCP route, not the ordinary GUI."""
    def __init__(self, app, *, access: LocalMcpAccess):
        self.app, self.access = app, access

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        try:
            token = await anyio.to_thread.run_sync(self.access.token)
        except (OSError, UnicodeError, SecretDecryptionError, SecretUnavailableError):
            await JSONResponse({'detail': {'code': 'mcp_credential_unavailable'}}, status_code=503)(scope, receive, send)
            return
        if token is None:
            await JSONResponse({'detail': {'code': 'mcp_disabled'}}, status_code=404)(scope, receive, send)
            return
        authorization = Headers(scope=scope).getlist('authorization')
        expected = ('Bearer ' + token).encode('ascii')
        if len(authorization) != 1 or not secrets.compare_digest(authorization[0].encode('latin-1'), expected):
            await JSONResponse({'detail': {'code': 'mcp_authentication_required'}}, status_code=401,
                               headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
            return
        await self.app(scope, receive, send)


def mount_mcp(app, config):
    from app.api.action_assets import router as upload_router

    server = create_action_server(ActionDispatcher(config))
    protocol_app = server.streamable_http_app(streamable_http_path='/', json_response=True, stateless_http=True)
    uploads = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    uploads.state.config = config
    uploads.include_router(upload_router)
    transport = Starlette(routes=[Mount('/uploads', uploads), Mount('/', protocol_app)])
    app.state.mcp_server = server
    app.mount('/mcp', LocalMcpAuthMiddleware(transport, access=LocalMcpAccess(config.secret_dir)))
