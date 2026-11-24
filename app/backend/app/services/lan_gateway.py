"""Authenticated same-origin transport for the existing full Asteria GUI.

This app does not dispatch AI tools or introduce another job queue. The core
and ComfyUI retain their loopback listeners. Launcher integration owns when
this app may listen on a selected LAN interface.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path
import secrets
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from app.services.lan_access import LanAccessError, LanAccessRepository


COOKIE_NAME = 'asteria_lan_device'
_HOP_HEADERS = frozenset({'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
                          'te', 'trailer', 'transfer-encoding', 'upgrade'})
_UPSTREAM_ORIGIN = 'http://127.0.0.1:3000'


def _error(status: int, code: str):
    return JSONResponse({'detail': {'code': code}}, status_code=status,
                        headers={'cache-control': 'no-store'})


def create_lan_gateway(access: LanAccessRepository, *, public_origin: str,
                       transport: httpx.AsyncBaseTransport | None = None,
                       frontend_origin: str = _UPSTREAM_ORIGIN,
                       backend_origin: str = 'http://127.0.0.1:8000') -> Starlette:
    public = urlsplit(public_origin)
    if public.scheme not in {'http', 'https'} or not public.netloc or public.path or public.query or public.fragment or public.username:
        raise ValueError('lan_origin_invalid')
    for target in [frontend_origin, backend_origin]:
        parts = urlsplit(target)
        if parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost', '::1'} or parts.path or parts.query or parts.fragment or parts.username:
            raise ValueError('lan_upstream_must_be_loopback')

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(None, connect=10)) as client:
            app.state.client = client
            yield

    def boundary(connection, *, websocket=False):
        hosts = connection.headers.getlist('host')
        if len(hosts) != 1 or hosts[0].lower() != public.netloc.lower():
            return 400, 'lan_host_not_allowed'
        origins = connection.headers.getlist('origin')
        if origins:
            if len(origins) != 1 or origins[0] != public_origin:
                return 403, 'lan_origin_not_allowed'
        else:
            referers = connection.headers.getlist('referer')
            if referers:
                ref = urlsplit(referers[0])
                if len(referers) != 1 or f'{ref.scheme}://{ref.netloc}' != public_origin:
                    return 403, 'lan_origin_not_allowed'
            elif websocket or connection.method not in {'GET', 'HEAD'}:
                return 403, 'lan_origin_required'
        if connection.headers.get('sec-fetch-site') == 'cross-site':
            return 403, 'lan_origin_not_allowed'
        return None

    def target_url(connection):
        base = backend_origin if connection.url.path.startswith('/api/') else frontend_origin
        raw = connection.scope.get('raw_path', connection.url.path.encode('utf-8'))
        query = connection.scope.get('query_string', b'')
        return httpx.URL(base).copy_with(raw_path=raw + (b'?' + query if query else b''))

    def upstream_headers(connection):
        connection_tokens = {part.strip().lower() for part in connection.headers.get('connection', '').split(',')}
        excluded = _HOP_HEADERS | connection_tokens | {'host', 'origin', 'referer', 'cookie', 'forwarded'}
        headers = [(key, value) for key, value in connection.headers.raw if key.decode('latin-1').lower() not in excluded
                   and not key.decode('latin-1').lower().startswith(('x-forwarded-', 'x-asteria-lan', 'sec-websocket-'))]
        return headers + [(b'origin', _UPSTREAM_ORIGIN.encode()), (b'x-asteria-lan', b'1')]

    async def authenticated(connection):
        return await asyncio.to_thread(access.authenticate, connection.cookies.get(COOKIE_NAME))

    async def connection_page(request: Request):
        if rejected := boundary(request):
            return _error(*rejected)
        nonce = secrets.token_urlsafe(24)
        markup = (Path(__file__).parent.parent / 'web/lan_connect.html').read_text('utf-8')
        return HTMLResponse(markup.replace('__NONCE__', nonce), headers={
            'cache-control': 'no-store', 'referrer-policy': 'no-referrer',
            'content-security-policy': f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"})

    async def session_status(request: Request):
        if rejected := boundary(request):
            return _error(*rejected)
        return JSONResponse({'registered': bool(await authenticated(request))}, headers={'cache-control': 'no-store'})

    async def pairing(request: Request):
        if rejected := boundary(request):
            return _error(*rejected)
        try:
            value = await request.json()
            if not isinstance(value, dict) or not isinstance(value.get('token'), str) or not isinstance(value.get('name'), str):
                return _error(422, 'lan_pairing_input_invalid')
            credential = await asyncio.to_thread(access.register, value['token'], value['name'])
        except (ValueError, LanAccessError) as error:
            code = str(error) if isinstance(error, LanAccessError) else 'lan_pairing_input_invalid'
            return _error(403, code)
        response = JSONResponse({'device': asdict(credential.device)}, headers={'cache-control': 'no-store'})
        response.set_cookie(COOKIE_NAME, credential.token, max_age=31536000, httponly=True,
                            secure=public.scheme == 'https', samesite='strict', path='/')
        return response

    async def proxy(request: Request):
        if rejected := boundary(request):
            return _error(*rejected)
        if not await asyncio.to_thread(access.enabled):
            return _error(503, 'lan_disabled')
        if not await authenticated(request):
            if request.method == 'GET' and not request.url.path.startswith('/api/') and 'text/html' in request.headers.get('accept', ''):
                return RedirectResponse('/_asteria/lan/connect', status_code=303, headers={'cache-control': 'no-store'})
            return _error(401, 'lan_device_required')
        client = request.app.state.client
        outgoing = client.build_request(request.method, target_url(request),
                                        headers=upstream_headers(request), content=request.stream())
        try:
            response = await client.send(outgoing, stream=True)
        except httpx.HTTPError:
            return _error(502, 'lan_core_unavailable')

        async def body():
            pending = None
            chunks = response.aiter_raw().__aiter__()
            try:
                while True:
                    pending = asyncio.create_task(anext(chunks))
                    while not (await asyncio.wait({pending}, timeout=1))[0]:
                        if not await authenticated(request):
                            return
                    if not await authenticated(request):
                        return
                    try:
                        yield pending.result()
                    except StopAsyncIteration:
                        return
                    pending = None
            finally:
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                await response.aclose()

        headers = [(key, value) for key, value in response.headers.raw
                   if key.decode('latin-1').lower() not in _HOP_HEADERS]
        result = StreamingResponse(body(), status_code=response.status_code)
        result.raw_headers = headers
        return result

    async def websocket_proxy(socket: WebSocket):
        if boundary(socket, websocket=True) or not await authenticated(socket):
            await socket.close(code=1008)
            return
        url = str(target_url(socket)).replace('http:', 'ws:', 1)
        headers = [(key, value) for key, value in upstream_headers(socket) if key != b'origin']
        try:
            async with connect(url, origin=_UPSTREAM_ORIGIN, additional_headers=[(k.decode(), v.decode('latin-1')) for k, v in headers],
                               subprotocols=socket.scope.get('subprotocols') or None, proxy=None, max_size=None) as upstream:
                await socket.accept(subprotocol=upstream.subprotocol)

                async def to_core():
                    while True:
                        item = await socket.receive()
                        if item['type'] == 'websocket.disconnect':
                            return
                        await upstream.send(item['bytes'] if item.get('bytes') is not None else item['text'])

                async def to_browser():
                    async for item in upstream:
                        if isinstance(item, bytes):
                            await socket.send_bytes(item)
                        else:
                            await socket.send_text(item)

                async def revoked():
                    while await authenticated(socket):
                        await asyncio.sleep(1)
                    await socket.close(code=1008)

                tasks = [asyncio.create_task(action()) for action in (to_core, to_browser, revoked)]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (OSError, WebSocketException, WebSocketDisconnect):
            pass
        finally:
            with suppress(RuntimeError, WebSocketDisconnect):
                await socket.close(code=1001)

    return Starlette(lifespan=lifespan, routes=[
        Route('/_asteria/lan/connect', connection_page, methods=['GET']),
        Route('/_asteria/lan/session', session_status, methods=['GET']),
        Route('/_asteria/lan/pair', pairing, methods=['POST']),
        Route('/{path:path}', proxy, methods=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']),
        WebSocketRoute('/{path:path}', websocket_proxy),
    ])
