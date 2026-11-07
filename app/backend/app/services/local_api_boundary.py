from __future__ import annotations

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


ASTERIA_UI_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")
_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def _authority(name: str) -> str:
    return f"[{name}]" if ":" in name else name


def _trusted_authorities(scope: Scope) -> set[str]:
    # Uvicorn supplies the listener address, not a value derived from Host
    # or X-Forwarded-Host. The normal launcher binds only to loopback.
    address, port = scope.get("server") or ("127.0.0.1", None)
    names = _LOOPBACK_NAMES if address in _LOOPBACK_NAMES else {address}
    authorities = {_authority(name) for name in names}
    if port is not None:
        authorities |= {f"{authority}:{port}" for authority in authorities}
    return authorities


def _referer_origin(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


class LocalApiBoundaryMiddleware:
    """Reject browser requests outside the local application's trust boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        hosts = headers.getlist("host")
        host = hosts[0].lower() if len(hosts) == 1 else ""
        if host not in _trusted_authorities(scope):
            await self._reject(scope, receive, send, 400, "local_api_host_not_allowed")
            return

        scheme = "https" if scope["scheme"] in {"https", "wss"} else "http"
        allowed_origins = {*ASTERIA_UI_ORIGINS, f"{scheme}://{host}"}
        origins = headers.getlist("origin")
        if origins:
            allowed = len(origins) == 1 and origins[0] in allowed_origins
        else:
            # Plain image and media loads may omit Origin. Check the referrer
            # and browser fetch metadata before allowing them to touch paths.
            referers = headers.getlist("referer")
            if referers:
                allowed = len(referers) == 1 and _referer_origin(referers[0]) in allowed_origins
            else:
                # Native local tools and browser address-bar navigation do
                # not send an Origin. This is not native-process authentication.
                allowed = headers.get("sec-fetch-site") in {None, "none", "same-origin"}
        if not allowed:
            await self._reject(scope, receive, send, 403, "local_api_origin_not_allowed")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, status: int, code: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"detail": {"code": code}}, status_code=status)(scope, receive, send)
