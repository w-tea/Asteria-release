from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from threading import BoundedSemaphore
from typing import Callable, Iterable, Mapping
from urllib.parse import urlsplit

import httpcore
import httpx

from app.schemas.extensions import (
    ASTERIA_RESERVED_EXTENSION_LOCALHOST_PORTS,
    ExtensionManifest,
)


_MAX_REQUEST_BYTES = 256 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_SAFE_REQUEST_HEADERS = frozenset({"accept", "content-type"})
_SAFE_RESPONSE_HEADERS = frozenset({"content-type", "content-length", "etag", "last-modified"})


class StandardExtensionNetworkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StandardNetworkResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class _ValidatedNetworkTarget:
    url: str
    external_host: str | None = None
    pinned_addresses: tuple[str, ...] = ()


class _PinnedResponseStream(httpx.SyncByteStream):
    def __init__(self, stream: Iterable[bytes]) -> None:
        self._stream = stream

    def __iter__(self):
        yield from self._stream

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class _PinnedNetworkBackend(httpcore.NetworkBackend):
    def __init__(
        self,
        *,
        expected_host: str,
        pinned_addresses: tuple[str, ...],
        backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        if not pinned_addresses:
            raise ValueError("pinned_addresses must not be empty")
        self._expected_host = expected_host.casefold()
        self._pinned_addresses = pinned_addresses
        self._backend = backend or httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        if host.casefold() != self._expected_host:
            raise httpcore.ConnectError("Unexpected host for pinned extension request")
        last_error: httpcore.ConnectError | httpcore.ConnectTimeout | None = None
        for address in self._pinned_addresses:
            try:
                return self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        del path, timeout, socket_options
        raise httpcore.ConnectError("Unix sockets are unavailable to standard extensions")

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class PinnedHTTPTransport(httpx.BaseTransport):
    """Connect to validated IPs while preserving the original HTTPS identity."""

    def __init__(
        self,
        *,
        expected_host: str,
        pinned_addresses: tuple[str, ...],
        network_backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        backend = _PinnedNetworkBackend(
            expected_host=expected_host,
            pinned_addresses=pinned_addresses,
            backend=network_backend,
        )
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpcore.default_ssl_context(),
            max_connections=1,
            max_keepalive_connections=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=backend,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if not isinstance(request.stream, httpx.SyncByteStream):
            raise TypeError("PinnedHTTPTransport requires a synchronous request stream")
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = self._pool.handle_request(core_request)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_PinnedResponseStream(response.stream),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._pool.close()


class StandardExtensionNetworkBroker:
    def __init__(
        self,
        *,
        resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
        transport: httpx.BaseTransport | None = None,
        external_transport_factory: Callable[[str, tuple[str, ...]], httpx.BaseTransport] = (
            lambda host, addresses: PinnedHTTPTransport(
                expected_host=host,
                pinned_addresses=addresses,
            )
        ),
        max_concurrent_requests: int = 4,
    ) -> None:
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        self.resolver = resolver
        self.transport = transport
        self.external_transport_factory = external_transport_factory
        self._request_slots = BoundedSemaphore(max_concurrent_requests)

    def fetch(
        self,
        manifest: ExtensionManifest,
        *,
        url: str,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> StandardNetworkResponse:
        if not self._request_slots.acquire(blocking=False):
            raise StandardExtensionNetworkError("extension_standard_network_busy")
        try:
            return self._fetch(manifest, url=url, method=method, headers=headers, body=body)
        finally:
            self._request_slots.release()

    def _fetch(
        self,
        manifest: ExtensionManifest,
        *,
        url: str,
        method: str,
        headers: Mapping[str, str] | None,
        body: bytes,
    ) -> StandardNetworkResponse:
        if manifest.execution_level != "standard":
            raise StandardExtensionNetworkError("extension_standard_network_denied")
        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST"} or len(body) > _MAX_REQUEST_BYTES:
            raise StandardExtensionNetworkError("extension_standard_network_request_invalid")
        target = self._validate_target(manifest, url)
        request_headers = {
            key: value
            for key, value in (headers or {}).items()
            if key.casefold() in _SAFE_REQUEST_HEADERS
            and isinstance(value, str)
            and "\r" not in value
            and "\n" not in value
            and len(value) <= 512
        }
        if len(request_headers) != len(headers or {}):
            raise StandardExtensionNetworkError("extension_standard_network_request_invalid")
        request_transport = self.transport
        if request_transport is None and target.external_host is not None:
            request_transport = self.external_transport_factory(
                target.external_host,
                target.pinned_addresses,
            )
        try:
            with httpx.Client(
                transport=request_transport,
                follow_redirects=False,
                timeout=httpx.Timeout(30.0, connect=10.0),
                trust_env=False,
            ) as client:
                with client.stream(
                    normalized_method,
                    target.url,
                    headers=request_headers,
                    content=body or None,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise StandardExtensionNetworkError("extension_standard_network_redirect_denied")
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > _MAX_RESPONSE_BYTES:
                            raise StandardExtensionNetworkError("extension_standard_network_response_too_large")
                        chunks.append(chunk)
                    response_headers = {
                        key: value
                        for key, value in response.headers.items()
                        if key.casefold() in _SAFE_RESPONSE_HEADERS
                    }
                    return StandardNetworkResponse(
                        status_code=response.status_code,
                        headers=response_headers,
                        body=b"".join(chunks),
                    )
        except StandardExtensionNetworkError:
            raise
        except (
            httpx.HTTPError,
            httpcore.TimeoutException,
            httpcore.NetworkError,
            httpcore.ProtocolError,
            httpcore.UnsupportedProtocol,
            OSError,
            ValueError,
        ) as exc:
            raise StandardExtensionNetworkError("extension_standard_network_failed") from exc

    def _validate_target(self, manifest: ExtensionManifest, url: str) -> _ValidatedNetworkTarget:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise StandardExtensionNetworkError("extension_standard_network_target_invalid") from exc
        if parsed.username or parsed.password or parsed.query and len(parsed.query) > 2048 or parsed.fragment:
            raise StandardExtensionNetworkError("extension_standard_network_target_invalid")
        host = (parsed.hostname or "").casefold()
        effective_port = port or (443 if parsed.scheme == "https" else 80)
        origin = f"{parsed.scheme}://{host}"
        if port is not None and not (parsed.scheme == "https" and port == 443):
            origin = f"{origin}:{port}"
        declared_external = {value.casefold() for value in manifest.permissions.network_hosts}
        declared_local = {(item.host.casefold(), item.port) for item in manifest.permissions.localhost}
        external_host: str | None = None
        pinned_addresses: tuple[str, ...] = ()
        if parsed.scheme == "https" and origin.casefold() in declared_external:
            external_host = host
            pinned_addresses = self._resolve_global_addresses(host, effective_port)
        elif host in {"localhost", "127.0.0.1", "::1"} and (host, effective_port) in declared_local:
            if parsed.scheme not in {"http", "https"}:
                raise StandardExtensionNetworkError("extension_standard_network_target_invalid")
            if effective_port in ASTERIA_RESERVED_EXTENSION_LOCALHOST_PORTS:
                raise StandardExtensionNetworkError("extension_standard_network_reserved_target")
        else:
            raise StandardExtensionNetworkError("extension_standard_network_denied")
        if not parsed.path.startswith("/") or len(url) > 4096:
            raise StandardExtensionNetworkError("extension_standard_network_target_invalid")
        return _ValidatedNetworkTarget(
            url=url,
            external_host=external_host,
            pinned_addresses=pinned_addresses,
        )

    def _resolve_global_addresses(self, host: str, port: int) -> tuple[str, ...]:
        try:
            results = self.resolver(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise StandardExtensionNetworkError("extension_standard_network_failed") from exc
        if not results:
            raise StandardExtensionNetworkError("extension_standard_network_failed")
        addresses: list[str] = []
        for result in results:
            address = result[4][0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise StandardExtensionNetworkError("extension_standard_network_failed") from exc
            if not ip.is_global:
                raise StandardExtensionNetworkError("extension_standard_network_private_target")
            normalized = str(ip)
            if normalized not in addresses and len(addresses) < 8:
                addresses.append(normalized)
        if not addresses:
            raise StandardExtensionNetworkError("extension_standard_network_failed")
        return tuple(addresses)
