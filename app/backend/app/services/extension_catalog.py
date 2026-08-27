from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import RLock
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.schemas.extension_catalog import (
    ASTERIA_EXTENSION_CATALOG_SCHEMA_ID,
    CatalogCompatibilityStatus,
    ExtensionCatalogDocument,
    ExtensionCatalogEntry,
    ExtensionCatalogRuntimeCompatibility,
)
from app.schemas.extensions import compare_semver, validate_semver
from app.services.atomic_json import write_json_atomic
from app.services.extension_sources import (
    ExtensionSourceDescriptor,
    SourceKind,
    SourceTrust,
)


EXTENSION_CATALOG_CACHE_SCHEMA = "asteria.extension.catalog.cache.v2"
CATALOG_MAX_BYTES = 4 * 1024 * 1024
CatalogResultSource = Literal["network", "cache", "empty"]


class ExtensionCatalogError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CatalogFetchResponse:
    status: int
    body: bytes
    etag: str | None
    final_url: str


@dataclass(frozen=True, slots=True)
class ExtensionCatalogResult:
    entries: tuple[ExtensionCatalogEntry, ...]
    source: CatalogResultSource
    stale: bool
    cache_age_seconds: int | None
    error_code: str | None
    rejected_count: int = 0


class _CatalogCacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[EXTENSION_CATALOG_CACHE_SCHEMA] = EXTENSION_CATALOG_CACHE_SCHEMA
    catalog: ExtensionCatalogDocument
    etag: str | None
    fetched_at: datetime
    checked_at: datetime
    rejected_count: int

    @field_validator("fetched_at", "checked_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("catalog cache timestamps must include a timezone")
        return value


CatalogFetcher = Callable[[str, str | None], CatalogFetchResponse]


class ExtensionCatalogService:
    def __init__(
        self,
        cache_root: Path,
        *,
        catalog_url: str,
        asteria_version: str,
        fetcher: CatalogFetcher | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.cache_path = cache_root / "catalog-cache-v1.json"
        self.catalog_url = _validated_https_url(catalog_url)
        self.asteria_version = validate_semver(asteria_version)
        self.fetcher = fetcher or _fetch_catalog
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()

    def refresh(self) -> ExtensionCatalogResult:
        with self._lock:
            cached = self._read_cache()
            try:
                response = self.fetcher(self.catalog_url, cached.etag if cached is not None else None)
                _validated_https_url(response.final_url)
                if response.status == 304 and cached is None:
                    response = self.fetcher(self.catalog_url, None)
                    _validated_https_url(response.final_url)
                if response.status == 304:
                    if cached is None:
                        raise ExtensionCatalogError("extension_catalog_invalid")
                    now = self._now()
                    checked = cached.model_copy(update={"checked_at": now})
                    self._write_cache(checked)
                    return self._result(
                        checked,
                        source="cache",
                        stale=False,
                        error_code=None,
                        now=now,
                    )
                if response.status != 200:
                    raise ExtensionCatalogError("extension_catalog_offline")
                if len(response.body) > CATALOG_MAX_BYTES:
                    raise ExtensionCatalogError("extension_catalog_invalid")
                document, rejected_count = self._parse_document(response.body)
                now = self._now()
                record = _CatalogCacheRecord(
                    catalog=document,
                    etag=response.etag,
                    fetched_at=now,
                    checked_at=now,
                    rejected_count=rejected_count,
                )
                self._write_cache(record)
                return self._result(
                    record,
                    source="network",
                    stale=False,
                    error_code=(
                        "extension_catalog_entries_rejected"
                        if rejected_count
                        else None
                    ),
                    now=now,
                )
            except ExtensionCatalogError as exc:
                return self._fallback(cached, exc.code)
            except (OSError, TimeoutError, HTTPError, URLError, ValueError):
                return self._fallback(cached, "extension_catalog_offline")

    def snapshot(self) -> ExtensionCatalogResult:
        with self._lock:
            cached = self._read_cache()
            if cached is None:
                return ExtensionCatalogResult(
                    entries=(),
                    source="empty",
                    stale=True,
                    cache_age_seconds=None,
                    error_code=None,
                )
            return self._result(cached, source="cache", stale=False, error_code=None)

    def source_for(self, extension_id: str, version: str) -> ExtensionSourceDescriptor:
        _entry, source = self.resolve_source(extension_id, version)
        return source

    def runtime_compatibility(
        self,
        entry: ExtensionCatalogEntry,
    ) -> ExtensionCatalogRuntimeCompatibility:
        status: CatalogCompatibilityStatus
        if compare_semver(self.asteria_version, entry.asteria.minimum) < 0:
            status = "requires_newer_asteria"
        elif compare_semver(
            self.asteria_version,
            entry.asteria.maximum_exclusive,
        ) >= 0:
            status = "requires_older_asteria"
        else:
            status = "compatible"
        return ExtensionCatalogRuntimeCompatibility(
            status=status,
            current=self.asteria_version,
            minimum=entry.asteria.minimum,
            maximum_exclusive=entry.asteria.maximum_exclusive,
        )

    def resolve_source(
        self,
        extension_id: str,
        version: str,
    ) -> tuple[ExtensionCatalogEntry, ExtensionSourceDescriptor]:
        snapshot = self.snapshot()
        entry = next(
            (
                item
                for item in snapshot.entries
                if item.id.casefold() == extension_id.casefold() and item.version == version
            ),
            None,
        )
        if entry is None:
            raise ExtensionCatalogError("extension_catalog_entry_not_found")
        if self.runtime_compatibility(entry).status != "compatible":
            raise ExtensionCatalogError("extension_catalog_incompatible")
        url = str(entry.source.url)
        source = ExtensionSourceDescriptor(
            kind=SourceKind(entry.source.kind),
            reference=url,
            expected_sha256=entry.source.sha256,
            expected_size_bytes=entry.source.size_bytes,
            trust=SourceTrust.CATALOG_VERIFIED,
        )
        return entry, source

    def _parse_document(self, body: bytes) -> tuple[ExtensionCatalogDocument, int]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ExtensionCatalogError("extension_catalog_invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
            raise ExtensionCatalogError("extension_catalog_invalid")
        schema_version = payload.get("schema_version")
        if schema_version != ASTERIA_EXTENSION_CATALOG_SCHEMA_ID:
            code = (
                "extension_catalog_version_unsupported"
                if isinstance(schema_version, str)
                and schema_version.startswith("asteria.extension.catalog.")
                else "extension_catalog_invalid"
            )
            raise ExtensionCatalogError(code)
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ExtensionCatalogError("extension_catalog_invalid")
        entries: list[ExtensionCatalogEntry] = []
        rejected_count = 0
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or not {
                "execution_level",
                "permissions",
            }.issubset(raw_entry):
                rejected_count += 1
                continue
            try:
                entries.append(
                    ExtensionCatalogEntry.model_validate_json(
                        json.dumps(raw_entry, ensure_ascii=False)
                    )
                )
            except ValidationError:
                rejected_count += 1
        if raw_entries and not entries:
            raise ExtensionCatalogError("extension_catalog_invalid")
        ids = [entry.id.casefold() for entry in entries]
        if len(ids) != len(set(ids)):
            raise ExtensionCatalogError("extension_catalog_duplicate_id")
        entries.sort(key=lambda entry: (entry.id.casefold(), entry.id))
        try:
            return (
                ExtensionCatalogDocument(
                    schema_version=ASTERIA_EXTENSION_CATALOG_SCHEMA_ID,
                    entries=tuple(entries),
                ),
                rejected_count,
            )
        except ValidationError as exc:
            raise ExtensionCatalogError("extension_catalog_invalid") from exc

    def _read_cache(self) -> _CatalogCacheRecord | None:
        try:
            return _CatalogCacheRecord.model_validate_json(
                self.cache_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValidationError):
            return None

    def _write_cache(self, record: _CatalogCacheRecord) -> None:
        try:
            write_json_atomic(self.cache_path, record.model_dump(mode="json"))
        except OSError as exc:
            raise ExtensionCatalogError("extension_catalog_cache_write_failed") from exc

    def _fallback(
        self,
        cached: _CatalogCacheRecord | None,
        error_code: str,
    ) -> ExtensionCatalogResult:
        if cached is None:
            return ExtensionCatalogResult(
                entries=(),
                source="empty",
                stale=True,
                cache_age_seconds=None,
                error_code=error_code,
            )
        return self._result(cached, source="cache", stale=True, error_code=error_code)

    def _result(
        self,
        record: _CatalogCacheRecord,
        *,
        source: CatalogResultSource,
        stale: bool,
        error_code: str | None,
        now: datetime | None = None,
    ) -> ExtensionCatalogResult:
        return ExtensionCatalogResult(
            entries=record.catalog.entries,
            source=source,
            stale=stale,
            cache_age_seconds=max(
                0,
                int(((now or self._now()) - record.checked_at).total_seconds()),
            ),
            error_code=error_code,
            rejected_count=record.rejected_count,
        )

    def _now(self) -> datetime:
        value = self.clock()
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _validated_https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme.casefold() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("extension catalog URL must be credential-free HTTPS")
    return value


def _fetch_catalog(url: str, etag: str | None) -> CatalogFetchResponse:
    headers = {"Accept": "application/json"}
    if etag is not None:
        headers["If-None-Match"] = etag
    request = Request(url, headers=headers)
    try:
        with build_opener(_CatalogRedirectHandler()).open(request, timeout=15) as response:
            body = response.read(CATALOG_MAX_BYTES + 1)
            return CatalogFetchResponse(
                status=int(getattr(response, "status", 200)),
                body=body,
                etag=response.headers.get("ETag"),
                final_url=response.geturl(),
            )
    except HTTPError as exc:
        if exc.code == 304:
            return CatalogFetchResponse(
                status=304,
                body=b"",
                etag=exc.headers.get("ETag") if exc.headers is not None else etag,
                final_url=exc.geturl(),
            )
        raise


class _CatalogRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validated_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)
