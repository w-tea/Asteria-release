from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal
import re

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.extensions import (
    AsteriaCompatibility,
    Capability,
    ExecutionLevel,
    ExtensionStandardPermissions,
    StrictSemVer,
    validate_extension_id,
    validate_semver,
)


ASTERIA_EXTENSION_CATALOG_SCHEMA_ID = "asteria.extension.catalog.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXTENSION_ID_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$"
_ASSET_PATH_PATTERN = (
    r"^(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.{1,2}(?:/|$))"
    r"(?!.*://)(?!.*[?#]).+$"
)
CatalogExtensionId = Annotated[str, Field(pattern=_EXTENSION_ID_PATTERN)]
CatalogAssetPath = Annotated[str, Field(json_schema_extra={"pattern": _ASSET_PATH_PATTERN})]
CatalogHttpsUrl = Annotated[
    AnyHttpUrl,
    Field(json_schema_extra={"pattern": r"^https://(?![^/?#]*@)"}),
]
CatalogCompatibilityStatus = Literal[
    "compatible",
    "requires_newer_asteria",
    "requires_older_asteria",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ExtensionCatalogCompatibility(AsteriaCompatibility):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        json_schema_extra={
            "$comment": (
                "Semantic validation requires maximum_exclusive to be greater than minimum; "
                "Asteria runtime validation enforces this rule in addition to JSON Schema."
            )
        },
    )


def _validate_relative_asset_path(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "://" in value
        or "?" in value
        or "#" in value
    ):
        raise ValueError("catalog asset path must be a safe relative URL")
    return posix.as_posix()


class ExtensionCatalogAuthor(_StrictFrozenModel):
    name: str = Field(min_length=1, max_length=160)
    url: CatalogHttpsUrl | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("author name must not have surrounding whitespace")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: AnyHttpUrl | None) -> AnyHttpUrl | None:
        if value is not None and (
            value.scheme != "https" or value.username is not None or value.password is not None
        ):
            raise ValueError("catalog author URL must be credential-free HTTPS")
        return value


class ExtensionCatalogLicense(_StrictFrozenModel):
    id: str = Field(min_length=1, max_length=120)
    url: CatalogHttpsUrl

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("license ID must not have surrounding whitespace")
        return value

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https" or value.username is not None or value.password is not None:
            raise ValueError("catalog license URL must be credential-free HTTPS")
        return value

class ExtensionCatalogSource(_StrictFrozenModel):
    kind: Literal["https_zip", "github_release_zip"]
    url: CatalogHttpsUrl
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=512 * 1024 * 1024)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("catalog source checksum must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_https_source(self) -> ExtensionCatalogSource:
        if self.url.scheme != "https" or self.url.username is not None or self.url.password is not None:
            raise ValueError("catalog source URL must be credential-free HTTPS")
        path = self.url.path or ""
        if not path.casefold().endswith(".zip"):
            raise ValueError("catalog source must be a direct ZIP URL")
        if self.kind == "github_release_zip":
            parts = tuple(part for part in path.split("/") if part)
            if (
                self.url.host != "github.com"
                or self.url.query is not None
                or self.url.fragment is not None
                or len(parts) != 6
                or parts[2:4] != ("releases", "download")
            ):
                raise ValueError(
                    "GitHub release source must be a direct github.com release ZIP URL"
                )
        return self


class ExtensionCatalogAssets(_StrictFrozenModel):
    icon: CatalogAssetPath | None = None
    screenshots: tuple[CatalogAssetPath, ...] = Field(default=(), max_length=8)

    @field_validator("icon")
    @classmethod
    def _validate_icon(cls, value: str | None) -> str | None:
        return _validate_relative_asset_path(value) if value is not None else None

    @field_validator("screenshots")
    @classmethod
    def _validate_screenshots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validate_relative_asset_path(item) for item in value)
        if len(set(item.casefold() for item in normalized)) != len(normalized):
            raise ValueError("catalog screenshot paths must be unique")
        return normalized


class ExtensionCatalogEntry(_StrictFrozenModel):
    id: CatalogExtensionId
    name: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=500)
    version: StrictSemVer
    asteria: ExtensionCatalogCompatibility
    execution_level: ExecutionLevel = "full_access"
    permissions: ExtensionStandardPermissions = Field(default_factory=ExtensionStandardPermissions)
    capabilities: tuple[Capability, ...] = Field(
        json_schema_extra={"uniqueItems": True},
    )
    author: ExtensionCatalogAuthor
    license: ExtensionCatalogLicense
    homepage: CatalogHttpsUrl
    review_status: Literal["official", "community", "unverified"]
    source: ExtensionCatalogSource
    assets: ExtensionCatalogAssets = ExtensionCatalogAssets()

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_extension_id(value)

    @field_validator("name", "summary")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("catalog text must not have surrounding whitespace")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return validate_semver(value)

    @field_validator("capabilities")
    @classmethod
    def _validate_capabilities(cls, value: tuple[Capability, ...]) -> tuple[Capability, ...]:
        if len(set(value)) != len(value):
            raise ValueError("catalog capabilities must be unique")
        return value

    @field_validator("homepage")
    @classmethod
    def _validate_homepage(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https" or value.username is not None or value.password is not None:
            raise ValueError("catalog homepage must be credential-free HTTPS")
        return value

    @model_validator(mode="after")
    def _validate_execution_level(self) -> ExtensionCatalogEntry:
        if self.execution_level == "standard" and self.capabilities:
            raise ValueError("standard catalog entry cannot declare full-access capabilities")
        if self.execution_level == "full_access" and not self.permissions.is_empty():
            raise ValueError("full-access catalog entry cannot declare standard permissions")
        return self


class ExtensionCatalogRuntimeCompatibility(_StrictFrozenModel):
    status: CatalogCompatibilityStatus
    current: StrictSemVer
    minimum: StrictSemVer
    maximum_exclusive: StrictSemVer

    @field_validator("current", "minimum", "maximum_exclusive")
    @classmethod
    def _validate_versions(cls, value: str) -> str:
        return validate_semver(value)


class ExtensionCatalogProjectedEntry(ExtensionCatalogEntry):
    runtime_compatibility: ExtensionCatalogRuntimeCompatibility


class ExtensionCatalogDocument(_StrictFrozenModel):
    schema_version: Literal[ASTERIA_EXTENSION_CATALOG_SCHEMA_ID]
    entries: tuple[ExtensionCatalogEntry, ...]

    @model_validator(mode="after")
    def _validate_unique_entries(self) -> ExtensionCatalogDocument:
        ids = [entry.id.casefold() for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("catalog extension IDs must be unique")
        return self


def extension_catalog_json_schema() -> dict[str, Any]:
    schema = ExtensionCatalogDocument.model_json_schema()
    entry_schema = schema["$defs"]["ExtensionCatalogEntry"]
    required = list(entry_schema["required"])
    insert_at = required.index("capabilities")
    entry_schema["required"] = [
        *required[:insert_at],
        "execution_level",
        "permissions",
        *required[insert_at:],
    ]
    entry_schema["properties"]["execution_level"].pop("default", None)
    entry_schema["allOf"] = [
        {
            "if": {
                "properties": {"execution_level": {"const": "standard"}},
                "required": ["execution_level"],
            },
            "then": {
                "properties": {
                    "capabilities": {"type": "array", "maxItems": 0}
                }
            },
        },
        {
            "if": {
                "properties": {"execution_level": {"const": "full_access"}},
                "required": ["execution_level"],
            },
            "then": {
                "properties": {
                    "permissions": {
                        "type": "object",
                        "properties": {
                            "asteria": {"type": "array", "maxItems": 0},
                            "network_hosts": {"type": "array", "maxItems": 0},
                            "localhost": {"type": "array", "maxItems": 0},
                        }
                    }
                }
            },
        },
    ]
    return {"$id": ASTERIA_EXTENSION_CATALOG_SCHEMA_ID, **schema}
