from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal
from urllib.parse import urlsplit
import ipaddress
import re

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


ASTERIA_EXTENSION_SCHEMA_ID = "asteria.extension.v1"
ASTERIA_EXTENSION_STATE_SCHEMA_ID = "asteria.extension.state.v1"
ASTERIA_EXTENSION_RECEIPT_SCHEMA_ID = "asteria.extension.receipt.v1"
ASTERIA_EXTENSION_REGISTRY_SCHEMA_ID = "asteria.extension.registry.v1"
SEMVER_PATTERN = (
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_SEMVER_RE = re.compile(SEMVER_PATTERN)
_EXTENSION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_CONTRIBUTION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_SAFE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BACKEND_ENTRYPOINT_PATTERN = (
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)
_FRONTEND_ENTRYPOINT_PATTERN = (
    r"^(?!/)(?![A-Za-z]:)(?!.*(?:^|/)\.\.?(?:/|$))(?!.*\\)"
    r"(?!.*[<>:\"|?*#%\x00-\x1f])"
    r"(?!.*(?:^|/)(?:[Cc][Oo][Nn]|[Pp][Rr][Nn]|[Aa][Uu][Xx]|[Nn][Uu][Ll]|"
    r"[Cc][Oo][Mm][1-9]|[Ll][Pp][Tt][1-9])(?:\.[^/]*)?(?:/|$))"
    r"(?!.*[ .](?:/|$)).+\.(?:js|mjs)$"
)
_WINDOWS_RESERVED_FILENAME_RE = re.compile(
    r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$",
    re.IGNORECASE,
)
_WINDOWS_UNSAFE_PATH_CHARS = frozenset('<>:"\\|?*#%')


StrictSemVer = Annotated[str, Field(pattern=SEMVER_PATTERN)]
Capability = Literal["network", "filesystem", "subprocess"]
ExecutionLevel = Literal["standard", "full_access"]
ASTERIA_RESERVED_EXTENSION_LOCALHOST_PORTS = frozenset({3000, 3001, 5173, 8000, 8188})
StandardAsteriaPermission = Literal[
    "prompt.read",
]
WorkspaceSurface = Literal["independent_workspace"]
PanelSurface = Literal[
    "generate.inspector_panel",
    "image_edit.inspector_panel",
    "media_library.details",
    "lora_training.tool",
    "merge_studio.tool",
    "settings.integration_card",
]
ActionSurface = Literal[
    "generate.inspector_action",
    "image_edit.inspector_action",
    "media_library.context_action",
    "lora_training.action",
    "merge_studio.action",
    "backend.api_router",
    "backend.operation",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def validate_extension_id(value: str) -> str:
    if not _EXTENSION_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError("extension id is invalid")
    return value


def validate_semver(value: str) -> str:
    if not _SEMVER_RE.fullmatch(value):
        raise ValueError("value must be a semantic version")
    return value


def compare_semver(left: str, right: str) -> int:
    left_match = _SEMVER_RE.fullmatch(left)
    right_match = _SEMVER_RE.fullmatch(right)
    if left_match is None or right_match is None:
        raise ValueError("value must be a semantic version")
    left_core = tuple(int(value) for value in left_match.groups()[:3])
    right_core = tuple(int(value) for value in right_match.groups()[:3])
    if left_core != right_core:
        return -1 if left_core < right_core else 1
    left_pre = left_match.group(4)
    right_pre = right_match.group(4)
    if left_pre == right_pre:
        return 0
    if left_pre is None:
        return 1
    if right_pre is None:
        return -1
    left_parts = left_pre.split(".")
    right_parts = right_pre.split(".")
    for left_part, right_part in zip(left_parts, right_parts, strict=False):
        if left_part == right_part:
            continue
        left_numeric = left_part.isdigit()
        right_numeric = right_part.isdigit()
        if left_numeric and right_numeric:
            return -1 if int(left_part) < int(right_part) else 1
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        return -1 if left_part < right_part else 1
    return -1 if len(left_parts) < len(right_parts) else 1


class AsteriaCompatibility(_StrictFrozenModel):
    minimum: StrictSemVer
    maximum_exclusive: StrictSemVer

    @field_validator("minimum", "maximum_exclusive")
    @classmethod
    def _validate_versions(cls, value: str) -> str:
        return validate_semver(value)

    @model_validator(mode="after")
    def _validate_range(self) -> AsteriaCompatibility:
        if compare_semver(self.maximum_exclusive, self.minimum) <= 0:
            raise ValueError("maximum_exclusive must be greater than minimum")
        return self


class ExtensionEntrypoints(_StrictFrozenModel):
    backend: str = Field(
        json_schema_extra={"pattern": _BACKEND_ENTRYPOINT_PATTERN},
    )
    frontend: str = Field(
        json_schema_extra={"pattern": _FRONTEND_ENTRYPOINT_PATTERN},
    )
    frontend_element: str = Field(
        json_schema_extra={"pattern": r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$"},
    )

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*",
            value,
        ):
            raise ValueError("backend entrypoint must use module:function")
        return value

    @field_validator("frontend")
    @classmethod
    def _validate_frontend(cls, value: str) -> str:
        try:
            normalized = validate_extension_relative_path(
                value,
                allowed_suffixes=(".js", ".mjs"),
            )
        except ValueError as exc:
            raise ValueError("frontend entrypoint must be a relative ESM path") from exc
        return normalized

    @field_validator("frontend_element")
    @classmethod
    def _validate_frontend_element(cls, value: str) -> str:
        if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+", value) is None:
            raise ValueError("frontend custom element name is invalid")
        return value


def validate_extension_relative_path(
    value: str,
    *,
    allowed_suffixes: tuple[str, ...] | None = None,
) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if (
        not value
        or "\\" in value
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("extension path is invalid")
    for part in parts:
        if (
            any(character in _WINDOWS_UNSAFE_PATH_CHARS or ord(character) < 32 for character in part)
            or part.endswith((" ", "."))
            or _WINDOWS_RESERVED_FILENAME_RE.fullmatch(part)
        ):
            raise ValueError("extension path is not Windows-safe")
    normalized = posix.as_posix()
    if allowed_suffixes is not None and not normalized.casefold().endswith(allowed_suffixes):
        raise ValueError("extension path has an unsupported suffix")
    return normalized


class _Contribution(_StrictFrozenModel):
    id: str = Field(json_schema_extra={"pattern": _CONTRIBUTION_ID_RE.pattern})
    name: str = Field(min_length=1, max_length=160)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not _CONTRIBUTION_ID_RE.fullmatch(value) or value in {".", ".."}:
            raise ValueError("contribution id is invalid")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("contribution name must not have surrounding whitespace")
        return value


class WorkspaceContribution(_Contribution):
    surface: WorkspaceSurface


class PanelContribution(_Contribution):
    surface: PanelSurface


class ActionContribution(_Contribution):
    surface: ActionSurface


class ExtensionContributions(_StrictFrozenModel):
    workspaces: tuple[WorkspaceContribution, ...] = ()
    panels: tuple[PanelContribution, ...] = ()
    actions: tuple[ActionContribution, ...] = ()

    @model_validator(mode="after")
    def _unique_ids(self) -> ExtensionContributions:
        seen: set[str] = set()
        for contribution in (*self.workspaces, *self.panels, *self.actions):
            normalized = contribution.id.casefold()
            if normalized in seen:
                raise ValueError("duplicate contribution id")
            seen.add(normalized)
        return self


class ExtensionLicense(_StrictFrozenModel):
    id: str = Field(min_length=1, max_length=120)
    url: AnyHttpUrl = Field(json_schema_extra={"pattern": r"^https?://"})

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("license id must not have surrounding whitespace")
        return value


class ExtensionLocalhostPermission(_StrictFrozenModel):
    host: Literal["localhost", "127.0.0.1", "::1"]
    port: int = Field(
        ge=1,
        le=65535,
        json_schema_extra={
            "not": {"enum": sorted(ASTERIA_RESERVED_EXTENSION_LOCALHOST_PORTS)},
        },
    )

    @field_validator("port")
    @classmethod
    def _reject_asteria_reserved_port(cls, value: int) -> int:
        if value in ASTERIA_RESERVED_EXTENSION_LOCALHOST_PORTS:
            raise ValueError("localhost port is reserved for Asteria")
        return value


class ExtensionStandardPermissions(_StrictFrozenModel):
    asteria: tuple[StandardAsteriaPermission, ...] = Field(
        default=(),
        json_schema_extra={"uniqueItems": True},
    )
    network_hosts: tuple[str, ...] = Field(
        default=(),
        json_schema_extra={"uniqueItems": True},
    )
    localhost: tuple[ExtensionLocalhostPermission, ...] = Field(
        default=(),
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("asteria")
    @classmethod
    def _unique_asteria_permissions(
        cls,
        value: tuple[StandardAsteriaPermission, ...],
    ) -> tuple[StandardAsteriaPermission, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate standard permission")
        return value

    @field_validator("network_hosts")
    @classmethod
    def _validate_network_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: set[str] = set()
        for item in value:
            parsed = urlsplit(item)
            hostname = parsed.hostname or ""
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError("network host declaration is invalid") from exc
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or "*" in item
                or hostname.casefold() == "localhost"
            ):
                raise ValueError("network host declaration is invalid")
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                pass
            else:
                raise ValueError("network host declaration is invalid")
            if "." not in hostname or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in hostname.split(".")
            ):
                raise ValueError("network host declaration is invalid")
            canonical = f"https://{hostname.casefold()}" + (
                f":{port}" if port not in (None, 443) else ""
            )
            if item != canonical:
                raise ValueError("network host declaration must use its canonical origin")
            if canonical in normalized:
                raise ValueError("duplicate network host declaration")
            normalized.add(canonical)
        return value

    @field_validator("localhost")
    @classmethod
    def _unique_localhost(
        cls,
        value: tuple[ExtensionLocalhostPermission, ...],
    ) -> tuple[ExtensionLocalhostPermission, ...]:
        identities = {(item.host, item.port) for item in value}
        if len(identities) != len(value):
            raise ValueError("duplicate localhost permission")
        return value

    def is_empty(self) -> bool:
        return not self.asteria and not self.network_hosts and not self.localhost


class ExtensionManifest(_StrictFrozenModel):
    schema_version: Literal[ASTERIA_EXTENSION_SCHEMA_ID]
    id: str = Field(json_schema_extra={"pattern": _EXTENSION_ID_RE.pattern})
    name: str = Field(min_length=1, max_length=160)
    version: StrictSemVer
    asteria: AsteriaCompatibility
    entrypoints: ExtensionEntrypoints
    contributions: ExtensionContributions
    execution_level: ExecutionLevel = "full_access"
    permissions: ExtensionStandardPermissions = Field(default_factory=ExtensionStandardPermissions)
    capabilities: tuple[Capability, ...] = Field(
        default=(),
        json_schema_extra={"uniqueItems": True},
    )
    license: ExtensionLicense

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_extension_id(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("extension name must not have surrounding whitespace")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return validate_semver(value)

    @field_validator("capabilities")
    @classmethod
    def _unique_capabilities(cls, value: tuple[Capability, ...]) -> tuple[Capability, ...]:
        if len({item.casefold() for item in value}) != len(value):
            raise ValueError("duplicate capability")
        return value

    @model_validator(mode="after")
    def _validate_execution_level(self) -> ExtensionManifest:
        if self.execution_level == "standard" and self.capabilities:
            raise ValueError("standard extension cannot declare full-access capabilities")
        if self.execution_level == "full_access" and not self.permissions.is_empty():
            raise ValueError("full-access extension cannot declare standard permissions")
        return self


class ExtensionSourceState(_StrictFrozenModel):
    kind: Literal["managed", "unmanaged"]
    install_url: AnyHttpUrl | None = None
    external_reference: str | None = Field(default=None, min_length=1, max_length=2048)

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> ExtensionSourceState:
        if self.kind == "managed" and self.external_reference is not None:
            raise ValueError("managed source cannot include an external reference")
        if self.kind == "unmanaged":
            if self.external_reference is None:
                raise ValueError("unmanaged source requires external reference metadata")
            if self.install_url is not None:
                raise ValueError("unmanaged source cannot include an install URL")
        return self


class ExtensionLastLoad(_StrictFrozenModel):
    status: Literal["never", "success", "failed"]
    version: StrictSemVer | None = None
    error_code: str | None = None

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str | None) -> str | None:
        return validate_semver(value) if value is not None else None

    @field_validator("error_code")
    @classmethod
    def _validate_error_code(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_CODE_RE.fullmatch(value):
            raise ValueError("last-load error code is invalid")
        return value

    @model_validator(mode="after")
    def _validate_status(self) -> ExtensionLastLoad:
        if self.status == "never" and (self.version is not None or self.error_code is not None):
            raise ValueError("never-loaded state cannot include a version or error")
        if self.status == "success" and (self.version is None or self.error_code is not None):
            raise ValueError("successful load requires a version and no error")
        if self.status == "failed" and (self.version is None or self.error_code is None):
            raise ValueError("failed load requires a version and error code")
        return self


class ExtensionRecoveryState(_StrictFrozenModel):
    status: Literal["ready", "restart_required", "missing_install", "load_failed"]
    guidance: Literal["none", "restart", "reinstall", "reconnect", "safe_mode"]


class ExtensionFileHash(_StrictFrozenModel):
    relative_path: str
    size: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        posix = PurePosixPath(value.replace("\\", "/"))
        windows = PureWindowsPath(value)
        if (
            not value
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ValueError("receipt file path is invalid")
        return posix.as_posix()

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("receipt file hash is invalid")
        return value


class ExtensionVersionReceipt(_StrictFrozenModel):
    schema_version: Literal[ASTERIA_EXTENSION_RECEIPT_SCHEMA_ID] = ASTERIA_EXTENSION_RECEIPT_SCHEMA_ID
    extension_id: str
    version: StrictSemVer
    package_checksum: str
    tree_checksum: str
    files: tuple[ExtensionFileHash, ...]
    source_trust: Literal["catalog_verified", "user_checksum", "unverified_direct"] | None = None
    resolved_commit: str | None = None

    @field_validator("extension_id")
    @classmethod
    def _validate_extension_id(cls, value: str) -> str:
        return validate_extension_id(value)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        return validate_semver(value)

    @field_validator("package_checksum", "tree_checksum")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("receipt checksum is invalid")
        return value

    @field_validator("resolved_commit")
    @classmethod
    def _validate_commit(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
            raise ValueError("receipt resolved commit is invalid")
        return value

    @model_validator(mode="after")
    def _validate_files(self) -> ExtensionVersionReceipt:
        keys = [item.relative_path.casefold() for item in self.files]
        if not keys or len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("receipt files must be non-empty, unique, and sorted")
        return self


class ExtensionState(_StrictFrozenModel):
    schema_version: Literal[ASTERIA_EXTENSION_STATE_SCHEMA_ID] = ASTERIA_EXTENSION_STATE_SCHEMA_ID
    active_version: StrictSemVer
    previous_version: StrictSemVer | None = None
    active_receipt_checksum: str | None
    previous_receipt_checksum: str | None = None
    enabled: bool
    source: ExtensionSourceState
    last_load: ExtensionLastLoad
    recovery: ExtensionRecoveryState
    pending_action: Literal["install", "update", "rollback", "uninstall"] | None = None

    @field_validator("active_version", "previous_version")
    @classmethod
    def _validate_version(cls, value: str | None) -> str | None:
        return validate_semver(value) if value is not None else None

    @field_validator("active_receipt_checksum", "previous_receipt_checksum")
    @classmethod
    def _validate_checksum(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("receipt checksum must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_source_lifecycle(self) -> ExtensionState:
        if self.previous_version == self.active_version:
            raise ValueError("previous version must differ from active version")
        if self.source.kind == "managed":
            if self.active_receipt_checksum is None:
                raise ValueError("managed active version requires a receipt checksum")
            if (self.previous_version is None) != (self.previous_receipt_checksum is None):
                raise ValueError("managed previous version and receipt checksum must be paired")
        elif (
            self.previous_version is not None
            or self.active_receipt_checksum is not None
            or self.previous_receipt_checksum is not None
        ):
            raise ValueError("unmanaged extensions do not have managed rollback or receipts")
        versions = {self.active_version, self.previous_version}
        if self.last_load.version is not None and self.last_load.version not in versions:
            raise ValueError("last-load version must be active or previous")
        if self.recovery.status == "ready" and (
            self.recovery.guidance != "none"
            or self.last_load.status != "success"
            or self.pending_action is not None
        ):
            raise ValueError("ready recovery state requires a successful load and no guidance")
        if self.recovery.status == "restart_required":
            if self.recovery.guidance != "restart" or self.pending_action is None:
                raise ValueError("restart-required state requires a pending action and restart guidance")
            if self.pending_action == "install":
                if self.previous_version is not None or self.last_load.status != "never":
                    raise ValueError("pending install must describe a never-loaded first version")
            elif self.pending_action in {"update", "rollback"}:
                if self.source.kind != "managed" or self.previous_version is None:
                    raise ValueError("pending update or rollback requires managed active and previous versions")
                if self.last_load.status == "never" or self.last_load.version != self.previous_version:
                    raise ValueError("pending update or rollback must retain the previously loaded version")
            elif self.pending_action == "uninstall" and self.enabled:
                raise ValueError("pending uninstall must be disabled")
        elif self.pending_action is not None:
            raise ValueError("pending action requires restart-required recovery")
        if self.recovery.status == "missing_install":
            expected = "reinstall" if self.source.kind == "managed" else "reconnect"
            if self.enabled or self.recovery.guidance != expected:
                raise ValueError("missing-install state must be disabled with source recovery guidance")
        if self.recovery.status == "load_failed" and (
            self.recovery.guidance != "safe_mode" or self.last_load.status != "failed"
        ):
            raise ValueError("load-failed state requires failed load and safe-mode guidance")
        if self.last_load.status == "never" and self.recovery.status not in {
            "restart_required",
            "missing_install",
        }:
            raise ValueError("never-loaded state requires restart or missing-install recovery")
        return self


ExtensionRegistryErrorCode = Literal[
    "extension_active_version_invalid",
    "extension_entrypoint_invalid",
    "extension_id_collision",
    "extension_id_invalid",
    "extension_manifest_incompatible",
    "extension_manifest_invalid",
    "extension_manifest_mismatch",
    "extension_path_escape",
    "extension_path_unreadable",
    "extension_previous_version_invalid",
    "extension_receipt_invalid",
    "extension_reparse_traversal",
    "extension_root_unreadable",
    "extension_state_invalid",
    "extension_unmanaged_reference_missing",
]


class ExtensionRegistryRecord(_StrictFrozenModel):
    extension_id: str
    status: Literal["ready", "recovery", "error"]
    state: ExtensionState | None = None
    manifest: ExtensionManifest | None = None
    error_code: ExtensionRegistryErrorCode | None = None

    @field_validator("extension_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_extension_id(value)

    @model_validator(mode="after")
    def _validate_record(self) -> ExtensionRegistryRecord:
        if self.status == "error":
            if self.error_code is None or self.state is not None or self.manifest is not None:
                raise ValueError("error registry record must contain only a known sanitized code")
            return self
        if self.state is None or self.error_code is not None:
            raise ValueError("non-error registry record requires state and no error")
        if self.status == "recovery":
            if self.state.recovery.status == "ready":
                raise ValueError("recovery registry record requires a recovery state")
            if self.state.recovery.status == "missing_install" and self.manifest is not None:
                raise ValueError("missing-install registry record cannot contain manifest")
            if (
                self.state.source.kind == "managed"
                and self.state.recovery.status != "missing_install"
                and self.manifest is None
            ):
                raise ValueError("managed recovery registry record requires manifest")
            if self.state.source.kind == "unmanaged" and self.manifest is not None:
                raise ValueError("unmanaged recovery registry record cannot contain manifest")
            return self
        if self.state.recovery.status != "ready":
            raise ValueError("ready registry record requires ready state")
        if self.state.source.kind == "managed" and self.manifest is None:
            raise ValueError("ready managed registry record requires manifest")
        if self.state.source.kind == "unmanaged" and self.manifest is not None:
            raise ValueError("ready unmanaged registry record cannot contain managed manifest")
        return self


class ExtensionRegistrySnapshotRecord(_StrictFrozenModel):
    extension_id: str
    status: Literal["ready", "recovery", "error"]
    active_version: StrictSemVer | None = None
    previous_version: StrictSemVer | None = None
    enabled: bool | None = None
    source_kind: Literal["managed", "unmanaged"] | None = None
    recovery_status: Literal["ready", "restart_required", "missing_install", "load_failed"] | None = None
    manifest_version: StrictSemVer | None = None
    error_code: ExtensionRegistryErrorCode | None = None

    @field_validator("extension_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return validate_extension_id(value)

    @model_validator(mode="after")
    def _validate_snapshot_record(self) -> ExtensionRegistrySnapshotRecord:
        all_metadata = (
            self.active_version,
            self.previous_version,
            self.enabled,
            self.source_kind,
            self.recovery_status,
        )
        if self.status == "error":
            if self.error_code is None or any(value is not None for value in all_metadata) or self.manifest_version is not None:
                raise ValueError("snapshot error record must contain only a known sanitized code")
            return self
        required_metadata = (
            self.active_version,
            self.enabled,
            self.source_kind,
            self.recovery_status,
        )
        if self.error_code is not None or any(value is None for value in required_metadata):
            raise ValueError("snapshot non-error record requires state metadata")
        if self.previous_version == self.active_version:
            raise ValueError("snapshot previous version must differ from active version")
        if self.status == "ready" and self.recovery_status != "ready":
            raise ValueError("snapshot ready record requires ready recovery status")
        if self.status == "recovery" and self.recovery_status == "ready":
            raise ValueError("snapshot recovery record requires non-ready recovery status")
        if self.source_kind == "managed":
            if self.recovery_status != "missing_install" and self.manifest_version != self.active_version:
                raise ValueError("snapshot managed record requires active manifest version")
            if self.recovery_status == "missing_install" and self.manifest_version is not None:
                raise ValueError("snapshot missing-install record cannot contain manifest")
        elif self.manifest_version is not None or self.previous_version is not None:
            raise ValueError("snapshot unmanaged record cannot contain manifest or rollback version")
        if self.recovery_status == "missing_install" and self.enabled:
            raise ValueError("snapshot missing-install record must be disabled")
        return self


class ExtensionRegistrySnapshot(_StrictFrozenModel):
    schema_version: Literal[ASTERIA_EXTENSION_REGISTRY_SCHEMA_ID] = ASTERIA_EXTENSION_REGISTRY_SCHEMA_ID
    records: tuple[ExtensionRegistrySnapshotRecord, ...]

    @model_validator(mode="after")
    def _unique_records(self) -> ExtensionRegistrySnapshot:
        ids = [record.extension_id.casefold() for record in self.records]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("registry snapshot IDs must be sorted and unique")
        return self
