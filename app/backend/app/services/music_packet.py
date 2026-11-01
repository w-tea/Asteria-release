from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import secrets
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.core.config import AppConfig
from app.schemas.music import MusicRequest, contains_secret_key
from app.schemas.provider_capabilities import ProviderCapabilityDescriptor
from app.services.music_runtime_adapters import MusicAdapterError, MusicRuntimeAdapter


MUSIC_MAPPING_SCHEMA_VERSION = "asteria.music.mapping.v1"
MUSIC_PACKET_VERSION = "asteria.music.submission.v1"
_MAPPING_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class MusicPacketError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class MusicProviderOptionMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value_type: Literal["enum", "boolean", "integer", "number", "string"]
    target: str
    allowed_values: tuple[JsonValue, ...] = ()
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if not value or any(not part for part in value.split(".")) or contains_secret_key({value: None}):
            raise ValueError("music provider option target cannot contain secret fields")
        return value

    @model_validator(mode="after")
    def validate_allowed(self) -> "MusicProviderOptionMapping":
        if self.value_type == "enum" and not self.allowed_values:
            raise ValueError("enum provider option requires allowed values")
        if self.value_type != "enum" and self.allowed_values:
            raise ValueError("only enum provider options may declare allowed values")
        if self.value_type not in {"integer", "number"} and (self.minimum is not None or self.maximum is not None):
            raise ValueError("only numeric provider options may declare ranges")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("provider option minimum cannot exceed maximum")
        return self


class MusicModeMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_task: str
    payload_template: dict[str, JsonValue]
    v2_payload_template: dict[str, JsonValue] | None = None
    value_maps: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)

    @field_validator("provider_task")
    @classmethod
    def validate_provider_task(cls, value: str) -> str:
        if not _MAPPING_IDENTIFIER.fullmatch(value):
            raise ValueError("music provider task is invalid")
        return value

    @field_validator("payload_template")
    @classmethod
    def validate_payload_template(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        workflow_template_id = value.get("workflow_template_id")
        if not isinstance(workflow_template_id, str) or not _MAPPING_IDENTIFIER.fullmatch(workflow_template_id):
            raise ValueError("music workflow template identifier is invalid")
        return value

    @field_validator("v2_payload_template")
    @classmethod
    def validate_v2_payload_template(
        cls,
        value: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        workflow_template_id = value.get("workflow_template_id")
        if not isinstance(workflow_template_id, str) or not _MAPPING_IDENTIFIER.fullmatch(workflow_template_id):
            raise ValueError("music v2 workflow template identifier is invalid")
        return value


class MusicMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["asteria.music.mapping.v1"]
    mapping_id: str
    mapping_revision: int = Field(ge=1)
    provider: Literal["local_music3"]
    adapter_id: str
    mode_mappings: dict[str, MusicModeMapping]
    provider_options: dict[str, MusicProviderOptionMapping] = Field(default_factory=dict)

    @field_validator("mapping_id", "adapter_id")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        if not _MAPPING_IDENTIFIER.fullmatch(value):
            raise ValueError("music mapping identifier is invalid")
        return value

    @model_validator(mode="after")
    def validate_mapping(self) -> "MusicMapping":
        if set(self.mode_mappings) != {"vocal", "instrumental"}:
            raise ValueError("music mapping must cover vocal and instrumental")
        if contains_secret_key(self.model_dump(mode="json")):
            raise ValueError("music mapping cannot contain secret fields")
        return self


class MusicMappingRepository:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else AppConfig().mapping_dir / "music"
        )

    def load(self, mapping_id: str) -> MusicMapping:
        if not mapping_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in mapping_id):
            raise MusicPacketError("music_mapping_not_found")
        try:
            raw = json.loads((self.root / f"{mapping_id}.json").read_text(encoding="utf-8"))
            mapping = MusicMapping.model_validate(raw)
        except FileNotFoundError as error:
            raise MusicPacketError("music_mapping_not_found") from error
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise MusicPacketError("music_mapping_invalid") from error
        if mapping.mapping_id != mapping_id:
            raise MusicPacketError("music_mapping_invalid")
        return mapping


@dataclass(frozen=True, slots=True)
class MusicSubmissionPacket:
    packet_version: str
    runtime_job_id: str
    adapter_id: str
    adapter_revision: str
    mapping_id: str
    mapping_revision: int
    _request_json: str
    _capability_json: str
    _payload_json: str

    @property
    def request_snapshot(self) -> dict[str, Any]:
        return json.loads(self._request_json)

    @property
    def capability_snapshot(self) -> dict[str, Any]:
        return json.loads(self._capability_json)

    @property
    def provider_payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {"packet_version": self.packet_version, "runtime_job_id": self.runtime_job_id, "adapter_id": self.adapter_id, "adapter_revision": self.adapter_revision, "mapping_id": self.mapping_id, "mapping_revision": self.mapping_revision, "request_snapshot": self.request_snapshot, "capability_snapshot": self.capability_snapshot, "provider_payload": self.provider_payload}


class MusicPacketBuilder:
    def __init__(self, mappings: MusicMappingRepository | None = None) -> None:
        self.mappings = mappings or MusicMappingRepository()

    def build(
        self,
        request: MusicRequest,
        adapter: MusicRuntimeAdapter,
        *,
        capability: ProviderCapabilityDescriptor | None = None,
    ) -> MusicSubmissionPacket:
        capability = capability or adapter.capabilities()
        request = _resolve_request_defaults(request, capability)
        try:
            adapter.validate_request(request)
        except MusicAdapterError as error:
            raise MusicPacketError(error.code) from error
        mapping = self.mappings.load(adapter.mapping_id)
        if mapping.provider != request.provider or mapping.adapter_id != adapter.id:
            raise MusicPacketError("music_mapping_mismatch")
        mode_mapping = mapping.mode_mappings.get(request.mode)
        if mode_mapping is None:
            raise MusicPacketError("music_mapping_mode_unsupported")
        values: dict[str, JsonValue] = {"description": request.description, "lyrics": request.lyrics, "seed": request.seed, "duration_intent": request.duration_intent, "language_intent": request.language_intent, "model_profile": request.model_profile, "provider_task": mode_mapping.provider_task}
        template = mode_mapping.payload_template
        if request.schema_version == "asteria.music.request.v2":
            if mode_mapping.v2_payload_template is None or request.model_selection is None or request.sampling is None:
                raise MusicPacketError("music_mapping_v2_unsupported")
            production = capability.music_production
            if production is None:
                raise MusicPacketError("music_mapping_v2_unsupported")
            values.update(
                {
                    "model_selection": request.model_selection.model_dump(mode="json"),
                    "text_guidance": request.sampling.text_guidance,
                    "top_k": request.sampling.top_k,
                    "steps": request.sampling.steps,
                    "sampler": request.sampling.sampler,
                    "scheduler": request.sampling.scheduler,
                    "denoise": request.sampling.denoise,
                    "tiled_decode": request.sampling.tiled_decode,
                    "execution_profile": request.sampling.execution_profile,
                    "ksampler_cfg": production.ksampler_cfg,
                }
            )
            template = mode_mapping.v2_payload_template
        payload = _render(template, values)
        if not isinstance(payload, dict):
            raise MusicPacketError("music_mapping_invalid")
        self._apply_options(payload, request.provider_options, mapping.provider_options)
        if contains_secret_key(payload):
            raise MusicPacketError("music_packet_secret_rejected")
        return MusicSubmissionPacket(MUSIC_PACKET_VERSION, f"asteria-music-{uuid4().hex}", adapter.id, adapter.revision, mapping.mapping_id, mapping.mapping_revision, _canonical(request.model_dump(mode="json")), _canonical(capability.model_dump(mode="json")), _canonical(payload))

    def _apply_options(self, payload: dict[str, JsonValue], options: dict[str, JsonValue], contracts: dict[str, MusicProviderOptionMapping]) -> None:
        if set(options) - set(contracts):
            raise MusicPacketError("music_provider_option_unsupported")
        for key, value in options.items():
            contract = contracts[key]
            if not _matches(value, contract.value_type) or (contract.value_type == "enum" and value not in contract.allowed_values):
                raise MusicPacketError("music_provider_option_invalid")
            if contract.minimum is not None and float(value) < contract.minimum:
                raise MusicPacketError("music_provider_option_invalid")
            if contract.maximum is not None and float(value) > contract.maximum:
                raise MusicPacketError("music_provider_option_invalid")
            _assign(payload, contract.target, deepcopy(value))


def _render(value: JsonValue, values: dict[str, JsonValue]) -> JsonValue:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        if key not in values:
            raise MusicPacketError("music_mapping_invalid")
        return deepcopy(values[key])
    if isinstance(value, dict):
        return {key: _render(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, values) for item in value]
    return deepcopy(value)


def _assign(payload: dict[str, JsonValue], target: str, value: JsonValue) -> None:
    parts = target.split(".")
    current = payload
    for part in parts[:-1]:
        nested = current.setdefault(part, {})
        if not isinstance(nested, dict):
            raise MusicPacketError("music_mapping_invalid")
        current = nested
    current[parts[-1]] = value


def _matches(value: JsonValue, value_type: str) -> bool:
    return {"boolean": isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool), "number": isinstance(value, (int, float)) and not isinstance(value, bool), "string": isinstance(value, str), "enum": isinstance(value, (str, int, float, bool)) and not isinstance(value, dict)}.get(value_type, False)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolve_request_defaults(request: MusicRequest, capability: Any) -> MusicRequest:
    updates: dict[str, Any] = {}
    if request.seed == -1:
        hard = capability.provider_hard_parameters.get("seed")
        minimum = int(hard.minimum) if hard and hard.minimum is not None else 0
        maximum = int(hard.maximum) if hard and hard.maximum is not None else 9223372036854775807
        updates["seed"] = minimum + secrets.randbelow(maximum - minimum + 1)
    if request.duration_intent is None:
        updates["duration_intent"] = capability.duration.default_seconds
    return request.model_copy(update=updates) if updates else request
