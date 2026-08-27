from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import secrets
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.core.config import AppConfig
from app.schemas.provider_capabilities import ProviderCapabilityDescriptor
from app.schemas.video import VideoRequest, contains_secret_key, is_secret_field_name
from app.services.video_runtime_adapters import VideoAdapterError, VideoRuntimeAdapter
from app.services.video_sequence_context import VideoSequenceContextReceipt


VIDEO_MAPPING_SCHEMA_VERSION = "asteria.video.mapping.v1"
VIDEO_PACKET_VERSION = "asteria.video.submission.v1"


class VideoPacketError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class VideoModelPreflightError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class VideoModelSelectionPreflight(Protocol):
    def validate(self, request: VideoRequest) -> None: ...


class VideoProviderOptionMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value_type: Literal["enum", "boolean", "integer", "number", "string"]
    target: str
    allowed_values: tuple[JsonValue, ...] = ()

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        parts = value.split(".")
        if not value or any(not part for part in parts):
            raise ValueError("video provider option target is invalid")
        if any(is_secret_field_name(part) for part in parts):
            raise ValueError("video provider option target cannot contain secret fields")
        return value

    @model_validator(mode="after")
    def validate_allowed_values(self) -> "VideoProviderOptionMapping":
        if self.value_type == "enum" and not self.allowed_values:
            raise ValueError("enum provider option requires allowed values")
        if self.value_type != "enum" and self.allowed_values:
            raise ValueError("only enum provider options may declare allowed values")
        return self


class VideoModeMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_task: str
    payload_template: dict[str, JsonValue]
    value_maps: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)


class VideoMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video.mapping.v1"]
    mapping_id: str
    mapping_version: int = Field(ge=1)
    provider: Literal["local_h3"]
    adapter_id: str
    mode_mappings: dict[str, VideoModeMapping]
    provider_options: dict[str, VideoProviderOptionMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mapping(self) -> "VideoMapping":
        if not self.mode_mappings:
            raise ValueError("video mapping requires a mode")
        if contains_secret_key(self.model_dump(mode="json")):
            raise ValueError("video mapping cannot contain secret fields")
        return self


class VideoMappingRepository:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else AppConfig().mapping_dir / "video"
        )

    def load(self, mapping_id: str) -> VideoMapping:
        if not mapping_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in mapping_id):
            raise VideoPacketError("video_mapping_not_found")
        path = self.root / f"{mapping_id}.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            mapping = VideoMapping.model_validate(raw)
        except FileNotFoundError as error:
            raise VideoPacketError("video_mapping_not_found") from error
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise VideoPacketError("video_mapping_invalid") from error
        if mapping.mapping_id != mapping_id:
            raise VideoPacketError("video_mapping_invalid")
        return mapping


@dataclass(frozen=True, slots=True)
class VideoSubmissionPacket:
    packet_version: str
    runtime_job_id: str
    adapter_id: str
    adapter_revision: str
    mapping_id: str
    mapping_version: int
    _request_snapshot_json: str
    _capability_snapshot_json: str
    _provider_payload_json: str
    _sequence_context_json: str | None = None

    @property
    def request_snapshot(self) -> dict[str, Any]:
        return json.loads(self._request_snapshot_json)

    @property
    def capability_snapshot(self) -> dict[str, Any]:
        return json.loads(self._capability_snapshot_json)

    @property
    def provider_payload(self) -> dict[str, Any]:
        return json.loads(self._provider_payload_json)

    @property
    def sequence_context(self) -> dict[str, Any] | None:
        if self._sequence_context_json is None:
            return None
        return json.loads(self._sequence_context_json)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "packet_version": self.packet_version,
            "runtime_job_id": self.runtime_job_id,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "mapping_id": self.mapping_id,
            "mapping_version": self.mapping_version,
            "request_snapshot": self.request_snapshot,
            "capability_snapshot": self.capability_snapshot,
            "provider_payload": self.provider_payload,
        }
        if self.sequence_context is not None:
            payload["sequence_context"] = self.sequence_context
        return payload


class VideoPacketBuilder:
    def __init__(
        self,
        mappings: VideoMappingRepository | None = None,
        *,
        model_preflight: VideoModelSelectionPreflight | None = None,
    ) -> None:
        self.mappings = mappings or VideoMappingRepository()
        self.model_preflight = model_preflight

    def build(
        self,
        request: VideoRequest,
        adapter: VideoRuntimeAdapter,
        *,
        capability: ProviderCapabilityDescriptor | None = None,
    ) -> VideoSubmissionPacket:
        if request.mode == "sequence":
            raise VideoPacketError("video_sequence_context_required")
        capability = capability or adapter.capabilities()
        request = _resolve_random_seed(request, capability)
        try:
            adapter.validate_request(request)
        except VideoAdapterError as error:
            raise VideoPacketError(error.code) from error
        self._validate_model_receipts(request)
        mapping = self.mappings.load(adapter.mapping_id)
        if mapping.provider != request.provider or mapping.adapter_id != adapter.id:
            raise VideoPacketError("video_mapping_mismatch")
        mode_mapping = mapping.mode_mappings.get(request.mode)
        if mode_mapping is None:
            raise VideoPacketError("video_mapping_mode_unsupported")
        values = self._request_values(request, mode_mapping)
        payload = _render_template(mode_mapping.payload_template, values)
        if not isinstance(payload, dict):
            raise VideoPacketError("video_mapping_invalid")
        self._apply_provider_options(payload, request.provider_options, mapping.provider_options)
        if contains_secret_key(payload):
            raise VideoPacketError("video_packet_secret_rejected")
        return VideoSubmissionPacket(
            packet_version=VIDEO_PACKET_VERSION,
            runtime_job_id=f"asteria-video-{uuid4().hex}",
            adapter_id=adapter.id,
            adapter_revision=adapter.revision,
            mapping_id=mapping.mapping_id,
            mapping_version=mapping.mapping_version,
            _request_snapshot_json=_canonical_json(request.model_dump(mode="json")),
            _capability_snapshot_json=_canonical_json(capability.model_dump(mode="json")),
            _provider_payload_json=_canonical_json(payload),
        )

    def build_sequence_initial(
        self,
        request: VideoRequest,
        adapter: VideoRuntimeAdapter,
        *,
        capability: ProviderCapabilityDescriptor | None = None,
    ) -> VideoSubmissionPacket:
        if request.mode != "sequence":
            raise VideoPacketError("video_sequence_request_required")
        capability = capability or adapter.capabilities()
        request = _resolve_random_seed(request, capability)
        try:
            adapter.validate_request(request)
        except VideoAdapterError as error:
            raise VideoPacketError(error.code) from error
        self._validate_model_receipts(request)
        mapping = self.mappings.load(adapter.sequence_mapping_id)
        if mapping.provider != request.provider or mapping.adapter_id != adapter.id:
            raise VideoPacketError("video_mapping_mismatch")
        mode_mapping = mapping.mode_mappings.get("sequence_initial")
        if mode_mapping is None:
            raise VideoPacketError("video_mapping_mode_unsupported")
        values = self._request_values(request, mode_mapping)
        payload = _render_template(mode_mapping.payload_template, values)
        if not isinstance(payload, dict):
            raise VideoPacketError("video_mapping_invalid")
        self._apply_provider_options(payload, request.provider_options, mapping.provider_options)
        if contains_secret_key(payload):
            raise VideoPacketError("video_packet_secret_rejected")
        return VideoSubmissionPacket(
            packet_version=VIDEO_PACKET_VERSION,
            runtime_job_id=f"asteria-video-{uuid4().hex}",
            adapter_id=adapter.id,
            adapter_revision=adapter.revision,
            mapping_id=mapping.mapping_id,
            mapping_version=mapping.mapping_version,
            _request_snapshot_json=_canonical_json(request.model_dump(mode="json")),
            _capability_snapshot_json=_canonical_json(capability.model_dump(mode="json")),
            _provider_payload_json=_canonical_json(payload),
        )

    def build_sequence(
        self,
        request: VideoRequest,
        adapter: VideoRuntimeAdapter,
        context: VideoSequenceContextReceipt,
        *,
        capability: ProviderCapabilityDescriptor | None = None,
    ) -> VideoSubmissionPacket:
        if request.mode != "sequence":
            raise VideoPacketError("video_sequence_request_required")
        context = VideoSequenceContextReceipt.model_validate(context.model_dump(mode="json"))
        capability = capability or adapter.capabilities()
        request = _resolve_random_seed(request, capability)
        try:
            adapter.validate_request(request)
        except VideoAdapterError as error:
            raise VideoPacketError(error.code) from error
        self._validate_model_receipts(request)
        if (
            context.adapter_id != adapter.id
            or context.adapter_revision != adapter.revision
            or context.mapping_id != adapter.sequence_mapping_id
        ):
            raise VideoPacketError("video_sequence_context_mismatch")
        mapping = self.mappings.load(adapter.sequence_mapping_id)
        if (
            mapping.provider != request.provider
            or mapping.adapter_id != adapter.id
            or context.mapping_version != mapping.mapping_version
        ):
            raise VideoPacketError("video_mapping_mismatch")
        mode_mapping = mapping.mode_mappings.get("sequence")
        if mode_mapping is None:
            raise VideoPacketError("video_mapping_mode_unsupported")
        values = self._request_values(request, mode_mapping)
        context_payload = context.model_dump(mode="json")
        values.update(
            {
                "sequence_context": context_payload,
                "continuation_provider_task": context.method,
                "continuation_workflow_template_id": context.method,
            }
        )
        for field, value_map in mode_mapping.value_maps.items():
            source = values.get(field)
            if source is None:
                continue
            mapped = value_map.get(str(source))
            if mapped is None:
                raise VideoPacketError("video_mapping_value_unsupported")
            values[field] = mapped
        payload = _render_template(mode_mapping.payload_template, values)
        if not isinstance(payload, dict):
            raise VideoPacketError("video_mapping_invalid")
        self._apply_provider_options(payload, request.provider_options, mapping.provider_options)
        if contains_secret_key(payload):
            raise VideoPacketError("video_packet_secret_rejected")
        return VideoSubmissionPacket(
            packet_version=VIDEO_PACKET_VERSION,
            runtime_job_id=f"asteria-video-{uuid4().hex}",
            adapter_id=adapter.id,
            adapter_revision=adapter.revision,
            mapping_id=mapping.mapping_id,
            mapping_version=mapping.mapping_version,
            _request_snapshot_json=_canonical_json(request.model_dump(mode="json")),
            _capability_snapshot_json=_canonical_json(capability.model_dump(mode="json")),
            _provider_payload_json=_canonical_json(payload),
            _sequence_context_json=_canonical_json(context_payload),
        )

    def _request_values(
        self,
        request: VideoRequest,
        mode_mapping: VideoModeMapping,
    ) -> dict[str, JsonValue]:
        values: dict[str, JsonValue] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "width": request.size.width,
            "height": request.size.height,
            "aspect_ratio": request.size.aspect_ratio,
            "resolution_intent": request.size.resolution_intent,
            "duration_seconds": request.duration_seconds,
            "fps": request.fps,
            "audio_enabled": request.audio.enabled,
            "audio_prompt": request.audio.prompt,
            "first_frame_artifact_id": request.first_frame_artifact_id,
            "last_frame_artifact_id": request.last_frame_artifact_id,
            "references": [reference.model_dump(mode="json") for reference in request.references],
            "model_profile": request.model_profile,
            "model_selection": (
                request.model_selection.model_dump(mode="json")
                if request.model_selection is not None
                else None
            ),
            "sampling": (
                request.sampling.model_dump(mode="json")
                if request.sampling is not None
                else None
            ),
            "loras": [lora.model_dump(mode="json") for lora in request.loras],
            "provider_task": mode_mapping.provider_task,
        }
        for field, mapping in mode_mapping.value_maps.items():
            source = values.get(field)
            if source is None:
                continue
            mapped = mapping.get(str(source))
            if mapped is None:
                raise VideoPacketError("video_mapping_value_unsupported")
            values[field] = mapped
        return values

    def _validate_model_receipts(self, request: VideoRequest) -> None:
        if request.schema_version != "asteria.video.request.v2":
            return
        if self.model_preflight is None:
            raise VideoPacketError("video_model_preflight_required")
        try:
            self.model_preflight.validate(request)
        except VideoModelPreflightError as error:
            raise VideoPacketError(error.code) from error

    def _apply_provider_options(
        self,
        payload: dict[str, JsonValue],
        options: dict[str, JsonValue],
        contracts: dict[str, VideoProviderOptionMapping],
    ) -> None:
        unknown = set(options) - set(contracts)
        if unknown:
            raise VideoPacketError("video_provider_option_unsupported")
        for key, value in options.items():
            contract = contracts[key]
            if not _value_matches_type(value, contract.value_type):
                raise VideoPacketError("video_provider_option_invalid")
            if contract.value_type == "enum" and value not in contract.allowed_values:
                raise VideoPacketError("video_provider_option_invalid")
            _assign_path(payload, contract.target, deepcopy(value))


def _render_template(value: JsonValue, values: dict[str, JsonValue]) -> JsonValue:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        if key not in values:
            raise VideoPacketError("video_mapping_invalid")
        return deepcopy(values[key])
    if isinstance(value, dict):
        return {key: _render_template(nested, values) for key, nested in value.items()}
    if isinstance(value, list):
        return [_render_template(nested, values) for nested in value]
    return deepcopy(value)


def _assign_path(payload: dict[str, JsonValue], target: str, value: JsonValue) -> None:
    parts = target.split(".")
    current: dict[str, JsonValue] = payload
    for part in parts[:-1]:
        nested = current.get(part)
        if nested is None:
            nested = {}
            current[part] = nested
        if not isinstance(nested, dict):
            raise VideoPacketError("video_mapping_invalid")
        current = nested
    current[parts[-1]] = value


def _value_matches_type(value: JsonValue, value_type: str) -> bool:
    if value_type == "enum":
        return isinstance(value, (str, int, float, bool)) and not isinstance(value, dict)
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_type == "string":
        return isinstance(value, str)
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolve_random_seed(
    request: VideoRequest,
    capability: ProviderCapabilityDescriptor,
) -> VideoRequest:
    if request.seed != -1:
        return request
    hard_range = capability.provider_hard_parameters.get("seed")
    minimum = int(hard_range.minimum) if hard_range is not None and hard_range.minimum is not None else 0
    maximum = (
        int(hard_range.maximum)
        if hard_range is not None and hard_range.maximum is not None
        else 9223372036854775807
    )
    resolved_seed = minimum + secrets.randbelow(maximum - minimum + 1)
    return request.model_copy(update={"seed": resolved_seed})
