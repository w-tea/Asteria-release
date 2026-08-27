from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROVIDER_CAPABILITY_SCHEMA_VERSION = "asteria.provider.capability.v1"
CapabilityDomain = Literal["video", "music"]
CapabilityAvailability = Literal["ready", "setup_required", "unavailable"]
CapabilityEnforcement = Literal["advisory", "provider_hard_limit"]
ProgressSupport = Literal["none", "indeterminate", "determinate"]
SetupKind = Literal["runtime", "model", "license", "secret"]
BillingStatus = Literal["local_no_api_billing", "provider_billed"]
AudioOutputSemantics = Literal["none", "optional", "always", "provider_defined"]
LyricsBehavior = Literal["required", "optional", "forbidden", "provider_defined"]
LanguageIntentSemantics = Literal["unsupported", "optional", "required", "provider_defined"]
VideoContinuationMethod = Literal["direct_context", "reference_continuation"]
VideoContinuationOverlapUnit = Literal["frames", "none"]

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")


def _validate_token(value: str, label: str) -> str:
    if not _TOKEN.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value


class CapabilityRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    step: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    enforcement: CapabilityEnforcement

    @model_validator(mode="after")
    def validate_range(self) -> "CapabilityRange":
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("capability minimum cannot exceed maximum")
        if self.minimum is None and self.maximum is None:
            raise ValueError("capability range requires a minimum or maximum")
        return self


class DurationCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: Literal["seconds"] = "seconds"
    default_seconds: float = Field(gt=0, allow_inf_nan=False)
    alignment: Literal["none", "adapter_defined"] = "none"
    validated_range: CapabilityRange | None = None
    provider_hard_limit: CapabilityRange | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "DurationCapability":
        if self.validated_range is not None and self.validated_range.enforcement != "advisory":
            raise ValueError("validated duration range must be advisory")
        if self.provider_hard_limit is not None and self.provider_hard_limit.enforcement != "provider_hard_limit":
            raise ValueError("provider duration limit must use provider_hard_limit enforcement")
        return self


class SizeCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_width: int = Field(gt=0)
    default_height: int = Field(gt=0)
    multiple: int | None = Field(default=None, gt=0)
    validated_megapixels: CapabilityRange | None = None
    provider_hard_resolutions: tuple[str, ...] = ()

    @field_validator("provider_hard_resolutions")
    @classmethod
    def validate_resolutions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("provider hard resolutions must be non-empty and unique")
        return normalized

    @model_validator(mode="after")
    def validate_semantics(self) -> "SizeCapability":
        if self.validated_megapixels is not None and self.validated_megapixels.enforcement != "advisory":
            raise ValueError("validated size range must be advisory")
        return self


class MusicLyricsCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    behavior: LyricsBehavior

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validate_token(value, "music lyrics task identifier")


class MusicDomainCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lyrics: tuple[MusicLyricsCapability, ...]
    language_intent: LanguageIntentSemantics = "provider_defined"
    duration_intent_supported: bool = True

    @model_validator(mode="after")
    def validate_lyrics(self) -> "MusicDomainCapability":
        task_ids = [item.task_id for item in self.lyrics]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("music lyrics capabilities must be non-empty and unique")
        return self


class VideoContinuationMethodCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: VideoContinuationMethod
    quality_description: str
    accepted_source_kinds: tuple[Literal["video"], ...]
    supports_frame_range: bool
    supports_audio_range: bool
    overlap_unit: VideoContinuationOverlapUnit
    mapping_id: str
    mapping_version: int = Field(ge=1)

    @field_validator("quality_description")
    @classmethod
    def validate_quality_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("video continuation quality description is required")
        if len(normalized) > 320:
            raise ValueError("video continuation quality description is too long")
        return normalized

    @field_validator("mapping_id")
    @classmethod
    def validate_mapping_id(cls, value: str) -> str:
        return _validate_token(value, "video continuation mapping identifier")

    @field_validator("accepted_source_kinds")
    @classmethod
    def validate_source_kinds(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("video continuation source kinds must be non-empty and unique")
        return value


class VideoSequenceCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    methods: tuple[VideoContinuationMethodCapability, ...]

    @model_validator(mode="after")
    def validate_methods(self) -> "VideoSequenceCapability":
        method_ids = [item.method for item in self.methods]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("video continuation methods must be unique")
        return self


class MusicProductionCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_text_guidance: float = Field(ge=0, allow_inf_nan=False)
    text_guidance: CapabilityRange
    default_top_k: int = Field(ge=1)
    top_k: CapabilityRange
    default_steps: int = Field(ge=1)
    steps: CapabilityRange
    samplers: tuple[str, ...]
    default_sampler: str
    schedulers: tuple[str, ...]
    default_scheduler: str
    default_denoise: float = Field(ge=0, le=1, allow_inf_nan=False)
    denoise: CapabilityRange
    tiled_decode_supported: bool
    execution_profiles: tuple[Literal["automatic"], ...]
    default_execution_profile: Literal["automatic"]
    ksampler_cfg: float = Field(ge=0, allow_inf_nan=False)
    lora_supported: bool
    lora_weight: CapabilityRange | None = None

    @field_validator("samplers", "schedulers")
    @classmethod
    def validate_choices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("music production choices must be non-empty and unique")
        return tuple(_validate_token(item, "music production choice") for item in value)

    @model_validator(mode="after")
    def validate_contract(self) -> "MusicProductionCapability":
        if self.default_sampler not in self.samplers:
            raise ValueError("default music sampler must be advertised")
        if self.default_scheduler not in self.schedulers:
            raise ValueError("default music scheduler must be advertised")
        if self.default_execution_profile not in self.execution_profiles:
            raise ValueError("default music execution profile must be advertised")
        if self.execution_profiles != ("automatic",):
            raise ValueError("music execution profile must use automatic runtime management")
        if any(
            item.enforcement != "advisory"
            for item in (self.text_guidance, self.top_k, self.steps, self.denoise)
        ):
            raise ValueError("music production ranges must be advisory")
        if self.ksampler_cfg != 1.7:
            raise ValueError("Music3 KSampler CFG must remain mapping-owned at 1.7")
        if self.lora_supported and self.lora_weight is None:
            raise ValueError("Music LoRA support requires an advertised weight range")
        if not self.lora_supported and self.lora_weight is not None:
            raise ValueError("disabled Music LoRA support cannot advertise a weight range")
        return self


class VideoProductionCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_steps: int = Field(ge=1)
    steps: CapabilityRange
    samplers: tuple[str, ...]
    default_sampler: str
    schedulers: tuple[str, ...]
    default_scheduler: str
    default_video_shift: float = Field(ge=0, allow_inf_nan=False)
    video_shift: CapabilityRange
    default_audio_shift: float = Field(ge=0, allow_inf_nan=False)
    audio_shift: CapabilityRange
    lora_supported: bool
    lora_weight: CapabilityRange | None = None
    default_batch_count: int = Field(default=1, ge=1)
    batch_count: CapabilityRange

    @field_validator("samplers", "schedulers")
    @classmethod
    def validate_choices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("video production choices must be non-empty and unique")
        return tuple(_validate_token(item, "video production choice") for item in value)

    @model_validator(mode="after")
    def validate_defaults(self) -> "VideoProductionCapability":
        if self.default_sampler not in self.samplers:
            raise ValueError("default video sampler must be advertised")
        if self.default_scheduler not in self.schedulers:
            raise ValueError("default video scheduler must be advertised")
        if self.steps.enforcement != "advisory":
            raise ValueError("video steps range must be advisory")
        if self.video_shift.enforcement != "advisory" or self.audio_shift.enforcement != "advisory":
            raise ValueError("H3 shift ranges must be advisory")
        if self.batch_count.enforcement != "advisory":
            raise ValueError("video batch range must be advisory")
        if self.lora_supported and self.lora_weight is None:
            raise ValueError("LoRA support requires an advertised weight range")
        if not self.lora_supported and self.lora_weight is not None:
            raise ValueError("disabled LoRA support cannot advertise a weight range")
        return self


class ProviderTaskCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    supported: bool
    reason_code: str | None = None
    supports_negative_prompt: bool = False
    audio_output: AudioOutputSemantics = "provider_defined"
    supports_audio_prompt: bool = True
    reference_kinds: tuple[Literal["image", "video", "audio"], ...] = ()
    reference_roles: tuple[Literal["identity", "style", "motion", "camera", "voice", "other"], ...] = ()
    provider_hard_reference_limits: dict[Literal["image", "video", "audio"], int] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validate_token(value, "task identifier")

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        return _validate_token(value, "task reason code") if value is not None else None

    @field_validator("reference_kinds", "reference_roles")
    @classmethod
    def validate_unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("capability values must be unique")
        return value

    @field_validator("provider_hard_reference_limits")
    @classmethod
    def validate_reference_limits(cls, value: dict[str, int]) -> dict[str, int]:
        if any(limit < 1 for limit in value.values()):
            raise ValueError("provider reference limits must be positive")
        return value

    @model_validator(mode="after")
    def validate_support(self) -> "ProviderTaskCapability":
        if not self.supported and self.reason_code is None:
            raise ValueError("unsupported task requires a reason code")
        if self.supported and self.reason_code is not None:
            raise ValueError("supported task must not carry a reason code")
        return self


class SetupRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    kind: SetupKind
    ready: bool
    reason_code: str | None = None

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        return _validate_token(value, "setup requirement identifier")

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        return _validate_token(value, "setup reason code") if value is not None else None

    @model_validator(mode="after")
    def validate_readiness(self) -> "SetupRequirement":
        if self.ready and self.reason_code is not None:
            raise ValueError("ready setup requirement must not carry a reason code")
        if not self.ready and self.reason_code is None:
            raise ValueError("unready setup requirement requires a reason code")
        return self


class OutputCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["video", "audio"]
    mime_types: tuple[str, ...]

    @field_validator("mime_types")
    @classmethod
    def validate_mime_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)) or any("/" not in item for item in value):
            raise ValueError("output MIME types must be non-empty and unique")
        return value


class BillingCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: BillingStatus
    confirmation_required: bool

    @model_validator(mode="after")
    def validate_billing(self) -> "BillingCapability":
        if self.status == "local_no_api_billing" and self.confirmation_required:
            raise ValueError("local provider billing cannot require paid confirmation")
        if self.status == "provider_billed" and not self.confirmation_required:
            raise ValueError("billed provider requires explicit confirmation")
        return self


class ProviderCapabilityDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.provider.capability.v1"] = PROVIDER_CAPABILITY_SCHEMA_VERSION
    domain: CapabilityDomain
    provider_id: str
    adapter_id: str
    adapter_revision: str
    model_profile: str
    attribution: str | None = None
    availability: CapabilityAvailability
    availability_reason_code: str | None = None
    tasks: tuple[ProviderTaskCapability, ...]
    duration: DurationCapability
    size: SizeCapability | None = None
    video_sequence: VideoSequenceCapability | None = None
    video_production: VideoProductionCapability | None = None
    music: MusicDomainCapability | None = None
    music_production: MusicProductionCapability | None = None
    required_setup: tuple[SetupRequirement, ...] = ()
    progress_support: ProgressSupport
    cancel_support: bool
    outputs: tuple[OutputCapability, ...]
    resource_estimate_inputs: tuple[str, ...] = ()
    provider_hard_parameters: dict[str, CapabilityRange] = Field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    billing: BillingCapability

    @field_validator("provider_id", "adapter_id", "adapter_revision", "model_profile")
    @classmethod
    def validate_identifiers(cls, value: str) -> str:
        return _validate_token(value, "capability identifier")

    @field_validator("attribution")
    @classmethod
    def validate_attribution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or len(normalized) > 192:
            raise ValueError("capability attribution is invalid")
        return normalized

    @field_validator("availability_reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        return _validate_token(value, "availability reason code") if value is not None else None

    @field_validator("resource_estimate_inputs", "reason_codes")
    @classmethod
    def validate_estimate_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("capability tokens must be unique")
        return tuple(_validate_token(item, "capability token") for item in value)

    @field_validator("provider_hard_parameters")
    @classmethod
    def validate_hard_parameters(
        cls,
        value: dict[str, CapabilityRange],
    ) -> dict[str, CapabilityRange]:
        for key, hard_range in value.items():
            _validate_token(key, "provider hard parameter")
            if hard_range.enforcement != "provider_hard_limit":
                raise ValueError("provider hard parameter must use provider_hard_limit enforcement")
        return value

    @model_validator(mode="after")
    def validate_descriptor(self) -> "ProviderCapabilityDescriptor":
        if self.availability == "ready" and self.availability_reason_code is not None:
            raise ValueError("ready provider must not carry a reason code")
        if self.availability != "ready" and self.availability_reason_code is None:
            raise ValueError("unavailable provider requires a reason code")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task identifiers must be unique")
        output_keys = [(output.kind, output.mime_types) for output in self.outputs]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("output capabilities must be unique")
        setup_ids = [requirement.requirement_id for requirement in self.required_setup]
        if len(setup_ids) != len(set(setup_ids)):
            raise ValueError("setup requirement identifiers must be unique")
        if self.domain == "video":
            if self.size is None:
                raise ValueError("video provider requires size capability")
            if self.music is not None:
                raise ValueError("video provider cannot declare music capability")
            if self.music_production is not None:
                raise ValueError("video provider cannot declare music production capability")
            sequence_task = next((task for task in self.tasks if task.task_id == "sequence"), None)
            if sequence_task is not None and sequence_task.supported:
                if self.video_sequence is None or not self.video_sequence.methods:
                    raise ValueError("supported video sequence task requires continuation methods")
            elif self.video_sequence is not None:
                raise ValueError("video sequence capability requires a supported sequence task")
            if any(output.kind != "video" for output in self.outputs):
                raise ValueError("video provider outputs must be video artifacts")
        else:
            if self.music is None:
                raise ValueError("music provider requires music capability")
            if self.size is not None:
                raise ValueError("music provider cannot declare size capability")
            if self.video_sequence is not None:
                raise ValueError("music provider cannot declare video sequence capability")
            if self.video_production is not None:
                raise ValueError("music provider cannot declare video production capability")
            lyric_task_ids = {item.task_id for item in self.music.lyrics}
            if lyric_task_ids != set(task_ids):
                raise ValueError("music lyrics capabilities must match music tasks")
            if any(output.kind != "audio" for output in self.outputs):
                raise ValueError("music provider outputs must be audio artifacts")
            if any(
                task.reference_kinds or task.reference_roles or task.provider_hard_reference_limits
                for task in self.tasks
            ):
                raise ValueError("music provider tasks cannot declare video references")
        if self.provider_id.startswith("local_"):
            if (
                self.duration.provider_hard_limit is not None
                or (self.size is not None and self.size.provider_hard_resolutions)
                or self.provider_hard_parameters
            ):
                raise ValueError("local providers cannot declare provider hard limits")
            if any(task.provider_hard_reference_limits for task in self.tasks):
                raise ValueError("local providers cannot declare provider hard limits")
        return self


def capability_after_verified_setup(
    capability: ProviderCapabilityDescriptor,
) -> ProviderCapabilityDescriptor:
    """Return the immutable capability state proven by a successful setup gate."""

    return capability.model_copy(
        update={
            "availability": "ready",
            "availability_reason_code": None,
            "required_setup": tuple(
                requirement.model_copy(update={"ready": True, "reason_code": None})
                for requirement in capability.required_setup
            ),
        }
    )
