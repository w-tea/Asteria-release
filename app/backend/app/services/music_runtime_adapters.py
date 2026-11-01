from __future__ import annotations

from typing import Protocol

from app.schemas.music import MusicRequest
from app.schemas.provider_capabilities import (
    BillingCapability,
    CapabilityRange,
    DurationCapability,
    MusicDomainCapability,
    MusicLyricsCapability,
    MusicProductionCapability,
    OutputCapability,
    ProviderCapabilityDescriptor,
    ProviderTaskCapability,
    SetupRequirement,
)


class MusicAdapterError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class MusicRuntimeAdapter(Protocol):
    id: str
    revision: str
    provider_id: str
    mapping_id: str

    def capabilities(self) -> ProviderCapabilityDescriptor: ...
    def validate_request(self, request: MusicRequest) -> None: ...


class LocalMusic3Adapter:
    id = "minimax_music3_local"
    revision = "asteria.music.local-music3.2026-08-18.1"
    provider_id = "local_music3"
    mapping_id = "minimax-music3-local.v1"

    def capabilities(self) -> ProviderCapabilityDescriptor:
        return ProviderCapabilityDescriptor(
            domain="music",
            provider_id=self.provider_id,
            adapter_id=self.id,
            adapter_revision=self.revision,
            model_profile="minimax-music3",
            attribution="MiniMax-Music3",
            availability="setup_required",
            availability_reason_code="music_local_setup_required",
            tasks=(
                ProviderTaskCapability(task_id="vocal", supported=True, audio_output="always", supports_audio_prompt=False),
                ProviderTaskCapability(task_id="instrumental", supported=True, audio_output="always", supports_audio_prompt=False),
            ),
            duration=DurationCapability(
                default_seconds=120,
                alignment="adapter_defined",
                validated_range=CapabilityRange(minimum=0.04, maximum=360, step=0.04, enforcement="advisory"),
            ),
            music=MusicDomainCapability(
                lyrics=(
                    MusicLyricsCapability(task_id="vocal", behavior="optional"),
                    MusicLyricsCapability(task_id="instrumental", behavior="forbidden"),
                ),
                language_intent="unsupported",
                duration_intent_supported=True,
            ),
            music_production=MusicProductionCapability(
                default_text_guidance=1.7,
                text_guidance=CapabilityRange(
                    minimum=0,
                    maximum=100,
                    step=0.1,
                    enforcement="advisory",
                ),
                default_top_k=50,
                top_k=CapabilityRange(
                    minimum=1,
                    maximum=16384,
                    step=1,
                    enforcement="advisory",
                ),
                default_steps=30,
                steps=CapabilityRange(
                    minimum=1,
                    maximum=1000,
                    step=1,
                    enforcement="advisory",
                ),
                samplers=("euler", "euler_ancestral"),
                default_sampler="euler",
                schedulers=("simple", "normal", "karras", "sgm_uniform"),
                default_scheduler="simple",
                default_denoise=1.0,
                denoise=CapabilityRange(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    enforcement="advisory",
                ),
                tiled_decode_supported=True,
                execution_profiles=("automatic",),
                default_execution_profile="automatic",
                ksampler_cfg=1.7,
                lora_supported=False,
            ),
            required_setup=(
                SetupRequirement(requirement_id="runtime:comfy", kind="runtime", ready=False, reason_code="music_local_runtime_required"),
                SetupRequirement(requirement_id="model:minimax-music3", kind="model", ready=False, reason_code="music_local_model_required"),
                SetupRequirement(requirement_id="license:minimax-music3", kind="license", ready=False, reason_code="music_local_license_required"),
            ),
            progress_support="determinate",
            cancel_support=True,
            outputs=(OutputCapability(kind="audio", mime_types=("audio/flac",)),),
            resource_estimate_inputs=("duration", "mode", "execution_profile", "steps"),
            reason_codes=("music_task_not_advertised", "music_lyrics_forbidden", "music_language_intent_unsupported", "music_lora_unsupported", "music_sampling_unsupported"),
            billing=BillingCapability(status="local_no_api_billing", confirmation_required=False),
        )

    def validate_request(self, request: MusicRequest) -> None:
        if request.provider != self.provider_id:
            raise MusicAdapterError("music_provider_mismatch")
        task = next((item for item in self.capabilities().tasks if item.task_id == request.mode), None)
        if task is None or not task.supported:
            raise MusicAdapterError("music_task_not_advertised")
        if request.mode == "instrumental" and request.lyrics is not None:
            raise MusicAdapterError("music_lyrics_forbidden")
        if request.language_intent is not None:
            raise MusicAdapterError("music_language_intent_unsupported")
        if request.schema_version == "asteria.music.request.v2":
            production = self.capabilities().music_production
            sampling = request.sampling
            if production is None or sampling is None:
                raise MusicAdapterError("music_sampling_unsupported")
            if sampling.sampler not in production.samplers or sampling.scheduler not in production.schedulers:
                raise MusicAdapterError("music_sampling_unsupported")
            if request.loras and not production.lora_supported:
                raise MusicAdapterError("music_lora_unsupported")


def default_music_adapters() -> tuple[MusicRuntimeAdapter, ...]:
    return (LocalMusic3Adapter(),)
