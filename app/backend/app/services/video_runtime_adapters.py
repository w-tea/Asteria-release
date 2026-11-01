from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.schemas.provider_capabilities import (
    BillingCapability,
    CapabilityRange,
    DurationCapability,
    OutputCapability,
    ProviderCapabilityDescriptor,
    ProviderTaskCapability,
    SetupRequirement,
    SizeCapability,
    VideoContinuationMethodCapability,
    VideoSequenceCapability,
    VideoProductionCapability,
)
from app.schemas.video import VideoRequest
from app.services.video_model_profiles import (
    VideoModelProfile,
    VideoProductionRange,
    load_default_video_model_profile,
)


VIDEO_MODES = ("text", "image", "first_last", "reference", "sequence")
BASE_VIDEO_MODES = frozenset({"text", "image", "first_last", "reference", "sequence"})
SEQUENCE_RUNTIME_DEPENDENCY_IDS = frozenset(
    {
        "comfyui-h3-motion-context",
        "asteria-h3-latent-assembly",
    }
)


class VideoAdapterError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class VideoRuntimeAdapter(Protocol):
    id: str
    revision: str
    provider_id: str
    mapping_id: str
    sequence_mapping_id: str
    sequence_initial_workflow_id: str

    def capabilities(self) -> ProviderCapabilityDescriptor: ...

    def validate_request(self, request: VideoRequest) -> None: ...

    def sequence_capabilities(self) -> VideoSequenceCapability: ...


class RuntimeDependencyStatusView(Protocol):
    id: str
    status: str


class LocalH3VideoAdapter:
    id = "minimax_h3_local"
    revision = "asteria.video.local-h3.2026-08-24.2"
    provider_id = "local_h3"
    mapping_id = "minimax-h3-local.v1"
    sequence_mapping_id = "minimax-h3-local-sequence.v1"
    sequence_initial_workflow_id = "asteria_minimax_h3_sequence_initial.v1"

    def __init__(self, model_profile: VideoModelProfile | None = None) -> None:
        self._model_profile = model_profile or load_default_video_model_profile()

    def capabilities(self) -> ProviderCapabilityDescriptor:
        production = self._model_profile.production
        return ProviderCapabilityDescriptor(
            domain="video",
            provider_id=self.provider_id,
            adapter_id=self.id,
            adapter_revision=self.revision,
            model_profile=self._model_profile.runtime_profile_id,
            availability="setup_required",
            availability_reason_code="video_local_setup_required",
            tasks=_task_capabilities(BASE_VIDEO_MODES),
            duration=DurationCapability(
                default_seconds=5,
                alignment="adapter_defined",
                validated_range=CapabilityRange(
                    minimum=5,
                    maximum=15,
                    enforcement="advisory",
                ),
            ),
            size=SizeCapability(
                default_width=832,
                default_height=480,
                multiple=32,
                validated_megapixels=CapabilityRange(maximum=1.0, enforcement="advisory"),
            ),
            video_sequence=self.sequence_capabilities(),
            video_production=VideoProductionCapability(
                default_steps=production.default_steps,
                steps=_advisory_range(production.steps),
                samplers=production.samplers,
                default_sampler=production.default_sampler,
                schedulers=production.schedulers,
                default_scheduler=production.default_scheduler,
                default_video_shift=production.default_video_shift,
                video_shift=_advisory_range(production.video_shift),
                default_audio_shift=production.default_audio_shift,
                audio_shift=_advisory_range(production.audio_shift),
                lora_supported=production.lora_supported,
                lora_weight=(
                    _advisory_range(production.lora_weight)
                    if production.lora_weight is not None
                    else None
                ),
                default_batch_count=production.default_batch_count,
                batch_count=_advisory_range(production.batch_count),
            ),
            required_setup=(
                SetupRequirement(
                    requirement_id="runtime:comfy-0.30",
                    kind="runtime",
                    ready=False,
                    reason_code="video_local_runtime_required",
                ),
                SetupRequirement(
                    requirement_id="model:minimax-h3-shared",
                    kind="model",
                    ready=False,
                    reason_code="video_local_shared_models_required",
                ),
                SetupRequirement(
                    requirement_id="model:minimax-h3-task",
                    kind="model",
                    ready=False,
                    reason_code="video_local_task_model_required",
                ),
                SetupRequirement(
                    requirement_id="license:minimax-h3-2026-08-02",
                    kind="license",
                    ready=False,
                    reason_code="video_local_license_required",
                ),
            ),
            progress_support="determinate",
            cancel_support=True,
            outputs=(OutputCapability(kind="video", mime_types=("video/mp4",)),),
            resource_estimate_inputs=(
                "duration",
                "resolution",
                "fps",
                "references",
                "audio",
                "precision",
                "offload",
            ),
            reason_codes=(
                "video_task_not_advertised",
                "video_negative_prompt_unsupported",
                "video_reference_kind_unsupported",
                "video_audio_disable_unsupported",
            ),
            billing=BillingCapability(
                status="local_no_api_billing",
                confirmation_required=False,
            ),
        )

    def validate_request(self, request: VideoRequest) -> None:
        _validate_request_against_capability(request, self.capabilities())

    def sequence_capabilities(self) -> VideoSequenceCapability:
        return VideoSequenceCapability(
            methods=(
                VideoContinuationMethodCapability(
                    method="direct_context",
                    quality_description=(
                        "Pins the accepted clip's exact H3 video and audio latent tail; "
                        "the overlapping head is removed from both streams once."
                    ),
                    accepted_source_kinds=("video",),
                    supports_frame_range=True,
                    supports_audio_range=True,
                    overlap_unit="frames",
                    mapping_id=self.sequence_mapping_id,
                    mapping_version=1,
                ),
                VideoContinuationMethodCapability(
                    method="reference_continuation",
                    quality_description=(
                        "Uses the accepted clip as synchronized video and audio guidance; "
                        "continuity is guided rather than frame-anchored."
                    ),
                    accepted_source_kinds=("video",),
                    supports_frame_range=True,
                    supports_audio_range=True,
                    overlap_unit="none",
                    mapping_id=self.sequence_mapping_id,
                    mapping_version=1,
                ),
            )
        )


def default_video_adapters() -> tuple[VideoRuntimeAdapter, ...]:
    return (LocalH3VideoAdapter(),)


def video_runtime_dependencies_ready(
    request: VideoRequest,
    statuses: Iterable[RuntimeDependencyStatusView],
) -> bool:
    if request.mode != "sequence":
        return True
    installed = {status.id for status in statuses if status.status == "installed"}
    return SEQUENCE_RUNTIME_DEPENDENCY_IDS.issubset(installed)


def _advisory_range(value: VideoProductionRange) -> CapabilityRange:
    return CapabilityRange(
        minimum=value.minimum,
        maximum=value.maximum,
        step=value.step,
        enforcement="advisory",
    )


def _task_capabilities(supported: frozenset[str]) -> tuple[ProviderTaskCapability, ...]:
    tasks: list[ProviderTaskCapability] = []
    for mode in VIDEO_MODES:
        is_supported = mode in supported
        if mode == "sequence" and not is_supported:
            reason = "video_sequence_not_implemented"
        elif not is_supported:
            reason = "video_task_not_advertised"
        else:
            reason = None
        reference_kinds: tuple[str, ...] = ()
        reference_roles: tuple[str, ...] = ()
        if mode == "reference":
            reference_kinds = ("image", "video", "audio")
            reference_roles = ("identity", "style", "motion", "camera", "voice", "other")
        elif mode == "sequence":
            reference_kinds = ("video",)
            reference_roles = ("motion",)
        tasks.append(
            ProviderTaskCapability(
                task_id=mode,
                supported=is_supported,
                reason_code=reason,
                supports_negative_prompt=False,
                audio_output="always" if is_supported else "none",
                supports_audio_prompt=is_supported,
                reference_kinds=reference_kinds,
                reference_roles=reference_roles,
                provider_hard_reference_limits={},
            )
        )
    return tuple(tasks)


def _validate_request_against_capability(
    request: VideoRequest,
    capability: ProviderCapabilityDescriptor,
) -> None:
    if request.provider != capability.provider_id:
        raise VideoAdapterError("video_provider_mismatch")
    task = next((item for item in capability.tasks if item.task_id == request.mode), None)
    if task is None or not task.supported:
        raise VideoAdapterError(task.reason_code if task is not None else "video_task_not_advertised")
    if request.negative_prompt and not task.supports_negative_prompt:
        raise VideoAdapterError("video_negative_prompt_unsupported")
    if request.audio.prompt and not task.supports_audio_prompt:
        raise VideoAdapterError("video_audio_prompt_unsupported")
    if task.audio_output == "always" and not request.audio.enabled:
        raise VideoAdapterError("video_audio_disable_unsupported")
    if task.audio_output == "none" and request.audio.enabled:
        raise VideoAdapterError("video_audio_generation_unsupported")
    if request.references:
        if any(reference.kind not in task.reference_kinds for reference in request.references):
            raise VideoAdapterError("video_reference_kind_unsupported")
        counts: dict[str, int] = {}
        for reference in request.references:
            counts[reference.kind] = counts.get(reference.kind, 0) + 1
        if any(counts.get(kind, 0) > limit for kind, limit in task.provider_hard_reference_limits.items()):
            raise VideoAdapterError("video_provider_reference_limit")
    hard_duration = capability.duration.provider_hard_limit
    if hard_duration is not None:
        if hard_duration.minimum is not None and request.duration_seconds < hard_duration.minimum:
            raise VideoAdapterError("video_provider_duration_limit")
        if hard_duration.maximum is not None and request.duration_seconds > hard_duration.maximum:
            raise VideoAdapterError("video_provider_duration_limit")
    if capability.size.provider_hard_resolutions:
        resolution = {
            "preview": "768P",
            "quality": "2K",
        }.get(request.size.resolution_intent)
        if resolution not in capability.size.provider_hard_resolutions:
            raise VideoAdapterError("video_provider_resolution_limit")
    parameter_values = {"seed": request.seed}
    for name, hard_range in capability.provider_hard_parameters.items():
        value = parameter_values.get(name)
        if value is None:
            continue
        if hard_range.minimum is not None and value < hard_range.minimum:
            raise VideoAdapterError(f"video_provider_{name}_limit")
        if hard_range.maximum is not None and value > hard_range.maximum:
            raise VideoAdapterError(f"video_provider_{name}_limit")
    if request.sampling is not None:
        production = capability.video_production
        if production is None:
            raise VideoAdapterError("video_production_controls_unsupported")
        if request.sampling.sampler not in production.samplers:
            raise VideoAdapterError("video_sampler_unsupported")
        if request.sampling.scheduler not in production.schedulers:
            raise VideoAdapterError("video_scheduler_unsupported")
    if request.loras and (
        capability.video_production is None
        or not capability.video_production.lora_supported
    ):
        raise VideoAdapterError("video_lora_unsupported")
