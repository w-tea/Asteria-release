from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import secrets
from typing import Callable
from uuid import uuid4

from app.schemas.generation_working_artifact import (
    GenerationWorkingArtifactRecord,
    GenerationWorkingArtifactSummary,
)
from app.schemas.media import MediaArtifact
from app.schemas.provider_capabilities import capability_after_verified_setup
from app.schemas.operations import OperationRecord
from app.schemas.resource_advisory import (
    ResourceAdvisoryAcknowledgement,
    ResourceAdvisoryAuthorization,
    ResourceObservationSnapshot,
    ResourceWorkload,
)
from app.schemas.video import VideoRequest
from app.services.video_sequence_context import VideoSequenceContextReceipt
from app.services.atomic_json import write_json_atomic
from app.services.generation_artifacts import finalize_non_image_output
from app.services.generation_working_artifacts import (
    GenerationWorkingArtifactError,
    GenerationWorkingArtifactRegistry,
)
from app.services.history import HistoryRepository
from app.services.media_artifacts import MediaFinalizationResult
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)
from app.services.operation_repository import OperationNotFoundError
from app.services.resource_advisor import ResourceAdvisoryError, ResourceAdvisor
from app.services.video_frame_policy import normalized_h3_frame_count
from app.services.video_local_adapter import (
    LocalH3VideoExecutionAdapter,
    LocalVideoExecutionResult,
    VideoLocalAdapterError,
)
from app.services.video_packet import VIDEO_PACKET_VERSION, VideoPacketBuilder, VideoSubmissionPacket
from app.services.video_runtime_adapters import LocalH3VideoAdapter
from app.services.video_upscale_bundles import (
    VideoUpscaleBundleError,
    stage_video_upscale_bundle,
)


VIDEO_OPERATION_KIND = "video.generate"
VIDEO_OPERATION_CONTRACT_VERSION = 1
VIDEO_BATCH_CONTRACT_VERSION = 1
MAX_VIDEO_SEED = 9223372036854775807


@dataclass(frozen=True, slots=True)
class SubmittedVideoOperation:
    operation: OperationRecord
    generation_id: int


class VideoOperationService:
    def __init__(
        self,
        *,
        coordinator: OperationCoordinator,
        history: HistoryRepository,
        packet_builder: VideoPacketBuilder | None = None,
        readiness_check: Callable[[VideoRequest], bool] | None = None,
        resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
        resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
        output_root_provider: Callable[[], Path] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.history = history
        self.packet_builder = packet_builder or VideoPacketBuilder()
        self.readiness_check = readiness_check
        self.resource_advisor_provider = resource_advisor_provider or ResourceAdvisor
        self.resource_observation_provider = resource_observation_provider or ResourceObservationSnapshot
        self.output_root_provider = output_root_provider

    def submit(
        self,
        request: VideoRequest,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ) -> SubmittedVideoOperation:
        if request.provider != "local_h3":
            raise VideoLocalAdapterError("video_provider_not_implemented")
        setup_verified = self.readiness_check is not None
        if setup_verified and not self.readiness_check(request):
            raise VideoLocalAdapterError("video_local_setup_required")
        resource_authorization = self._validate_resource_authorization(
            request,
            resource_authorization,
            run_anyway=run_anyway,
        )
        authorization_payload = (
            resource_authorization.model_dump(mode="json")
            if resource_authorization is not None
            else None
        )
        adapter = LocalH3VideoAdapter()
        capability = adapter.capabilities()
        if setup_verified:
            capability = capability_after_verified_setup(capability)
        packet = self.packet_builder.build(request, adapter, capability=capability)
        candidate_packets = self._candidate_packets(request, packet, adapter, capability)
        batch_id = str(uuid4()) if len(candidate_packets) > 1 else None
        requested_frames, actual_frames = normalized_h3_frame_count(
            float(packet.request_snapshot["duration_seconds"]),
            float(packet.request_snapshot["fps"]),
        )
        generation_ids: list[int] = []
        candidate_snapshots: list[dict] = []
        operation_output_root = (
            Path(self.output_root_provider()).expanduser().resolve()
            if self.output_root_provider is not None
            else None
        )
        for index, candidate_packet in enumerate(candidate_packets):
            candidate_request = VideoRequest.model_validate(candidate_packet.request_snapshot)
            settings = self._history_settings(
                candidate_request,
                candidate_packet,
                requested_frames=requested_frames,
                actual_frames=actual_frames,
                resource_authorization=authorization_payload,
                batch=(
                    {
                        "batch_id": batch_id,
                        "index": index,
                        "count": len(candidate_packets),
                        "seed_policy": request.sampling.seed_policy,
                        "seed": candidate_request.seed,
                    }
                    if batch_id is not None and request.sampling is not None
                    else None
                ),
                output_root=operation_output_root,
            )
            generation_id = self.history.add(
                status="queued",
                profile=candidate_request.model_profile,
                prompt=candidate_request.prompt,
                expanded_prompt=candidate_request.prompt,
                negative_prompt=candidate_request.negative_prompt or "",
                settings=settings,
                model_slots={},
                loras=[],
                embeddings=[],
                output_paths=[],
                media_kind="video",
                artifacts=[],
            )
            generation_ids.append(generation_id)
            candidate_snapshots.append(
                {
                    "index": index,
                    "generation_id": generation_id,
                    "seed": candidate_request.seed,
                    "packet": candidate_packet.to_dict(),
                }
            )
        generation_id = generation_ids[0]
        operation_snapshot = {
            "generation_id": generation_id,
            "packet": candidate_packets[0].to_dict(),
            "resource_authorization": authorization_payload,
        }
        if operation_output_root is not None:
            operation_snapshot["output_root"] = str(operation_output_root)
        if batch_id is not None:
            operation_snapshot["batch"] = {
                "contract_version": VIDEO_BATCH_CONTRACT_VERSION,
                "batch_id": batch_id,
                "count": len(candidate_packets),
                "seed_policy": request.sampling.seed_policy if request.sampling else "increment",
                "candidates": candidate_snapshots,
            }
        try:
            operation = self.coordinator.enqueue(
                kind=VIDEO_OPERATION_KIND,
                source_actor=source_actor,
                request_snapshot=operation_snapshot,
                request_contract_version=VIDEO_OPERATION_CONTRACT_VERSION,
                progress_total=actual_frames * len(candidate_packets),
                progress_unit="frames",
            )
        except Exception:
            for failed_generation_id in generation_ids:
                item = self.history.get(failed_generation_id)
                if item is not None:
                    self.history.update_generation(
                        failed_generation_id,
                        status="failed",
                        settings={**item["settings"], "error_code": "video_operation_enqueue_failed"},
                        output_paths=[],
                    )
            raise
        return SubmittedVideoOperation(operation=operation, generation_id=generation_id)

    def _candidate_packets(
        self,
        request: VideoRequest,
        resolved_packet: VideoSubmissionPacket,
        adapter: LocalH3VideoAdapter,
        capability,
    ) -> tuple[VideoSubmissionPacket, ...]:
        if request.schema_version != "asteria.video.request.v2" or request.sampling is None:
            return (resolved_packet,)
        count = request.sampling.batch_count
        if count == 1:
            candidate_request = VideoRequest.model_validate(resolved_packet.request_snapshot)
            candidate_sampling = candidate_request.sampling.model_copy(update={"batch_count": 1})
            return (
                self.packet_builder.build(
                    candidate_request.model_copy(update={"sampling": candidate_sampling}),
                    adapter,
                    capability=capability,
                ),
            )
        base_seed = int(resolved_packet.request_snapshot["seed"])
        seeds = self._candidate_seeds(base_seed, count, request.sampling.seed_policy)
        packets: list[VideoSubmissionPacket] = []
        for seed in seeds:
            candidate_request = VideoRequest.model_validate(resolved_packet.request_snapshot)
            candidate_sampling = candidate_request.sampling.model_copy(update={"batch_count": 1})
            packets.append(
                self.packet_builder.build(
                    candidate_request.model_copy(
                        update={"seed": seed, "sampling": candidate_sampling}
                    ),
                    adapter,
                    capability=capability,
                )
            )
        return tuple(packets)

    @staticmethod
    def _candidate_seeds(base_seed: int, count: int, policy: str) -> tuple[int, ...]:
        if policy == "increment":
            return tuple((base_seed + index) % (MAX_VIDEO_SEED + 1) for index in range(count))
        seeds = [base_seed]
        while len(seeds) < count:
            candidate = secrets.randbelow(MAX_VIDEO_SEED + 1)
            if candidate not in seeds:
                seeds.append(candidate)
        return tuple(seeds)

    @staticmethod
    def _history_settings(
        request: VideoRequest,
        packet: VideoSubmissionPacket,
        *,
        requested_frames: int,
        actual_frames: int,
        resource_authorization: dict | None,
        batch: dict | None,
        output_root: Path | None,
    ) -> dict:
        settings = {
            "schema_version": "asteria.video.history.v1",
            "request_snapshot": packet.request_snapshot,
            "capability_snapshot": packet.capability_snapshot,
            "provider": request.provider,
            "adapter_id": packet.adapter_id,
            "adapter_revision": packet.adapter_revision,
            "mapping_id": packet.mapping_id,
            "mapping_version": packet.mapping_version,
            "submission_runtime_job_id": packet.runtime_job_id,
            "requested_frame_count": requested_frames,
            "actual_frame_count": actual_frames,
            "resource_authorization": resource_authorization,
        }
        if batch is not None:
            settings["batch"] = batch
        if output_root is not None:
            settings["output_root"] = str(output_root)
        return settings

    def _validate_resource_authorization(
        self,
        request: VideoRequest,
        authorization: ResourceAdvisoryAuthorization | None,
        *,
        run_anyway: bool = False,
        workload: ResourceWorkload | None = None,
    ) -> ResourceAdvisoryAuthorization | None:
        advisor = self.resource_advisor_provider()
        try:
            current = advisor.advise(
                workload or _resource_workload(request),
                LocalH3VideoAdapter().capabilities(),
                self.resource_observation_provider(),
            )
        except ResourceAdvisoryError as error:
            mapped_code = {
                "resource_advisory_profile_mismatch": "video_model_profile_unsupported",
                "resource_advisory_task_unavailable": "video_task_not_advertised",
                "resource_advisory_provider_mismatch": "video_provider_mismatch",
            }.get(error.code, "video_resource_authorization_mismatch")
            raise VideoLocalAdapterError(mapped_code) from error
        if run_anyway:
            acknowledgement = (
                ResourceAdvisoryAcknowledgement(
                    advisory_fingerprint=current.fingerprint,
                    run_anyway=True,
                )
                if current.acknowledgement_required
                else None
            )
            return advisor.authorize(current, acknowledgement)
        if authorization is not None and authorization.advisory.fingerprint != current.fingerprint:
            raise VideoLocalAdapterError("video_resource_authorization_mismatch")
        try:
            verified = advisor.authorize(
                current,
                authorization.acknowledgement if authorization is not None else None,
            )
        except ResourceAdvisoryError as error:
            if error.code == "resource_advisory_acknowledgement_required":
                raise VideoLocalAdapterError("video_resource_acknowledgement_required") from error
            raise VideoLocalAdapterError("video_resource_authorization_mismatch") from error
        if authorization is None and not current.acknowledgement_required:
            return None
        return verified

    def validate_resource_authorization(
        self,
        request: VideoRequest,
        authorization: ResourceAdvisoryAuthorization | None,
        *,
        run_anyway: bool = False,
        workload: ResourceWorkload | None = None,
    ) -> ResourceAdvisoryAuthorization | None:
        """Validate a composed video operation against the shared advisory contract."""
        return self._validate_resource_authorization(
            request,
            authorization,
            run_anyway=run_anyway,
            workload=workload,
        )

    def retry(
        self,
        operation_id: str,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ) -> OperationRecord:
        operation = self.coordinator.get(operation_id)
        if operation.kind != "video.generate":
            raise OperationNotFoundError("operation_not_found")
        try:
            request = VideoRequest.model_validate(
                operation.request_snapshot["packet"]["request_snapshot"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VideoLocalAdapterError("video_request_snapshot_invalid") from error
        if self.readiness_check is not None and not self.readiness_check(request):
            raise VideoLocalAdapterError("video_local_setup_required")
        verified_authorization = self._validate_resource_authorization(
            request,
            resource_authorization,
            run_anyway=run_anyway,
        )
        authorization_payload = (
            verified_authorization.model_dump(mode="json")
            if verified_authorization is not None
            else None
        )
        request_snapshot = dict(operation.request_snapshot)
        request_snapshot["resource_authorization"] = authorization_payload
        generation_ids = self._snapshot_generation_ids(request_snapshot)
        originals: dict[int, dict] = {}
        for generation_id in generation_ids:
            item = self.history.get(generation_id)
            if item is None:
                raise VideoLocalAdapterError("video_history_not_found")
            originals[generation_id] = item
            if item["status"] == "completed":
                continue
            settings = dict(item["settings"])
            settings["resource_authorization"] = authorization_payload
            settings.pop("error_code", None)
            settings.pop("warning_codes", None)
            self.history.update_generation(
                generation_id,
                status="queued",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
        try:
            return self.coordinator.retry(
                operation_id,
                source_actor=source_actor,
                request_snapshot=request_snapshot,
            )
        except Exception:
            for generation_id, item in originals.items():
                self.history.update_generation(
                    generation_id,
                    status=item["status"],
                    settings=dict(item["settings"]),
                    output_paths=list(item["output_paths"]),
                )
            raise

    @staticmethod
    def _snapshot_generation_ids(request_snapshot: dict) -> tuple[int, ...]:
        batch = request_snapshot.get("batch")
        if isinstance(batch, dict) and isinstance(batch.get("candidates"), list):
            generation_ids = tuple(
                candidate.get("generation_id")
                for candidate in batch["candidates"]
                if isinstance(candidate, dict) and isinstance(candidate.get("generation_id"), int)
            )
            if generation_ids:
                return generation_ids
        generation_id = request_snapshot.get("generation_id")
        if not isinstance(generation_id, int) or generation_id < 1:
            raise VideoLocalAdapterError("video_request_snapshot_invalid")
        return (generation_id,)

    def cleanup_working_artifacts(self) -> None:
        roots: set[Path] = set()
        if self.output_root_provider is not None:
            try:
                roots.add(Path(self.output_root_provider()).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                pass
        for item in self.history.list_all():
            settings = item.get("settings")
            if not isinstance(settings, dict):
                continue
            value = settings.get("output_root")
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                roots.add(Path(value).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                continue

        def operation_status(operation_id: str) -> str | None:
            try:
                return self.coordinator.get(operation_id).status
            except OperationNotFoundError:
                return None

        for root in roots:
            try:
                GenerationWorkingArtifactRegistry(root).cleanup(
                    operation_status=operation_status
                )
            except GenerationWorkingArtifactError:
                continue


def _resource_workload(
    request: VideoRequest,
    *,
    guide_duration_seconds: float = 0,
) -> ResourceWorkload:
    precision = request.provider_options.get("precision", "auto")
    offload = request.provider_options.get("offload", "auto")
    if not isinstance(precision, str) or precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise VideoLocalAdapterError("video_provider_options_invalid")
    if not isinstance(offload, str) or offload not in {"auto", "none", "balanced", "maximum", "layer_streaming"}:
        raise VideoLocalAdapterError("video_provider_options_invalid")
    guide_count = len(request.references)
    if request.mode in {"image", "first_last"}:
        guide_count = int(request.first_frame_artifact_id is not None) + int(
            request.last_frame_artifact_id is not None
        )
    elif request.mode == "sequence":
        guide_count = 1
    return ResourceWorkload(
        domain="video",
        provider_id=request.provider,
        adapter_id=LocalH3VideoAdapter.id,
        model_profile=request.model_profile,
        task_id=request.mode,
        width=request.size.width,
        height=request.size.height,
        duration_seconds=request.duration_seconds,
        fps=request.fps,
        # Candidate clips are queued sequentially under one GPU-exclusive operation.
        batch=1,
        guide_count=guide_count,
        guide_duration_seconds=guide_duration_seconds,
        audio=request.audio.enabled,
        sequence_context=request.mode == "sequence",
        precision=precision,
        offload=offload,
    )


class VideoOperationExecutor:
    version = "video-local-h3-v1"

    def __init__(
        self,
        *,
        history: HistoryRepository,
        local_adapter: LocalH3VideoExecutionAdapter,
        poster_root: Path,
        finalizer: Callable[..., MediaFinalizationResult] = finalize_non_image_output,
        working_registry_factory: Callable[[Path], GenerationWorkingArtifactRegistry] = (
            GenerationWorkingArtifactRegistry
        ),
    ) -> None:
        self.history = history
        self.local_adapter = local_adapter
        self.poster_root = Path(poster_root)
        self.finalizer = finalizer
        self.working_registry_factory = working_registry_factory

    def resource_claims(self, _request_snapshot: dict) -> tuple[str, ...]:
        return ("comfy_gpu_exclusive",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        return self._execute(
            context.request_snapshot,
            context.operation.operation_id,
            progress=lambda current, total: context.report_progress(
                current,
                total,
                "frames",
                phase="generating",
            ),
            cancel_requested=lambda: context.cancel_requested,
        )

    def execute_for_test(self, request_snapshot: dict, operation_id: str) -> OperationExecutionResult:
        return self._execute(
            request_snapshot,
            operation_id,
            progress=lambda _current, _total: None,
            cancel_requested=lambda: False,
        )

    def _execute(
        self,
        request_snapshot: dict,
        operation_id: str,
        *,
        progress: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
    ) -> OperationExecutionResult:
        batch = request_snapshot.get("batch")
        if isinstance(batch, dict):
            return self._execute_batch(
                batch,
                operation_id,
                output_root=request_snapshot.get("output_root"),
                progress=progress,
                cancel_requested=cancel_requested,
            )
        generation_id, packet, sequence_metadata = self._validated_snapshot(request_snapshot)
        item = self.history.get(generation_id)
        if item is None:
            raise OperationExecutionError("video_history_not_found", retryable=False)
        if not self._history_matches_packet(item, packet):
            raise OperationExecutionError("video_request_snapshot_invalid", retryable=False)
        if item["status"] == "completed":
            return self._registered_result(generation_id, item)
        settings = dict(item["settings"])
        settings.pop("error_code", None)
        try:
            output_root = self._operation_output_root(request_snapshot, settings)
        except OperationExecutionError as error:
            settings["error_code"] = error.error_code
            self.history.update_generation(
                generation_id,
                status="failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            raise
        settings["output_root"] = str(output_root)
        self.history.update_generation(
            generation_id,
            status="running",
            settings=settings,
            output_paths=list(item["output_paths"]),
        )
        runtime_job_id = ""
        finalized_paths: list[Path] = []
        working_registry: GenerationWorkingArtifactRegistry | None = None
        registered_working: list[GenerationWorkingArtifactRecord] = []
        working_operation_id = str(request_snapshot.get("working_operation_id") or operation_id)
        try:
            execution = self.local_adapter.execute_video(
                packet,
                operation_id=operation_id,
                output_root=output_root,
                progress_callback=progress,
                cancel_requested=cancel_requested,
                runtime_job_callback=lambda value: self._record_runtime_job(
                    generation_id,
                    settings,
                    value,
                ),
            )
            runtime_job_id = execution.runtime_job_id
            finalized_paths = [
                *execution.output_paths,
                *(output.path for output in execution.working_outputs),
            ]
            self._raise_if_cancelled(cancel_requested, finalized_paths)
            artifacts, output_paths = self._finalize_outputs(
                generation_id,
                packet,
                execution,
                settings=settings,
                sequence_metadata=sequence_metadata,
                cancel_requested=cancel_requested,
            )
            finalized_paths = [
                *execution.output_paths,
                *(output.path for output in execution.working_outputs),
            ]
            finalized_paths.extend(Path(value) for value in output_paths)
            finalized_paths.extend(Path(artifact.path) for artifact in artifacts)
            primary = next(
                artifact
                for artifact in artifacts
                if artifact.kind == "video" and artifact.role == "primary"
            )
            if execution.working_outputs:
                working_registry = self.working_registry_factory(output_root)
                for working_output in execution.working_outputs:
                    registered_working.append(
                        working_registry.register_path(
                            operation_id=working_operation_id,
                            generation_id=generation_id,
                            media_artifact_id=primary.artifact_id,
                            kind=working_output.kind,
                            source_path=working_output.path,
                            source_sha256=working_output.source_sha256,
                            conditioning_sha256=working_output.conditioning_sha256,
                        )
                    )
                latent_records = [
                    record for record in registered_working if record.kind == "av_latent"
                ]
                if len(latent_records) == 1:
                    bundle, bundle_path = stage_video_upscale_bundle(
                        output_root=output_root,
                        operation_id=working_operation_id,
                        generation_id=generation_id,
                        packet=packet,
                        execution=execution,
                        primary=primary,
                        latent=latent_records[0],
                    )
                    finalized_paths.append(bundle_path)
                    registered_working.append(
                        working_registry.register_path(
                            operation_id=working_operation_id,
                            generation_id=generation_id,
                            media_artifact_id=primary.artifact_id,
                            kind="upscale_work_bundle",
                            source_path=bundle_path,
                            source_sha256=(primary.sha256 or "", latent_records[0].sha256),
                            conditioning_sha256=bundle.conditioning_sha256,
                        )
                    )
                    settings["upscale_work_bundle_id"] = registered_working[-1].artifact_id
                    settings["upscale_work_bundle_operation_id"] = working_operation_id
                settings["working_artifacts"] = [
                    self._working_summary(record).model_dump(mode="json")
                    for record in registered_working
                ]
            settings.update(
                {
                    "runtime_job_id": runtime_job_id,
                    "workflow_id": execution.workflow_id,
                    "requested_frame_count": execution.requested_frame_count,
                    "actual_frame_count": execution.actual_frame_count,
                    "actual_duration_seconds": execution.actual_duration_seconds,
                }
            )
            self._raise_if_cancelled(cancel_requested, finalized_paths)
            self.history.complete_media_generation(
                generation_id,
                status="completed",
                settings=settings,
                media_kind="video",
                artifacts=artifacts,
                output_paths=output_paths,
            )
            return self._artifact_result(
                generation_id,
                artifacts,
                working_artifact_ids=tuple(record.artifact_id for record in registered_working),
            )
        except VideoLocalAdapterError as error:
            self._discard_paths(finalized_paths)
            self._discard_working(working_registry, working_operation_id, registered_working)
            settings.update(
                {
                    "error_code": error.code,
                    "runtime_job_id": runtime_job_id or settings.get("runtime_job_id"),
                }
            )
            self.history.update_generation(
                generation_id,
                status="cancelled" if error.code == "video_generation_cancelled" else "failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            if error.code == "video_generation_cancelled":
                raise OperationCancelled() from error
            raise OperationExecutionError(error.code, retryable=error.retryable) from error
        except (GenerationWorkingArtifactError, VideoUpscaleBundleError, OSError, ValueError) as error:
            self._discard_paths(finalized_paths)
            self._discard_working(working_registry, working_operation_id, registered_working)
            settings.update(
                {
                    "error_code": (
                        "video_upscale_bundle_finalize_failed"
                        if isinstance(error, VideoUpscaleBundleError)
                        else "video_working_artifact_finalize_failed"
                        if isinstance(error, GenerationWorkingArtifactError)
                        else "video_artifact_finalize_failed"
                    ),
                    "runtime_job_id": runtime_job_id or settings.get("runtime_job_id"),
                }
            )
            self.history.update_generation(
                generation_id,
                status="failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            raise OperationExecutionError(
                settings["error_code"],
                retryable=True,
            ) from error

    def _execute_batch(
        self,
        batch: dict,
        operation_id: str,
        *,
        output_root: object,
        progress: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
    ) -> OperationExecutionResult:
        candidates = batch.get("candidates")
        if (
            batch.get("contract_version") != VIDEO_BATCH_CONTRACT_VERSION
            or not isinstance(candidates, list)
            or len(candidates) < 2
            or batch.get("count") != len(candidates)
        ):
            raise OperationExecutionError("video_request_snapshot_invalid", retryable=False)
        frame_totals: list[int] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise OperationExecutionError("video_request_snapshot_invalid", retryable=False)
            _generation_id, packet, _sequence = self._validated_snapshot(candidate)
            frame_totals.append(
                normalized_h3_frame_count(
                    float(packet.request_snapshot["duration_seconds"]),
                    float(packet.request_snapshot["fps"]),
                )[1]
            )
        total_frames = sum(frame_totals)
        completed_frames = 0
        artifact_refs: list[str] = []
        warning_codes: list[str] = []
        for index, candidate in enumerate(candidates):
            if cancel_requested():
                self._mark_unstarted_batch_candidates_cancelled(candidates[index:])
                raise OperationCancelled(artifact_refs=tuple(artifact_refs))
            try:
                candidate_snapshot = dict(candidate)
                if output_root is not None:
                    candidate_snapshot["output_root"] = output_root
                candidate_snapshot["working_operation_id"] = operation_id
                result = self._execute(
                    candidate_snapshot,
                    f"{operation_id}-{index + 1}",
                    progress=lambda current, _total, base=completed_frames: progress(
                        min(base + current, total_frames),
                        total_frames,
                    ),
                    cancel_requested=cancel_requested,
                )
            except OperationCancelled as error:
                self._mark_unstarted_batch_candidates_cancelled(candidates[index + 1:])
                raise OperationCancelled(
                    artifact_refs=tuple((*artifact_refs, *error.artifact_refs))
                ) from error
            except OperationExecutionError as error:
                self._mark_unstarted_batch_candidates_failed(candidates[index + 1:])
                raise OperationExecutionError(
                    error.error_code,
                    retryable=error.retryable,
                    artifact_refs=tuple((*artifact_refs, *error.artifact_refs)),
                ) from error
            artifact_refs.extend(result.artifact_refs)
            warning_codes.extend(result.warning_codes)
            completed_frames += frame_totals[index]
            progress(completed_frames, total_frames)
        return OperationExecutionResult(
            artifact_refs=tuple(dict.fromkeys(artifact_refs)),
            warning_codes=tuple(dict.fromkeys(warning_codes)),
        )

    def _mark_unstarted_batch_candidates_cancelled(self, candidates: list[dict]) -> None:
        for candidate in candidates:
            try:
                generation_id = int(candidate["generation_id"])
            except (KeyError, TypeError, ValueError):
                continue
            item = self.history.get(generation_id)
            if item is None or item["status"] == "completed":
                continue
            settings = dict(item["settings"])
            settings["error_code"] = "video_generation_cancelled"
            self.history.update_generation(
                generation_id,
                status="cancelled",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )

    def _mark_unstarted_batch_candidates_failed(self, candidates: list[dict]) -> None:
        for candidate in candidates:
            generation_id = candidate.get("generation_id") if isinstance(candidate, dict) else None
            if not isinstance(generation_id, int) or isinstance(generation_id, bool):
                continue
            item = self.history.get(generation_id)
            if item is None or item["status"] == "completed":
                continue
            settings = dict(item["settings"])
            settings["error_code"] = "video_batch_stopped_after_failure"
            self.history.update_generation(
                generation_id,
                status="failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )

    def _finalize_outputs(
        self,
        generation_id: int,
        packet: VideoSubmissionPacket,
        execution: LocalVideoExecutionResult,
        *,
        settings: dict,
        sequence_metadata: dict | None = None,
        cancel_requested: Callable[[], bool],
    ) -> tuple[list[MediaArtifact], list[str]]:
        artifacts: list[MediaArtifact] = []
        output_paths: list[str] = []
        created_paths: list[Path] = list(execution.output_paths)
        try:
            for source in execution.output_paths:
                self._raise_if_cancelled(cancel_requested, created_paths)
                finalized = self.finalizer(
                    generation_id=generation_id,
                    source_path=source,
                    poster_root=self.poster_root,
                )
                created_paths.extend(Path(value) for value in finalized.output_paths)
                created_paths.extend(Path(artifact.path) for artifact in finalized.artifacts)
                self._raise_if_cancelled(cancel_requested, created_paths)
                if finalized.media_kind != "video":
                    raise ValueError("video_artifact_kind_invalid")
                finalized_artifacts = list(finalized.artifacts)
                if sequence_metadata is not None:
                    parent_id = sequence_metadata.get("parent_artifact_id")
                    finalized_artifacts = [
                        artifact.model_copy(
                            update={
                                "sequence_id": sequence_metadata["sequence_id"],
                                "segment_id": sequence_metadata["segment_id"],
                                "parent_artifact_ids": (
                                    (*artifact.parent_artifact_ids, parent_id)
                                    if parent_id is not None
                                    and parent_id not in artifact.parent_artifact_ids
                                    else artifact.parent_artifact_ids
                                ),
                            }
                        )
                        for artifact in finalized_artifacts
                    ]
                artifacts.extend(finalized_artifacts)
                output_paths.extend(finalized.output_paths)
            if not any(artifact.kind == "video" and artifact.role == "primary" for artifact in artifacts):
                raise ValueError("video_primary_artifact_missing")
            sidecar_path = execution.output_paths[0].with_suffix(".asteria.json")
            sidecar_payload = {
                "schema_version": "asteria.video.artifact.v1",
                "generation_id": generation_id,
                "request_snapshot": packet.request_snapshot,
                "resource_authorization": settings.get("resource_authorization"),
                "capability_snapshot": packet.capability_snapshot,
                "adapter": {
                    "id": packet.adapter_id,
                    "revision": packet.adapter_revision,
                    "mapping_id": packet.mapping_id,
                    "mapping_version": packet.mapping_version,
                    "workflow_id": execution.workflow_id,
                    "runtime_job_id": execution.runtime_job_id,
                },
                "timing": {
                    "requested_frame_count": execution.requested_frame_count,
                    "actual_frame_count": execution.actual_frame_count,
                    "actual_duration_seconds": execution.actual_duration_seconds,
                },
                "artifacts": [
                    artifact.model_dump(mode="json", exclude={"frame_rate"})
                    for artifact in artifacts
                ],
            }
            batch_metadata = settings.get("batch")
            if isinstance(batch_metadata, dict):
                sidecar_payload["batch"] = batch_metadata
            if sequence_metadata is not None:
                sidecar_payload["sequence"] = sequence_metadata
                sidecar_payload["sequence_context"] = packet.sequence_context
            self._raise_if_cancelled(cancel_requested, created_paths)
            write_json_atomic(sidecar_path, sidecar_payload)
            created_paths.append(sidecar_path)
            self._raise_if_cancelled(cancel_requested, created_paths)
            sidecar_sha = _file_sha256(sidecar_path)
            primary_ids = tuple(
                artifact.artifact_id
                for artifact in artifacts
                if artifact.kind == "video" and artifact.role == "primary"
            )
            sidecar = MediaArtifact(
                artifact_id=f"media:sidecar:{generation_id}:{sidecar_sha[:20]}",
                kind="sidecar",
                role="sidecar",
                path=str(sidecar_path),
                mime_type="application/json",
                size_bytes=sidecar_path.stat().st_size,
                sha256=sidecar_sha,
                parent_artifact_ids=primary_ids,
                sequence_id=(
                    sequence_metadata["sequence_id"]
                    if sequence_metadata is not None
                    else None
                ),
                segment_id=(
                    sequence_metadata["segment_id"]
                    if sequence_metadata is not None
                    else None
                ),
            )
            artifacts.append(sidecar)
            output_paths.append(str(sidecar_path))
            return artifacts, list(dict.fromkeys(output_paths))
        except Exception:
            self._discard_paths(created_paths)
            raise

    @classmethod
    def _raise_if_cancelled(
        cls,
        cancel_requested: Callable[[], bool],
        paths: tuple[Path, ...] | list[Path],
    ) -> None:
        if not cancel_requested():
            return
        cls._discard_paths(paths)
        raise VideoLocalAdapterError("video_generation_cancelled", retryable=True)

    @staticmethod
    def _discard_paths(paths: tuple[Path, ...] | list[Path]) -> None:
        for path in dict.fromkeys(Path(value) for value in paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _record_runtime_job(self, generation_id: int, settings: dict, runtime_job_id: str) -> None:
        settings["runtime_job_id"] = runtime_job_id
        self.history.update_settings(generation_id, settings)

    @staticmethod
    def _validated_snapshot(
        request_snapshot: dict,
    ) -> tuple[int, VideoSubmissionPacket, dict | None]:
        try:
            generation_id = int(request_snapshot["generation_id"])
            payload = request_snapshot["packet"]
            sequence_context = payload.get("sequence_context")
            packet = VideoSubmissionPacket(
                packet_version=str(payload["packet_version"]),
                runtime_job_id=str(payload["runtime_job_id"]),
                adapter_id=str(payload["adapter_id"]),
                adapter_revision=str(payload["adapter_revision"]),
                mapping_id=str(payload["mapping_id"]),
                mapping_version=int(payload["mapping_version"]),
                _request_snapshot_json=_canonical_json(payload["request_snapshot"]),
                _capability_snapshot_json=_canonical_json(payload["capability_snapshot"]),
                _provider_payload_json=_canonical_json(payload["provider_payload"]),
                _sequence_context_json=(
                    _canonical_json(sequence_context)
                    if sequence_context is not None
                    else None
                ),
            )
            video_request = VideoRequest.model_validate(packet.request_snapshot)
            sequence_metadata = request_snapshot.get("sequence")
            if sequence_metadata is not None:
                if not isinstance(sequence_metadata, dict):
                    raise ValueError("sequence metadata must be an object")
                sequence_id = str(sequence_metadata["sequence_id"])
                segment_id = str(sequence_metadata["segment_id"])
                sequence_revision = int(sequence_metadata["sequence_revision"])
                segment_revision = int(sequence_metadata["segment_revision"])
                parent_artifact_id = sequence_metadata.get("parent_artifact_id")
                if (
                    not sequence_id
                    or not segment_id
                    or sequence_revision < 1
                    or segment_revision < 1
                    or (parent_artifact_id is not None and not isinstance(parent_artifact_id, str))
                ):
                    raise ValueError("sequence metadata is invalid")
                sequence_metadata = {
                    "sequence_id": sequence_id,
                    "segment_id": segment_id,
                    "sequence_revision": sequence_revision,
                    "segment_revision": segment_revision,
                    "parent_artifact_id": parent_artifact_id,
                }
            if sequence_context is not None:
                VideoSequenceContextReceipt.model_validate(sequence_context)
        except (KeyError, TypeError, ValueError) as error:
            raise OperationExecutionError("video_request_snapshot_invalid", retryable=False) from error
        expected_mapping_id = (
            LocalH3VideoAdapter.sequence_mapping_id
            if sequence_metadata is not None and video_request.mode == "sequence"
            else LocalH3VideoAdapter.mapping_id
        )
        if (
            generation_id < 1
            or packet.packet_version != VIDEO_PACKET_VERSION
            or packet.adapter_id != LocalH3VideoAdapter.id
            or packet.adapter_revision != LocalH3VideoAdapter.revision
            or packet.mapping_id != expected_mapping_id
            or packet.mapping_version != 1
            or video_request.provider != "local_h3"
            or (
                packet.sequence_context is not None
                and (video_request.mode != "sequence" or sequence_metadata is None)
            )
        ):
            raise OperationExecutionError("video_request_snapshot_invalid", retryable=False)
        return generation_id, packet, sequence_metadata

    @staticmethod
    def _history_matches_packet(item: dict, packet: VideoSubmissionPacket) -> bool:
        settings = item.get("settings")
        return (
            item.get("media_kind") == "video"
            and isinstance(settings, dict)
            and settings.get("submission_runtime_job_id") == packet.runtime_job_id
            and settings.get("adapter_id") == packet.adapter_id
            and settings.get("mapping_id") == packet.mapping_id
            and settings.get("request_snapshot") == packet.request_snapshot
        )

    @staticmethod
    def _artifact_result(
        generation_id: int,
        artifacts: list[MediaArtifact],
        *,
        working_artifact_ids: tuple[str, ...] = (),
    ) -> OperationExecutionResult:
        return OperationExecutionResult(
            artifact_refs=(
                f"history:{generation_id}",
                *(artifact.artifact_id for artifact in artifacts),
                *working_artifact_ids,
            )
        )

    def _registered_result(self, generation_id: int, item: dict) -> OperationExecutionResult:
        artifacts = [MediaArtifact.model_validate(value) for value in item.get("artifacts", [])]
        working_ids: list[str] = []
        for value in item.get("settings", {}).get("working_artifacts", []):
            if not isinstance(value, dict):
                continue
            try:
                working_ids.append(
                    GenerationWorkingArtifactSummary.model_validate(value).artifact_id
                )
            except ValueError:
                continue
        return self._artifact_result(
            generation_id,
            artifacts,
            working_artifact_ids=tuple(working_ids),
        )

    def _operation_output_root(self, request_snapshot: dict, settings: dict) -> Path:
        snapshot_value = request_snapshot.get("output_root")
        recorded_value = settings.get("output_root")
        if isinstance(snapshot_value, str) and snapshot_value.strip():
            selected = Path(snapshot_value).expanduser().resolve()
            if (
                isinstance(recorded_value, str)
                and recorded_value.strip()
                and Path(recorded_value).expanduser().resolve() != selected
            ):
                raise OperationExecutionError(
                    "video_output_root_snapshot_invalid",
                    retryable=False,
                )
            return selected
        adapter_root = getattr(self.local_adapter, "output_root", None)
        if isinstance(adapter_root, Path):
            return adapter_root.expanduser().resolve()
        raise OperationExecutionError("video_output_root_unavailable", retryable=False)

    @staticmethod
    def _working_summary(
        record: GenerationWorkingArtifactRecord,
    ) -> GenerationWorkingArtifactSummary:
        return GenerationWorkingArtifactSummary(
            artifact_id=record.artifact_id,
            generation_id=record.generation_id,
            media_artifact_id=record.media_artifact_id,
            kind=record.kind,
            state=record.state,
            size_bytes=record.size_bytes,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _discard_working(
        registry: GenerationWorkingArtifactRegistry | None,
        operation_id: str,
        records: list[GenerationWorkingArtifactRecord],
    ) -> None:
        if registry is None:
            return
        for record in records:
            try:
                registry.retire(
                    operation_id,
                    record.artifact_id,
                    outcome="discarded",
                    operation_status="failed",
                )
            except GenerationWorkingArtifactError:
                continue
        try:
            registry.cleanup(operation_status=lambda _operation_id: "failed")
        except GenerationWorkingArtifactError:
            pass


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
