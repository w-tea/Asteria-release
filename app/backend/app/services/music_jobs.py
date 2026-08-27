from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable

from app.schemas.media import MediaArtifact
from app.schemas.music import MusicRequest
from app.schemas.resource_advisory import (
    ResourceAdvisoryAcknowledgement,
    ResourceAdvisoryAuthorization,
    ResourceObservationSnapshot,
    ResourceWorkload,
)
from app.schemas.operations import OperationRecord
from app.schemas.provider_capabilities import capability_after_verified_setup
from app.services.atomic_json import write_json_atomic
from app.services.history import HistoryRepository
from app.services.media_artifacts import MediaFinalizationResult, finalize_media_artifact
from app.services.music_local_adapter import (
    LocalMusic3ExecutionAdapter,
    LocalMusicExecutionResult,
    MusicLocalAdapterError,
    normalized_music3_duration,
)
from app.services.music_packet import (
    MUSIC_PACKET_VERSION,
    MusicPacketBuilder,
    MusicPacketError,
    MusicSubmissionPacket,
)
from app.services.music_model_bundle_resolver import MusicModelBundlePreflightError
from app.services.music_runtime_adapters import LocalMusic3Adapter
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)
from app.services.operation_repository import OperationNotFoundError
from app.services.resource_advisor import ResourceAdvisoryError, ResourceAdvisor


MUSIC_OPERATION_KIND = "music.generate"
MUSIC_OPERATION_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SubmittedMusicOperation:
    operation: OperationRecord
    generation_id: int


class MusicOperationService:
    def __init__(
        self,
        *,
        coordinator: OperationCoordinator,
        history: HistoryRepository,
        packet_builder: MusicPacketBuilder | None = None,
        readiness_check: Callable[[MusicRequest], bool] | None = None,
        resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
        resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
        execution_profile_provider: Callable[[], str] | None = None,
        model_preflight: Callable[[MusicRequest], object] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.history = history
        self.packet_builder = packet_builder or MusicPacketBuilder()
        self.readiness_check = readiness_check
        self.resource_advisor_provider = resource_advisor_provider or ResourceAdvisor
        self.resource_observation_provider = resource_observation_provider or ResourceObservationSnapshot
        self.execution_profile_provider = execution_profile_provider or (lambda: "layer_streaming")
        self.model_preflight = model_preflight

    def draft(self, request: MusicRequest) -> dict:
        if request.provider != "local_music3":
            raise MusicLocalAdapterError("music_provider_not_implemented")
        self._run_model_preflight(request)
        packet = self.packet_builder.build(request, LocalMusic3Adapter())
        timing = normalized_music3_duration(float(packet.request_snapshot["duration_intent"]))
        ready = self.readiness_check is None or self.readiness_check(request)
        return {
            "schema_version": "asteria.music.draft.v1",
            "request": packet.request_snapshot,
            "provider": request.provider,
            "adapter_id": packet.adapter_id,
            "adapter_revision": packet.adapter_revision,
            "mapping_id": packet.mapping_id,
            "mapping_revision": packet.mapping_revision,
            "timing": {
                "requested_duration_seconds": timing.requested_duration_seconds,
                "effective_duration_seconds": timing.effective_duration_seconds,
                "effective_audio_frames": timing.effective_audio_frames,
                "provider_clamp_reason": timing.provider_clamp_reason,
            },
            "ready": ready,
            "reason_code": None if ready else "music_local_setup_required",
        }

    def submit(
        self,
        request: MusicRequest,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ) -> SubmittedMusicOperation:
        if request.provider != "local_music3":
            raise MusicLocalAdapterError("music_provider_not_implemented")
        setup_verified = self.readiness_check is not None
        if setup_verified and not self.readiness_check(request):
            raise MusicLocalAdapterError("music_local_setup_required")
        self._run_model_preflight(request)
        resource_authorization = self._validate_resource_authorization(
            request,
            resource_authorization,
            run_anyway=run_anyway,
        )
        adapter = LocalMusic3Adapter()
        capability = adapter.capabilities()
        if setup_verified:
            capability = capability_after_verified_setup(capability)
        packet = self.packet_builder.build(request, adapter, capability=capability)
        timing = normalized_music3_duration(float(packet.request_snapshot["duration_intent"]))
        settings = {
            "schema_version": "asteria.music.history.v1",
            "request_snapshot": packet.request_snapshot,
            "capability_snapshot": packet.capability_snapshot,
            "provider": request.provider,
            "adapter_id": packet.adapter_id,
            "adapter_revision": packet.adapter_revision,
            "mapping_id": packet.mapping_id,
            "mapping_revision": packet.mapping_revision,
            "submission_runtime_job_id": packet.runtime_job_id,
            "requested_duration_seconds": timing.requested_duration_seconds,
            "effective_duration_seconds": timing.effective_duration_seconds,
            "effective_audio_frames": timing.effective_audio_frames,
            "provider_clamp_reason": timing.provider_clamp_reason,
            "resource_authorization": (
                resource_authorization.model_dump(mode="json")
                if resource_authorization is not None
                else None
            ),
        }
        generation_id = self.history.add(
            status="queued",
            profile=request.model_profile,
            prompt=request.description,
            expanded_prompt=request.description,
            negative_prompt="",
            settings=settings,
            model_slots={},
            loras=[],
            embeddings=[],
            output_paths=[],
            media_kind="audio",
            artifacts=[],
        )
        try:
            operation = self.coordinator.enqueue(
                kind=MUSIC_OPERATION_KIND,
                source_actor=source_actor,
                request_snapshot={
                    "generation_id": generation_id,
                    "packet": packet.to_dict(),
                    "resource_authorization": (
                        resource_authorization.model_dump(mode="json")
                        if resource_authorization is not None
                        else None
                    ),
                },
                request_contract_version=MUSIC_OPERATION_CONTRACT_VERSION,
                progress_total=max(1, math.ceil(timing.effective_duration_seconds)),
                progress_unit="seconds",
            )
        except Exception:
            self.history.update_generation(
                generation_id,
                status="failed",
                settings={**settings, "error_code": "music_operation_enqueue_failed"},
                output_paths=[],
            )
            raise
        return SubmittedMusicOperation(operation, generation_id)

    def _validate_resource_authorization(
        self,
        request: MusicRequest,
        authorization: ResourceAdvisoryAuthorization | None,
        *,
        run_anyway: bool = False,
    ) -> ResourceAdvisoryAuthorization | None:
        advisor = self.resource_advisor_provider()
        try:
            current = advisor.advise(
                _resource_workload(
                    request,
                    request.sampling.execution_profile
                    if request.sampling is not None
                    else self.execution_profile_provider(),
                ),
                LocalMusic3Adapter().capabilities(),
                self.resource_observation_provider(),
            )
        except ResourceAdvisoryError as error:
            mapped_code = {
                "resource_advisory_profile_mismatch": "music_model_profile_unsupported",
                "resource_advisory_task_unavailable": "music_task_not_advertised",
                "resource_advisory_provider_mismatch": "music_provider_mismatch",
            }.get(error.code, "music_resource_authorization_mismatch")
            raise MusicLocalAdapterError(mapped_code) from error
        if run_anyway:
            acknowledgement = (
                ResourceAdvisoryAcknowledgement(advisory_fingerprint=current.fingerprint, run_anyway=True)
                if current.acknowledgement_required
                else None
            )
            return advisor.authorize(current, acknowledgement)
        if authorization is not None and authorization.advisory.fingerprint != current.fingerprint:
            raise MusicLocalAdapterError("music_resource_authorization_mismatch")
        try:
            verified = advisor.authorize(
                current,
                authorization.acknowledgement if authorization is not None else None,
            )
        except ResourceAdvisoryError as error:
            if error.code == "resource_advisory_acknowledgement_required":
                raise MusicLocalAdapterError("music_resource_acknowledgement_required") from error
            raise MusicLocalAdapterError("music_resource_authorization_mismatch") from error
        if authorization is None and not current.acknowledgement_required:
            return None
        return verified

    def retry(
        self,
        operation_id: str,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ) -> OperationRecord:
        operation = self.coordinator.get(operation_id)
        if operation.kind != MUSIC_OPERATION_KIND:
            raise OperationNotFoundError("operation_not_found")
        try:
            request = MusicRequest.model_validate(
                operation.request_snapshot["packet"]["request_snapshot"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MusicLocalAdapterError("music_request_snapshot_invalid") from error
        if self.readiness_check is not None and not self.readiness_check(request):
            raise MusicLocalAdapterError("music_local_setup_required")
        self._run_model_preflight(request)
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
        generation_id = request_snapshot.get("generation_id")
        if not isinstance(generation_id, int) or generation_id < 1:
            raise MusicLocalAdapterError("music_request_snapshot_invalid")
        item = self.history.get(generation_id)
        if item is None:
            raise MusicLocalAdapterError("music_history_not_found")
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
            self.history.update_generation(
                generation_id,
                status=item["status"],
                settings=dict(item["settings"]),
                output_paths=list(item["output_paths"]),
            )
            raise

    def _run_model_preflight(self, request: MusicRequest) -> None:
        if request.schema_version != "asteria.music.request.v2":
            return
        if self.model_preflight is None:
            raise MusicLocalAdapterError("music_model_resolver_not_configured")
        try:
            self.model_preflight(request)
        except MusicModelBundlePreflightError as error:
            raise MusicLocalAdapterError(error.code) from error


class _PartialFinalizationError(RuntimeError):
    def __init__(self, artifacts: list[MediaArtifact], output_paths: list[str]) -> None:
        self.artifacts = artifacts
        self.output_paths = output_paths
        super().__init__("music_partial_output_preserved")


class MusicOperationExecutor:
    version = "music-local-minimax3-v1"

    def __init__(
        self,
        *,
        history: HistoryRepository,
        local_adapter: LocalMusic3ExecutionAdapter,
        poster_root: Path,
        finalizer: Callable[..., MediaFinalizationResult] = finalize_media_artifact,
    ) -> None:
        self.history = history
        self.local_adapter = local_adapter
        self.poster_root = Path(poster_root)
        self.finalizer = finalizer

    def resource_claims(self, _request_snapshot: dict) -> tuple[str, ...]:
        return ("comfy_gpu_exclusive",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        return self._execute(
            context.request_snapshot,
            context.operation.operation_id,
            progress=lambda current, total: context.report_progress(
                current, total, "seconds", phase="generating"
            ),
            cancel_requested=lambda: context.cancel_requested,
        )

    def execute_for_test(self, request_snapshot: dict, operation_id: str) -> OperationExecutionResult:
        return self._execute(
            request_snapshot,
            operation_id,
            progress=lambda *_args: None,
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
        generation_id, packet = self._validated_snapshot(request_snapshot)
        item = self.history.get(generation_id)
        if item is None:
            raise OperationExecutionError("music_history_not_found", retryable=False)
        if not self._history_matches_packet(item, packet):
            raise OperationExecutionError("music_request_snapshot_invalid", retryable=False)
        if item["status"] == "completed":
            return self._registered_result(generation_id, item)
        settings = dict(item["settings"])
        settings.pop("error_code", None)
        settings.pop("warning_codes", None)
        output_root = getattr(self.local_adapter, "output_root", None)
        if isinstance(output_root, Path):
            settings["output_root"] = str(output_root)
        self.history.update_generation(
            generation_id,
            status="running",
            settings=settings,
            output_paths=list(item["output_paths"]),
        )
        execution: LocalMusicExecutionResult | None = None
        try:
            execution = self.local_adapter.execute_music(
                packet,
                operation_id=operation_id,
                output_root=output_root,
                progress_callback=progress,
                cancel_requested=cancel_requested,
                runtime_job_callback=lambda value: self._record_runtime_job(
                    generation_id, settings, value
                ),
            )
            settings.update(
                {
                    "runtime_job_id": execution.runtime_job_id,
                    "workflow_id": execution.workflow_id,
                    "workflow_source_revision": execution.workflow_source_revision,
                    "execution_profile": execution.execution_profile,
                    "model_revision": execution.model_revision,
                    "model_asset_ids": list(execution.model_asset_ids),
                    "model_bundle_id": execution.model_bundle_id,
                    "model_components": _music_model_component_receipts(execution),
                    "sampling": _music_sampling_receipt(execution),
                    "requested_duration_seconds": execution.timing.requested_duration_seconds,
                    "effective_duration_seconds": execution.timing.effective_duration_seconds,
                    "effective_audio_frames": execution.timing.effective_audio_frames,
                    "provider_clamp_reason": execution.timing.provider_clamp_reason,
                }
            )
            self._raise_if_cancelled(cancel_requested, execution.output_paths)
            artifacts, output_paths = self._finalize_outputs(
                generation_id,
                packet,
                execution,
                settings=settings,
                cancel_requested=cancel_requested,
            )
            self._raise_if_cancelled(
                cancel_requested,
                [*execution.output_paths, *(Path(value) for value in output_paths)],
            )
            self.history.complete_media_generation(
                generation_id,
                status="completed",
                settings=settings,
                media_kind="audio",
                artifacts=artifacts,
                output_paths=output_paths,
            )
            return self._artifact_result(generation_id, artifacts)
        except _PartialFinalizationError as error:
            settings.update(
                {
                    "error_code": "music_artifact_finalize_failed",
                    "warning_codes": ["music_partial_output_preserved"],
                }
            )
            self.history.complete_media_generation(
                generation_id,
                status="failed",
                settings=settings,
                media_kind="audio",
                artifacts=error.artifacts,
                output_paths=error.output_paths,
            )
            refs = self._artifact_result(generation_id, error.artifacts).artifact_refs
            raise OperationExecutionError(
                "music_artifact_finalize_failed", retryable=True, artifact_refs=refs
            ) from error
        except MusicLocalAdapterError as error:
            partial = self._finalize_partial_paths(generation_id, error.partial_paths)
            settings.update(
                {
                    "error_code": error.code,
                    "runtime_job_id": settings.get("runtime_job_id"),
                }
            )
            if partial is not None:
                artifacts, paths = partial
                settings["warning_codes"] = ["music_partial_output_preserved"]
                self.history.complete_media_generation(
                    generation_id,
                    status="failed",
                    settings=settings,
                    media_kind="audio",
                    artifacts=artifacts,
                    output_paths=paths,
                )
                refs = self._artifact_result(generation_id, artifacts).artifact_refs
            else:
                refs = ()
                self.history.update_generation(
                    generation_id,
                    status="cancelled" if error.code == "music_generation_cancelled" else "failed",
                    settings=settings,
                    output_paths=list(item["output_paths"]),
                )
            if error.code == "music_generation_cancelled":
                raise OperationCancelled(artifact_refs=refs) from error
            raise OperationExecutionError(
                error.code, retryable=error.retryable, artifact_refs=refs
            ) from error
        except (OSError, ValueError) as error:
            if execution is not None:
                self._discard_paths(execution.output_paths)
            settings["error_code"] = "music_artifact_finalize_failed"
            self.history.update_generation(
                generation_id,
                status="failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            raise OperationExecutionError(
                "music_artifact_finalize_failed", retryable=True
            ) from error

    def _finalize_outputs(
        self,
        generation_id: int,
        packet: MusicSubmissionPacket,
        execution: LocalMusicExecutionResult,
        *,
        settings: dict,
        cancel_requested: Callable[[], bool],
    ) -> tuple[list[MediaArtifact], list[str]]:
        artifacts: list[MediaArtifact] = []
        output_paths: list[str] = []
        sidecar_path: Path | None = None
        try:
            for source in execution.output_paths:
                self._raise_if_cancelled(cancel_requested, execution.output_paths)
                finalized = self.finalizer(
                    generation_id=generation_id,
                    source_path=source,
                    poster_root=self.poster_root,
                )
                if finalized.media_kind != "audio":
                    raise ValueError("music_artifact_kind_invalid")
                artifacts.extend(finalized.artifacts)
                output_paths.extend(finalized.output_paths)
            if not any(a.kind == "audio" and a.role == "primary" for a in artifacts):
                raise ValueError("music_primary_artifact_missing")
            sidecar_path = execution.output_paths[0].with_suffix(".asteria.json")
            sidecar_payload = {
                "schema_version": "asteria.music.artifact.v1",
                "generation_id": generation_id,
                "request_snapshot": packet.request_snapshot,
                "resource_authorization": settings.get("resource_authorization"),
                "capability_snapshot": packet.capability_snapshot,
                "adapter": {
                    "id": packet.adapter_id,
                    "revision": packet.adapter_revision,
                    "mapping_id": packet.mapping_id,
                    "mapping_revision": packet.mapping_revision,
                    "workflow_id": execution.workflow_id,
                    "workflow_source_revision": execution.workflow_source_revision,
                    "execution_profile": execution.execution_profile,
                    "model_profile": packet.request_snapshot["model_profile"],
                    "model_revision": execution.model_revision,
                    "model_asset_ids": list(execution.model_asset_ids),
                    "model_bundle_id": execution.model_bundle_id,
                    "model_components": _music_model_component_receipts(execution),
                    "runtime_job_id": execution.runtime_job_id,
                    "provider_output": execution.provider_output,
                },
                "sampling": _music_sampling_receipt(execution),
                "timing": {
                    "requested_duration_seconds": execution.timing.requested_duration_seconds,
                    "effective_duration_seconds": execution.timing.effective_duration_seconds,
                    "effective_audio_frames": execution.timing.effective_audio_frames,
                    "provider_clamp_reason": execution.timing.provider_clamp_reason,
                },
                "artifacts": [
                    artifact.model_dump(mode="json", exclude={"frame_rate"})
                    for artifact in artifacts
                ],
            }
            self._raise_if_cancelled(cancel_requested, execution.output_paths)
            write_json_atomic(sidecar_path, sidecar_payload)
            self._raise_if_cancelled(
                cancel_requested, [*execution.output_paths, sidecar_path]
            )
            sidecar_sha = _file_sha256(sidecar_path)
            primary_ids = tuple(a.artifact_id for a in artifacts if a.kind == "audio")
            artifacts.append(
                MediaArtifact(
                    artifact_id=f"media:sidecar:{generation_id}:{sidecar_sha[:20]}",
                    kind="sidecar",
                    role="sidecar",
                    path=str(sidecar_path),
                    mime_type="application/json",
                    size_bytes=sidecar_path.stat().st_size,
                    sha256=sidecar_sha,
                    parent_artifact_ids=primary_ids,
                )
            )
            output_paths.append(str(sidecar_path))
            return artifacts, list(dict.fromkeys(output_paths))
        except MusicLocalAdapterError:
            if sidecar_path is not None:
                self._discard_paths([sidecar_path])
            raise
        except Exception as error:
            if sidecar_path is not None:
                self._discard_paths([sidecar_path])
            if artifacts:
                registered = {Path(artifact.path) for artifact in artifacts if artifact.kind == "audio"}
                self._discard_paths(
                    [path for path in execution.output_paths if Path(path) not in registered]
                )
                raise _PartialFinalizationError(
                    [artifact for artifact in artifacts if artifact.kind == "audio"],
                    [artifact.path for artifact in artifacts if artifact.kind == "audio"],
                ) from error
            raise

    def _finalize_partial_paths(
        self,
        generation_id: int,
        paths: tuple[Path, ...],
    ) -> tuple[list[MediaArtifact], list[str]] | None:
        artifacts: list[MediaArtifact] = []
        output_paths: list[str] = []
        for path in paths:
            try:
                result = self.finalizer(
                    generation_id=generation_id,
                    source_path=path,
                    poster_root=self.poster_root,
                )
            except (OSError, ValueError):
                self._discard_paths([path])
                continue
            if result.media_kind == "audio":
                artifacts.extend(result.artifacts)
                output_paths.extend(result.output_paths)
        return (artifacts, list(dict.fromkeys(output_paths))) if artifacts else None

    @classmethod
    def _raise_if_cancelled(
        cls, callback: Callable[[], bool], paths: tuple[Path, ...] | list[Path]
    ) -> None:
        if callback():
            cls._discard_paths(paths)
            raise MusicLocalAdapterError("music_generation_cancelled", retryable=True)

    @staticmethod
    def _discard_paths(paths: tuple[Path, ...] | list[Path]) -> None:
        for path in dict.fromkeys(Path(value) for value in paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _record_runtime_job(self, generation_id: int, settings: dict, value: str) -> None:
        settings["runtime_job_id"] = value
        self.history.update_settings(generation_id, settings)

    @staticmethod
    def _validated_snapshot(snapshot: dict) -> tuple[int, MusicSubmissionPacket]:
        try:
            generation_id = int(snapshot["generation_id"])
            payload = snapshot["packet"]
            packet = MusicSubmissionPacket(
                packet_version=str(payload["packet_version"]),
                runtime_job_id=str(payload["runtime_job_id"]),
                adapter_id=str(payload["adapter_id"]),
                adapter_revision=str(payload["adapter_revision"]),
                mapping_id=str(payload["mapping_id"]),
                mapping_revision=int(payload["mapping_revision"]),
                _request_json=_canonical_json(payload["request_snapshot"]),
                _capability_json=_canonical_json(payload["capability_snapshot"]),
                _payload_json=_canonical_json(payload["provider_payload"]),
            )
            MusicRequest.model_validate(packet.request_snapshot)
        except (KeyError, TypeError, ValueError) as error:
            raise OperationExecutionError("music_request_snapshot_invalid", retryable=False) from error
        if (
            generation_id < 1
            or packet.packet_version != MUSIC_PACKET_VERSION
            or packet.adapter_id != LocalMusic3Adapter.id
            or packet.adapter_revision != LocalMusic3Adapter.revision
            or packet.mapping_id != LocalMusic3Adapter.mapping_id
            or packet.mapping_revision not in {1, 2}
            or packet.request_snapshot.get("provider") != "local_music3"
        ):
            raise OperationExecutionError("music_request_snapshot_invalid", retryable=False)
        return generation_id, packet

    @staticmethod
    def _history_matches_packet(item: dict, packet: MusicSubmissionPacket) -> bool:
        settings = item.get("settings")
        return (
            item.get("media_kind") == "audio"
            and isinstance(settings, dict)
            and settings.get("submission_runtime_job_id") == packet.runtime_job_id
            and settings.get("adapter_id") == packet.adapter_id
            and settings.get("mapping_id") == packet.mapping_id
            and settings.get("request_snapshot") == packet.request_snapshot
        )

    @staticmethod
    def _artifact_result(generation_id: int, artifacts: list[MediaArtifact]) -> OperationExecutionResult:
        return OperationExecutionResult(
            artifact_refs=(
                f"history:{generation_id}",
                *(artifact.artifact_id for artifact in artifacts),
            )
        )

    def _registered_result(self, generation_id: int, item: dict) -> OperationExecutionResult:
        artifacts = [MediaArtifact.model_validate(value) for value in item.get("artifacts", [])]
        return self._artifact_result(generation_id, artifacts)


def _resource_workload(request: MusicRequest, execution_profile: str) -> ResourceWorkload:
    offload = {
        "automatic": "auto",
        "full": "none",
        "offload": "balanced",
        "layer_streaming": "layer_streaming",
    }.get(execution_profile)
    if offload is None:
        raise MusicLocalAdapterError("music_execution_profile_invalid")
    return ResourceWorkload(
        domain="music",
        provider_id=request.provider,
        adapter_id=LocalMusic3Adapter.id,
        model_profile=request.model_profile,
        task_id=request.mode,
        duration_seconds=request.duration_intent,
        batch=1,
        guide_count=0,
        guide_duration_seconds=0,
        audio=True,
        sequence_context=False,
        precision="auto",
        offload=offload,
    )


def _music_model_component_receipts(
    execution: LocalMusicExecutionResult,
) -> list[dict[str, object]]:
    return [
        {
            "role": component.role.value
            if hasattr(component.role, "value")
            else str(component.role),
            "stable_asset_id": component.stable_asset_id,
            "loader_name": component.loader_name,
            "fingerprint_algorithm": component.fingerprint_algorithm,
            "fingerprint_version": component.fingerprint_version,
            "fingerprint_value": component.fingerprint_value,
        }
        for component in execution.model_components
    ]


def _music_sampling_receipt(
    execution: LocalMusicExecutionResult,
) -> dict[str, object] | None:
    if execution.sampling is None:
        return None
    return {
        "text_guidance": execution.sampling.text_guidance,
        "top_k": execution.sampling.top_k,
        "steps": execution.sampling.steps,
        "sampler": execution.sampling.sampler,
        "scheduler": execution.sampling.scheduler,
        "denoise": execution.sampling.denoise,
        "tiled_decode": execution.sampling.tiled_decode,
        "ksampler_cfg": execution.sampling.ksampler_cfg,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
