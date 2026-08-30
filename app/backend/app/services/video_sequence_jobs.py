from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from app.schemas.media import MediaArtifact
from app.schemas.generation_working_artifact import GenerationWorkingArtifactSummary
from app.schemas.provider_capabilities import (
    ProviderCapabilityDescriptor,
    capability_after_verified_setup,
)
from app.schemas.operations import ACTIVE_OPERATION_STATUSES, OperationRecord
from app.schemas.resource_advisory import (
    ResourceAdvisoryAuthorization,
    ResourceObservationSnapshot,
)
from app.schemas.video import VideoRequest
from app.schemas.video_sequence import VideoSequence, VideoSequenceSegment
from app.services.history import HistoryRepository
from app.services.atomic_json import write_json_atomic
from app.services.media_artifacts import (
    MediaArtifactAccessError,
    file_sha256,
    finalize_media_artifact,
    resolve_registered_media_artifact,
)
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_executor import (
    OperationCancelled,
    OperationExecutionContext,
    OperationExecutionError,
    OperationExecutionResult,
)
from app.services.operation_repository import OperationNotFoundError
from app.services.resource_advisor import ResourceAdvisor
from app.services.video_executor import (
    VideoOperationExecutor,
    VideoOperationService,
    _resource_workload,
)
from app.services.video_frame_policy import normalized_h3_frame_count
from app.services.video_local_adapter import VideoLocalAdapterError
from app.services.video_local_adapter import HistoryVideoWorkingArtifactResolver
from app.services.video_packet import VideoPacketBuilder, VideoSubmissionPacket
from app.services.video_runtime_adapters import LocalH3VideoAdapter
from app.services.video_sequence_context import (
    VideoSequenceContextPreparer,
    VideoSequenceSourceArtifact,
)
from app.services.video_sequence_assembly import (
    ASSEMBLY_REVISION,
    SequenceAssemblyPlan,
    VideoSequenceAssemblyError,
    VideoSequenceAssemblyService,
    build_active_chain_plan,
)
from app.services.video_sequence_repository import (
    VideoSequenceConflictError,
    VideoSequenceNotFoundError,
    VideoSequenceRepository,
)


VIDEO_SEQUENCE_CLIP_OPERATION_KIND = "video.sequence.clip"
VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND = "video.sequence.assemble"
VIDEO_SEQUENCE_OPERATION_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SubmittedVideoSequenceClip:
    sequence: VideoSequence
    operation: OperationRecord
    generation_id: int


@dataclass(frozen=True, slots=True)
class SubmittedVideoSequenceAssembly:
    sequence: VideoSequence
    operation: OperationRecord
    generation_id: int


class VideoSequenceClipExecutor:
    version = "video-sequence-clip-v1"

    def __init__(
        self,
        *,
        delegate: VideoOperationExecutor,
        sequences: VideoSequenceRepository,
        history: HistoryRepository,
    ) -> None:
        self.delegate = delegate
        self.sequences = sequences
        self.history = history

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]:
        return self.delegate.resource_claims(request_snapshot)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        metadata = self._metadata(context.request_snapshot)
        try:
            result = self.delegate.execute(context)
            item = self.history.get(metadata["generation_id"])
            if item is None:
                raise OperationExecutionError("video_history_not_found", retryable=False)
            candidate_id = self._primary_candidate(item, metadata)
            packet = context.request_snapshot.get("packet", {})
            context_receipt = packet.get("sequence_context")
            working_artifact_receipt = self._working_artifact_receipt(
                result,
                item,
                metadata,
                context.operation.operation_id,
                candidate_id,
            )
            self.sequences.enter_review(
                metadata["sequence_id"],
                segment_id=metadata["segment_id"],
                candidate_artifact_id=candidate_id,
                candidate_generation_id=metadata["generation_id"],
                context_receipt=context_receipt,
                working_artifact_receipt=working_artifact_receipt,
                expected_revision=metadata["sequence_revision"],
                expected_segment_revision=metadata["segment_revision"],
                operation_id=context.operation.operation_id,
                receipt_id=f"review:{context.operation.operation_id}",
            )
            return result
        except (OperationCancelled, OperationExecutionError):
            self.sequences.mark_operation_retryable(
                metadata["sequence_id"],
                segment_id=metadata["segment_id"],
                operation_id=context.operation.operation_id,
            )
            raise
        except Exception as error:
            self.sequences.mark_operation_retryable(
                metadata["sequence_id"],
                segment_id=metadata["segment_id"],
                operation_id=context.operation.operation_id,
            )
            code = getattr(error, "code", "video_sequence_review_failed")
            raise OperationExecutionError(str(code), retryable=True) from error

    @staticmethod
    def _metadata(snapshot: dict) -> dict[str, Any]:
        try:
            sequence = snapshot["sequence"]
            result = {
                "sequence_id": str(sequence["sequence_id"]),
                "segment_id": str(sequence["segment_id"]),
                "sequence_revision": int(sequence["sequence_revision"]),
                "segment_revision": int(sequence["segment_revision"]),
                "generation_id": int(snapshot["generation_id"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise OperationExecutionError(
                "video_sequence_snapshot_invalid", retryable=False
            ) from error
        if any(not result[key] for key in ("sequence_id", "segment_id")) or any(
            result[key] < 1
            for key in ("sequence_revision", "segment_revision", "generation_id")
        ):
            raise OperationExecutionError("video_sequence_snapshot_invalid", retryable=False)
        return result

    @staticmethod
    def _primary_candidate(item: dict, metadata: dict[str, Any]) -> str:
        for raw in item.get("artifacts", []):
            try:
                artifact = MediaArtifact.model_validate(raw)
            except ValueError:
                continue
            if (
                artifact.kind == "video"
                and artifact.role == "primary"
                and artifact.sequence_id == metadata["sequence_id"]
                and artifact.segment_id == metadata["segment_id"]
            ):
                return artifact.artifact_id
        raise OperationExecutionError("video_sequence_candidate_missing", retryable=True)

    @staticmethod
    def _working_artifact_receipt(
        result: OperationExecutionResult,
        item: dict,
        metadata: dict[str, Any],
        operation_id: str,
        candidate_id: str,
    ) -> dict[str, Any] | None:
        working_ids = [
            value
            for value in result.artifact_refs
            if value.startswith("working:av_latent:")
        ]
        if not working_ids:
            return None
        if len(working_ids) != 1:
            raise OperationExecutionError(
                "video_sequence_working_artifact_ambiguous", retryable=True
            )
        matches: list[GenerationWorkingArtifactSummary] = []
        settings = item.get("settings")
        if isinstance(settings, dict):
            for raw in settings.get("working_artifacts", []):
                try:
                    summary = GenerationWorkingArtifactSummary.model_validate(raw)
                except ValueError:
                    continue
                if summary.artifact_id == working_ids[0]:
                    matches.append(summary)
        if len(matches) != 1:
            raise OperationExecutionError(
                "video_sequence_working_artifact_missing", retryable=True
            )
        summary = matches[0]
        if (
            summary.generation_id != metadata["generation_id"]
            or summary.media_artifact_id != candidate_id
            or summary.kind != "av_latent"
            or summary.state != "available"
        ):
            raise OperationExecutionError(
                "video_sequence_working_artifact_mismatch", retryable=True
            )
        return {
            "artifact_id": summary.artifact_id,
            "operation_id": operation_id,
            "generation_id": summary.generation_id,
            "kind": summary.kind,
        }


class VideoSequenceAssemblyBoundaryExecutor:
    version = "video-sequence-assembly-boundary-v1"

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]:
        identity = request_snapshot.get("plan_identity")
        if (
            isinstance(identity, dict)
            and identity.get("assembly_mode") == "accepted_latent_crossfade"
        ):
            return ("comfy_gpu_exclusive", "disk_write_heavy")
        return ("cpu_heavy", "disk_write_heavy")

    def execute(self, _context: OperationExecutionContext) -> OperationExecutionResult:
        raise OperationExecutionError(
            "video_sequence_assembly_not_configured",
            retryable=False,
        )


class VideoSequenceAssemblyExecutor:
    version = "video-sequence-assembly-v1"

    def __init__(
        self,
        *,
        sequences: VideoSequenceRepository,
        history: HistoryRepository,
        assembler: VideoSequenceAssemblyService,
        poster_root: Path,
        accepted_roots: set[Path] | Callable[[], set[Path]],
        finalizer: Callable[..., Any] = finalize_media_artifact,
    ) -> None:
        self.sequences = sequences
        self.history = history
        self.assembler = assembler
        self.poster_root = Path(poster_root)
        self._accepted_roots_provider = (
            accepted_roots
            if callable(accepted_roots)
            else lambda: {Path(root) for root in accepted_roots}
        )
        self.finalizer = finalizer
        self.working_artifact_resolver = HistoryVideoWorkingArtifactResolver(
            history
        ).resolve

    @property
    def accepted_roots(self) -> set[Path]:
        return {Path(root) for root in self._accepted_roots_provider()}

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]:
        identity = request_snapshot.get("plan_identity")
        if (
            isinstance(identity, dict)
            and identity.get("assembly_mode") == "accepted_latent_crossfade"
        ):
            return ("comfy_gpu_exclusive", "disk_write_heavy")
        return ("cpu_heavy", "disk_write_heavy")

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        metadata = self._metadata(context.request_snapshot)
        generation_id = metadata["generation_id"]
        created_paths: list[Path] = []
        try:
            sequence = self.sequences.get(metadata["sequence_id"])
            plan = build_active_chain_plan(
                sequence,
                artifact_resolver=self._resolve_artifact,
                working_artifact_resolver=self.working_artifact_resolver,
                assembly_revision=metadata["assembly_revision"],
            )
            if self._plan_identity(plan) != metadata["plan_identity"]:
                raise VideoSequenceAssemblyError("video_sequence_assembly_snapshot_stale")
            context.raise_if_cancelled()
            item = self.history.get(generation_id)
            if item is None:
                raise VideoSequenceAssemblyError(
                    "video_sequence_assembly_history_missing"
                )
            settings = item.get("settings")
            recorded_root = settings.get("output_root") if isinstance(settings, dict) else None
            output_root = Path(metadata["output_root"]).expanduser().resolve()
            if (
                not isinstance(recorded_root, str)
                or Path(recorded_root).expanduser().resolve() != output_root
            ):
                raise VideoSequenceAssemblyError(
                    "video_sequence_assembly_output_root_stale"
                )
            assembled = self.assembler.assemble(
                plan,
                cancel_requested=lambda: context.cancel_requested,
                output_root=output_root,
                operation_id=context.operation.operation_id,
            )
            created_paths.append(assembled.output_path)
            context.raise_if_cancelled()
            finalized = self.finalizer(
                generation_id=generation_id,
                source_path=assembled.output_path,
                poster_root=self.poster_root,
            )
            created_paths.extend(Path(value) for value in finalized.output_paths)
            source_ids = tuple(clip.artifact.artifact_id for clip in plan.clips)
            artifacts = [
                artifact.model_copy(
                    update={
                        "sequence_id": plan.sequence_id,
                        "parent_artifact_ids": tuple(
                            dict.fromkeys((*artifact.parent_artifact_ids, *source_ids))
                        ),
                        "adapter_metadata": {
                            **artifact.adapter_metadata,
                            "active_chain_revision": plan.active_chain_revision,
                            "assembly_revision": plan.assembly_revision,
                            "assembly_mode": plan.assembly_mode,
                            "assembly_fallback_reason": plan.fallback_reason,
                            "assembly_receipt_sha256": assembled.receipt_sha256,
                        },
                    }
                )
                for artifact in finalized.artifacts
            ]
            primary = next(
                (
                    artifact
                    for artifact in artifacts
                    if artifact.kind == "video" and artifact.role == "primary"
                ),
                None,
            )
            if primary is None:
                raise VideoSequenceAssemblyError("video_sequence_assembly_primary_missing")
            sidecar_path = assembled.output_path.with_suffix(".asteria.json")
            sidecar_payload = {
                "schema_version": "asteria.video.sequence.artifact.v1",
                "generation_id": generation_id,
                "assembly_receipt": assembled.receipt,
                "assembly_receipt_sha256": assembled.receipt_sha256,
                "artifacts": [
                    artifact.model_dump(mode="json", exclude={"frame_rate"})
                    for artifact in artifacts
                ],
            }
            write_json_atomic(sidecar_path, sidecar_payload)
            created_paths.append(sidecar_path)
            sidecar_sha = file_sha256(sidecar_path)
            sidecar = MediaArtifact(
                artifact_id=f"media:sidecar:{generation_id}:{sidecar_sha[:20]}",
                kind="sidecar",
                role="sidecar",
                path=str(sidecar_path),
                mime_type="application/json",
                size_bytes=sidecar_path.stat().st_size,
                sha256=sidecar_sha,
                parent_artifact_ids=(primary.artifact_id, *source_ids),
                sequence_id=plan.sequence_id,
                adapter_metadata={
                    "active_chain_revision": plan.active_chain_revision,
                    "assembly_revision": plan.assembly_revision,
                    "assembly_mode": plan.assembly_mode,
                    "assembly_fallback_reason": plan.fallback_reason,
                    "assembly_receipt_sha256": assembled.receipt_sha256,
                },
            )
            artifacts.append(sidecar)
            output_paths = list(
                dict.fromkeys((*finalized.output_paths, str(sidecar_path)))
            )
            settings = dict(item["settings"])
            settings["assembly_receipt"] = assembled.receipt
            settings["assembly_receipt_sha256"] = assembled.receipt_sha256
            settings["sidecar_sha256"] = sidecar_sha
            settings["output_root"] = str(output_root)
            context.raise_if_cancelled()
            self.sequences.complete_assembly(
                plan.sequence_id,
                operation_id=context.operation.operation_id,
                assembled_artifact_id=primary.artifact_id,
                expected_active_chain_revision=plan.active_chain_revision,
                expected_assembly_revision=plan.assembly_revision,
            )
            self.history.complete_media_generation(
                generation_id,
                status="completed",
                settings=settings,
                media_kind="video",
                artifacts=artifacts,
                output_paths=output_paths,
            )
            return OperationExecutionResult(
                artifact_refs=tuple(artifact.artifact_id for artifact in artifacts)
            )
        except OperationCancelled:
            self._fail(metadata, context.operation.operation_id, created_paths)
            raise
        except (VideoSequenceAssemblyError, MediaArtifactAccessError) as error:
            self._fail(metadata, context.operation.operation_id, created_paths)
            raise OperationExecutionError(
                getattr(error, "code", "video_sequence_assembly_failed"),
                retryable=True,
            ) from error
        except Exception as error:
            self._fail(metadata, context.operation.operation_id, created_paths)
            raise OperationExecutionError(
                "video_sequence_assembly_failed", retryable=True
            ) from error

    def _resolve_artifact(self, artifact_id: str) -> tuple[MediaArtifact, Path]:
        matches: list[tuple[MediaArtifact, Path]] = []
        for item in self.history.list_all():
            try:
                artifact, path = resolve_registered_media_artifact(
                    item,
                    artifact_id,
                    accepted_roots=self._accepted_roots_for(item),
                    allowed_kinds={"video"},
                )
            except MediaArtifactAccessError:
                continue
            matches.append((artifact, path))
        if len(matches) != 1:
            raise VideoSequenceAssemblyError("video_sequence_accepted_artifact_missing")
        return matches[0]

    def _accepted_roots_for(self, item: dict[str, Any]) -> set[Path]:
        roots = self.accepted_roots
        settings = item.get("settings")
        if isinstance(settings, dict):
            for key in ("output_root", "output_directory"):
                value = settings.get(key)
                if isinstance(value, str) and value.strip():
                    roots.add(Path(value))
        return roots

    def _fail(self, metadata: dict[str, Any], operation_id: str, paths: list[Path]) -> None:
        for path in dict.fromkeys(paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        item = self.history.get(metadata["generation_id"])
        if item is not None and item["status"] != "completed":
            self.history.update_generation(
                metadata["generation_id"],
                status="failed",
                settings=dict(item["settings"]),
                output_paths=list(item["output_paths"]),
            )
        self.sequences.mark_assembly_failed(
            metadata["sequence_id"], operation_id=operation_id
        )

    @staticmethod
    def _metadata(snapshot: dict) -> dict[str, Any]:
        try:
            sequence_id = str(snapshot["sequence_id"])
            generation_id = int(snapshot["generation_id"])
            assembly_revision = int(snapshot["assembly_revision"])
            plan_identity = snapshot["plan_identity"]
            output_root = str(snapshot["output_root"])
        except (KeyError, TypeError, ValueError) as error:
            raise OperationExecutionError(
                "video_sequence_assembly_snapshot_invalid", retryable=False
            ) from error
        if (
            not sequence_id
            or generation_id < 1
            or assembly_revision < 1
            or not isinstance(plan_identity, dict)
            or not output_root.strip()
        ):
            raise OperationExecutionError(
                "video_sequence_assembly_snapshot_invalid", retryable=False
            )
        return {
            "sequence_id": sequence_id,
            "generation_id": generation_id,
            "assembly_revision": assembly_revision,
            "plan_identity": plan_identity,
            "output_root": output_root,
        }

    @staticmethod
    def _plan_identity(plan: SequenceAssemblyPlan) -> dict[str, Any]:
        return {
            "active_chain_revision": plan.active_chain_revision,
            "segment_ids": [clip.segment_id for clip in plan.clips],
            "accepted_revisions": [clip.accepted_revision for clip in plan.clips],
            "artifact_ids": [clip.artifact.artifact_id for clip in plan.clips],
            "source_sha256": [clip.artifact.sha256 for clip in plan.clips],
            "context_sha256": [clip.context_sha256 for clip in plan.clips],
            "assembly_mode": plan.assembly_mode,
            "working_artifact_ids": [
                clip.working_artifact.artifact_id
                if clip.working_artifact is not None
                else None
                for clip in plan.clips
            ],
            "working_operation_ids": [
                clip.working_artifact.operation_id
                if clip.working_artifact is not None
                else None
                for clip in plan.clips
            ],
            "working_generation_ids": [
                clip.working_artifact.generation_id
                if clip.working_artifact is not None
                else None
                for clip in plan.clips
            ],
        }


class VideoSequenceJobService:
    def __init__(
        self,
        *,
        sequences: VideoSequenceRepository,
        coordinator: OperationCoordinator,
        history: HistoryRepository,
        packet_builder: VideoPacketBuilder | None = None,
        context_preparer: VideoSequenceContextPreparer | None = None,
        readiness_check: Callable[[VideoRequest], bool] | None = None,
        resource_advisor_provider: Callable[[], ResourceAdvisor] | None = None,
        resource_observation_provider: Callable[[], ResourceObservationSnapshot] | None = None,
        output_root_provider: Callable[[], Path] | None = None,
    ) -> None:
        self.sequences = sequences
        self.coordinator = coordinator
        self.history = history
        self.packet_builder = packet_builder or VideoPacketBuilder()
        self.context_preparer = context_preparer or VideoSequenceContextPreparer()
        self.readiness_check = readiness_check
        self.output_root_provider = output_root_provider
        self._authorization = VideoOperationService(
            coordinator=coordinator,
            history=history,
            packet_builder=self.packet_builder,
            readiness_check=readiness_check,
            resource_advisor_provider=resource_advisor_provider,
            resource_observation_provider=resource_observation_provider,
        )

    def create(self, request: VideoRequest, *, receipt_id: str) -> VideoSequence:
        if request.mode != "sequence":
            raise VideoLocalAdapterError("video_sequence_request_required")
        now = datetime.now(UTC)
        sequence_id = f"sequence-{sha256(receipt_id.encode('utf-8')).hexdigest()[:32]}"
        segment_id = f"{sequence_id}:segment:0"
        adapter = LocalH3VideoAdapter()
        capability = adapter.capabilities()
        if self.readiness_check is not None and self.readiness_check(request):
            capability = capability_after_verified_setup(capability)
        sequence = VideoSequence(
            sequence_id=sequence_id,
            provider=request.provider,
            model_profile=request.model_profile,
            shared_settings_snapshot={
                "size": request.size.model_dump(mode="json"),
                "fps": request.fps,
                "audio": request.audio.model_dump(mode="json"),
            },
            capability_snapshot=capability.model_dump(mode="json"),
            active_chain_revision=1,
            active_segment_ids=(),
            status="draft",
            segments=(
                VideoSequenceSegment(
                    segment_id=segment_id,
                    ordinal=0,
                    request_snapshot=request,
                    status="draft",
                ),
            ),
            revision=1,
            created_at=now,
            updated_at=now,
        )
        return self.sequences.create(sequence, receipt_id=receipt_id)

    def list(self, *, limit: int = 100) -> list[VideoSequence]:
        return self.sequences.list(limit=limit)

    def get(self, sequence_id: str) -> VideoSequence:
        return self.sequences.get(sequence_id)

    def generate(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int,
        receipt_id: str,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
        retry: bool = False,
    ) -> SubmittedVideoSequenceClip:
        submission_hash = self._submission_hash(
            sequence_id=sequence_id,
            segment_id=segment_id,
            expected_revision=expected_revision,
            expected_segment_revision=expected_segment_revision,
            resource_authorization=resource_authorization,
            run_anyway=run_anyway,
            retry=retry,
        )
        operation_id = sha256(
            f"video-sequence:{sequence_id}:{segment_id}:{receipt_id}".encode("utf-8")
        ).hexdigest()
        try:
            existing = self.coordinator.get(operation_id)
        except OperationNotFoundError:
            existing = None
        if existing is not None:
            existing_sequence = existing.request_snapshot.get("sequence", {})
            if existing_sequence.get("submission_hash") != submission_hash:
                raise VideoSequenceConflictError(
                    "video sequence receipt was reused for another request"
                )
            generation_id = int(existing.request_snapshot["generation_id"])
            return SubmittedVideoSequenceClip(
                sequence=self.sequences.get(sequence_id),
                operation=existing,
                generation_id=generation_id,
            )

        current = self.sequences.get(sequence_id)
        segment = self._checked_segment(
            current,
            segment_id,
            expected_revision,
            expected_segment_revision,
        )
        setup_verified = self.readiness_check is not None
        if setup_verified and not self.readiness_check(segment.request_snapshot):
            raise VideoLocalAdapterError("video_local_setup_required")
        capability = LocalH3VideoAdapter().capabilities()
        if setup_verified:
            capability = capability_after_verified_setup(capability)
        packet, parent_artifact_id, guide_duration_seconds = self._packet_for(
            current,
            segment,
            capability=capability,
        )
        execution_request = VideoRequest.model_validate(packet.request_snapshot)
        accepted_authorization = self._authorization._validate_resource_authorization(
            execution_request,
            resource_authorization,
            run_anyway=run_anyway,
            workload=_resource_workload(
                execution_request,
                guide_duration_seconds=guide_duration_seconds,
            ),
        )
        operation_output_root = (
            Path(self.output_root_provider()).expanduser().resolve()
            if self.output_root_provider is not None
            else None
        )
        generation_id, settings, actual_frames = self._create_history(
            current,
            segment,
            packet,
            accepted_authorization,
            output_root=operation_output_root,
        )
        transition = self.sequences.retry_segment if retry else self.sequences.start_generation
        try:
            generating = transition(
                sequence_id,
                segment_id=segment_id,
                expected_revision=expected_revision,
                expected_segment_revision=expected_segment_revision,
                operation_id=operation_id,
                receipt_id=f"generate:{receipt_id}",
            )
            active_segment = next(
                item for item in generating.segments if item.segment_id == segment_id
            )
            operation = self.coordinator.enqueue(
                kind=VIDEO_SEQUENCE_CLIP_OPERATION_KIND,
                source_actor=source_actor,
                request_snapshot={
                    "generation_id": generation_id,
                    "packet": packet.to_dict(),
                    "sequence": {
                        "sequence_id": sequence_id,
                        "segment_id": segment_id,
                        "sequence_revision": generating.revision,
                        "segment_revision": active_segment.revision,
                        "parent_artifact_id": parent_artifact_id,
                        "submission_hash": submission_hash,
                    },
                    **(
                        {"output_root": str(operation_output_root)}
                        if operation_output_root is not None
                        else {}
                    ),
                },
                idempotency_key=f"video-sequence:{operation_id[:40]}",
                request_contract_version=VIDEO_SEQUENCE_OPERATION_CONTRACT_VERSION,
                progress_total=actual_frames,
                progress_unit="frames",
                operation_id=operation_id,
            )
        except Exception:
            self.history.update_generation(
                generation_id,
                status="failed",
                settings={**settings, "error_code": "video_sequence_enqueue_failed"},
                output_paths=[],
            )
            self.sequences.mark_operation_retryable(
                sequence_id,
                segment_id=segment_id,
                operation_id=operation_id,
                transition_receipt_id=f"generate:{receipt_id}",
            )
            raise
        return SubmittedVideoSequenceClip(
            sequence=generating,
            operation=operation,
            generation_id=generation_id,
        )

    @staticmethod
    def _submission_hash(
        *,
        sequence_id: str,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int,
        resource_authorization: ResourceAdvisoryAuthorization | None,
        run_anyway: bool,
        retry: bool,
    ) -> str:
        payload = {
            "sequence_id": sequence_id,
            "segment_id": segment_id,
            "expected_revision": expected_revision,
            "expected_segment_revision": expected_segment_revision,
            "resource_authorization": (
                resource_authorization.model_dump(mode="json")
                if resource_authorization is not None
                else None
            ),
            "run_anyway": run_anyway,
            "retry": retry,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()

    def cancel(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        operation_id: str,
        expected_revision: int,
        expected_segment_revision: int,
        receipt_id: str,
    ) -> OperationRecord:
        receipt_payload = {
            "segment_id": segment_id,
            "operation_id": operation_id,
            "expected_revision": expected_revision,
            "expected_segment_revision": expected_segment_revision,
        }
        if self.sequences.action_receipt_replay(
            receipt_id=receipt_id,
            sequence_id=sequence_id,
            action="cancel",
            request_payload=receipt_payload,
        ):
            return self.coordinator.get(operation_id)
        current = self.sequences.get(sequence_id)
        segment = self._checked_segment(
            current, segment_id, expected_revision, expected_segment_revision
        )
        if segment.operation_id != operation_id:
            raise VideoSequenceConflictError("video sequence operation is stale")
        if segment.status != "generating":
            raise VideoSequenceConflictError("video sequence segment is not generating")
        active = self.coordinator.get(operation_id)
        if active.status not in ACTIVE_OPERATION_STATUSES:
            raise VideoSequenceConflictError("video sequence operation is not active")
        operation = self.coordinator.request_cancel(operation_id)
        if operation.status not in {"cancelling", "cancelled"}:
            raise VideoSequenceConflictError("video sequence operation is not cancellable")
        self.sequences.store_action_receipt(
            receipt_id=receipt_id,
            sequence_id=sequence_id,
            action="cancel",
            request_payload=receipt_payload,
        )
        if operation.status == "cancelled":
            self.sequences.mark_operation_retryable(
                sequence_id, segment_id=segment_id, operation_id=operation_id
            )
        return operation

    def accept_and_continue(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        next_request: VideoRequest,
        expected_revision: int,
        expected_segment_revision: int,
        receipt_id: str,
    ) -> VideoSequence:
        current = self.sequences.get(sequence_id)
        segment = self._checked_segment(
            current, segment_id, expected_revision, expected_segment_revision
        )
        if next_request.mode != "sequence":
            raise VideoLocalAdapterError("video_sequence_request_required")
        next_id = f"{sequence_id}:segment:{segment.ordinal + 1}"
        return self.sequences.accept_and_continue(
            sequence_id,
            segment_id=segment_id,
            candidate_artifact_id=candidate_artifact_id,
            next_segment_id=next_id,
            next_request_snapshot=next_request,
            expected_revision=expected_revision,
            expected_segment_revision=expected_segment_revision,
            receipt_id=receipt_id,
        )

    def accept_and_finish(
        self,
        sequence_id: str,
        *,
        segment_id: str,
        candidate_artifact_id: str,
        expected_revision: int,
        expected_segment_revision: int,
        receipt_id: str,
    ) -> SubmittedVideoSequenceAssembly:
        current = self.sequences.get(sequence_id)
        assembly_revision = max(1, current.assembly_revision + 1)
        operation_id = sha256(
            f"video-sequence-assembly:{sequence_id}:{receipt_id}".encode("utf-8")
        ).hexdigest()
        accepted = self.sequences.accept_and_finish(
            sequence_id,
            segment_id=segment_id,
            candidate_artifact_id=candidate_artifact_id,
            expected_revision=expected_revision,
            expected_segment_revision=expected_segment_revision,
            assembly_operation_id=operation_id,
            assembly_revision=assembly_revision,
            receipt_id=receipt_id,
        )
        generation_id = self._ensure_assembly_history(
            accepted,
            operation_id,
            output_root=self._assembly_output_root(accepted),
        )
        accepted = self.sequences.attach_assembly_generation(
            sequence_id,
            operation_id=operation_id,
            generation_id=generation_id,
        )
        snapshot = self._assembly_operation_snapshot(accepted)
        try:
            operation = self.coordinator.get(operation_id)
            if operation.request_snapshot != snapshot:
                raise VideoSequenceConflictError(
                    "video sequence assembly receipt was reused for another request"
                )
        except OperationNotFoundError:
            operation = self.coordinator.enqueue(
                kind=VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND,
                source_actor="local",
                request_snapshot=snapshot,
                idempotency_key=receipt_id,
                request_contract_version=VIDEO_SEQUENCE_OPERATION_CONTRACT_VERSION,
                progress_total=len(accepted.active_segment_ids),
                progress_unit="clips",
                operation_id=operation_id,
            )
        return SubmittedVideoSequenceAssembly(
            sequence=accepted,
            operation=operation,
            generation_id=generation_id,
        )

    def retry_assembly(
        self,
        sequence_id: str,
        *,
        expected_revision: int,
        receipt_id: str,
        source_actor: str,
    ) -> SubmittedVideoSequenceAssembly:
        current = self.sequences.get(sequence_id)
        if current.revision != expected_revision:
            raise VideoSequenceConflictError("video sequence revision is stale")
        if (
            current.status != "failed"
            or current.assembly_operation_id is None
            or current.assembly_generation_id is None
        ):
            raise VideoSequenceConflictError("video sequence assembly is not retryable")
        self._ensure_existing_assembly_output_root(current)
        previous_operation_id = current.assembly_operation_id
        operation_id = sha256(
            f"video-sequence-assembly-retry:{sequence_id}:{receipt_id}".encode("utf-8")
        ).hexdigest()
        assembling = self.sequences.start_assembly_retry(
            sequence_id,
            previous_operation_id=previous_operation_id,
            operation_id=operation_id,
            expected_revision=expected_revision,
            receipt_id=receipt_id,
        )
        snapshot = self._assembly_operation_snapshot(assembling)
        try:
            self.coordinator.get(previous_operation_id)
            parent_operation_id: str | None = previous_operation_id
        except OperationNotFoundError:
            parent_operation_id = None
        try:
            operation = self.coordinator.get(operation_id)
            if operation.request_snapshot != snapshot:
                raise VideoSequenceConflictError(
                    "video sequence assembly receipt was reused for another request"
                )
        except OperationNotFoundError:
            operation = self.coordinator.enqueue(
                kind=VIDEO_SEQUENCE_ASSEMBLE_OPERATION_KIND,
                source_actor=source_actor,
                request_snapshot=snapshot,
                idempotency_key=receipt_id,
                request_contract_version=VIDEO_SEQUENCE_OPERATION_CONTRACT_VERSION,
                progress_total=len(assembling.active_segment_ids),
                progress_unit="clips",
                parent_operation_id=parent_operation_id,
                operation_id=operation_id,
            )
        return SubmittedVideoSequenceAssembly(
            sequence=assembling,
            operation=operation,
            generation_id=assembling.assembly_generation_id or 0,
        )

    def _ensure_assembly_history(
        self,
        sequence: VideoSequence,
        operation_id: str,
        *,
        output_root: Path,
    ) -> int:
        for item in self.history.list_all():
            settings = item.get("settings", {})
            if (
                isinstance(settings, dict)
                and settings.get("assembly_operation_id") == operation_id
                and settings.get("sequence_id") == sequence.sequence_id
            ):
                return int(item["id"])
        accepted = [
            segment
            for segment in sequence.segments
            if segment.segment_id in sequence.active_segment_ids
        ]
        return self.history.add(
            status="queued",
            profile=sequence.model_profile,
            prompt="\n\n".join(segment.request_snapshot.prompt for segment in accepted),
            expanded_prompt="\n\n".join(
                segment.request_snapshot.prompt for segment in accepted
            ),
            negative_prompt="",
            settings={
                "schema_version": "asteria.video.sequence.assembly.history.v1",
                "sequence_id": sequence.sequence_id,
                "active_chain_revision": sequence.active_chain_revision,
                "assembly_revision": sequence.assembly_revision,
                "assembly_operation_id": operation_id,
                "output_root": str(output_root),
                "segment_ids": list(sequence.active_segment_ids),
                "accepted_revisions": [segment.accepted_revision for segment in accepted],
            },
            model_slots={},
            loras=[],
            embeddings=[],
            output_paths=[],
            media_kind="video",
            artifacts=[],
        )

    def _assembly_operation_snapshot(self, sequence: VideoSequence) -> dict[str, Any]:
        if sequence.assembly_generation_id is None:
            raise VideoSequenceConflictError("video sequence assembly generation is missing")
        identity = {
            "active_chain_revision": sequence.active_chain_revision,
            "segment_ids": [],
            "accepted_revisions": [],
            "artifact_ids": [],
            "source_sha256": [],
            "context_sha256": [],
            "assembly_mode": "encoded_artifact_fallback",
            "working_artifact_ids": [],
            "working_operation_ids": [],
            "working_generation_ids": [],
        }
        all_working = True
        direct_context_chain = True
        for segment_id in sequence.active_segment_ids:
            segment = next(item for item in sequence.segments if item.segment_id == segment_id)
            if segment.accepted_candidate_id is None or segment.accepted_revision is None:
                raise VideoSequenceConflictError("video sequence active chain is invalid")
            source = self._source_artifact(
                segment.accepted_candidate_id,
                segment.accepted_revision,
            )
            identity["segment_ids"].append(segment.segment_id)
            identity["accepted_revisions"].append(segment.accepted_revision)
            identity["artifact_ids"].append(segment.accepted_candidate_id)
            identity["source_sha256"].append(source.sha256)
            identity["context_sha256"].append(
                segment.context_receipt.get("context_sha256")
                if segment.context_receipt is not None
                else None
            )
            working = segment.candidate_working_artifacts.get(
                segment.accepted_candidate_id
            )
            if not isinstance(working, dict):
                all_working = False
                identity["working_artifact_ids"].append(None)
                identity["working_operation_ids"].append(None)
                identity["working_generation_ids"].append(None)
            else:
                identity["working_artifact_ids"].append(working.get("artifact_id"))
                identity["working_operation_ids"].append(working.get("operation_id"))
                identity["working_generation_ids"].append(working.get("generation_id"))
            if segment.ordinal > 0 and (
                not isinstance(segment.context_receipt, dict)
                or segment.context_receipt.get("method") != "direct_context"
            ):
                direct_context_chain = False
        if all_working and direct_context_chain:
            identity["assembly_mode"] = "accepted_latent_crossfade"
        history_item = self.history.get(sequence.assembly_generation_id)
        settings = history_item.get("settings") if history_item is not None else None
        output_root = settings.get("output_root") if isinstance(settings, dict) else None
        if not isinstance(output_root, str) or not output_root.strip():
            raise VideoSequenceConflictError(
                "video sequence assembly output root is missing"
            )
        return {
            "sequence_id": sequence.sequence_id,
            "generation_id": sequence.assembly_generation_id,
            "assembly_revision": sequence.assembly_revision,
            "plan_identity": identity,
            "output_root": output_root,
        }

    def _assembly_output_root(self, sequence: VideoSequence) -> Path:
        if self.output_root_provider is not None:
            return Path(self.output_root_provider()).expanduser().resolve()
        roots: list[Path] = []
        accepted_ids = {
            segment.accepted_candidate_id
            for segment in sequence.segments
            if segment.segment_id in sequence.active_segment_ids
        }
        for item in self.history.list_all():
            if not any(
                isinstance(raw, dict) and raw.get("artifact_id") in accepted_ids
                for raw in item.get("artifacts", [])
            ):
                continue
            settings = item.get("settings")
            value = settings.get("output_root") if isinstance(settings, dict) else None
            if isinstance(value, str) and value.strip():
                roots.append(Path(value).expanduser().resolve())
        if roots:
            return roots[-1]
        raise VideoSequenceConflictError(
            "video sequence assembly output root is unavailable"
        )

    def _ensure_existing_assembly_output_root(self, sequence: VideoSequence) -> None:
        generation_id = sequence.assembly_generation_id
        if generation_id is None:
            raise VideoSequenceConflictError(
                "video sequence assembly generation is missing"
            )
        item = self.history.get(generation_id)
        if item is None:
            raise VideoSequenceConflictError(
                "video sequence assembly history is missing"
            )
        settings = item.get("settings")
        if isinstance(settings, dict) and isinstance(settings.get("output_root"), str):
            if settings["output_root"].strip():
                return
        repaired = dict(settings) if isinstance(settings, dict) else {}
        repaired["output_root"] = str(self._assembly_output_root(sequence))
        self.history.update_generation(
            generation_id,
            status=str(item["status"]),
            settings=repaired,
            output_paths=list(item["output_paths"]),
        )

    def _packet_for(
        self,
        sequence: VideoSequence,
        segment: VideoSequenceSegment,
        *,
        capability: ProviderCapabilityDescriptor | None = None,
    ) -> tuple[VideoSubmissionPacket, str | None, float]:
        adapter = LocalH3VideoAdapter()
        if segment.ordinal == 0:
            return (
                self.packet_builder.build_sequence_initial(
                    segment.request_snapshot,
                    adapter,
                    capability=capability,
                ),
                None,
                0,
            )
        predecessor = sequence.segments[segment.ordinal - 1]
        if (
            predecessor.segment_id not in sequence.active_segment_ids
            or predecessor.accepted_candidate_id is None
            or predecessor.accepted_revision is None
        ):
            raise VideoSequenceConflictError("video sequence predecessor is not accepted")
        source = self._source_artifact(
            predecessor.accepted_candidate_id,
            predecessor.accepted_revision,
            working_artifact_receipt=predecessor.candidate_working_artifacts.get(
                predecessor.accepted_candidate_id
            ),
        )
        context = self.context_preparer.prepare(segment.request_snapshot, source, adapter)
        return (
            self.packet_builder.build_sequence(
                segment.request_snapshot,
                adapter,
                context,
                capability=capability,
            ),
            predecessor.accepted_candidate_id,
            source.duration_seconds,
        )

    def _source_artifact(
        self,
        artifact_id: str,
        accepted_revision: int,
        *,
        working_artifact_receipt: Any = None,
    ) -> VideoSequenceSourceArtifact:
        for item in self.history.list_all():
            for raw in item.get("artifacts", []):
                try:
                    artifact = MediaArtifact.model_validate(raw)
                except ValueError:
                    continue
                if artifact.artifact_id != artifact_id or artifact.kind != "video":
                    continue
                fps = artifact.frame_rate
                if (
                    artifact.sha256 is None
                    or artifact.width is None
                    or artifact.height is None
                    or artifact.duration_ms is None
                    or fps is None
                ):
                    raise VideoLocalAdapterError("video_sequence_source_metadata_missing")
                duration = artifact.duration_ms / 1000
                sample_count = (
                    max(1, round(duration * artifact.sample_rate))
                    if artifact.has_audio and artifact.sample_rate
                    else None
                )
                return VideoSequenceSourceArtifact(
                    artifact_id=artifact.artifact_id,
                    accepted_revision=accepted_revision,
                    sha256=artifact.sha256,
                    width=artifact.width,
                    height=artifact.height,
                    fps=fps,
                    frame_count=max(1, round(duration * fps)),
                    duration_seconds=duration,
                    audio_sample_rate=(artifact.sample_rate if sample_count is not None else None),
                    audio_sample_count=sample_count,
                    working_artifact_id=(
                        str(working_artifact_receipt["artifact_id"])
                        if isinstance(working_artifact_receipt, dict)
                        else None
                    ),
                    working_operation_id=(
                        str(working_artifact_receipt["operation_id"])
                        if isinstance(working_artifact_receipt, dict)
                        else None
                    ),
                )
        raise VideoLocalAdapterError("video_sequence_source_not_found")

    def _create_history(
        self,
        sequence: VideoSequence,
        segment: VideoSequenceSegment,
        packet: VideoSubmissionPacket,
        authorization: ResourceAdvisoryAuthorization | None,
        *,
        output_root: Path | None,
    ) -> tuple[int, dict[str, Any], int]:
        requested_frames, actual_frames = normalized_h3_frame_count(
            float(packet.request_snapshot["duration_seconds"]),
            float(packet.request_snapshot["fps"]),
        )
        settings: dict[str, Any] = {
            "schema_version": "asteria.video.history.v1",
            "request_snapshot": packet.request_snapshot,
            "sequence_request_snapshot": segment.request_snapshot.model_dump(mode="json"),
            "capability_snapshot": packet.capability_snapshot,
            "provider": sequence.provider,
            "adapter_id": packet.adapter_id,
            "adapter_revision": packet.adapter_revision,
            "mapping_id": packet.mapping_id,
            "mapping_version": packet.mapping_version,
            "submission_runtime_job_id": packet.runtime_job_id,
            "requested_frame_count": requested_frames,
            "actual_frame_count": actual_frames,
            "sequence_id": sequence.sequence_id,
            "segment_id": segment.segment_id,
            "active_chain_revision": sequence.active_chain_revision,
            "resource_authorization": (
                authorization.model_dump(mode="json") if authorization is not None else None
            ),
        }
        if output_root is not None:
            settings["output_root"] = str(output_root)
        generation_id = self.history.add(
            status="queued",
            profile=sequence.model_profile,
            prompt=segment.request_snapshot.prompt,
            expanded_prompt=segment.request_snapshot.prompt,
            negative_prompt=segment.request_snapshot.negative_prompt or "",
            settings=settings,
            model_slots={},
            loras=[],
            embeddings=[],
            output_paths=[],
            media_kind="video",
            artifacts=[],
        )
        return generation_id, settings, actual_frames

    @staticmethod
    def _checked_segment(
        sequence: VideoSequence,
        segment_id: str,
        expected_revision: int,
        expected_segment_revision: int,
    ) -> VideoSequenceSegment:
        if sequence.revision != expected_revision:
            raise VideoSequenceConflictError("video sequence revision is stale")
        segment = next(
            (item for item in sequence.segments if item.segment_id == segment_id),
            None,
        )
        if segment is None:
            raise VideoSequenceNotFoundError("video sequence segment was not found")
        if segment.revision != expected_segment_revision:
            raise VideoSequenceConflictError("video sequence segment revision is stale")
        return segment
