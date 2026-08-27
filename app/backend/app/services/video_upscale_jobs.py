from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import stat
from typing import Callable

from pydantic import ValidationError

from app.schemas.media import MediaArtifact
from app.schemas.operations import OperationRecord
from app.schemas.resource_advisory import ResourceAdvisoryAuthorization
from app.schemas.video import VideoSize
from app.schemas.video_upscale import VideoUpscaleRequest, VideoUpscaleWorkBundle
from app.services.atomic_json import write_json_atomic
from app.services.generation_working_artifacts import (
    GenerationWorkingArtifactError,
    GenerationWorkingArtifactRegistry,
)
from app.services.history import HistoryRepository
from app.services.media_artifacts import (
    MediaArtifactAccessError,
    MediaFinalizationResult,
    file_sha256,
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
from app.services.video_executor import VideoOperationService, finalize_non_image_output
from app.services.video_upscale_adapters import (
    VideoUpscaleAdapter,
    VideoUpscaleAdapterError,
    VideoUpscaleExecutionInput,
    VideoUpscaleExecutionResult,
)
from app.services.video_upscale_bundles import canonical_sha256


VIDEO_UPSCALE_OPERATION_KIND = "video.upscale"
VIDEO_UPSCALE_OPERATION_CONTRACT_VERSION = 1
VIDEO_UPSCALE_ADAPTER_UPSTREAM_COMMIT = (
    "2553ad1b66ee0956df02e9146dd78b93395f9f69"
)
MAX_UPSCALE_BUNDLE_BYTES = 16 * 1024 * 1024


class VideoUpscaleJobError(ValueError):
    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ResolvedVideoUpscaleSource:
    request: VideoUpscaleRequest
    source_item: dict
    source_artifact: MediaArtifact
    source_path: Path
    source_output_root: Path
    bundle: VideoUpscaleWorkBundle
    bundle_path: Path
    latent_path: Path
    bundle_operation_id: str
    bundle_artifact_id: str


@dataclass(frozen=True, slots=True)
class SubmittedVideoUpscaleOperation:
    operation: OperationRecord
    generation_id: int


def resolve_video_upscale_target(
    request: VideoUpscaleRequest,
    bundle: VideoUpscaleWorkBundle,
) -> tuple[int, int]:
    if request.target.scale is not None:
        width = int(round(bundle.actual_width * request.target.scale / 32.0)) * 32
        height = int(round(bundle.actual_height * request.target.scale / 32.0)) * 32
    else:
        assert request.target.width is not None and request.target.height is not None
        width = request.target.width
        height = request.target.height
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise VideoUpscaleJobError("video_upscale_target_unsupported")
    if width <= bundle.actual_width and height <= bundle.actual_height:
        raise VideoUpscaleJobError("video_upscale_target_not_larger")
    return width, height


class VideoUpscaleSourceResolver:
    def __init__(self, history: HistoryRepository) -> None:
        self.history = history

    def resolve(self, request: VideoUpscaleRequest) -> ResolvedVideoUpscaleSource:
        item = self.history.get(request.source_generation_id)
        if item is None or item.get("status") != "completed" or item.get("media_kind") != "video":
            raise VideoUpscaleJobError("video_upscale_source_unavailable")
        settings = item.get("settings")
        if not isinstance(settings, dict):
            raise VideoUpscaleJobError("video_upscale_source_unavailable")
        output_root_value = settings.get("output_root")
        operation_id = settings.get("upscale_work_bundle_operation_id")
        bundle_artifact_id = settings.get("upscale_work_bundle_id")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (output_root_value, operation_id, bundle_artifact_id)
        ):
            raise VideoUpscaleJobError("video_upscale_work_bundle_unavailable")
        try:
            output_root = Path(output_root_value).expanduser().resolve(strict=True)
            source_artifact, source_path = resolve_registered_media_artifact(
                item,
                request.source_artifact_id,
                accepted_roots={output_root},
                allowed_kinds={"video"},
            )
        except (OSError, RuntimeError, ValueError, MediaArtifactAccessError) as error:
            raise VideoUpscaleJobError("video_upscale_source_unavailable") from error
        registry = GenerationWorkingArtifactRegistry(output_root)
        try:
            bundle_path = registry.resolve_for_adapter(
                operation_id=operation_id,
                artifact_id=bundle_artifact_id,
                expected_kind="upscale_work_bundle",
            )
            if bundle_path.stat().st_size > MAX_UPSCALE_BUNDLE_BYTES:
                raise VideoUpscaleJobError("video_upscale_work_bundle_invalid")
            bundle = VideoUpscaleWorkBundle.model_validate_json(
                bundle_path.read_text(encoding="utf-8")
            )
            latent_path = registry.resolve_for_adapter(
                operation_id=operation_id,
                artifact_id=bundle.latent_artifact_id,
                expected_kind="av_latent",
            )
        except VideoUpscaleJobError:
            raise
        except (GenerationWorkingArtifactError, OSError, ValidationError, ValueError) as error:
            raise VideoUpscaleJobError("video_upscale_work_bundle_unavailable") from error
        source_sha = file_sha256(source_path)
        latent_sha = file_sha256(latent_path)
        packet = dict(bundle.packet_snapshot)
        if bundle.request_snapshot.schema_version != "asteria.video.request.v2":
            raise VideoUpscaleJobError("video_upscale_source_request_unsupported")
        if bundle.request_snapshot.mode == "sequence":
            raise VideoUpscaleJobError("video_upscale_sequence_source_unsupported")
        if (
            bundle.source_generation_id != request.source_generation_id
            or bundle.source_media_artifact_id != request.source_artifact_id
            or Path(bundle.actual_output_root).expanduser().resolve() != output_root
            or bundle.working_operation_id != operation_id
            or source_artifact.sha256 is None
            or bundle.source_media_sha256 != source_artifact.sha256
            or source_sha != bundle.source_media_sha256
            or latent_sha != bundle.latent_sha256
            or canonical_sha256(packet) != bundle.packet_sha256
            or packet.get("request_snapshot")
            != bundle.request_snapshot.model_dump(mode="json")
            or canonical_sha256(
                {
                    "request": packet.get("request_snapshot"),
                    "conditioning": packet.get("provider_payload", {}).get("conditioning"),
                    "model_selection": packet.get("provider_payload", {}).get("model_selection"),
                    "loras": packet.get("provider_payload", {}).get("loras"),
                    "mapping_id": packet.get("mapping_id"),
                    "mapping_revision": packet.get("mapping_version"),
                }
            )
            != bundle.conditioning_sha256
        ):
            raise VideoUpscaleJobError("video_upscale_work_bundle_changed")
        resolve_video_upscale_target(request, bundle)
        return ResolvedVideoUpscaleSource(
            request=request,
            source_item=item,
            source_artifact=source_artifact,
            source_path=source_path,
            source_output_root=output_root,
            bundle=bundle,
            bundle_path=bundle_path,
            latent_path=latent_path,
            bundle_operation_id=operation_id,
            bundle_artifact_id=bundle_artifact_id,
        )


class VideoUpscaleOperationService:
    def __init__(
        self,
        *,
        coordinator: OperationCoordinator,
        history: HistoryRepository,
        source_resolver: VideoUpscaleSourceResolver,
        video_service: VideoOperationService,
        output_root_provider: Callable[[], Path],
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.history = history
        self.source_resolver = source_resolver
        self.video_service = video_service
        self.output_root_provider = output_root_provider
        self.readiness_check = readiness_check

    def submit(
        self,
        request: VideoUpscaleRequest,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ) -> SubmittedVideoUpscaleOperation:
        if self.readiness_check is not None and not self.readiness_check():
            raise VideoUpscaleJobError("video_upscale_setup_required")
        self._validate_supported_request(request)
        source = self.source_resolver.resolve(request)
        target_width, target_height = resolve_video_upscale_target(request, source.bundle)
        self._validate_supported_request(
            request,
            target=(target_width, target_height),
        )
        resolved_seed = (
            secrets.randbelow(9223372036854775808)
            if request.sampling.seed == -1
            else request.sampling.seed
        )
        advisory_request = source.bundle.request_snapshot.model_copy(
            update={
                "size": VideoSize(
                    width=target_width,
                    height=target_height,
                    aspect_ratio="adaptive",
                    resolution_intent="custom",
                ),
                "seed": resolved_seed,
            }
        )
        verified = self.video_service.validate_resource_authorization(
            advisory_request,
            resource_authorization,
            run_anyway=run_anyway,
        )
        authorization_payload = (
            verified.model_dump(mode="json") if verified is not None else None
        )
        output_root = Path(self.output_root_provider()).expanduser().resolve()
        settings = {
            "schema_version": "asteria.video-upscale.history.v1",
            "request_snapshot": request.model_dump(mode="json"),
            "source_generation_id": request.source_generation_id,
            "source_artifact_id": request.source_artifact_id,
            "source_media_sha256": source.bundle.source_media_sha256,
            "source_bundle_artifact_id": source.bundle_artifact_id,
            "source_bundle_sha256": file_sha256(source.bundle_path),
            "source_latent_artifact_id": source.bundle.latent_artifact_id,
            "source_latent_sha256": source.bundle.latent_sha256,
            "adapter_id": "minimax_h3_ultimate_upscale",
            "adapter_revision": "1",
            "upstream_commit": VIDEO_UPSCALE_ADAPTER_UPSTREAM_COMMIT,
            "resource_authorization": authorization_payload,
            "target_width": target_width,
            "target_height": target_height,
            "resolved_seed": str(resolved_seed),
            "output_root": str(output_root),
        }
        generation_id = self.history.add(
            status="queued",
            profile=source.bundle.request_snapshot.model_profile,
            prompt=source.source_item.get("prompt", ""),
            expanded_prompt=source.source_item.get("expanded_prompt", ""),
            negative_prompt=source.source_item.get("negative_prompt", ""),
            settings=settings,
            model_slots={},
            loras=[],
            embeddings=[],
            output_paths=[],
            media_kind="video",
            artifacts=[],
        )
        snapshot = {
            "generation_id": generation_id,
            "request": request.model_dump(mode="json"),
            "source_output_root": str(source.source_output_root),
            "output_root": str(output_root),
            "bundle_operation_id": source.bundle_operation_id,
            "bundle_artifact_id": source.bundle_artifact_id,
            "bundle_sha256": settings["source_bundle_sha256"],
            "latent_artifact_id": source.bundle.latent_artifact_id,
            "latent_sha256": source.bundle.latent_sha256,
            "source_media_sha256": source.bundle.source_media_sha256,
            "target_width": target_width,
            "target_height": target_height,
            "resolved_seed": resolved_seed,
            "resource_authorization": authorization_payload,
        }
        try:
            operation = self.coordinator.enqueue(
                kind=VIDEO_UPSCALE_OPERATION_KIND,
                source_actor=source_actor,
                request_snapshot=snapshot,
                request_contract_version=VIDEO_UPSCALE_OPERATION_CONTRACT_VERSION,
                progress_total=1,
                progress_unit="result",
            )
        except Exception:
            failed = self.history.get(generation_id)
            if failed is not None:
                self.history.update_generation(
                    generation_id,
                    status="failed",
                    settings={**failed["settings"], "error_code": "video_upscale_enqueue_failed"},
                    output_paths=[],
                )
            raise
        return SubmittedVideoUpscaleOperation(operation=operation, generation_id=generation_id)

    def retry(
        self,
        operation_id: str,
        *,
        source_actor: str,
        resource_authorization: ResourceAdvisoryAuthorization | None = None,
        run_anyway: bool = False,
    ):
        operation = self.coordinator.get(operation_id)
        if operation.kind != VIDEO_UPSCALE_OPERATION_KIND:
            raise OperationNotFoundError("operation_not_found")
        if self.readiness_check is not None and not self.readiness_check():
            raise VideoUpscaleJobError("video_upscale_setup_required")
        try:
            request = VideoUpscaleRequest.model_validate(operation.request_snapshot["request"])
        except (KeyError, ValidationError, ValueError) as error:
            raise VideoUpscaleJobError("video_upscale_request_snapshot_invalid") from error
        self._validate_supported_request(request)
        generation_id = operation.request_snapshot.get("generation_id")
        if not isinstance(generation_id, int):
            raise VideoUpscaleJobError("video_upscale_request_snapshot_invalid")
        item = self.history.get(generation_id)
        if item is None:
            raise VideoUpscaleJobError("video_upscale_history_not_found")
        source = self.source_resolver.resolve(request)
        expected_target = resolve_video_upscale_target(request, source.bundle)
        try:
            stored_target = (
                int(operation.request_snapshot["target_width"]),
                int(operation.request_snapshot["target_height"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VideoUpscaleJobError(
                "video_upscale_request_snapshot_invalid"
            ) from error
        if stored_target != expected_target:
            raise VideoUpscaleJobError("video_upscale_request_snapshot_changed")
        self._validate_supported_request(
            request,
            target=stored_target,
        )
        advisory_request = source.bundle.request_snapshot.model_copy(
            update={
                "size": VideoSize(
                    width=stored_target[0],
                    height=stored_target[1],
                    aspect_ratio="adaptive",
                    resolution_intent="custom",
                )
            }
        )
        verified = self.video_service.validate_resource_authorization(
            advisory_request,
            resource_authorization,
            run_anyway=run_anyway,
        )
        authorization_payload = (
            verified.model_dump(mode="json") if verified is not None else None
        )
        retry_snapshot = dict(operation.request_snapshot)
        retry_snapshot["resource_authorization"] = authorization_payload
        settings = dict(item["settings"])
        settings.pop("error_code", None)
        settings["resource_authorization"] = authorization_payload
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
                request_snapshot=retry_snapshot,
            )
        except Exception:
            self.history.update_generation(
                generation_id,
                status=item["status"],
                settings=dict(item["settings"]),
                output_paths=list(item["output_paths"]),
            )
            raise

    @staticmethod
    def _validate_supported_request(
        request: VideoUpscaleRequest,
        *,
        target: tuple[int, int] | None = None,
    ) -> None:
        if request.sampling.cfg != 1:
            raise VideoUpscaleJobError("video_upscale_cfg_unsupported")
        if target is not None and request.spatial is not None:
            width, height = target
            if width < request.spatial.tile_width or height < request.spatial.tile_height:
                raise VideoUpscaleJobError("video_upscale_tile_exceeds_target")


class VideoUpscaleOperationExecutor:
    version = "video-h3-upscale-v1"

    def __init__(
        self,
        *,
        history: HistoryRepository,
        source_resolver: VideoUpscaleSourceResolver,
        adapter: VideoUpscaleAdapter,
        poster_root: Path,
        finalizer: Callable[..., MediaFinalizationResult] = finalize_non_image_output,
    ) -> None:
        self.history = history
        self.source_resolver = source_resolver
        self.adapter = adapter
        self.poster_root = Path(poster_root)
        self.finalizer = finalizer

    def resource_claims(self, _request_snapshot: dict) -> tuple[str, ...]:
        return ("comfy_gpu_exclusive",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        return self._execute(
            context.request_snapshot,
            context.operation.operation_id,
            progress=lambda current, total: context.report_progress(
                current, total, "result", phase="upscaling"
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
        snapshot: dict,
        operation_id: str,
        *,
        progress: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
    ) -> OperationExecutionResult:
        generation_id, request = self._validate_snapshot(snapshot)
        item = self.history.get(generation_id)
        if item is None:
            raise OperationExecutionError("video_upscale_history_not_found", retryable=False)
        if item["status"] == "completed":
            return self._registered_result(generation_id, item)
        settings = dict(item["settings"])
        settings.pop("error_code", None)
        self.history.update_generation(
            generation_id,
            status="running",
            settings=settings,
            output_paths=list(item["output_paths"]),
        )
        source_registry: GenerationWorkingArtifactRegistry | None = None
        begun: list[str] = []
        created: list[Path] = []
        runtime_job_id = ""
        try:
            source = self.source_resolver.resolve(request)
            self._snapshot_matches(snapshot, source)
            self._history_matches_snapshot(item, snapshot, request)
            source_registry = GenerationWorkingArtifactRegistry(source.source_output_root)
            for artifact_id in (source.bundle_artifact_id, source.bundle.latent_artifact_id):
                source_registry.begin_use(source.bundle_operation_id, artifact_id)
                begun.append(artifact_id)
            bundle_path = source_registry.resolve_for_adapter(
                operation_id=source.bundle_operation_id,
                artifact_id=source.bundle_artifact_id,
                expected_kind="upscale_work_bundle",
                allow_in_use=True,
            )
            latent_path = source_registry.resolve_for_adapter(
                operation_id=source.bundle_operation_id,
                artifact_id=source.bundle.latent_artifact_id,
                expected_kind="av_latent",
                allow_in_use=True,
            )
            output_root = Path(snapshot["output_root"]).expanduser().resolve()
            result = self.adapter.execute(
                VideoUpscaleExecutionInput(
                    request=request,
                    bundle=source.bundle,
                    bundle_path=bundle_path,
                    latent_path=latent_path,
                    output_root=output_root,
                    target_width=int(snapshot["target_width"]),
                    target_height=int(snapshot["target_height"]),
                    resolved_seed=int(snapshot["resolved_seed"]),
                ),
                operation_id=operation_id,
                progress_callback=progress,
                cancel_requested=cancel_requested,
                runtime_job_callback=lambda value: self._record_runtime_job(
                    generation_id, settings, value
                ),
            )
            runtime_job_id = result.runtime_job_id
            self._validate_output_paths(result.output_paths, output_root)
            created.extend(result.output_paths)
            if cancel_requested():
                raise VideoUpscaleAdapterError("video_upscale_cancelled", retryable=True)
            artifacts, output_paths = self._finalize(
                generation_id=generation_id,
                request=request,
                source=source,
                result=result,
                settings=settings,
            )
            created.extend(Path(value) for value in output_paths)
            settings.update(
                {
                    "runtime_job_id": runtime_job_id,
                    "workflow_id": result.workflow_id,
                    "actual_width": result.target_width,
                    "actual_height": result.target_height,
                }
            )
            self.history.complete_media_generation(
                generation_id,
                status="completed",
                settings=settings,
                media_kind="video",
                artifacts=artifacts,
                output_paths=output_paths,
            )
            self._consume(source_registry, source.bundle_operation_id, begun)
            return OperationExecutionResult(
                artifact_refs=(
                    f"history:{generation_id}",
                    *(artifact.artifact_id for artifact in artifacts),
                )
            )
        except VideoUpscaleAdapterError as error:
            self._discard(created)
            self._release(source_registry, snapshot.get("bundle_operation_id"), begun)
            settings.update(
                {
                    "error_code": error.code,
                    "runtime_job_id": runtime_job_id or settings.get("runtime_job_id"),
                }
            )
            self.history.update_generation(
                generation_id,
                status="cancelled" if error.code == "video_upscale_cancelled" else "failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            if error.code == "video_upscale_cancelled":
                raise OperationCancelled() from error
            raise OperationExecutionError(error.code, retryable=error.retryable) from error
        except (VideoUpscaleJobError, GenerationWorkingArtifactError, OSError, ValueError) as error:
            self._discard(created)
            self._release(source_registry, snapshot.get("bundle_operation_id"), begun)
            code = error.code if isinstance(error, VideoUpscaleJobError) else "video_upscale_finalize_failed"
            settings["error_code"] = code
            self.history.update_generation(
                generation_id,
                status="failed",
                settings=settings,
                output_paths=list(item["output_paths"]),
            )
            raise OperationExecutionError(code, retryable=False) from error

    @staticmethod
    def _validate_snapshot(snapshot: dict) -> tuple[int, VideoUpscaleRequest]:
        try:
            generation_id = int(snapshot["generation_id"])
            request = VideoUpscaleRequest.model_validate(snapshot["request"])
            if generation_id < 1:
                raise ValueError
        except (KeyError, TypeError, ValidationError, ValueError) as error:
            raise OperationExecutionError(
                "video_upscale_request_snapshot_invalid", retryable=False
            ) from error
        return generation_id, request

    @staticmethod
    def _snapshot_matches(snapshot: dict, source: ResolvedVideoUpscaleSource) -> None:
        expected = {
            "bundle_operation_id": source.bundle_operation_id,
            "bundle_artifact_id": source.bundle_artifact_id,
            "bundle_sha256": file_sha256(source.bundle_path),
            "latent_artifact_id": source.bundle.latent_artifact_id,
            "latent_sha256": source.bundle.latent_sha256,
            "source_media_sha256": source.bundle.source_media_sha256,
        }
        if any(snapshot.get(key) != value for key, value in expected.items()):
            raise VideoUpscaleJobError("video_upscale_request_snapshot_changed")
        target = resolve_video_upscale_target(source.request, source.bundle)
        if target != (snapshot.get("target_width"), snapshot.get("target_height")):
            raise VideoUpscaleJobError("video_upscale_request_snapshot_changed")

    @staticmethod
    def _history_matches_snapshot(
        item: dict,
        snapshot: dict,
        request: VideoUpscaleRequest,
    ) -> None:
        settings = item.get("settings")
        if not isinstance(settings, dict):
            raise VideoUpscaleJobError("video_upscale_request_snapshot_changed")
        expected = {
            "request_snapshot": request.model_dump(mode="json"),
            "source_generation_id": request.source_generation_id,
            "source_artifact_id": request.source_artifact_id,
            "source_media_sha256": snapshot.get("source_media_sha256"),
            "source_bundle_artifact_id": snapshot.get("bundle_artifact_id"),
            "source_bundle_sha256": snapshot.get("bundle_sha256"),
            "source_latent_artifact_id": snapshot.get("latent_artifact_id"),
            "source_latent_sha256": snapshot.get("latent_sha256"),
            "target_width": snapshot.get("target_width"),
            "target_height": snapshot.get("target_height"),
            "resolved_seed": str(snapshot.get("resolved_seed")),
            "output_root": snapshot.get("output_root"),
        }
        if any(settings.get(key) != value for key, value in expected.items()):
            raise VideoUpscaleJobError("video_upscale_request_snapshot_changed")

    @staticmethod
    def _validate_output_paths(paths: tuple[Path, ...], output_root: Path) -> None:
        if not paths:
            raise VideoUpscaleJobError("video_upscale_output_missing")
        try:
            root = output_root.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise VideoUpscaleJobError("video_upscale_output_path_unsafe") from error
        if not root.is_dir():
            raise VideoUpscaleJobError("video_upscale_output_path_unsafe")
        for path in paths:
            try:
                raw = Path(os.path.abspath(os.fspath(path)))
                resolved = raw.resolve(strict=True)
            except (OSError, RuntimeError, ValueError) as error:
                raise VideoUpscaleJobError("video_upscale_output_path_unsafe") from error
            if (
                not resolved.is_file()
                or resolved.is_symlink()
                or not resolved.is_relative_to(root)
                or _contains_reparse_component(raw, root)
            ):
                raise VideoUpscaleJobError("video_upscale_output_path_unsafe")

    def _finalize(
        self,
        *,
        generation_id: int,
        request: VideoUpscaleRequest,
        source: ResolvedVideoUpscaleSource,
        result: VideoUpscaleExecutionResult,
        settings: dict,
    ) -> tuple[tuple[MediaArtifact, ...], tuple[str, ...]]:
        artifacts: list[MediaArtifact] = []
        output_paths: list[str] = []
        created: list[Path] = list(result.output_paths)
        try:
            for path in result.output_paths:
                finalized = self.finalizer(
                    generation_id=generation_id,
                    source_path=path,
                    poster_root=self.poster_root,
                )
                if finalized.media_kind != "video":
                    raise ValueError("video_upscale_artifact_kind_invalid")
                for artifact in finalized.artifacts:
                    update = {}
                    if artifact.kind == "video" and artifact.role == "primary":
                        update = {
                            "parent_artifact_ids": tuple(
                                dict.fromkeys(
                                    (*artifact.parent_artifact_ids, source.source_artifact.artifact_id)
                                )
                            ),
                            "adapter_metadata": {
                                **artifact.adapter_metadata,
                                "upscale": {
                                    "adapter_id": self.adapter.id,
                                    "adapter_revision": self.adapter.revision,
                                    "upstream_commit": VIDEO_UPSCALE_ADAPTER_UPSTREAM_COMMIT,
                                    "source_generation_id": source.request.source_generation_id,
                                    "source_artifact_id": source.source_artifact.artifact_id,
                                    "source_media_sha256": source.bundle.source_media_sha256,
                                    "bundle_artifact_id": source.bundle_artifact_id,
                                    "bundle_sha256": file_sha256(source.bundle_path),
                                    "latent_artifact_id": source.bundle.latent_artifact_id,
                                    "latent_sha256": source.bundle.latent_sha256,
                                    "conditioning_sha256": source.bundle.conditioning_sha256,
                                    "request": request.model_dump(mode="json"),
                                },
                            },
                        }
                    artifacts.append(artifact.model_copy(update=update))
                output_paths.extend(finalized.output_paths)
                created.extend(Path(value) for value in finalized.output_paths)
            primary_ids = tuple(
                artifact.artifact_id
                for artifact in artifacts
                if artifact.kind == "video" and artifact.role == "primary"
            )
            if not primary_ids:
                raise ValueError("video_upscale_primary_artifact_missing")
            sidecar_path = result.output_paths[0].with_suffix(".asteria.json")
            payload = {
                "schema_version": "asteria.video-upscale.artifact.v1",
                "generation_id": generation_id,
                "request_snapshot": request.model_dump(mode="json"),
                "resource_authorization": settings.get("resource_authorization"),
                "source": {
                    "generation_id": source.request.source_generation_id,
                    "artifact_id": source.source_artifact.artifact_id,
                    "media_sha256": source.bundle.source_media_sha256,
                    "bundle_artifact_id": source.bundle_artifact_id,
                    "bundle_sha256": file_sha256(source.bundle_path),
                    "latent_artifact_id": source.bundle.latent_artifact_id,
                    "latent_sha256": source.bundle.latent_sha256,
                    "conditioning_sha256": source.bundle.conditioning_sha256,
                },
                "adapter": {
                    "id": self.adapter.id,
                    "revision": self.adapter.revision,
                    "upstream_commit": VIDEO_UPSCALE_ADAPTER_UPSTREAM_COMMIT,
                    "workflow_id": result.workflow_id,
                    "runtime_job_id": result.runtime_job_id,
                },
                "result": {
                    "width": result.target_width,
                    "height": result.target_height,
                    "artifacts": [
                        artifact.model_dump(mode="json", exclude={"frame_rate"})
                        for artifact in artifacts
                    ],
                },
            }
            write_json_atomic(sidecar_path, payload)
            created.append(sidecar_path)
            sidecar_sha = file_sha256(sidecar_path)
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
            return tuple(artifacts), tuple(dict.fromkeys(output_paths))
        except Exception:
            self._discard(created)
            raise

    def _record_runtime_job(self, generation_id: int, settings: dict, runtime_job_id: str) -> None:
        settings["runtime_job_id"] = runtime_job_id
        self.history.update_settings(generation_id, settings)

    @staticmethod
    def _release(
        registry: GenerationWorkingArtifactRegistry | None,
        operation_id: object,
        artifact_ids: list[str],
    ) -> None:
        if registry is None or not isinstance(operation_id, str):
            return
        try:
            registry.finish_use_many(
                operation_id,
                tuple(artifact_ids),
                outcome="available",
                operation_status="failed",
            )
        except GenerationWorkingArtifactError:
            return

    @staticmethod
    def _consume(
        registry: GenerationWorkingArtifactRegistry | None,
        operation_id: object,
        artifact_ids: list[str],
    ) -> None:
        if registry is None or not isinstance(operation_id, str):
            return
        try:
            registry.finish_use_many(
                operation_id,
                tuple(artifact_ids),
                outcome="consumed",
                operation_status="completed",
            )
            registry.cleanup(operation_status=lambda _operation_id: "completed")
        except GenerationWorkingArtifactError:
            return

    @staticmethod
    def _discard(paths: list[Path]) -> None:
        for path in dict.fromkeys(Path(value) for value in paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _registered_result(generation_id: int, item: dict) -> OperationExecutionResult:
        artifacts = tuple(
            MediaArtifact.model_validate(value).artifact_id
            for value in item.get("artifacts", [])
        )
        return OperationExecutionResult(
            artifact_refs=(f"history:{generation_id}", *artifacts)
        )


def _contains_reparse_component(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    current = path
    while True:
        try:
            details = os.lstat(current)
        except OSError:
            return True
        attributes = getattr(details, "st_file_attributes", 0)
        junction_check = getattr(os.path, "isjunction", None)
        try:
            is_junction = (
                bool(junction_check(current)) if junction_check is not None else False
            )
        except OSError:
            return True
        if (
            stat.S_ISLNK(details.st_mode)
            or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            or is_junction
        ):
            return True
        if current == root:
            return False
        current = current.parent
