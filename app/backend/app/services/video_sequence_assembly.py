from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import av

from app.schemas.media import MediaArtifact
from app.schemas.video import VideoRequest
from app.schemas.video_sequence import VideoSequence
from app.services.media_probe import MediaProbeResult, probe_media
from app.services.atomic_json import write_json_atomic
from app.services.runtime import ComfyClient, RuntimeCancellationConfirmed, RuntimeErrorResponse
from app.services.video_model_resolver import VideoModelResolver
from app.services.video_sequence_context import (
    VideoSequenceContextError,
    VideoSequenceContextPreparer,
    VideoSequenceContextReceipt,
    VideoSequenceSourceArtifact,
)


ASSEMBLY_SCHEMA_VERSION = "asteria.video.sequence.assembly.v1"
ASSEMBLY_REVISION = 1


class VideoSequenceAssemblyError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SequenceAssemblyClip:
    ordinal: int
    segment_id: str
    accepted_revision: int
    artifact: MediaArtifact
    source_path: Path
    context_receipt: dict[str, Any] | None
    context_sha256: str | None
    overlap_frames: int
    overlap_seconds: float
    trim_strategy: str
    audio_crossfade_seconds: float
    working_artifact: SequenceAssemblyWorkingArtifact | None = None
    request_snapshot: VideoRequest | None = None


@dataclass(frozen=True, slots=True)
class SequenceAssemblyWorkingArtifact:
    artifact_id: str
    operation_id: str
    generation_id: int
    source_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SequenceAssemblyPlan:
    sequence_id: str
    active_chain_revision: int
    assembly_revision: int
    clips: tuple[SequenceAssemblyClip, ...]
    assembly_mode: str = "encoded_artifact_fallback"
    fallback_reason: str | None = "explicit_encoded_plan"

    def __post_init__(self) -> None:
        if not self.clips:
            raise VideoSequenceAssemblyError("video_sequence_active_chain_empty")
        if tuple(clip.ordinal for clip in self.clips) != tuple(range(len(self.clips))):
            raise VideoSequenceAssemblyError("video_sequence_active_chain_invalid")
        if self.assembly_mode not in {
            "accepted_latent_crossfade",
            "encoded_artifact_fallback",
        }:
            raise VideoSequenceAssemblyError("video_sequence_assembly_mode_invalid")
        if self.assembly_mode == "accepted_latent_crossfade":
            if self.fallback_reason is not None or any(
                clip.working_artifact is None for clip in self.clips
            ):
                raise VideoSequenceAssemblyError("video_sequence_latent_chain_invalid")
        elif not self.fallback_reason:
            raise VideoSequenceAssemblyError("video_sequence_assembly_fallback_reason_missing")


@dataclass(frozen=True, slots=True)
class SequenceAssemblyResult:
    output_path: Path
    probe: MediaProbeResult
    receipt: dict[str, Any]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class LatentAssemblyExecution:
    output_path: Path
    runtime_job_id: str
    manifest_sha256: str


ArtifactResolver = Callable[[str], tuple[MediaArtifact, Path]]
WorkingArtifactResolver = Callable[..., Path]


def build_active_chain_plan(
    sequence: VideoSequence,
    *,
    artifact_resolver: ArtifactResolver,
    working_artifact_resolver: WorkingArtifactResolver | None = None,
    assembly_revision: int = ASSEMBLY_REVISION,
) -> SequenceAssemblyPlan:
    if sequence.status not in {"assembling", "failed"}:
        raise VideoSequenceAssemblyError("video_sequence_not_assembling")
    if not sequence.active_segment_ids:
        raise VideoSequenceAssemblyError("video_sequence_active_chain_empty")
    segment_by_id = {segment.segment_id: segment for segment in sequence.segments}
    active_segments = []
    for ordinal, segment_id in enumerate(sequence.active_segment_ids):
        segment = segment_by_id.get(segment_id)
        if (
            segment is None
            or segment.ordinal != ordinal
            or segment.status != "accepted"
            or segment.accepted_candidate_id is None
            or segment.accepted_revision is None
        ):
            raise VideoSequenceAssemblyError("video_sequence_active_chain_invalid")
        if ordinal == 0:
            if segment.parent_segment_revision is not None:
                raise VideoSequenceAssemblyError("video_sequence_parent_revision_stale")
        else:
            predecessor = active_segments[-1]
            if segment.parent_segment_revision != predecessor.accepted_revision:
                raise VideoSequenceAssemblyError("video_sequence_parent_revision_stale")
        active_segments.append(segment)

    accepted_working_receipts: list[dict[str, Any] | None] = []
    for segment in active_segments:
        raw = segment.candidate_working_artifacts.get(
            segment.accepted_candidate_id or ""
        )
        accepted_working_receipts.append(dict(raw) if isinstance(raw, dict) else None)
    all_working_receipts = all(receipt is not None for receipt in accepted_working_receipts)
    direct_context_chain = all(
        isinstance(segment.context_receipt, dict)
        and segment.context_receipt.get("method") == "direct_context"
        for segment in active_segments[1:]
    )
    latent_mode = bool(
        all_working_receipts
        and direct_context_chain
        and working_artifact_resolver is not None
    )
    if latent_mode:
        assembly_mode = "accepted_latent_crossfade"
        fallback_reason = None
    elif not all_working_receipts:
        assembly_mode = "encoded_artifact_fallback"
        fallback_reason = "accepted_latent_receipt_missing"
    elif not direct_context_chain:
        assembly_mode = "encoded_artifact_fallback"
        fallback_reason = "latent_context_chain_unavailable"
    else:
        assembly_mode = "encoded_artifact_fallback"
        fallback_reason = "latent_assembly_adapter_unavailable"

    clips: list[SequenceAssemblyClip] = []
    for ordinal, segment in enumerate(active_segments):
        artifact, source_path = artifact_resolver(segment.accepted_candidate_id or "")
        if (
            artifact.artifact_id != segment.accepted_candidate_id
            or artifact.kind != "video"
            or artifact.role != "primary"
            or artifact.sequence_id != sequence.sequence_id
            or artifact.segment_id != segment.segment_id
            or artifact.sha256 is None
        ):
            raise VideoSequenceAssemblyError("video_sequence_accepted_artifact_mismatch")
        try:
            resolved_source = Path(source_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise VideoSequenceAssemblyError("video_sequence_accepted_artifact_missing") from error
        if not resolved_source.is_file() or _file_sha256(resolved_source) != artifact.sha256:
            raise VideoSequenceAssemblyError("video_sequence_accepted_artifact_changed")

        working_artifact: SequenceAssemblyWorkingArtifact | None = None
        working_receipt = accepted_working_receipts[ordinal]
        if latent_mode:
            if working_receipt is None or working_artifact_resolver is None:
                raise VideoSequenceAssemblyError("video_sequence_working_artifact_missing")
            try:
                working_path = Path(
                    working_artifact_resolver(
                        operation_id=str(working_receipt["operation_id"]),
                        artifact_id=str(working_receipt["artifact_id"]),
                        expected_kind="av_latent",
                        media_artifact_id=artifact.artifact_id,
                    )
                ).resolve(strict=True)
                working_generation_id = int(working_receipt["generation_id"])
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                raise VideoSequenceAssemblyError(
                    "video_sequence_working_artifact_missing"
                ) from error
            expected_generation_id = segment.candidate_generation_ids.get(
                segment.accepted_candidate_id or ""
            )
            if (
                working_receipt.get("kind") != "av_latent"
                or expected_generation_id != working_generation_id
                or not working_path.is_file()
            ):
                raise VideoSequenceAssemblyError(
                    "video_sequence_working_artifact_mismatch"
                )
            working_artifact = SequenceAssemblyWorkingArtifact(
                artifact_id=str(working_receipt["artifact_id"]),
                operation_id=str(working_receipt["operation_id"]),
                generation_id=working_generation_id,
                source_path=working_path,
                sha256=_file_sha256(working_path),
            )

        context_sha: str | None = None
        context_payload: dict[str, Any] | None = None
        overlap_frames = 0
        overlap_seconds = 0.0
        trim_strategy = "none"
        if ordinal > 0:
            if segment.context_receipt is None:
                raise VideoSequenceAssemblyError("video_sequence_context_receipt_missing")
            try:
                receipt = VideoSequenceContextReceipt.model_validate(segment.context_receipt)
            except ValueError as error:
                raise VideoSequenceAssemblyError("video_sequence_context_receipt_invalid") from error
            predecessor_clip = clips[-1]
            if (
                receipt.source_artifact_id != predecessor_clip.artifact.artifact_id
                or receipt.source_accepted_revision != predecessor_clip.accepted_revision
                or receipt.source_sha256 != predecessor_clip.artifact.sha256
            ):
                raise VideoSequenceAssemblyError("video_sequence_context_receipt_stale")
            predecessor_artifact = predecessor_clip.artifact
            predecessor_working = predecessor_clip.working_artifact
            predecessor_working_receipt = accepted_working_receipts[ordinal - 1]
            frame_rate = predecessor_artifact.frame_rate
            if (
                predecessor_artifact.width is None
                or predecessor_artifact.height is None
                or predecessor_artifact.duration_ms is None
                or frame_rate is None
            ):
                raise VideoSequenceAssemblyError("video_sequence_source_metadata_missing")
            duration_seconds = predecessor_artifact.duration_ms / 1000
            sample_count = (
                max(1, round(duration_seconds * predecessor_artifact.sample_rate))
                if predecessor_artifact.has_audio and predecessor_artifact.sample_rate
                else None
            )
            try:
                VideoSequenceContextPreparer().validate_source(
                    receipt,
                    VideoSequenceSourceArtifact(
                        artifact_id=predecessor_artifact.artifact_id,
                        accepted_revision=predecessor_clip.accepted_revision,
                        sha256=predecessor_artifact.sha256 or "",
                        width=predecessor_artifact.width,
                        height=predecessor_artifact.height,
                        fps=frame_rate,
                        frame_count=max(1, round(duration_seconds * frame_rate)),
                        duration_seconds=duration_seconds,
                        audio_sample_rate=(
                            predecessor_artifact.sample_rate
                            if sample_count is not None
                            else None
                        ),
                        audio_sample_count=sample_count,
                        working_artifact_id=(
                            predecessor_working.artifact_id
                            if predecessor_working is not None
                            else (
                                str(predecessor_working_receipt["artifact_id"])
                                if predecessor_working_receipt is not None
                                else None
                            )
                        ),
                        working_operation_id=(
                            predecessor_working.operation_id
                            if predecessor_working is not None
                            else (
                                str(predecessor_working_receipt["operation_id"])
                                if predecessor_working_receipt is not None
                                else None
                            )
                        ),
                    ),
                )
            except (ValueError, VideoSequenceContextError) as error:
                raise VideoSequenceAssemblyError("video_sequence_context_receipt_stale") from error
            context_sha = receipt.context_sha256
            context_payload = receipt.model_dump(mode="json")
            if latent_mode:
                overlap_frames = receipt.overlap_frames
                overlap_seconds = receipt.overlap_seconds
                trim_strategy = (
                    "latent_av_crossfade_once" if overlap_frames else "none"
                )
            else:
                # Direct-context previews are already trimmed by the generation
                # workflow. Encoded compatibility assembly must not remove the
                # same carried context a second time.
                overlap_frames = 0
                overlap_seconds = 0.0
                trim_strategy = (
                    "encoded_preview_already_trimmed"
                    if receipt.method == "direct_context"
                    else "none"
                )
        clips.append(
            SequenceAssemblyClip(
                ordinal=ordinal,
                segment_id=segment.segment_id,
                accepted_revision=segment.accepted_revision,
                artifact=artifact.model_copy(update={"accepted_revision": segment.accepted_revision}),
                source_path=resolved_source,
                context_receipt=context_payload,
                context_sha256=context_sha,
                overlap_frames=overlap_frames,
                overlap_seconds=overlap_seconds,
                trim_strategy=trim_strategy,
                audio_crossfade_seconds=(overlap_seconds if latent_mode else 0),
                working_artifact=working_artifact,
                request_snapshot=segment.request_snapshot,
            )
        )
    return SequenceAssemblyPlan(
        sequence_id=sequence.sequence_id,
        active_chain_revision=sequence.active_chain_revision,
        assembly_revision=assembly_revision,
        clips=tuple(clips),
        assembly_mode=assembly_mode,
        fallback_reason=fallback_reason,
    )


class ComfyH3LatentAssemblyAdapter:
    def __init__(
        self,
        *,
        client: ComfyClient,
        workflow_path: Path,
        model_resolver: VideoModelResolver | None = None,
    ) -> None:
        self.client = client
        self.workflow_path = Path(workflow_path)
        self.model_resolver = model_resolver

    def assemble(
        self,
        plan: SequenceAssemblyPlan,
        *,
        destination: Path,
        output_root: Path,
        operation_id: str,
        cancel_requested: Callable[[], bool],
    ) -> LatentAssemblyExecution:
        if plan.assembly_mode != "accepted_latent_crossfade":
            raise VideoSequenceAssemblyError("video_sequence_latent_chain_invalid")
        manifest_path: Path | None = None
        prompt_id: str | None = None
        destination = Path(destination)
        try:
            manifest, manifest_sha = self._manifest(plan, operation_id=operation_id)
            manifest_path = self._write_manifest(
                manifest,
                output_root=Path(output_root),
                operation_id=operation_id,
            )
            workflow = self._workflow(
                plan,
                manifest_path=manifest_path,
                operation_id=operation_id,
            )
            self._validate_runtime_contract()
            if cancel_requested():
                raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
            prompt_id = self.client.queue_prompt(workflow)
            history = self.client.wait_for_history(
                prompt_id,
                timeout_seconds=None,
                runtime_stop_confirmed=cancel_requested,
            )
            if cancel_requested():
                self._cancel(prompt_id)
                raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
            outputs = self._video_outputs(history, prompt_id)
            if len(outputs) != 1:
                raise VideoSequenceAssemblyError("video_sequence_assembly_output_invalid")
            payload = self.client.download_output(outputs[0])
            if not payload:
                raise VideoSequenceAssemblyError("video_sequence_assembly_output_invalid")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return LatentAssemblyExecution(
                output_path=destination,
                runtime_job_id=prompt_id,
                manifest_sha256=manifest_sha,
            )
        except VideoSequenceAssemblyError:
            destination.unlink(missing_ok=True)
            raise
        except RuntimeCancellationConfirmed as error:
            if prompt_id is not None:
                self._cancel(prompt_id)
            destination.unlink(missing_ok=True)
            raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled") from error
        except (OSError, RuntimeErrorResponse, TypeError, ValueError, KeyError) as error:
            destination.unlink(missing_ok=True)
            raise VideoSequenceAssemblyError("video_sequence_latent_assembly_failed") from error
        finally:
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)
                try:
                    manifest_path.parent.rmdir()
                except OSError:
                    pass

    def _manifest(
        self,
        plan: SequenceAssemblyPlan,
        *,
        operation_id: str,
    ) -> tuple[dict[str, Any], str]:
        allowed_roots: list[str] = []
        clips: list[dict[str, Any]] = []
        fps: float | None = None
        for clip in plan.clips:
            working = clip.working_artifact
            request = clip.request_snapshot
            if working is None or request is None:
                raise VideoSequenceAssemblyError("video_sequence_latent_chain_invalid")
            if fps is None:
                fps = float(request.fps)
            elif float(request.fps) != fps:
                raise VideoSequenceAssemblyError("video_sequence_assembly_fps_mismatch")
            actual_hash = _file_sha256(working.source_path)
            if actual_hash != working.sha256:
                raise VideoSequenceAssemblyError("video_sequence_working_artifact_changed")
            root = _working_artifact_output_root(working.source_path)
            root_value = str(root)
            if root_value not in allowed_roots:
                allowed_roots.append(root_value)
            clips.append(
                {
                    "ordinal": clip.ordinal,
                    "segment_id": clip.segment_id,
                    "accepted_revision": clip.accepted_revision,
                    "media_artifact_id": clip.artifact.artifact_id,
                    "media_sha256": clip.artifact.sha256,
                    "working_artifact_id": working.artifact_id,
                    "working_operation_id": working.operation_id,
                    "working_sha256": working.sha256,
                    "path": str(working.source_path),
                    "overlap_frames": clip.overlap_frames,
                }
            )
        manifest = {
            "schema_version": "asteria.video.sequence.latent-assembly-input.v1",
            "operation_id": operation_id,
            "sequence_id": plan.sequence_id,
            "active_chain_revision": plan.active_chain_revision,
            "assembly_revision": plan.assembly_revision,
            "allowed_roots": allowed_roots,
            "fps": fps,
            "clips": clips,
        }
        return manifest, sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()

    @staticmethod
    def _write_manifest(
        manifest: dict[str, Any],
        *,
        output_root: Path,
        operation_id: str,
    ) -> Path:
        operation_token = sha256(operation_id.encode("utf-8")).hexdigest()[:32]
        directory = (
            Path(output_root).expanduser().resolve()
            / ".asteria"
            / "sequence-assembly"
            / f"op-{operation_token}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "accepted-latents.json"
        write_json_atomic(path, manifest)
        return path

    def _workflow(
        self,
        plan: SequenceAssemblyPlan,
        *,
        manifest_path: Path,
        operation_id: str,
    ) -> dict[str, dict]:
        try:
            payload = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid") from error
        workflow = deepcopy(payload.get("workflow"))
        bindings = payload.get("bindings")
        if (
            payload.get("schema_version")
            != "asteria.video.workflow.v1"
            or payload.get("workflow_id")
            != "asteria_minimax_h3_sequence_latent_assembly.v1"
            or not isinstance(workflow, dict)
            or not isinstance(bindings, dict)
        ):
            raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid")
        first_request = plan.clips[0].request_snapshot
        if first_request is None:
            raise VideoSequenceAssemblyError("video_sequence_latent_chain_invalid")
        video_vae = "minimax_h3_video_vae_fp16.safetensors"
        audio_vae = "minimax_h3_audio_vae_fp32.safetensors"
        if first_request.schema_version == "asteria.video.request.v2":
            if self.model_resolver is None:
                raise VideoSequenceAssemblyError("video_model_preflight_required")
            try:
                resolved = self.model_resolver.resolve(first_request)
                video_vae = resolved.components["video_vae"].loader_name
                audio_vae = resolved.components["audio_vae"].loader_name
            except (KeyError, ValueError) as error:
                raise VideoSequenceAssemblyError("video_model_preflight_failed") from error
        values: dict[str, Any] = {
            "manifest_path": str(manifest_path),
            "stitch_fps": float(first_request.fps),
            "video_fps": float(first_request.fps),
            "video_vae_name": video_vae,
            "audio_vae_name": audio_vae,
            "output_prefix": (
                "asteria/video-sequence-assembly/"
                + re.sub(r"[^A-Za-z0-9._-]+", "_", operation_id).strip("._-")
            ),
        }
        for name, targets in bindings.items():
            if name not in values:
                raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid")
            for target in (targets if isinstance(targets, list) else [targets]):
                self._assign(workflow, str(target), values[name])
        if _contains_placeholder(workflow):
            raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid")
        return workflow

    @staticmethod
    def _assign(workflow: dict, target: str, value: Any) -> None:
        parts = target.split(".")
        if len(parts) != 3 or parts[1] != "inputs":
            raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid")
        node = workflow.get(parts[0])
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise VideoSequenceAssemblyError("video_sequence_assembly_workflow_invalid")
        node["inputs"][parts[2]] = value

    @staticmethod
    def _video_outputs(history: dict, prompt_id: str) -> tuple[dict, ...]:
        prompt = history.get(prompt_id)
        outputs = prompt.get("outputs") if isinstance(prompt, dict) else None
        if not isinstance(outputs, dict):
            return ()
        found: list[dict] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                filename = value.get("filename")
                if (
                    isinstance(filename, str)
                    and Path(filename).suffix.lower() in {".mp4", ".webm"}
                ):
                    found.append(value)
                    return
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(outputs)
        return tuple(found)

    def _cancel(self, prompt_id: str) -> None:
        try:
            state = self.client.queue_state()
            if prompt_id in state.get("pending", []):
                self.client.delete_queued_prompt(prompt_id)
            elif prompt_id in state.get("running", []):
                self.client.interrupt()
        except (OSError, RuntimeErrorResponse, TypeError, ValueError):
            pass

    def _validate_runtime_contract(self) -> None:
        info = self.client.object_info()
        required = {
            "VAELoader": {"vae_name"},
            "AsteriaH3AcceptedLatentStitcher": {
                "manifest_path",
                "video_vae",
                "audio_vae",
                "fps",
                "contract_version",
            },
            "CreateVideo": {"images", "audio", "fps", "bit_depth"},
            "SaveVideo": {"video", "filename_prefix", "format", "codec"},
        }
        for class_type, expected_inputs in required.items():
            node = info.get(class_type) if isinstance(info, dict) else None
            inputs = node.get("input") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                raise VideoSequenceAssemblyError(
                    "video_sequence_latent_assembly_runtime_unavailable"
                )
            available = set()
            for group in ("required", "optional"):
                values = inputs.get(group)
                if isinstance(values, dict):
                    available.update(values)
            if not expected_inputs.issubset(available):
                raise VideoSequenceAssemblyError(
                    "video_sequence_latent_assembly_contract_changed"
                )
        contract = info["AsteriaH3AcceptedLatentStitcher"]["input"]["required"].get(
            "contract_version"
        )
        options = contract[0] if isinstance(contract, (list, tuple)) and contract else None
        if not isinstance(options, (list, tuple)) or (
            "asteria.video.sequence.latent-assembly.v1" not in options
        ):
            raise VideoSequenceAssemblyError(
                "video_sequence_latent_assembly_contract_changed"
            )


class VideoSequenceAssemblyService:
    def __init__(
        self,
        *,
        output_root: Path | Callable[[], Path],
        latent_adapter: ComfyH3LatentAssemblyAdapter | None = None,
    ) -> None:
        self._output_root_provider = (
            output_root if callable(output_root) else lambda: Path(output_root)
        )
        self.latent_adapter = latent_adapter

    @property
    def output_root(self) -> Path:
        return Path(self._output_root_provider())

    def assemble(
        self,
        plan: SequenceAssemblyPlan,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        output_root: Path | None = None,
        operation_id: str | None = None,
    ) -> SequenceAssemblyResult:
        cancel_requested = cancel_requested or (lambda: False)
        probes = self._preflight(plan)
        operation_output_root = Path(output_root) if output_root is not None else self.output_root
        output_dir = operation_output_root / "video-sequences" / plan.sequence_id
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / (
            f"chain-{plan.active_chain_revision}.assembly-{plan.assembly_revision}.mp4"
        )
        staging_path = output_dir / f".{final_path.name}.{uuid4().hex}.staging.mp4"
        latent_execution: LatentAssemblyExecution | None = None
        try:
            if plan.assembly_mode == "accepted_latent_crossfade":
                if self.latent_adapter is None or not operation_id:
                    raise VideoSequenceAssemblyError(
                        "video_sequence_latent_assembly_not_configured"
                    )
                latent_execution = self.latent_adapter.assemble(
                    plan,
                    destination=staging_path,
                    output_root=operation_output_root,
                    operation_id=operation_id,
                    cancel_requested=cancel_requested,
                )
            else:
                self._encode(plan, probes, staging_path, cancel_requested=cancel_requested)
            if cancel_requested():
                raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
            result_probe = probe_media(staging_path)
            expected = probes[0]
            if (
                result_probe.width != expected.width
                or result_probe.height != expected.height
                or result_probe.fps_numerator != expected.fps_numerator
                or result_probe.fps_denominator != expected.fps_denominator
                or result_probe.has_audio != expected.has_audio
            ):
                raise VideoSequenceAssemblyError("video_sequence_assembly_probe_mismatch")
            receipt = self._receipt(
                plan,
                result_probe,
                latent_execution=latent_execution,
            )
            canonical = _canonical_json(receipt)
            os.replace(staging_path, final_path)
            return SequenceAssemblyResult(
                output_path=final_path,
                probe=result_probe,
                receipt=receipt,
                receipt_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
            )
        except VideoSequenceAssemblyError:
            staging_path.unlink(missing_ok=True)
            raise
        except (av.FFmpegError, EOFError, OSError, RuntimeError, ValueError) as error:
            staging_path.unlink(missing_ok=True)
            raise VideoSequenceAssemblyError("video_sequence_assembly_failed") from error

    def _preflight(self, plan: SequenceAssemblyPlan) -> tuple[MediaProbeResult, ...]:
        probes: list[MediaProbeResult] = []
        for clip in plan.clips:
            if clip.artifact.sha256 is None or _file_sha256(clip.source_path) != clip.artifact.sha256:
                raise VideoSequenceAssemblyError("video_sequence_accepted_artifact_changed")
            probe = probe_media(clip.source_path)
            if probe.kind != "video" or probe.fps_numerator is None or probe.fps_denominator is None:
                raise VideoSequenceAssemblyError("video_sequence_assembly_input_invalid")
            if probes:
                first = probes[0]
                if (probe.width, probe.height) != (first.width, first.height):
                    raise VideoSequenceAssemblyError("video_sequence_assembly_resolution_mismatch")
                if (probe.fps_numerator, probe.fps_denominator) != (
                    first.fps_numerator,
                    first.fps_denominator,
                ):
                    raise VideoSequenceAssemblyError("video_sequence_assembly_fps_mismatch")
                if (
                    probe.video_time_base_numerator,
                    probe.video_time_base_denominator,
                ) != (
                    first.video_time_base_numerator,
                    first.video_time_base_denominator,
                ):
                    raise VideoSequenceAssemblyError(
                        "video_sequence_assembly_video_time_base_mismatch"
                    )
                if probe.has_audio != first.has_audio:
                    raise VideoSequenceAssemblyError("video_sequence_assembly_audio_mismatch")
                if probe.has_audio and (
                    probe.sample_rate != first.sample_rate or probe.channels != first.channels
                ):
                    raise VideoSequenceAssemblyError("video_sequence_assembly_audio_mismatch")
                if probe.has_audio and (
                    probe.audio_time_base_numerator,
                    probe.audio_time_base_denominator,
                ) != (
                    first.audio_time_base_numerator,
                    first.audio_time_base_denominator,
                ):
                    raise VideoSequenceAssemblyError(
                        "video_sequence_assembly_audio_time_base_mismatch"
                    )
            estimated_frames = round(
                probe.duration_ms * probe.fps_numerator / (1000 * probe.fps_denominator)
            )
            if clip.overlap_frames > estimated_frames:
                raise VideoSequenceAssemblyError("video_sequence_assembly_overlap_invalid")
            probes.append(probe)
        return tuple(probes)

    @staticmethod
    def _encode(
        plan: SequenceAssemblyPlan,
        probes: tuple[MediaProbeResult, ...],
        output_path: Path,
        *,
        cancel_requested: Callable[[], bool],
    ) -> None:
        first = probes[0]
        rate = Fraction(first.fps_numerator or 1, first.fps_denominator or 1)
        with av.open(str(output_path), mode="w", format="mp4") as output:
            video_out = output.add_stream("libx264", rate=rate)
            video_out.width = int(first.width or 0)
            video_out.height = int(first.height or 0)
            video_out.pix_fmt = "yuv420p"
            video_out.options = {"crf": "18", "preset": "medium"}
            audio_out = None
            audio_layout = None
            if first.has_audio:
                audio_layout = "mono" if first.channels == 1 else "stereo"
                audio_out = output.add_stream(
                    "aac", rate=int(first.sample_rate or 0), layout=audio_layout
                )

            video_pts = 0
            audio_pts = 0
            video_time_base = Fraction(rate.denominator, rate.numerator)
            for clip in plan.clips:
                if cancel_requested():
                    raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
                with av.open(str(clip.source_path), mode="r") as source:
                    video_stream = next(iter(source.streams.video), None)
                    if video_stream is None or video_stream.time_base is None or video_stream.time_base <= 0:
                        raise VideoSequenceAssemblyError("video_sequence_assembly_time_base_invalid")
                    skipped = 0
                    clip_frame_count = 0
                    for frame in source.decode(video=0):
                        if skipped < clip.overlap_frames:
                            skipped += 1
                            continue
                        if cancel_requested():
                            raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
                        normalized = frame.reformat(
                            width=video_out.width,
                            height=video_out.height,
                            format="yuv420p",
                        )
                        normalized.pts = video_pts
                        normalized.time_base = video_time_base
                        video_pts += 1
                        clip_frame_count += 1
                        for packet in video_out.encode(normalized):
                            output.mux(packet)
                if audio_out is not None and audio_layout is not None:
                    max_audio_samples = round(
                        clip_frame_count * int(first.sample_rate or 0) / float(rate)
                    )
                    audio_pts = VideoSequenceAssemblyService._encode_audio_clip(
                        output,
                        audio_out,
                        clip,
                        sample_rate=int(first.sample_rate or 0),
                        layout=audio_layout,
                        output_pts=audio_pts,
                        max_output_samples=max_audio_samples,
                        cancel_requested=cancel_requested,
                    )
            for packet in video_out.encode():
                output.mux(packet)
            if audio_out is not None:
                for packet in audio_out.encode():
                    output.mux(packet)

    @staticmethod
    def _encode_audio_clip(
        output,
        audio_out,
        clip: SequenceAssemblyClip,
        *,
        sample_rate: int,
        layout: str,
        output_pts: int,
        max_output_samples: int,
        cancel_requested: Callable[[], bool],
    ) -> int:
        skip_samples = max(0, round(clip.overlap_seconds * sample_rate))
        written_samples = 0
        with av.open(str(clip.source_path), mode="r") as source:
            audio_stream = next(iter(source.streams.audio), None)
            if audio_stream is None or audio_stream.time_base is None or audio_stream.time_base <= 0:
                raise VideoSequenceAssemblyError("video_sequence_assembly_time_base_invalid")
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            for decoded in source.decode(audio=0):
                for frame in resampler.resample(decoded):
                    if cancel_requested():
                        raise VideoSequenceAssemblyError("video_sequence_assembly_cancelled")
                    if skip_samples >= frame.samples:
                        skip_samples -= frame.samples
                        continue
                    if skip_samples:
                        values = frame.to_ndarray()[:, skip_samples:]
                        frame = av.AudioFrame.from_ndarray(values, format="fltp", layout=layout)
                        frame.sample_rate = sample_rate
                        skip_samples = 0
                    remaining = max_output_samples - written_samples
                    if remaining <= 0:
                        return output_pts
                    if frame.samples > remaining:
                        values = frame.to_ndarray()[:, :remaining]
                        frame = av.AudioFrame.from_ndarray(values, format="fltp", layout=layout)
                        frame.sample_rate = sample_rate
                    frame.pts = output_pts
                    frame.time_base = Fraction(1, sample_rate)
                    output_pts += frame.samples
                    written_samples += frame.samples
                    for packet in audio_out.encode(frame):
                        output.mux(packet)
                    if written_samples >= max_output_samples:
                        return output_pts
        return output_pts

    @staticmethod
    def _receipt(
        plan: SequenceAssemblyPlan,
        probe: MediaProbeResult,
        *,
        latent_execution: LatentAssemblyExecution | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": ASSEMBLY_SCHEMA_VERSION,
            "sequence_id": plan.sequence_id,
            "active_chain_revision": plan.active_chain_revision,
            "assembly_revision": plan.assembly_revision,
            "assembly_mode": plan.assembly_mode,
            "fallback_reason": plan.fallback_reason,
            "segment_ids": [clip.segment_id for clip in plan.clips],
            "accepted_revisions": [clip.accepted_revision for clip in plan.clips],
            "source_artifact_ids": [clip.artifact.artifact_id for clip in plan.clips],
            "source_sha256": [clip.artifact.sha256 for clip in plan.clips],
            "context_sha256": [clip.context_sha256 for clip in plan.clips],
            "context_receipts": [clip.context_receipt for clip in plan.clips],
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
            "working_sha256": [
                clip.working_artifact.sha256
                if clip.working_artifact is not None
                else None
                for clip in plan.clips
            ],
            "runtime_job_id": (
                latent_execution.runtime_job_id
                if latent_execution is not None
                else None
            ),
            "input_manifest_sha256": (
                latent_execution.manifest_sha256
                if latent_execution is not None
                else None
            ),
            "decisions": [
                {
                    "segment_id": clip.segment_id,
                    "trim_strategy": clip.trim_strategy,
                    "video_trim_frames": clip.overlap_frames,
                    "audio_trim_seconds": clip.overlap_seconds,
                    "audio_crossfade_seconds": float(clip.audio_crossfade_seconds),
                }
                for clip in plan.clips
            ],
            "final_probe": {
                "mime_type": probe.mime_type,
                "container_format": probe.container_format,
                "width": probe.width,
                "height": probe.height,
                "duration_ms": probe.duration_ms,
                "fps_numerator": probe.fps_numerator,
                "fps_denominator": probe.fps_denominator,
                "codec": probe.codec,
                "has_audio": probe.has_audio,
                "sample_rate": probe.sample_rate,
                "channels": probe.channels,
                "video_time_base_numerator": probe.video_time_base_numerator,
                "video_time_base_denominator": probe.video_time_base_denominator,
                "audio_time_base_numerator": probe.audio_time_base_numerator,
                "audio_time_base_denominator": probe.audio_time_base_denominator,
            },
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _working_artifact_output_root(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    for parent in resolved.parents:
        if parent.name == "working-artifacts" and parent.parent.name == ".asteria":
            return parent.parent.parent.resolve(strict=True)
    raise VideoSequenceAssemblyError("video_sequence_working_artifact_path_invalid")


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("${") and value.endswith("}")
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False
