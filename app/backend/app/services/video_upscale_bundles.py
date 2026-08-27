from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from app.schemas.generation_working_artifact import GenerationWorkingArtifactRecord
from app.schemas.media import MediaArtifact
from app.schemas.video import VideoRequest
from app.schemas.video_upscale import VideoUpscaleWorkBundle
from app.services.atomic_json import write_json_atomic
from app.services.video_local_adapter import LocalVideoExecutionResult
from app.services.video_packet import VideoSubmissionPacket


class VideoUpscaleBundleError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def stage_video_upscale_bundle(
    *,
    output_root: Path,
    operation_id: str,
    generation_id: int,
    packet: VideoSubmissionPacket,
    execution: LocalVideoExecutionResult,
    primary: MediaArtifact,
    latent: GenerationWorkingArtifactRecord,
) -> tuple[VideoUpscaleWorkBundle, Path]:
    if (
        latent.kind != "av_latent"
        or latent.generation_id != generation_id
        or latent.media_artifact_id != primary.artifact_id
        or latent.operation_id != operation_id
        or primary.kind != "video"
        or primary.sha256 is None
        or primary.width is None
        or primary.height is None
        or primary.fps_numerator is None
        or primary.fps_denominator is None
    ):
        raise VideoUpscaleBundleError("video_upscale_bundle_source_mismatch")
    metadata = primary.adapter_metadata
    video_time_base = metadata.get("video_time_base")
    audio_time_base = metadata.get("audio_time_base")
    if not isinstance(video_time_base, str):
        raise VideoUpscaleBundleError("video_upscale_bundle_probe_missing")
    if primary.has_audio is True:
        if not isinstance(primary.sample_rate, int) or not isinstance(audio_time_base, str):
            raise VideoUpscaleBundleError("video_upscale_bundle_probe_missing")
    elif audio_time_base is not None:
        raise VideoUpscaleBundleError("video_upscale_bundle_probe_invalid")

    request = VideoRequest.model_validate(packet.request_snapshot)
    packet_payload = packet.to_dict()
    conditioning_receipt = {
        "request": packet_payload["request_snapshot"],
        "conditioning": packet_payload["provider_payload"].get("conditioning"),
        "model_selection": packet_payload["provider_payload"].get("model_selection"),
        "loras": packet_payload["provider_payload"].get("loras"),
        "mapping_id": packet.mapping_id,
        "mapping_revision": packet.mapping_version,
    }
    bundle = VideoUpscaleWorkBundle(
        source_generation_id=generation_id,
        source_media_artifact_id=primary.artifact_id,
        source_media_sha256=primary.sha256,
        actual_output_root=str(Path(output_root).expanduser().resolve()),
        working_operation_id=operation_id,
        latent_artifact_id=latent.artifact_id,
        latent_sha256=latent.sha256,
        conditioning_sha256=canonical_sha256(conditioning_receipt),
        packet_sha256=canonical_sha256(packet_payload),
        packet_snapshot=packet_payload,
        request_snapshot=request,
        workflow_id=execution.workflow_id,
        mapping_id=packet.mapping_id,
        mapping_revision=packet.mapping_version,
        actual_width=primary.width,
        actual_height=primary.height,
        actual_frame_count=execution.actual_frame_count,
        fps_numerator=primary.fps_numerator,
        fps_denominator=primary.fps_denominator,
        audio_sample_rate=primary.sample_rate if primary.has_audio is True else None,
        video_time_base=video_time_base,
        audio_time_base=audio_time_base if primary.has_audio is True else None,
        created_at=datetime.now(timezone.utc),
    )
    staging = Path(output_root).resolve() / ".asteria" / "upscale-bundle-staging"
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / f"{uuid4().hex}.json"
    write_json_atomic(path, bundle.model_dump(mode="json"))
    return bundle, path
