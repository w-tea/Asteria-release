from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas.generation import (
    VideoSequenceHistoryArtifact,
    VideoSequenceHistoryDetails,
    VideoSequenceHistorySegment,
)
from app.schemas.media import MediaArtifact
from app.schemas.video_sequence import VideoSequenceSegment
from app.services.media_artifacts import MediaArtifactAccessError, resolve_registered_media_artifact
from app.services.video_sequence_repository import (
    VideoSequenceNotFoundError,
    VideoSequenceRepository,
)


ASSEMBLY_HISTORY_SCHEMA = "asteria.video.sequence.assembly.history.v1"
CLIP_HISTORY_SCHEMA = "asteria.video.history.v1"
SEQUENCE_HISTORY_SCHEMA = "asteria.video.sequence.history.v1"


def is_hidden_sequence_child(item: dict[str, Any]) -> bool:
    settings = item.get("settings")
    if not isinstance(settings, dict):
        return False
    return (
        settings.get("schema_version") == CLIP_HISTORY_SCHEMA
        and isinstance(settings.get("sequence_id"), str)
        and isinstance(settings.get("segment_id"), str)
        and settings.get("library_visibility") != "standalone"
    )


class VideoSequenceHistoryProjector:
    def __init__(self, database_path: Path, *, accepted_roots: list[Path] | tuple[Path, ...] = ()) -> None:
        self._sequences = VideoSequenceRepository(database_path)
        self._accepted_roots = tuple(accepted_roots)

    def project(
        self,
        item: dict[str, Any],
        *,
        history_items: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        settings = item.get("settings")
        if not isinstance(settings, dict) or settings.get("schema_version") != ASSEMBLY_HISTORY_SCHEMA:
            return None
        sequence_id = settings.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            return None
        try:
            sequence = self._sequences.get(sequence_id)
        except VideoSequenceNotFoundError:
            return None

        history_by_id = {
            int(candidate["id"]): candidate
            for candidate in history_items
            if isinstance(candidate.get("id"), int)
        }
        active_ids = set(sequence.active_segment_ids)
        accepted_segments: list[VideoSequenceHistorySegment] = []
        historical_artifacts: list[VideoSequenceHistoryArtifact] = []
        for segment in sequence.segments:
            for artifact_id in segment.candidate_artifact_ids:
                generation_id = segment.candidate_generation_ids.get(artifact_id)
                if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 1:
                    continue
                context_method = self._context_method(segment, artifact_id)
                available = self._artifact_available(
                    history_by_id.get(generation_id),
                    artifact_id=artifact_id,
                )
                common = {
                    "segment_id": segment.segment_id,
                    "ordinal": segment.ordinal,
                    "accepted_revision": (
                        segment.accepted_revision
                        if artifact_id == segment.accepted_candidate_id
                        else None
                    ),
                    "generation_id": generation_id,
                    "artifact_id": artifact_id,
                    "prompt": segment.request_snapshot.prompt,
                    "context_method": context_method,
                    "available": available,
                }
                if (
                    segment.segment_id in active_ids
                    and segment.status == "accepted"
                    and artifact_id == segment.accepted_candidate_id
                ):
                    accepted_segments.append(VideoSequenceHistorySegment.model_validate(common))
                    continue
                historical_artifacts.append(
                    VideoSequenceHistoryArtifact.model_validate(
                        {
                            **common,
                            "state": (
                                "invalidated"
                                if segment.segment_id not in active_ids
                                or segment.status == "needs_regeneration"
                                else "history"
                            ),
                        }
                    )
                )

        assembled_artifact_id = self._assembled_artifact_id(item, sequence_id)
        sidecar_artifact_ids = self._sidecar_artifact_ids(item, sequence_id)
        details = VideoSequenceHistoryDetails(
            schema_version=SEQUENCE_HISTORY_SCHEMA,
            sequence_id=sequence.sequence_id,
            sequence_revision=sequence.revision,
            active_chain_revision=sequence.active_chain_revision,
            assembly_revision=int(settings.get("assembly_revision") or 0),
            status=sequence.status,
            assembled_generation_id=int(item["id"]),
            assembled_artifact_id=assembled_artifact_id,
            sidecar_artifact_ids=sidecar_artifact_ids,
            accepted_segments=accepted_segments,
            historical_artifacts=historical_artifacts,
        )
        return details.model_dump(mode="json")

    def project_items(
        self,
        items: list[dict[str, Any]],
        *,
        history_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        for item in items:
            details = self.project(item, history_items=history_items)
            projected.append({**item, "video_sequence": details})
        return projected

    @staticmethod
    def _context_method(segment: VideoSequenceSegment, artifact_id: str) -> str | None:
        receipt = segment.candidate_context_receipts.get(artifact_id)
        if not isinstance(receipt, dict) and artifact_id == segment.accepted_candidate_id:
            receipt = segment.context_receipt
        method = receipt.get("method") if isinstance(receipt, dict) else None
        return method if method in {"direct_context", "reference_continuation"} else None

    def _artifact_available(
        self,
        item: dict[str, Any] | None,
        *,
        artifact_id: str,
    ) -> bool:
        if item is None or not self._accepted_roots:
            return False
        try:
            resolve_registered_media_artifact(
                item,
                artifact_id,
                accepted_roots=list(self._accepted_roots),
                allowed_kinds={"video"},
            )
        except MediaArtifactAccessError:
            return False
        return True

    @staticmethod
    def _assembled_artifact_id(item: dict[str, Any], sequence_id: str) -> str | None:
        for raw in item.get("artifacts", []):
            try:
                artifact = MediaArtifact.model_validate(raw)
            except ValueError:
                continue
            if artifact.kind == "video" and artifact.role == "primary" and artifact.sequence_id == sequence_id:
                return artifact.artifact_id
        return None

    @staticmethod
    def _sidecar_artifact_ids(item: dict[str, Any], sequence_id: str) -> list[str]:
        result: list[str] = []
        for raw in item.get("artifacts", []):
            try:
                artifact = MediaArtifact.model_validate(raw)
            except ValueError:
                continue
            if artifact.kind == "sidecar" and artifact.role == "sidecar" and artifact.sequence_id == sequence_id:
                result.append(artifact.artifact_id)
        return result
