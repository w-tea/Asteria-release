from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import AppConfig
from app.schemas.generation_working_artifact import (
    GenerationWorkingArtifactSummary,
    WorkingArtifactKind,
)
from app.schemas.video import VideoReference, VideoRequest
from app.services.history import HistoryRepository
from app.services.generation_working_artifacts import (
    GenerationWorkingArtifactError,
    GenerationWorkingArtifactRegistry,
)
from app.services.media_artifacts import MediaArtifactAccessError, resolve_registered_media_artifact
from app.services.runtime import ComfyClient, RuntimeCancellationConfirmed, RuntimeErrorResponse
from app.services.user_settings import UserSettingsRepository, resolve_output_root
from app.services.video_packet import VideoSubmissionPacket
from app.services.video_model_resolver import (
    ResolvedVideoModelSelection,
    VideoModelResolver,
)
from app.services.video_frame_policy import normalized_h3_frame_count
from app.services.video_sequence_context import VideoSequenceContextReceipt


VIDEO_WORKFLOW_SCHEMA_VERSION = "asteria.video.workflow.v1"
_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_V2_PRODUCTION_BINDINGS = frozenset(
    {
        "diffusion_model_name",
        "text_encoder_name",
        "video_vae_name",
        "audio_vae_name",
        "video_shift",
        "audio_shift",
        "steps",
        "scheduler",
        "sampler",
    }
)


class VideoLocalAdapterError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class LocalVideoWorkingOutput:
    kind: WorkingArtifactKind
    path: Path
    source_sha256: tuple[str, ...] = ()
    conditioning_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LocalVideoExecutionResult:
    runtime_job_id: str
    workflow_id: str
    output_paths: tuple[Path, ...]
    requested_frame_count: int
    actual_frame_count: int
    actual_duration_seconds: float
    working_outputs: tuple[LocalVideoWorkingOutput, ...] = ()


class VideoSourceResolver(Protocol):
    def resolve(self, artifact_id: str, expected_kind: str) -> Path: ...


class VideoWorkingArtifactResolver(Protocol):
    def resolve(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        expected_kind: WorkingArtifactKind,
        media_artifact_id: str,
    ) -> Path: ...


class VideoWorkflowWorkingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: WorkingArtifactKind
    node_id: str
    path_input: str
    index_input: str
    suffix: str

    @model_validator(mode="after")
    def validate_contract(self) -> "VideoWorkflowWorkingOutput":
        if not self.node_id or not self.path_input or not self.index_input:
            raise ValueError("video workflow working output binding is incomplete")
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,12}", self.suffix):
            raise ValueError("video workflow working output suffix is invalid")
        return self


class VideoWorkflowAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    workflow_id: str
    mode: str
    minimum_comfyui_version: str
    required_nodes: tuple[str, ...]
    required_node_inputs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    workflow: dict[str, dict]
    bindings: dict[str, str]
    dynamic_bindings: dict[str, dict] = Field(default_factory=dict)
    working_outputs: tuple[VideoWorkflowWorkingOutput, ...] = ()

    @model_validator(mode="after")
    def validate_contract(self) -> "VideoWorkflowAsset":
        if self.schema_version != VIDEO_WORKFLOW_SCHEMA_VERSION:
            raise ValueError("video workflow schema is unsupported")
        if not _WORKFLOW_ID.fullmatch(self.workflow_id):
            raise ValueError("video workflow identifier is invalid")
        if not self.workflow or not self.bindings:
            raise ValueError("video workflow is incomplete")
        class_types = {
            str(node.get("class_type") or "")
            for node in self.workflow.values()
            if isinstance(node, dict)
        }
        class_types.update(_declared_dynamic_class_types(self.dynamic_bindings))
        if not set(self.required_nodes).issubset(class_types):
            raise ValueError("video workflow required nodes are missing")
        for output in self.working_outputs:
            node = self.workflow.get(output.node_id)
            if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                raise ValueError("video workflow working output node is missing")
            if output.path_input not in node["inputs"] or output.index_input not in node["inputs"]:
                raise ValueError("video workflow working output inputs are missing")
        if not set(self.required_node_inputs).issubset(class_types):
            raise ValueError("video workflow required input node is missing")
        if any(
            not inputs or len(inputs) != len(set(inputs)) or any(not value for value in inputs)
            for inputs in self.required_node_inputs.values()
        ):
            raise ValueError("video workflow required input contract is invalid")
        return self


class VideoWorkflowRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def load(self, workflow_id: str) -> VideoWorkflowAsset:
        if not _WORKFLOW_ID.fullmatch(workflow_id):
            raise VideoLocalAdapterError("video_workflow_not_found")
        matches: list[VideoWorkflowAsset] = []
        try:
            paths = sorted(self.root.glob("*.json"))
        except OSError as error:
            raise VideoLocalAdapterError("video_workflow_unavailable") from error
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                candidate = VideoWorkflowAsset.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise VideoLocalAdapterError("video_workflow_invalid") from error
            if candidate.workflow_id == workflow_id:
                matches.append(candidate)
        if len(matches) != 1:
            code = "video_workflow_not_found" if not matches else "video_workflow_invalid"
            raise VideoLocalAdapterError(code)
        return matches[0]


class HistoryVideoSourceResolver:
    def __init__(self, history: HistoryRepository, config: AppConfig) -> None:
        self.history = history
        self.config = config

    def resolve(self, artifact_id: str, expected_kind: str) -> Path:
        matches: list[tuple[dict, set[Path]]] = []
        for item in self.history.list_all():
            if not any(
                isinstance(artifact, dict) and artifact.get("artifact_id") == artifact_id
                for artifact in item.get("artifacts", [])
            ):
                continue
            matches.append((item, self._accepted_roots(item)))
        if len(matches) != 1:
            raise VideoLocalAdapterError("video_source_artifact_not_found")
        item, roots = matches[0]
        try:
            _artifact, path = resolve_registered_media_artifact(
                item,
                artifact_id,
                accepted_roots=roots,
                allowed_kinds={expected_kind},
            )
        except MediaArtifactAccessError as error:
            raise VideoLocalAdapterError("video_source_artifact_not_found") from error
        return path

    def _accepted_roots(self, item: dict) -> set[Path]:
        roots = {self.config.data_dir / "outputs"}
        settings = UserSettingsRepository(self.config.data_dir / "user-settings.json").load().output
        roots.add(resolve_output_root(self.config.data_dir, settings))
        item_settings = item.get("settings")
        if isinstance(item_settings, dict):
            for key in ("output_root", "output_directory"):
                value = item_settings.get(key)
                if isinstance(value, str) and value.strip():
                    roots.add(Path(value))
        return roots


class HistoryVideoWorkingArtifactResolver:
    def __init__(self, history: HistoryRepository) -> None:
        self.history = history

    def resolve(
        self,
        *,
        operation_id: str,
        artifact_id: str,
        expected_kind: WorkingArtifactKind,
        media_artifact_id: str,
    ) -> Path:
        matches: list[tuple[GenerationWorkingArtifactSummary, Path]] = []
        for item in self.history.list_all():
            settings = item.get("settings")
            if not isinstance(settings, dict):
                continue
            root_value = settings.get("output_root")
            if not isinstance(root_value, str) or not root_value.strip():
                continue
            for raw in settings.get("working_artifacts", []):
                try:
                    summary = GenerationWorkingArtifactSummary.model_validate(raw)
                except ValueError:
                    continue
                if summary.artifact_id == artifact_id:
                    matches.append((summary, Path(root_value).expanduser().resolve()))
        if len(matches) != 1:
            raise VideoLocalAdapterError("video_sequence_working_artifact_not_found")
        summary, output_root = matches[0]
        if (
            summary.kind != expected_kind
            or summary.media_artifact_id != media_artifact_id
            or summary.state not in {"available", "in_use"}
        ):
            raise VideoLocalAdapterError("video_sequence_working_artifact_mismatch")
        try:
            return GenerationWorkingArtifactRegistry(output_root).resolve_for_adapter(
                operation_id=operation_id,
                artifact_id=artifact_id,
                expected_kind=expected_kind,
                allow_in_use=True,
            )
        except GenerationWorkingArtifactError as error:
            raise VideoLocalAdapterError(str(error)) from error


class LocalH3VideoExecutionAdapter:
    def __init__(
        self,
        *,
        client: ComfyClient,
        workflow_root: Path,
        source_resolver: VideoSourceResolver,
        output_root: Path | Callable[[], Path],
        working_artifact_resolver: VideoWorkingArtifactResolver | None = None,
        comfy_output_root: Path | None = None,
        model_resolver: VideoModelResolver | None = None,
    ) -> None:
        self.client = client
        self.workflows = VideoWorkflowRepository(workflow_root)
        self.source_resolver = source_resolver
        self.working_artifact_resolver = working_artifact_resolver
        self.comfy_output_root = (
            Path(comfy_output_root).expanduser().resolve()
            if comfy_output_root is not None
            else None
        )
        self.model_resolver = model_resolver
        self._output_root_provider = (
            output_root if callable(output_root) else lambda: Path(output_root)
        )

    @property
    def output_root(self) -> Path:
        return Path(self._output_root_provider())

    def execute_video(
        self,
        packet: VideoSubmissionPacket,
        *,
        operation_id: str,
        output_root: Path | None = None,
        progress_callback: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
        runtime_job_callback: Callable[[str], None],
    ) -> LocalVideoExecutionResult:
        operation_output_root = Path(output_root) if output_root is not None else self.output_root
        payload = packet.provider_payload
        workflow_id = str(payload.get("workflow_template_id") or "")
        asset = self.workflows.load(workflow_id)
        request = packet.request_snapshot
        if asset.mode != request.get("mode"):
            raise VideoLocalAdapterError("video_workflow_mismatch")
        requested_frames, actual_frames = normalized_h3_frame_count(
            float(request["duration_seconds"]),
            float(request["fps"]),
        )
        prompt_id: str | None = None
        written: list[Path] = []
        staged_working: list[LocalVideoWorkingOutput] = []
        try:
            if cancel_requested():
                raise VideoLocalAdapterError("video_generation_cancelled", retryable=True)
            workflow = self._build_workflow(asset, packet, actual_frames, operation_id)
            staged_working = self._expected_working_outputs(asset, workflow, packet)
            preview_destination = self._prepare_output_subdirectory(
                "_previews",
                _safe_token(operation_id),
                output_root=operation_output_root,
            )
            listener = getattr(self.client, "preview_listener", None)
            preview_context = (
                listener(
                    preview_destination,
                    None,
                    lambda current, total: self._report_progress(
                        current,
                        total,
                        actual_frames,
                        progress_callback,
                    ),
                )
                if callable(listener)
                else nullcontext()
            )
            with preview_context:
                prompt_id = self.client.queue_prompt(workflow)
                runtime_job_callback(prompt_id)
                history = self.client.wait_for_history(
                    prompt_id,
                    timeout_seconds=None,
                    runtime_stop_confirmed=cancel_requested,
                )
            if cancel_requested():
                self._cancel_prompt(prompt_id)
                raise VideoLocalAdapterError("video_generation_cancelled", retryable=True)
            outputs = self._video_outputs(history, prompt_id)
            if not outputs:
                raise VideoLocalAdapterError("video_runtime_output_missing", retryable=True)
            destination = self._prepare_output_subdirectory(
                "video",
                output_root=operation_output_root,
            )
            for index, output in enumerate(outputs):
                self._raise_if_cancelled(cancel_requested, prompt_id)
                filename = self._output_filename(output, operation_id, index)
                path = destination / filename
                payload = self.client.download_output(output)
                self._raise_if_cancelled(cancel_requested, prompt_id)
                path.write_bytes(payload)
                written.append(path)
                self._raise_if_cancelled(cancel_requested, prompt_id)
            for output in staged_working:
                self._validate_working_output_path(output.path)
            progress_callback(actual_frames, actual_frames)
            return LocalVideoExecutionResult(
                runtime_job_id=prompt_id,
                workflow_id=asset.workflow_id,
                output_paths=tuple(written),
                requested_frame_count=requested_frames,
                actual_frame_count=actual_frames,
                actual_duration_seconds=actual_frames / float(request["fps"]),
                working_outputs=tuple(staged_working),
            )
        except VideoLocalAdapterError:
            self._discard([*written, *(item.path for item in staged_working)])
            raise
        except RuntimeCancellationConfirmed as error:
            if prompt_id is not None:
                self._cancel_prompt(prompt_id)
            self._discard([*written, *(item.path for item in staged_working)])
            raise VideoLocalAdapterError("video_generation_cancelled", retryable=True) from error
        except (OSError, RuntimeErrorResponse, TypeError, ValueError, KeyError) as error:
            self._discard([*written, *(item.path for item in staged_working)])
            message = str(error).casefold()
            if "h3_motion_context" in message and any(
                token in message
                for token in (
                    "already patch",
                    "patch owner",
                    "second copy",
                    "different copy",
                    "another pack",
                )
            ):
                code = "video_motion_context_conflict"
            elif "h3_motion_context" in message and any(
                token in message
                for token in (
                    "patch could not be applied",
                    "self-test failed",
                    "upstream layout change",
                )
            ):
                code = "video_motion_context_runtime_incompatible"
            else:
                code = "video_runtime_failed"
            raise VideoLocalAdapterError(code, retryable=True) from error

    def _build_workflow(
        self,
        asset: VideoWorkflowAsset,
        packet: VideoSubmissionPacket,
        frame_count: int,
        operation_id: str,
    ) -> dict:
        request = packet.request_snapshot
        workflow = deepcopy(asset.workflow)
        resolved_models: ResolvedVideoModelSelection | None = None
        if request.get("schema_version") == "asteria.video.request.v2":
            if self.model_resolver is None:
                raise VideoLocalAdapterError("video_model_preflight_required")
            try:
                resolved_models = self.model_resolver.resolve(
                    VideoRequest.model_validate(request)
                )
            except ValueError as error:
                code = getattr(error, "code", "video_model_preflight_failed")
                raise VideoLocalAdapterError(code) from error
        prompt = str(request["prompt"])
        audio = request.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("prompt"), str):
            prompt = f"{prompt}\nAudio: {audio['prompt']}"
        values: dict[str, object] = {
            "prompt": prompt,
            "width": int(request["size"]["width"]),
            "height": int(request["size"]["height"]),
            "frame_count": frame_count,
            "fps": float(request["fps"]),
            "trim_fps": float(request["fps"]),
            "seed": int(request["seed"]),
            "output_prefix": f"asteria/video/{_safe_token(operation_id)}",
            "working_output_prefix": (
                f"asteria/working/{_safe_token(operation_id)}/latent"
            ),
        }
        if resolved_models is not None:
            components = resolved_models.components
            sampling = request.get("sampling")
            if not isinstance(sampling, dict):
                raise VideoLocalAdapterError("video_sampling_settings_required")
            values.update(
                {
                    "diffusion_model_name": components["diffusion_model"].loader_name,
                    "text_encoder_name": components["text_encoder"].loader_name,
                    "video_vae_name": components["video_vae"].loader_name,
                    "audio_vae_name": components["audio_vae"].loader_name,
                    "steps": int(sampling["steps"]),
                    "sampler": str(sampling["sampler"]),
                    "scheduler": str(sampling["scheduler"]),
                    "video_shift": float(sampling["video_shift"]),
                    "audio_shift": float(sampling["audio_shift"]),
                }
            )
        if request.get("mode") == "sequence":
            context_payload = packet.sequence_context
            if asset.workflow_id == "asteria_minimax_h3_sequence_initial.v1":
                if context_payload is not None:
                    raise VideoLocalAdapterError("video_sequence_context_mismatch")
                context_payload = None
            elif context_payload is None:
                raise VideoLocalAdapterError("video_sequence_context_required")
            if context_payload is None:
                context = None
            else:
                try:
                    context = VideoSequenceContextReceipt.model_validate(context_payload)
                except ValueError as error:
                    raise VideoLocalAdapterError("video_sequence_context_invalid") from error
            if context is None:
                pass
            elif context.mapping_id != packet.mapping_id or context.mapping_version != packet.mapping_version:
                raise VideoLocalAdapterError("video_sequence_context_mismatch")
            elif context.method == "direct_context":
                if (
                    asset.workflow_id != "asteria_minimax_h3_sequence_motion_context.v1"
                    or self.working_artifact_resolver is None
                    or context.source_working_artifact_id is None
                    or context.source_working_operation_id is None
                ):
                    raise VideoLocalAdapterError("video_sequence_context_mismatch")
                source = self.working_artifact_resolver.resolve(
                    operation_id=context.source_working_operation_id,
                    artifact_id=context.source_working_artifact_id,
                    expected_kind="av_latent",
                    media_artifact_id=context.source_artifact_id,
                )
                values["context_latent_path"] = str(source)
                values["video_context_frames"] = str(context.video_context_frames)
                values["audio_context_frames"] = context.audio_context_frames
            elif context.method == "reference_continuation":
                if asset.workflow_id != "asteria_minimax_h3_sequence_reference.v1":
                    raise VideoLocalAdapterError("video_sequence_context_mismatch")
                source = self.source_resolver.resolve(context.source_artifact_id, "video")
                values["context_video_filename"] = self.client.upload_file(source)
                reference_labels = "<Video 1>"
                if context.source_audio_range is not None:
                    reference_labels += " and its paired <Audio 1>"
                prompt = (
                    f"Continue directly from {reference_labels} without a visible cut.\n"
                    f"{prompt}"
                )
                values["prompt"] = prompt
            else:
                raise VideoLocalAdapterError("video_sequence_context_mismatch")
        first_id = request.get("first_frame_artifact_id")
        last_id = request.get("last_frame_artifact_id")
        if isinstance(first_id, str):
            values["first_frame_filename"] = self.client.upload_file(
                self.source_resolver.resolve(first_id, "image")
            )
        if isinstance(last_id, str):
            values["last_frame_filename"] = self.client.upload_file(
                self.source_resolver.resolve(last_id, "image")
            )
        for name, target in asset.bindings.items():
            if name not in values:
                if resolved_models is None and name in _V2_PRODUCTION_BINDINGS:
                    continue
                raise VideoLocalAdapterError("video_workflow_invalid")
            self._assign_binding(workflow, target, values[name])
        references = tuple(VideoReference.model_validate(value) for value in request.get("references", []))
        if references:
            self._apply_reference_bindings(workflow, asset.dynamic_bindings, references)
        if resolved_models is not None:
            self._apply_lora_bindings(
                workflow,
                asset.dynamic_bindings,
                resolved_models,
            )
        if self._contains_placeholder(workflow):
            raise VideoLocalAdapterError("video_workflow_invalid")
        if resolved_models is not None:
            self._validate_runtime_contract(asset, workflow, resolved_models)
        return workflow

    def build_workflow_for_adapter(
        self,
        asset: WorkflowAsset,
        packet: VideoSubmissionPacket,
        actual_frames: int,
        operation_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Build the validated H3 base graph for a composed Asteria adapter."""
        return self._build_workflow(asset, packet, actual_frames, operation_id)

    def _expected_working_outputs(
        self,
        asset: VideoWorkflowAsset,
        workflow: dict,
        packet: VideoSubmissionPacket,
    ) -> list[LocalVideoWorkingOutput]:
        if not asset.working_outputs:
            return []
        if self.comfy_output_root is None:
            raise VideoLocalAdapterError("video_working_output_root_unavailable")
        context = (
            VideoSequenceContextReceipt.model_validate(packet.sequence_context)
            if packet.sequence_context is not None
            else None
        )
        outputs: list[LocalVideoWorkingOutput] = []
        for contract in asset.working_outputs:
            node = workflow.get(contract.node_id)
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                raise VideoLocalAdapterError("video_workflow_invalid")
            prefix_value = inputs.get(contract.path_input)
            index_value = inputs.get(contract.index_input)
            if not isinstance(prefix_value, str):
                raise VideoLocalAdapterError("video_workflow_invalid")
            try:
                clip_index = int(index_value)
            except (TypeError, ValueError) as error:
                raise VideoLocalAdapterError("video_workflow_invalid") from error
            if clip_index < 1:
                raise VideoLocalAdapterError("video_working_output_identity_invalid")
            prefix = Path(prefix_value)
            if prefix.is_absolute() or ".." in prefix.parts or not prefix.name:
                raise VideoLocalAdapterError("video_working_output_identity_invalid")
            path = (
                self.comfy_output_root
                / prefix.parent
                / f"{prefix.name}_{clip_index:05d}{contract.suffix}"
            ).resolve()
            try:
                path.relative_to(self.comfy_output_root)
            except ValueError as error:
                raise VideoLocalAdapterError("video_working_output_path_escape") from error
            outputs.append(
                LocalVideoWorkingOutput(
                    kind=contract.kind,
                    path=path,
                    source_sha256=(
                        (context.source_sha256,) if context is not None else ()
                    ),
                    conditioning_sha256=(
                        context.context_sha256 if context is not None else None
                    ),
                )
            )
        return outputs

    def _validate_working_output_path(self, path: Path) -> None:
        if self.comfy_output_root is None:
            raise VideoLocalAdapterError("video_working_output_root_unavailable")
        resolved = path.resolve()
        try:
            resolved.relative_to(self.comfy_output_root)
        except ValueError as error:
            raise VideoLocalAdapterError("video_working_output_path_escape") from error
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or _contains_reparse_component(resolved, self.comfy_output_root)
        ):
            raise VideoLocalAdapterError("video_working_output_missing", retryable=True)

    def _apply_lora_bindings(
        self,
        workflow: dict,
        dynamic_bindings: dict[str, dict],
        resolved: ResolvedVideoModelSelection,
    ) -> None:
        if not resolved.loras:
            return
        contract = dynamic_bindings.get("loras")
        if not isinstance(contract, dict):
            raise VideoLocalAdapterError("video_workflow_invalid")
        class_type = str(contract.get("class_type") or "")
        source_node_id = str(contract.get("source_node_id") or "")
        target_node_id = str(contract.get("target_node_id") or "")
        target_input = str(contract.get("target_input") or "")
        model_input = str(contract.get("model_input") or "")
        name_input = str(contract.get("lora_name_input") or "")
        strength_input = str(contract.get("strength_input") or "")
        output_index = contract.get("output_index")
        target = workflow.get(target_node_id)
        if (
            not class_type
            or not source_node_id
            or not model_input
            or not name_input
            or not strength_input
            or not isinstance(output_index, int)
            or not isinstance(target, dict)
            or not isinstance(target.get("inputs"), dict)
            or not target_input
        ):
            raise VideoLocalAdapterError("video_workflow_invalid")
        previous_node_id = source_node_id
        for index, lora in enumerate(resolved.loras):
            node_id = f"asteria-video-lora-{index}"
            if node_id in workflow:
                raise VideoLocalAdapterError("video_workflow_invalid")
            workflow[node_id] = {
                "class_type": class_type,
                "inputs": {
                    model_input: [previous_node_id, output_index],
                    name_input: lora.loader_name,
                    strength_input: lora.weight,
                },
            }
            previous_node_id = node_id
        target["inputs"][target_input] = [previous_node_id, output_index]

    def _validate_runtime_contract(
        self,
        asset: VideoWorkflowAsset,
        workflow: dict,
        resolved: ResolvedVideoModelSelection,
    ) -> None:
        object_info_provider = getattr(self.client, "object_info", None)
        if not callable(object_info_provider):
            raise VideoLocalAdapterError("video_runtime_contract_unavailable")
        try:
            object_info = object_info_provider()
        except (OSError, RuntimeErrorResponse, TypeError, ValueError) as error:
            raise VideoLocalAdapterError(
                "video_runtime_contract_unavailable", retryable=True
            ) from error
        if not isinstance(object_info, dict):
            raise VideoLocalAdapterError("video_runtime_contract_unavailable")
        class_types = {
            str(node.get("class_type") or "")
            for node in workflow.values()
            if isinstance(node, dict)
        }
        missing = {
            class_type
            for class_type in class_types
            if not isinstance(object_info.get(class_type), dict)
        }
        if missing:
            raise VideoLocalAdapterError("video_runtime_nodes_unavailable")

        for class_type, input_names in asset.required_node_inputs.items():
            for input_name in input_names:
                if self._runtime_input_definition(object_info, class_type, input_name) is None:
                    raise VideoLocalAdapterError("video_runtime_input_contract_changed")

        for binding_name in (
            "diffusion_model_name",
            "text_encoder_name",
            "video_vae_name",
            "audio_vae_name",
            "steps",
            "sampler",
            "scheduler",
            "video_shift",
            "audio_shift",
        ):
            target = asset.bindings.get(binding_name)
            if not target:
                raise VideoLocalAdapterError("video_workflow_invalid")
            node_id, _, input_name = target.partition(".inputs.")
            node = workflow.get(node_id)
            class_type = node.get("class_type") if isinstance(node, dict) else None
            if (
                not isinstance(class_type, str)
                or self._runtime_input_definition(object_info, class_type, input_name)
                is None
            ):
                raise VideoLocalAdapterError("video_runtime_input_contract_changed")

        selected_names = {
            "diffusion_model": ("UNETLoader", "unet_name"),
            "text_encoder": ("CLIPLoader", "clip_name"),
            "video_vae": ("VAELoader", "vae_name"),
            "audio_vae": ("VAELoader", "vae_name"),
        }
        for role, (class_type, input_name) in selected_names.items():
            available = self._runtime_input_values(object_info, class_type, input_name)
            if available is None:
                raise VideoLocalAdapterError("video_runtime_input_contract_changed")
            if resolved.components[role].loader_name not in available:
                raise VideoLocalAdapterError("video_model_runtime_unavailable")

        sampling = {
            "sampler": ("KSamplerSelect", "sampler_name", "video_runtime_sampler_unavailable"),
            "scheduler": ("BasicScheduler", "scheduler", "video_runtime_scheduler_unavailable"),
        }
        for binding_name, (class_type, input_name, error_code) in sampling.items():
            target = asset.bindings[binding_name]
            node_id = target.split(".", 1)[0]
            value = str(workflow[node_id]["inputs"][input_name])
            available = self._runtime_input_values(object_info, class_type, input_name)
            if available is None:
                raise VideoLocalAdapterError("video_runtime_input_contract_changed")
            if value not in available:
                raise VideoLocalAdapterError(error_code)

        if resolved.loras:
            contract = asset.dynamic_bindings.get("loras")
            if not isinstance(contract, dict):
                raise VideoLocalAdapterError("video_workflow_invalid")
            class_type = str(contract.get("class_type") or "")
            for input_key in ("model_input", "lora_name_input", "strength_input"):
                input_name = str(contract.get(input_key) or "")
                if self._runtime_input_definition(object_info, class_type, input_name) is None:
                    raise VideoLocalAdapterError("video_runtime_lora_contract_changed")
            available = self._runtime_input_values(
                object_info,
                class_type,
                str(contract["lora_name_input"]),
            )
            if available is None:
                raise VideoLocalAdapterError("video_runtime_lora_contract_changed")
            if any(lora.loader_name not in available for lora in resolved.loras):
                raise VideoLocalAdapterError("video_lora_runtime_unavailable")

    @staticmethod
    def _runtime_input_definition(
        object_info: dict,
        class_type: str,
        input_name: str,
    ) -> object | None:
        node = object_info.get(class_type)
        if not isinstance(node, dict):
            return None
        inputs = node.get("input")
        if not isinstance(inputs, dict):
            return None
        for section in ("required", "optional"):
            values = inputs.get(section)
            if isinstance(values, dict) and input_name in values:
                return values[input_name]
        return None

    @classmethod
    def _runtime_input_values(
        cls,
        object_info: dict,
        class_type: str,
        input_name: str,
    ) -> set[str] | None:
        raw = cls._runtime_input_definition(object_info, class_type, input_name)
        if not isinstance(raw, (list, tuple)) or not raw:
            return None
        values: object = raw[0]
        if (
            values == "COMBO"
            and len(raw) >= 2
            and isinstance(raw[1], dict)
        ):
            values = raw[1].get("options")
        if not isinstance(values, (list, tuple)):
            return None
        return {str(value) for value in values if isinstance(value, str)}

    def _apply_reference_bindings(
        self,
        workflow: dict,
        dynamic_bindings: dict[str, dict],
        references: tuple[VideoReference, ...],
    ) -> None:
        contract = dynamic_bindings.get("references")
        if not isinstance(contract, dict):
            raise VideoLocalAdapterError("video_workflow_invalid")
        target_id = str(contract.get("target_node_id") or "")
        target = workflow.get(target_id)
        groups = contract.get("groups")
        if not isinstance(target, dict) or not isinstance(target.get("inputs"), dict) or not isinstance(groups, dict):
            raise VideoLocalAdapterError("video_workflow_invalid")
        ordinals: dict[str, int] = {"image": 0, "video": 0, "audio": 0}
        for reference in references:
            group = groups.get(reference.kind)
            if not isinstance(group, dict):
                raise VideoLocalAdapterError("video_workflow_invalid")
            ordinal = ordinals[reference.kind]
            ordinals[reference.kind] += 1
            maximum = int(group.get("maximum", 0))
            index = int(group.get("index_base", 0)) + ordinal
            if maximum < 1 or ordinal >= maximum:
                raise VideoLocalAdapterError("video_reference_limit")
            source = self.source_resolver.resolve(reference.artifact_id, reference.kind)
            uploaded = self.client.upload_file(source)
            loader = group.get("loader")
            if not isinstance(loader, dict):
                raise VideoLocalAdapterError("video_workflow_invalid")
            loader_id = f"asteria-ref-{reference.kind}-{ordinal}"
            workflow[loader_id] = {
                "class_type": loader["class_type"],
                "inputs": {str(loader["filename_input"]): uploaded},
            }
            output_index = int(loader.get("output_index", 0))
            if reference.kind == "video":
                components = group.get("components")
                if not isinstance(components, dict):
                    raise VideoLocalAdapterError("video_workflow_invalid")
                component_id = f"asteria-ref-video-components-{ordinal}"
                workflow[component_id] = {
                    "class_type": components["class_type"],
                    "inputs": {str(components["video_input"]): [loader_id, output_index]},
                }
                target["inputs"][f"{group['target_input_prefix']}{index}"] = [
                    component_id,
                    int(components["frames_output_index"]),
                ]
                target["inputs"][f"{group['paired_audio_input_prefix']}{index}"] = [
                    component_id,
                    int(components["audio_output_index"]),
                ]
            else:
                target["inputs"][f"{group['target_input_prefix']}{index}"] = [loader_id, output_index]

    @staticmethod
    def _assign_binding(workflow: dict, target: str, value: object) -> None:
        parts = target.split(".")
        if len(parts) != 3 or parts[1] != "inputs":
            raise VideoLocalAdapterError("video_workflow_invalid")
        node = workflow.get(parts[0])
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise VideoLocalAdapterError("video_workflow_invalid")
        node["inputs"][parts[2]] = value

    @staticmethod
    def _contains_placeholder(value: object) -> bool:
        if isinstance(value, str):
            return value.startswith("${") and value.endswith("}")
        if isinstance(value, dict):
            return any(LocalH3VideoExecutionAdapter._contains_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(LocalH3VideoExecutionAdapter._contains_placeholder(item) for item in value)
        return False

    @staticmethod
    def _video_outputs(history: dict, prompt_id: str) -> tuple[dict, ...]:
        prompt = history.get(prompt_id)
        if not isinstance(prompt, dict) or not isinstance(prompt.get("outputs"), dict):
            return ()
        outputs: list[dict] = []
        for node in prompt["outputs"].values():
            outputs.extend(LocalH3VideoExecutionAdapter._saved_video_descriptors(node))
        return tuple(outputs)

    @staticmethod
    def _saved_video_descriptors(value: object) -> list[dict]:
        if isinstance(value, dict):
            filename = value.get("filename")
            if (
                isinstance(filename, str)
                and Path(filename).suffix.lower() in {".mp4", ".webm"}
                and isinstance(value.get("subfolder", ""), str)
                and isinstance(value.get("type", "output"), str)
            ):
                return [value]
            found: list[dict] = []
            for nested in value.values():
                found.extend(LocalH3VideoExecutionAdapter._saved_video_descriptors(nested))
            return found
        if isinstance(value, list):
            found = []
            for nested in value:
                found.extend(LocalH3VideoExecutionAdapter._saved_video_descriptors(nested))
            return found
        return []

    @staticmethod
    def _output_filename(output: dict, operation_id: str, index: int) -> str:
        source = Path(str(output.get("filename") or "clip.mp4")).name
        suffix = Path(source).suffix.lower()
        if suffix not in {".mp4", ".webm"}:
            raise VideoLocalAdapterError("video_runtime_output_invalid")
        stem = _SAFE_FILENAME.sub("_", Path(source).stem).strip("._-") or "clip"
        return f"{_safe_token(operation_id)}-{index + 1}-{stem}{suffix}"

    def _cancel_prompt(self, prompt_id: str) -> None:
        try:
            state = self.client.queue_state()
            if prompt_id in state.get("pending", []):
                self.client.delete_queued_prompt(prompt_id)
            elif prompt_id in state.get("running", []):
                self.client.interrupt()
        except (OSError, RuntimeErrorResponse, TypeError, ValueError):
            return

    def _raise_if_cancelled(
        self,
        cancel_requested: Callable[[], bool],
        prompt_id: str,
    ) -> None:
        if not cancel_requested():
            return
        self._cancel_prompt(prompt_id)
        raise VideoLocalAdapterError("video_generation_cancelled", retryable=True)

    def _prepare_output_subdirectory(self, *parts: str, output_root: Path | None = None) -> Path:
        try:
            selected_root = Path(output_root) if output_root is not None else self.output_root
            selected_root.mkdir(parents=True, exist_ok=True)
            resolved_root = selected_root.resolve(strict=True)
            current = selected_root
            for part in parts:
                if not part or part in {".", ".."} or Path(part).name != part:
                    raise VideoLocalAdapterError("video_output_path_unsafe")
                current = current / part
                current.mkdir(exist_ok=True)
                if _is_reparse_path(current):
                    raise VideoLocalAdapterError("video_output_path_unsafe")
                resolved = current.resolve(strict=True)
                if not resolved.is_relative_to(resolved_root):
                    raise VideoLocalAdapterError("video_output_path_unsafe")
            return current
        except VideoLocalAdapterError:
            raise
        except OSError as error:
            raise VideoLocalAdapterError("video_output_path_unsafe") from error

    @staticmethod
    def _report_progress(
        current: int,
        total: int,
        frame_count: int,
        callback: Callable[[int, int], None],
    ) -> None:
        if total <= 0:
            return
        normalized = min(frame_count, max(0, math.floor(frame_count * current / total)))
        callback(normalized, frame_count)

    @staticmethod
    def _discard(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue


def _declared_dynamic_class_types(value: object) -> set[str]:
    declared: set[str] = set()
    if isinstance(value, dict):
        class_type = value.get("class_type")
        if isinstance(class_type, str):
            declared.add(class_type)
        for nested in value.values():
            declared.update(_declared_dynamic_class_types(nested))
    elif isinstance(value, list):
        for nested in value:
            declared.update(_declared_dynamic_class_types(nested))
    return declared


def _safe_token(value: str) -> str:
    normalized = _SAFE_FILENAME.sub("_", value).strip("._-")
    if not normalized:
        raise VideoLocalAdapterError("video_operation_id_invalid")
    return normalized[:128]


def _is_reparse_path(path: Path) -> bool:
    try:
        value = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if attributes & reparse_flag:
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction(path)) if callable(isjunction) else False


def _contains_reparse_component(path: Path, root: Path) -> bool:
    current = path
    root = root.resolve()
    while True:
        if _is_reparse_path(current):
            return True
        if current == root:
            return False
        if current.parent == current:
            return True
        current = current.parent
