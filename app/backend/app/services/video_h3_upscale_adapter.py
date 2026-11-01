from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable

from app.schemas.video import VideoSamplingSettings, VideoSize
from app.services.runtime import RuntimeCancellationConfirmed, RuntimeErrorResponse
from app.services.video_local_adapter import (
    LocalH3VideoExecutionAdapter,
    VideoLocalAdapterError,
)
from app.services.video_model_resolver import VideoModelResolver
from app.services.video_packet import VideoPacketBuilder
from app.services.video_runtime_adapters import LocalH3VideoAdapter
from app.services.video_upscale_adapters import (
    VideoUpscaleAdapterError,
    VideoUpscaleExecutionInput,
    VideoUpscaleExecutionResult,
)
from app.services.video_upscale_bundles import canonical_sha256


H3_UPSCALE_WORKFLOW_ID = "asteria_minimax_h3_upscale_after_preview.v1"
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")
_REMOVE_OUTPUT_CLASSES = {
    "SaveVideo",
    "MiniMaxH3MotionContextSaveLatent",
}
_UPSCALE_INPUTS = {
    "MMH3UltimateUpscale": {
        "model",
        "conditioning",
        "latent",
        "noise",
        "sampler",
        "sigmas",
        "cfg",
    },
    "MMH3TemporalSplitParams": {
        "chunk_length",
        "temporal_overlap",
        "anchor_strength",
    },
    "MMH3SpatialSplitParams": {
        "tile_width",
        "tile_height",
        "spatial_w_overlap",
        "spatial_h_overlap",
        "fade_width",
        "fade_height",
        "min_tile_size",
        "overlap_mode",
        "overlap_blend",
    },
    "MMH3LatentUpscaleWithModelParams": {
        "model_name",
        "width",
        "height",
        "device",
        "precision",
    },
    "MMH3LatentUpscaleParams": {"method", "width", "height"},
    "AsteriaH3UpscaleLoadLatent": {"latent_path"},
}


class ComfyH3VideoUpscaleAdapter:
    id = "minimax_h3_ultimate_upscale"
    revision = "1"

    def __init__(
        self,
        *,
        base_adapter: LocalH3VideoExecutionAdapter,
        model_resolver: VideoModelResolver,
        packet_builder: VideoPacketBuilder | None = None,
    ) -> None:
        self.base_adapter = base_adapter
        self.client = base_adapter.client
        self.model_resolver = model_resolver
        self.packet_builder = packet_builder or VideoPacketBuilder(
            model_preflight=model_resolver
        )

    def execute(
        self,
        execution_input: VideoUpscaleExecutionInput,
        *,
        operation_id: str,
        progress_callback: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
        runtime_job_callback: Callable[[str], None],
    ) -> VideoUpscaleExecutionResult:
        prompt_id: str | None = None
        written: list[Path] = []
        try:
            if cancel_requested():
                raise VideoUpscaleAdapterError("video_upscale_cancelled", retryable=True)
            workflow = self._build_workflow(execution_input, operation_id)
            destination = self._output_directory(
                execution_input.output_root,
                "video-upscale",
            )
            listener = getattr(self.client, "preview_listener", None)
            preview_context = (
                listener(destination / ".previews", None, progress_callback)
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
                self._cancel(prompt_id)
                raise VideoUpscaleAdapterError("video_upscale_cancelled", retryable=True)
            descriptors = self._video_outputs(history, prompt_id)
            if not descriptors:
                raise VideoUpscaleAdapterError(
                    "video_upscale_output_missing", retryable=True
                )
            for index, descriptor in enumerate(descriptors):
                if cancel_requested():
                    self._cancel(prompt_id)
                    raise VideoUpscaleAdapterError(
                        "video_upscale_cancelled", retryable=True
                    )
                payload = self.client.download_output(descriptor)
                path = destination / self._filename(operation_id, descriptor, index)
                temporary = path.with_name(f".{path.name}.part")
                written.append(temporary)
                temporary.write_bytes(payload)
                temporary.replace(path)
                written[-1] = path
            progress_callback(1, 1)
            return VideoUpscaleExecutionResult(
                runtime_job_id=prompt_id,
                workflow_id=H3_UPSCALE_WORKFLOW_ID,
                output_paths=tuple(written),
                target_width=execution_input.target_width,
                target_height=execution_input.target_height,
            )
        except VideoUpscaleAdapterError:
            self._discard(written)
            raise
        except RuntimeCancellationConfirmed as error:
            if prompt_id is not None:
                self._cancel(prompt_id)
            self._discard(written)
            raise VideoUpscaleAdapterError(
                "video_upscale_cancelled", retryable=True
            ) from error
        except (OSError, RuntimeErrorResponse, TypeError, ValueError, KeyError) as error:
            self._discard(written)
            raise VideoUpscaleAdapterError(
                "video_upscale_runtime_failed", retryable=True
            ) from error

    def _build_workflow(
        self,
        execution_input: VideoUpscaleExecutionInput,
        operation_id: str,
    ) -> dict[str, dict[str, Any]]:
        bundle = execution_input.bundle
        if canonical_sha256(dict(bundle.packet_snapshot)) != bundle.packet_sha256:
            raise VideoUpscaleAdapterError("video_upscale_bundle_changed")
        source_request = bundle.request_snapshot
        if source_request.schema_version != "asteria.video.request.v2":
            raise VideoUpscaleAdapterError("video_upscale_source_request_unsupported")
        if source_request.mode == "sequence":
            raise VideoUpscaleAdapterError("video_upscale_sequence_source_unsupported")
        if source_request.model_selection is None or source_request.sampling is None:
            raise VideoUpscaleAdapterError("video_upscale_source_request_unsupported")
        if execution_input.request.sampling.cfg != 1:
            raise VideoUpscaleAdapterError("video_upscale_cfg_unsupported")
        sampling = source_request.sampling.model_copy(
            update={
                "steps": execution_input.request.sampling.steps,
                "sampler": execution_input.request.sampling.sampler,
                "scheduler": execution_input.request.sampling.scheduler,
                "batch_count": 1,
                "seed_policy": "increment",
            }
        )
        if not isinstance(sampling, VideoSamplingSettings):
            raise VideoUpscaleAdapterError("video_upscale_sampling_invalid")
        derived_request = source_request.model_copy(
            update={
                "seed": execution_input.resolved_seed,
                "size": VideoSize(
                    width=execution_input.target_width,
                    height=execution_input.target_height,
                    aspect_ratio="adaptive",
                    resolution_intent="custom",
                ),
                "duration_seconds": (
                    bundle.actual_frame_count
                    / (bundle.fps_numerator / bundle.fps_denominator)
                ),
                "sampling": sampling,
                "provider_options": {
                    **source_request.provider_options,
                    "offload": execution_input.request.offload,
                },
            }
        )
        packet = self.packet_builder.build(derived_request, LocalH3VideoAdapter())
        asset = self.base_adapter.workflows.load(
            str(packet.provider_payload.get("workflow_template_id") or "")
        )
        base_workflow = self.base_adapter.build_workflow_for_adapter(
            asset,
            packet,
            bundle.actual_frame_count,
            operation_id,
        )
        workflow = deepcopy(base_workflow)
        refs = self._base_refs(workflow)
        for node_id in tuple(workflow):
            node = workflow[node_id]
            if node.get("class_type") in _REMOVE_OUTPUT_CLASSES:
                workflow.pop(node_id)

        workflow["asteria-upscale-load-latent"] = {
            "class_type": "AsteriaH3UpscaleLoadLatent",
            "inputs": {
                "latent_path": str(execution_input.latent_path),
            },
        }
        upscale_link: list[object]
        request = execution_input.request
        if request.mode == "h3_3d_model":
            assert request.upscaler_model is not None
            resolved_upscaler = self.model_resolver.resolve_upscaler(
                request.upscaler_model,
                profile_id=source_request.model_profile,
                variant=source_request.model_selection.variant,
            )
            workflow["asteria-upscale-params"] = {
                "class_type": "MMH3LatentUpscaleWithModelParams",
                "inputs": {
                    "model_name": resolved_upscaler.loader_name,
                    "width": execution_input.target_width,
                    "height": execution_input.target_height,
                    "device": request.device,
                    "precision": request.precision,
                },
            }
        else:
            assert request.interpolation is not None
            workflow["asteria-upscale-params"] = {
                "class_type": "MMH3LatentUpscaleParams",
                "inputs": {
                    "method": request.interpolation,
                    "width": execution_input.target_width,
                    "height": execution_input.target_height,
                },
            }
        upscale_link = ["asteria-upscale-params", 0]
        ultimate_inputs: dict[str, object] = {
            "model": refs["model"],
            "conditioning": refs["conditioning"],
            "latent": ["asteria-upscale-load-latent", 0],
            "noise": refs["noise"],
            "sampler": refs["sampler"],
            "sigmas": refs["sigmas"],
            "cfg": request.sampling.cfg,
            "latent_upscale_param": upscale_link,
        }
        if request.temporal is not None:
            temporal = request.temporal
            workflow["asteria-upscale-temporal"] = {
                "class_type": "MMH3TemporalSplitParams",
                "inputs": {
                    "chunk_length": temporal.chunk_length,
                    "temporal_overlap": temporal.overlap,
                    "anchor_strength": temporal.anchor_strength,
                },
            }
            ultimate_inputs["temporal_split_param"] = ["asteria-upscale-temporal", 0]
        if request.spatial is not None:
            spatial = request.spatial
            if (
                execution_input.target_width < spatial.tile_width
                or execution_input.target_height < spatial.tile_height
            ):
                raise VideoUpscaleAdapterError("video_upscale_tile_exceeds_target")
            workflow["asteria-upscale-spatial"] = {
                "class_type": "MMH3SpatialSplitParams",
                "inputs": {
                    "tile_width": spatial.tile_width,
                    "tile_height": spatial.tile_height,
                    "spatial_w_overlap": spatial.width_overlap,
                    "spatial_h_overlap": spatial.height_overlap,
                    "fade_width": spatial.fade_width,
                    "fade_height": spatial.fade_height,
                    "min_tile_size": spatial.minimum_tile_size,
                    "overlap_mode": spatial.overlap_mode,
                    "overlap_blend": spatial.overlap_blend,
                },
            }
            ultimate_inputs["spatial_split_param"] = ["asteria-upscale-spatial", 0]
        workflow["asteria-upscale-run"] = {
            "class_type": "MMH3UltimateUpscale",
            "inputs": ultimate_inputs,
        }
        workflow["asteria-upscale-video-decode"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["asteria-upscale-run", 0], "vae": refs["video_vae"]},
        }
        workflow["asteria-upscale-audio-decode"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["asteria-upscale-run", 0], "vae": refs["audio_vae"]},
        }
        workflow["asteria-upscale-create-video"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["asteria-upscale-video-decode", 0],
                "audio": ["asteria-upscale-audio-decode", 0],
                "fps": bundle.fps_numerator / bundle.fps_denominator,
                "bit_depth": 8,
            },
        }
        workflow["asteria-upscale-save"] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["asteria-upscale-create-video", 0],
                "filename_prefix": f"asteria/video-upscale/{self._safe(operation_id)}",
                "format": "auto",
                "codec": "auto",
            },
        }
        self._validate_runtime_contract(workflow)
        return workflow

    @staticmethod
    def _base_refs(workflow: dict[str, dict[str, Any]]) -> dict[str, list[object]]:
        def first(class_type: str) -> tuple[str, dict[str, Any]]:
            matches = [
                (node_id, node)
                for node_id, node in workflow.items()
                if node.get("class_type") == class_type
            ]
            if len(matches) != 1:
                raise VideoUpscaleAdapterError("video_upscale_base_workflow_invalid")
            return matches[0]

        shift_id, _shift = first("MiniMaxH3SigmaShift")
        noise_id, _noise = first("RandomNoise")
        sampler_id, _sampler = first("KSamplerSelect")
        scheduler_id, _scheduler = first("BasicScheduler")
        conditioning_matches = [
            (node_id, node)
            for node_id, node in workflow.items()
            if node.get("class_type")
            in {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"}
        ]
        if len(conditioning_matches) != 1:
            raise VideoUpscaleAdapterError("video_upscale_base_workflow_invalid")
        conditioning_id = conditioning_matches[0][0]
        _, video_decode = first("VAEDecode")
        _, audio_decode = first("VAEDecodeAudio")
        video_vae = video_decode.get("inputs", {}).get("vae")
        audio_vae = audio_decode.get("inputs", {}).get("vae")
        if not isinstance(video_vae, list) or not isinstance(audio_vae, list):
            raise VideoUpscaleAdapterError("video_upscale_base_workflow_invalid")
        return {
            "model": [shift_id, 0],
            "conditioning": [conditioning_id, 0],
            "noise": [noise_id, 0],
            "sampler": [sampler_id, 0],
            "sigmas": [scheduler_id, 0],
            "video_vae": video_vae,
            "audio_vae": audio_vae,
        }

    def _validate_runtime_contract(self, workflow: dict[str, dict[str, Any]]) -> None:
        provider = getattr(self.client, "object_info", None)
        if not callable(provider):
            raise VideoUpscaleAdapterError("video_upscale_runtime_contract_unavailable")
        info = provider()
        if not isinstance(info, dict):
            raise VideoUpscaleAdapterError("video_upscale_runtime_contract_unavailable")
        for class_type, required_inputs in _UPSCALE_INPUTS.items():
            if not any(node.get("class_type") == class_type for node in workflow.values()):
                continue
            node = info.get(class_type)
            if not isinstance(node, dict):
                raise VideoUpscaleAdapterError("video_upscale_runtime_nodes_unavailable")
            inputs = node.get("input")
            if not isinstance(inputs, dict):
                raise VideoUpscaleAdapterError("video_upscale_runtime_contract_changed")
            available: set[str] = set()
            for section in ("required", "optional"):
                values = inputs.get(section, {})
                if isinstance(values, dict):
                    available.update(str(key) for key in values)
            if not required_inputs.issubset(available):
                raise VideoUpscaleAdapterError("video_upscale_runtime_contract_changed")
        model_node = workflow.get("asteria-upscale-params")
        if model_node and model_node.get("class_type") == "MMH3LatentUpscaleWithModelParams":
            allowed = LocalH3VideoExecutionAdapter._runtime_input_values(
                info,
                "MMH3LatentUpscaleWithModelParams",
                "model_name",
            )
            selected = model_node["inputs"]["model_name"]
            if allowed is None or selected not in allowed:
                raise VideoUpscaleAdapterError("video_upscale_model_unavailable")

    @staticmethod
    def _video_outputs(history: dict, prompt_id: str) -> list[dict]:
        prompt = history.get(prompt_id)
        outputs = prompt.get("outputs", {}) if isinstance(prompt, dict) else {}
        values: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        if not isinstance(outputs, dict):
            return values
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            for key in ("videos", "images", "gifs"):
                items = output.get(key)
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        filename = str(item.get("filename") or "")
                        if Path(filename).suffix.lower() not in {".mp4", ".webm"}:
                            continue
                        identity = (
                            filename,
                            str(item.get("subfolder") or ""),
                            str(item.get("type") or ""),
                        )
                        if identity in seen:
                            continue
                        seen.add(identity)
                        values.append(item)
        return values

    @staticmethod
    def _output_directory(root: Path, name: str) -> Path:
        resolved_root = Path(root).resolve()
        destination = resolved_root / name
        destination.mkdir(parents=True, exist_ok=True)
        if (
            not destination.resolve().is_relative_to(resolved_root)
            or _contains_reparse_component(destination, resolved_root)
        ):
            raise VideoUpscaleAdapterError("video_upscale_output_path_unsafe")
        return destination

    @classmethod
    def _filename(cls, operation_id: str, descriptor: dict, index: int) -> str:
        suffix = Path(str(descriptor.get("filename") or "")).suffix.lower()
        if suffix not in {".mp4", ".webm"}:
            suffix = ".mp4"
        return f"{cls._safe(operation_id)}_{index + 1}{suffix}"

    @staticmethod
    def _safe(value: str) -> str:
        normalized = _SAFE_TOKEN.sub("-", value).strip("-._")
        return normalized[:96] or "video-upscale"

    def _cancel(self, prompt_id: str) -> None:
        try:
            state = self.client.queue_state()
            if prompt_id in state.get("pending", []):
                self.client.delete_queued_prompt(prompt_id)
            elif prompt_id in state.get("running", []):
                self.client.interrupt()
        except (OSError, RuntimeErrorResponse, TypeError, ValueError):
            pass

    @staticmethod
    def _discard(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue


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
