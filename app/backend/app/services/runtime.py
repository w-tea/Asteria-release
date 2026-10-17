from __future__ import annotations

import json
import math
import random
import re
import struct
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from uuid import uuid4

import httpx
from PIL import Image, ImageFilter, ImageOps

try:
    import websocket
except ImportError:  # pragma: no cover - optional runtime enhancement.
    websocket = None

from app.core.config import AppConfig
from app.services import api_errors
from app.services.comfy_enhancement_stages import mark_stage, decoded_stage_kind, append_stage_outputs
from app.services.enhancement_stages import stage_receipt, copy_stage
from app.services.comfy_admission import (
    ComfyAdmissionError,
    ComfyAdmissionService,
    ComfyAdmissionTimeoutError,
    ComfyCleanupPendingError,
    comfy_admission,
)
from app.services.comfy_model_paths import comfy_model_name
from app.services.external_model_libraries import ExternalModelRoot
from app.services.controlnet_preprocessor import (
    COMFY_CONTROLNET_PREPROCESSOR_NODES,
    ControlNetPreprocessorError,
    preprocess_control_image,
)
from app.services.controlnet_mode import (
    controlnet_mode_contract_status,
    enabled_priority_controlnet_units,
    resolve_controlnet_mode_contract,
)
from app.services.detailer_catalog import detailer_detector_model
from app.services.inpaint_region import (
    InpaintRegionError,
    compose_only_masked_output,
    prepare_only_masked_region,
)
from app.services.sampling_catalog import sampler_runtime_name, scheduler_runtime_name
from app.services.seed_resize import seed_resize_active
from app.services.topaz import TopazSettings, process_topaz_image
from app.services.upscaler_catalog import upscaler_model_name, upscaler_option
from app.services.prompt_syntax import PromptSyntaxError, PromptSyntaxPlan, prompt_syntax_plan_from_dict
from app.schemas.prompt_input import join_prompt_contract_fragments
from app.schemas.spatial_prompt_runtime import (
    SpatialMaskBundle,
    SpatialModelProfileReceipt,
    SpatialPromptPlan,
    canonical_runtime_sha256,
)
from app.services.spatial_conditioning_builder import (
    SpatialConditioningBuilderError,
    build_spatial_conditioning_bundle,
    spatial_prompt_stage,
)
from app.services.spatial_prompt_mapping import (
    ASTERIA_SPATIAL_NODE_CONTRACT_VERSION,
    SpatialPromptMappingError,
    append_spatial_execution_receipt,
    append_spatial_prompt_mapping,
    spatial_runtime_mask_receipt_sha256,
)
from app.services.spatial_generation_context import (
    SpatialGenerationContextError,
    resize_spatial_runtime_stage,
    upload_spatial_runtime_masks,
)
from app.services.workflow_mapping import WorkflowMapping, WorkflowMappingRepository


class RuntimeErrorResponse(Exception):
    pass


class RuntimeCancellationConfirmed(RuntimeErrorResponse):
    pass


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_exists_safely(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_file_safely(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _unlink_safely(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _log_messages(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _size_tuple(value: object) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _size_dict(size: tuple[int, int] | None) -> dict[str, int] | None:
    if size is None:
        return None
    return {"width": size[0], "height": size[1]}


def _hires_checkpoint_setting(settings: dict) -> str:
    checkpoint = str(settings.get("hires_checkpoint") or "").strip()
    if not checkpoint or checkpoint == "Use same checkpoint":
        return ""
    return checkpoint


def _hires_noise_mode(settings: dict) -> str:
    return str(settings.get("hires_noise_mode") or "CPU")


def _inspire_noise_inputs(settings: dict, seed: int) -> dict:
    # GPU initial noise must not reuse the ancestral/SDE sampler's RNG stream.
    noise_mode = str(settings.get("noise_mode") or "GPU+internal_seed")
    inputs = {"noise_mode": noise_mode}
    if "internal_seed" in noise_mode:
        internal_seed = settings.get("internal_seed")
        inputs["internal_seed"] = (
            int(internal_seed) if internal_seed is not None else (seed + 1) % (2**64)
        )
    return inputs


def _refiner_checkpoint_setting(settings: dict) -> str:
    if not settings.get("refiner_enabled"):
        return ""
    return str(settings.get("refiner_checkpoint") or "").strip()


LORA_BLOCK_WEIGHT_CLASS = "LoraLoaderBlockWeight //Inspire"
INSPIRE_KSAMPLER_CLASS = "KSampler //Inspire"
CONTROLNET_COMFY_PREPROCESSOR_DEFAULTS: dict[str, dict[str, object]] = {
    "canny": {"low_threshold": 100, "high_threshold": 200},
    "lineart": {"guassian_sigma": 6.0, "intensity_threshold": 8},
    "scribble": {},
    "depth": {"ckpt_name": "depth_anything_vitl14.pth"},
    "tile": {"pyrUp_iters": 3},
    "softedge": {"safe": "enable"},
    "normal": {},
    "openpose": {
        "detect_hand": "enable",
        "detect_body": "enable",
        "detect_face": "enable",
        "bbox_detector": "yolox_l.onnx",
        "pose_estimator": "dw-ll_ucoco_384.onnx",
        "scale_stick_for_xinsr_cn": "disable",
    },
}
CONTROLNET_UNION_TYPES: dict[str, str] = {
    "canny": "canny/lineart/anime_lineart/mlsd",
    "lineart": "canny/lineart/anime_lineart/mlsd",
    "scribble": "hed/pidi/scribble/ted",
    "depth": "depth",
    "tile": "tile",
    "softedge": "hed/pidi/scribble/ted",
    "normal": "normal",
    "openpose": "openpose",
}


def _is_union_controlnet_model(model: object) -> bool:
    return "union" in str(model or "").lower()


class ComfyClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = timeout_seconds
        self.client_id = f"asteria-{uuid4().hex}"

    def health(self) -> bool:
        return self.system_stats() is not None

    def system_stats(self) -> dict | None:
        try:
            return self._request_json("GET", "system_stats")
        except RuntimeErrorResponse:
            return None

    def object_info(self) -> dict:
        return self._request_json("GET", "object_info")

    def queue_prompt(self, prompt: dict) -> str:
        response = self._request_json("POST", "prompt", {"prompt": prompt, "client_id": self.client_id})
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str):
            raise RuntimeErrorResponse("ComfyUI did not return prompt_id")
        return prompt_id

    def interrupt(self) -> dict:
        return self._request_json("POST", "interrupt", {})

    def free_memory(self) -> dict:
        return self._request_json(
            "POST",
            "free",
            {"unload_models": True, "free_memory": True},
            allow_empty=True,
        )

    def queue_state(self) -> dict[str, list[str]]:
        payload = self._request_json("GET", "queue")
        return {
            "running": self._queue_prompt_ids(payload.get("queue_running")),
            "pending": self._queue_prompt_ids(payload.get("queue_pending")),
        }

    def delete_queued_prompt(self, prompt_id: str) -> dict:
        return self._request_json("POST", "queue", {"delete": [prompt_id]})

    def wait_for_history(
        self,
        prompt_id: str,
        timeout_seconds: float | None = 600.0,
        interval_seconds: float = 1.0,
        runtime_stop_confirmed=None,
    ) -> dict:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while deadline is None or time.monotonic() < deadline:
            if callable(runtime_stop_confirmed) and runtime_stop_confirmed():
                raise RuntimeCancellationConfirmed(f"ComfyUI prompt stopped: {prompt_id}")
            history = self._request_json("GET", f"history/{prompt_id}")
            if prompt_id in history:
                return history
            if callable(runtime_stop_confirmed) and runtime_stop_confirmed():
                raise RuntimeCancellationConfirmed(f"ComfyUI prompt stopped: {prompt_id}")
            time.sleep(interval_seconds)
        raise RuntimeErrorResponse(f"ComfyUI generation timed out: {prompt_id}")

    def _queue_prompt_ids(self, rows: object) -> list[str]:
        if not isinstance(rows, list):
            return []
        prompt_ids: list[str] = []
        for row in rows:
            if isinstance(row, list) and len(row) > 1 and isinstance(row[1], str):
                prompt_ids.append(row[1])
        return prompt_ids

    def download_output(self, output: dict) -> bytes:
        query = urlencode(
            {
                "filename": output.get("filename", ""),
                "subfolder": output.get("subfolder", ""),
                "type": output.get("type", "output"),
            }
        )
        url = urljoin(self.base_url, f"view?{query}")
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                return response.read()
        except URLError as exc:
            raise RuntimeErrorResponse(f"Failed to download ComfyUI output: {exc}") from exc

    def download_image(self, image: dict) -> bytes:
        return self.download_output(image)

    def upload_file(self, path: Path) -> str:
        if not _path_exists_safely(path) or not _path_is_file_safely(path):
            raise RuntimeErrorResponse(f"Input file not found: {path}")
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with path.open("rb") as file:
                    response = client.post(
                        urljoin(self.base_url, "upload/image"),
                        data={"type": "input", "overwrite": "true"},
                        files={"image": (path.name, file, "application/octet-stream")},
                    )
            response.raise_for_status()
            payload = response.json()
        except (OSError, httpx.HTTPError, ValueError) as exc:
            raise RuntimeErrorResponse(f"Failed to upload input to ComfyUI: {exc}") from exc
        name = payload.get("name")
        if not isinstance(name, str):
            raise RuntimeErrorResponse("ComfyUI upload did not return image name")
        subfolder = payload.get("subfolder")
        return f"{subfolder}/{name}" if subfolder else name

    def upload_image(self, path: Path) -> str:
        if not _path_exists_safely(path) or not _path_is_file_safely(path):
            raise RuntimeErrorResponse(f"Image not found: {path}")
        return self.upload_file(path)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        allow_empty: bool = False,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            urljoin(self.base_url, path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
                if allow_empty and not body.strip():
                    return {}
                return json.loads(body.decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeErrorResponse(f"ComfyUI request failed: {method} /{path}: {exc}") from exc

    @contextmanager
    def preview_listener(
        self,
        preview_dir: Path,
        callback: Callable[[str], None] | None,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        if websocket is None or (callback is None and progress_callback is None):
            yield
            return
        try:
            preview_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            yield
            return
        stop_event = threading.Event()
        socket = None

        def listen() -> None:
            nonlocal socket
            try:
                socket = websocket.create_connection(
                    self._websocket_url(),
                    timeout=1,
                    enable_multithread=True,
                )
            except Exception:
                return
            while not stop_event.is_set():
                try:
                    message = socket.recv()
                except Exception:
                    if stop_event.is_set():
                        return
                    continue
                if isinstance(message, bytes):
                    path = self._write_preview_message(preview_dir, message)
                    if path and callback is not None:
                        callback(str(path))
                elif isinstance(message, str):
                    self._handle_preview_status_message(message, progress_callback)

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        try:
            time.sleep(0.1)
            yield
        finally:
            stop_event.set()
            if socket is not None:
                try:
                    socket.close()
                except Exception:
                    pass
            thread.join(timeout=1)

    def _websocket_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/") + "/ws"
        return urlunparse((scheme, parsed.netloc, path, "", urlencode({"clientId": self.client_id}), ""))

    def _write_preview_message(self, preview_dir: Path, message: bytes) -> Path | None:
        if len(message) < 8:
            return None
        event_type = struct.unpack(">I", message[:4])[0]
        payload = message[4:]
        if event_type == 1:
            if len(payload) < 8:
                return None
            image_type = struct.unpack(">I", payload[:4])[0]
            image_bytes = payload[4:]
            suffix = self._preview_suffix(image_bytes, image_type)
        elif event_type == 4:
            if len(payload) < 4:
                return None
            metadata_length = struct.unpack(">I", payload[:4])[0]
            image_bytes = payload[4 + metadata_length :]
            suffix = self._preview_suffix(image_bytes)
        else:
            return None
        if not image_bytes:
            return None
        path = preview_dir / f"{uuid4().hex}_preview{suffix}"
        try:
            path.write_bytes(image_bytes)
        except OSError:
            return None
        return path

    def _preview_suffix(self, image_bytes: bytes, image_type: int | None = None) -> str:
        if image_bytes.startswith(b"\x89PNG"):
            return ".png"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        if image_type == 2:
            return ".png"
        return ".jpg"

    def _handle_preview_status_message(self, message: str, progress_callback: Callable[[int, int], None] | None) -> None:
        if progress_callback is None:
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if payload.get("type") != "progress":
            return
        data = payload.get("data")
        if not isinstance(data, dict):
            return
        try:
            value = int(data.get("value", 0))
            maximum = int(data.get("max", 0))
        except (TypeError, ValueError):
            return
        if maximum > 0:
            progress_callback(value, maximum)


class ComfyTxt2ImgWorkflowBuilder:
    def __init__(
        self,
        mapping_dir: Path | None = None,
        model_root: Path | None = None,
        external_roots: tuple[ExternalModelRoot, ...] = (),
    ) -> None:
        self.mapping_repository = WorkflowMappingRepository(mapping_dir or AppConfig().mapping_dir)
        self.model_root = model_root
        self.external_roots = external_roots

    def build(self, request: dict) -> dict:
        if isinstance(request.get("_spatial_runtime_context"), dict):
            settings = request.get("settings")
            if not isinstance(settings, dict):
                raise RuntimeErrorResponse("Generation settings are invalid.")
            seed = settings.get("seed", -1)
            if seed is None or int(seed) < 0:
                settings["seed"] = random.randint(0, 2**32 - 1)
        profile = request.get("profile")
        builders = {"sdxl": self._build_sdxl, "flux": self._build_flux,
                    "flux1": self._build_flux, "flux2": self._build_flux2,
                    "anima": self._build_anima}
        if profile not in builders:
            raise RuntimeErrorResponse(f"Unsupported ComfyUI profile: {profile}")
        settings = request.setdefault("settings", {})
        capture = bool(request.get("_image_check_capture"))
        settings.pop('_enhancement_points', None)
        settings.pop('_branch_sample_node', None)
        if capture:
            settings["_image_check_capture"] = True
        try:
            workflow = builders[profile](request)
            word_operation = request.get('_word_maps')
            if word_operation and settings.get('_prompt_trace_stage') != 'hires':
                from app.services.comfy_word_masks import attach_word_capture
                attach_word_capture(workflow, settings['_branch_sample_node'], {
                    **word_operation, 'seed': settings['seed'], 'width': settings['width'],
                    'height': settings['height']}, self._mapping(profile))
            branch_operation = request.get('_generation_branch')
            if branch_operation and settings.get('mode', 'txt2img') == 'txt2img' and settings.get('_prompt_trace_stage') != 'hires':
                from app.services.comfy_generation_branch import attach_branch
                attach_branch(workflow, settings['_branch_sample_node'], branch_operation, self._mapping(profile))
            if request.get('_comparison_sampling') is not None:
                from app.services.comparison_sampling import attach_comparison_sampling
                attach_comparison_sampling(workflow, request['_comparison_sampling'], self._mapping(profile))
            append_stage_outputs(workflow, settings, self._mapping(profile))
            if capture:
                boundary = settings.pop("_image_check_boundary")
                boundary["output_nodes"] = [
                    key for key, node in workflow.items()
                    if node["class_type"] == self._mapping(profile).class_type("save_image", "SaveImage")
                ]
                request["_image_check_boundary"] = boundary
            return workflow
        finally:
            settings.pop("_image_check_capture", None)
            settings.pop("_image_check_boundary", None)
            settings.pop('_branch_sample_node', None)

    @staticmethod
    def _capture_image_check_boundary(settings: dict, sample_node: str, vae_ref: list, mapping) -> None:
        settings['_branch_sample_node'] = sample_node
        if settings.get('hires_enabled') and settings.get('hires_upscaler') != 'Topaz Photo AI':
            mark_stage(settings, 'base', samples=[sample_node, 0], vae=vae_ref)
        if settings.get("_image_check_capture"):
            settings["_image_check_boundary"] = {
                "samples": [sample_node, 0], "vae": list(vae_ref),
                "batch_size": int(settings.get("batch_size", 1)),
                "save_class": mapping.class_type("generation_stage_save", "AsteriaSaveGenerationStage"),
                "load_class": mapping.class_type("generation_stage_load", "AsteriaLoadGenerationStage"),
                "sampler_class": mapping.class_type("generation_stage_sampler", "AsteriaGenerationStageSampler"),
                "noise_class": mapping.class_type("generation_stage_noise", "AsteriaSelectedGenerationNoise"),
                "sample_filter_class": mapping.class_type("generation_stage_sample_filter", "AsteriaSelectedGenerationSampler"),
            }

    def _mapping(self, profile: str) -> WorkflowMapping:
        return self.mapping_repository.get_txt2img(profile)

    def prepare_output_writes(self, workflow: dict, profile: str) -> None:
        # Comfy caches SaveImage's receipt, not the file. Asteria removes its local
        # intermediate after importing it, so every submission needs a fresh write.
        # Change only output leaves; retain cached model/sampling results and seeds.
        save_class = self._mapping(profile).class_type("save_image", "SaveImage")
        operation_id = uuid4().hex
        for node in workflow.values():
            if node.get("class_type") == save_class:
                inputs = node["inputs"]
                inputs["filename_prefix"] = f"{inputs['filename_prefix']}_{operation_id}"

    def _build_sdxl(self, request: dict) -> dict:
        mapping = self._mapping("sdxl")
        settings = request["settings"]
        spatial_runtime_stage = str(request.get("_spatial_runtime_stage") or "base")
        positive_plan = self._request_prompt_syntax_plan(request, "positive")
        negative_plan = self._request_prompt_syntax_plan(request, "negative")
        model_slots = request.get("model_slots", {})
        checkpoint = model_slots.get("checkpoint")
        if not checkpoint:
            raise RuntimeErrorResponse("Checkpoint is required before sending a real ComfyUI generation request.")

        checkpoint_node = mapping.node("checkpoint_loader", "1")
        positive_node = mapping.node("clip_positive", "2")
        negative_node = mapping.node("clip_negative", "3")
        latent_node = mapping.node("latent", "4")
        sampler_node = mapping.node("sampler", "5")
        vae_loader_node = mapping.node("vae_loader", "6")
        decode_preferred_node = mapping.node("vae_decode", "7")
        save_preferred_node = mapping.node("save_image", "9")
        workflow: dict[str, dict] = {
            checkpoint_node: {
                "class_type": mapping.class_type("checkpoint_loader", "CheckpointLoaderSimple"),
                "inputs": {"ckpt_name": self._model_name(checkpoint, "checkpoints")},
            }
        }

        model_ref: list[str | int] = [checkpoint_node, 0]
        clip_ref: list[str | int] = [checkpoint_node, 1]
        vae_ref: list[str | int] = [checkpoint_node, 2]

        merge_context = request.get("_merge_runtime_context")
        if merge_context is not None:
            if not isinstance(merge_context, dict):
                raise RuntimeErrorResponse("Invalid internal merge runtime context.")
            source_b = merge_context.get("source_b")
            source_c = merge_context.get("source_c")
            if not isinstance(source_b, dict) or not source_b.get("checkpoint"):
                raise RuntimeErrorResponse("Internal merge runtime source B checkpoint is required.")
            source_b_node = mapping.node("merge_source_b", "900")
            source_c_node = mapping.node("merge_source_c", "901")
            merge_node = mapping.node("merge_runtime", "902")
            workflow[source_b_node] = {
                "class_type": mapping.class_type("checkpoint_loader", "CheckpointLoaderSimple"),
                "inputs": {"ckpt_name": self._model_name(source_b["checkpoint"], "checkpoints")},
            }
            merge_inputs: dict[str, object] = {
                "model_a": [checkpoint_node, 0],
                "model_b": [source_b_node, 0],
                "clip_a": [checkpoint_node, 1],
                "clip_b": [source_b_node, 1],
                "vae_a": [checkpoint_node, 2],
                "vae_b": [source_b_node, 2],
                "recipe_payload": str(merge_context.get("recipe_payload") or ""),
                "contract_version": str(merge_context.get("contract_version") or ""),
            }
            if isinstance(source_c, dict) and source_c.get("checkpoint"):
                workflow[source_c_node] = {
                    "class_type": mapping.class_type("checkpoint_loader", "CheckpointLoaderSimple"),
                    "inputs": {"ckpt_name": self._model_name(source_c["checkpoint"], "checkpoints")},
                }
                merge_inputs.update(
                    model_c=[source_c_node, 0],
                    clip_c=[source_c_node, 1],
                    vae_c=[source_c_node, 2],
                )
            workflow[merge_node] = {
                "class_type": mapping.class_type("merge_runtime", "AsteriaSDXLMergeRuntime"),
                "inputs": merge_inputs,
            }
            model_ref = [merge_node, 0]
            clip_ref = [merge_node, 1]
            vae_ref = [merge_node, 2]

        next_id = 8
        for lora in request.get("loras", []):
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = self._lora_loader_node(lora, model_ref, clip_ref, mapping, settings)
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]

        clip_skip = int(settings.get("clip_skip") or 1)
        if clip_skip > 1:
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = {
                "class_type": mapping.class_type("clip_skip", "CLIPSetLastLayer"),
                "inputs": {
                    "clip": clip_ref,
                    "stop_at_clip_layer": -clip_skip,
                },
            }
            clip_ref = [node_id, 0]

        vae = model_slots.get("vae")
        if vae:
            workflow[vae_loader_node] = {
                "class_type": mapping.class_type("vae_loader", "VAELoader"),
                "inputs": {"vae_name": self._model_name(vae, "vae")},
            }
            vae_ref = [vae_loader_node, 0]

        if settings.get("mode") == "remaster":
            latent_ref, next_id = self._append_remaster_latent(
                workflow,
                next_id,
                uploaded_image=str(settings.get("init_image_path") or ""),
                vae_ref=vae_ref,
                settings=settings,
                mapping=mapping,
            )
        elif settings.get("mode") in {"img2img", "inpaint"}:
            init_image = settings.get("init_image_path") or settings.get("init_image")
            if not init_image:
                raise RuntimeErrorResponse("init_image_path is required for img2img/inpaint.")
            image_node = str(next_id)
            encode_node = str(next_id + 1)
            next_id += 2
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": init_image},
            }
            # "original" retains the source under the mask; VAEEncodeForInpaint erases it.
            workflow[encode_node] = {
                "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                "inputs": {"pixels": [image_node, 0], "vae": vae_ref},
            }
            latent_ref: list[str | int] = [encode_node, 0]
            if settings.get("mode") == "inpaint":
                mask_image = settings.get("mask_image_path")
                if not mask_image:
                    raise RuntimeErrorResponse("mask_image_path is required for inpaint.")
                mask_node = str(next_id)
                next_id += 1
                workflow[mask_node] = {
                    "class_type": mapping.class_type("load_image", "LoadImage"),
                    "inputs": {"image": mask_image},
                }
                masked_latent_node = str(next_id)
                next_id += 1
                workflow[masked_latent_node] = {
                    "class_type": mapping.class_type("set_latent_noise_mask", "SetLatentNoiseMask"),
                    "inputs": {
                        "samples": [encode_node, 0],
                        "mask": [mask_node, 1],
                    },
                }
                latent_ref = [masked_latent_node, 0]
        else:
            workflow[latent_node] = {
                "class_type": mapping.class_type("latent", "EmptyLatentImage"),
                "inputs": {
                    "width": settings["width"],
                    "height": settings["height"],
                    "batch_size": settings.get("batch_size", 1),
                },
            }
            latent_ref = [latent_node, 0]

        if self._identity_prompt_plan(positive_plan) and self._identity_prompt_plan(negative_plan):
            workflow[positive_node] = self._sdxl_conditioning_node(
                self._comfy_prompt_text(request["expanded_prompt"], request.get("embeddings", [])),
                clip_ref,
                mapping,
                int(settings["width"]),
                int(settings["height"]),
            )
            workflow[negative_node] = self._sdxl_conditioning_node(
                self._comfy_prompt_text(request["negative_prompt"], request.get("embeddings", [])),
                clip_ref,
                mapping,
                int(settings["width"]),
                int(settings["height"]),
            )
            positive_ref: list[str | int] = [positive_node, 0]
            negative_ref: list[str | int] = [negative_node, 0]
        else:
            positive_ref, next_id = self._append_sdxl_prompt_conditioning(
                workflow=workflow,
                next_id=next_id,
                preferred_node=positive_node,
                plan=positive_plan,
                fallback_text=request["expanded_prompt"],
                clip_ref=clip_ref,
                model_ref=model_ref,
                mapping=mapping,
                width=int(settings["width"]),
                height=int(settings["height"]),
                stage_settings=settings,
                embeddings=request.get("embeddings", []),
            )
            negative_ref, next_id = self._append_sdxl_prompt_conditioning(
                workflow=workflow,
                next_id=next_id,
                preferred_node=negative_node,
                plan=negative_plan,
                fallback_text=request["negative_prompt"],
                clip_ref=clip_ref,
                model_ref=model_ref,
                mapping=mapping,
                width=int(settings["width"]),
                height=int(settings["height"]),
                stage_settings=settings,
                embeddings=request.get("embeddings", []),
            )
        spatial_global = self._append_sdxl_spatial_global(
            request=request,
            stage=spatial_runtime_stage,
            workflow=workflow,
            next_id=next_id,
            mapping=mapping,
            clip_ref=clip_ref,
            model_ref=model_ref,
            width=int(settings["width"]),
            height=int(settings["height"]),
            stage_settings=settings,
        )
        if spatial_global is not None:
            positive_ref, next_id = spatial_global
        for unit in settings.get("controlnet_units", []):
            if not unit.get("enabled") or not unit.get("model") or not unit.get("image_path"):
                continue
            control_mode = self._controlnet_mode(unit)
            effective_strength = self._controlnet_strength(unit)
            unit["control_mode_normalized"] = control_mode
            unit["effective_strength"] = effective_strength
            unit.pop("control_mode_base_multiplier", None)
            unit.pop("control_mode_uncond_multiplier", None)
            if control_mode == "balanced":
                unit["control_mode_mapping"] = "balanced_native_strength"
                unit["control_mode_runtime"] = "balanced_standard_controlnet_apply"
            else:
                unit["control_mode_mapping"] = (
                    "prompt_priority_soft_weights"
                    if control_mode == "prompt_priority"
                    else "controlnet_priority_soft_weights"
                )
                unit["control_mode_runtime"] = "advanced_controlnet_soft_weights"
                unit["control_mode_base_multiplier"] = 0.825
                unit["control_mode_uncond_multiplier"] = 1.0 if control_mode == "prompt_priority" else 0.0
            image_node = str(next_id)
            next_id += 1
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": unit["image_path"]},
            }
            control_image_ref: list[str | int] = [image_node, 0]
            preprocessor_node = self._controlnet_preprocessor_workflow_node(unit, control_image_ref, settings)
            if preprocessor_node:
                preprocessor_node_id = str(next_id)
                next_id += 1
                workflow[preprocessor_node_id] = preprocessor_node
                control_image_ref = [preprocessor_node_id, 0]
                unit["preprocessor_workflow_node_id"] = preprocessor_node_id
            control_loader_node = str(next_id)
            next_id += 1
            workflow[control_loader_node] = {
                "class_type": mapping.class_type("controlnet_loader", "ControlNetLoader"),
                "inputs": {"control_net_name": self._model_name(unit["model"], "controlnet")},
            }
            control_net_ref: list[str | int] = [control_loader_node, 0]
            union_type = self._controlnet_union_type(unit)
            if union_type:
                control_union_node = str(next_id)
                next_id += 1
                workflow[control_union_node] = {
                    "class_type": mapping.class_type("controlnet_union_type", "SetUnionControlNetType"),
                    "inputs": {"control_net": control_net_ref, "type": union_type},
                }
                control_net_ref = [control_union_node, 0]
                unit["controlnet_union_type"] = union_type
                unit["controlnet_union_type_node_id"] = control_union_node
            apply_inputs = {
                "positive": positive_ref,
                "negative": negative_ref,
                "control_net": control_net_ref,
                "image": control_image_ref,
                "strength": effective_strength,
                "start_percent": unit.get("guidance_start", 0.0),
                "end_percent": unit.get("guidance_end", 1.0),
            }
            if control_mode == "balanced":
                apply_class = mapping.class_type("controlnet_apply", "ControlNetApplyAdvanced")
                apply_inputs["vae"] = vae_ref
            else:
                soft_weights_node = str(next_id)
                next_id += 1
                workflow[soft_weights_node] = {
                    "class_type": mapping.class_type("controlnet_soft_weights", ""),
                    "inputs": {
                        "base_multiplier": unit["control_mode_base_multiplier"],
                        "uncond_multiplier": unit["control_mode_uncond_multiplier"],
                    },
                }
                apply_class = mapping.class_type("controlnet_advanced_apply", "")
                apply_inputs["weights_override"] = [soft_weights_node, 0]
                apply_inputs["vae_optional"] = vae_ref
            control_apply_node = str(next_id)
            next_id += 1
            workflow[control_apply_node] = {
                "class_type": apply_class,
                "inputs": apply_inputs,
                "_meta": {"control_mode": control_mode},
            }
            positive_ref = [control_apply_node, 0]
            negative_ref = [control_apply_node, 1]
        base_unpatched_model_ref = model_ref
        base_execution_state_ref: list[str | int] | None = None
        spatial_base = self._append_sdxl_spatial_stage(
            request=request,
            stage=spatial_runtime_stage,
            workflow=workflow,
            next_id=next_id,
            mapping=mapping,
            model_ref=base_unpatched_model_ref,
            clip_ref=clip_ref,
            global_ref=positive_ref,
            negative_ref=negative_ref,
            width=int(settings["width"]),
            height=int(settings["height"]),
            stage_settings=settings,
        )
        if spatial_base is not None:
            (
                model_ref,
                positive_ref,
                negative_ref,
                base_execution_state_ref,
                next_id,
            ) = spatial_base
        refiner_checkpoint = _refiner_checkpoint_setting(settings)
        should_refine_base_stage = bool(refiner_checkpoint and not self._uses_comfy_hires(settings))
        sample_node = sampler_node
        seed_resize_noise_ref: list[str | int] | None = None
        seed_resize_seed: int | None = None
        if settings.get("mode") == "txt2img" and seed_resize_active(settings):
            seed_resize_noise_ref, seed_resize_seed, next_id = self._append_seed_resize_noise(
                workflow,
                next_id,
                latent_ref,
                settings,
                mapping,
            )
        if seed_resize_noise_ref is not None and seed_resize_seed is not None:
            workflow[sample_node] = self._ksampler_advanced_inspire_node(
                settings,
                model_ref,
                positive_ref,
                negative_ref,
                latent_ref,
                seed_resize_noise_ref,
                mapping,
                seed=seed_resize_seed,
                start_at_step=0,
                end_at_step=(
                    self._refiner_switch_step(settings, int(settings["steps"]))
                    if should_refine_base_stage
                    else int(settings["steps"])
                ),
                return_with_leftover_noise=should_refine_base_stage,
            )
        elif should_refine_base_stage:
            workflow[sample_node] = self._ksampler_advanced_node(
                settings,
                model_ref,
                positive_ref,
                negative_ref,
                latent_ref,
                mapping,
                start_at_step=0,
                end_at_step=self._refiner_switch_step(settings, int(settings["steps"])),
                add_noise=True,
                return_with_leftover_noise=True,
            )
        else:
            workflow[sample_node] = self._ksampler_node(
                settings,
                model_ref,
                positive_ref,
                negative_ref,
                latent_ref,
                mapping,
                use_inspire=bool(settings.get("use_inspire_ksampler", True)),
            )
        if base_execution_state_ref is not None:
            next_id = append_spatial_execution_receipt(
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                execution_state_ref=base_execution_state_ref,
                samples_ref=[sample_node, 0],
            )
        self._capture_image_check_boundary(settings, sample_node, vae_ref, mapping)
        current_width = int(settings["width"])
        current_height = int(settings["height"])

        if self._uses_comfy_hires(settings):
            if settings.get("hires_keep_first_pass"):
                first_pass_decode_node = str(next_id)
                first_pass_save_node = str(next_id + 1)
                next_id += 2
                workflow[first_pass_decode_node] = {
                    "class_type": mapping.class_type("vae_decode", "VAEDecode"),
                    "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
                }
                workflow[first_pass_save_node] = {
                    "class_type": mapping.class_type("save_image", "SaveImage"),
                    "inputs": {"images": [first_pass_decode_node, 0], "filename_prefix": "Asteria_first_pass"},
                }
            if self._hires_is_latent(settings):
                upscale_node = str(next_id)
                next_id += 1
                resize_dimensions = self._hires_resize_dimensions(settings)
                if resize_dimensions is not None:
                    workflow[upscale_node] = {
                        "class_type": mapping.class_type("latent_upscale", "LatentUpscale"),
                        "inputs": {
                            "samples": [sample_node, 0],
                            "upscale_method": self._latent_upscale_method(settings),
                            "width": resize_dimensions[0],
                            "height": resize_dimensions[1],
                            "crop": "disabled",
                        },
                    }
                else:
                    workflow[upscale_node] = {
                        "class_type": mapping.class_type("latent_upscale_by", "LatentUpscaleBy"),
                        "inputs": {
                            "samples": [sample_node, 0],
                            "upscale_method": self._latent_upscale_method(settings),
                            "scale_by": settings.get("hires_scale", 2.0),
                        },
                    }
                hires_latent_ref = [upscale_node, 0]
            else:
                first_decode_node = str(next_id)
                upscale_start_node = next_id + 1
                workflow[first_decode_node] = {
                    "class_type": mapping.class_type("vae_decode", "VAEDecode"),
                    "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
                }
                upscale_ref, next_id = self._append_image_hires_upscale(
                    workflow,
                    upscale_start_node,
                    [first_decode_node, 0],
                    settings,
                    mapping,
                )
                encode_node = str(next_id)
                next_id += 1
                workflow[encode_node] = {
                    "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                    "inputs": {"pixels": upscale_ref, "vae": vae_ref},
                }
                hires_latent_ref = [encode_node, 0]
            hires_model_ref = base_unpatched_model_ref
            hires_clip_ref = clip_ref
            hires_vae_ref = vae_ref
            hires_checkpoint = _hires_checkpoint_setting(settings)
            if hires_checkpoint:
                hires_checkpoint_node = str(next_id)
                next_id += 1
                workflow[hires_checkpoint_node] = {
                    "class_type": mapping.class_type("checkpoint_loader", "CheckpointLoaderSimple"),
                    "inputs": {"ckpt_name": self._model_name(hires_checkpoint, "checkpoints")},
                }
                hires_model_ref = [hires_checkpoint_node, 0]
                hires_clip_ref = [hires_checkpoint_node, 1]
                if not vae:
                    hires_vae_ref = [hires_checkpoint_node, 2]
                settings["hires_checkpoint_runtime"] = "second_pass_checkpoint_loader"
            hires_width, hires_height = self._hires_target_dimensions(settings)
            positive_node = str(next_id)
            negative_node = str(next_id + 1)
            next_id += 2
            hires_prompt = settings.get("hires_prompt") or request["expanded_prompt"]
            hires_negative_prompt = settings.get("hires_negative_prompt") or request["negative_prompt"]
            hires_stage_settings = settings | {
                "_prompt_trace_stage": "hires",
                "steps": settings.get("hires_steps") or settings["steps"],
                "cfg_scale": settings.get("hires_cfg_scale") or settings["cfg_scale"],
                "sampler": self._hires_sampler(settings),
                "scheduler": self._hires_scheduler(settings),
                "denoise": settings.get("hires_denoising_strength", 0.7),
                "noise_mode": _hires_noise_mode(settings),
            }
            hires_positive_plan = (
                self._request_prompt_syntax_plan(request, "hires_positive")
                if settings.get("hires_prompt")
                else positive_plan
            )
            hires_negative_plan = (
                self._request_prompt_syntax_plan(request, "hires_negative")
                if settings.get("hires_negative_prompt")
                else negative_plan
            )
            if self._identity_prompt_plan(hires_positive_plan) and self._identity_prompt_plan(hires_negative_plan):
                workflow[positive_node] = self._sdxl_conditioning_node(
                    self._comfy_prompt_text(hires_prompt, request.get("embeddings", [])),
                    hires_clip_ref,
                    mapping,
                    hires_width,
                    hires_height,
                )
                workflow[negative_node] = self._sdxl_conditioning_node(
                    self._comfy_prompt_text(hires_negative_prompt, request.get("embeddings", [])),
                    hires_clip_ref,
                    mapping,
                    hires_width,
                    hires_height,
                )
                positive_ref = [positive_node, 0]
                negative_ref = [negative_node, 0]
            else:
                positive_ref, next_id = self._append_sdxl_prompt_conditioning(
                    workflow=workflow,
                    next_id=next_id,
                    preferred_node=positive_node,
                    plan=hires_positive_plan,
                    fallback_text=hires_prompt,
                    clip_ref=hires_clip_ref,
                    model_ref=hires_model_ref,
                    mapping=mapping,
                    width=hires_width,
                    height=hires_height,
                    stage_settings=hires_stage_settings,
                    embeddings=request.get("embeddings", []),
                )
                negative_ref, next_id = self._append_sdxl_prompt_conditioning(
                    workflow=workflow,
                    next_id=next_id,
                    preferred_node=negative_node,
                    plan=hires_negative_plan,
                    fallback_text=hires_negative_prompt,
                    clip_ref=hires_clip_ref,
                    model_ref=hires_model_ref,
                    mapping=mapping,
                    width=hires_width,
                    height=hires_height,
                    stage_settings=hires_stage_settings,
                    embeddings=request.get("embeddings", []),
                )
            spatial_hires_global = self._append_sdxl_spatial_global(
                request=request,
                stage="hires",
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                clip_ref=hires_clip_ref,
                model_ref=hires_model_ref,
                width=hires_width,
                height=hires_height,
                stage_settings=hires_stage_settings,
            )
            if spatial_hires_global is not None:
                positive_ref, next_id = spatial_hires_global
            hires_execution_state_ref: list[str | int] | None = None
            spatial_hires = self._append_sdxl_spatial_stage(
                request=request,
                stage="hires",
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                model_ref=hires_model_ref,
                clip_ref=hires_clip_ref,
                global_ref=positive_ref,
                negative_ref=negative_ref,
                width=hires_width,
                height=hires_height,
                stage_settings=hires_stage_settings,
            )
            if spatial_hires is not None:
                (
                    hires_model_ref,
                    positive_ref,
                    negative_ref,
                    hires_execution_state_ref,
                    next_id,
                ) = spatial_hires
            sample_node = str(next_id)
            next_id += 1
            use_inspire_hires = bool(settings.get("use_inspire_ksampler", True))
            if refiner_checkpoint:
                workflow[sample_node] = self._ksampler_advanced_node(
                    hires_stage_settings,
                    hires_model_ref,
                    positive_ref,
                    negative_ref,
                    hires_latent_ref,
                    mapping,
                    start_at_step=0,
                    end_at_step=self._refiner_switch_step(settings, int(hires_stage_settings["steps"])),
                    add_noise=True,
                    return_with_leftover_noise=True,
                )
            else:
                workflow[sample_node] = self._ksampler_node(
                    hires_stage_settings,
                    hires_model_ref,
                    positive_ref,
                    negative_ref,
                    hires_latent_ref,
                    mapping,
                    use_inspire=use_inspire_hires,
                )
            if hires_execution_state_ref is not None:
                next_id = append_spatial_execution_receipt(
                    workflow=workflow,
                    next_id=next_id,
                    mapping=mapping,
                    execution_state_ref=hires_execution_state_ref,
                    samples_ref=[sample_node, 0],
                )
            current_width, current_height = hires_width, hires_height
            model_ref = hires_model_ref
            clip_ref = hires_clip_ref
            vae_ref = hires_vae_ref

        if refiner_checkpoint:
            refiner_checkpoint_node = str(next_id)
            refiner_positive_node = str(next_id + 1)
            refiner_negative_node = str(next_id + 2)
            refiner_sample_node = str(next_id + 3)
            next_id += 4
            workflow[refiner_checkpoint_node] = {
                "class_type": mapping.class_type("checkpoint_loader", "CheckpointLoaderSimple"),
                "inputs": {"ckpt_name": self._model_name(refiner_checkpoint, "checkpoints")},
            }
            refiner_prompt = (
                settings.get("hires_prompt")
                if self._uses_comfy_hires(settings) and settings.get("hires_prompt")
                else request["expanded_prompt"]
            )
            refiner_negative_prompt = (
                settings.get("hires_negative_prompt")
                if self._uses_comfy_hires(settings) and settings.get("hires_negative_prompt")
                else request["negative_prompt"]
            )
            workflow[refiner_positive_node] = self._sdxl_refiner_conditioning_node(
                self._comfy_prompt_text(refiner_prompt, request.get("embeddings", [])),
                [refiner_checkpoint_node, 1],
                mapping,
                current_width,
                current_height,
            )
            workflow[refiner_negative_node] = self._sdxl_refiner_conditioning_node(
                self._comfy_prompt_text(refiner_negative_prompt, request.get("embeddings", [])),
                [refiner_checkpoint_node, 1],
                mapping,
                current_width,
                current_height,
            )
            uses_hires_refiner_stage = self._uses_comfy_hires(settings)
            refiner_steps = int(
                (settings.get("hires_steps") or settings["steps"])
                if uses_hires_refiner_stage
                else settings["steps"]
            )
            refiner_cfg_scale = settings["cfg_scale"]
            refiner_sampler = settings["sampler"]
            refiner_scheduler = settings["scheduler"]
            if uses_hires_refiner_stage:
                refiner_cfg_scale = settings.get("hires_cfg_scale") or settings["cfg_scale"]
                refiner_sampler = self._hires_sampler(settings)
                refiner_scheduler = self._hires_scheduler(settings)
            refiner_stage_settings = settings | {
                "_prompt_trace_stage": "refiner",
                "steps": refiner_steps,
                "cfg_scale": refiner_cfg_scale,
                "sampler": refiner_sampler,
                "scheduler": refiner_scheduler,
            }
            refiner_model_ref: list[str | int] = [refiner_checkpoint_node, 0]
            refiner_positive_ref: list[str | int] = [refiner_positive_node, 0]
            refiner_negative_ref: list[str | int] = [refiner_negative_node, 0]
            spatial_refiner_global = self._append_sdxl_spatial_global(
                request=request,
                stage="refiner",
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                clip_ref=[refiner_checkpoint_node, 1],
                model_ref=refiner_model_ref,
                width=current_width,
                height=current_height,
                stage_settings=refiner_stage_settings,
                refiner=True,
            )
            if spatial_refiner_global is not None:
                refiner_positive_ref, next_id = spatial_refiner_global
            spatial_refiner = self._append_sdxl_spatial_stage(
                request=request,
                stage="refiner",
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                model_ref=refiner_model_ref,
                clip_ref=[refiner_checkpoint_node, 1],
                global_ref=refiner_positive_ref,
                negative_ref=refiner_negative_ref,
                width=current_width,
                height=current_height,
                stage_settings=refiner_stage_settings,
                refiner=True,
            )
            refiner_execution_state_ref: list[str | int] | None = None
            if spatial_refiner is not None:
                (
                    refiner_model_ref,
                    refiner_positive_ref,
                    refiner_negative_ref,
                    refiner_execution_state_ref,
                    next_id,
                ) = spatial_refiner
            workflow[refiner_sample_node] = self._ksampler_advanced_node(
                refiner_stage_settings,
                refiner_model_ref,
                refiner_positive_ref,
                refiner_negative_ref,
                [sample_node, 0],
                mapping,
                start_at_step=self._refiner_switch_step(settings, refiner_steps),
                end_at_step=refiner_steps,
                add_noise=False,
                return_with_leftover_noise=False,
            )
            if refiner_execution_state_ref is not None:
                next_id = append_spatial_execution_receipt(
                    workflow=workflow,
                    next_id=next_id,
                    mapping=mapping,
                    execution_state_ref=refiner_execution_state_ref,
                    samples_ref=[refiner_sample_node, 0],
                )
            sample_node = refiner_sample_node
            model_ref = refiner_model_ref
            clip_ref = [refiner_checkpoint_node, 1]
            settings["refiner_runtime"] = "sdxl_advanced_sampler"

        decode_node = decode_preferred_node if decode_preferred_node not in workflow else str(next_id)
        if decode_node == str(next_id):
            next_id += 1

        workflow[decode_node] = {
            "class_type": mapping.class_type("vae_decode", "VAEDecode"),
            "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
        }
        image_ref: list[str | int] = [decode_node, 0]
        mark_stage(settings, decoded_stage_kind(settings), images=image_ref)
        if settings.get("restore_faces"):
            image_ref, next_id = self._append_face_detailer_node(
                workflow,
                next_id,
                image_ref,
                model_ref,
                clip_ref,
                vae_ref,
                positive_ref,
                negative_ref,
                mapping,
                settings,
                {
                    "detector": "face_yolo",
                    "confidence": 0.3,
                    "mask_blur": 4,
                    "denoising_strength": 0.35,
                    "inpaint_only_masked": True,
                    "guide_size": 512,
                    "max_size": 1024,
                    "bbox_dilation": 10,
                    "bbox_crop_factor": 3.0,
                },
            )
            settings["restore_faces_runtime"] = "comfy_face_detailer"
            settings["face_restoration_model"] = "FaceDetailer"
        for unit in settings.get("detailer_units", []):
            if not unit.get("enabled"):
                continue
            image_ref, next_id = self._append_face_detailer_node(
                workflow,
                next_id,
                image_ref,
                model_ref,
                clip_ref,
                vae_ref,
                positive_ref,
                negative_ref,
                mapping,
                settings,
                unit,
                request.get("embeddings", []),
            )
        save_node = save_preferred_node if save_preferred_node not in workflow else str(next_id)
        if save_node == str(next_id):
            next_id += 1
        workflow[save_node] = {
            "class_type": mapping.class_type("save_image", "SaveImage"),
            "inputs": {"images": image_ref, "filename_prefix": "Asteria"},
        }
        settings["primary_output_node_ids"] = [save_node]
        return workflow

    def _build_flux(self, request: dict) -> dict:
        mapping = self._mapping("flux")
        settings = request["settings"]
        model_slots = request.get("model_slots", {})
        diffusion_model = model_slots.get("diffusion_model")
        clip_l = model_slots.get("clip_l")
        t5xxl = model_slots.get("t5xxl")
        vae = model_slots.get("vae")
        missing = [
            name
            for name, value in {
                "diffusion_model": diffusion_model,
                "clip_l": clip_l,
                "t5xxl": t5xxl,
                "vae": vae,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeErrorResponse(f"Flux.1 requires model slots before generation: {', '.join(missing)}")

        unet_node = mapping.node("unet_loader", "1")
        clip_node = mapping.node("dual_clip_loader", "2")
        vae_node = mapping.node("vae_loader", "3")
        model_sampling_node = mapping.node("model_sampling", "4")
        conditioning_node = mapping.node("conditioning", "5")
        latent_node = mapping.node("latent", "6")
        model_ref: list[str | int] = [unet_node, 0]
        clip_ref: list[str | int] = [clip_node, 0]
        workflow: dict[str, dict] = {
            unet_node: {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(diffusion_model, "diffusion_models"),
                    "weight_dtype": "fp8_e4m3fn",
                },
            },
            clip_node: {
                "class_type": mapping.class_type("dual_clip_loader", "DualCLIPLoader"),
                "inputs": {
                    "clip_name1": self._model_name(clip_l, "text_encoders"),
                    "clip_name2": self._model_name(t5xxl, "text_encoders"),
                    "type": "flux",
                },
            },
            vae_node: {
                "class_type": mapping.class_type("vae_loader", "VAELoader"),
                "inputs": {"vae_name": self._model_name(vae, "vae")},
            },
            model_sampling_node: {
                "class_type": mapping.class_type("model_sampling", "ModelSamplingFlux"),
                "inputs": {
                    "model": model_ref,
                    "max_shift": 1.15,
                    "base_shift": 0.5,
                    "width": settings["width"],
                    "height": settings["height"],
                },
            },
            conditioning_node: {
                "class_type": mapping.class_type("conditioning", "CLIPTextEncodeFlux"),
                "inputs": {
                    "clip": clip_ref,
                    "clip_l": request["expanded_prompt"],
                    "t5xxl": request["expanded_prompt"],
                    "guidance": settings.get("distilled_cfg_scale", 3.5),
                },
            },
        }
        if settings.get("mode") == "remaster":
            latent_ref, _ = self._append_remaster_latent(
                workflow,
                max(int(node_id) for node_id in workflow) + 1,
                uploaded_image=str(settings.get("init_image_path") or ""),
                vae_ref=[vae_node, 0],
                settings=settings,
                mapping=mapping,
            )
            workflow[model_sampling_node]["inputs"]["width"] = settings["width"]
            workflow[model_sampling_node]["inputs"]["height"] = settings["height"]
        elif settings.get("mode") == "inpaint":
            init_image = settings.get("init_image_path")
            mask_image = settings.get("mask_image_path")
            if not init_image:
                raise RuntimeErrorResponse("init_image_path is required for Flux.1 inpaint.")
            if not mask_image:
                raise RuntimeErrorResponse("mask_image_path is required for Flux.1 inpaint.")
            image_node = str(int(latent_node) + 1)
            mask_node = str(int(image_node) + 1)
            encode_node = str(int(mask_node) + 1)
            noise_mask_node = str(int(encode_node) + 1)
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": init_image},
            }
            workflow[mask_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": mask_image},
            }
            workflow[encode_node] = {
                "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                "inputs": {
                    "pixels": [image_node, 0],
                    "vae": [vae_node, 0],
                },
            }
            workflow[noise_mask_node] = {
                "class_type": mapping.class_type("set_latent_noise_mask", "SetLatentNoiseMask"),
                "inputs": {"samples": [encode_node, 0], "mask": [mask_node, 1]},
            }
            settings["inpaint_runtime"] = "flux1_vae_noise_mask_img2img"
            latent_ref: list[str | int] = [noise_mask_node, 0]
        elif settings.get("mode") == "img2img":
            init_image = settings.get("init_image_path")
            if not init_image:
                raise RuntimeErrorResponse("init_image_path is required for Flux.1 img2img.")
            image_node = str(int(latent_node) + 1)
            encode_node = str(int(image_node) + 1)
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": init_image},
            }
            workflow[encode_node] = {
                "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                "inputs": {"pixels": [image_node, 0], "vae": [vae_node, 0]},
            }
            latent_ref: list[str | int] = [encode_node, 0]
        else:
            workflow[latent_node] = {
                "class_type": mapping.class_type("latent", "EmptySD3LatentImage"),
                "inputs": {
                    "width": settings["width"],
                    "height": settings["height"],
                    "batch_size": settings.get("batch_size", 1),
                },
            }
            latent_ref = [latent_node, 0]
        next_id = max(int(node_id) for node_id in workflow) + 1
        hires_model_ref: list[str | int] | None = None
        hires_checkpoint = _hires_checkpoint_setting(settings) if self._uses_comfy_hires(settings) else None
        if hires_checkpoint:
            hires_model_node = str(next_id)
            next_id += 1
            workflow[hires_model_node] = {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(hires_checkpoint, "diffusion_models"),
                    "weight_dtype": "fp8_e4m3fn",
                },
            }
            hires_model_ref = [hires_model_node, 0]
            settings["hires_checkpoint_runtime"] = "second_pass_diffusion_model_loader"
            settings["hires_checkpoint_kind"] = "diffusion_model"
        for lora in request.get("loras", []):
            previous_clip_ref = clip_ref
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = self._lora_loader_node(lora, model_ref, clip_ref, mapping, settings)
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]
            if hires_model_ref is not None:
                hires_node_id = str(next_id)
                next_id += 1
                workflow[hires_node_id] = self._lora_loader_node(
                    lora,
                    hires_model_ref,
                    previous_clip_ref,
                    mapping,
                    settings,
                )
                hires_model_ref = [hires_node_id, 0]
        workflow[model_sampling_node]["inputs"]["model"] = model_ref
        workflow[conditioning_node]["inputs"]["clip"] = clip_ref
        hires_sampling_ref: list[str | int] | None = None
        if hires_model_ref is not None:
            hires_sampling_node = str(next_id)
            hires_width, hires_height = self._hires_target_dimensions(settings)
            workflow[hires_sampling_node] = {
                "class_type": mapping.class_type("model_sampling", "ModelSamplingFlux"),
                "inputs": {
                    "model": hires_model_ref,
                    "max_shift": 1.15,
                    "base_shift": 0.5,
                    "width": hires_width,
                    "height": hires_height,
                },
            }
            hires_sampling_ref = [hires_sampling_node, 0]
        return self._build_custom_sampler_tail(
            workflow,
            settings,
            model_ref=[model_sampling_node, 0],
            conditioning_ref=[conditioning_node, 0],
            latent_ref=latent_ref,
            vae_ref=[vae_node, 0],
            mapping=mapping,
            hires_model_ref=hires_sampling_ref,
        )

    def _build_flux2(self, request: dict) -> dict:
        mapping = self._mapping("flux2")
        settings = request["settings"]
        model_slots = request.get("model_slots", {})
        diffusion_model = model_slots.get("diffusion_model")
        text_encoder = model_slots.get("text_encoder")
        vae = model_slots.get("vae")
        missing = [
            name
            for name, value in {
                "diffusion_model": diffusion_model,
                "text_encoder": text_encoder,
                "vae": vae,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeErrorResponse(f"Flux.2 requires model slots before generation: {', '.join(missing)}")

        unet_node = mapping.node("unet_loader", "1")
        clip_node = mapping.node("clip_loader", "2")
        vae_node = mapping.node("vae_loader", "3")
        conditioning_node = mapping.node("conditioning", "4")
        guidance_node = mapping.node("guidance", "5")
        latent_node = mapping.node("latent", "6")
        model_ref: list[str | int] = [unet_node, 0]
        clip_ref: list[str | int] = [clip_node, 0]
        workflow: dict[str, dict] = {
            unet_node: {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(diffusion_model, "diffusion_models"),
                    "weight_dtype": "default",
                },
            },
            clip_node: {
                "class_type": mapping.class_type("clip_loader", "CLIPLoader"),
                "inputs": {
                    "clip_name": self._model_name(text_encoder, "text_encoders"),
                    "type": "flux2",
                },
            },
            vae_node: {
                "class_type": mapping.class_type("vae_loader", "VAELoader"),
                "inputs": {"vae_name": self._model_name(vae, "vae")},
            },
            conditioning_node: {
                "class_type": mapping.class_type("conditioning", "CLIPTextEncode"),
                "inputs": {
                    "clip": clip_ref,
                    "text": request["expanded_prompt"],
                },
            },
            guidance_node: {
                "class_type": mapping.class_type("guidance", "FluxGuidance"),
                "inputs": {
                    "conditioning": [conditioning_node, 0],
                    "guidance": settings.get("cfg_scale", 3.5),
                },
            },
        }
        if settings.get("mode") == "remaster":
            latent_ref, _ = self._append_remaster_latent(
                workflow,
                max(int(node_id) for node_id in workflow) + 1,
                uploaded_image=str(settings.get("init_image_path") or ""),
                vae_ref=[vae_node, 0],
                settings=settings,
                mapping=mapping,
                latent_dimension_divisor=2,
            )
        elif settings.get("mode") == "inpaint":
            init_image = settings.get("init_image_path")
            mask_image = settings.get("mask_image_path")
            if not init_image:
                raise RuntimeErrorResponse("init_image_path is required for Flux.2 inpaint.")
            if not mask_image:
                raise RuntimeErrorResponse("mask_image_path is required for Flux.2 inpaint.")
            image_node = str(int(latent_node) + 1)
            mask_node = str(int(image_node) + 1)
            encode_node = str(int(mask_node) + 1)
            noise_mask_node = str(int(encode_node) + 1)
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": init_image},
            }
            workflow[mask_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": mask_image},
            }
            workflow[encode_node] = {
                "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                "inputs": {"pixels": [image_node, 0], "vae": [vae_node, 0]},
            }
            workflow[noise_mask_node] = {
                "class_type": mapping.class_type("set_latent_noise_mask", "SetLatentNoiseMask"),
                "inputs": {"samples": [encode_node, 0], "mask": [mask_node, 1]},
            }
            settings["inpaint_runtime"] = "flux2_vae_noise_mask_img2img"
            latent_ref: list[str | int] = [noise_mask_node, 0]
        elif settings.get("mode") == "img2img":
            init_image = settings.get("init_image_path")
            if not init_image:
                raise RuntimeErrorResponse("init_image_path is required for Flux.2 img2img.")
            image_node = str(int(latent_node) + 1)
            encode_node = str(int(image_node) + 1)
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": init_image},
            }
            workflow[encode_node] = {
                "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                "inputs": {"pixels": [image_node, 0], "vae": [vae_node, 0]},
            }
            latent_ref: list[str | int] = [encode_node, 0]
        else:
            workflow[latent_node] = {
                "class_type": mapping.class_type("latent", "EmptyFlux2LatentImage"),
                "inputs": {
                    "width": settings["width"],
                    "height": settings["height"],
                    "batch_size": settings.get("batch_size", 1),
                },
            }
            latent_ref = [latent_node, 0]
        next_id = max(int(node_id) for node_id in workflow) + 1
        hires_model_ref: list[str | int] | None = None
        hires_checkpoint = _hires_checkpoint_setting(settings) if self._uses_comfy_hires(settings) else None
        if hires_checkpoint:
            hires_model_node = str(next_id)
            next_id += 1
            workflow[hires_model_node] = {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(hires_checkpoint, "diffusion_models"),
                    "weight_dtype": "default",
                },
            }
            hires_model_ref = [hires_model_node, 0]
            settings["hires_checkpoint_runtime"] = "second_pass_diffusion_model_loader"
            settings["hires_checkpoint_kind"] = "diffusion_model"
        for lora in request.get("loras", []):
            previous_clip_ref = clip_ref
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = self._lora_loader_node(lora, model_ref, clip_ref, mapping, settings)
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]
            if hires_model_ref is not None:
                hires_node_id = str(next_id)
                next_id += 1
                workflow[hires_node_id] = self._lora_loader_node(
                    lora,
                    hires_model_ref,
                    previous_clip_ref,
                    mapping,
                    settings,
                )
                hires_model_ref = [hires_node_id, 0]
        workflow[conditioning_node]["inputs"]["clip"] = clip_ref
        return self._build_custom_sampler_tail(
            workflow,
            settings,
            model_ref=model_ref,
            conditioning_ref=[guidance_node, 0],
            latent_ref=latent_ref,
            vae_ref=[vae_node, 0],
            mapping=mapping,
            scheduler_kind="flux2",
            hires_model_ref=hires_model_ref,
        )

    def _build_anima(self, request: dict) -> dict:
        mapping = self._mapping("anima")
        settings = request["settings"]
        spatial_runtime_stage = str(request.get("_spatial_runtime_stage") or "base")
        if any(
            isinstance(unit, dict) and bool(unit.get("regional_prompts"))
            for unit in settings.get("anima_lllite_units", [])
        ):
            raise RuntimeErrorResponse(
                "Legacy Anima regional prompts cannot be used for a new generation. "
                "Open the prompt layout editor and import the saved regions first."
            )
        model_slots = request.get("model_slots", {})
        diffusion_model = model_slots.get("diffusion_model")
        text_encoder = model_slots.get("text_encoder")
        vae = model_slots.get("vae")
        bundle = settings.get("anima_generation_bundle")
        is_anima_3_8b = (
            isinstance(bundle, dict)
            and bundle.get("variant_id") == "anima_3_8b_52"
        )
        if is_anima_3_8b and settings.get("mode", "txt2img") not in {
            "txt2img",
            "img2img",
            "inpaint",
        }:
            raise RuntimeErrorResponse(
                "Anima 3.8B supports txt2img, img2img, and inpaint."
            )
        qwen35_text_encoder = model_slots.get("qwen35_text_encoder")
        progressive_adapter = model_slots.get("progressive_adapter")
        missing = [
            name
            for name, value in {
                "diffusion_model": diffusion_model,
                "text_encoder": text_encoder,
                "vae": vae,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeErrorResponse(f"Anima requires model slots before generation: {', '.join(missing)}")
        if is_anima_3_8b and (not qwen35_text_encoder or not progressive_adapter):
            raise RuntimeErrorResponse(
                "Anima 3.8B requires Qwen3.5 4B and its progressive adapter."
            )

        diffusion_node = mapping.node("diffusion_loader", "1")
        text_encoder_node = mapping.node("text_encoder_loader", "2")
        vae_node = mapping.node("vae_loader", "3")
        conditioning_node = mapping.node("conditioning", "4")
        latent_node = mapping.node("latent", "5")
        negative_conditioning_node = mapping.node("negative_conditioning", "6")
        qwen35_loader_node = mapping.node("qwen35_loader", "7")
        model_ref: list[str | int] = [diffusion_node, 0]
        clip_ref: list[str | int] = [text_encoder_node, 0]
        workflow: dict[str, dict] = {
            diffusion_node: {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(diffusion_model, "diffusion_models"),
                    "weight_dtype": "default",
                },
            },
            text_encoder_node: {
                "class_type": mapping.class_type("text_encoder_loader", "CLIPLoader"),
                "inputs": {
                    "clip_name": self._model_name(text_encoder, "text_encoders"),
                    "type": "stable_diffusion" if is_anima_3_8b else "qwen_image",
                },
            },
            vae_node: {
                "class_type": mapping.class_type("vae_loader", "VAELoader"),
                "inputs": {"vae_name": self._model_name(vae, "vae")},
            },
            latent_node: {
                "class_type": mapping.class_type(
                    "anima_3_8b_latent" if is_anima_3_8b else "latent",
                    "EmptySD3LatentImage"
                    if is_anima_3_8b
                    else "EmptyQwenImageLayeredLatentImage",
                ),
                "inputs": {
                    "width": settings["width"],
                    "height": settings["height"],
                    **({} if is_anima_3_8b else {"layers": 0}),
                    "batch_size": settings.get("batch_size", 1),
                },
            },
        }
        if is_anima_3_8b:
            workflow[qwen35_loader_node] = {
                "class_type": mapping.class_type(
                    "qwen35_loader", "AnimaQwen35Loader"
                ),
                "inputs": {
                    "qwen35_model": self._model_name(
                        qwen35_text_encoder, "text_encoders"
                    )
                },
            }
            workflow[conditioning_node] = {
                "class_type": mapping.class_type(
                    "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                ),
                "inputs": {
                    "model": model_ref,
                    "native_clip": clip_ref,
                    "qwen35_clip": [qwen35_loader_node, 0],
                    "adapter_name": self._model_name(
                        progressive_adapter, "text_encoders"
                    ),
                    "prompt": request["expanded_prompt"],
                    "adapter_strength": settings.get(
                        "anima_qwen35_adapter_strength", 1.0
                    ),
                },
            }
            negative_strength = float(
                settings.get("anima_qwen35_negative_adapter_strength") or 0.0
            )
            workflow[negative_conditioning_node] = (
                {
                    "class_type": mapping.class_type(
                        "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                    ),
                    "inputs": {
                        "model": model_ref,
                        "native_clip": clip_ref,
                        "qwen35_clip": [qwen35_loader_node, 0],
                        "adapter_name": self._model_name(
                            progressive_adapter, "text_encoders"
                        ),
                        "prompt": request.get("negative_prompt", ""),
                        "adapter_strength": negative_strength,
                    },
                }
                if negative_strength > 0
                else {
                    "class_type": mapping.class_type(
                        "native_text_encode", "CLIPTextEncode"
                    ),
                    "inputs": {
                        "clip": clip_ref,
                        "text": request.get("negative_prompt", ""),
                    },
                }
            )
        else:
            workflow[conditioning_node] = {
                "class_type": mapping.class_type(
                    "conditioning", "TextEncodeQwenImageEdit"
                ),
                "inputs": {
                    "clip": clip_ref,
                    "prompt": request["expanded_prompt"],
                    "vae": [vae_node, 0],
                },
            }
            workflow[negative_conditioning_node] = {
                "class_type": mapping.class_type(
                    "conditioning", "TextEncodeQwenImageEdit"
                ),
                "inputs": {
                    "clip": clip_ref,
                    "prompt": request.get("negative_prompt", ""),
                    "vae": [vae_node, 0],
                },
            }
        merge_context = request.get("_merge_runtime_context")
        if merge_context is not None:
            if not isinstance(merge_context, dict):
                raise RuntimeErrorResponse("Invalid internal merge runtime context.")
            source_b = merge_context.get("source_b")
            source_c = merge_context.get("source_c")
            if not isinstance(source_b, dict) or not source_b.get("diffusion_model"):
                raise RuntimeErrorResponse("Internal merge runtime source B diffusion model is required.")
            source_b_node = mapping.node("merge_source_b", "900")
            source_c_node = mapping.node("merge_source_c", "901")
            merge_node = mapping.node("merge_runtime", "902")
            workflow[source_b_node] = {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(source_b["diffusion_model"], "diffusion_models"),
                    "weight_dtype": "default",
                },
            }
            merge_inputs: dict[str, object] = {
                "model_a": [diffusion_node, 0],
                "model_b": [source_b_node, 0],
                "recipe_payload": str(merge_context.get("recipe_payload") or ""),
                "contract_version": str(merge_context.get("contract_version") or ""),
            }
            if isinstance(source_c, dict) and source_c.get("diffusion_model"):
                workflow[source_c_node] = {
                    "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                    "inputs": {
                        "unet_name": self._model_name(source_c["diffusion_model"], "diffusion_models"),
                        "weight_dtype": "default",
                    },
                }
                merge_inputs["model_c"] = [source_c_node, 0]
            workflow[merge_node] = {
                "class_type": mapping.class_type("merge_runtime", "AsteriaAnimaMergeRuntime"),
                "inputs": merge_inputs,
            }
            model_ref = [merge_node, 0]
        next_id = 8 if is_anima_3_8b else 7
        hires_model_ref: list[str | int] | None = None
        hires_checkpoint = _hires_checkpoint_setting(settings) if self._uses_comfy_hires(settings) else None
        if hires_checkpoint:
            hires_model_node = str(next_id)
            next_id += 1
            workflow[hires_model_node] = {
                "class_type": mapping.class_type("unet_loader", "UNETLoader"),
                "inputs": {
                    "unet_name": self._model_name(hires_checkpoint, "diffusion_models"),
                    "weight_dtype": "default",
                },
            }
            hires_model_ref = [hires_model_node, 0]
            settings["hires_checkpoint_runtime"] = "second_pass_diffusion_model_loader"
            settings["hires_checkpoint_kind"] = "diffusion_model"
        latent_ref: list[str | int] = [latent_node, 0]
        if settings.get("mode") == "remaster":
            workflow.pop(latent_node, None)
            latent_ref, next_id = self._append_remaster_latent(
                workflow,
                next_id,
                uploaded_image=str(settings.get("init_image_path") or ""),
                vae_ref=[vae_node, 0],
                settings=settings,
                mapping=mapping,
            )
            image_ref = [str(settings["remaster_source_image_node_id"]), 0]
            workflow[conditioning_node]["inputs"]["image"] = image_ref
            workflow[negative_conditioning_node]["inputs"]["image"] = image_ref
        elif settings.get("mode") in {"img2img", "inpaint"} and settings.get("init_image_path"):
            image_node = str(next_id)
            next_id += 1
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": settings["init_image_path"]},
            }
            image_ref = [image_node, 0]
            if not is_anima_3_8b:
                workflow[conditioning_node]["inputs"]["image"] = image_ref
                workflow[negative_conditioning_node]["inputs"]["image"] = image_ref
            if settings.get("mode") == "inpaint":
                mask_image = settings.get("mask_image_path")
                if not mask_image:
                    raise RuntimeErrorResponse("mask_image_path is required for Anima inpaint.")
                mask_node = str(next_id)
                next_id += 1
                workflow[mask_node] = {
                    "class_type": mapping.class_type("load_image", "LoadImage"),
                    "inputs": {"image": mask_image},
                }
                encode_node = str(next_id)
                next_id += 1
                workflow[encode_node] = {
                    "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                    "inputs": {
                        "pixels": image_ref,
                        "vae": [vae_node, 0],
                    },
                }
                workflow[latent_node] = {
                    "class_type": mapping.class_type("set_latent_noise_mask", "SetLatentNoiseMask"),
                    "inputs": {
                        "samples": [encode_node, 0],
                        "mask": [mask_node, 1],
                    },
                }
                settings["inpaint_runtime"] = "anima_vae_masked_img2img"
            else:
                workflow[latent_node] = {
                    "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                    "inputs": {"pixels": image_ref, "vae": [vae_node, 0]},
                }
        for lora in request.get("loras", []):
            previous_clip_ref = clip_ref
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = self._anima_lora_loader_node(
                lora,
                model_ref,
                clip_ref,
                mapping,
                settings,
            )
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]
            if hires_model_ref is not None:
                hires_node_id = str(next_id)
                next_id += 1
                workflow[hires_node_id] = self._anima_lora_loader_node(
                    lora,
                    hires_model_ref,
                    previous_clip_ref,
                    mapping,
                    settings,
                    remap_key="hires_anima_remap",
                )
                hires_model_ref = [hires_node_id, 0]
        if is_anima_3_8b:
            workflow[conditioning_node]["inputs"]["model"] = model_ref
            workflow[conditioning_node]["inputs"]["native_clip"] = clip_ref
            if (
                workflow[negative_conditioning_node]["class_type"]
                == mapping.class_type(
                    "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                )
            ):
                workflow[negative_conditioning_node]["inputs"]["model"] = model_ref
                workflow[negative_conditioning_node]["inputs"]["native_clip"] = clip_ref
            else:
                workflow[negative_conditioning_node]["inputs"]["clip"] = clip_ref
        else:
            workflow[conditioning_node]["inputs"]["clip"] = clip_ref
            workflow[negative_conditioning_node]["inputs"]["clip"] = clip_ref
        for unit in settings.get("anima_lllite_units", []):
            if not unit.get("enabled") or not unit.get("model") or not unit.get("image_path"):
                continue
            image_node = str(next_id)
            next_id += 1
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": unit["image_path"]},
            }
            contract = settings.get("anima_lllite_runtime_contract", "model_patch")
            inputs = {
                "model": model_ref,
                "image": [image_node, 0],
                "strength": unit.get("strength", 1.0),
                "start_percent": unit.get("guidance_start", 0.0),
                "end_percent": unit.get("guidance_end", 0.45),
            }
            if contract == "legacy_direct":
                inputs["lllite_name"] = self._model_name(unit["model"], "controlnet")
                inputs["preserve_wrapper"] = unit.get("preserve_wrapper", True)
            else:
                loader_node = str(next_id)
                next_id += 1
                workflow[loader_node] = {
                    "class_type": mapping.class_type("model_patch_loader", "ModelPatchLoader"),
                    "inputs": {
                        "name": self._model_name(unit["model"], "model_patches"),
                    },
                }
                inputs["model_patch"] = [loader_node, 0]
            if unit.get("mask_image_path"):
                mask_node = str(next_id)
                next_id += 1
                workflow[mask_node] = {
                    "class_type": mapping.class_type("load_image", "LoadImage"),
                    "inputs": {"image": unit["mask_image_path"]},
                }
                inputs["mask"] = [mask_node, 1]
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = {
                "class_type": mapping.class_type("anima_lllite_apply", "AnimaLLLiteApply"),
                "inputs": inputs,
            }
            if hires_model_ref is not None:
                hires_node_id = str(next_id)
                next_id += 1
                workflow[hires_node_id] = {
                    "class_type": mapping.class_type("anima_lllite_apply", "AnimaLLLiteApply"),
                    "inputs": inputs | {"model": hires_model_ref},
                }
                hires_model_ref = [hires_node_id, 0]
            unit["runtime"] = "comfy_workflow"
            model_ref = [node_id, 0]
        base_unpatched_model_ref = model_ref
        hires_unpatched_model_ref = hires_model_ref or model_ref
        base_positive_ref: list[str | int] = [conditioning_node, 0]
        base_negative_ref: list[str | int] = [negative_conditioning_node, 0]
        base_execution_state_ref: list[str | int] | None = None
        spatial_base = self._append_anima_spatial_stage(
            request=request,
            stage=spatial_runtime_stage,
            workflow=workflow,
            next_id=next_id,
            mapping=mapping,
            model_ref=base_unpatched_model_ref,
            clip_ref=clip_ref,
            vae_ref=[vae_node, 0],
            negative_ref=base_negative_ref,
            width=int(settings["width"]),
            height=int(settings["height"]),
            is_anima_3_8b=is_anima_3_8b,
            qwen35_ref=[qwen35_loader_node, 0] if is_anima_3_8b else None,
            progressive_adapter=(
                self._model_name(progressive_adapter, "text_encoders")
                if is_anima_3_8b and progressive_adapter
                else None
            ),
            output_index=0,
        )
        if spatial_base is not None:
            (
                model_ref,
                base_positive_ref,
                base_negative_ref,
                base_execution_state_ref,
                next_id,
            ) = spatial_base

        hires_positive_ref: list[str | int] | None = None
        hires_negative_ref: list[str | int] | None = None
        if self._uses_comfy_hires(settings) and request.get("_spatial_runtime_context") is not None:
            hires_payload = self._spatial_stage_payload(request, "hires")
            if hires_payload is None:
                raise RuntimeErrorResponse("Spatial Hires context is missing.")
            hires_negative_prompt = settings.get("hires_negative_prompt") or request.get(
                "negative_prompt", ""
            )
            hires_negative_node, next_id = self._allocate_prompt_node_id(
                workflow, next_id
            )
            if is_anima_3_8b:
                if negative_strength > 0:
                    workflow[hires_negative_node] = {
                        "class_type": mapping.class_type(
                            "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                        ),
                        "inputs": {
                            "model": hires_unpatched_model_ref,
                            "native_clip": clip_ref,
                            "qwen35_clip": [qwen35_loader_node, 0],
                            "adapter_name": self._model_name(
                                progressive_adapter, "text_encoders"
                            ),
                            "prompt": hires_negative_prompt,
                            "adapter_strength": negative_strength,
                        },
                    }
                    hires_negative_ref = [hires_negative_node, 1]
                else:
                    workflow[hires_negative_node] = {
                        "class_type": mapping.class_type(
                            "native_text_encode", "CLIPTextEncode"
                        ),
                        "inputs": {"clip": clip_ref, "text": hires_negative_prompt},
                    }
                    hires_negative_ref = [hires_negative_node, 0]
            else:
                workflow[hires_negative_node] = {
                    "class_type": mapping.class_type(
                        "conditioning", "TextEncodeQwenImageEdit"
                    ),
                    "inputs": {
                        "clip": clip_ref,
                        "prompt": hires_negative_prompt,
                        "vae": [vae_node, 0],
                    },
                }
                hires_negative_ref = [hires_negative_node, 0]
            hires_execution_state_ref: list[str | int] | None = None
            spatial_hires = self._append_anima_spatial_stage(
                request=request,
                stage="hires",
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                model_ref=hires_unpatched_model_ref,
                clip_ref=clip_ref,
                vae_ref=[vae_node, 0],
                negative_ref=hires_negative_ref,
                width=self._hires_target_dimensions(settings)[0],
                height=self._hires_target_dimensions(settings)[1],
                is_anima_3_8b=is_anima_3_8b,
                qwen35_ref=[qwen35_loader_node, 0] if is_anima_3_8b else None,
                progressive_adapter=(
                    self._model_name(progressive_adapter, "text_encoders")
                    if is_anima_3_8b and progressive_adapter
                    else None
                ),
                output_index=1 if is_anima_3_8b else 0,
            )
            if spatial_hires is not None:
                (
                    hires_model_ref,
                    hires_positive_ref,
                    hires_negative_ref,
                    hires_execution_state_ref,
                    next_id,
                ) = spatial_hires
        else:
            hires_execution_state_ref = None
        workflow = self._build_custom_sampler_tail(
            workflow,
            settings,
            model_ref=model_ref,
            conditioning_ref=base_positive_ref,
            negative_ref=base_negative_ref,
            hires_conditioning_ref=(
                hires_positive_ref
                if hires_positive_ref is not None
                else ([conditioning_node, 1] if is_anima_3_8b else None)
            ),
            hires_negative_ref=(
                hires_negative_ref
                if hires_negative_ref is not None
                else (
                    [negative_conditioning_node, 1]
                    if is_anima_3_8b and negative_strength > 0
                    else None
                )
            ),
            latent_ref=latent_ref,
            vae_ref=[vae_node, 0],
            mapping=mapping,
            hires_model_ref=hires_model_ref,
            base_execution_state_ref=base_execution_state_ref,
            hires_execution_state_ref=hires_execution_state_ref,
        )
        detailer_units = [
            unit
            for unit in settings.get("detailer_units", [])
            if isinstance(unit, dict) and unit.get("enabled", True)
        ]
        if settings.get("restore_faces") or detailer_units:
            previous_save_node = str(settings["primary_output_node_ids"][0])
            image_ref = list(workflow.pop(previous_save_node)["inputs"]["images"])
            next_id = max(int(node_id) for node_id in workflow) + 1
            if settings.get("restore_faces"):
                image_ref, next_id = self._append_face_detailer_node(
                    workflow,
                    next_id,
                    image_ref,
                    hires_model_ref or model_ref,
                    clip_ref,
                    [vae_node, 0],
                    [conditioning_node, 0],
                    [negative_conditioning_node, 0],
                    mapping,
                    settings,
                    {
                        "detector": "face_yolo",
                        "confidence": 0.3,
                        "mask_blur": 4,
                        "denoising_strength": 0.35,
                        "inpaint_only_masked": True,
                        "guide_size": 512,
                        "max_size": 1024,
                        "bbox_dilation": 10,
                        "bbox_crop_factor": 3.0,
                    },
                )
                settings["restore_faces_runtime"] = "comfy_face_detailer"
                settings["face_restoration_model"] = "FaceDetailer"
            for unit in detailer_units:
                image_ref, next_id = self._append_face_detailer_node(
                    workflow,
                    next_id,
                    image_ref,
                    hires_model_ref or model_ref,
                    clip_ref,
                    [vae_node, 0],
                    [conditioning_node, 0],
                    [negative_conditioning_node, 0],
                    mapping,
                    settings,
                    unit,
                    request.get("embeddings", []),
                )
            save_node = str(next_id)
            workflow[save_node] = {
                "class_type": mapping.class_type("save_image", "SaveImage"),
                "inputs": {"images": image_ref, "filename_prefix": "Asteria"},
            }
            settings["primary_output_node_ids"] = [save_node]
        return workflow

    def _build_custom_sampler_tail(
        self,
        workflow: dict[str, dict],
        settings: dict,
        model_ref: list[str | int],
        conditioning_ref: list[str | int],
        latent_ref: list[str | int],
        vae_ref: list[str | int],
        mapping: WorkflowMapping,
        negative_ref: list[str | int] | None = None,
        scheduler_kind: str = "basic",
        hires_model_ref: list[str | int] | None = None,
        hires_conditioning_ref: list[str | int] | None = None,
        hires_negative_ref: list[str | int] | None = None,
        base_execution_state_ref: list[str | int] | None = None,
        hires_execution_state_ref: list[str | int] | None = None,
    ) -> dict:
        next_id = max(int(node_id) for node_id in workflow) + 1
        sample_node = self._append_custom_sample(
            workflow,
            next_id,
            settings,
            model_ref,
            conditioning_ref,
            latent_ref,
            mapping,
            negative_ref=negative_ref,
            scheduler_kind=scheduler_kind,
        )
        next_id += self._custom_sample_node_count(settings, scheduler_kind)
        self._capture_image_check_boundary(settings, sample_node, vae_ref, mapping)
        if base_execution_state_ref is not None:
            next_id = append_spatial_execution_receipt(
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                execution_state_ref=base_execution_state_ref,
                samples_ref=[sample_node, 0],
            )
        if self._uses_comfy_hires(settings):
            if settings.get("hires_keep_first_pass"):
                first_pass_decode_node = str(next_id)
                first_pass_save_node = str(next_id + 1)
                next_id += 2
                workflow[first_pass_decode_node] = {
                    "class_type": mapping.class_type("vae_decode", "VAEDecode"),
                    "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
                }
                workflow[first_pass_save_node] = {
                    "class_type": mapping.class_type("save_image", "SaveImage"),
                    "inputs": {"images": [first_pass_decode_node, 0], "filename_prefix": "Asteria_first_pass"},
                }
            if self._hires_is_latent(settings):
                upscale_node = str(next_id)
                next_id += 1
                resize_dimensions = self._hires_resize_dimensions(settings)
                if resize_dimensions is not None:
                    workflow[upscale_node] = {
                        "class_type": mapping.class_type("latent_upscale", "LatentUpscale"),
                        "inputs": {
                            "samples": [sample_node, 0],
                            "upscale_method": self._latent_upscale_method(settings),
                            "width": resize_dimensions[0],
                            "height": resize_dimensions[1],
                            "crop": "disabled",
                        },
                    }
                else:
                    workflow[upscale_node] = {
                        "class_type": mapping.class_type("latent_upscale_by", "LatentUpscaleBy"),
                        "inputs": {
                            "samples": [sample_node, 0],
                            "upscale_method": self._latent_upscale_method(settings),
                            "scale_by": settings.get("hires_scale", 2.0),
                        },
                    }
                hires_latent_ref = [upscale_node, 0]
            else:
                first_decode_node = str(next_id)
                upscale_start_node = next_id + 1
                workflow[first_decode_node] = {
                    "class_type": mapping.class_type("vae_decode", "VAEDecode"),
                    "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
                }
                upscale_ref, next_id = self._append_image_hires_upscale(
                    workflow,
                    upscale_start_node,
                    [first_decode_node, 0],
                    settings,
                    mapping,
                )
                encode_node = str(next_id)
                next_id += 1
                workflow[encode_node] = {
                    "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                    "inputs": {"pixels": upscale_ref, "vae": vae_ref},
                }
                hires_latent_ref = [encode_node, 0]
            hires_sample_settings = settings | {
                "_prompt_trace_stage": "hires",
                    "steps": settings.get("hires_steps") or settings["steps"],
                    "cfg_scale": settings.get("hires_cfg_scale") or settings["cfg_scale"],
                    "sampler": self._hires_sampler(settings),
                    "scheduler": self._hires_scheduler(settings),
                    "denoise": settings.get("hires_denoising_strength", 0.7),
                    **(
                        dict(zip(("width", "height"), self._hires_target_dimensions(settings), strict=True))
                        if scheduler_kind == "flux2"
                        else {}
                    ),
                }
            sample_node = self._append_custom_sample(
                workflow,
                next_id,
                hires_sample_settings,
                hires_model_ref or model_ref,
                hires_conditioning_ref or conditioning_ref,
                hires_latent_ref,
                mapping,
                negative_ref=hires_negative_ref or negative_ref,
                scheduler_kind=scheduler_kind,
            )
            next_id += self._custom_sample_node_count(
                hires_sample_settings,
                scheduler_kind,
            )
            if hires_execution_state_ref is not None:
                next_id = append_spatial_execution_receipt(
                    workflow=workflow,
                    next_id=next_id,
                    mapping=mapping,
                    execution_state_ref=hires_execution_state_ref,
                    samples_ref=[sample_node, 0],
                )

        decode_node = str(next_id)
        save_node = str(next_id + 1)
        workflow[decode_node] = {
            "class_type": mapping.class_type("vae_decode", "VAEDecode"),
            "inputs": {"samples": [sample_node, 0], "vae": vae_ref},
        }
        workflow[save_node] = {
            "class_type": mapping.class_type("save_image", "SaveImage"),
            "inputs": {"images": [decode_node, 0], "filename_prefix": "Asteria"},
        }
        mark_stage(settings, decoded_stage_kind(settings), images=[decode_node, 0])
        settings["primary_output_node_ids"] = [save_node]
        return workflow

    def _append_custom_sample(
        self,
        workflow: dict[str, dict],
        next_id: int,
        settings: dict,
        model_ref: list[str | int],
        conditioning_ref: list[str | int],
        latent_ref: list[str | int],
        mapping: WorkflowMapping,
        negative_ref: list[str | int] | None = None,
        scheduler_kind: str = "basic",
    ) -> str:
        seed = settings.get("seed", -1)
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        noise_node = str(next_id)
        guider_node = str(next_id + 1)
        sampler_node = str(next_id + 2)
        scheduler_node = str(next_id + 3)
        sample_node = str(next_id + 4)
        workflow[noise_node] = {
            "class_type": mapping.class_type("random_noise", "RandomNoise"),
            "inputs": {"noise_seed": seed},
        }
        if negative_ref is not None:
            workflow[guider_node] = {
                "class_type": mapping.class_type("cfg_guider", "CFGGuider"),
                "inputs": {
                    "model": model_ref,
                    "positive": conditioning_ref,
                    "negative": negative_ref,
                    "cfg": settings.get("cfg_scale", 4.5),
                },
            }
        else:
            workflow[guider_node] = {
                "class_type": mapping.class_type("basic_guider", "BasicGuider"),
                "inputs": {"model": model_ref, "conditioning": conditioning_ref},
            }
        workflow[sampler_node] = {
            "class_type": mapping.class_type("sampler_select", "KSamplerSelect"),
            "inputs": {"sampler_name": self._sampler_name(settings.get("sampler", "Euler"))},
        }
        if scheduler_kind == "flux2":
            workflow[scheduler_node] = {
                "class_type": mapping.class_type("flux2_scheduler", "Flux2Scheduler"),
                "inputs": {
                    "steps": settings["steps"],
                    "width": settings["width"],
                    "height": settings["height"],
                },
            }
        else:
            workflow[scheduler_node] = {
                "class_type": mapping.class_type("basic_scheduler", "BasicScheduler"),
                "inputs": {
                    "model": model_ref,
                    "scheduler": self._scheduler_name(settings.get("scheduler", "Simple")),
                    "steps": settings["steps"],
                    "denoise": settings.get("denoise", 1.0),
                },
            }
        sigmas_ref: list[str | int] = [scheduler_node, 0]
        if scheduler_kind == "flux2" and float(settings.get("denoise", 1.0)) < 1.0:
            split_node = str(next_id + 4)
            sample_node = str(next_id + 5)
            steps = int(settings["steps"])
            split_step = max(0, min(steps, round(steps * (1.0 - float(settings.get("denoise", 1.0))))))
            workflow[split_node] = {
                "class_type": mapping.class_type("split_sigmas", "SplitSigmas"),
                "inputs": {"sigmas": [scheduler_node, 0], "step": split_step},
            }
            sigmas_ref = [split_node, 1]
        workflow[sample_node] = {
            "class_type": mapping.class_type("sampler_custom", "SamplerCustomAdvanced"),
            "_meta": {"asteria_stage": settings.get("_prompt_trace_stage", "base")},
            "inputs": {
                "noise": [noise_node, 0],
                "guider": [guider_node, 0],
                "sampler": [sampler_node, 0],
                "sigmas": sigmas_ref,
                "latent_image": latent_ref,
            },
        }
        return sample_node

    def _custom_sample_node_count(self, settings: dict, scheduler_kind: str) -> int:
        if scheduler_kind == "flux2" and float(settings.get("denoise", 1.0)) < 1.0:
            return 6
        return 5

    def _ksampler_node(
        self,
        settings: dict,
        model_ref: list[str | int],
        positive_ref: list[str | int],
        negative_ref: list[str | int],
        latent_ref: list[str | int],
        mapping: WorkflowMapping | None = None,
        use_inspire: bool = False,
    ) -> dict:
        seed = settings.get("seed", -1)
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        inputs = {
            "seed": seed,
            "steps": settings["steps"],
            "cfg": settings["cfg_scale"],
            "sampler_name": self._sampler_name(settings.get("sampler", "Euler a")),
            "scheduler": self._scheduler_name(settings.get("scheduler", "Automatic")),
            "denoise": settings.get("denoise", 1.0),
            "model": model_ref,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": latent_ref,
        }
        if use_inspire:
            variation_seed = int(settings.get("subseed") or 0)
            if variation_seed < 0:
                variation_seed = 0
            inputs = {
                **inputs,
                **_inspire_noise_inputs(settings, seed),
                "batch_seed_mode": settings.get("batch_seed_mode") or "incremental",
                "variation_seed": variation_seed,
                "variation_strength": float(settings.get("subseed_strength") or 0),
                "variation_method": settings.get("variation_method") or "linear",
            }
        return {
            "class_type": (
                mapping.class_type("ksampler_inspire", INSPIRE_KSAMPLER_CLASS)
                if use_inspire and mapping
                else mapping.class_type("ksampler", "KSampler")
                if mapping
                else "KSampler"
            ),
            "inputs": inputs,
            "_meta": {"asteria_stage": settings.get("_prompt_trace_stage", "base")},
        }

    def _ksampler_advanced_node(
        self,
        settings: dict,
        model_ref: list[str | int],
        positive_ref: list[str | int],
        negative_ref: list[str | int],
        latent_ref: list[str | int],
        mapping: WorkflowMapping,
        *,
        start_at_step: int,
        end_at_step: int,
        add_noise: bool,
        return_with_leftover_noise: bool,
    ) -> dict:
        seed = settings.get("seed", -1)
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        return {
            "class_type": mapping.class_type("ksampler_advanced", "KSamplerAdvanced"),
            "_meta": {"asteria_stage": settings.get("_prompt_trace_stage", "base")},
            "inputs": {
                "model": model_ref,
                "add_noise": "enable" if add_noise else "disable",
                "noise_seed": seed,
                "steps": settings["steps"],
                "cfg": settings["cfg_scale"],
                "sampler_name": self._sampler_name(settings.get("sampler", "Euler a")),
                "scheduler": self._scheduler_name(settings.get("scheduler", "Automatic")),
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": latent_ref,
                "start_at_step": start_at_step,
                "end_at_step": end_at_step,
                "return_with_leftover_noise": "enable" if return_with_leftover_noise else "disable",
            },
        }

    def _refiner_switch_step(self, settings: dict, steps: int) -> int:
        if steps <= 1:
            return max(0, steps)
        switch_at = float(settings.get("refiner_switch_at") or 0.8)
        return max(1, min(steps - 1, int(round(steps * switch_at))))

    def _request_prompt_syntax_plan(
        self,
        request: dict,
        key: str,
    ) -> PromptSyntaxPlan | None:
        namespace = request.get("prompt_syntax")
        if not isinstance(namespace, dict) or namespace.get(key) is None:
            return None
        try:
            plan = prompt_syntax_plan_from_dict(namespace[key])
        except PromptSyntaxError as exc:
            raise RuntimeErrorResponse(f"Prompt syntax plan is invalid: {exc.code}") from exc
        if plan.profile_id != request.get("profile"):
            raise RuntimeErrorResponse("Prompt syntax profile does not match the runtime profile.")
        return plan

    @staticmethod
    def _identity_prompt_plan(plan: PromptSyntaxPlan | None) -> bool:
        return plan is None or plan.operation == "identity"

    @staticmethod
    def _allocate_prompt_node_id(workflow: dict[str, dict], next_id: int) -> tuple[str, int]:
        while str(next_id) in workflow:
            next_id += 1
        return str(next_id), next_id + 1

    def _append_sdxl_prompt_conditioning(
        self,
        *,
        workflow: dict[str, dict],
        next_id: int,
        preferred_node: str,
        plan: PromptSyntaxPlan | None,
        fallback_text: str,
        clip_ref: list[str | int],
        model_ref: list[str | int],
        mapping: WorkflowMapping,
        width: int,
        height: int,
        stage_settings: dict,
        embeddings: list[str],
        refiner: bool = False,
    ) -> tuple[list[str | int], int]:
        def conditioning_node(text: str) -> dict:
            normalized = self._comfy_prompt_text(text, embeddings)
            if refiner:
                return self._sdxl_refiner_conditioning_node(
                    normalized,
                    clip_ref,
                    mapping,
                    width,
                    height,
                )
            return self._sdxl_conditioning_node(
                normalized,
                clip_ref,
                mapping,
                width,
                height,
            )

        if plan is None or plan.operation == "identity":
            workflow[preferred_node] = conditioning_node(fallback_text)
            return [preferred_node, 0], next_id

        encoded_refs: list[list[str | int]] = []
        for index, segment in enumerate(plan.segments):
            if index == 0:
                node_id = preferred_node
            else:
                node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
            workflow[node_id] = conditioning_node(segment.text)
            encoded_refs.append([node_id, 0])

        if plan.operation == "token_weight":
            return encoded_refs[0], next_id
        if plan.operation == "scheduled_switch":
            ranged_refs: list[list[str | int]] = []
            for encoded_ref, segment in zip(encoded_refs, plan.segments, strict=True):
                node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[node_id] = {
                    "class_type": mapping.class_type(
                        "conditioning_time_range", "ConditioningSetTimestepRange"
                    ),
                    "inputs": {
                        "conditioning": encoded_ref,
                        "start": float(segment.start or 0),
                        "end": float(segment.end or 1),
                    },
                }
                ranged_refs.append([node_id, 0])
            node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
            workflow[node_id] = {
                "class_type": mapping.class_type(
                    "conditioning_combine", "ConditioningCombine"
                ),
                "inputs": {
                    "conditioning_1": ranged_refs[0],
                    "conditioning_2": ranged_refs[1],
                },
            }
            return [node_id, 0], next_id
        if plan.operation == "alternating_switch":
            node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
            workflow[node_id] = {
                "class_type": mapping.class_type(
                    "prompt_step_schedule", "AsteriaPromptStepSchedule"
                ),
                "inputs": {
                    "model": model_ref,
                    "conditioning_a": encoded_refs[0],
                    "conditioning_b": encoded_refs[1],
                    "steps": int(stage_settings["steps"]),
                    "sampler_name": self._sampler_name(
                        stage_settings.get("sampler", "Euler a")
                    ),
                    "scheduler": self._scheduler_name(
                        stage_settings.get("scheduler", "Automatic")
                    ),
                    "denoise": float(stage_settings.get("denoise", 1.0)),
                },
            }
            return [node_id, 0], next_id
        if plan.operation == "conditioning_and":
            weighted_refs: list[list[str | int]] = []
            for encoded_ref, segment in zip(encoded_refs, plan.segments, strict=True):
                node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[node_id] = {
                    "class_type": mapping.class_type(
                        "conditioning_multiply", "ConditioningMultiply"
                    ),
                    "inputs": {
                        "conditioning": encoded_ref,
                        "multiplier": float(segment.weight),
                    },
                }
                weighted_refs.append([node_id, 0])
            combined_ref = weighted_refs[0]
            for weighted_ref in weighted_refs[1:]:
                node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[node_id] = {
                    "class_type": mapping.class_type(
                        "conditioning_combine", "ConditioningCombine"
                    ),
                    "inputs": {
                        "conditioning_1": combined_ref,
                        "conditioning_2": weighted_ref,
                    },
                }
                combined_ref = [node_id, 0]
            return combined_ref, next_id
        if plan.operation == "conditioning_break":
            concatenated_ref = encoded_refs[0]
            for encoded_ref in encoded_refs[1:]:
                node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[node_id] = {
                    "class_type": mapping.class_type(
                        "conditioning_concat", "ConditioningConcat"
                    ),
                    "inputs": {
                        "conditioning_to": concatenated_ref,
                        "conditioning_from": encoded_ref,
                    },
                }
                concatenated_ref = [node_id, 0]
            return concatenated_ref, next_id
        raise RuntimeErrorResponse(f"Unsupported prompt syntax operation: {plan.operation}")

    def _sdxl_conditioning_node(
        self,
        text: str,
        clip_ref: list[str | int],
        mapping: WorkflowMapping,
        width: int,
        height: int,
    ) -> dict:
        return {
            "class_type": mapping.class_type("clip_text_encode_sdxl", "CLIPTextEncodeSDXL"),
            "inputs": {
                "clip": clip_ref,
                "width": width,
                "height": height,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": width,
                "target_height": height,
                "text_g": text,
                "text_l": text,
            },
        }

    def _sdxl_refiner_conditioning_node(
        self,
        text: str,
        clip_ref: list[str | int],
        mapping: WorkflowMapping,
        width: int,
        height: int,
    ) -> dict:
        return {
            "class_type": mapping.class_type("clip_text_encode_sdxl_refiner", "CLIPTextEncodeSDXLRefiner"),
            "inputs": {
                "ascore": 6.0,
                "width": width,
                "height": height,
                "text": text,
                "clip": clip_ref,
            },
        }

    def _spatial_stage_payload(
        self,
        request: dict,
        stage: str,
    ) -> tuple[
        SpatialPromptPlan,
        SpatialMaskBundle,
        SpatialModelProfileReceipt,
        str,
        dict[str, str],
    ] | None:
        context = request.get("_spatial_runtime_context")
        if context is None:
            return None
        try:
            if (
                not isinstance(context, dict)
                or context.get("schema_version")
                != "asteria.spatial-workflow-context.v1"
            ):
                raise ValueError("context")
            prompt_plan = SpatialPromptPlan.model_validate(context["prompt_plan"])
            if prompt_plan.profile_id != request.get("profile"):
                raise ValueError("profile")
            stages = context["stages"]
            if not isinstance(stages, dict):
                raise ValueError("stages")
            stage_payload = stages[stage]
            if not isinstance(stage_payload, dict):
                raise ValueError("stage")
            mask_bundle = SpatialMaskBundle.model_validate(stage_payload["mask_bundle"])
            profile_receipt = SpatialModelProfileReceipt.model_validate(
                stage_payload["profile_receipt"]
            )
            compatible_model_families = {
                "sdxl": {"sdxl", "illustrious"},
                "anima": {"anima"},
            }.get(prompt_plan.profile_id, {prompt_plan.profile_id})
            if profile_receipt.model_family not in compatible_model_families:
                raise ValueError("profile_receipt")
            profile_json = stage_payload["profile_json"]
            mask_paths = stage_payload["mask_paths"]
            if not isinstance(profile_json, str) or not isinstance(mask_paths, dict):
                raise ValueError("payload")
            normalized_paths = {
                target_id: path
                for target_id, path in mask_paths.items()
                if isinstance(target_id, str)
                and isinstance(path, str)
                and path.strip()
            }
            if len(normalized_paths) != len(mask_paths):
                raise ValueError("mask_paths")
            spatial_prompt_stage(prompt_plan, stage)
        except (KeyError, TypeError, ValueError, SpatialConditioningBuilderError) as error:
            raise RuntimeErrorResponse(
                f"Invalid internal spatial runtime context for {stage}."
            ) from error
        return prompt_plan, mask_bundle, profile_receipt, profile_json, normalized_paths

    def _append_spatial_mask_loaders(
        self,
        *,
        workflow: dict[str, dict],
        next_id: int,
        mask_paths: dict[str, str],
        mapping: WorkflowMapping,
    ) -> tuple[dict[str, list[str | int]], int]:
        mask_refs: dict[str, list[str | int]] = {}
        for target_id, mask_path in mask_paths.items():
            node_id, next_id = self._allocate_prompt_node_id(workflow, next_id)
            workflow[node_id] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": mask_path},
            }
            mask_refs[target_id] = [node_id, 1]
        return mask_refs, next_id

    @staticmethod
    def _record_spatial_fragments_sha256(
        request: dict,
        *,
        stage: str,
        fragments_sha256: str,
    ) -> None:
        context = request.get("_spatial_runtime_context")
        if not isinstance(context, dict):
            raise RuntimeErrorResponse("Spatial prompt runtime context is invalid.")
        stages = context.get("stages")
        stage_payload = stages.get(stage) if isinstance(stages, dict) else None
        if not isinstance(stage_payload, dict):
            raise RuntimeErrorResponse("Spatial prompt runtime stage is invalid.")
        stage_payload["fragments_sha256"] = fragments_sha256

    @staticmethod
    def _spatial_combined_prompt(global_prompt: str, target_prompt: str) -> str:
        return join_prompt_contract_fragments((global_prompt, target_prompt))

    def _append_sdxl_spatial_stage(
        self,
        *,
        request: dict,
        stage: str,
        workflow: dict[str, dict],
        next_id: int,
        mapping: WorkflowMapping,
        model_ref: list[str | int],
        clip_ref: list[str | int],
        global_ref: list[str | int],
        negative_ref: list[str | int],
        width: int,
        height: int,
        stage_settings: dict,
        refiner: bool = False,
    ) -> tuple[
        list[str | int],
        list[str | int],
        list[str | int],
        list[str | int],
        int,
    ] | None:
        payload = self._spatial_stage_payload(request, stage)
        if payload is None:
            return None
        # V5 establishes composition during the base denoise only. Hires and
        # refiner stages keep the full prompt, but must not register another
        # Cross/Self boundary router at the enlarged latent resolution.
        if stage != "base":
            return None
        prompt_plan, mask_bundle, profile_receipt, profile_json, mask_paths = payload
        if mask_bundle.width != width or mask_bundle.height != height:
            raise RuntimeErrorResponse(
                f"Spatial {stage} mask dimensions do not match the generation stage."
            )
        mask_refs, next_id = self._append_spatial_mask_loaders(
            workflow=workflow,
            next_id=next_id,
            mask_paths=mask_paths,
            mapping=mapping,
        )
        try:
            bundle = build_spatial_conditioning_bundle(
                prompt_plan=prompt_plan,
                mask_bundle=mask_bundle,
                profile_receipt=profile_receipt,
                stage=stage,
                encoder_id=(
                    mapping.class_type(
                        "clip_text_encode_sdxl_refiner", "CLIPTextEncodeSDXLRefiner"
                    )
                    if refiner
                    else mapping.class_type(
                        "clip_text_encode_sdxl", "CLIPTextEncodeSDXL"
                    )
                ),
                text_encoder_ref=clip_ref,
                full_positive_conditioning_ref=global_ref,
                negative_conditioning_ref=negative_ref,
            )
            mapped = append_spatial_prompt_mapping(
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                model_ref=model_ref,
                conditioning_bundle=bundle,
                mask_bundle=mask_bundle,
                mask_refs=mask_refs,
                profile_json=profile_json,
                initial_noise_seed=int(stage_settings["seed"]),
                routing_geometry=request["_spatial_runtime_context"]["stages"][stage].get("routing_geometry"),
                width=width,
                height=height,
            )
            self._record_spatial_fragments_sha256(
                request,
                stage=stage,
                fragments_sha256=mapped.fragments_sha256,
            )
        except (SpatialConditioningBuilderError, SpatialPromptMappingError) as error:
            raise RuntimeErrorResponse(
                f"Spatial prompt workflow mapping failed: {error}"
            ) from error
        return (
            mapped.model_ref,
            mapped.positive_ref,
            mapped.negative_ref,
            mapped.execution_state_ref,
            mapped.next_id,
        )

    def _append_sdxl_spatial_global(
        self,
        *,
        request: dict,
        stage: str,
        workflow: dict[str, dict],
        next_id: int,
        mapping: WorkflowMapping,
        clip_ref: list[str | int],
        model_ref: list[str | int],
        width: int,
        height: int,
        stage_settings: dict,
        refiner: bool = False,
    ) -> tuple[list[str | int], int] | None:
        payload = self._spatial_stage_payload(request, stage)
        if payload is None:
            return None
        stage_plan = spatial_prompt_stage(payload[0], stage)
        if stage_plan.full_text is None:
            raise RuntimeErrorResponse("Spatial full prompt is missing.")
        if stage_plan.scheduled_bindings:
            ranged_refs: list[list[str | int]] = []
            embeddings = list(payload[0].embeddings)
            for scheduled in stage_plan.scheduled_bindings:
                encode_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
                normalized_text = self._comfy_prompt_text(scheduled.full_text, embeddings)
                workflow[encode_node] = (
                    self._sdxl_refiner_conditioning_node(
                        normalized_text,
                        clip_ref,
                        mapping,
                        width,
                        height,
                    )
                    if refiner
                    else self._sdxl_conditioning_node(
                        normalized_text,
                        clip_ref,
                        mapping,
                        width,
                        height,
                    )
                )
                range_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[range_node] = {
                    "class_type": mapping.class_type(
                        "conditioning_time_range", "ConditioningSetTimestepRange"
                    ),
                    "inputs": {
                        "conditioning": [encode_node, 0],
                        "start": scheduled.start_percent,
                        "end": scheduled.end_percent,
                    },
                }
                ranged_refs.append([range_node, 0])
            combined = ranged_refs[0]
            for ranged_ref in ranged_refs[1:]:
                combine_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[combine_node] = {
                    "class_type": mapping.class_type(
                        "conditioning_combine", "ConditioningCombine"
                    ),
                    "inputs": {
                        "conditioning_1": combined,
                        "conditioning_2": ranged_ref,
                    },
                }
                combined = [combine_node, 0]
            return combined, next_id
        preferred_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
        return self._append_sdxl_prompt_conditioning(
            workflow=workflow,
            next_id=next_id,
            preferred_node=preferred_node,
            plan=None,
            fallback_text=stage_plan.full_text,
            clip_ref=clip_ref,
            model_ref=model_ref,
            mapping=mapping,
            width=width,
            height=height,
            stage_settings=stage_settings,
            embeddings=list(payload[0].embeddings),
            refiner=refiner,
        )

    def _append_anima_spatial_stage(
        self,
        *,
        request: dict,
        stage: str,
        workflow: dict[str, dict],
        next_id: int,
        mapping: WorkflowMapping,
        model_ref: list[str | int],
        clip_ref: list[str | int],
        vae_ref: list[str | int],
        negative_ref: list[str | int],
        width: int,
        height: int,
        is_anima_3_8b: bool,
        qwen35_ref: list[str | int] | None = None,
        progressive_adapter: str | None = None,
        output_index: int = 0,
    ) -> tuple[
        list[str | int],
        list[str | int],
        list[str | int],
        list[str | int] | None,
        int,
    ] | None:
        payload = self._spatial_stage_payload(request, stage)
        if payload is None:
            return None
        prompt_plan, mask_bundle, profile_receipt, profile_json, mask_paths = payload
        if mask_bundle.width != width or mask_bundle.height != height:
            raise RuntimeErrorResponse(
                f"Spatial {stage} mask dimensions do not match the generation stage."
            )
        stage_plan = spatial_prompt_stage(prompt_plan, stage)
        if stage_plan.full_text is None:
            raise RuntimeErrorResponse("Spatial full prompt is missing.")
        full_positive_prompt = stage_plan.full_text
        schedule = tuple(stage_plan.scheduled_bindings or ())
        prompts = (
            tuple((item.full_text, item.start_percent, item.end_percent) for item in schedule)
            if schedule
            else ((full_positive_prompt, 0.0, 1.0),)
        )
        ranged_refs: list[list[str | int]] = []
        for prompt_text, start_percent, end_percent in prompts:
            global_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
            if is_anima_3_8b:
                if qwen35_ref is None or not progressive_adapter:
                    raise RuntimeErrorResponse(
                        "Anima 3.8B spatial prompt encoder inputs are incomplete."
                    )
                workflow[global_node] = {
                    "class_type": mapping.class_type(
                        "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                    ),
                    "inputs": {
                        "model": model_ref,
                        "native_clip": clip_ref,
                        "qwen35_clip": qwen35_ref,
                        "adapter_name": progressive_adapter,
                        "prompt": prompt_text,
                        "adapter_strength": request["settings"].get(
                            "anima_qwen35_adapter_strength", 1.0
                        ),
                    },
                }
            else:
                workflow[global_node] = {
                    "class_type": mapping.class_type(
                        "conditioning", "TextEncodeQwenImageEdit"
                    ),
                    "inputs": {
                        "clip": clip_ref,
                        "prompt": prompt_text,
                        "vae": vae_ref,
                    },
                }
            encoded_ref: list[str | int] = [global_node, output_index]
            if schedule:
                range_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
                workflow[range_node] = {
                    "class_type": mapping.class_type(
                        "conditioning_time_range", "ConditioningSetTimestepRange"
                    ),
                    "inputs": {
                        "conditioning": encoded_ref,
                        "start": start_percent,
                        "end": end_percent,
                    },
                }
                encoded_ref = [range_node, 0]
            ranged_refs.append(encoded_ref)
        global_ref = ranged_refs[0]
        for ranged_ref in ranged_refs[1:]:
            combine_node, next_id = self._allocate_prompt_node_id(workflow, next_id)
            workflow[combine_node] = {
                "class_type": mapping.class_type(
                    "conditioning_combine", "ConditioningCombine"
                ),
                "inputs": {
                    "conditioning_1": global_ref,
                    "conditioning_2": ranged_ref,
                },
            }
            global_ref = [combine_node, 0]
        # As with SDXL, the Anima V5 router is a base-stage composition aid.
        # A later Hires/Topaz pass receives the complete conditioning but no
        # second boundary mask, which bounds both memory and runtime overhead.
        if stage != "base":
            return model_ref, global_ref, negative_ref, None, next_id
        mask_refs, next_id = self._append_spatial_mask_loaders(
            workflow=workflow,
            next_id=next_id,
            mask_paths=mask_paths,
            mapping=mapping,
        )
        try:
            bundle = build_spatial_conditioning_bundle(
                prompt_plan=prompt_plan,
                mask_bundle=mask_bundle,
                profile_receipt=profile_receipt,
                stage=stage,
                encoder_id=(
                    mapping.class_type(
                        "qwen35_unified_prompt", "AnimaQwen35UnifiedPrompt"
                    )
                    if is_anima_3_8b
                    else mapping.class_type(
                        "conditioning", "TextEncodeQwenImageEdit"
                    )
                ),
                text_encoder_ref=clip_ref,
                full_positive_conditioning_ref=global_ref,
                negative_conditioning_ref=negative_ref,
            )
            mapped = append_spatial_prompt_mapping(
                workflow=workflow,
                next_id=next_id,
                mapping=mapping,
                model_ref=model_ref,
                conditioning_bundle=bundle,
                mask_bundle=mask_bundle,
                mask_refs=mask_refs,
                profile_json=profile_json,
                initial_noise_seed=int(request["settings"]["seed"]),
                routing_geometry=request["_spatial_runtime_context"]["stages"][stage].get("routing_geometry"),
                width=width,
                height=height,
                secondary_text_encoder_ref=(qwen35_ref if is_anima_3_8b else None),
                progressive_adapter_name=(
                    progressive_adapter if is_anima_3_8b else None
                ),
                progressive_adapter_strength=float(
                    request["settings"].get("anima_qwen35_adapter_strength", 1.0)
                ),
            )
            self._record_spatial_fragments_sha256(
                request,
                stage=stage,
                fragments_sha256=mapped.fragments_sha256,
            )
        except (SpatialConditioningBuilderError, SpatialPromptMappingError) as error:
            raise RuntimeErrorResponse(
                f"Spatial prompt workflow mapping failed: {error}"
            ) from error
        return (
            mapped.model_ref,
            mapped.positive_ref,
            mapped.negative_ref,
            mapped.execution_state_ref,
            mapped.next_id,
        )

    def _uses_comfy_hires(self, settings: dict) -> bool:
        if not settings.get("hires_enabled"):
            return False
        return settings.get("hires_upscaler") != "Topaz Photo AI"

    def _append_remaster_latent(
        self,
        workflow: dict[str, dict],
        next_id: int,
        *,
        uploaded_image: str,
        vae_ref: list[str | int],
        settings: dict,
        mapping: WorkflowMapping,
        latent_dimension_divisor: int = 1,
    ) -> tuple[list[str | int], int]:
        if not uploaded_image:
            raise RuntimeErrorResponse("init_image_path is required for Remaster.")
        source_width = int(settings.get("remaster_source_width") or settings.get("width") or 0)
        source_height = int(settings.get("remaster_source_height") or settings.get("height") or 0)
        target_scale = float(settings.get("remaster_target_scale", 2.0))
        buffer_ratio = float(settings.get("remaster_buffer_ratio", 0.75))
        dimension_step = 8 * latent_dimension_divisor
        final_width = self._round_to_step(source_width * target_scale, dimension_step)
        final_height = self._round_to_step(source_height * target_scale, dimension_step)
        latent_width = self._round_to_step(final_width / latent_dimension_divisor)
        latent_height = self._round_to_step(final_height / latent_dimension_divisor)
        buffer_width = self._round_to_step(final_width * buffer_ratio, dimension_step)
        buffer_height = self._round_to_step(final_height * buffer_ratio, dimension_step)
        anchor_upscaler = str(settings.get("remaster_anchor_upscaler") or "R-ESGAN4x")
        anchor_model = upscaler_model_name(anchor_upscaler)
        if not anchor_model:
            raise RuntimeErrorResponse(f"Remaster upscaler does not use an upscale model: {anchor_upscaler}")

        image_node = str(next_id)
        loader_node = str(next_id + 1)
        model_upscale_node = str(next_id + 2)
        buffer_node = str(next_id + 3)
        encode_node = str(next_id + 4)
        latent_node = str(next_id + 5)
        workflow[image_node] = {
            "class_type": mapping.class_type("load_image", "LoadImage"),
            "inputs": {"image": uploaded_image},
        }

        workflow[loader_node] = {
            "class_type": mapping.class_type("upscale_model_loader", "UpscaleModelLoader"),
            "inputs": {"model_name": anchor_model},
        }
        workflow[model_upscale_node] = {
            "class_type": mapping.class_type("image_upscale_with_model", "ImageUpscaleWithModel"),
            "inputs": {"upscale_model": [loader_node, 0], "image": [image_node, 0]},
        }
        workflow[buffer_node] = {
            "class_type": mapping.class_type("image_scale", "ImageScale"),
            "inputs": {
                "image": [model_upscale_node, 0],
                "upscale_method": "area",
                "width": buffer_width,
                "height": buffer_height,
                "crop": "disabled",
            },
        }
        workflow[encode_node] = {
            "class_type": mapping.class_type("vae_encode", "VAEEncode"),
            "inputs": {"pixels": [buffer_node, 0], "vae": vae_ref},
        }
        workflow[latent_node] = {
            "class_type": mapping.class_type("latent_upscale", "LatentUpscale"),
            "inputs": {
                "samples": [encode_node, 0],
                "upscale_method": settings.get("remaster_latent_interpolation", "bilinear"),
                "width": latent_width,
                "height": latent_height,
                "crop": "disabled",
            },
        }
        settings["width"] = final_width
        settings["height"] = final_height
        settings["remaster_source_image_node_id"] = image_node
        settings["remaster_node_ids"] = {
            "source": image_node,
            "anchor_loader": loader_node,
            "anchor_upscale": model_upscale_node,
            "buffer": buffer_node,
            "encode": encode_node,
            "latent_target": latent_node,
        }
        settings["remaster_stage_sizes"] = {
            "source": {"width": source_width, "height": source_height},
            "anchor_requested": {"width": source_width * 4, "height": source_height * 4},
            "buffer_requested": {
                "width": final_width * buffer_ratio,
                "height": final_height * buffer_ratio,
            },
            "buffer_actual": {"width": buffer_width, "height": buffer_height},
            "final_requested": {"width": final_width, "height": final_height},
        }
        return [latent_node, 0], next_id + 6

    def _append_seed_resize_noise(
        self,
        workflow: dict[str, dict],
        next_id: int,
        target_latent_ref: list[str | int],
        settings: dict,
        mapping: WorkflowMapping,
    ) -> tuple[list[str | int], int, int]:
        seed = int(settings.get("seed") if settings.get("seed") is not None else -1)
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            settings["seed"] = seed
        source_width = int(settings["seed_resize_from_w"])
        source_height = int(settings["seed_resize_from_h"])
        batch_size = int(settings.get("batch_size", 1))
        source_latent_node = str(next_id)
        target_noise_node = str(next_id + 1)
        source_noise_node = str(next_id + 2)
        composite_node = str(next_id + 3)
        next_id += 4
        workflow[source_latent_node] = {
            "class_type": mapping.class_type("latent", "EmptyLatentImage"),
            "inputs": {
                "width": source_width,
                "height": source_height,
                "batch_size": batch_size,
            },
        }
        shared_seed_inputs = {
            "seed_prompt": str(seed),
            "enable_additional": False,
            "additional_seed": 0,
            "additional_strength": 0.0,
            "noise_mode": "GPU(=A1111)",
            "initial_batch_seed_mode": "incremental",
        }
        workflow[target_noise_node] = {
            "class_type": mapping.class_type("seed_explorer", "SeedExplorer //Inspire"),
            "inputs": {"latent": target_latent_ref, **shared_seed_inputs},
        }
        workflow[source_noise_node] = {
            "class_type": mapping.class_type("seed_explorer", "SeedExplorer //Inspire"),
            "inputs": {"latent": [source_latent_node, 0], **shared_seed_inputs},
        }
        workflow[composite_node] = {
            "class_type": mapping.class_type("composite_noise", "CompositeNoise //Inspire"),
            "inputs": {
                "destination": [target_noise_node, 0],
                "source": [source_noise_node, 0],
                "mode": "center",
                "x": 0,
                "y": 0,
            },
        }
        settings["seed_resize_runtime"] = "inspire_center_composite_noise"
        settings["seed_resize_noise_mode"] = "GPU(=A1111)"
        settings["seed_resize_source_size"] = {
            "width": source_width,
            "height": source_height,
        }
        settings["seed_resize_target_size"] = {
            "width": int(settings["width"]),
            "height": int(settings["height"]),
        }
        return [composite_node, 0], seed, next_id

    def _ksampler_advanced_inspire_node(
        self,
        settings: dict,
        model_ref: list[str | int],
        positive_ref: list[str | int],
        negative_ref: list[str | int],
        latent_ref: list[str | int],
        noise_ref: list[str | int],
        mapping: WorkflowMapping,
        *,
        seed: int,
        start_at_step: int,
        end_at_step: int,
        return_with_leftover_noise: bool,
    ) -> dict:
        return {
            "_meta": {"asteria_stage": settings.get("_prompt_trace_stage", "base")},
            "class_type": mapping.class_type(
                "ksampler_advanced_inspire",
                "KSamplerAdvanced //Inspire",
            ),
            "inputs": {
                "model": model_ref,
                "add_noise": True,
                "noise_seed": seed,
                "steps": settings["steps"],
                "cfg": settings["cfg_scale"],
                "sampler_name": self._sampler_name(settings.get("sampler", "Euler a")),
                "scheduler": self._scheduler_name(settings.get("scheduler", "Automatic")),
                "positive": positive_ref,
                "negative": negative_ref,
                "latent_image": latent_ref,
                "start_at_step": start_at_step,
                "end_at_step": end_at_step,
                **_inspire_noise_inputs(settings, seed),
                "return_with_leftover_noise": return_with_leftover_noise,
                "batch_seed_mode": "incremental",
                "variation_seed": 0,
                "variation_strength": 0.0,
                "noise_opt": noise_ref,
            },
        }

    def _hires_is_latent(self, settings: dict) -> bool:
        return upscaler_option(settings.get("hires_upscaler")).kind == "latent"

    def _hires_uses_upscale_model(self, settings: dict) -> bool:
        return upscaler_option(settings.get("hires_upscaler")).kind == "model"

    def _latent_upscale_method(self, settings: dict) -> str:
        return upscaler_option(settings.get("hires_upscaler")).runtime_name

    def _image_upscale_method(self, settings: dict) -> str:
        option = upscaler_option(settings.get("hires_upscaler"))
        return option.runtime_name if option.kind == "image" else "lanczos"

    def _upscale_model_name(self, settings: dict) -> str:
        model_name = upscaler_model_name(settings.get("hires_upscaler"), settings.get("hires_upscale_model"))
        if model_name is None:
            raise RuntimeErrorResponse(f"Hires upscaler does not use an upscale model: {settings.get('hires_upscaler')}")
        return model_name

    def _append_image_hires_upscale(
        self,
        workflow: dict[str, dict],
        next_id: int,
        image_ref: list[str | int],
        settings: dict,
        mapping: WorkflowMapping,
    ) -> tuple[list[str | int], int]:
        if self._hires_uses_upscale_model(settings):
            loader_node = str(next_id)
            model_upscale_node = str(next_id + 1)
            resize_node = str(next_id + 2)
            target_width, target_height = self._hires_target_dimensions(settings)
            workflow[loader_node] = {
                "class_type": mapping.class_type("upscale_model_loader", "UpscaleModelLoader"),
                "inputs": {"model_name": self._upscale_model_name(settings)},
            }
            workflow[model_upscale_node] = {
                "class_type": mapping.class_type("image_upscale_with_model", "ImageUpscaleWithModel"),
                "inputs": {"upscale_model": [loader_node, 0], "image": image_ref},
            }
            workflow[resize_node] = {
                "class_type": mapping.class_type("image_scale", "ImageScale"),
                "inputs": {
                    "image": [model_upscale_node, 0],
                    "upscale_method": "lanczos",
                    "width": target_width,
                    "height": target_height,
                    "crop": "disabled",
                },
            }
            return [resize_node, 0], next_id + 3

        resize_dimensions = self._hires_resize_dimensions(settings)
        upscale_node = str(next_id)
        if resize_dimensions is not None:
            workflow[upscale_node] = {
                "class_type": mapping.class_type("image_scale", "ImageScale"),
                "inputs": {
                    "image": image_ref,
                    "upscale_method": self._image_upscale_method(settings),
                    "width": resize_dimensions[0],
                    "height": resize_dimensions[1],
                    "crop": "disabled",
                },
            }
        else:
            workflow[upscale_node] = {
                "class_type": mapping.class_type("image_scale_by", "ImageScaleBy"),
                "inputs": {
                    "image": image_ref,
                    "upscale_method": self._image_upscale_method(settings),
                    "scale_by": settings.get("hires_scale", 2.0),
                },
            }
        return [upscale_node, 0], next_id + 1

    def _hires_sampler(self, settings: dict) -> str:
        sampler = settings.get("hires_sampler")
        if sampler and sampler != "Use same sampler":
            return sampler
        return settings["sampler"]

    def _hires_scheduler(self, settings: dict) -> str:
        scheduler = settings.get("hires_scheduler")
        if scheduler and scheduler != "Use same scheduler":
            return scheduler
        return settings["scheduler"]

    def _hires_resize_dimensions(self, settings: dict) -> tuple[int, int] | None:
        resize_width = int(settings.get("hires_resize_width") or 0)
        resize_height = int(settings.get("hires_resize_height") or 0)
        width = int(settings.get("width") or 0)
        height = int(settings.get("height") or 0)
        if resize_width <= 0 and resize_height <= 0:
            return None
        if resize_width > 0 and resize_height > 0:
            return self._round_to_step(resize_width), self._round_to_step(resize_height)
        if width <= 0 or height <= 0:
            return None
        if resize_width > 0:
            return self._round_to_step(resize_width), self._round_to_step(resize_width * (height / width))
        return self._round_to_step(resize_height * (width / height)), self._round_to_step(resize_height)

    def _hires_target_dimensions(self, settings: dict) -> tuple[int, int]:
        resize_dimensions = self._hires_resize_dimensions(settings)
        if resize_dimensions is not None:
            return resize_dimensions
        scale = float(settings.get("hires_scale") or 1)
        return self._round_to_step(int(settings["width"]) * scale), self._round_to_step(int(settings["height"]) * scale)

    def _round_to_step(self, value: int | float, step: int = 8) -> int:
        return int(float(value) / step + 0.5) * step

    def _model_name(self, value: str, model_dir_name: str) -> str:
        return comfy_model_name(
            value,
            model_dir_name,
            model_root=self.model_root,
            external_roots=self.external_roots,
        )

    def _comfy_prompt_text(self, text: str, embeddings: list[str]) -> str:
        result = text
        for name in sorted({str(item) for item in embeddings if item}, key=len, reverse=True):
            pattern = rf"(?<!embedding:)(?<![\w.-]){re.escape(name)}(?![\w.-])"
            result = re.sub(pattern, f"embedding:{name}", result)
        return result

    def _lora_loader_node(
        self,
        lora: dict,
        model_ref: list[str | int],
        clip_ref: list[str | int],
        mapping: WorkflowMapping,
        settings: dict,
    ) -> dict:
        strength_model = lora.get("weight", 1.0)
        strength_clip = lora.get("clip_weight")
        if strength_clip is None:
            strength_clip = strength_model
        block_vector = self._lora_block_vector(lora)
        if block_vector and lora.get("hierarchy_mode") != "disabled":
            return {
                "class_type": mapping.class_type("lora_block_weight_loader", LORA_BLOCK_WEIGHT_CLASS),
                "inputs": {
                    "model": model_ref,
                    "clip": clip_ref,
                    "category_filter": "All",
                    "lora_name": self._model_name(lora["name"], "loras"),
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                    "inverse": False,
                    "seed": int(settings.get("seed") or 0),
                    "A": 4.0,
                    "B": 1.0,
                    "preset": lora.get("block_weight_preset") or "Preset",
                    "block_vector": block_vector,
                    "bypass": False,
                },
                "_meta": {
                    "lora_runtime": "inspire_block_weight",
                    "hierarchy_mode": lora.get("hierarchy_mode") or "chain",
                    "block_weight_preset": lora.get("block_weight_preset") or "",
                },
            }
        return {
            "class_type": mapping.class_type("lora_loader", "LoraLoader"),
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": self._model_name(lora["name"], "loras"),
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            },
        }

    def _anima_lora_loader_node(
        self,
        lora: dict,
        model_ref: list[str | int],
        clip_ref: list[str | int],
        mapping: WorkflowMapping,
        settings: dict,
        *,
        remap_key: str = "anima_remap",
    ) -> dict:
        remap = lora.get(remap_key)
        if (
            remap_key == "hires_anima_remap"
            and not isinstance(remap, dict)
            and isinstance(lora.get("anima_remap"), dict)
        ):
            raise RuntimeErrorResponse(
                "A verified Anima LoRA remap receipt is required for the Hires target model."
            )
        if not isinstance(remap, dict):
            return self._lora_loader_node(
                lora,
                model_ref,
                clip_ref,
                mapping,
                settings,
            )
        strength_model = lora.get("weight", 1.0)
        strength_clip = lora.get("clip_weight")
        if strength_clip is None:
            strength_clip = strength_model
        return {
            "class_type": mapping.class_type(
                "anima_lora_remap",
                "AsteriaAnimaLoraRemap",
            ),
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": self._model_name(lora["name"], "loras"),
                "expected_lora_sha256": str(remap.get("source_sha256") or ""),
                "source_layout_id": str(remap.get("source_layout_id") or ""),
                "target_layout_id": str(remap.get("target_layout_id") or ""),
                "mode": str(remap.get("mode") or "preserve"),
                "mapping_revision": int(remap.get("mapping_revision") or 0),
                "mapping_hash": str(remap.get("mapping_hash") or ""),
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            },
        }

    def _lora_block_vector(self, lora: dict) -> str:
        block_weight = str(lora.get("block_weight") or "").strip()
        if block_weight:
            return block_weight
        return str(lora.get("block_weight_preset") or "").strip()

    def _append_face_detailer_node(
        self,
        workflow: dict[str, dict],
        next_id: int,
        image_ref: list[str | int],
        model_ref: list[str | int],
        clip_ref: list[str | int],
        vae_ref: list[str | int],
        positive_ref: list[str | int],
        negative_ref: list[str | int],
        mapping: WorkflowMapping,
        settings: dict,
        unit: dict,
        embeddings: list[str] | None = None,
    ) -> tuple[list[str | int], int]:
        detector_node = str(next_id)
        detailer_node = str(next_id + 1)
        next_id += 2
        detailer_positive_ref = positive_ref
        detailer_negative_ref = negative_ref
        embeddings = embeddings or []
        if unit.get("prompt"):
            positive_node = str(next_id)
            next_id += 1
            if mapping.profile == "anima":
                workflow[positive_node] = {
                    "class_type": mapping.class_type("conditioning", "TextEncodeQwenImageEdit"),
                    "inputs": {"prompt": str(unit["prompt"]), "clip": clip_ref, "vae": vae_ref},
                }
            else:
                workflow[positive_node] = {
                    "class_type": mapping.class_type("clip_text_encode", "CLIPTextEncode"),
                    "inputs": {"text": self._comfy_prompt_text(unit["prompt"], embeddings), "clip": clip_ref},
                }
            detailer_positive_ref = [positive_node, 0]
        if unit.get("negative_prompt"):
            negative_node = str(next_id)
            next_id += 1
            if mapping.profile == "anima":
                workflow[negative_node] = {
                    "class_type": mapping.class_type("conditioning", "TextEncodeQwenImageEdit"),
                    "inputs": {"prompt": str(unit["negative_prompt"]), "clip": clip_ref, "vae": vae_ref},
                }
            else:
                workflow[negative_node] = {
                    "class_type": mapping.class_type("clip_text_encode", "CLIPTextEncode"),
                    "inputs": {
                        "text": self._comfy_prompt_text(unit["negative_prompt"], embeddings),
                        "clip": clip_ref,
                    },
                }
            detailer_negative_ref = [negative_node, 0]
        workflow[detector_node] = {
            "class_type": mapping.class_type("detailer_detector", "UltralyticsDetectorProvider"),
            "inputs": {"model_name": detailer_detector_model(unit.get("detector", "face_yolo"))},
        }
        workflow[detailer_node] = {
            "class_type": mapping.class_type("face_detailer", "FaceDetailer"),
            "_meta": {"asteria_stage": "detailer"},
            "inputs": {
                "image": image_ref,
                "model": model_ref,
                "clip": clip_ref,
                "vae": vae_ref,
                "guide_size": unit.get("guide_size", 512),
                "guide_size_for": True,
                "max_size": unit.get("max_size", 1024),
                "seed": settings["seed"] if settings.get("seed", -1) >= 0 else random.randint(0, 2**32 - 1),
                "steps": unit.get("steps") or settings["steps"],
                "cfg": unit.get("cfg_scale") or settings["cfg_scale"],
                "sampler_name": self._sampler_name(unit.get("sampler") or settings["sampler"]),
                "scheduler": self._scheduler_name(unit.get("scheduler") or settings["scheduler"]),
                "positive": detailer_positive_ref,
                "negative": detailer_negative_ref,
                "denoise": unit.get("denoising_strength", 0.4),
                "feather": unit.get("mask_blur", 4),
                "noise_mask": unit.get("inpaint_only_masked", True),
                "force_inpaint": True,
                "bbox_threshold": unit.get("confidence", 0.3),
                "bbox_dilation": unit.get("bbox_dilation", 10),
                "bbox_crop_factor": unit.get("bbox_crop_factor", 3.0),
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": [detector_node, 0],
                "wildcard": "",
                "cycle": 1,
                "noise_mask_feather": 20,
            },
        }
        mark_stage(settings, 'detailer', images=[detailer_node, 0])
        return [detailer_node, 0], next_id

    def _sampler_name(self, value: str) -> str:
        return sampler_runtime_name(value)

    def _scheduler_name(self, value: str) -> str:
        return scheduler_runtime_name(value)

    def _controlnet_mode(self, unit: dict) -> str:
        raw_mode = str(unit["control_mode"]) if "control_mode" in unit else "Balanced"
        mapped = {
            "Balanced": "balanced",
            "My prompt is more important": "prompt_priority",
            "ControlNet is more important": "controlnet_priority",
        }.get(raw_mode)
        if mapped is None:
            raise RuntimeErrorResponse(f"unsupported_controlnet_mode:{raw_mode}")
        return mapped

    def _controlnet_strength(self, unit: dict) -> float:
        return float(unit.get("weight", 1.0))

    def _controlnet_union_type(self, unit: dict) -> str:
        if not _is_union_controlnet_model(unit.get("model")):
            return ""
        mode = str(unit.get("preprocessor") or "none").lower()
        return CONTROLNET_UNION_TYPES.get(mode, "auto")

    def _controlnet_preprocessor_workflow_node(
        self,
        unit: dict,
        image_ref: list[str | int],
        settings: dict,
    ) -> dict | None:
        mode = str(unit.get("preprocessor") or "none").lower()
        if mode == "none" or unit.get("preprocessor_runtime") != "comfy_node":
            return None
        node_class = str(unit.get("preprocessor_node_class") or COMFY_CONTROLNET_PREPROCESSOR_NODES.get(mode) or "")
        if not node_class:
            return None
        inputs = {"image": image_ref}
        inputs.update(CONTROLNET_COMFY_PREPROCESSOR_DEFAULTS.get(mode, {}))
        resolution = unit.get("preprocessor_resolution")
        if resolution is None:
            resolution = max(int(settings.get("width") or 512), int(settings.get("height") or 512))
        inputs["resolution"] = int(resolution)
        return {
            "class_type": node_class,
            "inputs": inputs,
        }

class ComfyGenerationRuntime:
    def __init__(
        self,
        client: ComfyClient,
        workflow_builder: ComfyTxt2ImgWorkflowBuilder | None = None,
        model_root: Path | None = None,
        external_roots: tuple[ExternalModelRoot, ...] = (),
        comfy_output_dir: Path | None = None,
        topaz_executable: Path | None = None,
        topaz_processor: Callable[..., dict[str, str | list[str]]] = process_topaz_image,
        admission_service: ComfyAdmissionService = comfy_admission,
        admission_timeout_seconds: float | None = 30.0,
        image_checker: Callable | None = None,
    ) -> None:
        self.client = client
        self.workflow_builder = workflow_builder or ComfyTxt2ImgWorkflowBuilder(
            model_root=model_root,
            external_roots=external_roots,
        )
        self.comfy_output_dir = Path(comfy_output_dir) if comfy_output_dir else None
        self.topaz_executable = topaz_executable
        self.topaz_processor = topaz_processor
        self.admission_service = admission_service
        self.admission_timeout_seconds = admission_timeout_seconds
        self.image_checker = image_checker

    def generate_txt2img(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_txt2img_admitted(request, output_dir)

    def _generate_txt2img_admitted(self, request: dict, output_dir: Path) -> dict:
        request.setdefault("settings", {})["mode"] = "txt2img"
        self._ensure_controlnet_mode_runtime(request)
        self._apply_runtime_node_fallbacks(request)
        if self._uses_image_check(request):
            return self._generate_checked(request, output_dir)
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
        self._prepare_spatial_runtime_masks(request)
        return self._generate(request, output_dir)

    def generate_img2img(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_img2img_admitted(request, output_dir)

    def _generate_img2img_admitted(self, request: dict, output_dir: Path) -> dict:
        request.setdefault("settings", {})["mode"] = "img2img"
        self._ensure_controlnet_mode_runtime(request)
        self._apply_runtime_node_fallbacks(request)
        self._prepare_init_image(request)
        if self._uses_image_check(request):
            return self._generate_checked(request, output_dir)
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
        self._prepare_spatial_runtime_masks(request)
        return self._generate(request, output_dir)

    def generate_inpaint(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_inpaint_admitted(request, output_dir)

    def _generate_inpaint_admitted(self, request: dict, output_dir: Path) -> dict:
        settings = request.setdefault("settings", {})
        settings["mode"] = "inpaint"
        self._ensure_controlnet_mode_runtime(request)
        only_masked = str(settings.get("inpaint_area") or "whole_picture").strip().lower() == "only_masked"
        if only_masked:
            self._ensure_supported_only_masked_runtime_request(request)
        self._apply_runtime_node_fallbacks(request)
        if only_masked:
            self._prepare_only_masked_inpaint(request, output_dir / "_inpaint_regions")
        else:
            self._prepare_init_image(request)
            self._prepare_mask_image(request)
        if self._uses_image_check(request):
            result = self._generate_checked(request, output_dir)
            if only_masked:
                settings["width"], settings["height"] = _size_tuple(settings["inpaint_full_size"])
            return result
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
        self._prepare_spatial_runtime_masks(request)
        result = self._generate(request, output_dir)
        if only_masked:
            full_size = _size_tuple(settings.get("inpaint_full_size"))
            if full_size is None:
                raise RuntimeErrorResponse(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
            settings["width"], settings["height"] = full_size
        return result

    @staticmethod
    def _uses_image_check(request: dict) -> bool:
        operation = request.get("_image_check_operation")
        return operation is not None and operation.settings.enabled

    def _generate_checked(self, request: dict, output_dir: Path) -> dict:
        from app.services.image_check_runtime import run_checked_generation
        try:
            return run_checked_generation(self, request, output_dir)
        except (OSError, ValueError) as exc:
            raise RuntimeErrorResponse(str(exc)) from exc

    def _ensure_supported_only_masked_runtime_request(self, request: dict) -> None:
        settings = request.get("settings", {})
        supported = (
            request.get("profile") == "sdxl"
            and str(settings.get("masked_content") or "original").strip().lower() == "original"
            and not settings.get("hires_enabled")
            and not settings.get("tiling")
        )
        if not supported:
            raise RuntimeErrorResponse(api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING)

    def _prepare_only_masked_inpaint(self, request: dict, output_dir: Path) -> None:
        settings = request.setdefault("settings", {})
        source_path = settings.get("init_image_path")
        mask_path = settings.get("mask_image_path")
        try:
            prepared = prepare_only_masked_region(
                str(source_path or ""),
                str(mask_path or ""),
                output_dir,
                mask_blur=int(settings.get("mask_blur") or 0),
                requested_size=(int(settings.get("width") or 0), int(settings.get("height") or 0)),
            )
        except InpaintRegionError as exc:
            raise RuntimeErrorResponse(exc.code) from None
        except OSError:
            raise RuntimeErrorResponse("inpaint_only_masked_preparation_failed") from None
        settings.update(prepared.metadata)
        settings["source_init_image_path"] = str(prepared.source_path)
        settings["source_mask_image_path"] = str(prepared.mask_path)
        settings["processed_init_image_path"] = str(prepared.init_crop_path)
        settings["processed_mask_image_path"] = str(prepared.comfy_mask_crop_path)
        settings["inpaint_canonical_source_path"] = str(prepared.canonical_source_path)
        settings["inpaint_positive_mask_crop_path"] = str(prepared.positive_mask_crop_path)
        settings["width"], settings["height"] = prepared.region.size
        settings["init_image_path"] = self.client.upload_image(prepared.init_crop_path)
        settings["mask_image_path"] = self.client.upload_image(prepared.comfy_mask_crop_path)
        settings["inpaint_runtime"] = "sdxl_only_masked_crop_composite"
        settings["inpaint_composited"] = False

    def generate_remaster(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_remaster_admitted(request, output_dir)

    def _generate_remaster_admitted(self, request: dict, output_dir: Path) -> dict:
        settings = request.setdefault("settings", {})
        settings["mode"] = "remaster"
        self._ensure_controlnet_mode_runtime(request)
        remaster = settings.get("remaster", {})
        if isinstance(remaster, dict):
            for key in (
                "target_scale",
                "anchor_upscaler",
                "buffer_ratio",
                "latent_interpolation",
            ):
                if key in remaster:
                    settings[f"remaster_{key}"] = remaster[key]
            for key in ("denoise", "steps", "sampler", "scheduler"):
                if key in remaster:
                    settings[key] = remaster[key]
        self._apply_runtime_node_fallbacks(request)
        self._prepare_init_image(request)
        source_size = settings.get("source_init_size", {})
        settings["remaster_source_width"] = int(source_size.get("width") or settings.get("width") or 0)
        settings["remaster_source_height"] = int(source_size.get("height") or settings.get("height") or 0)
        return self._generate(request, output_dir)

    @contextmanager
    def _admission_scope(self, request: dict):
        lease_token = request.get("_comfy_admission_lease_token")
        try:
            if isinstance(lease_token, str) and lease_token:
                with self.admission_service.borrow(lease_token, required_kind="xyz"):
                    yield
                return
            with self.admission_service.acquire(
                "operation",
                timeout_seconds=self.admission_timeout_seconds,
            ):
                yield
        except ComfyCleanupPendingError as exc:
            raise RuntimeErrorResponse(api_errors.COMFY_CLEANUP_PENDING) from exc
        except (ComfyAdmissionTimeoutError, ComfyAdmissionError) as exc:
            raise RuntimeErrorResponse(api_errors.COMFY_ADMISSION_BUSY) from exc

    def _apply_runtime_node_fallbacks(self, request: dict) -> None:
        if request.get("profile") != "sdxl":
            return
        settings = request.setdefault("settings", {})
        if "use_inspire_ksampler" in settings:
            return
        try:
            nodes = self.client.object_info()
        except (AttributeError, RuntimeErrorResponse):
            return
        if INSPIRE_KSAMPLER_CLASS in nodes:
            settings["use_inspire_ksampler"] = True
            settings["inspire_ksampler_runtime"] = "available"
        else:
            settings["use_inspire_ksampler"] = False
            settings["inspire_ksampler_runtime"] = "fallback_standard_ksampler"

    def _ensure_controlnet_mode_runtime(self, request: dict) -> None:
        settings = request.get("settings", {})
        if request.get("profile") != "sdxl" or not enabled_priority_controlnet_units(
            settings.get("controlnet_units")
        ):
            return
        try:
            object_info = self.client.object_info()
        except (AttributeError, RuntimeErrorResponse) as exc:
            raise RuntimeErrorResponse(api_errors.MISSING_CONTROLNET_MODE_RUNTIME_NODES) from exc
        repository = getattr(self.workflow_builder, "mapping_repository", None)
        if not isinstance(repository, WorkflowMappingRepository):
            repository = WorkflowMappingRepository(AppConfig().mapping_dir)
        try:
            contract = resolve_controlnet_mode_contract(repository, "sdxl")
            status = controlnet_mode_contract_status(object_info, contract)
        except ValueError as exc:
            raise RuntimeErrorResponse(api_errors.MISSING_CONTROLNET_MODE_RUNTIME_NODES) from exc
        if not status.ready:
            raise RuntimeErrorResponse(api_errors.MISSING_CONTROLNET_MODE_RUNTIME_NODES)

    def _raise_if_generation_cancelled(
        self,
        request: dict,
        unpublished_paths: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> None:
        cancel_requested = request.get("cancel_requested")
        runtime_stop_confirmed = request.get("runtime_stop_confirmed")
        if not (
            (callable(cancel_requested) and cancel_requested())
            or (callable(runtime_stop_confirmed) and runtime_stop_confirmed())
        ):
            return
        self._discard_unpublished_paths(unpublished_paths)
        raise RuntimeCancellationConfirmed("generation_cancelled")

    @staticmethod
    def _discard_unpublished_paths(
        unpublished_paths: list[str | Path] | tuple[str | Path, ...],
    ) -> None:
        for path in unpublished_paths:
            _unlink_safely(Path(path))

    def _generate(self, request: dict, output_dir: Path) -> dict:
        try:
            if output_dir.exists() and not output_dir.is_dir():
                raise RuntimeErrorResponse(f"ComfyUI output path is not a directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        except RuntimeErrorResponse:
            raise
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to prepare ComfyUI output directory: {output_dir}") from exc
        if "_prepared_workflow" not in request:
            self._prepare_control_images(request, output_dir / "controlnet")
            self._prepare_anima_lllite_images(request)
        self._raise_if_generation_cancelled(request)
        batch_count = max(1, int(request.get("settings", {}).get("batch_count", 1)))
        base_seed = int(request.get("settings", {}).get("seed", -1))
        progress_callback = request.get("progress_callback")
        self._publish_generation_progress(progress_callback, 0, batch_count)
        prompt_ids: list[str] = []
        generated_seeds: list[int] = []
        sampler_noise_runtime: list[dict] = []
        output_paths: list[str] = []
        primary_outputs: list[dict[str, str | int]] = []
        enhancement_stages: list[dict] = []
        image_outputs: list[dict] = []
        stage_paths: list[str] = []
        batch_state = {"index": 0}
        with self._preview_listener(
            Path(request.get('_preview_output_dir') or output_dir / '_previews'),
            self._preview_callback(progress_callback, batch_count, batch_state),
            self._step_callback(progress_callback, batch_count, batch_state),
        ):
            for batch_index in range(batch_count):
                self._raise_if_generation_cancelled(request, [*output_paths, *stage_paths])
                batch_state["index"] = batch_index
                iteration_request = deepcopy(request)
                settings = iteration_request.setdefault("settings", {})
                settings["batch_count"] = 1
                seed = self._iteration_seed(base_seed, batch_index)
                settings["seed"] = seed
                generated_seeds.append(seed)
                workflow = (deepcopy(iteration_request["_prepared_workflow"])
                            if "_prepared_workflow" in iteration_request
                            else self.workflow_builder.build(iteration_request))
                self._merge_builder_runtime_settings(request, iteration_request)
                prepare_outputs = getattr(self.workflow_builder, "prepare_output_writes", None)
                if callable(prepare_outputs):
                    prepare_outputs(workflow, request["profile"])
                from app.services.comfy_prompt_trace import instrument_prompt_encoders, record_prompt_execution
                if request.get("_record_prompt_inputs", True):
                    instrument_prompt_encoders(workflow, capture_words=bool(request.get('_word_maps')))
                prompt_id = self.client.queue_prompt(workflow)
                prompt_ids.append(prompt_id)
                from app.services.comfy_sampler_receipts import sampler_noise_receipts
                sampler_noise_runtime.extend(sampler_noise_receipts(workflow, prompt_id))
                runtime_job_callback = iteration_request.get("runtime_job_callback")
                if callable(runtime_job_callback):
                    runtime_job_callback(prompt_id)
                runtime_stop_confirmed = iteration_request.get("runtime_stop_confirmed")
                if callable(runtime_stop_confirmed):
                    try:
                        history = self.client.wait_for_history(
                            prompt_id,
                            timeout_seconds=None,
                            runtime_stop_confirmed=runtime_stop_confirmed,
                        )
                    except RuntimeCancellationConfirmed:
                        self._discard_unpublished_paths([*output_paths, *stage_paths])
                        raise
                else:
                    history = self.client.wait_for_history(prompt_id)
                self._raise_if_generation_cancelled(request, [*output_paths, *stage_paths])
                if request.get("_record_prompt_inputs", True):
                    record_prompt_execution(iteration_request, workflow, history, prompt_id,
                                            batch_index=batch_index + int(request.get('_batch_index_offset', 0)))
                    self._merge_builder_runtime_settings(request, iteration_request)
                spatial_receipts = self._spatial_receipts_from_history(
                    history,
                    prompt_id,
                )
                if iteration_request.get("_spatial_runtime_context") is not None:
                    spatial_stage = str(
                        iteration_request.get("_spatial_runtime_stage") or "base"
                    )
                    if spatial_stage == "base":
                        if not spatial_receipts:
                            raise RuntimeErrorResponse(
                                "Spatial prompt execution receipt is missing."
                            )
                        self._validate_spatial_receipts_against_context(
                            iteration_request["_spatial_runtime_context"],
                            spatial_receipts,
                        )
                        if any(
                            receipt.get("initial_noise_seed") != seed
                            for receipt in spatial_receipts
                        ):
                            raise RuntimeErrorResponse(
                                "Spatial prompt execution seed does not match the sampler seed."
                            )
                        recorded_receipts = request.setdefault("settings", {}).setdefault(
                            "spatial_execution_receipts",
                            [],
                        )
                        recorded_receipts.extend(
                            {
                                **receipt,
                                "runtime_job_id": prompt_id,
                                "batch_index": batch_index,
                                "seed": seed,
                            }
                            for receipt in spatial_receipts
                        )
                        settings["spatial_execution_receipts"] = deepcopy(
                            recorded_receipts
                        )
                    elif spatial_receipts:
                        raise RuntimeErrorResponse(
                            "Spatial prompt execution receipt is unexpected outside the base stage."
                        )
                images = self._images_from_history(history, prompt_id)
                primary_node_ids = {
                    str(node_id) for node_id in settings.get("primary_output_node_ids", [])
                }
                stage_nodes = settings.get('_enhancement_output_nodes', {})
                stage_image_indices: dict[str, int] = {}
                output_image_indices: dict[str, int] = {}
                for image in images:
                    filename = self._safe_filename(image.get("filename", "comfy-output.png"))
                    path = output_dir / f"{uuid4().hex}_{filename}"
                    stage_meta = stage_nodes.get(str(image.get('node_id', '')))
                    if stage_meta:
                        directory = output_dir / '_stages'
                        directory.mkdir(exist_ok=True)
                        path = directory / path.name
                    image_bytes = self.client.download_image(image)
                    try:
                        path.write_bytes(image_bytes)
                    except OSError as exc:
                        raise RuntimeErrorResponse(f"Failed to save ComfyUI output image: {path}") from exc
                    if stage_meta:
                        stage_paths.append(str(path))
                        self._raise_if_generation_cancelled(request, [*output_paths, *stage_paths])
                        key = str(image['node_id'])
                        index = stage_image_indices.get(key, 0)
                        stage_image_indices[key] = index + 1
                        enhancement_stages.append(stage_receipt(path, stage_meta['kind'], seed=seed,
                            batch_index=batch_index, image_index=index, runtime_job_id=prompt_id, ordinal=stage_meta['ordinal']))
                        if self.comfy_output_dir is not None:
                            self._remove_local_comfy_intermediate(image)
                        continue
                    output_paths.append(str(path))
                    image_node = str(image.get('node_id', ''))
                    image_index = output_image_indices.get(image_node, 0)
                    output_image_indices[image_node] = image_index + 1
                    image_outputs.append({'path': str(path), 'batch_index': batch_index,
                                          'image_index': image_index, 'seed': seed})
                    self._raise_if_generation_cancelled(request, [*output_paths, *stage_paths])
                    if settings.get("mode") == "inpaint":
                        if settings.get("inpaint_runtime") == "sdxl_only_masked_crop_composite":
                            try:
                                compose_only_masked_output(
                                    path,
                                    canonical_source_path=str(settings.get("inpaint_canonical_source_path") or ""),
                                    positive_mask_crop_path=str(
                                        settings.get("inpaint_positive_mask_crop_path") or ""
                                    ),
                                    crop_rect=settings.get("inpaint_crop_rect"),
                                )
                                settings["inpaint_composited"] = True
                            except InpaintRegionError as exc:
                                settings["inpaint_composited"] = False
                                self._merge_builder_runtime_settings(request, iteration_request)
                                raise RuntimeErrorResponse(exc.code) from None
                        else:
                            self._composite_inpaint_output(path, settings)
                        self._merge_builder_runtime_settings(request, iteration_request)
                    if settings.get("tiling"):
                        self._apply_tiling_postprocess(path, settings)
                        self._merge_builder_runtime_settings(request, iteration_request)
                    if str(image.get("node_id", "")) in primary_node_ids:
                        primary_outputs.append(
                            {
                                "path": str(path),
                                "seed": seed,
                                "output_index": len(primary_outputs),
                                "runtime_job_id": prompt_id,
                            }
                        )
                    if self.comfy_output_dir is not None and not self._remove_local_comfy_intermediate(image):
                        settings["comfy_intermediate_cleanup_warning"] = True
                self._publish_generation_progress(
                    progress_callback,
                    batch_index + 1,
                    batch_count,
                    output_paths[-1] if output_paths else "",
                )
        self._raise_if_generation_cancelled(request, [*output_paths, *stage_paths])
        if not output_paths:
            raise RuntimeErrorResponse(f"ComfyUI completed without image outputs: {', '.join(prompt_ids)}")
        request.setdefault("settings", {})["generated_seeds"] = generated_seeds
        request["settings"]["sampler_noise_runtime"] = sampler_noise_runtime
        if base_seed < 0 and generated_seeds:
            request["settings"]["seed"] = generated_seeds[0]
        return {
            "status": "completed",
            "output_paths": output_paths,
            "runtime_adapter": "comfy_workflow",
            "runtime_job_id": prompt_ids[0],
            "runtime_job_ids": prompt_ids,
            "comfy_prompt_id": prompt_ids[0],
            "comfy_prompt_ids": prompt_ids,
            "generated_seeds": generated_seeds,
            "primary_outputs": primary_outputs,
            "enhancement_stages": enhancement_stages,
            "image_outputs": image_outputs,
        }

    def _remove_local_comfy_intermediate(self, image: dict) -> bool:
        if self.comfy_output_dir is None or str(image.get("type") or "output") != "output":
            return False
        filename = str(image.get("filename") or "")
        subfolder = str(image.get("subfolder") or "")
        if not filename or Path(filename).name != filename:
            return False
        try:
            root = self.comfy_output_dir.resolve()
            candidate = (root / subfolder / filename).resolve()
            if candidate.parent != root and root not in candidate.parents:
                return False
            if not candidate.is_file():
                return False
            candidate.unlink()
        except OSError:
            return False
        return True

    def _merge_builder_runtime_settings(self, target_request: dict, iteration_request: dict) -> None:
        target_settings = target_request.setdefault("settings", {})
        iteration_settings = iteration_request.get("settings", {})
        for key in (
            "restore_faces_runtime",
            "face_restoration_model",
            "tiling_runtime",
            "tiling_blend_band",
            "tiling_blend_axes",
            "inpaint_runtime",
            "inpaint_composited",
            "inpaint_only_masked_padding",
            "inpaint_crop_rect",
            "inpaint_crop_size",
            "inpaint_full_size",
            "inpaint_canonical_source_path",
            "inpaint_positive_mask_crop_path",
            "processed_init_image_path",
            "processed_mask_image_path",
            "primary_output_node_ids",
            "remaster_source_image_node_id",
            "remaster_node_ids",
            "remaster_stage_sizes",
            "seed_resize_runtime",
            "seed_resize_noise_mode",
            "seed_resize_source_size",
            "seed_resize_target_size",
            "sampler_noise_runtime",
            "prompt_execution",
            "prompt_execution_truncated",
        ):
            if key in iteration_settings:
                target_settings[key] = iteration_settings[key]
        if iteration_settings.get("mode") == "remaster":
            target_settings["width"] = iteration_settings.get("width", target_settings.get("width"))
            target_settings["height"] = iteration_settings.get("height", target_settings.get("height"))
        self._merge_controlnet_unit_runtime_settings(target_settings, iteration_settings)

    def _merge_controlnet_unit_runtime_settings(self, target_settings: dict, iteration_settings: dict) -> None:
        target_units = target_settings.get("controlnet_units")
        iteration_units = iteration_settings.get("controlnet_units")
        if not isinstance(target_units, list) or not isinstance(iteration_units, list):
            return
        runtime_keys = (
            "controlnet_union_type",
            "controlnet_union_type_node_id",
            "preprocessor_workflow_node_id",
            "control_mode_normalized",
            "control_mode_mapping",
            "control_mode_runtime",
            "control_mode_base_multiplier",
            "control_mode_uncond_multiplier",
            "effective_strength",
        )
        for target_unit, iteration_unit in zip(target_units, iteration_units, strict=False):
            if not isinstance(target_unit, dict) or not isinstance(iteration_unit, dict):
                continue
            for key in runtime_keys:
                if key in iteration_unit:
                    target_unit[key] = iteration_unit[key]

    def _apply_tiling_postprocess(self, path: Path, settings: dict) -> None:
        try:
            with Image.open(path) as image:
                tiled = self._blend_tiling_edges(image.convert("RGB"))
                tiled.save(path)
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to apply tiling postprocess to ComfyUI output image: {path}") from exc
        settings["tiling_runtime"] = "postprocess_seam_blend"
        settings["tiling_blend_band"] = self._tiling_blend_band(tiled.size)
        settings["tiling_blend_axes"] = ["x", "y"]

    def _blend_tiling_edges(self, image: Image.Image) -> Image.Image:
        tiled = image.copy()
        width, height = tiled.size
        band = self._tiling_blend_band((width, height))
        if band <= 0:
            return tiled
        pixels = tiled.load()
        for offset in range(band):
            alpha = (band - offset) / band
            right_x = width - 1 - offset
            for y in range(height):
                left = pixels[offset, y]
                right = pixels[right_x, y]
                average = tuple((left[channel] + right[channel]) // 2 for channel in range(3))
                pixels[offset, y] = tuple(round(left[channel] * (1 - alpha) + average[channel] * alpha) for channel in range(3))
                pixels[right_x, y] = tuple(round(right[channel] * (1 - alpha) + average[channel] * alpha) for channel in range(3))
        for offset in range(band):
            alpha = (band - offset) / band
            bottom_y = height - 1 - offset
            for x in range(width):
                top = pixels[x, offset]
                bottom = pixels[x, bottom_y]
                average = tuple((top[channel] + bottom[channel]) // 2 for channel in range(3))
                pixels[x, offset] = tuple(round(top[channel] * (1 - alpha) + average[channel] * alpha) for channel in range(3))
                pixels[x, bottom_y] = tuple(round(bottom[channel] * (1 - alpha) + average[channel] * alpha) for channel in range(3))
        return tiled

    def _tiling_blend_band(self, size: tuple[int, int]) -> int:
        width, height = size
        if width < 8 or height < 8:
            return 0
        return max(4, min(256, min(width, height) // 8))

    def _composite_inpaint_output(
        self,
        path: Path,
        settings: dict,
        base_image_path: Path | None = None,
    ) -> None:
        init_path = base_image_path or Path(str(settings.get("source_init_image_path", "")))
        source_mask_path = settings.get("source_mask_image_path")
        processed_mask_path = settings.get("processed_mask_image_path")
        mask_path = Path(str(source_mask_path or processed_mask_path or ""))
        using_processed_mask = not source_mask_path and bool(processed_mask_path)
        if (
            not _path_exists_safely(init_path)
            or not _path_is_file_safely(init_path)
            or not _path_exists_safely(mask_path)
            or not _path_is_file_safely(mask_path)
        ):
            settings["inpaint_composited"] = False
            return
        try:
            with Image.open(path).convert("RGB") as generated:
                with Image.open(init_path).convert("RGB") as init_image:
                    with Image.open(mask_path) as mask_image:
                        init_resized = init_image.resize(generated.size, Image.Resampling.LANCZOS) if init_image.size != generated.size else init_image
                        if mask_image.mode == "RGBA":
                            alpha = mask_image.getchannel("A")
                            if using_processed_mask:
                                mask = ImageOps.invert(alpha)
                            else:
                                luminance = ImageOps.grayscale(mask_image.convert("RGB"))
                                mask = (
                                    luminance
                                    if alpha.getextrema() == (255, 255)
                                    else Image.composite(luminance, alpha, alpha)
                                )
                        else:
                            mask = ImageOps.grayscale(mask_image.convert("RGB"))
                        if mask.size != generated.size:
                            mask = mask.resize(generated.size, Image.Resampling.LANCZOS)
                        composited = Image.composite(generated, init_resized, mask)
                        composited.save(path)
            settings["inpaint_composited"] = True
        except OSError:
            settings["inpaint_composited"] = False

    def _generate_topaz_hires(self, request: dict, output_dir: Path, *, first_result: dict | None = None) -> dict:
        settings = request.setdefault("settings", {})
        source_mode = str(settings.get("mode") or "txt2img")
        settings["topaz_hires_source_mode"] = source_mode
        if self.topaz_executable is None or not _path_is_file_safely(self.topaz_executable):
            settings["topaz_hires_runtime"] = "skipped_missing_executable"
            self._prepare_spatial_runtime_masks(request)
            return self._generate(request, output_dir)

        first_request = deepcopy(request)
        first_settings = first_request.setdefault("settings", {})
        first_settings["hires_enabled"] = False
        first_settings["mode"] = source_mode
        first_pass_count = max(1, int(settings.get("batch_size", 1))) * max(
            1,
            int(settings.get("batch_count", 1)),
        )
        first_settings["batch_size"] = 1
        first_settings["batch_count"] = first_pass_count
        if request.get("profile") == "anima":
            first_settings["restore_faces"] = False
            first_settings["detailer_units"] = []
        settings["topaz_first_pass_batch_size"] = 1
        settings["topaz_first_pass_batch_count"] = first_pass_count
        self._prepare_spatial_runtime_masks(first_request)
        if first_result is None:
            first_result = self._generate(first_request, output_dir)
        self._merge_builder_runtime_settings(request, first_request)
        # Topaz uses a copied request; retain the base execution evidence in the
        # owning request so history and final-image sidecars receive it too.
        if first_settings.get("spatial_execution_receipts"):
            settings["spatial_execution_receipts"] = deepcopy(
                first_settings["spatial_execution_receipts"]
            )
        first_paths = [Path(path) for path in first_result["output_paths"]]
        enhancement_stages = list(first_result.get('enhancement_stages', []))
        self._raise_if_generation_cancelled(request, [*first_paths, *(s['path'] for s in enhancement_stages)])
        generated_seeds = list(first_result.get("generated_seeds", []))
        if len(first_paths) != len(generated_seeds):
            raise RuntimeErrorResponse(
                "Topaz Hires first-pass output count did not match generated seed count: "
                f"{len(first_paths)} outputs, {len(generated_seeds)} seeds"
            )
        settings["topaz_output_seeds"] = generated_seeds

        topaz_settings = TopazSettings(**request.get("topaz_settings", {}))
        settings["topaz"] = topaz_settings.model_dump()
        topaz_paths: list[Path] = []
        topaz_logs: list[str] = []
        topaz_sizes: list[tuple[int, int]] = []
        topaz_source_sizes: list[tuple[int, int] | None] = []
        topaz_size_evidence: list[dict[str, object]] = []
        for source_index, path in enumerate(first_paths):
            self._raise_if_generation_cancelled(request, [*first_paths, *topaz_paths, *(s['path'] for s in enhancement_stages)])
            source_size = self._image_size(path)
            if source_size is None:
                raise RuntimeErrorResponse(f"Could not read Topaz Hires first-pass image size: {path}")
            target_size = self._hires_target_size_for_source(settings, source_size)
            result = self.topaz_processor(
                executable=self.topaz_executable,
                input_path=path,
                output_dir=output_dir,
                settings=topaz_settings,
                target_size=target_size,
            )
            topaz_path = Path(str(result["output_path"]))
            if not _path_exists_safely(topaz_path) or not _path_is_file_safely(topaz_path):
                raise RuntimeErrorResponse(f"Topaz Hires did not produce a valid image file: {topaz_path}")
            topaz_paths.append(topaz_path)
            if settings.get('record_enhancement_stages'):
                identity = dict(seed=generated_seeds[source_index], batch_index=source_index,
                                image_index=0, runtime_job_id=first_result.get('runtime_job_ids',
                                    [first_result['runtime_job_id']])[source_index])
                enhancement_stages.append(copy_stage(path, output_dir, 'before_topaz', ordinal=900, **identity))
                enhancement_stages.append(copy_stage(topaz_path, output_dir, 'topaz', ordinal=1000, **identity))
            self._raise_if_generation_cancelled(request, [*first_paths, *topaz_paths, *(s['path'] for s in enhancement_stages)])
            source_size = _size_tuple(result.get("source_size")) or source_size
            output_size = _size_tuple(result.get("output_size")) or self._image_size(topaz_path) or target_size
            topaz_source_sizes.append(source_size)
            topaz_sizes.append(output_size)
            topaz_size_evidence.append(
                {
                    "source_path": str(path),
                    "output_path": str(topaz_path),
                    "source_size": _size_dict(source_size),
                    "requested_target_size": _size_dict(target_size),
                    "output_size": _size_dict(output_size),
                }
            )
            topaz_logs.extend(_log_messages(result.get("logs")))
        if topaz_logs:
            settings["topaz_logs"] = topaz_logs
        requested_target_sizes = [evidence["requested_target_size"] for evidence in topaz_size_evidence]
        settings["topaz_requested_target_sizes"] = requested_target_sizes
        if requested_target_sizes:
            settings["topaz_requested_target_size"] = requested_target_sizes[0]
        settings["topaz_source_sizes"] = [_size_dict(size) for size in topaz_source_sizes]
        settings["topaz_output_sizes"] = [_size_dict(size) for size in topaz_sizes]
        settings["topaz_actual_upscaled_sizes"] = [
            {"width": width, "height": height} for width, height in topaz_sizes
        ]
        settings["topaz_size_evidence"] = topaz_size_evidence

        final_paths: list[str] = []
        second_prompt_ids: list[str] = []
        topaz_image_outputs: list[dict] = []
        primary_outputs: list[dict[str, str | int]] = []
        for index, topaz_path in enumerate(topaz_paths):
            self._raise_if_generation_cancelled(
                request,
                [*first_paths, *topaz_paths, *(Path(path) for path in final_paths), *(s['path'] for s in enhancement_stages)],
            )
            second_request = deepcopy(request)
            second_request["_spatial_runtime_stage"] = "hires"
            second_settings = second_request.setdefault("settings", {})
            second_settings["_prompt_trace_stage"] = "hires"
            second_width, second_height = topaz_sizes[index]
            second_settings["mode"] = "img2img"
            second_settings["seed_resize_from_w"] = 0
            second_settings["seed_resize_from_h"] = 0
            second_settings["hires_enabled"] = False
            second_settings["init_image_path"] = self.client.upload_image(topaz_path)
            second_settings["source_init_image_path"] = str(topaz_path)
            second_settings["width"], second_settings["height"] = second_width, second_height
            second_settings["steps"] = settings.get("hires_steps") or settings["steps"]
            second_settings["cfg_scale"] = settings.get("hires_cfg_scale") or settings["cfg_scale"]
            second_settings["distilled_cfg_scale"] = (
                settings.get("hires_distilled_cfg_scale")
                or settings.get("distilled_cfg_scale")
                or second_settings["cfg_scale"]
            )
            second_settings["sampler"] = self._hires_sampler(settings)
            second_settings["scheduler"] = self._hires_scheduler(settings)
            second_settings["denoise"] = settings.get("hires_denoising_strength", 0.7)
            second_settings["noise_mode"] = _hires_noise_mode(settings)
            second_settings["use_inspire_ksampler"] = bool(settings.get("use_inspire_ksampler", True))
            second_settings["batch_size"] = 1
            second_settings["batch_count"] = 1
            second_settings["seed"] = generated_seeds[index]
            hires_checkpoint = _hires_checkpoint_setting(settings)
            if hires_checkpoint:
                if request.get("profile") in {"anima", "flux1", "flux2"}:
                    second_request.setdefault("model_slots", {})["diffusion_model"] = hires_checkpoint
                    second_settings["hires_checkpoint_runtime"] = "topaz_second_pass_diffusion_model_slot"
                    second_settings["hires_checkpoint_kind"] = "diffusion_model"
                    settings["hires_checkpoint_runtime"] = "topaz_second_pass_diffusion_model_slot"
                    settings["hires_checkpoint_kind"] = "diffusion_model"
                elif request.get("profile") == "sdxl":
                    second_request.setdefault("model_slots", {})["checkpoint"] = hires_checkpoint
                    second_settings["hires_checkpoint_runtime"] = "topaz_second_pass_model_slot"
            if settings.get("hires_prompt"):
                second_request["expanded_prompt"] = settings["hires_prompt"]
            if settings.get("hires_negative_prompt"):
                second_request["negative_prompt"] = settings["hires_negative_prompt"]

            spatial_context = second_request.get("_spatial_runtime_context")
            if spatial_context is not None:
                try:
                    second_request["_spatial_runtime_context"] = resize_spatial_runtime_stage(
                        spatial_context,
                        stage="hires",
                        width=second_width,
                        height=second_height,
                    )
                except SpatialGenerationContextError as exc:
                    raise RuntimeErrorResponse(exc.code) from exc
            self._prepare_spatial_runtime_masks(second_request)

            try:
                second_result = self._generate(second_request, output_dir)
            except RuntimeCancellationConfirmed:
                self._discard_unpublished_paths(
                    [*first_paths, *topaz_paths, *(Path(path) for path in final_paths), *(s['path'] for s in enhancement_stages)],
                )
                raise
            self._raise_if_generation_cancelled(
                request,
                [
                    *first_paths,
                    *topaz_paths,
                    *(Path(path) for path in final_paths),
                    *(Path(path) for path in second_result["output_paths"]),
                    *(s['path'] for s in enhancement_stages),
                    *(s['path'] for s in second_result.get('enhancement_stages', [])),
                ],
            )
            if request.get("profile") == "anima":
                self._merge_builder_runtime_settings(request, second_request)
            for trace_key in ("prompt_execution", "prompt_execution_truncated"):
                if trace_key in second_settings:
                    settings[trace_key] = deepcopy(second_settings[trace_key])
            if source_mode == "inpaint":
                composite_failed_paths: list[str] = []
                for final_path_value in second_result["output_paths"]:
                    final_path = Path(final_path_value)
                    self._composite_inpaint_output(final_path, settings, base_image_path=topaz_path)
                    if not settings.get("inpaint_composited"):
                        composite_failed_paths.append(str(final_path))
                settings.setdefault("topaz_inpaint_composite_base_paths", []).append(str(topaz_path))
                if composite_failed_paths:
                    settings["topaz_inpaint_composite_failed_paths"] = composite_failed_paths
                    raise RuntimeErrorResponse(
                        "Topaz Hires failed to composite inpaint second-pass outputs: "
                        + ", ".join(composite_failed_paths)
                    )
            final_paths.extend(second_result["output_paths"])
            topaz_image_outputs.extend({**item, 'batch_index': index, 'image_index': 0}
                                       for item in second_result.get('image_outputs', []))
            enhancement_stages.extend({**item, 'batch_index': index, 'image_index': 0,
                                       'ordinal': 2000 + item['ordinal']}
                                      for item in second_result.get('enhancement_stages', []))
            for primary_output in second_result.get("primary_outputs", []):
                primary_outputs.append(
                    {
                        **primary_output,
                        "output_index": len(primary_outputs),
                    }
                )
            second_prompt_ids.extend(second_result.get("runtime_job_ids", [second_result["runtime_job_id"]]))
            settings.setdefault("sampler_noise_runtime", []).extend(
                second_settings.get("sampler_noise_runtime", [])
            )

        settings["topaz_hires_runtime"] = "completed"
        settings["topaz_second_pass_runtime"] = {
            "sdxl": "sdxl_vae_img2img",
            "anima": "anima_vae_img2img",
            "flux1": "flux1_vae_img2img",
            "flux2": "flux2_vae_img2img",
        }.get(str(request.get("profile")), "vae_img2img")
        settings["topaz_source_output_paths"] = [str(path) for path in first_paths]
        settings["topaz_upscaled_paths"] = [str(path) for path in topaz_paths]
        settings["generated_seeds"] = generated_seeds
        if not settings.get("hires_keep_first_pass"):
            skipped_deletes = []
            for path in first_paths:
                if not _unlink_safely(path):
                    skipped_deletes.append(str(path))
            if skipped_deletes:
                settings["topaz_source_output_paths_delete_skipped"] = skipped_deletes
            else:
                settings["topaz_source_output_paths_deleted"] = True
        else:
            final_paths.extend(
                str(path) for path in first_paths if _path_exists_safely(path) and _path_is_file_safely(path)
            )
            final_paths.extend(
                str(path) for path in topaz_paths if _path_exists_safely(path) and _path_is_file_safely(path)
            )
        if not settings.get("hires_keep_first_pass"):
            skipped_deletes = []
            for path in topaz_paths:
                if not _unlink_safely(path):
                    skipped_deletes.append(str(path))
            if skipped_deletes:
                settings["topaz_upscaled_paths_delete_skipped"] = skipped_deletes
            else:
                settings["topaz_upscaled_paths_deleted"] = True
        prompt_ids = [*first_result.get("runtime_job_ids", [first_result["runtime_job_id"]]), *second_prompt_ids]
        topaz_image_outputs.extend({'path': str(path), 'batch_index': index, 'image_index': 0,
                                    'seed': generated_seeds[index]}
                                   for index, pair in enumerate(zip(first_paths, topaz_paths, strict=True))
                                   for path in pair if str(path) in final_paths)
        return {
            "status": "completed",
            "output_paths": final_paths,
            "runtime_adapter": "comfy_workflow",
            "runtime_job_id": prompt_ids[-1],
            "runtime_job_ids": prompt_ids,
            "comfy_prompt_id": prompt_ids[-1],
            "comfy_prompt_ids": prompt_ids,
            "generated_seeds": generated_seeds,
            "primary_outputs": primary_outputs,
            "enhancement_stages": enhancement_stages,
            "image_outputs": topaz_image_outputs,
        }

    def _uses_topaz_hires(self, request: dict) -> bool:
        settings = request.get("settings", {})
        return bool(settings.get("hires_enabled") and settings.get("hires_upscaler") == "Topaz Photo AI")

    def _image_size(self, path: Path) -> tuple[int, int] | None:
        try:
            with Image.open(path) as image:
                return image.size
        except OSError:
            return None

    def _hires_target_size(self, settings: dict) -> tuple[int, int]:
        resize_dimensions = self._hires_resize_dimensions(settings)
        if resize_dimensions is not None:
            return resize_dimensions
        return self._hires_scaled_dimensions(settings)

    def _hires_target_size_for_source(
        self,
        settings: dict,
        source_size: tuple[int, int],
    ) -> tuple[int, int]:
        resize_width = int(settings.get("hires_resize_width") or 0)
        resize_height = int(settings.get("hires_resize_height") or 0)
        source_width, source_height = source_size
        if resize_width > 0 and resize_height > 0:
            return self._round_to_step(resize_width), self._round_to_step(resize_height)
        if resize_width > 0:
            return self._round_to_step(resize_width), self._round_to_step(
                resize_width * (source_height / source_width)
            )
        if resize_height > 0:
            return self._round_to_step(resize_height * (source_width / source_height)), self._round_to_step(
                resize_height
            )
        scale = float(settings.get("hires_scale") or 1)
        return self._round_to_step(source_width * scale), self._round_to_step(source_height * scale)

    def _hires_sampler(self, settings: dict) -> str:
        sampler = settings.get("hires_sampler")
        if sampler and sampler != "Use same sampler":
            return sampler
        return settings["sampler"]

    def _hires_scheduler(self, settings: dict) -> str:
        scheduler = settings.get("hires_scheduler")
        if scheduler and scheduler != "Use same scheduler":
            return scheduler
        return settings["scheduler"]

    def _hires_resize_dimensions(self, settings: dict) -> tuple[int, int] | None:
        resize_width = int(settings.get("hires_resize_width") or 0)
        resize_height = int(settings.get("hires_resize_height") or 0)
        width = int(settings.get("width") or 0)
        height = int(settings.get("height") or 0)
        if resize_width <= 0 and resize_height <= 0:
            return None
        if resize_width > 0 and resize_height > 0:
            return self._round_to_step(resize_width), self._round_to_step(resize_height)
        if width <= 0 or height <= 0:
            return None
        if resize_width > 0:
            return self._round_to_step(resize_width), self._round_to_step(resize_width * (height / width))
        return self._round_to_step(resize_height * (width / height)), self._round_to_step(resize_height)

    def _hires_scaled_dimensions(self, settings: dict) -> tuple[int, int]:
        width = int(settings.get("width") or 0)
        height = int(settings.get("height") or 0)
        scale = float(settings.get("hires_scale") or 1)
        return self._round_to_step(width * scale), self._round_to_step(height * scale)

    def _round_to_step(self, value: int | float, step: int = 8) -> int:
        return int(float(value) / step + 0.5) * step

    def _iteration_seed(self, base_seed: int, batch_index: int) -> int:
        if base_seed >= 0:
            return base_seed + batch_index
        return random.randint(0, 2**32 - 1)

    def _publish_generation_progress(
        self,
        callback,
        current: int,
        total: int,
        preview_path: str = "",
        step: int | None = None,
        steps: int | None = None,
        phase: str = "",
    ) -> None:
        if not callable(callback):
            return
        try:
            callback(current, total, preview_path, step, steps, phase)
        except TypeError:
            try:
                callback(current, total, preview_path)
            except TypeError:
                callback(current, total)

    @contextmanager
    def _preview_listener(self, preview_dir: Path, callback, progress_callback=None):
        listener = getattr(self.client, "preview_listener", None)
        if listener is None:
            yield
            return
        with listener(preview_dir, callback, progress_callback):
            yield

    def _preview_callback(self, callback, total: int, batch_state: dict[str, int]):
        if not callable(callback):
            return None

        def publish(preview_path: str) -> None:
            self._publish_generation_progress(callback, batch_state.get("index", 0), total, preview_path, phase="sampling")

        return publish

    def _step_callback(self, callback, total: int, batch_state: dict[str, int]):
        if not callable(callback):
            return None

        def publish(step: int, steps: int) -> None:
            self._publish_generation_progress(
                callback,
                batch_state.get("index", 0),
                total,
                step=step,
                steps=steps,
                phase="sampling",
            )

        return publish

    def _prepare_init_image(self, request: dict) -> None:
        if request.get("profile") not in {"sdxl", "anima", "flux1", "flux2"}:
            raise RuntimeErrorResponse("ComfyUI img2img currently supports the sdxl, anima, flux1, and flux2 profiles.")
        settings = request.get("settings", {})
        image_path = settings.get("init_image_path")
        if not image_path:
            raise RuntimeErrorResponse("init_image_path is required for img2img.")
        path = Path(str(image_path))
        if not _path_exists_safely(path) or not _path_is_file_safely(path):
            raise RuntimeErrorResponse(f"Init image not found: {path}")
        try:
            with Image.open(path) as image:
                source_width, source_height = image.size
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to read init image dimensions: {path}") from exc
        if not (256 <= source_width <= 4096 and 256 <= source_height <= 4096):
            raise RuntimeErrorResponse("Init image dimensions must be between 256 and 4096 pixels.")
        alignment = 16 if request.get("profile") == "flux2" else 8
        effective_width = source_width // alignment * alignment
        effective_height = source_height // alignment * alignment
        settings["requested_base_size"] = {
            "width": int(settings.get("width") or 0),
            "height": int(settings.get("height") or 0),
        }
        settings["source_init_size"] = {"width": source_width, "height": source_height}
        settings["effective_base_size"] = {"width": effective_width, "height": effective_height}
        settings["base_size_runtime"] = "source_image_dimensions_aligned_to_vae"
        settings["width"] = effective_width
        settings["height"] = effective_height
        settings["source_init_image_path"] = str(path)
        settings["init_image_path"] = self.client.upload_image(path)

    def _prepare_spatial_runtime_masks(self, request: dict) -> None:
        context = request.get("_spatial_runtime_context")
        if context is None:
            return
        if isinstance(context, dict) and context.get("mask_transport") == "comfy_upload":
            return
        try:
            uploaded = upload_spatial_runtime_masks(
                context,
                upload=self.client.upload_image,
            )
        except SpatialGenerationContextError as exc:
            raise RuntimeErrorResponse(exc.code) from exc
        uploaded["mask_transport"] = "comfy_upload"
        request["_spatial_runtime_context"] = uploaded

    def _prepare_mask_image(self, request: dict) -> None:
        if request.get("profile") not in {"sdxl", "anima", "flux1", "flux2"}:
            raise RuntimeErrorResponse("ComfyUI inpaint currently supports the sdxl, anima, flux1, and flux2 profiles.")
        settings = request.get("settings", {})
        image_path = settings.get("mask_image_path")
        if not image_path:
            raise RuntimeErrorResponse("mask_image_path is required for inpaint.")
        path = Path(str(image_path))
        if not _path_exists_safely(path) or not _path_is_file_safely(path):
            raise RuntimeErrorResponse(f"Mask image not found: {path}")
        settings["source_mask_image_path"] = str(path)
        blur_radius = int(settings.get("mask_blur") or 0)
        normalized_path = self._normalize_inpaint_mask(path, blur_radius)
        if normalized_path != path:
            settings["processed_mask_image_path"] = str(normalized_path)
        settings["mask_image_path"] = self.client.upload_image(normalized_path)

    def _normalize_inpaint_mask(self, path: Path, blur_radius: int = 0) -> Path:
        output_path = path.with_name(f"{path.stem}_comfy_mask.png")
        try:
            with Image.open(path) as image:
                if image.mode == "RGBA":
                    alpha = image.getchannel("A")
                    rgb = image.convert("RGB")
                    luminance = ImageOps.grayscale(rgb)
                    alpha_extrema = alpha.getextrema()
                    if alpha_extrema == (255, 255):
                        mask = luminance
                    else:
                        mask = Image.composite(luminance, alpha, alpha)
                else:
                    mask = ImageOps.grayscale(image.convert("RGB"))
                if blur_radius > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                rgba = Image.new("RGBA", image.size, (255, 255, 255, 0))
                rgba.putalpha(ImageOps.invert(mask))
                rgba.save(output_path)
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to prepare inpaint mask: {path}") from exc
        return output_path

    def _prepare_control_images(self, request: dict, output_dir: Path) -> None:
        if request.get("profile") != "sdxl":
            return
        available_preprocessor_nodes = self._available_controlnet_preprocessor_nodes()
        for unit in request.get("settings", {}).get("controlnet_units", []):
            image_path = unit.get("image_path")
            if not image_path:
                continue
            path = Path(str(image_path))
            if not _path_exists_safely(path) or not _path_is_file_safely(path):
                raise RuntimeErrorResponse(f"Control image not found: {path}")
            preprocessor = str(unit.get("preprocessor") or "none").lower()
            node_class = COMFY_CONTROLNET_PREPROCESSOR_NODES.get(preprocessor)
            if preprocessor != "none" and node_class and node_class in available_preprocessor_nodes:
                processed_path = path
                unit["preprocessor_runtime"] = "comfy_node"
                unit["preprocessor_node_class"] = node_class
            else:
                try:
                    processed_path = preprocess_control_image(path, preprocessor, output_dir)
                except ControlNetPreprocessorError as exc:
                    raise RuntimeErrorResponse(str(exc)) from exc
                unit["preprocessor_runtime"] = "none" if preprocessor == "none" else "local_fallback"
            processed_path = self._prepare_controlnet_resize(processed_path, unit, request.get("settings", {}), output_dir)
            if processed_path != path:
                unit["source_image_path"] = str(path)
                unit["processed_image_path"] = str(processed_path)
            else:
                unit["source_image_path"] = str(path)
            unit["image_path"] = self.client.upload_image(processed_path)
            unit["uploaded_image_name"] = unit["image_path"]

    def _available_controlnet_preprocessor_nodes(self) -> set[str]:
        try:
            object_info = self.client.object_info()
        except (AttributeError, RuntimeErrorResponse):
            return set()
        if not isinstance(object_info, dict):
            return set()
        return set(object_info.keys())

    def _prepare_controlnet_resize(self, path: Path, unit: dict, settings: dict, output_dir: Path) -> Path:
        if not unit.get("pixel_perfect"):
            return path
        width = int(settings.get("width") or 0)
        height = int(settings.get("height") or 0)
        if width <= 0 or height <= 0:
            return path
        unit["pixel_perfect_target_size"] = {"width": width, "height": height}
        resized_path = output_dir / f"{path.stem}_pixel_perfect{path.suffix or '.png'}"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                mode = str(unit.get("resize_mode") or "Crop and Resize")
                if mode == "Just Resize":
                    resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
                elif mode == "Resize and Fill":
                    resized = ImageOps.pad(rgb, (width, height), method=Image.Resampling.LANCZOS, color=(0, 0, 0))
                else:
                    resized = ImageOps.fit(rgb, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                resized.save(resized_path)
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to prepare ControlNet pixel-perfect image: {path}") from exc
        return resized_path

    def _prepare_anima_lllite_images(self, request: dict) -> None:
        if request.get("profile") != "anima":
            return
        for unit in request.get("settings", {}).get("anima_lllite_units", []):
            if not unit.get("enabled"):
                continue
            image_path = unit.get("image_path")
            if not image_path:
                continue
            path = Path(str(image_path))
            if not _path_exists_safely(path) or not _path_is_file_safely(path):
                raise RuntimeErrorResponse(f"Anima LLLite image not found: {path}")
            unit["source_image_path"] = str(path)
            unit["image_path"] = self.client.upload_image(path)
            unit["uploaded_image_name"] = unit["image_path"]
            mask_path_value = unit.get("mask_image_path")
            if mask_path_value:
                mask_path = Path(str(mask_path_value))
                if not _path_exists_safely(mask_path) or not _path_is_file_safely(mask_path):
                    raise RuntimeErrorResponse(f"Anima LLLite mask not found: {mask_path}")
                unit["mask_source_image_path"] = str(mask_path)
                unit["mask_image_path"] = self.client.upload_image(mask_path)
                unit["uploaded_mask_name"] = unit["mask_image_path"]

    def _images_from_history(self, history: dict, prompt_id: str) -> list[dict]:
        outputs = history.get(prompt_id, {}).get("outputs", {})
        images = []
        for node_id, output in outputs.items():
            images.extend({**image, "node_id": str(node_id)} for image in output.get("images", []))
        return sorted(images, key=lambda image: "first_pass" in str(image.get("filename", "")).lower())

    @staticmethod
    def _validate_spatial_receipts_against_context(
        context: object,
        receipts: list[dict],
    ) -> None:
        try:
            if (
                not isinstance(context, dict)
                or context.get("schema_version")
                != "asteria.spatial-workflow-context.v1"
                or len(receipts) != 1
            ):
                raise ValueError("context")
            stages = context["stages"]
            stage = stages["base"]
            prompt_stage = context["prompt_plan"]["base"]
            profile_receipt = stage["profile_receipt"]
            profile_payload = json.loads(stage["profile_json"])
            mask_bundle = stage["mask_bundle"]
            ordered_masks = sorted(
                mask_bundle["targets"],
                key=lambda item: int(item["depth_index"]),
            )
            expected_target_ids = [item["target_id"] for item in ordered_masks]
            expected_field_hashes = [item["mask_sha256"] for item in ordered_masks]
            scheduled = prompt_stage.get("scheduled_bindings") or []
            expected_full_text_hash: str | list[str] = (
                [item["full_text_sha256"] for item in scheduled]
                if scheduled
                else prompt_stage["full_text_sha256"]
            )
            receipt = receipts[0]
            matches = (
                receipt.get("stage") == "base"
                and receipt.get("target_ids") == expected_target_ids
                and profile_receipt.get("target_ids") == expected_target_ids
                and receipt.get("profile_id") == profile_receipt.get("profile_id")
                and receipt.get("profile_revision")
                == profile_receipt.get("profile_revision")
                and receipt.get("profile_receipt_sha256")
                == profile_receipt.get("receipt_sha256")
                and receipt.get("profile_payload_sha256")
                == profile_payload.get("profile_payload_sha256")
                and receipt.get("mask_bundle_sha256")
                == spatial_runtime_mask_receipt_sha256(mask_bundle)
                and receipt.get("field_sha256") == expected_field_hashes
                and (
                    receipt.get("fragments_sha256")
                    == stage.get("fragments_sha256")
                    if receipt.get("schema_version")
                    == "asteria.spatial-node-execution.v6"
                    else receipt.get("full_text_sha256")
                    == expected_full_text_hash
                )
            )
            if receipt.get("schema_version") == "asteria.spatial-node-execution.v9":
                matches = matches and receipt.get("geometry_sha256") == canonical_runtime_sha256(stage.get("routing_geometry"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeErrorResponse(
                "Spatial prompt execution receipt does not match the submitted context."
            ) from error
        if not matches:
            raise RuntimeErrorResponse(
                "Spatial prompt execution receipt does not match the submitted context."
            )

    @staticmethod
    def _spatial_receipts_from_history(
        history: dict,
        prompt_id: str,
    ) -> list[dict]:
        outputs = history.get(prompt_id, {}).get("outputs", {})
        receipts: list[dict] = []
        for output in outputs.values():
            raw_receipts = output.get("asteria_spatial_receipts", [])
            if not isinstance(raw_receipts, list):
                raise RuntimeErrorResponse("Spatial prompt execution receipt is invalid.")
            for raw in raw_receipts:
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise RuntimeErrorResponse(
                        "Spatial prompt execution receipt is invalid."
                    ) from error
                if not isinstance(payload, dict):
                    raise RuntimeErrorResponse(
                        "Spatial prompt execution receipt is invalid."
                    )
                execution_sha256 = payload.get("execution_sha256")
                body = {
                    key: value
                    for key, value in payload.items()
                    if key != "execution_sha256"
                }
                target_ids = payload.get("target_ids")
                if payload.get("schema_version") == "asteria.spatial-node-execution.v9":
                    from app.services.spatial_native_receipt import validate_native_spatial_receipt
                    try:
                        validate_native_spatial_receipt(payload)
                    except (KeyError, TypeError, ValueError) as error:
                        raise RuntimeErrorResponse("Spatial prompt execution receipt is invalid.") from error
                    receipts.append(payload)
                    continue
                if payload.get("schema_version") == "asteria.spatial-node-execution.v6":
                    runtime = payload.get("runtime")
                    structure_id = payload.get("verified_structure_id")
                    expected_block_count = {
                        "sdxl_unet_v1": 24,
                        "anima_base_28": 28,
                        "anima_2_9b_40": 40,
                        "anima_3_8b_52": 52,
                    }.get(structure_id)
                    forbidden_counts = (
                        "target_latent_count",
                        "target_noise_prediction_count",
                        "target_denoised_output_count",
                        "target_only_attention_forward_count",
                        "global_only_attention_forward_count",
                        "spatial_output_blend_count",
                        "background_substitution_count",
                        "self_attention_spatial_mask_count",
                        "hires_self_attention_mask_count",
                        "negative_token_change_count",
                        "global_token_block_count",
                        "query_key_recompute_count",
                        "extra_model_evaluation_count",
                        "latent_update_count",
                        "candidate_search_count",
                    )
                    runtime_forbidden_counts = (
                        "extra_model_evaluation_count",
                        "target_only_attention_forward_count",
                        "spatial_output_blend_count",
                        "latent_update_count",
                    )
                    declared_layers = payload.get("cross_attention_layers")
                    applied_layers = (
                        runtime.get("cross_attention_layers")
                        if isinstance(runtime, dict)
                        else None
                    )
                    target_ranges = (
                        runtime.get("target_context_ranges")
                        if isinstance(runtime, dict)
                        else None
                    )
                    cross_count = (
                        runtime.get("cross_attention_mask_count")
                        if isinstance(runtime, dict)
                        else None
                    )
                    cross_min = runtime.get("cross_mask_min") if isinstance(runtime, dict) else None
                    cross_max = runtime.get("cross_mask_max") if isinstance(runtime, dict) else None
                    schedule_min = runtime.get("cross_schedule_min") if isinstance(runtime, dict) else None
                    schedule_max = runtime.get("cross_schedule_max") if isinstance(runtime, dict) else None
                    query_counts = runtime.get("cross_query_token_counts") if isinstance(runtime, dict) else None
                    key_counts = runtime.get("cross_key_token_counts") if isinstance(runtime, dict) else None
                    latent_shape = payload.get("latent_shape")
                    registered = payload.get("registered_patches")
                    isolated_context_length = payload.get("isolated_context_length")
                    aligned_value_count = (
                        runtime.get("aligned_value_patch_count")
                        if isinstance(runtime, dict)
                        else None
                    )
                    router_revision = (
                        runtime.get("router_revision")
                        if isinstance(runtime, dict)
                        else None
                    )
                    sdxl_v8 = payload.get("model_family") == "sdxl"
                    sdxl_v6_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-isolated-conditioning-v6"
                        and payload.get("adapter_id")
                        == "sdxl.isolated-conditioning.v6"
                        and payload.get("routing_mode")
                        == "isolated_context_cross_attention_mask"
                        and registered == ["attn2_replace", "attn1_replace"]
                        and router_revision == "asteria-isolated-conditioning-v6"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                    )
                    sdxl_v16_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-group-binding-vector-v16"
                        and payload.get("adapter_id")
                        == "sdxl.group-binding-vector.v16"
                        and payload.get("routing_mode")
                        == "full_prompt_group_binding_sparse_attention"
                        and registered == ["attn2_replace"]
                        and router_revision == "asteria-group-binding-vector-v16"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                    )
                    sdxl_v17_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-attribute-value-binding-v17"
                        and payload.get("adapter_id")
                        == "sdxl.attribute-value-binding.v17"
                        and payload.get("routing_mode")
                        == "full_prompt_attribute_value_sparse_attention"
                        and registered == ["attn2_patch", "attn2_replace"]
                        and router_revision == "asteria-attribute-value-binding-v17"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count > 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                    )
                    sdxl_v20_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-appended-appearance-v20-research"
                        and payload.get("adapter_id")
                        == "sdxl.appended-appearance.v20.research"
                        and payload.get("routing_mode")
                        == "single_attention_appended_multilayer_appearance_tokens"
                        and registered == ["attn2_replace", "attn1_replace"]
                        and router_revision
                        == "asteria-appended-appearance-v20-research"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                    )
                    sparse_grounding = (
                        runtime.get("sparse_grounding")
                        if isinstance(runtime, dict)
                        else None
                    )
                    sdxl_v21_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-sparse-gligen-mid-v21-research"
                        and payload.get("adapter_id")
                        == "sdxl.sparse-gligen-mid.v21.research"
                        and payload.get("routing_mode")
                        == "single_latent_sparse_grounding_with_isolated_cross_attention"
                        and registered == ["attn2_replace", "middle_patch"]
                        and router_revision
                        == "asteria-sparse-gligen-mid-v21-research"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                        and isinstance(sparse_grounding, dict)
                        and sparse_grounding.get("revision")
                        == "asteria-sparse-gligen-mid-v21-research"
                        and sparse_grounding.get("layers")
                        == (
                            [f"block:input:7:{index}" for index in range(10)]
                            + [f"block:input:8:{index}" for index in range(10)]
                            + [f"block:middle:0:{index}" for index in range(10)]
                            + [f"block:output:0:{index}" for index in range(10)]
                            + [f"block:output:1:{index}" for index in range(10)]
                            + [f"block:output:2:{index}" for index in range(10)]
                        )
                        and sparse_grounding.get("fuser_count") == 60
                        and isinstance(
                            sparse_grounding.get("layer_call_counts"), dict
                        )
                        and set(sparse_grounding["layer_call_counts"])
                        == set(sparse_grounding["layers"])
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 1
                            for value in sparse_grounding["layer_call_counts"].values()
                        )
                        and isinstance(sparse_grounding.get("call_count"), int)
                        and not isinstance(sparse_grounding.get("call_count"), bool)
                        and sparse_grounding["call_count"] >= 1
                        and isinstance(sparse_grounding.get("query_token_counts"), list)
                        and bool(sparse_grounding["query_token_counts"])
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and 1 <= value <= 1024
                            for value in sparse_grounding["query_token_counts"]
                        )
                        and sparse_grounding.get("target_count") == len(target_ids or [])
                        and sparse_grounding.get("object_capacity") == 30
                        and isinstance(sparse_grounding.get("weight_bytes"), int)
                        and not isinstance(sparse_grounding.get("weight_bytes"), bool)
                        and sparse_grounding["weight_bytes"] > 0
                        and all(
                            isinstance(sparse_grounding.get(name), (int, float))
                            and not isinstance(sparse_grounding.get(name), bool)
                            and math.isfinite(float(sparse_grounding[name]))
                            for name in ("gain", "schedule_min", "schedule_max")
                        )
                        and 0.25 <= float(sparse_grounding["gain"]) <= 16.0
                        and 0.0
                        < float(sparse_grounding["schedule_min"])
                        <= float(sparse_grounding["schedule_max"])
                        <= 1.0
                        and all(
                            sparse_grounding.get(name) == 0
                            for name in runtime_forbidden_counts
                        )
                    )
                    fused_adapter_layers = (
                        runtime.get("fused_adapter_layers")
                        if isinstance(runtime, dict)
                        else None
                    )
                    expected_ifadapter_layers = (
                        [f"block:middle:0:{index}" for index in range(4)]
                        + [
                            f"block:output:{block}:{index}"
                            for block in range(3)
                            for index in range(4)
                        ]
                    )
                    sdxl_v22_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-faithful-ifadapter-v22-research"
                        and payload.get("adapter_id")
                        == "sdxl.faithful-ifadapter.v22.research"
                        and payload.get("routing_mode")
                        == "single_latent_learned_instance_feature_attention"
                        and registered == ["attn2_replace"]
                        and router_revision
                        == "asteria-faithful-ifadapter-v22-research"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                        and fused_adapter_layers == expected_ifadapter_layers
                        and isinstance(
                            runtime.get("fused_adapter_layer_call_counts"), dict
                        )
                        and set(runtime["fused_adapter_layer_call_counts"])
                        == set(expected_ifadapter_layers)
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 1
                            for value in runtime[
                                "fused_adapter_layer_call_counts"
                            ].values()
                        )
                        and isinstance(
                            runtime.get("fused_adapter_attention_count"), int
                        )
                        and runtime["fused_adapter_attention_count"] >= 1
                        and runtime.get("fused_adapter_cbam_fusion_count")
                        == runtime["fused_adapter_attention_count"]
                        and isinstance(
                            runtime.get("fused_adapter_cfg_positive_row_count"), int
                        )
                        and runtime["fused_adapter_cfg_positive_row_count"] >= 1
                        and isinstance(
                            runtime.get(
                                "fused_adapter_cfg_unconditional_row_count"
                            ),
                            int,
                        )
                        and runtime[
                            "fused_adapter_cfg_unconditional_row_count"
                        ]
                        >= 1
                        and runtime.get("fused_adapter_aspect_ratio_aware") is True
                        and runtime.get("fused_adapter_max_objects") == 10
                        and isinstance(
                            runtime.get("fused_adapter_control_ratio"), (int, float)
                        )
                        and not isinstance(
                            runtime.get("fused_adapter_control_ratio"), bool
                        )
                        and 0.05
                        <= float(runtime["fused_adapter_control_ratio"])
                        <= 1.0
                    )
                    v20_self_count = (
                        runtime.get("self_attention_boundary_mask_count")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_layers = (
                        runtime.get("self_attention_layers")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_queries = (
                        runtime.get("self_query_token_counts")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_stats = (
                        runtime.get("self_boundary_stats")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_min = (
                        runtime.get("self_mask_min")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_max = (
                        runtime.get("self_mask_max")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_schedule_min = (
                        runtime.get("self_schedule_min")
                        if isinstance(runtime, dict)
                        else None
                    )
                    v20_self_schedule_max = (
                        runtime.get("self_schedule_max")
                        if isinstance(runtime, dict)
                        else None
                    )
                    valid_sparse_self = (
                        (sdxl_v6_research or sdxl_v20_research)
                        and isinstance(v20_self_count, int)
                        and not isinstance(v20_self_count, bool)
                        and v20_self_count >= 1
                        and v20_self_layers == ["block:middle:0:0"]
                        and isinstance(v20_self_queries, list)
                        and bool(v20_self_queries)
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and 1 <= value <= 1024
                            for value in v20_self_queries
                        )
                        and isinstance(v20_self_stats, dict)
                        and isinstance(
                            v20_self_stats.get("different_target_block_count"), int
                        )
                        and v20_self_stats["different_target_block_count"] > 0
                        and all(
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            for value in (
                                v20_self_min,
                                v20_self_max,
                                v20_self_schedule_min,
                                v20_self_schedule_max,
                            )
                        )
                        and -6.0 <= float(v20_self_min) <= float(v20_self_max) <= 0.0
                        and 0.0
                        < float(v20_self_schedule_min)
                        <= float(v20_self_schedule_max)
                        <= 1.0
                    )
                    sdxl_v19_research = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-alternating-conditioning-v19-research"
                        and payload.get("adapter_id")
                        == "sdxl.alternating-conditioning.v19.research"
                        and payload.get("routing_mode")
                        == "early_alternating_target_context"
                        and registered == ["attn2_replace"]
                        and router_revision
                        == "asteria-alternating-conditioning-v19-research"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                    )
                    valid_mechanism = (
                        sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-full-fragment-context-v8"
                        and payload.get("adapter_id")
                        == "sdxl.full-fragment-context.v8"
                        and payload.get("routing_mode")
                        == "full_fragment_context_sparse_attention_mask"
                        and registered == ["attn2_replace"]
                        and router_revision == "asteria-full-fragment-context-v8"
                        and isinstance(aligned_value_count, int)
                        and not isinstance(aligned_value_count, bool)
                        and aligned_value_count == 0
                        and isinstance(payload.get("adapter_context_length"), int)
                        and payload.get("adapter_context_length")
                        == isolated_context_length
                        and isolated_context_length % 77 == 0
                    ) or sdxl_v6_research or sdxl_v16_research or sdxl_v17_research or sdxl_v20_research or sdxl_v21_research or sdxl_v22_research or sdxl_v19_research or (
                        not sdxl_v8
                        and payload.get("adapter_revision")
                        == "asteria-isolated-conditioning-v6"
                        and payload.get("routing_mode")
                        == "isolated_context_cross_attention_mask"
                        and registered
                        == [
                            "attn2_patch",
                            "optimized_attention_override",
                            "diffusion_model.preprocess_text_embeds",
                        ]
                    )
                    valid_anima = payload.get("model_family") != "anima" or (
                        payload.get("token_position_source") == "T5_adapter_context"
                        and payload.get("context_length") == 512
                        and payload.get("padding_positions_preserved") is True
                    )
                    valid_ranges = (
                        isinstance(isolated_context_length, int)
                        and not isinstance(isolated_context_length, bool)
                        and isolated_context_length > 0
                        and isinstance(target_ranges, list)
                        and isinstance(target_ids, list)
                        and len(target_ranges) == len(target_ids)
                        and [item.get("target_id") for item in target_ranges if isinstance(item, dict)]
                        == target_ids
                        and all(
                            isinstance(item, dict)
                            and isinstance(item.get("start"), int)
                            and isinstance(item.get("end"), int)
                            and 0 <= item["start"] < item["end"]
                            <= isolated_context_length
                            for item in target_ranges
                        )
                    )
                    if (
                        payload.get("contract_version")
                        != ASTERIA_SPATIAL_NODE_CONTRACT_VERSION
                        or not valid_mechanism
                        or payload.get("stage") != "base"
                        or payload.get("hires_mode") != "base_only"
                        or expected_block_count is None
                        or payload.get("diffusion_block_count") != expected_block_count
                        or not isinstance(payload.get("model_slot_id"), str)
                        or not payload["model_slot_id"]
                        or not isinstance(payload.get("encoder_kind"), str)
                        or not payload["encoder_kind"]
                        or not all(
                            _valid_sha256(payload.get(name))
                            for name in (
                                "layout_fingerprint",
                                "structure_fingerprint",
                                "profile_receipt_sha256",
                                "profile_payload_sha256",
                                "mask_bundle_sha256",
                                "isolated_context_sha256",
                                "fragments_sha256",
                            )
                        )
                        or not _valid_sha256(execution_sha256)
                        or canonical_runtime_sha256(body) != execution_sha256
                        or not isinstance(target_ids, list)
                        or not target_ids
                        or len(target_ids) != len(set(target_ids))
                        or not isinstance(runtime, dict)
                        or any(payload.get(name) != 0 for name in forbidden_counts)
                        or any(runtime.get(name) != 0 for name in runtime_forbidden_counts)
                        or (
                            not valid_sparse_self
                            if (sdxl_v6_research or sdxl_v20_research)
                            else runtime.get("self_attention_boundary_mask_count") != 0
                        )
                        or payload.get("initial_noise_candidate_count") != 1
                        or isinstance(payload.get("initial_noise_seed"), bool)
                        or not isinstance(payload.get("initial_noise_seed"), int)
                        or not isinstance(declared_layers, list)
                        or not declared_layers
                        or not isinstance(applied_layers, list)
                        or not applied_layers
                        or not all(
                            applied in declared_layers
                            or any(
                                applied.startswith(f"{declared}:")
                                for declared in declared_layers
                            )
                            for applied in applied_layers
                        )
                        or isinstance(cross_count, bool)
                        or not isinstance(cross_count, int)
                        or cross_count < 1
                        or not isinstance(query_counts, list)
                        or not query_counts
                        or not isinstance(key_counts, list)
                        or not key_counts
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in (*query_counts, *key_counts)
                        )
                        or not valid_ranges
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in (
                                cross_min,
                                cross_max,
                                schedule_min,
                                schedule_max,
                                payload.get("cross_penalty_cap"),
                            )
                        )
                        or not -float(payload["cross_penalty_cap"])
                        <= float(cross_min)
                        <= float(cross_max)
                        <= float(payload["cross_penalty_cap"])
                        or (
                            router_revision == "asteria-full-fragment-context-v8"
                            and float(cross_max) > 0.0
                        )
                        or not 0.0 < float(schedule_min) <= float(schedule_max) <= 1.0
                        or not isinstance(payload.get("field_sha256"), list)
                        or len(payload["field_sha256"]) != len(target_ids)
                        or not all(_valid_sha256(value) for value in payload["field_sha256"])
                        or not isinstance(latent_shape, list)
                        or not latent_shape
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in latent_shape
                        )
                        or not valid_anima
                    ):
                        raise RuntimeErrorResponse(
                            "Spatial prompt execution receipt is invalid."
                        )
                    receipts.append(payload)
                    continue
                if payload.get("schema_version") == "asteria.spatial-node-execution.v5":
                    runtime = payload.get("runtime")
                    forbidden_counts = (
                        "target_latent_count",
                        "target_noise_prediction_count",
                        "target_denoised_output_count",
                        "target_only_attention_forward_count",
                        "global_only_attention_forward_count",
                        "spatial_output_blend_count",
                        "background_substitution_count",
                        "hires_self_attention_mask_count",
                        "negative_token_change_count",
                        "global_token_block_count",
                        "query_key_recompute_count",
                        "extra_model_evaluation_count",
                        "latent_update_count",
                        "candidate_search_count",
                    )
                    runtime_forbidden_counts = (
                        "target_only_attention_forward_count",
                        "global_only_attention_forward_count",
                        "spatial_output_blend_count",
                        "background_substitution_count",
                        "target_latent_count",
                        "target_noise_prediction_count",
                        "target_denoised_output_count",
                        "hires_self_attention_mask_count",
                        "negative_token_change_count",
                        "global_token_block_count",
                        "query_key_recompute_count",
                        "extra_model_evaluation_count",
                        "latent_update_count",
                        "candidate_search_count",
                    )
                    declared_cross_layers = payload.get("cross_attention_layers")
                    declared_self_layers = payload.get("self_attention_layers")
                    applied_cross_layers = (
                        runtime.get("cross_attention_layers")
                        if isinstance(runtime, dict)
                        else None
                    )
                    applied_self_layers = (
                        runtime.get("self_attention_layers")
                        if isinstance(runtime, dict)
                        else None
                    )
                    cross_count = (
                        runtime.get("cross_attention_boundary_mask_count")
                        if isinstance(runtime, dict)
                        else None
                    )
                    self_count = (
                        runtime.get("self_attention_boundary_mask_count")
                        if isinstance(runtime, dict)
                        else None
                    )
                    cross_schedule_min = (
                        runtime.get("cross_schedule_min")
                        if isinstance(runtime, dict)
                        else None
                    )
                    cross_schedule_max = (
                        runtime.get("cross_schedule_max")
                        if isinstance(runtime, dict)
                        else None
                    )
                    self_schedule_min = (
                        runtime.get("self_schedule_min")
                        if isinstance(runtime, dict)
                        else None
                    )
                    self_schedule_max = (
                        runtime.get("self_schedule_max")
                        if isinstance(runtime, dict)
                        else None
                    )
                    self_stats = runtime.get("self_boundary_stats") if isinstance(runtime, dict) else None
                    cross_cap = payload.get("cross_penalty_cap")
                    self_cap = payload.get("self_penalty_cap")
                    cross_min = runtime.get("cross_mask_min") if isinstance(runtime, dict) else None
                    cross_max = runtime.get("cross_mask_max") if isinstance(runtime, dict) else None
                    self_min = runtime.get("self_mask_min") if isinstance(runtime, dict) else None
                    self_max = runtime.get("self_mask_max") if isinstance(runtime, dict) else None
                    latent_shape = payload.get("latent_shape")
                    full_text_sha256 = payload.get("full_text_sha256")
                    structure_id = payload.get("verified_structure_id")
                    expected_block_count = {
                        "sdxl_unet_v1": 24,
                        "anima_base_28": 28,
                        "anima_2_9b_40": 40,
                        "anima_3_8b_52": 52,
                    }.get(structure_id)
                    cross_query_tokens = runtime.get("cross_query_token_counts") if isinstance(runtime, dict) else None
                    self_query_tokens = runtime.get("self_query_token_counts") if isinstance(runtime, dict) else None
                    declared_self_token_counts = payload.get(
                        "self_attention_token_counts"
                    )
                    self_token_limit = payload.get("self_attention_token_limit")
                    target_token_indices = runtime.get("target_token_indices") if isinstance(runtime, dict) else None
                    sigma_values = (
                        runtime.get("stage_sigma_start") if isinstance(runtime, dict) else None,
                        runtime.get("stage_sigma_end") if isinstance(runtime, dict) else None,
                        runtime.get("cross_sigma_start") if isinstance(runtime, dict) else None,
                        runtime.get("cross_sigma_end") if isinstance(runtime, dict) else None,
                        runtime.get("self_sigma_start") if isinstance(runtime, dict) else None,
                        runtime.get("self_sigma_end") if isinstance(runtime, dict) else None,
                    )
                    valid_full_text_hash = _valid_sha256(full_text_sha256) or (
                        isinstance(full_text_sha256, list)
                        and bool(full_text_sha256)
                        and all(_valid_sha256(value) for value in full_text_sha256)
                    )
                    valid_anima = payload.get("model_family") != "anima" or (
                        payload.get("token_position_source") == "T5_adapter_context"
                        and payload.get("context_length") == 512
                        and payload.get("padding_positions_preserved") is True
                    )
                    if (
                        payload.get("contract_version")
                        != ASTERIA_SPATIAL_NODE_CONTRACT_VERSION
                        or payload.get("adapter_revision")
                        != "asteria-cross-self-boundary-v5"
                        or payload.get("routing_mode")
                        != "cross_self_boundary_attention_mask"
                        or payload.get("stage") != "base"
                        or payload.get("hires_mode") != "base_only"
                        or not isinstance(payload.get("model_slot_id"), str)
                        or not payload["model_slot_id"]
                        or not isinstance(payload.get("encoder_kind"), str)
                        or not payload["encoder_kind"]
                        or expected_block_count is None
                        or payload.get("diffusion_block_count") != expected_block_count
                        or not _valid_sha256(payload.get("layout_fingerprint"))
                        or not _valid_sha256(payload.get("structure_fingerprint"))
                        or not _valid_sha256(execution_sha256)
                        or canonical_runtime_sha256(body) != execution_sha256
                        or not isinstance(target_ids, list)
                        or not target_ids
                        or len(target_ids) != len(set(target_ids))
                        or not isinstance(runtime, dict)
                        or any(payload.get(name) != 0 for name in forbidden_counts)
                        or any(runtime.get(name) != 0 for name in runtime_forbidden_counts)
                        or payload.get("initial_noise_candidate_count") != 1
                        or runtime.get("initial_noise_candidate_count") != 1
                        or isinstance(payload.get("initial_noise_seed"), bool)
                        or not isinstance(payload.get("initial_noise_seed"), int)
                        or payload.get("registered_patches") != [
                            "attn1_patch",
                            "attn2_patch",
                            "optimized_attention_override",
                        ]
                        or not isinstance(declared_cross_layers, list)
                        or not declared_cross_layers
                        or not isinstance(declared_self_layers, list)
                        or not declared_self_layers
                        or not isinstance(applied_cross_layers, list)
                        or not applied_cross_layers
                        or not set(applied_cross_layers).issubset(set(declared_cross_layers))
                        or not isinstance(applied_self_layers, list)
                        or (len(target_ids) > 1 and not applied_self_layers)
                        or not set(applied_self_layers).issubset(set(declared_self_layers))
                        or isinstance(cross_count, bool)
                        or not isinstance(cross_count, int)
                        or cross_count < 1
                        or isinstance(self_count, bool)
                        or not isinstance(self_count, int)
                        or self_count < (1 if len(target_ids) > 1 else 0)
                        or runtime.get("cross_original_attention_call_count") != cross_count
                        or runtime.get("self_original_attention_call_count") != self_count
                        or not isinstance(cross_query_tokens, list)
                        or not cross_query_tokens
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in cross_query_tokens
                        )
                        or not isinstance(self_query_tokens, list)
                        or (self_count > 0 and not self_query_tokens)
                        or not isinstance(declared_self_token_counts, list)
                        or not declared_self_token_counts
                        or len(declared_self_token_counts)
                        != len(set(declared_self_token_counts))
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in declared_self_token_counts
                        )
                        or isinstance(self_token_limit, bool)
                        or not isinstance(self_token_limit, int)
                        or self_token_limit < 1
                        or self_token_limit > 65536
                        or any(
                            value > self_token_limit
                            for value in declared_self_token_counts
                        )
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            or value > self_token_limit
                            for value in self_query_tokens
                        )
                        or not isinstance(target_token_indices, list)
                        or len(target_token_indices) != len(target_ids)
                        or any(
                            not isinstance(indices, list)
                            or not indices
                            or any(
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                for index in indices
                            )
                            for indices in target_token_indices
                        )
                        or len(
                            [index for indices in target_token_indices for index in indices]
                        )
                        != len(
                            set(index for indices in target_token_indices for index in indices)
                        )
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in sigma_values
                        )
                        or float(sigma_values[0]) <= float(sigma_values[1])
                        or not float(sigma_values[0]) >= float(sigma_values[2]) > float(sigma_values[3]) >= float(sigma_values[1])
                        or not float(sigma_values[0]) >= float(sigma_values[4]) > float(sigma_values[5]) >= float(sigma_values[1])
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in (
                                cross_schedule_min,
                                cross_schedule_max,
                                *(
                                    (self_schedule_min, self_schedule_max)
                                    if self_count
                                    else ()
                                ),
                            )
                        )
                        or not 0.0 < float(cross_schedule_min)
                        <= float(cross_schedule_max)
                        <= 1.0
                        or (self_count and not 0.0 < float(self_schedule_min) <= float(self_schedule_max) <= 1.0)
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            for value in (
                                cross_cap,
                                self_cap,
                                cross_min,
                                cross_max,
                                self_min,
                                self_max,
                            )
                        )
                        or float(cross_cap) <= 0.0
                        or float(self_cap) <= 0.0
                        or not -float(cross_cap) <= float(cross_min) <= float(cross_max) <= 0.0
                        or not -float(self_cap) <= float(self_min) <= float(self_max) <= 0.0
                        or not isinstance(self_stats, dict)
                        or self_stats.get("target_to_background_block_count") != 0
                        or self_stats.get("background_to_target_block_count") != 0
                        or not all(
                            _valid_sha256(payload.get(name))
                            for name in (
                                "profile_receipt_sha256",
                                "profile_payload_sha256",
                                "mask_bundle_sha256",
                                "token_binding_sha256",
                            )
                        )
                        or not valid_full_text_hash
                        or not isinstance(payload.get("field_sha256"), list)
                        or len(payload["field_sha256"]) != len(target_ids)
                        or not all(_valid_sha256(value) for value in payload["field_sha256"])
                        or not isinstance(latent_shape, list)
                        or not latent_shape
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in latent_shape
                        )
                        or not valid_anima
                    ):
                        raise RuntimeErrorResponse(
                            "Spatial prompt execution receipt is invalid."
                        )
                    receipts.append(payload)
                    continue
                solver_steps = payload.get("solver_steps")
                loss_steps = payload.get("loss_steps")
                forbidden_counts = (
                    "target_latent_count",
                    "target_noise_prediction_count",
                    "target_denoised_output_count",
                    "target_only_attention_forward_count",
                    "global_only_attention_forward_count",
                    "spatial_output_blend_count",
                    "background_substitution_count",
                )
                valid_solver_steps = (
                    isinstance(solver_steps, list)
                    and bool(solver_steps)
                    and all(
                        isinstance(item, dict)
                        and item.get("sampler_consumption_count") == 1
                        and all(
                            isinstance(item.get(field), str)
                            and bool(re.fullmatch(r"[0-9a-f]{64}", item[field]))
                            for field in (
                                "input_latent_sha256",
                                "guidance_input_latent_sha256",
                                "updated_latent_sha256",
                                "sampler_input_latent_sha256",
                                "sampler_output_latent_sha256",
                            )
                        )
                        and item.get("input_latent_sha256")
                        == item.get("guidance_input_latent_sha256")
                        and item.get("updated_latent_sha256")
                        == item.get("sampler_input_latent_sha256")
                        for item in solver_steps
                    )
                )
                if valid_solver_steps:
                    for index, item in enumerate(solver_steps):
                        if item.get("step_index") != index:
                            valid_solver_steps = False
                            break
                        next_input = item.get("next_step_input_latent_sha256")
                        if index + 1 < len(solver_steps):
                            if (
                                next_input != item.get("sampler_output_latent_sha256")
                                or next_input
                                != solver_steps[index + 1].get("input_latent_sha256")
                            ):
                                valid_solver_steps = False
                                break
                        elif next_input is not None:
                            valid_solver_steps = False
                            break
                valid_loss_steps = (
                    isinstance(loss_steps, list)
                    and bool(loss_steps)
                    and all(
                        isinstance(item, dict)
                        and isinstance(item.get("cross_attention_call_count"), int)
                        and item["cross_attention_call_count"] > 0
                        and isinstance(item.get("self_attention_call_count"), int)
                        and item["self_attention_call_count"] >= 0
                        and isinstance(item.get("total"), (int, float))
                        for item in loss_steps
                    )
                )
                if (
                    payload.get("schema_version") != "asteria.spatial-node-execution.v3"
                    or payload.get("contract_version")
                    != "asteria.spatial-prompt.node.v2"
                    or payload.get("adapter_revision")
                    != "asteria-layout-latent-guidance-v1"
                    or payload.get("routing_mode")
                    != "single_full_text_latent_guidance"
                    or not isinstance(execution_sha256, str)
                    or canonical_runtime_sha256(body) != execution_sha256
                    or not isinstance(target_ids, list)
                    or not target_ids
                    or len(target_ids) != len(set(target_ids))
                    or any(payload.get(name) != 0 for name in forbidden_counts)
                    or not isinstance(payload.get("latent_update_count"), int)
                    or payload["latent_update_count"] < 1
                    or not isinstance(
                        payload.get("sampler_attention_prior_count"), int
                    )
                    or payload["sampler_attention_prior_count"] < 1
                    or payload.get("sampler_step_count") != len(solver_steps or ())
                    or payload.get("sampler_consumption_count") != len(solver_steps or ())
                    or not valid_solver_steps
                    or not valid_loss_steps
                    or not isinstance(payload.get("latent_shape"), list)
                    or not payload["latent_shape"]
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 1
                        for value in payload["latent_shape"]
                    )
                ):
                    raise RuntimeErrorResponse(
                        "Spatial prompt execution receipt is invalid."
                    )
                receipts.append(payload)
        return sorted(
            receipts,
            key=lambda item: {"base": 0, "hires": 1, "refiner": 2}.get(
                str(item.get("stage")),
                99,
            ),
        )

    def _safe_filename(self, filename: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
