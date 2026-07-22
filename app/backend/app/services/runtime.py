from __future__ import annotations

import json
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
from app.services.comfy_admission import (
    ComfyAdmissionError,
    ComfyAdmissionService,
    ComfyAdmissionTimeoutError,
    ComfyCleanupPendingError,
    comfy_admission,
)
from app.services.comfy_model_paths import comfy_model_name
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
from app.services.workflow_mapping import WorkflowMapping, WorkflowMappingRepository


class RuntimeErrorResponse(Exception):
    pass


class RuntimeCancellationConfirmed(RuntimeErrorResponse):
    pass


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


def _regional_color_value(value: object) -> int:
    normalized = str(value or "").strip().lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", normalized):
        raise RuntimeErrorResponse(f"Invalid Anima regional prompt color: {value}")
    return int(normalized, 16)


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
        timeout_seconds: float = 600.0,
        interval_seconds: float = 1.0,
        runtime_stop_confirmed=None,
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
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

    def download_image(self, image: dict) -> bytes:
        query = urlencode(
            {
                "filename": image.get("filename", ""),
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        url = urljoin(self.base_url, f"view?{query}")
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                return response.read()
        except URLError as exc:
            raise RuntimeErrorResponse(f"Failed to download ComfyUI image: {exc}") from exc

    def upload_image(self, path: Path) -> str:
        if not _path_exists_safely(path) or not _path_is_file_safely(path):
            raise RuntimeErrorResponse(f"Image not found: {path}")
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
            raise RuntimeErrorResponse(f"Failed to upload image to ComfyUI: {exc}") from exc
        name = payload.get("name")
        if not isinstance(name, str):
            raise RuntimeErrorResponse("ComfyUI upload did not return image name")
        subfolder = payload.get("subfolder")
        return f"{subfolder}/{name}" if subfolder else name

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
    def __init__(self, mapping_dir: Path | None = None, model_root: Path | None = None) -> None:
        self.mapping_repository = WorkflowMappingRepository(mapping_dir or AppConfig().mapping_dir)
        self.model_root = model_root

    def build(self, request: dict) -> dict:
        profile = request.get("profile")
        if profile == "sdxl":
            return self._build_sdxl(request)
        if profile in {"flux", "flux1"}:
            return self._build_flux(request)
        if profile == "flux2":
            return self._build_flux2(request)
        if profile == "anima":
            return self._build_anima(request)
        raise RuntimeErrorResponse(f"Unsupported ComfyUI profile: {profile}")

    def _mapping(self, profile: str) -> WorkflowMapping:
        return self.mapping_repository.get_txt2img(profile)

    def _build_sdxl(self, request: dict) -> dict:
        mapping = self._mapping("sdxl")
        settings = request["settings"]
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
                workflow[encode_node] = {
                    "class_type": mapping.class_type("vae_encode_inpaint", "VAEEncodeForInpaint"),
                    "inputs": {
                        "pixels": [image_node, 0],
                        "vae": vae_ref,
                        "mask": [mask_node, 1],
                        "grow_mask_by": settings.get("mask_blur", 4),
                    },
                }
            else:
                workflow[encode_node] = {
                    "class_type": mapping.class_type("vae_encode", "VAEEncode"),
                    "inputs": {"pixels": [image_node, 0], "vae": vae_ref},
                }
            latent_ref: list[str | int] = [encode_node, 0]
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
            hires_model_ref = model_ref
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
            sample_node = str(next_id)
            next_id += 1
            use_inspire_hires = bool(settings.get("use_inspire_ksampler", True))
            hires_stage_settings = settings | {
                "steps": settings.get("hires_steps") or settings["steps"],
                "cfg_scale": settings.get("hires_cfg_scale") or settings["cfg_scale"],
                "sampler": self._hires_sampler(settings),
                "scheduler": self._hires_scheduler(settings),
                "denoise": settings.get("hires_denoising_strength", 0.7),
                "noise_mode": settings.get("hires_noise_mode") or "CPU",
            }
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
                "steps": refiner_steps,
                "cfg_scale": refiner_cfg_scale,
                "sampler": refiner_sampler,
                "scheduler": refiner_scheduler,
            }
            workflow[refiner_sample_node] = self._ksampler_advanced_node(
                refiner_stage_settings,
                [refiner_checkpoint_node, 0],
                [refiner_positive_node, 0],
                [refiner_negative_node, 0],
                [sample_node, 0],
                mapping,
                start_at_step=self._refiner_switch_step(settings, refiner_steps),
                end_at_step=refiner_steps,
                add_noise=False,
                return_with_leftover_noise=False,
            )
            sample_node = refiner_sample_node
            model_ref = [refiner_checkpoint_node, 0]
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
            raise RuntimeErrorResponse(f"Anima requires model slots before generation: {', '.join(missing)}")

        diffusion_node = mapping.node("diffusion_loader", "1")
        text_encoder_node = mapping.node("text_encoder_loader", "2")
        vae_node = mapping.node("vae_loader", "3")
        conditioning_node = mapping.node("conditioning", "4")
        latent_node = mapping.node("latent", "5")
        negative_conditioning_node = mapping.node("negative_conditioning", "6")
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
                    "type": "qwen_image",
                },
            },
            vae_node: {
                "class_type": mapping.class_type("vae_loader", "VAELoader"),
                "inputs": {"vae_name": self._model_name(vae, "vae")},
            },
            conditioning_node: {
                "class_type": mapping.class_type("conditioning", "TextEncodeQwenImageEdit"),
                "inputs": {
                    "clip": clip_ref,
                    "prompt": request["expanded_prompt"],
                    "vae": [vae_node, 0],
                },
            },
            negative_conditioning_node: {
                "class_type": mapping.class_type("conditioning", "TextEncodeQwenImageEdit"),
                "inputs": {
                    "clip": clip_ref,
                    "prompt": request.get("negative_prompt", ""),
                    "vae": [vae_node, 0],
                },
            },
            latent_node: {
                "class_type": mapping.class_type("latent", "EmptyQwenImageLayeredLatentImage"),
                "inputs": {
                    "width": settings["width"],
                    "height": settings["height"],
                    "layers": 0,
                    "batch_size": settings.get("batch_size", 1),
                },
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
        next_id = 7
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
                workflow[latent_node] = {
                    "class_type": mapping.class_type("vae_encode_inpaint", "VAEEncodeForInpaint"),
                    "inputs": {
                        "pixels": image_ref,
                        "vae": [vae_node, 0],
                        "mask": [mask_node, 1],
                        "grow_mask_by": settings.get("mask_blur", 4),
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
        workflow[negative_conditioning_node]["inputs"]["clip"] = clip_ref
        regional_prompts: list[tuple[dict, dict, list[str | int]]] = []
        for unit in settings.get("anima_lllite_units", []):
            if not unit.get("enabled") or not unit.get("model") or not unit.get("image_path"):
                continue
            image_node = str(next_id)
            node_id = str(next_id + 1)
            next_id += 2
            workflow[image_node] = {
                "class_type": mapping.class_type("load_image", "LoadImage"),
                "inputs": {"image": unit["image_path"]},
            }
            inputs = {
                "model": model_ref,
                "lllite_name": self._model_name(unit["model"], "controlnet"),
                "image": [image_node, 0],
                "strength": unit.get("strength", 1.0),
                "start_percent": unit.get("guidance_start", 0.0),
                "end_percent": unit.get("guidance_end", 0.45),
                "preserve_wrapper": unit.get("preserve_wrapper", True),
            }
            if unit.get("mask_image_path"):
                mask_node = str(next_id)
                next_id += 1
                workflow[mask_node] = {
                    "class_type": mapping.class_type("load_image", "LoadImage"),
                    "inputs": {"image": unit["mask_image_path"]},
                }
                inputs["mask"] = [mask_node, 1]
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
            for region in unit.get("regional_prompts", []):
                if not isinstance(region, dict) or not str(region.get("prompt") or "").strip():
                    continue
                regional_prompts.append((unit, region, [image_node, 0]))
            unit["runtime"] = "comfy_workflow"
            model_ref = [node_id, 0]
        if regional_prompts:
            base_mask_node = str(next_id)
            next_id += 1
            workflow[base_mask_node] = {
                "class_type": mapping.class_type("solid_mask", "SolidMask"),
                "inputs": {
                    "value": 1.0,
                    "width": settings["width"],
                    "height": settings["height"],
                },
            }
            couple_inputs: dict[str, object] = {
                "model": model_ref,
                "base_cond": [conditioning_node, 0],
                "base_mask": [base_mask_node, 0],
            }
            coupled_units: set[int] = set()
            for index, (unit, region, image_ref) in enumerate(regional_prompts, start=1):
                conditioning_id = str(next_id)
                mask_id = str(next_id + 1)
                next_id += 2
                workflow[conditioning_id] = {
                    "class_type": mapping.class_type("conditioning", "TextEncodeQwenImageEdit"),
                    "inputs": {
                        "clip": clip_ref,
                        "prompt": str(region["prompt"]).strip(),
                        "vae": [vae_node, 0],
                    },
                }
                workflow[mask_id] = {
                    "class_type": mapping.class_type("image_color_to_mask", "ImageColorToMask"),
                    "inputs": {
                        "image": image_ref,
                        "color": _regional_color_value(region.get("color")),
                    },
                }
                couple_inputs[f"cond_{index}"] = [conditioning_id, 0]
                couple_inputs[f"mask_{index}"] = [mask_id, 0]
                region["runtime"] = "attention_couple_ppm"
                coupled_units.add(id(unit))
            couple_node = str(next_id)
            next_id += 1
            workflow[couple_node] = {
                "class_type": mapping.class_type("attention_couple", "AttentionCouplePPM"),
                "inputs": couple_inputs,
            }
            hires_couple_node: str | None = None
            if hires_model_ref is not None:
                hires_couple_node = str(next_id)
                next_id += 1
                workflow[hires_couple_node] = {
                    "class_type": mapping.class_type("attention_couple", "AttentionCouplePPM"),
                    "inputs": couple_inputs | {"model": hires_model_ref},
                }
            for unit in settings.get("anima_lllite_units", []):
                if id(unit) in coupled_units:
                    unit["regional_prompt_runtime"] = "attention_couple_ppm"
            model_ref = [couple_node, 0]
            if hires_couple_node is not None:
                hires_model_ref = [hires_couple_node, 0]
        workflow = self._build_custom_sampler_tail(
            workflow,
            settings,
            model_ref=model_ref,
            conditioning_ref=[conditioning_node, 0],
            negative_ref=[negative_conditioning_node, 0],
            latent_ref=latent_ref,
            vae_ref=[vae_node, 0],
            mapping=mapping,
            hires_model_ref=hires_model_ref,
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
            sample_node = self._append_custom_sample(
                workflow,
                next_id,
                settings | {
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
                },
                hires_model_ref or model_ref,
                conditioning_ref,
                hires_latent_ref,
                mapping,
                negative_ref=negative_ref,
                scheduler_kind=scheduler_kind,
            )
            next_id += self._custom_sample_node_count(
                settings | {"denoise": settings.get("hires_denoising_strength", 0.7)},
                scheduler_kind,
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
                "noise_mode": settings.get("noise_mode") or "GPU(=A1111)",
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
                "noise_mode": "GPU(=A1111)",
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
        return comfy_model_name(value, model_dir_name, model_root=self.model_root)

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
        topaz_executable: Path | None = None,
        topaz_processor: Callable[..., dict[str, str | list[str]]] = process_topaz_image,
        admission_service: ComfyAdmissionService = comfy_admission,
        admission_timeout_seconds: float | None = 30.0,
    ) -> None:
        self.client = client
        self.workflow_builder = workflow_builder or ComfyTxt2ImgWorkflowBuilder(model_root=model_root)
        self.topaz_executable = topaz_executable
        self.topaz_processor = topaz_processor
        self.admission_service = admission_service
        self.admission_timeout_seconds = admission_timeout_seconds

    def generate_txt2img(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_txt2img_admitted(request, output_dir)

    def _generate_txt2img_admitted(self, request: dict, output_dir: Path) -> dict:
        request.setdefault("settings", {})["mode"] = "txt2img"
        self._ensure_controlnet_mode_runtime(request)
        self._apply_runtime_node_fallbacks(request)
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
        return self._generate(request, output_dir)

    def generate_img2img(self, request: dict, output_dir: Path) -> dict:
        with self._admission_scope(request):
            return self._generate_img2img_admitted(request, output_dir)

    def _generate_img2img_admitted(self, request: dict, output_dir: Path) -> dict:
        request.setdefault("settings", {})["mode"] = "img2img"
        self._ensure_controlnet_mode_runtime(request)
        self._apply_runtime_node_fallbacks(request)
        self._prepare_init_image(request)
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
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
        if self._uses_topaz_hires(request):
            return self._generate_topaz_hires(request, output_dir)
        result = self._generate(request, output_dir)
        if only_masked:
            full_size = _size_tuple(settings.get("inpaint_full_size"))
            if full_size is None:
                raise RuntimeErrorResponse(api_errors.INPAINT_ONLY_MASKED_GEOMETRY_INVALID)
            settings["width"], settings["height"] = full_size
        return result

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

    def _generate(self, request: dict, output_dir: Path) -> dict:
        try:
            if output_dir.exists() and not output_dir.is_dir():
                raise RuntimeErrorResponse(f"ComfyUI output path is not a directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        except RuntimeErrorResponse:
            raise
        except OSError as exc:
            raise RuntimeErrorResponse(f"Failed to prepare ComfyUI output directory: {output_dir}") from exc
        self._prepare_control_images(request, output_dir / "controlnet")
        self._prepare_anima_lllite_images(request)
        batch_count = max(1, int(request.get("settings", {}).get("batch_count", 1)))
        base_seed = int(request.get("settings", {}).get("seed", -1))
        progress_callback = request.get("progress_callback")
        self._publish_generation_progress(progress_callback, 0, batch_count)
        prompt_ids: list[str] = []
        generated_seeds: list[int] = []
        output_paths: list[str] = []
        primary_outputs: list[dict[str, str | int]] = []
        batch_state = {"index": 0}
        with self._preview_listener(
            output_dir / "_previews",
            self._preview_callback(progress_callback, batch_count, batch_state),
            self._step_callback(progress_callback, batch_count, batch_state),
        ):
            for batch_index in range(batch_count):
                batch_state["index"] = batch_index
                iteration_request = deepcopy(request)
                settings = iteration_request.setdefault("settings", {})
                settings["batch_count"] = 1
                seed = self._iteration_seed(base_seed, batch_index)
                settings["seed"] = seed
                generated_seeds.append(seed)
                workflow = self.workflow_builder.build(iteration_request)
                self._merge_builder_runtime_settings(request, iteration_request)
                prompt_id = self.client.queue_prompt(workflow)
                prompt_ids.append(prompt_id)
                runtime_job_callback = iteration_request.get("runtime_job_callback")
                if callable(runtime_job_callback):
                    runtime_job_callback(prompt_id)
                runtime_stop_confirmed = iteration_request.get("runtime_stop_confirmed")
                if callable(runtime_stop_confirmed):
                    history = self.client.wait_for_history(
                        prompt_id,
                        runtime_stop_confirmed=runtime_stop_confirmed,
                    )
                else:
                    history = self.client.wait_for_history(prompt_id)
                images = self._images_from_history(history, prompt_id)
                primary_node_ids = {
                    str(node_id) for node_id in settings.get("primary_output_node_ids", [])
                }
                for image in images:
                    filename = self._safe_filename(image.get("filename", "comfy-output.png"))
                    path = output_dir / f"{uuid4().hex}_{filename}"
                    image_bytes = self.client.download_image(image)
                    try:
                        path.write_bytes(image_bytes)
                    except OSError as exc:
                        raise RuntimeErrorResponse(f"Failed to save ComfyUI output image: {path}") from exc
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
                    output_paths.append(str(path))
                    if str(image.get("node_id", "")) in primary_node_ids:
                        primary_outputs.append(
                            {
                                "path": str(path),
                                "seed": seed,
                                "output_index": len(primary_outputs),
                                "runtime_job_id": prompt_id,
                            }
                        )
                self._publish_generation_progress(
                    progress_callback,
                    batch_index + 1,
                    batch_count,
                    output_paths[-1] if output_paths else "",
                )
        if not output_paths:
            raise RuntimeErrorResponse(f"ComfyUI completed without image outputs: {', '.join(prompt_ids)}")
        request.setdefault("settings", {})["generated_seeds"] = generated_seeds
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
        }

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

    def _generate_topaz_hires(self, request: dict, output_dir: Path) -> dict:
        settings = request.setdefault("settings", {})
        source_mode = str(settings.get("mode") or "txt2img")
        settings["topaz_hires_source_mode"] = source_mode
        if self.topaz_executable is None or not _path_is_file_safely(self.topaz_executable):
            settings["topaz_hires_runtime"] = "skipped_missing_executable"
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
        first_result = self._generate(first_request, output_dir)
        self._merge_builder_runtime_settings(request, first_request)
        first_paths = [Path(path) for path in first_result["output_paths"]]
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
        for path in first_paths:
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
        primary_outputs: list[dict[str, str | int]] = []
        for index, topaz_path in enumerate(topaz_paths):
            second_request = deepcopy(request)
            second_settings = second_request.setdefault("settings", {})
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

            second_result = self._generate(second_request, output_dir)
            if request.get("profile") == "anima":
                self._merge_builder_runtime_settings(request, second_request)
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
            for primary_output in second_result.get("primary_outputs", []):
                primary_outputs.append(
                    {
                        **primary_output,
                        "output_index": len(primary_outputs),
                    }
                )
            second_prompt_ids.extend(second_result.get("runtime_job_ids", [second_result["runtime_job_id"]]))

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
        blur_radius = int(settings.get("mask_blur") or 0) if request.get("profile") in {"flux1", "flux2"} else 0
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

    def _safe_filename(self, filename: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
