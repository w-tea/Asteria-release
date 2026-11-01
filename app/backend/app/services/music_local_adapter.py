from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.schemas.media_model_catalog import MediaModelComponentRole
from app.schemas.music import MusicRequest
from app.services.music_local_setup import MusicModelProfileRepository
from app.services.music_packet import MusicSubmissionPacket
from app.services.runtime import ComfyClient, RuntimeCancellationConfirmed, RuntimeErrorResponse


MUSIC_WORKFLOW_SCHEMA_VERSION = "asteria.music.workflow.v1"
MUSIC3_AUDIO_FRAMES_PER_SECOND = 25
MUSIC3_MAX_AUDIO_FRAMES = 9000
_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_AUDIO_SUFFIXES = {".flac", ".mp3", ".opus", ".wav"}


class MusicLocalAdapterError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        partial_paths: tuple[Path, ...] = (),
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.partial_paths = partial_paths
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MusicDurationNormalization:
    requested_duration_seconds: float
    effective_duration_seconds: float
    effective_audio_frames: int
    provider_clamp_reason: str | None


@dataclass(frozen=True, slots=True)
class LocalMusicExecutionResult:
    runtime_job_id: str
    workflow_id: str
    execution_profile: str
    output_paths: tuple[Path, ...]
    provider_output: dict
    timing: MusicDurationNormalization
    model_revision: str = ""
    model_asset_ids: tuple[str, ...] = ()
    model_bundle_id: str | None = None
    model_components: tuple[MusicWorkflowModelAsset, ...] = ()
    sampling: MusicWorkflowSamplingReceipt | None = None
    workflow_source_revision: str = ""


@dataclass(frozen=True, slots=True)
class MusicWorkflowModelAsset:
    role: MediaModelComponentRole
    stable_asset_id: str
    loader_name: str
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str


@dataclass(frozen=True, slots=True)
class MusicWorkflowModelBundle:
    bundle_id: str
    architecture: str
    execution_profile: str
    model_revision: str
    components: tuple[MusicWorkflowModelAsset, ...]


@dataclass(frozen=True, slots=True)
class MusicWorkflowSamplingReceipt:
    text_guidance: float
    top_k: int
    steps: int
    sampler: str
    scheduler: str
    denoise: float
    tiled_decode: bool
    ksampler_cfg: float


class MusicWorkflowAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    workflow_id: str
    minimum_comfyui_version: str
    source_revision: str
    required_nodes: tuple[str, ...]
    workflow: dict[str, dict]
    bindings: dict[str, str]

    @field_validator("source_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("music workflow source revision is invalid")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "MusicWorkflowAsset":
        if self.schema_version != MUSIC_WORKFLOW_SCHEMA_VERSION:
            raise ValueError("music workflow schema is unsupported")
        if not _WORKFLOW_ID.fullmatch(self.workflow_id) or not self.workflow or not self.bindings:
            raise ValueError("music workflow is incomplete")
        classes = {
            str(node.get("class_type") or "")
            for node in self.workflow.values()
            if isinstance(node, dict)
        }
        if not set(self.required_nodes).issubset(classes):
            raise ValueError("music workflow required nodes are missing")
        return self


class MusicWorkflowRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def load(self, workflow_id: str) -> MusicWorkflowAsset:
        if not _WORKFLOW_ID.fullmatch(workflow_id):
            raise MusicLocalAdapterError("music_workflow_not_found")
        matches: list[MusicWorkflowAsset] = []
        try:
            paths = sorted(self.root.glob("*.json"))
        except OSError as error:
            raise MusicLocalAdapterError("music_workflow_unavailable") from error
        for path in paths:
            try:
                candidate = MusicWorkflowAsset.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise MusicLocalAdapterError("music_workflow_invalid") from error
            if candidate.workflow_id == workflow_id:
                matches.append(candidate)
        if len(matches) != 1:
            raise MusicLocalAdapterError(
                "music_workflow_not_found" if not matches else "music_workflow_invalid"
            )
        return matches[0]


def normalized_music3_duration(value: float) -> MusicDurationNormalization:
    requested = float(value)
    if not math.isfinite(requested) or requested <= 0:
        raise MusicLocalAdapterError("music_duration_invalid")
    requested_frames = max(1, round(requested * MUSIC3_AUDIO_FRAMES_PER_SECOND))
    frames = min(MUSIC3_MAX_AUDIO_FRAMES, requested_frames)
    return MusicDurationNormalization(
        requested_duration_seconds=requested,
        effective_duration_seconds=frames / MUSIC3_AUDIO_FRAMES_PER_SECOND,
        effective_audio_frames=frames,
        provider_clamp_reason=(
            "minimax_music3_max_audio_frames"
            if requested > MUSIC3_MAX_AUDIO_FRAMES / MUSIC3_AUDIO_FRAMES_PER_SECOND
            else None
        ),
    )


class LocalMusic3ExecutionAdapter:
    def __init__(
        self,
        *,
        client: ComfyClient,
        workflow_root: Path,
        output_root: Path | Callable[[], Path],
        model_profile_path: Path,
        execution_profile: str | Callable[[], str],
        model_bundle_resolver: Callable[[MusicRequest], MusicWorkflowModelBundle] | None = None,
    ) -> None:
        self.client = client
        self.workflows = MusicWorkflowRepository(workflow_root)
        self._output_root_provider = (
            output_root if callable(output_root) else lambda: Path(output_root)
        )
        self.model_profile_path = Path(model_profile_path)
        self.execution_profile = execution_profile
        self.model_bundle_resolver = model_bundle_resolver

    @property
    def output_root(self) -> Path:
        return Path(self._output_root_provider())

    def execute_music(
        self,
        packet: MusicSubmissionPacket,
        *,
        operation_id: str,
        output_root: Path | None = None,
        progress_callback: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
        runtime_job_callback: Callable[[str], None],
    ) -> LocalMusicExecutionResult:
        operation_output_root = Path(output_root) if output_root is not None else self.output_root
        payload = packet.provider_payload
        workflow_id = str(payload.get("workflow_template_id") or "")
        asset = self.workflows.load(workflow_id)
        profile = self._execution_profile()
        (
            workflow,
            timing,
            model_revision,
            model_asset_ids,
            effective_profile,
            model_bundle,
            sampling,
        ) = self._build_workflow(asset, packet, operation_id, profile)
        total = max(1, math.ceil(timing.effective_duration_seconds))
        prompt_id: str | None = None
        written: list[Path] = []
        try:
            self._raise_if_cancelled(cancel_requested, prompt_id)
            progress_callback(0, total)
            prompt_id = self.client.queue_prompt(workflow)
            runtime_job_callback(prompt_id)
            history = self.client.wait_for_history(
                prompt_id,
                timeout_seconds=None,
                runtime_stop_confirmed=cancel_requested,
            )
            self._raise_if_cancelled(cancel_requested, prompt_id)
            descriptors = self._audio_outputs(history, prompt_id)
            if not descriptors:
                raise MusicLocalAdapterError("music_runtime_output_missing", retryable=True)
            destination = self._prepare_output_subdirectory(
                "music",
                _safe_token(operation_id),
                output_root=operation_output_root,
            )
            for index, descriptor in enumerate(descriptors):
                self._raise_if_cancelled(cancel_requested, prompt_id)
                filename = self._output_filename(descriptor, operation_id, index)
                data = self.client.download_output(descriptor)
                if not data:
                    raise MusicLocalAdapterError("music_runtime_output_invalid", retryable=True)
                path = destination / filename
                temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(data)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
                written.append(path)
            progress_callback(total, total)
            return LocalMusicExecutionResult(
                runtime_job_id=prompt_id,
                workflow_id=asset.workflow_id,
                execution_profile=effective_profile,
                output_paths=tuple(written),
                provider_output=dict(descriptors[0]),
                timing=timing,
                model_revision=model_revision,
                model_asset_ids=model_asset_ids,
                model_bundle_id=model_bundle.bundle_id if model_bundle is not None else None,
                model_components=model_bundle.components if model_bundle is not None else (),
                sampling=sampling,
                workflow_source_revision=asset.source_revision,
            )
        except MusicLocalAdapterError as error:
            if error.code == "music_generation_cancelled":
                self._discard(written)
                raise
            if written and not error.partial_paths:
                raise MusicLocalAdapterError(
                    error.code,
                    retryable=error.retryable,
                    partial_paths=tuple(written),
                ) from error
            raise
        except RuntimeCancellationConfirmed as error:
            if prompt_id is not None:
                self._cancel_prompt(prompt_id)
            self._discard(written)
            raise MusicLocalAdapterError("music_generation_cancelled", retryable=True) from error
        except (OSError, RuntimeErrorResponse, TypeError, ValueError, KeyError) as error:
            if written:
                raise MusicLocalAdapterError(
                    "music_runtime_failed", retryable=True, partial_paths=tuple(written)
                ) from error
            raise MusicLocalAdapterError("music_runtime_failed", retryable=True) from error

    def build_workflow_for_test(
        self,
        asset: MusicWorkflowAsset,
        packet: MusicSubmissionPacket,
        operation_id: str,
    ) -> tuple[dict, MusicDurationNormalization]:
        (
            workflow,
            timing,
            _revision,
            _asset_ids,
            _effective_profile,
            _model_bundle,
            _sampling,
        ) = self._build_workflow(asset, packet, operation_id, self._execution_profile())
        return workflow, timing

    def _build_workflow(
        self,
        asset: MusicWorkflowAsset,
        packet: MusicSubmissionPacket,
        operation_id: str,
        execution_profile: str,
    ) -> tuple[
        dict,
        MusicDurationNormalization,
        str,
        tuple[str, ...],
        str,
        MusicWorkflowModelBundle | None,
        MusicWorkflowSamplingReceipt,
    ]:
        payload = packet.provider_payload
        conditioning = payload.get("conditioning")
        generation = payload.get("generation")
        runtime = payload.get("runtime")
        if not isinstance(conditioning, dict) or not isinstance(generation, dict) or not isinstance(runtime, dict):
            raise MusicLocalAdapterError("music_workflow_mismatch")
        timing = normalized_music3_duration(float(generation.get("max_duration", 0)))
        try:
            request = MusicRequest.model_validate(packet.request_snapshot)
        except ValueError as error:
            raise MusicLocalAdapterError("music_workflow_mismatch") from error
        effective_profile = execution_profile
        model_bundle: MusicWorkflowModelBundle | None = None
        if request.schema_version == "asteria.music.request.v2":
            if self.model_bundle_resolver is None or request.sampling is None or request.model_selection is None:
                raise MusicLocalAdapterError("music_model_resolver_not_configured")
            bundle = self.model_bundle_resolver(request)
            model_bundle = bundle
            effective_profile = request.sampling.execution_profile
            if (
                bundle.bundle_id != request.model_selection.bundle_id
                or bundle.architecture != request.model_selection.architecture
                or bundle.execution_profile != effective_profile
            ):
                raise MusicLocalAdapterError("music_model_bundle_mismatch")
            unet_name, clip_name, vae_name, asset_ids = self._workflow_bundle_values(bundle)
            model_revision = bundle.model_revision
        else:
            unet_name, clip_name, vae_name, model_revision, asset_ids = self._profile_asset_names(
                execution_profile
            )
        ksampler_cfg = float(generation.get("ksampler_cfg", 1.7))
        if ksampler_cfg != 1.7:
            raise MusicLocalAdapterError("music_workflow_mismatch")
        values: dict[str, object] = {
            "unet_name": unet_name,
            "clip_name": clip_name,
            "vae_name": vae_name,
            "caption": str(conditioning.get("caption") or ""),
            "lyrics": str(conditioning.get("lyrics") or ""),
            "seed": int(generation.get("seed", 0)),
            "max_duration": timing.effective_duration_seconds,
            "text_guidance": float(
                generation.get("text_guidance", generation.get("cfg_scale", 1.7))
            ),
            "top_k": int(generation.get("top_k", 50)),
            "steps": int(generation.get("steps", 30)),
            "ksampler_cfg": ksampler_cfg,
            "sampler": str(generation.get("sampler", "euler")),
            "scheduler": str(generation.get("scheduler", "simple")),
            "denoise": float(generation.get("denoise", 1.0)),
            "tiled_decode": bool(runtime.get("tiled_decode", True)),
            "output_prefix": f"asteria/music/{_safe_token(operation_id)}",
        }
        sampling = MusicWorkflowSamplingReceipt(
            text_guidance=float(values["text_guidance"]),
            top_k=int(values["top_k"]),
            steps=int(values["steps"]),
            sampler=str(values["sampler"]),
            scheduler=str(values["scheduler"]),
            denoise=float(values["denoise"]),
            tiled_decode=bool(values["tiled_decode"]),
            ksampler_cfg=ksampler_cfg,
        )
        workflow = deepcopy(asset.workflow)
        aliases = {"sampler_seed": "seed"}
        for name, target in asset.bindings.items():
            value_name = aliases.get(name, name)
            if value_name not in values:
                raise MusicLocalAdapterError("music_workflow_invalid")
            self._assign_binding(workflow, target, values[value_name])
        if self._contains_placeholder(workflow):
            raise MusicLocalAdapterError("music_workflow_invalid")
        return (
            workflow,
            timing,
            model_revision,
            asset_ids,
            effective_profile,
            model_bundle,
            sampling,
        )

    def _execution_profile(self) -> str:
        value = self.execution_profile() if callable(self.execution_profile) else self.execution_profile
        if value not in {"full", "offload", "layer_streaming"}:
            raise MusicLocalAdapterError("music_execution_profile_invalid")
        return value

    def _profile_asset_names(
        self, profile_id: str
    ) -> tuple[str, str, str, str, tuple[str, ...]]:
        try:
            profile = MusicModelProfileRepository(self.model_profile_path).load()
            selected = profile.execution_profile(profile_id)  # type: ignore[arg-type]
            bundle = profile.model_bundle(selected.recommended_bundle_id)
            assets = {asset.id: asset for asset in profile.assets}
            selected_assets = [assets[value] for value in bundle.asset_ids]
        except (KeyError, ValueError) as error:
            raise MusicLocalAdapterError("music_execution_profile_invalid") from error
        unets = [PurePosixPath(value.relative_path).name for value in selected_assets if value.component_role == "diffusion_model"]
        clips = [PurePosixPath(value.relative_path).name for value in selected_assets if value.component_role == "text_encoder"]
        vaes = [PurePosixPath(value.relative_path).name for value in selected_assets if value.component_role == "audio_vae"]
        if len(unets) != 1 or len(clips) != 1 or len(vaes) != 1:
            raise MusicLocalAdapterError("music_execution_profile_invalid")
        return (
            unets[0],
            clips[0],
            vaes[0],
            profile.model_repository.revision,
            bundle.asset_ids,
        )

    @staticmethod
    def _workflow_bundle_values(
        bundle: MusicWorkflowModelBundle,
    ) -> tuple[str, str, str, tuple[str, ...]]:
        by_role: dict[MediaModelComponentRole, MusicWorkflowModelAsset] = {}
        for component in bundle.components:
            if component.role in by_role or not component.loader_name:
                raise MusicLocalAdapterError("music_model_bundle_mismatch")
            by_role[component.role] = component
        required = (
            MediaModelComponentRole.DIFFUSION_MODEL,
            MediaModelComponentRole.TEXT_ENCODER,
            MediaModelComponentRole.AUDIO_VAE,
        )
        if set(by_role) != set(required):
            raise MusicLocalAdapterError("music_model_bundle_mismatch")
        ordered = tuple(by_role[role] for role in required)
        return (
            ordered[0].loader_name,
            ordered[1].loader_name,
            ordered[2].loader_name,
            tuple(item.stable_asset_id for item in ordered),
        )

    @staticmethod
    def _assign_binding(workflow: dict, target: str, value: object) -> None:
        parts = target.split(".")
        if len(parts) != 3 or parts[1] != "inputs":
            raise MusicLocalAdapterError("music_workflow_invalid")
        node = workflow.get(parts[0])
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise MusicLocalAdapterError("music_workflow_invalid")
        node["inputs"][parts[2]] = value

    @staticmethod
    def _contains_placeholder(value: object) -> bool:
        if isinstance(value, str):
            return value.startswith("${") and value.endswith("}")
        if isinstance(value, dict):
            return any(LocalMusic3ExecutionAdapter._contains_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(LocalMusic3ExecutionAdapter._contains_placeholder(item) for item in value)
        return False

    @staticmethod
    def _audio_outputs(history: dict, prompt_id: str) -> tuple[dict, ...]:
        prompt = history.get(prompt_id)
        if not isinstance(prompt, dict) or not isinstance(prompt.get("outputs"), dict):
            return ()
        results: list[dict] = []
        for node in prompt["outputs"].values():
            if not isinstance(node, dict) or not isinstance(node.get("audio"), list):
                continue
            for descriptor in node["audio"]:
                if LocalMusic3ExecutionAdapter._valid_output_descriptor(descriptor):
                    results.append(dict(descriptor))
                else:
                    raise MusicLocalAdapterError("music_runtime_output_invalid", retryable=True)
        return tuple(results)

    @staticmethod
    def _valid_output_descriptor(value: object) -> bool:
        if not isinstance(value, dict) or value.get("type", "output") != "output":
            return False
        filename = value.get("filename")
        subfolder = value.get("subfolder", "")
        if not isinstance(filename, str) or not isinstance(subfolder, str):
            return False
        if Path(filename).name != filename or Path(filename).suffix.lower() not in _AUDIO_SUFFIXES:
            return False
        if any(ord(char) < 32 or char in {"/", "\\", ":"} for char in filename):
            return False
        normalized = subfolder.replace("\\", "/")
        return not normalized.startswith("/") and ".." not in normalized.split("/") and ":" not in normalized

    @staticmethod
    def _output_filename(output: dict, operation_id: str, index: int) -> str:
        source = str(output.get("filename") or "")
        suffix = Path(source).suffix.lower()
        if Path(source).name != source or suffix not in _AUDIO_SUFFIXES:
            raise MusicLocalAdapterError("music_runtime_output_invalid")
        stem = _SAFE_FILENAME.sub("_", Path(source).stem).strip("._-") or "song"
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

    def _raise_if_cancelled(self, callback: Callable[[], bool], prompt_id: str | None) -> None:
        if not callback():
            return
        if prompt_id is not None:
            self._cancel_prompt(prompt_id)
        raise MusicLocalAdapterError("music_generation_cancelled", retryable=True)

    def _prepare_output_subdirectory(self, *parts: str, output_root: Path | None = None) -> Path:
        try:
            selected_root = Path(output_root) if output_root is not None else self.output_root
            selected_root.mkdir(parents=True, exist_ok=True)
            if _is_reparse_path(selected_root):
                raise MusicLocalAdapterError("music_output_path_unsafe")
            root = selected_root.resolve(strict=True)
            current = selected_root
            for part in parts:
                if not part or part in {".", ".."} or Path(part).name != part:
                    raise MusicLocalAdapterError("music_output_path_unsafe")
                current = current / part
                current.mkdir(exist_ok=True)
                if _is_reparse_path(current) or not current.resolve(strict=True).is_relative_to(root):
                    raise MusicLocalAdapterError("music_output_path_unsafe")
            return current
        except MusicLocalAdapterError:
            raise
        except OSError as error:
            raise MusicLocalAdapterError("music_output_path_unsafe") from error

    @staticmethod
    def _discard(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue


def _safe_token(value: str) -> str:
    normalized = _SAFE_FILENAME.sub("_", value).strip("._-")
    if not normalized:
        raise MusicLocalAdapterError("music_operation_id_invalid")
    return normalized[:128]


def _is_reparse_path(path: Path) -> bool:
    try:
        value = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(value.st_mode):
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if attributes & reparse:
        return True
    junction = getattr(os.path, "isjunction", None)
    return bool(junction(path)) if callable(junction) else False
