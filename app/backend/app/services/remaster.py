from dataclasses import dataclass
from contextlib import ExitStack, nullcontext
from pathlib import Path
import shutil
from collections.abc import Callable

from PIL import Image

from app.core.config import AppConfig
from app.schemas.remaster import RemasterRequest, RemasterResponse, ResolvedRemasterSettings, resolve_remaster_settings
from app.services import api_errors
from app.services.generation_artifacts import (
    OutputFinalizationError,
    apply_filename_template,
    apply_output_image_format,
    unique_output_path,
)
from app.services.history import HistoryRepository
from app.services.image_check_operation import ImageCheckOperation
from app.services.image_check_finalization import FinalImageSession, finalize_completed_batch, is_private_image_candidate
from app.services.metadata import MetadataWriter
from app.services.remaster_comparison import (
    RemasterComparisonResult,
    build_remaster_comparison,
    comparison_disabled_result,
    refresh_remaster_comparison_final_info,
)
from app.services.runtime import RuntimeCancellationConfirmed, RuntimeErrorResponse
from app.services.runtime_adapters import RuntimeAdapterError
from app.services.upscaler_catalog import upscaler_model_name, upscaler_option
from app.services.user_settings import UserSettingsRepository, resolve_current_output_root


REMASTER_REQUIRED_NODES = {
    "LoadImage",
    "UpscaleModelLoader",
    "ImageUpscaleWithModel",
    "ImageScale",
    "VAEEncode",
    "LatentUpscale",
    "VAEDecode",
    "SaveImage",
}


class RemasterValidationError(Exception):
    def __init__(self, code: str, **details: object) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


class RemasterExecutionError(Exception):
    def __init__(self, remaster_id: int, code: str, **details: object) -> None:
        super().__init__(code)
        self.remaster_id = remaster_id
        self.code = code
        self.details = details


def validate_remaster_scheduler_contract(request: RemasterRequest) -> None:
    if request.profile != "flux2":
        return
    if request.remaster.scheduler not in {"", "Flux2"}:
        raise RemasterValidationError(
            api_errors.UNSUPPORTED_COMFY_RUNTIME_SETTING,
            unsupported=[
                {
                    "field": "remaster.scheduler",
                    "label": request.remaster.scheduler,
                    "accepted": "Flux2",
                }
            ],
        )
    request.remaster.scheduler = "Flux2"


@dataclass(frozen=True)
class RemasterValidationResult:
    source_path: Path
    source_size: tuple[int, int]
    anchor_size: tuple[int, int]
    buffer_size: tuple[int, int]
    target_size: tuple[int, int]
    anchor_model_name: str
    settings: ResolvedRemasterSettings


class RemasterOrchestrator:
    def __init__(
        self,
        *,
        config: AppConfig,
        history: HistoryRepository,
        runtime: object,
        object_info_provider: Callable[[], dict],
        comparison_builder: Callable[[Path, Path, Path, dict], RemasterComparisonResult] = build_remaster_comparison,
        metadata_writer: MetadataWriter | None = None,
    ) -> None:
        self.config = config
        self.history = history
        self.runtime = runtime
        self.object_info_provider = object_info_provider
        self.comparison_builder = comparison_builder
        self.metadata_writer = metadata_writer or MetadataWriter()

    def run(
        self,
        request: RemasterRequest,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        runtime_job_callback: Callable[[str], None] | None = None,
        runtime_stop_confirmed: Callable[[], bool] | None = None,
        image_check_operation: ImageCheckOperation | None = None,
        image_checker: Callable | None = None,
    ) -> RemasterResponse:
        validate_remaster_scheduler_contract(request)
        request_snapshot = request.model_dump(mode="json")
        checked_operation = image_check_operation if image_check_operation and image_check_operation.settings.enabled else None
        output_root = (checked_operation.output_root if checked_operation else resolve_current_output_root(self.config.data_dir)).resolve()
        settings: dict = {
            "mode": "remaster",
            "remaster_request_snapshot": request_snapshot,
            "source_generation_id": request.source_generation_id,
            "source_output_index": request.source_output_index,
            "source_image_path": request.source_image_path,
            "output_root": str(output_root),
        }
        remaster_id = self.history.add(
            status="queued",
            profile=request.profile,
            prompt=request.prompt,
            expanded_prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            settings=settings,
            model_slots=request.model_slot_overrides,
            loras=[lora.model_dump(mode="json") for lora in request.loras],
            embeddings=[],
            output_paths=[],
        )

        if callable(cancel_requested) and cancel_requested():
            self._persist_cancelled(remaster_id, settings, [], {})
            raise RemasterExecutionError(remaster_id, "remaster_cancelled")

        try:
            validation = validate_remaster_request(
                request,
                self.config,
                self.object_info_provider(),
            )
        except RemasterValidationError as exc:
            settings["error_code"] = exc.code
            settings["error_details"] = exc.details
            self.history.update_generation(
                remaster_id,
                status="failed",
                settings=settings,
                output_paths=[],
            )
            raise RemasterExecutionError(remaster_id, exc.code, **exc.details) from exc
        except (OSError, RuntimeError, RuntimeErrorResponse, RuntimeAdapterError) as exc:
            code = api_errors.GENERATION_RUNTIME_UNREACHABLE
            settings["error_code"] = code
            self.history.update_generation(
                remaster_id,
                status="failed",
                settings=settings,
                output_paths=[],
            )
            raise RemasterExecutionError(remaster_id, code) from exc

        settings.update(
            {
                "resolved_remaster": validation.settings.model_dump(mode="json"),
                "remaster_stage_sizes": {
                    "source": _size_dict(validation.source_size),
                    "anchor": _size_dict(validation.anchor_size),
                    "buffer": _size_dict(validation.buffer_size),
                    "target": _size_dict(validation.target_size),
                },
                "anchor_model_name": validation.anchor_model_name,
            }
        )
        self.history.update_generation(
            remaster_id,
            status="running",
            settings=settings,
            output_paths=[],
        )

        working_dir = output_root / ".remaster-working" / str(remaster_id)
        runtime_result: dict = {}
        persisted_paths: list[str] = []
        finalization_stack = ExitStack()
        image_session = None

        def failure_paths() -> list[str]:
            if image_session and not image_session.committed:
                # The session rolls these files back. Never register a path
                # that will disappear as soon as the stack closes.
                check = settings.get('image_check', {})
                for record in check.get('results', []):
                    record.pop('saved_path', None)
                check['auxiliary_paths'] = []
                settings.pop('final_output_path', None)
                return []
            return persisted_paths

        try:
            image_session = (finalization_stack.enter_context(FinalImageSession(checked_operation, output_root / request.profile, history=self.history))
                             if checked_operation else None)
            if image_session:
                working_dir = image_session.directory
            runtime_request = self._runtime_request(request, validation)
            runtime_request['cancel_requested'] = cancel_requested
            runtime_request["runtime_job_callback"] = runtime_job_callback
            runtime_request["runtime_stop_confirmed"] = runtime_stop_confirmed
            runtime_result = self.runtime.generate_remaster(runtime_request, working_dir)
            if callable(cancel_requested) and cancel_requested():
                self._persist_cancelled(remaster_id, settings, self._working_paths(working_dir), runtime_result)
                raise RemasterExecutionError(remaster_id, "remaster_cancelled")
            primary_outputs = runtime_result.get("primary_outputs", [])
            if not isinstance(primary_outputs, list) or len(primary_outputs) != 1:
                raise RemasterExecutionError(remaster_id, "remaster_primary_output_invalid")
            runtime_path = Path(str(primary_outputs[0].get("path") or ""))
            if not _is_file(runtime_path):
                raise RemasterExecutionError(remaster_id, "remaster_primary_output_invalid")

            check_passed = True
            if image_session:
                checked = finalize_completed_batch(checked_operation, runtime_result, runtime_request['settings'],
                    checker=image_checker, cancel_requested=cancel_requested)
                records = checked['image_check_results']
                if checked['status'] == 'cancelled' and not records:
                    raise RuntimeCancellationConfirmed()
                check_passed = bool(checked['primary_outputs'])
                settings['image_check'] = {
                    'settings': checked_operation.settings.model_dump(mode='json'), 'target_stage': 'final_output',
                    'results': records, 'status': records[0]['status'], 'passed': int(check_passed),
                    'rejected': sum(r['status'] == 'reject' for r in records),
                    'pause': checked['image_check_pause'], 'cancelled': checked['status'] == 'cancelled',
                    'auxiliary_paths': [],
                }
                settings['filename_template'] = checked_operation.output_settings.filename_template if checked_operation.output_settings else ''
                settings['generated_seeds'] = [request.seed]
                settings['seed'] = request.seed
                if not checked['output_paths']:
                    settings['remaster_result_status'] = 'completed_with_warnings'
                    with self.history.publication():
                        self.history.update_generation(remaster_id, status='completed', settings=settings, output_paths=[])
                        image_session.seal_history(remaster_id)
                    image_session.commit()
                    return RemasterResponse(id=remaster_id, status='completed_with_warnings',
                        source_generation_id=request.source_generation_id, source_image_path=request.source_image_path,
                        source_output_index=request.source_output_index, settings=settings,
                        model_slots=request.model_slot_overrides, loras=request.loras,
                        warnings=['image_check_' + records[0]['status']])
                final_path = Path(image_session.publish(checked['output_paths'], settings, request.profile, remaster_id)[0])
            else:
                final_path = self._finalize_runtime_output(remaster_id, request, validation, runtime_path, output_root)
            persisted_paths.append(str(final_path))
            if callable(cancel_requested) and cancel_requested():
                self._persist_cancelled(remaster_id, settings, failure_paths(), runtime_result)
                raise RemasterExecutionError(remaster_id, "remaster_cancelled")
            comparison = (self._comparison(remaster_id, request, validation, final_path, runtime_result)
                          if check_passed else comparison_disabled_result())
            warnings = list(comparison.warnings)
            if not check_passed:
                warnings.append('image_check_reject')
            result_status = "completed_with_warnings" if warnings else "completed"
            settings.update(
                {
                    "runtime_adapter": "comfy_workflow",
                    "runtime_job_id": str(runtime_result.get("runtime_job_id") or ""),
                    "runtime_job_ids": list(runtime_result.get("runtime_job_ids") or []),
                    "remaster_result_status": result_status,
                    "warnings": warnings,
                    "final_output_path": str(final_path),
                    "comparison_contact_sheet_path": comparison.contact_sheet_path,
                    "comparison_report_path": comparison.report_path,
                    "comparison_metrics": comparison.metrics,
                }
            )
            metadata_payload = {
                "generation_id": remaster_id,
                "status": "completed",
                "profile": request.profile,
                "prompt": request.prompt,
                "expanded_prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "settings": settings,
                "model_slots": request.model_slot_overrides,
                "loras": [lora.model_dump(mode="json") for lora in request.loras],
                "embeddings": [],
            }
            sidecar_paths = self.metadata_writer.write_generation_sidecars(
                metadata_payload,
                [str(final_path)],
            )
            if comparison.report_path and not refresh_remaster_comparison_final_info(
                Path(comparison.report_path),
                final_path,
            ):
                warnings.append("comparison_report_refresh_failed")
                result_status = "completed_with_warnings"
                settings["warnings"] = warnings
                settings["remaster_result_status"] = result_status
                sidecar_paths = self.metadata_writer.write_generation_sidecars(
                    metadata_payload,
                    [str(final_path)],
                )
                refresh_remaster_comparison_final_info(Path(comparison.report_path), final_path)
            output_paths = [
                str(final_path),
                *sidecar_paths,
                *([comparison.contact_sheet_path] if comparison.contact_sheet_path else []),
                *([comparison.report_path] if comparison.report_path else []),
            ]
            with self.history.publication() if image_session else nullcontext():
                self.history.update_generation(
                    remaster_id,
                    status="completed",
                    settings=settings,
                    output_paths=output_paths,
                )
                if image_session:
                    image_session.seal_history(remaster_id)
            if image_session:
                image_session.commit()
            else:
                shutil.rmtree(working_dir, ignore_errors=True)
            return RemasterResponse(
                id=remaster_id,
                status=result_status,
                source_generation_id=request.source_generation_id,
                source_image_path=request.source_image_path,
                source_output_index=request.source_output_index,
                output_path=str(final_path),
                contact_sheet_path=comparison.contact_sheet_path,
                report_path=comparison.report_path,
                settings=settings,
                model_slots=request.model_slot_overrides,
                loras=request.loras,
                runtime_job_id=str(runtime_result.get("runtime_job_id") or ""),
                runtime_job_ids=list(runtime_result.get("runtime_job_ids") or []),
                warnings=warnings,
            )
        except RuntimeCancellationConfirmed as exc:
            self._persist_cancelled(
                remaster_id,
                settings,
                [*failure_paths(), *self._working_paths(working_dir)],
                runtime_result,
            )
            raise RemasterExecutionError(remaster_id, "remaster_cancelled") from exc
        except RemasterExecutionError as exc:
            if exc.code == "remaster_cancelled":
                raise
            self._persist_failure(remaster_id, settings, exc.code, working_dir, runtime_result, failure_paths())
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            RuntimeErrorResponse,
            RuntimeAdapterError,
            OutputFinalizationError,
        ) as exc:
            code = "remaster_execution_failed"
            self._persist_failure(remaster_id, settings, code, working_dir, runtime_result, failure_paths())
            raise RemasterExecutionError(remaster_id, code) from exc
        finally:
            finalization_stack.close()

    def _runtime_request(self, request: RemasterRequest, validation: RemasterValidationResult) -> dict:
        resolved = validation.settings.model_dump(mode="json")
        return {
            "profile": request.profile,
            "expanded_prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "settings": {
                "init_image_path": request.source_image_path,
                "seed": request.seed,
                "batch_size": 1,
                "batch_count": 1,
                "cfg_scale": request.cfg_scale,
                "distilled_cfg_scale": request.distilled_cfg_scale,
                "hires_enabled": False,
                "remaster": resolved,
            },
            "model_slots": request.model_slot_overrides,
            "loras": [lora.model_dump(mode="json") for lora in request.loras],
            "embeddings": [],
        }

    def _finalize_runtime_output(
        self,
        remaster_id: int,
        request: RemasterRequest,
        validation: RemasterValidationResult,
        runtime_path: Path,
        output_root: Path,
    ) -> Path:
        final_dir = output_root / request.profile
        final_dir.mkdir(parents=True, exist_ok=True)
        staged_path = unique_output_path(final_dir / runtime_path.name)
        runtime_path.replace(staged_path)
        output_settings = UserSettingsRepository(self.config.data_dir / "user-settings.json").load().output
        paths = apply_output_image_format([str(staged_path)], output_settings.image_format)
        artifact_settings = {
            "filename_template": output_settings.filename_template,
            "generated_seeds": [request.seed],
            "seed": request.seed,
            "sampler": validation.settings.sampler,
            "scheduler": validation.settings.scheduler,
        }
        paths = apply_filename_template(paths, artifact_settings, request.profile, remaster_id)
        final_path = Path(paths[0])
        if not _is_file(final_path):
            raise OutputFinalizationError("Remaster final output was not persisted")
        return final_path

    def _comparison(
        self,
        remaster_id: int,
        request: RemasterRequest,
        validation: RemasterValidationResult,
        final_path: Path,
        runtime_result: dict,
    ) -> RemasterComparisonResult:
        if not validation.settings.create_comparison:
            return comparison_disabled_result()
        return self.comparison_builder(
            validation.source_path,
            final_path,
            self.config.data_dir / "reports",
            {
                "remaster_id": remaster_id,
                "profile": request.profile,
                "request": request.model_dump(mode="json"),
                "stage_sizes": {
                    "source": _size_dict(validation.source_size),
                    "anchor": _size_dict(validation.anchor_size),
                    "buffer": _size_dict(validation.buffer_size),
                    "target": _size_dict(validation.target_size),
                },
                "job_ids": list(runtime_result.get("runtime_job_ids") or []),
            },
        )

    def _persist_failure(
        self,
        remaster_id: int,
        settings: dict,
        code: str,
        working_dir: Path,
        runtime_result: dict,
        persisted_paths: list[str],
    ) -> None:
        settings["error_code"] = code
        working_paths = [
            str(path)
            for path in working_dir.rglob("*")
            if _is_file(path) and not is_private_image_candidate(path)
        ] if working_dir.exists() else []
        retained_paths = [*persisted_paths, *working_paths]
        retained_paths = [path for path in retained_paths if not is_private_image_candidate(path)]
        settings["working_output_paths"] = retained_paths
        settings["runtime_job_id"] = str(runtime_result.get("runtime_job_id") or "")
        self.history.update_generation(
            remaster_id,
            status="failed",
            settings=settings,
            output_paths=retained_paths,
        )

    def _persist_cancelled(
        self,
        remaster_id: int,
        settings: dict,
        retained_paths: list[str],
        runtime_result: dict,
    ) -> None:
        settings["remaster_result_status"] = "cancelled"
        settings["runtime_job_id"] = str(runtime_result.get("runtime_job_id") or "")
        retained_paths = [path for path in retained_paths if not is_private_image_candidate(path)]
        settings["working_output_paths"] = retained_paths
        self.history.update_generation(
            remaster_id,
            status="cancelled",
            settings=settings,
            output_paths=retained_paths,
        )

    def _working_paths(self, working_dir: Path) -> list[str]:
        if not working_dir.exists():
            return []
        return [str(path) for path in working_dir.rglob("*") if _is_file(path)]


def validate_remaster_request(
    request: RemasterRequest,
    config: AppConfig,
    object_info: dict,
) -> RemasterValidationResult:
    if request.runtime != "comfy_workflow":
        raise RemasterValidationError(
            api_errors.REMASTER_RUNTIME_NOT_SUPPORTED,
            runtime=request.runtime,
        )

    source_path = Path(request.source_image_path)
    if _inside_documentation(source_path, config.root_dir) or not _is_file(source_path):
        raise RemasterValidationError(api_errors.REMASTER_SOURCE_INVALID)
    try:
        with Image.open(source_path) as image:
            image.verify()
        with Image.open(source_path) as image:
            source_size = image.size
    except (OSError, ValueError, TypeError) as exc:
        raise RemasterValidationError(api_errors.REMASTER_SOURCE_INVALID) from exc

    source_width, source_height = source_size
    if not (256 <= source_width <= 4096 and 256 <= source_height <= 4096):
        raise RemasterValidationError(
            api_errors.REMASTER_DIMENSIONS_UNSUPPORTED,
            stage="source",
            width=source_width,
            height=source_height,
        )

    settings = resolve_remaster_settings(request.profile, request.remaster)
    option = upscaler_option(settings.anchor_upscaler)
    model_name = upscaler_model_name(settings.anchor_upscaler)
    if option.kind != "model" or not model_name:
        raise RemasterValidationError(api_errors.REMASTER_UPSCALER_UNAVAILABLE)

    dimension_step = 16 if request.profile == "flux2" else 8
    target_size = (
        _round_to_step(source_width * settings.target_scale, dimension_step),
        _round_to_step(source_height * settings.target_scale, dimension_step),
    )
    buffer_size = (
        _round_to_step(target_size[0] * settings.buffer_ratio, dimension_step),
        _round_to_step(target_size[1] * settings.buffer_ratio, dimension_step),
    )
    anchor_size = (source_width * 4, source_height * 4)
    if (
        min(buffer_size) < 8
        or max(target_size) > 8192
        or target_size[0] * target_size[1] > config.remaster_max_final_pixels
        or anchor_size[0] * anchor_size[1] > config.remaster_max_anchor_pixels
    ):
        raise RemasterValidationError(
            api_errors.REMASTER_DIMENSIONS_UNSUPPORTED,
            source_size=source_size,
            anchor_size=anchor_size,
            buffer_size=buffer_size,
            target_size=target_size,
        )

    nodes = {name for name, value in object_info.items() if isinstance(name, str) and isinstance(value, dict)}
    missing_nodes = sorted(REMASTER_REQUIRED_NODES - nodes)
    if request.profile == "flux2" and settings.denoise < 1 and "SplitSigmas" not in nodes:
        missing_nodes.append("SplitSigmas")
    if missing_nodes:
        raise RemasterValidationError(
            api_errors.MISSING_REMASTER_RUNTIME_NODES,
            missing_nodes=sorted(set(missing_nodes)),
        )

    available_models = _node_input_values(object_info, "UpscaleModelLoader", "model_name")
    normalized_model = model_name.replace("/", "\\")
    if not available_models or normalized_model not in {value.replace("/", "\\") for value in available_models}:
        raise RemasterValidationError(
            api_errors.REMASTER_UPSCALER_UNAVAILABLE,
            model_name=model_name,
        )

    return RemasterValidationResult(
        source_path=source_path,
        source_size=source_size,
        anchor_size=anchor_size,
        buffer_size=buffer_size,
        target_size=target_size,
        anchor_model_name=model_name,
        settings=settings,
    )


def _round_to_step(value: int | float, step: int = 8) -> int:
    return int(float(value) / step + 0.5) * step


def _size_dict(size: tuple[int, int]) -> dict[str, int]:
    return {"width": size[0], "height": size[1]}


def _is_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def _inside_documentation(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to((root_dir / "docs").resolve())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _node_input_values(object_info: dict, node_name: str, input_name: str) -> set[str]:
    node = object_info.get(node_name)
    if not isinstance(node, dict):
        return set()
    required = ((node.get("input") or {}).get("required") or {})
    input_spec = required.get(input_name)
    if not isinstance(input_spec, list) or not input_spec:
        return set()
    values = input_spec[0]
    if values == "COMBO" and len(input_spec) > 1 and isinstance(input_spec[1], dict):
        values = input_spec[1].get("options")
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}
