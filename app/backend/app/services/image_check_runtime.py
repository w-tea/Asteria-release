"""The real base-image/check/enhancement boundary, owned by the Comfy adapter."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
from uuid import uuid4

from app.adapters.image_check_llm import check_image_with_llm
from app.adapters.image_check_codex import check_image_with_codex
from app.services.image_check import decide_image_check
from app.services.image_check_workflow import split_image_check_workflow
from app.services.comfy_sampler_receipts import sampler_noise_receipts
from app.services.comfy_prompt_trace import instrument_prompt_encoders, record_prompt_execution


def assess_image(operation, path: Path, cancel_requested):
    async def run():
        event = asyncio.Event()

        async def watch_cancel():
            while not event.is_set():
                if callable(cancel_requested) and cancel_requested():
                    event.set()
                    return
                await asyncio.sleep(0.1)

        watcher = asyncio.create_task(watch_cancel())
        try:
            if operation.settings.method == "codex":
                return await check_image_with_codex(operation.settings, path,
                    output_root=operation.output_root, cancel_event=event)
            return await check_image_with_llm(operation.settings, operation.providers, path,
                                             output_root=operation.output_root, cancel_event=event)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    return asyncio.run(run())


def _stage_images(directory: Path, history: dict, prompt_id: str, batch_size: int):
    path = directory / "stage.json"
    if path.stat().st_size > 128 * 1024:
        raise ValueError("image_check_stage_manifest_invalid")
    body = path.read_bytes()
    digest = sha256(body).hexdigest()
    receipt = history.get(prompt_id, {}).get("outputs", {}).get("asteria_image_check_stage", {})
    if receipt.get("asteria_generation_stage") != [digest]:
        raise ValueError("image_check_stage_receipt_mismatch")
    manifest = json.loads(body)
    images = manifest.get("images", [])
    if (manifest.get("schema_version") != "asteria.generation-stage.v1"
            or manifest.get("batch_size") != batch_size or len(images) != batch_size):
        raise ValueError("image_check_stage_batch_mismatch")
    result = []
    for index, item in enumerate(images):
        if item.get("index") != index or item.get("filename") != f"base-{index:04}.png":
            raise ValueError("image_check_stage_image_invalid")
        image = directory / item["filename"]
        if image.is_symlink() or image.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("image_check_stage_image_invalid")
        if sha256(image.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("image_check_stage_image_changed")
        result.append(image)
    return digest, result


def run_checked_generation(runtime, request: dict, output_dir: Path) -> dict:
    from app.services.runtime import RuntimeCancellationConfirmed

    operation = request["_image_check_operation"]
    nodes = runtime.client.object_info()
    if not {"AsteriaSaveGenerationStage", "AsteriaLoadGenerationStage",
            "AsteriaGenerationStageSampler", "AsteriaSelectedGenerationNoise",
            "AsteriaSelectedGenerationSampler"}.issubset(nodes):
        raise ValueError("image_check_stage_not_available")
    root = operation.output_root.resolve()
    if not output_dir.resolve().is_relative_to(root):
        raise ValueError("image_check_output_root_mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_parent = root / ".image-check"
    stage_parent.mkdir(exist_ok=True)
    if stage_parent.resolve() != stage_parent:
        raise ValueError("image_check_stage_root_mismatch")
    runtime._prepare_control_images(request, output_dir / "controlnet")
    runtime._prepare_anima_lllite_images(request)
    settings = request["settings"]
    topaz = runtime._uses_topaz_hires(request)
    batch_size = int(settings.get("batch_size", 1))
    batch_count = int(settings.get("batch_count", 1))
    if topaz:
        batch_count *= batch_size
        batch_size = 1
    seed = int(settings.get("seed", -1))
    results, paths, primary, prompt_ids, seeds, noise_receipts = [], [], [], [], [], []
    enhancement_stages = []
    checker = runtime.image_checker or assess_image
    paused = False
    cancelled = False
    for batch_index in range(batch_count):
        def continuation_progress(current, total, preview_path="", step=None, steps=None, phase=""):
            completed = batch_index + int(total > 0 and current >= total)
            runtime._publish_generation_progress(request.get("progress_callback"), completed,
                                                 batch_count, preview_path, step, steps, phase)

        try:
            runtime._raise_if_generation_cancelled(request)
        except RuntimeCancellationConfirmed:
            if not paths:
                raise
            cancelled = True
            break
        iteration = deepcopy(request)
        iteration.pop("_image_check_operation", None)
        iteration["_image_check_capture"] = True
        current = iteration["settings"]
        current.update(batch_count=1, batch_size=batch_size, seed=runtime._iteration_seed(seed, batch_index))
        # Topaz's first pass has no enhancement; its existing adapter owns stage two.
        if topaz:
            current["hires_enabled"] = False
            current["refiner_enabled"] = False
            current["restore_faces"] = False
            current["detailer_units"] = []
        full = runtime.workflow_builder.build(iteration)
        if request.get("_record_prompt_inputs", True):
            instrument_prompt_encoders(full, capture_words=bool(request.get('_word_maps')))
        runtime._merge_builder_runtime_settings(request, iteration)
        directory = stage_parent / uuid4().hex
        directory.mkdir()
        try:
            (directory / "owner.json").write_text(json.dumps({
                "schema_version": "asteria.generation-stage.v1", "output_root": str(root),
                "operation_id": directory.name,
            }), encoding="utf-8")
            split = split_image_check_workflow(full, iteration["_image_check_boundary"], directory)
            with runtime._preview_listener(
                output_dir / "_previews",
                runtime._preview_callback(request.get("progress_callback"), batch_count, {"index": batch_index}),
                runtime._step_callback(request.get("progress_callback"), batch_count, {"index": batch_index}),
            ):
                prompt_id = runtime.client.queue_prompt(split.first)
                prompt_ids.append(prompt_id)
                noise_receipts.extend(sampler_noise_receipts(split.first, prompt_id))
                callback = request.get("runtime_job_callback")
                if callable(callback):
                    callback(prompt_id)
                stop = request.get("runtime_stop_confirmed")
                kwargs = {"timeout_seconds": None, "runtime_stop_confirmed": stop} if callable(stop) else {}
                history = runtime.client.wait_for_history(prompt_id, **kwargs)
            runtime._raise_if_generation_cancelled(request)
            if request.get("_record_prompt_inputs", True):
                record_prompt_execution(iteration, split.first, history, prompt_id, batch_index=batch_index)
                runtime._merge_builder_runtime_settings(request, iteration)
            digest, images = _stage_images(directory, history, prompt_id, batch_size)
            passed = []
            for index, image in enumerate(images):
                runtime._raise_if_generation_cancelled(request)
                # The judgement sees the same visible image as inpaint output, not its crop alone.
                if current.get("mode") == "inpaint":
                    if current.get("inpaint_runtime") == "sdxl_only_masked_crop_composite":
                        from app.services.inpaint_region import compose_only_masked_output
                        compose_only_masked_output(image,
                            canonical_source_path=current["inpaint_canonical_source_path"],
                            positive_mask_crop_path=current["inpaint_positive_mask_crop_path"],
                            crop_rect=current["inpaint_crop_rect"])
                    else:
                        runtime._composite_inpaint_output(image, current)
                runtime._publish_generation_progress(request.get("progress_callback"), batch_index,
                                                     batch_count, str(image), phase="image_check")
                checked = checker(operation, image, request.get("cancel_requested"))
                runtime._raise_if_generation_cancelled(request)
                decision = decide_image_check(operation.settings, checked.assessment)
                record = {
                    "batch_index": batch_index, "image_index": index, "seed": current["seed"],
                    "runtime_job_id": prompt_id, "status": decision.status,
                    "stage_sha256": digest, "source_sha256": sha256(image.read_bytes()).hexdigest(),
                    "assessment": checked.assessment.model_dump(mode="json") if checked.assessment else None,
                    "receipt": asdict(checked.receipt),
                }
                results.append(record)
                if decision.continue_postprocessing:
                    passed.append(index)
                elif decision.save_rejected_image:
                    saved = output_dir / f"{uuid4().hex}_rejected.png"
                    shutil.copyfile(image, saved)
                    paths.append(str(saved))
                    record["saved_path"] = str(saved)
                if decision.pause:
                    paused = True
                    break
            if passed:
                runtime._raise_if_generation_cancelled(request)
                if topaz:
                    if runtime.topaz_executable is None or not runtime.topaz_executable.is_file():
                        raise ValueError("image_check_topaz_executable_missing")
                    accepted_paths = []
                    for index in passed:
                        saved = output_dir / f"{uuid4().hex}_first_pass.png"
                        shutil.copyfile(images[index], saved)
                        accepted_paths.append(str(saved))
                    continuation = deepcopy(request)
                    continuation.pop("_image_check_operation", None)
                    continuation["progress_callback"] = continuation_progress
                    continuation["settings"].update(batch_size=1, batch_count=1, seed=current["seed"])
                    first_result = {"output_paths": accepted_paths, "generated_seeds": [current["seed"]],
                                    "runtime_job_id": prompt_id, "runtime_job_ids": [prompt_id]}
                    result = runtime._generate_topaz_hires(continuation, output_dir, first_result=first_result)
                else:
                    continuation = deepcopy(iteration)
                    continuation.pop("_image_check_capture", None)
                    continuation["progress_callback"] = continuation_progress
                    continuation["_prepared_workflow"] = split.continuation(passed, digest)
                    result = runtime._generate(continuation, output_dir)
                runtime._merge_builder_runtime_settings(request, continuation)
                paths.extend(result["output_paths"])
                enhancement_stages.extend({**item, 'batch_index': batch_index,
                                           'image_index': passed[item['image_index']]}
                                          for item in result.get('enhancement_stages', []))
                primary_offset = len(primary)
                primary.extend({**item, "output_index": primary_offset + i,
                                **({'batch_index': batch_index, 'image_index': passed[i]}
                                   if settings.get('record_enhancement_stages') or request.get('_word_maps') else {})}
                               for i, item in enumerate(result.get("primary_outputs", [])))
                prompt_ids.extend(p for p in result.get("runtime_job_ids", []) if p != prompt_id)
                seeds.extend(result.get("generated_seeds", []))
                for key, value in continuation["settings"].items():
                    if key.startswith("topaz_"):
                        settings[key] = value
                noise_receipts.extend(continuation["settings"].get("sampler_noise_runtime", []))
        except RuntimeCancellationConfirmed:
            if not paths:
                raise
            cancelled = True
        finally:
            # Only this operation's random, root-checked directory is removed.
            if directory.resolve().parent == stage_parent:
                shutil.rmtree(directory)
        if paused or cancelled:
            break
        runtime._publish_generation_progress(request.get("progress_callback"), batch_index + 1,
                                             batch_count, primary[-1]["path"] if primary else "")
    passed_count = sum(item["status"] == "pass" for item in results)
    settings["image_check"] = {
        "settings": operation.settings.model_dump(mode="json"), "results": results,
        "status": "unknown" if paused else "pass" if passed_count else "reject",
        "passed": passed_count, "rejected": sum(item["status"] == "reject" for item in results),
        "pause": paused or cancelled, "cancelled": cancelled,
    }
    settings["generated_seeds"] = seeds
    settings["sampler_noise_runtime"] = noise_receipts
    if seed < 0 and results:
        settings["seed"] = results[0]["seed"]
    return {"status": "cancelled" if cancelled else "completed", "output_paths": paths, "primary_outputs": primary,
            "enhancement_stages": enhancement_stages,
            "runtime_adapter": "comfy_workflow", "runtime_job_id": prompt_ids[0],
            "runtime_job_ids": prompt_ids, "comfy_prompt_id": prompt_ids[0],
            "comfy_prompt_ids": prompt_ids, "generated_seeds": seeds}
