"""Complete ordinary Comfy batches before the shared final-image decision."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

from app.adapters.image_check_llm import check_image_with_llm
from app.adapters.image_check_codex import check_image_with_codex
from app.services.image_check_finalization import finalize_completed_batch


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
                    output_root=operation.output_root, cancel_event=event, generated_final=True)
            return await check_image_with_llm(operation.settings, operation.providers, path,
                output_root=operation.output_root, cancel_event=event, generated_final=True)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    return asyncio.run(run())


def run_checked_generation(runtime, request: dict, output_dir: Path) -> dict:
    from app.services.runtime import RuntimeCancellationConfirmed

    operation = request["_image_check_operation"]
    if not output_dir.resolve().is_relative_to(operation.output_root.resolve()):
        raise ValueError("image_check_output_root_mismatch")
    settings = request["settings"]
    requested_batch_size = int(settings.get("batch_size", 1))
    batch_size = requested_batch_size
    batch_count = int(settings.get("batch_count", 1))
    topaz = runtime._uses_topaz_hires(request)
    if topaz:
        batch_count *= batch_size
        batch_size = 1
    seed = int(settings.get("seed", -1))
    records, paths, primary, stages, auxiliary, jobs, seeds, noises = [], [], [], [], [], [], [], []
    paused = cancelled = False
    unjudged = 0
    progress = request.get("progress_callback")
    for batch_index in range(batch_count):
        iteration = deepcopy(request)
        iteration.pop("_image_check_operation", None)
        iteration.pop("_image_check_capture", None)
        iteration["settings"].update(batch_size=batch_size, batch_count=1,
                                    seed=runtime._iteration_seed(seed, batch_index))
        iteration["_batch_index_offset"] = batch_index

        def batch_progress(current, total, preview_path="", step=None, steps=None, phase=""):
            # Only the runtime's live preview frames may be shown before judgment.
            preview = preview_path if preview_path and Path(preview_path).parent.name == "_previews" else ""
            runtime._publish_generation_progress(progress, batch_index, batch_count, preview, step, steps, phase)

        iteration["progress_callback"] = batch_progress
        try:
            runtime._raise_if_generation_cancelled(request)
            runtime._prepare_spatial_runtime_masks(iteration)
            completed = (runtime._generate_topaz_hires(iteration, output_dir) if topaz
                         else runtime._generate(iteration, output_dir))
            runtime._merge_builder_runtime_settings(request, iteration)
            for key, value in iteration["settings"].items():
                if key.startswith("topaz_"):
                    settings[key] = value
            current_seeds = list(completed.get("generated_seeds", []))
            seeds.extend(current_seeds)
            jobs.extend(completed.get("runtime_job_ids", []))
            noises.extend(iteration["settings"].get("sampler_noise_runtime", []))
            for image_index, item in enumerate(completed.get("primary_outputs", [])):
                item["output_index"] = batch_index * batch_size + image_index
            prepare = request.get("_prepare_final_image_batch")
            if callable(prepare):
                completed = prepare(completed, iteration["settings"], output_dir)
            checked = finalize_completed_batch(operation, completed, iteration["settings"], batch_index=batch_index,
                checker=runtime.image_checker or assess_image, cancel_requested=request.get("cancel_requested"),
                progress=lambda phase: batch_progress(0, 1, phase=phase))
            if checked['status'] == 'cancelled' and not records and not checked['image_check_results']:
                raise RuntimeCancellationConfirmed()
            if topaz:
                for item in checked['primary_outputs']:
                    item['word_map_row'] = item['image_index']
                for item in [*checked["image_check_results"], *checked["primary_outputs"], *checked["enhancement_stages"]]:
                    item["batch_index"] = batch_index // requested_batch_size
                    item["image_index"] = batch_index % requested_batch_size
            records.extend(checked["image_check_results"])
            paths.extend(checked["output_paths"])
            primary.extend(checked["primary_outputs"])
            stages.extend(checked["enhancement_stages"])
            auxiliary.extend(checked["approved_auxiliary_paths"])
            unjudged += checked["image_check_unjudged"]
            paused = checked["image_check_pause"]
            cancelled = checked["status"] == "cancelled"
        except RuntimeCancellationConfirmed:
            if not records:
                raise
            cancelled = True
        if paused or cancelled:
            break
        runtime._publish_generation_progress(progress, batch_index + 1, batch_count)
    passed = sum(item["status"] == "pass" for item in records)
    settings["image_check"] = {
        "settings": operation.settings.model_dump(mode="json"), "target_stage": "final_output",
        "results": records, "status": "unknown" if paused else "pass" if passed else "reject",
        "passed": passed, "rejected": sum(item["status"] == "reject" for item in records),
        "unjudged": unjudged, "pause": paused or cancelled, "cancelled": cancelled,
        "auxiliary_paths": auxiliary,
    }
    settings["generated_seeds"] = seeds
    settings["sampler_noise_runtime"] = noises
    if seed < 0 and seeds:
        settings["seed"] = seeds[0]
    return {"status": "cancelled" if cancelled else "completed", "output_paths": paths,
            "primary_outputs": primary, "enhancement_stages": stages, "image_check_finalized": True,
            "runtime_adapter": "comfy_workflow", "runtime_job_id": jobs[0] if jobs else "",
            "runtime_job_ids": jobs, "comfy_prompt_id": jobs[0] if jobs else "", "comfy_prompt_ids": jobs,
            "generated_seeds": seeds}
