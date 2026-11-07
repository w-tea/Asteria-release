"""Record executed noise settings, including the image-check sampler wrapper."""
from __future__ import annotations

import json


def sampler_noise_receipts(workflow: dict, prompt_id: str) -> list[dict]:
    receipts = []
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        wrapped = node["class_type"] == "AsteriaGenerationStageSampler"
        if wrapped:
            inputs = json.loads(inputs["sampling"])
        if "noise_mode" not in inputs or "sampler_name" not in inputs:
            continue
        initial_seed = inputs.get("seed", inputs.get("noise_seed"))
        receipt = {
            "runtime_job_id": prompt_id, "node_id": node_id, "class_type": node["class_type"],
            "noise_mode": inputs["noise_mode"], "initial_noise_seed": initial_seed,
            "sampling_noise_seed": inputs["internal_seed"] if "internal_seed" in inputs["noise_mode"] else initial_seed,
        }
        if wrapped:
            receipt["source_sampler_class"] = inputs["class_type"]
        receipts.append(receipt)
    return receipts
