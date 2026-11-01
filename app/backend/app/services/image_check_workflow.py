"""Adapter-owned split at the original base latent, before enhancement nodes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path


def _stage_sampler(node: dict, boundary: dict, selection=None) -> dict:
    inputs, parameters = {}, {"class_type": node["class_type"]}
    for key, value in node["inputs"].items():
        if key in {"model", "positive", "negative", "latent_image", "noise_opt"}:
            inputs[key] = value
        else:
            parameters[key] = value
    inputs["sampling"] = json.dumps(parameters, separators=(",", ":"))
    if selection is not None:
        inputs["selection"] = selection
    return {"class_type": boundary["sampler_class"], "inputs": inputs,
            "_meta": deepcopy(node.get("_meta", {}))}


def _ancestors(workflow: dict, roots: list[str]) -> dict:
    selected: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in selected:
            return
        if node_id not in workflow:
            raise ValueError("image_check_workflow_reference_missing")
        selected.add(node_id)
        for value in workflow[node_id].get("inputs", {}).values():
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str) and isinstance(value[1], int)):
                visit(value[0])

    for root in roots:
        visit(root)
    return {key: deepcopy(node) for key, node in workflow.items() if key in selected}


@dataclass(frozen=True)
class ImageCheckWorkflow:
    first: dict
    original: dict
    boundary: dict
    staging_dir: Path

    def continuation(self, indices: list[int], stage_sha256: str) -> dict:
        if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("image_check_indices_invalid")
        workflow = deepcopy(self.original)
        base_id = self.boundary["samples"][0]
        workflow[base_id] = {
            "class_type": self.boundary.get("load_class", "AsteriaLoadGenerationStage"),
            "inputs": {
                "staging_dir": str(self.staging_dir),
                "indices": json.dumps(indices, separators=(",", ":")),
                "stage_sha256": stage_sha256,
            },
        }
        workflow = _ancestors(workflow, self.boundary["output_nodes"])
        if indices != list(range(self.boundary["batch_size"])):
            # VAEEncode intentionally drops latent metadata. Carry selection as
            # a separate graph edge, not an index that silently disappears there.
            selection = [base_id, 1]
            for node_id, node in list(workflow.items()):
                if node["class_type"] in {"KSampler", "KSampler //Inspire", "KSamplerAdvanced", "KSamplerAdvanced //Inspire"}:
                    workflow[node_id] = _stage_sampler(node, self.boundary, selection)
                elif node["class_type"] == "SamplerCustomAdvanced":
                    noise_id = f"asteria_checked_noise_{node_id}"
                    if noise_id in workflow:
                        raise ValueError("image_check_workflow_node_collision")
                    workflow[noise_id] = {"class_type": self.boundary["noise_class"], "inputs": {
                        "noise": node["inputs"]["noise"], "selection": selection,
                    }}
                    node["inputs"]["noise"] = [noise_id, 0]
                    sampler_id = f"asteria_checked_sampler_{node_id}"
                    if sampler_id in workflow:
                        raise ValueError("image_check_workflow_node_collision")
                    workflow[sampler_id] = {"class_type": self.boundary["sample_filter_class"], "inputs": {
                        "sampler": node["inputs"]["sampler"], "selection": selection,
                    }}
                    node["inputs"]["sampler"] = [sampler_id, 0]
        return workflow


def split_image_check_workflow(workflow: dict, boundary: dict, staging_dir: Path) -> ImageCheckWorkflow:
    samples = boundary.get("samples")
    vae = boundary.get("vae")
    if (not isinstance(samples, list) or len(samples) != 2 or samples[1] != 0
            or samples[0] not in workflow or not isinstance(vae, list) or len(vae) != 2
            or vae[0] not in workflow or not boundary.get("output_nodes")):
        raise ValueError("image_check_workflow_boundary_invalid")
    first = _ancestors(workflow, [samples[0], vae[0]])
    stage_id = "asteria_image_check_stage"
    if stage_id in workflow:
        raise ValueError("image_check_workflow_node_collision")
    first[stage_id] = {
        "class_type": boundary.get("save_class", "AsteriaSaveGenerationStage"),
        "inputs": {"samples": list(samples), "vae": list(vae), "staging_dir": str(staging_dir)},
    }
    if first[samples[0]]["inputs"].get("return_with_leftover_noise") in (True, "enable"):
        # A split base/refiner schedule yields noisy solver state. Judge its
        # already-computed x0 preview, while saving the untouched solver latent.
        first[samples[0]] = _stage_sampler(first[samples[0]], boundary)
        first[stage_id]["inputs"]["check_samples"] = [samples[0], 1]
    return ImageCheckWorkflow(first, deepcopy(workflow), deepcopy(boundary), staging_dir)
