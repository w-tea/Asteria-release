"""Adapter-owned input observation. Never infer an image effect from token weights."""
from __future__ import annotations

from copy import deepcopy
import json

ENCODERS = {"CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner", "CLIPTextEncodeFlux", "TextEncodeQwenImageEdit", "AnimaQwen35UnifiedPrompt"}
TRACE_NODES = {"AsteriaPromptEncodeTrace", "AsteriaAnimaPromptTrace"}
MAX_BYTES = 4 * 1024 * 1024


def instrument_prompt_encoders(workflow: dict, *, capture_words: bool = False) -> None:
    for node in workflow.values():
        if capture_words and node.get('class_type') in TRACE_NODES:
            node['inputs']['capture_words'] = True
        if node.get("class_type") not in ENCODERS:
            continue
        inputs = node["inputs"]
        unified = node["class_type"] == "AnimaQwen35UnifiedPrompt"
        connections = {"model", "native_clip", "qwen35_clip"} if unified else {"clip", "vae", "image"}
        if any(isinstance(v, list) for k, v in inputs.items() if k not in connections):
            continue
        node["inputs"] = {**{k: v for k, v in inputs.items() if k in connections},
                          **({} if unified else {"encoder": node["class_type"]}),
                          "parameters": json.dumps({k: v for k, v in inputs.items() if k not in connections}, ensure_ascii=False)}
        node["class_type"] = "AsteriaAnimaPromptTrace" if unified else "AsteriaPromptEncodeTrace"
        if capture_words:
            node['inputs']['capture_words'] = True


def _uses(workflow: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}

    def walk(ref, usage, operations, ancestors):
        if not isinstance(ref, list) or len(ref) != 2 or str(ref[0]) not in workflow:
            return
        key = str(ref[0])
        if key in ancestors:
            return
        node = workflow[key]
        inputs = node.get("inputs", {})
        if node.get("class_type") in TRACE_NODES:
            result.setdefault(key, []).append({**usage, "operations": operations})
            return
        operation = {"type": node.get("class_type"), **{k: v for k, v in inputs.items() if not isinstance(v, (list, dict))}}
        for name, value in inputs.items():
            if name in {"model", "clip", "vae", "control_net", "image", "mask"}:
                continue
            walk(value, usage, [*operations, operation], {*ancestors, key})

    for key, node in workflow.items():
        inputs = node.get("inputs", {})
        if "guider" in inputs:
            guider = workflow.get(str(inputs["guider"][0]), {}).get("inputs", {})
            scheduler = workflow.get(str(inputs.get("sigmas", [""])[0]), {}).get("inputs", {})
            for polarity, field in (("positive", "positive"), ("negative", "negative"), ("positive", "conditioning")):
                if field in guider:
                    walk(guider[field], {"stage": node.get("_meta", {}).get("asteria_stage", "base"), "consumer": key,
                         "polarity": polarity, "sampling": {k: scheduler[k] for k in ("steps", "denoise") if k in scheduler}}, [], set())
            continue
        if not ("positive" in inputs and "negative" in inputs) or "guider" in node.get("class_type", "").lower():
            continue
        if not ("sampler_name" in inputs or "Sampler" in node.get("class_type", "") or "Detailer" in node.get("class_type", "") or "guider" in node.get("class_type", "").lower()):
            continue
        sampling = json.loads(inputs["sampling"]) if node.get("class_type") == "AsteriaGenerationStageSampler" else inputs
        for polarity in ("positive", "negative"):
            usage = {"stage": node.get("_meta", {}).get("asteria_stage", "base"),
                     "consumer": key, "polarity": polarity,
                     "sampling": {k: sampling[k] for k in ("steps", "start_at_step", "end_at_step", "denoise") if k in sampling}}
            walk(inputs[polarity], usage, [], set())
    return result


def record_prompt_execution(request: dict, workflow: dict, history: dict, prompt_id: str, *, batch_index: int = 0) -> None:
    settings = request.setdefault("settings", {})
    records = settings.setdefault("prompt_execution", [])
    # A bounded record must not become a new generation input limit.
    size = len(json.dumps(records, ensure_ascii=False).encode("utf-8"))
    if size > MAX_BYTES:
        settings["prompt_execution_truncated"] = True
        return
    output = history.get(prompt_id, {}).get("outputs", {})
    uses = _uses(workflow)
    encodings, missing = [], []
    for node_id, node in workflow.items():
        if node.get("class_type") not in TRACE_NODES:
            continue
        values = output.get(node_id, {}).get("asteria_prompt_trace", [])
        if not values:
            missing.append(node_id)
            continue
        for value in values:
            if not isinstance(value, dict) or value.get("version") != 1:
                missing.append(node_id)
                continue
            item = {"node_id": node_id, "receipt": value, "uses": uses.get(node_id, [])}
            item_size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if size + item_size > MAX_BYTES:
                missing.append(node_id)
                settings["prompt_execution_truncated"] = True
                continue
            size += item_size
            encodings.append(deepcopy(item))
    record = {"version": 1, "runtime_job_id": prompt_id, "batch_index": batch_index,
                    "status": "partial" if missing or any(x["receipt"].get("partial") for x in encodings) else "recorded" if encodings else "unavailable",
                    "encodings": encodings, "missing": list(dict.fromkeys(missing))}
    # Include JSON delimiters and job/missing-node metadata in the real byte cap.
    while len(json.dumps([*records, record], ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        settings["prompt_execution_truncated"] = True
        record["status"] = "partial"
        if record["encodings"]:
            record["encodings"].pop()
        elif record["missing"]:
            record["missing"].pop()
        else:
            return
    records.append(record)
