from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
from typing import Any

import numpy as np
import scipy.ndimage
import torch

try:
    from .calculation_modes import (
        ASTERIA_CALCULATION_MODES,
        CALCULATION_CONTRACT_VERSION,
        CALCULATION_IMPLEMENTATION_REVISION,
        CALCULATION_PARAMETER_SCHEMA_VERSION,
        V11_CALCULATION_MODES,
        V12_CALCULATION_CONTRACT_VERSION,
        V12_CALCULATION_IMPLEMENTATION_REVISION,
        V12_CALCULATION_PARAMETER_SCHEMA_VERSION,
        CalculationModeError,
        calculation_implementation_revision_supported,
        merge_weight_sum_tensor,
    )
except ImportError:
    _calculation_path = Path(__file__).with_name("calculation_modes.py")
    _calculation_spec = importlib.util.spec_from_file_location(
        "asteria_merge_runtime_calculation_modes",
        _calculation_path,
    )
    if _calculation_spec is None or _calculation_spec.loader is None:
        raise RuntimeError("Asteria Merge calculation modes could not be loaded")
    _calculation_module = importlib.util.module_from_spec(_calculation_spec)
    _calculation_spec.loader.exec_module(_calculation_module)
    CALCULATION_CONTRACT_VERSION = _calculation_module.CALCULATION_CONTRACT_VERSION
    CALCULATION_IMPLEMENTATION_REVISION = (
        _calculation_module.CALCULATION_IMPLEMENTATION_REVISION
    )
    CALCULATION_PARAMETER_SCHEMA_VERSION = (
        _calculation_module.CALCULATION_PARAMETER_SCHEMA_VERSION
    )
    V11_CALCULATION_MODES = _calculation_module.V11_CALCULATION_MODES
    V12_CALCULATION_CONTRACT_VERSION = (
        _calculation_module.V12_CALCULATION_CONTRACT_VERSION
    )
    V12_CALCULATION_IMPLEMENTATION_REVISION = (
        _calculation_module.V12_CALCULATION_IMPLEMENTATION_REVISION
    )
    V12_CALCULATION_PARAMETER_SCHEMA_VERSION = (
        _calculation_module.V12_CALCULATION_PARAMETER_SCHEMA_VERSION
    )
    ASTERIA_CALCULATION_MODES = _calculation_module.ASTERIA_CALCULATION_MODES
    CalculationModeError = _calculation_module.CalculationModeError
    calculation_implementation_revision_supported = (
        _calculation_module.calculation_implementation_revision_supported
    )
    merge_weight_sum_tensor = _calculation_module.merge_weight_sum_tensor


CONTRACT_VERSION = "asteria.merge.runtime.v1"
SDXL_BLOCKS = ["BASE", *[f"IN{x:02d}" for x in range(17)], "M00", *[f"OUT{x:02d}" for x in range(17)]]
ANIMA_BLOCKS = ["BASE", *[f"B{x:03d}" for x in range(28)], *[f"LLM{x:03d}" for x in range(6)], "OUT"]
BLOCK_SCHEMAS = {"sdxl": SDXL_BLOCKS, "anima": ANIMA_BLOCKS}
SELF_SKIP_KEY_PARTS = ("to_out", "proj_out", "norm")
SUPER_MERGER_FINETUNE_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.input_blocks.0.0.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
    "model.diffusion_model.out.2.weight",
    "model.diffusion_model.out.2.bias",
]
ADJUST_COLORS_SDXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]
ADJUST_COLORS_ANIMA = [[-1, 1 / 3, 2 / 3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]


class MergeRuntimeMathError(ValueError):
    pass


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise MergeRuntimeMathError("incompatible merge runtime contract")
    family = payload.get("model_family")
    if family not in BLOCK_SCHEMAS or payload.get("block_schema") != family:
        raise MergeRuntimeMathError("unsupported merge runtime family")
    if payload.get("merge_mode") not in {"weight_sum", "add_difference", "triple_sum", "sum_twice"}:
        raise MergeRuntimeMathError("unsupported merge mode")
    calculation_mode = payload.get("calculation_mode_id") or payload.get("calculation_mode")
    if calculation_mode in ASTERIA_CALCULATION_MODES:
        if payload.get("calculation_mode_id") != payload.get("calculation_mode"):
            raise MergeRuntimeMathError("calculation mode identity mismatch")
        if payload.get("merge_mode") != "weight_sum":
            raise MergeRuntimeMathError(
                "Asteria calculation modes require weight_sum"
            )
        if calculation_mode in V11_CALCULATION_MODES:
            contract_version = CALCULATION_CONTRACT_VERSION
            parameter_schema_version = CALCULATION_PARAMETER_SCHEMA_VERSION
        else:
            contract_version = V12_CALCULATION_CONTRACT_VERSION
            parameter_schema_version = V12_CALCULATION_PARAMETER_SCHEMA_VERSION
        if payload.get("calculation_contract_version") != contract_version:
            raise MergeRuntimeMathError("incompatible calculation contract")
        if (
            payload.get("calculation_parameter_schema_version")
            != parameter_schema_version
        ):
            raise MergeRuntimeMathError("incompatible calculation parameter schema")
        if not calculation_implementation_revision_supported(
            calculation_mode,
            str(payload.get("calculation_implementation_revision") or ""),
        ):
            raise MergeRuntimeMathError("incompatible calculation implementation")


def _canonical_key(key: str, domain: str) -> str:
    if domain == "model" and key.startswith("diffusion_model."):
        return f"model.{key}"
    if domain == "clip":
        lowered = key.lower()
        if lowered.startswith("clip_g."):
            return f"conditioner.embedders.1.{key[len('clip_g.'):]}"
        if lowered.startswith("clip_l."):
            return f"conditioner.embedders.0.{key[len('clip_l.'):]}"
        if not lowered.startswith(("conditioner.", "cond_stage_model.")):
            return f"conditioner.embedders.0.{key}"
    if domain == "vae" and not key.startswith("first_stage_model."):
        return f"first_stage_model.{key}"
    return key


def _float_list(value: str) -> list[float]:
    values: list[float] = []
    for chunk in value.replace("\n", ",").split(","):
        try:
            values.append(float(chunk.strip()))
        except ValueError:
            continue
    return values


def _block_id_for_key(key: str, schema: str) -> str:
    lowered = key.lower()
    if "llm_adapter.blocks." in lowered:
        digits = "".join(ch for ch in lowered.split("llm_adapter.blocks.", 1)[1][:5] if ch.isdigit())
        return f"LLM{int(digits):03d}" if digits else "BASE"
    if schema == "anima" and ".blocks." in lowered:
        digits = "".join(ch for ch in lowered.split(".blocks.", 1)[1][:5] if ch.isdigit())
        return f"B{int(digits):03d}" if digits else "BASE"
    if schema == "anima" and lowered.startswith("net.blocks."):
        digits = "".join(ch for ch in lowered.split("net.blocks.", 1)[1][:5] if ch.isdigit())
        return f"B{int(digits):03d}" if digits else "BASE"
    if "final_layer" in lowered and schema == "anima":
        return "OUT"
    if schema == "sdxl" and "conditioner.embedders." in lowered:
        return "BASE"
    if schema == "sdxl" and ("time_embed" in lowered or "label_emb" in lowered):
        return "IN00"
    if schema == "sdxl" and "model.diffusion_model.out." in lowered:
        return "OUT08"
    for marker, prefix in (("input_blocks_", "IN"), ("input_blocks.", "IN"), ("output_blocks_", "OUT"), ("output_blocks.", "OUT")):
        if marker in lowered:
            match = re.match(r"(\d+)", lowered.split(marker, 1)[1])
            return f"{prefix}{int(match.group(1)):02d}" if match else ("BASE" if prefix == "IN" else "OUT")
    if "middle_block" in lowered or ".mid." in lowered:
        return "M00" if schema == "sdxl" else "BASE"
    if "first_stage_model" in lowered:
        return "VAE"
    return "BASE"


def _extended_sdxl_block_id(key: str) -> str:
    lowered = key.lower()
    if "conditioner.embedders.0." in lowered:
        return "CLIPL"
    if "conditioner.embedders.1." in lowered:
        return "CLIPG"
    if "time_embed" in lowered:
        return "TIME"
    if "label_emb" in lowered:
        return "LABEL"
    if "first_stage_model" in lowered:
        return "VAE"
    if "model.diffusion_model.out." in lowered:
        return "OUT9"
    for pattern, prefix in (
        (r"input_blocks[._](\d+)[._](\d+).*?transformer_blocks[._](\d+)", "IN"),
        (r"output_blocks[._](\d+)[._](\d+).*?transformer_blocks[._](\d+)", "OUT"),
    ):
        match = re.search(pattern, lowered)
        if match:
            return prefix + "".join(str(int(value)) for value in match.groups())
    match = re.search(r"middle_block[._](\d+).*?transformer_blocks[._](\d+)", lowered)
    return f"MID{int(match.group(1))}{int(match.group(2))}" if match else ""


def _elemental_block_ids(key: str, block_id: str, schema: str) -> set[str]:
    values = {block_id}
    if schema == "sdxl":
        extended = _extended_sdxl_block_id(key)
        if extended:
            values.add(extended)
    return values


def _expand_block_tokens(value: str, schema: list[str]) -> tuple[set[str], bool]:
    tokens = [part.strip().upper() for part in re.split(r"\s+", value.strip()) if part.strip()]
    invert = bool(tokens and tokens[0] == "NOT")
    if invert:
        tokens = tokens[1:]
    if not tokens or tokens == ["*"]:
        return set(), invert
    selected: set[str] = set()
    for token in tokens:
        if "-" in token:
            start, end = [part.strip().upper() for part in token.split("-", 1)]
            if start in schema and end in schema:
                low, high = sorted((schema.index(start), schema.index(end)))
                selected.update(schema[low : high + 1])
        else:
            selected.add(token)
    return selected, invert


def _element_matches(element: str, key: str) -> bool:
    if not element or element not in key:
        return not element
    if re.fullmatch(r"[a-z]{1,4}\d*", element):
        return bool(re.search(rf"(^|[._]){re.escape(element)}(?:\d+)?([._]|$)", key))
    return True


def _dice_values(value: str, count: int, default: float) -> list[float]:
    parsed = _float_list(value)
    if not parsed:
        return [default] * count
    if len(parsed) == 1:
        parsed *= count
    return (parsed + [default] * count)[:count]


def _elemental_random(payload: dict[str, Any], token: str, block_index: int, tensor_index: int) -> float:
    schema = BLOCK_SCHEMAS[payload["block_schema"]]
    ranges = payload.get("dice_ranges") or {}
    lower = _dice_values(ranges.get("lower", ""), len(schema), 0.0)
    upper = _dice_values(ranges.get("upper", ""), len(schema), 1.0)
    seed = payload.get("random_seed")
    if seed is None:
        material = "|".join([payload.get("recipe_id", ""), payload.get("recipe_name", ""), "", payload["merge_mode"], ""])
        seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    random_value = float(np.random.RandomState(seed).rand(3000)[tensor_index + 100])
    block_index = max(0, min(block_index, len(schema) - 1))
    value = random_value if token == "R" else (-2 * random_value + 1.0 if token == "U" else (lower[block_index] - upper[block_index]) * random_value + upper[block_index])
    try:
        precision = max(0, int(float(ranges.get("round", 3))))
    except (TypeError, ValueError):
        precision = 3
    return round(value, precision)


def _elemental_weight(payload: dict[str, Any], key: str, block_id: str, fallback: float, tensor_index: int) -> float:
    result = fallback
    schema = BLOCK_SCHEMAS[payload["block_schema"]]
    for raw in str(payload.get("elemental_rules", "")).replace(",", "\n").splitlines():
        parts = [part.strip() for part in raw.strip().split(":")]
        if len(parts) < 3:
            continue
        blocks, invert_blocks = _expand_block_tokens(parts[0], schema)
        tokens = [part.lower() for part in re.split(r"\s+", parts[1]) if part]
        invert_elements = bool(tokens and tokens[0].upper() == "NOT")
        if invert_elements:
            tokens = tokens[1:]
        block_match = True if not blocks else bool(blocks.intersection(_elemental_block_ids(key, block_id, payload["block_schema"])))
        if invert_blocks:
            block_match = not block_match
        element_match = True if not tokens else any(_element_matches(token, key.lower()) for token in tokens)
        if invert_elements:
            element_match = not element_match
        if not (block_match and element_match):
            continue
        ratio_text = parts[2]
        if ratio_text.upper() in {"R", "U", "X"} and block_id in schema:
            result = _elemental_random(payload, ratio_text.upper(), schema.index(block_id), tensor_index)
        else:
            try:
                result = float(ratio_text)
            except ValueError:
                pass
    return result


def _block_weight(payload: dict[str, Any], key: str, value: str, fallback: float, tensor_index: int) -> float:
    schema = BLOCK_SCHEMAS[payload["block_schema"]]
    block_id = _block_id_for_key(key, payload["block_schema"])
    weights = _float_list(value)
    if len(weights) >= len(schema) and block_id in schema:
        return _elemental_weight(payload, key, block_id, weights[schema.index(block_id)], tensor_index)
    result = fallback
    for raw in value.splitlines():
        if ":" not in raw:
            continue
        pattern, raw_weight = raw.split(":", 1)
        if pattern.strip() and (pattern.strip() in key or pattern.strip().upper() in _elemental_block_ids(key, block_id, payload["block_schema"])):
            try:
                result = float(raw_weight.strip())
            except ValueError:
                pass
    return _elemental_weight(payload, key, block_id, result, tensor_index)


def _should_merge(payload: dict[str, Any], key: str) -> bool:
    block_id = _block_id_for_key(key, payload["block_schema"])

    def matches(token: str) -> bool:
        normalized = token.strip()
        if not normalized or normalized.upper() == "PRINT":
            return False
        if normalized.upper() == block_id.upper():
            return True
        if normalized.upper() == "ADJUST":
            return key in SUPER_MERGER_FINETUNE_KEYS
        if normalized.upper() == "VAE":
            return block_id.upper() == "VAE" or "first_stage_model" in key.lower()
        return normalized.lower() in key.lower()

    include = payload.get("include_blocks") or []
    exclude = payload.get("exclude_blocks") or []
    return not (include and not any(matches(token) for token in include)) and not any(matches(token) for token in exclude)


def _midpoint_percentile(values: torch.Tensor, percentile: float) -> float:
    sorted_values = torch.sort(values.flatten()).values
    if sorted_values.numel() == 0:
        return 0.0
    if sorted_values.numel() == 1:
        return float(sorted_values[0].item())
    rank = (percentile / 100.0) * (sorted_values.numel() - 1)
    low = int(rank)
    high = min(low + 1, sorted_values.numel() - 1)
    return float(sorted_values[low].item()) if float(low) == rank else float(((sorted_values[low] + sorted_values[high]) / 2.0).item())


def _cosine_scores(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    cosine = torch.nn.CosineSimilarity(dim=0)
    a_float, b_float = a.float(), b.float()
    if mode == "cosineA":
        return cosine(torch.nn.functional.normalize(a_float, p=2, dim=0), torch.nn.functional.normalize(b_float, p=2, dim=0))
    simab = cosine(a_float, b_float)
    magnitude = torch.dot(a_float.flatten(), b_float.flatten()) / (torch.norm(a_float) * torch.norm(b_float))
    return (simab + magnitude) / 2.0


def _cosine_context(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], mode: str) -> dict[str, float]:
    scores = [_cosine_scores(value, b[key], mode).flatten() for key, value in a.items() if "first_stage_model" not in key and "model" in key and key in b and value.shape == b[key].shape]
    if not scores:
        return {"min": 0.0, "max": 0.0}
    sims = torch.cat(scores)
    sims = sims[~torch.isnan(sims)]
    if sims.numel() == 0:
        return {"min": 0.0, "max": 0.0}
    low, high = _midpoint_percentile(sims, 1), _midpoint_percentile(sims, 99)
    sims = sims[(sims >= low) & (sims <= high)]
    return {"min": float(sims.min().item()), "max": float(sims.max().item())} if sims.numel() else {"min": 0.0, "max": 0.0}


def _adjust(payload: dict[str, Any], key: str, value: torch.Tensor) -> torch.Tensor:
    if key not in SUPER_MERGER_FINETUNE_KEYS or not payload.get("adjust_values"):
        return value
    values = payload["adjust_values"]
    get = lambda name: float(values.get(name, 0.0))
    factors: list[Any] = [1 - get("detail_noise_1") * 0.01, 1 + get("detail_noise_1") * 0.02, 1 - get("detail_noise_2") * 0.01, 1 + get("detail_noise_2") * 0.02, 1 - get("detail_noise_3") * 0.01]
    colors = ADJUST_COLORS_SDXL if payload["block_schema"] == "sdxl" else ADJUST_COLORS_ANIMA
    color_values = [get("brightness"), get("color_tone_1"), get("color_tone_2"), get("color_tone_3")]
    additions = [get("contrast_detail") * 0.02] + [sum(colors[i][component] * color_values[i] * 0.02 for i in range(4)) for component in range(3)]
    index = SUPER_MERGER_FINETUNE_KEYS.index(key)
    return value * factors[index] if index < 5 else value + torch.tensor(additions, device=value.device, dtype=torch.float32)


def _replace_slice(a: torch.Tensor, b: torch.Tensor, alpha: float, beta: float, second_axis: bool) -> torch.Tensor:
    if a.ndim == 0:
        return a.clone()
    axis = 1 if second_axis and a.ndim > 1 and a.shape[1] > 100 else 0
    length = a.shape[axis]
    start_ratio = beta if alpha + beta <= 1 else alpha + beta - 1
    end_ratio = alpha + beta if alpha + beta <= 1 else beta
    start, end = max(0, min(length, int(length * start_ratio))), max(0, min(length, int(length * end_ratio)))
    if start >= end:
        return a.clone()
    result, replacement = (a.clone(), b) if alpha + beta <= 1 else (b.clone(), a)
    selector = [slice(None)] * a.ndim
    selector[axis] = slice(start, end)
    result[tuple(selector)] = replacement[tuple(selector)].clone()
    return result


def _merge_tensor(payload: dict[str, Any], key: str, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None, context: dict[str, float] | None, tensor_index: int) -> torch.Tensor:
    alpha = _block_weight(payload, key, payload.get("block_weights_alpha") or payload.get("block_weights", ""), float(payload["alpha"]), tensor_index)
    beta = _block_weight(payload, key, payload.get("block_weights_beta", ""), float(payload["beta"]), tensor_index)
    mode = payload.get("calculation_mode_id") or payload["calculation_mode"]
    a_float, b_float = a.float(), b.float()
    c_float = c.float() if c is not None else None
    if mode in ASTERIA_CALCULATION_MODES:
        try:
            value = merge_weight_sum_tensor(
                mode,
                a_float,
                b_float,
                alpha,
                payload.get("calculation_parameters"),
                implementation_revision=str(
                    payload.get("calculation_implementation_revision") or ""
                ),
            )
        except CalculationModeError as exc:
            raise MergeRuntimeMathError(str(exc)) from exc
    elif mode == "trainDifference":
        if c_float is None:
            raise MergeRuntimeMathError("source C is required")
        if torch.allclose(b_float, c_float.to(device=b_float.device), rtol=0, atol=0):
            value = a_float
        else:
            diff = b_float - c_float
            distances = diff.abs() + (b_float - a_float).abs()
            scale = torch.where(distances != 0, (b_float - a_float).abs() / distances, torch.tensor(0.0, device=a.device))
            value = a_float + torch.sign(diff) * scale.abs() * diff.abs() * (alpha * 1.8)
    elif mode == "extract":
        if c_float is None:
            raise MergeRuntimeMathError("source C is required")
        b_delta, c_delta = b_float - a_float, c_float - a_float
        similarity = torch.nn.functional.cosine_similarity(b_delta, c_delta, dim=-1).clamp(-1, 1).unsqueeze(-1)
        selectivity = ((similarity + 1.0) / 2.0) ** float(payload["calculation_option"])
        value = a_float + torch.lerp(b_delta, c_delta, alpha) * torch.lerp(selectivity, 1.0 - selectivity, beta)
    elif mode in {"smoothAdd", "smoothAdd MT"}:
        if c_float is None:
            raise MergeRuntimeMathError("source C is required")
        diff = (b_float - c_float).cpu().numpy()
        value = a_float + alpha * torch.tensor(scipy.ndimage.gaussian_filter(scipy.ndimage.median_filter(diff, size=3), sigma=1), device=a.device, dtype=torch.float32)
    elif mode in {"cosineA", "cosineB"}:
        if "first_stage_model" in key or "model" not in key or not context or context["max"] == context["min"]:
            value = a_float
        else:
            k = (_cosine_scores(a, b, mode) - context["min"]) / (context["max"] - context["min"])
            k = (k - (abs(alpha) if mode == "cosineA" else alpha)).clip(min=0, max=1.0)
            value = b_float * (1.0 - k) + a_float * k
    elif mode in {"tensor", "tensor2"}:
        value = _replace_slice(a_float, b_float, alpha, beta, mode == "tensor2")
    elif mode == "self":
        value = a_float if alpha == 0.0 or any(part in key for part in SELF_SKIP_KEY_PARTS) else a_float * alpha
    elif mode == "plus random":
        if alpha == 0.0 or any(part in key for part in SELF_SKIP_KEY_PARTS):
            value = a_float
        else:
            seed = payload.get("random_seed")
            if seed is None:
                seed = int(hashlib.sha256(f"{payload.get('recipe_id', '')}|{payload.get('recipe_name', '')}|{key}".encode("utf-8")).hexdigest()[:16], 16)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) % (2**63 - 1))
            value = a_float + torch.randn(a.shape, generator=generator, dtype=torch.float32, device="cpu").to(a.device) * alpha
    else:
        merge_mode = payload["merge_mode"]
        if merge_mode == "add_difference":
            if c_float is None:
                raise MergeRuntimeMathError("source C is required")
            value = a_float + (b_float - c_float) * alpha
        elif merge_mode == "triple_sum":
            if c_float is None:
                raise MergeRuntimeMathError("source C is required")
            value = a_float * (1.0 - alpha - beta) + b_float * alpha + c_float * beta
        elif merge_mode == "sum_twice":
            if c_float is None:
                raise MergeRuntimeMathError("source C is required")
            value = (a_float * (1.0 - alpha) + b_float * alpha) * (1.0 - beta) + c_float * beta
        else:
            value = a_float * (1.0 - alpha) + b_float * alpha
    output_dtype = (
        torch.float16
        if bool(payload.get("export_controls", {}).get("fp16"))
        else a.dtype
    )
    return _adjust(payload, key, value).to(dtype=output_dtype)


def cosine_context_for_pairs(payload: dict[str, Any], pairs, *, domain: str) -> dict[str, float] | None:
    mode = payload["calculation_mode"]
    if mode not in {"cosineA", "cosineB"}:
        return None
    scores: list[torch.Tensor] = []
    for key, tensor_a, tensor_b in pairs:
        canonical = _canonical_key(key, domain)
        if "first_stage_model" in canonical or "model" not in canonical or tensor_a.shape != tensor_b.shape:
            continue
        scores.append(_cosine_scores(tensor_a, tensor_b, mode).flatten())
    if not scores:
        return {"min": 0.0, "max": 0.0}
    similarities = torch.cat(scores)
    similarities = similarities[~torch.isnan(similarities)]
    if similarities.numel() == 0:
        return {"min": 0.0, "max": 0.0}
    low = _midpoint_percentile(similarities, 1)
    high = _midpoint_percentile(similarities, 99)
    similarities = similarities[(similarities >= low) & (similarities <= high)]
    return (
        {"min": float(similarities.min().item()), "max": float(similarities.max().item())}
        if similarities.numel()
        else {"min": 0.0, "max": 0.0}
    )


def merge_single_tensor(
    payload: dict[str, Any],
    key: str,
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor | None,
    tensor_c: torch.Tensor | None,
    *,
    domain: str,
    tensor_index: int,
    cosine_context: dict[str, float] | None,
) -> torch.Tensor:
    canonical = _canonical_key(key, domain)
    if not _should_merge(payload, canonical) or not tensor_a.is_floating_point() or not any(
        part in canonical.lower() for part in ("weight", "bias")
    ):
        return tensor_a.clone()
    if tensor_b is None:
        return tensor_a.clone()
    if tensor_a.shape != tensor_b.shape:
        raise MergeRuntimeMathError("source tensor shapes do not match")
    if payload.get("requires_source_c") and tensor_c is None:
        raise MergeRuntimeMathError("source C is missing a tensor")
    if tensor_c is not None and tensor_c.shape != tensor_b.shape:
        raise MergeRuntimeMathError("source C tensor shape does not match")
    return _merge_tensor(
        payload,
        canonical,
        tensor_a,
        tensor_b,
        tensor_c,
        cosine_context,
        tensor_index,
    )


def merge_tensor_dict(payload: dict[str, Any], tensors_a: dict[str, torch.Tensor], tensors_b: dict[str, torch.Tensor], tensors_c: dict[str, torch.Tensor] | None = None, *, domain: str) -> dict[str, torch.Tensor]:
    validate_payload(payload)
    if domain not in {"model", "clip", "vae"}:
        raise MergeRuntimeMathError("unsupported tensor domain")
    if payload.get("requires_source_c") and tensors_c is None:
        raise MergeRuntimeMathError("source C is required")
    context = cosine_context_for_pairs(
        payload,
        ((key, value, tensors_b[key]) for key, value in tensors_a.items() if key in tensors_b),
        domain=domain,
    )
    merged: dict[str, torch.Tensor] = {}
    for tensor_index, (key, tensor_a) in enumerate(tensors_a.items()):
        tensor_b = tensors_b.get(key)
        tensor_c = tensors_c.get(key) if tensors_c is not None else None
        merged[key] = merge_single_tensor(
            payload,
            key,
            tensor_a,
            tensor_b,
            tensor_c,
            domain=domain,
            tensor_index=tensor_index,
            cosine_context=context,
        )
    return merged
