from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch


CALCULATION_CONTRACT_VERSION = "asteria.merge.calculation.v1"
CALCULATION_PARAMETER_SCHEMA_VERSION = 1
CALCULATION_IMPLEMENTATION_REVISION = "asteria-merge-math-2026-07-28.3"
V11_CALCULATION_MODES = ("slerp", "nuslerp", "nearswap")


class CalculationModeError(ValueError):
    pass


_SHARED_CONTRACT = {
    "contract_version": CALCULATION_CONTRACT_VERSION,
    "parameter_schema_version": CALCULATION_PARAMETER_SCHEMA_VERSION,
    "implementation_revision": CALCULATION_IMPLEMENTATION_REVISION,
    "merge_modes": ["weight_sum"],
    "merge_types": ["checkpoint"],
    "model_families": ["sdxl", "anima"],
    "xyz_axis_parameters": ["alpha"],
    "alpha": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
    "tensor_domain": "floating_weight_or_bias",
    "dtype_policy": "float32_arithmetic_then_requested_output_dtype",
    "shape_policy": "matching_shapes_required",
    "missing_or_non_floating_policy": "preserve_source_a",
}

_MODE_CONTRACTS: dict[str, dict[str, Any]] = {
    "slerp": {
        **_SHARED_CONTRACT,
        "id": "slerp",
        "label": "SLERP",
        "description": "Global spherical interpolation over the flattened tensor",
        "parameters": {
            "dot_threshold": {
                "type": "number",
                "default": 0.9995,
                "minimum": 0.9,
                "maximum": 1.0,
            },
        },
    },
    "nuslerp": {
        **_SHARED_CONTRACT,
        "id": "nuslerp",
        "label": "NuSLERP",
        "description": "Vector-wise spherical interpolation with stable linear fallback",
        "parameters": {
            "vector_mode": {
                "type": "string",
                "default": "row",
                "enum": ["row", "column", "flatten"],
            },
            "dot_threshold": {
                "type": "number",
                "default": 0.9995,
                "minimum": 0.9,
                "maximum": 1.0,
            },
            "epsilon": {
                "type": "number",
                "default": 1e-8,
                "minimum": 1e-12,
                "maximum": 1e-3,
            },
        },
    },
    "nearswap": {
        **_SHARED_CONTRACT,
        "id": "nearswap",
        "label": "Nearswap",
        "description": "Element-wise interpolation weighted toward already-similar parameters",
        "parameters": {
            "epsilon": {
                "type": "number",
                "default": 1e-6,
                "minimum": 1e-12,
                "maximum": 1e-3,
            },
        },
    },
}


def calculation_mode_contract(mode: str) -> dict[str, Any]:
    contract = _MODE_CONTRACTS.get(mode)
    if contract is None:
        raise CalculationModeError(f"unsupported calculation mode: {mode}")
    return deepcopy(contract)


def _number_parameter(
    mode: str,
    name: str,
    raw: object,
    definition: dict[str, Any],
) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise CalculationModeError(f"{mode} {name} must be a number")
    value = float(raw)
    minimum = float(definition["minimum"])
    maximum = float(definition["maximum"])
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise CalculationModeError(
            f"{mode} {name} must be between {minimum:g} and {maximum:g}"
        )
    return value


def normalize_calculation_parameters(
    mode: str,
    parameters: dict[str, object] | None,
) -> dict[str, object]:
    contract = calculation_mode_contract(mode)
    definitions: dict[str, dict[str, Any]] = contract["parameters"]
    supplied = dict(parameters or {})
    unknown = sorted(set(supplied).difference(definitions))
    if unknown:
        raise CalculationModeError(
            f"{mode} has unsupported parameters: {', '.join(unknown)}"
        )
    normalized: dict[str, object] = {}
    for name, definition in definitions.items():
        raw = supplied.get(name, definition["default"])
        if definition["type"] == "number":
            normalized[name] = _number_parameter(mode, name, raw, definition)
            continue
        if not isinstance(raw, str) or raw not in definition["enum"]:
            allowed = ", ".join(definition["enum"])
            raise CalculationModeError(
                f"{mode} {name} must be one of: {allowed}"
            )
        normalized[name] = raw
    return normalized


def _validate_inputs(
    mode: str,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
) -> float:
    if mode not in V11_CALCULATION_MODES:
        raise CalculationModeError(f"unsupported calculation mode: {mode}")
    if a.shape != b.shape:
        raise CalculationModeError(
            f"{mode} source tensor shapes do not match: {tuple(a.shape)} != {tuple(b.shape)}"
        )
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CalculationModeError(f"{mode} alpha must be between 0 and 1")
    return value


def _linear(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return a * (1.0 - alpha) + b * alpha


def _global_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    dot_threshold: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_norm = torch.linalg.vector_norm(a_flat)
    b_norm = torch.linalg.vector_norm(b_flat)
    linear = _linear(a_flat, b_flat, alpha)
    safe_norms = (a_norm > epsilon) & (b_norm > epsilon)
    denominator = (a_norm * b_norm).clamp_min(epsilon)
    dot = ((a_flat * b_flat).sum() / denominator).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    spherical = (
        torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(epsilon) * a_flat
        + torch.sin(alpha * theta) / sin_theta.clamp_min(epsilon) * b_flat
    )
    use_linear = (
        ~safe_norms
        | ~torch.isfinite(spherical).all()
        | (dot.abs() >= dot_threshold)
        | (sin_theta.abs() <= epsilon)
    )
    return torch.where(use_linear, linear, spherical).reshape(a.shape)


def _vector_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    epsilon: float,
    vector_mode: str,
    dot_threshold: float,
) -> torch.Tensor:
    original_shape = a.shape
    if vector_mode == "flatten" or a.ndim == 0:
        working_a = a.reshape(1, -1)
        working_b = b.reshape(1, -1)
        transposed = False
    elif vector_mode == "column" and a.ndim > 1:
        working_a = a.transpose(0, -1)
        working_b = b.transpose(0, -1)
        transposed = True
    else:
        working_a = a
        working_b = b
        transposed = False

    norm_a = torch.linalg.vector_norm(working_a, dim=-1, keepdim=True)
    norm_b = torch.linalg.vector_norm(working_b, dim=-1, keepdim=True)
    unit_a = working_a / norm_a.clamp_min(epsilon)
    unit_b = working_b / norm_b.clamp_min(epsilon)
    cosine = (unit_a * unit_b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    sin_theta = torch.sin(theta)
    denominator = sin_theta.clamp_min(epsilon)
    spherical = (
        torch.sin((1.0 - alpha) * theta) * working_a
        + torch.sin(alpha * theta) * working_b
    ) / denominator
    linear = _linear(working_a, working_b, alpha)
    unsafe = (
        (norm_a <= epsilon)
        | (norm_b <= epsilon)
        | (cosine.abs() >= dot_threshold)
        | (sin_theta.abs() <= epsilon)
        | ~torch.isfinite(spherical).all(dim=-1, keepdim=True)
    )
    result = torch.where(unsafe, linear, spherical)
    if transposed:
        result = result.transpose(0, -1)
    return result.reshape(original_shape)


def merge_weight_sum_tensor(
    mode: str,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    parameters: dict[str, object] | None,
) -> torch.Tensor:
    resolved_alpha = _validate_inputs(mode, a, b, alpha)
    resolved = normalize_calculation_parameters(mode, parameters)
    if resolved_alpha == 0.0:
        return a.clone()
    if resolved_alpha == 1.0:
        return b.clone()
    if mode == "slerp":
        return _global_slerp(
            a,
            b,
            resolved_alpha,
            float(resolved["dot_threshold"]),
        )
    if mode == "nuslerp":
        return _vector_slerp(
            a,
            b,
            resolved_alpha,
            float(resolved["epsilon"]),
            str(resolved["vector_mode"]),
            float(resolved["dot_threshold"]),
        )
    epsilon = float(resolved["epsilon"])
    difference = (a - b).abs()
    weight = (resolved_alpha / difference.clamp_min(epsilon)).clamp(0.0, 1.0)
    return weight * b + (1.0 - weight) * a
