from __future__ import annotations

import math

import torch


CALCULATION_CONTRACT_VERSION = "asteria.merge.calculation.v1"
CALCULATION_PARAMETER_SCHEMA_VERSION = 1
CALCULATION_IMPLEMENTATION_REVISION = "asteria-merge-math-2026-07-28.3"
V11_CALCULATION_MODES = ("slerp", "nuslerp", "nearswap")


class CalculationModeError(ValueError):
    pass


def normalize_calculation_parameters(
    mode: str,
    parameters: dict[str, object] | None,
) -> dict[str, object]:
    supplied = dict(parameters or {})
    if mode == "slerp":
        allowed = {"dot_threshold"}
        unknown = sorted(set(supplied).difference(allowed))
        if unknown:
            raise CalculationModeError(
                f"slerp has unsupported parameters: {', '.join(unknown)}"
            )
        threshold = supplied.get("dot_threshold", 0.9995)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.9 <= float(threshold) <= 1.0
        ):
            raise CalculationModeError(
                "slerp dot_threshold must be between 0.9 and 1"
            )
        return {"dot_threshold": float(threshold)}
    if mode == "nuslerp":
        allowed = {"vector_mode", "dot_threshold", "epsilon"}
        unknown = sorted(set(supplied).difference(allowed))
        if unknown:
            raise CalculationModeError(
                f"nuslerp has unsupported parameters: {', '.join(unknown)}"
            )
        vector_mode = supplied.get("vector_mode", "row")
        if vector_mode not in {"row", "column", "flatten"}:
            raise CalculationModeError(
                "nuslerp vector_mode must be one of: row, column, flatten"
            )
        epsilon = supplied.get("epsilon", 1e-8)
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or not 1e-12 <= float(epsilon) <= 1e-3
        ):
            raise CalculationModeError(
                "nuslerp epsilon must be between 1e-12 and 0.001"
            )
        threshold = supplied.get("dot_threshold", 0.9995)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.9 <= float(threshold) <= 1.0
        ):
            raise CalculationModeError(
                "nuslerp dot_threshold must be between 0.9 and 1"
            )
        return {
            "dot_threshold": float(threshold),
            "epsilon": float(epsilon),
            "vector_mode": str(vector_mode),
        }
    if mode == "nearswap":
        allowed = {"epsilon"}
        unknown = sorted(set(supplied).difference(allowed))
        if unknown:
            raise CalculationModeError(
                f"nearswap has unsupported parameters: {', '.join(unknown)}"
            )
        epsilon = supplied.get("epsilon", 1e-6)
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or not 1e-12 <= float(epsilon) <= 1e-3
        ):
            raise CalculationModeError(
                "nearswap epsilon must be between 1e-12 and 0.001"
            )
        return {"epsilon": float(epsilon)}
    raise CalculationModeError(f"unsupported calculation mode: {mode}")


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
    if a.shape != b.shape:
        raise CalculationModeError(
            f"{mode} source tensor shapes do not match: {tuple(a.shape)} != {tuple(b.shape)}"
        )
    resolved_alpha = float(alpha)
    if not math.isfinite(resolved_alpha) or not 0.0 <= resolved_alpha <= 1.0:
        raise CalculationModeError(f"{mode} alpha must be between 0 and 1")
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
    difference = (a - b).abs()
    epsilon = float(resolved["epsilon"])
    weight = (resolved_alpha / difference.clamp_min(epsilon)).clamp(0.0, 1.0)
    return weight * b + (1.0 - weight) * a
