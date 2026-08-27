from __future__ import annotations

import math

import torch


CALCULATION_CONTRACT_VERSION = "asteria.merge.calculation.v1"
CALCULATION_PARAMETER_SCHEMA_VERSION = 1
CALCULATION_IMPLEMENTATION_REVISION = "asteria-merge-math-2026-07-28.3"
V11_CALCULATION_MODES = ("slerp", "nuslerp", "nearswap")
V12_CALCULATION_CONTRACT_VERSION = "asteria.merge.calculation.v2"
V12_CALCULATION_PARAMETER_SCHEMA_VERSION = 2
V12_CALCULATION_IMPLEMENTATION_REVISION = "asteria-merge-math-2026-08-26.2"
V12_LEGACY_CALCULATION_IMPLEMENTATION_REVISIONS = (
    "asteria-merge-math-2026-08-21.1",
)
DUAL_RESIDUAL_OPPOSITION_COSINE_THRESHOLD = -0.9995
MOMENT_WEAVE_SHAPE_COSINE_THRESHOLD = 0.5
V12_RESEARCH_CALCULATION_MODES = (
    "moment_weave",
    "log_weave",
    "dual_residual",
)
V12_PUBLIC_CALCULATION_MODES = V12_RESEARCH_CALCULATION_MODES
ASTERIA_CALCULATION_MODES = V11_CALCULATION_MODES + V12_PUBLIC_CALCULATION_MODES


class CalculationModeError(ValueError):
    pass


def calculation_implementation_revision_supported(mode: str, revision: str) -> bool:
    if mode in V11_CALCULATION_MODES:
        return revision == CALCULATION_IMPLEMENTATION_REVISION
    if mode in V12_RESEARCH_CALCULATION_MODES:
        return revision in {
            V12_CALCULATION_IMPLEMENTATION_REVISION,
            *V12_LEGACY_CALCULATION_IMPLEMENTATION_REVISIONS,
        }
    return False


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
    if mode in V12_RESEARCH_CALCULATION_MODES:
        allowed = {"epsilon"}
        if mode == "dual_residual":
            allowed.update({"residual_exponent", "energy_cap"})
        unknown = sorted(set(supplied).difference(allowed))
        if unknown:
            raise CalculationModeError(
                f"{mode} has unsupported parameters: {', '.join(unknown)}"
            )

        def bounded_number(
            name: str,
            default: float,
            minimum: float,
            maximum: float,
        ) -> float:
            raw = supplied.get(name, default)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or not minimum <= float(raw) <= maximum
            ):
                raise CalculationModeError(
                    f"{mode} {name} must be between {minimum:g} and {maximum:g}"
                )
            return float(raw)

        result = {
            "epsilon": bounded_number("epsilon", 1e-8, 1e-12, 1e-3),
        }
        if mode == "dual_residual":
            result.update(
                {
                    "residual_exponent": bounded_number(
                        "residual_exponent", 0.5, 0.0, 1.0
                    ),
                    "energy_cap": bounded_number("energy_cap", 1.25, 1.0, 4.0),
                }
            )
        return result
    raise CalculationModeError(f"unsupported calculation mode: {mode}")


def _linear(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    return a * (1.0 - alpha) + b * alpha


def _moment_weave(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    epsilon: float,
    shape_cosine_threshold: float | None,
) -> torch.Tensor:
    if a.ndim == 0:
        return _linear(a, b, alpha)
    output_dtype = a.dtype
    working_a = a.to(torch.float32)
    working_b = b.to(torch.float32)
    linear = _linear(working_a, working_b, alpha)
    mean_a = working_a.mean(dim=-1, keepdim=True)
    mean_b = working_b.mean(dim=-1, keepdim=True)
    centered_a = working_a - mean_a
    centered_b = working_b - mean_b
    rms_a = centered_a.square().mean(dim=-1, keepdim=True).sqrt()
    rms_b = centered_b.square().mean(dim=-1, keepdim=True).sqrt()
    unit_a = centered_a / rms_a.clamp_min(epsilon)
    unit_b = centered_b / rms_b.clamp_min(epsilon)
    shape_cosine = (unit_a * unit_b).mean(dim=-1, keepdim=True)
    low_shape_cosine = (
        torch.zeros_like(shape_cosine, dtype=torch.bool)
        if shape_cosine_threshold is None
        else shape_cosine < shape_cosine_threshold
    )
    shape = _linear(unit_a, unit_b, alpha)
    shape_rms = shape.square().mean(dim=-1, keepdim=True).sqrt()
    normalized_shape = shape / shape_rms.clamp_min(epsilon)
    blended_mean = _linear(mean_a, mean_b, alpha)
    blended_rms = torch.exp(
        (1.0 - alpha) * torch.log(rms_a.clamp_min(epsilon))
        + alpha * torch.log(rms_b.clamp_min(epsilon))
    )
    candidate = blended_mean + blended_rms * normalized_shape
    unsafe = (
        ~torch.isfinite(working_a).all(dim=-1, keepdim=True)
        | ~torch.isfinite(working_b).all(dim=-1, keepdim=True)
        | ~torch.isfinite(rms_a)
        | ~torch.isfinite(rms_b)
        | (rms_a <= epsilon)
        | (rms_b <= epsilon)
        | ~torch.isfinite(shape_cosine)
        | low_shape_cosine
        | ~torch.isfinite(shape_rms)
        | (shape_rms <= epsilon)
        | ~torch.isfinite(candidate).all(dim=-1, keepdim=True)
    )
    return torch.where(unsafe, linear, candidate).to(output_dtype)


def _log_weave(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    epsilon: float,
) -> torch.Tensor:
    if a.ndim == 0:
        return _linear(a, b, alpha)
    output_dtype = a.dtype
    working_a = a.to(torch.float32)
    working_b = b.to(torch.float32)
    linear = _linear(working_a, working_b, alpha)
    scale_a = working_a.abs().square().mean(dim=-1, keepdim=True).sqrt()
    scale_b = working_b.abs().square().mean(dim=-1, keepdim=True).sqrt()
    scale = torch.maximum(scale_a, scale_b).clamp_min(epsilon)
    transformed_a = working_a.sign() * torch.log1p(working_a.abs() / scale)
    transformed_b = working_b.sign() * torch.log1p(working_b.abs() / scale)
    transformed = _linear(transformed_a, transformed_b, alpha)
    candidate = transformed.sign() * scale * torch.expm1(transformed.abs())
    unsafe = (
        ~torch.isfinite(working_a).all(dim=-1, keepdim=True)
        | ~torch.isfinite(working_b).all(dim=-1, keepdim=True)
        | ~torch.isfinite(scale)
        | ~torch.isfinite(transformed).all(dim=-1, keepdim=True)
        | ~torch.isfinite(candidate).all(dim=-1, keepdim=True)
    )
    return torch.where(unsafe, linear, candidate).to(output_dtype)


def _dual_residual(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    epsilon: float,
    residual_exponent: float,
    energy_cap: float,
) -> torch.Tensor:
    if a.ndim == 0:
        return _linear(a, b, alpha)
    output_dtype = a.dtype
    working_a = a.to(torch.float32)
    working_b = b.to(torch.float32)
    linear = _linear(working_a, working_b, alpha)
    norm_a = torch.linalg.vector_norm(working_a, dim=-1, keepdim=True)
    norm_b = torch.linalg.vector_norm(working_b, dim=-1, keepdim=True)
    unit_a = working_a / norm_a.clamp_min(epsilon)
    unit_b = working_b / norm_b.clamp_min(epsilon)
    cosine = (unit_a * unit_b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    shared_raw = unit_a + unit_b
    shared_norm = torch.linalg.vector_norm(shared_raw, dim=-1, keepdim=True)
    shared = shared_raw / shared_norm.clamp_min(epsilon)
    projection_a = (working_a * shared).sum(dim=-1, keepdim=True) * shared
    projection_b = (working_b * shared).sum(dim=-1, keepdim=True) * shared
    residual_a = working_a - projection_a
    residual_b = working_b - projection_b
    candidate = (
        _linear(projection_a, projection_b, alpha)
        + (1.0 - alpha) ** residual_exponent * residual_a
        + alpha**residual_exponent * residual_b
    )
    rms_a = working_a.square().mean(dim=-1, keepdim=True).sqrt()
    rms_b = working_b.square().mean(dim=-1, keepdim=True).sqrt()
    envelope = _linear(rms_a, rms_b, alpha)
    candidate_rms = candidate.square().mean(dim=-1, keepdim=True).sqrt()
    maximum_rms = envelope * energy_cap
    capped = candidate * (maximum_rms / candidate_rms.clamp_min(epsilon))
    candidate = torch.where(candidate_rms > maximum_rms, capped, candidate)
    unsafe = (
        ~torch.isfinite(working_a).all(dim=-1, keepdim=True)
        | ~torch.isfinite(working_b).all(dim=-1, keepdim=True)
        | ~torch.isfinite(norm_a)
        | ~torch.isfinite(norm_b)
        | (norm_a <= epsilon)
        | (norm_b <= epsilon)
        | ~torch.isfinite(cosine)
        | (cosine <= DUAL_RESIDUAL_OPPOSITION_COSINE_THRESHOLD)
        | ~torch.isfinite(shared_norm)
        | (shared_norm <= epsilon)
        | ~torch.isfinite(envelope)
        | ~torch.isfinite(candidate).all(dim=-1, keepdim=True)
    )
    return torch.where(unsafe, linear, candidate).to(output_dtype)


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
    *,
    implementation_revision: str | None = None,
) -> torch.Tensor:
    if a.shape != b.shape:
        raise CalculationModeError(
            f"{mode} source tensor shapes do not match: {tuple(a.shape)} != {tuple(b.shape)}"
        )
    resolved_alpha = float(alpha)
    if not math.isfinite(resolved_alpha) or not 0.0 <= resolved_alpha <= 1.0:
        raise CalculationModeError(f"{mode} alpha must be between 0 and 1")
    resolved = normalize_calculation_parameters(mode, parameters)
    revision = implementation_revision or (
        CALCULATION_IMPLEMENTATION_REVISION
        if mode in V11_CALCULATION_MODES
        else V12_CALCULATION_IMPLEMENTATION_REVISION
    )
    if not calculation_implementation_revision_supported(mode, revision):
        raise CalculationModeError(
            f"unsupported calculation implementation revision: {revision}"
        )
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
    if mode == "moment_weave":
        return _moment_weave(
            a,
            b,
            resolved_alpha,
            float(resolved["epsilon"]),
            None
            if revision in V12_LEGACY_CALCULATION_IMPLEMENTATION_REVISIONS
            else MOMENT_WEAVE_SHAPE_COSINE_THRESHOLD,
        )
    if mode == "log_weave":
        return _log_weave(
            a,
            b,
            resolved_alpha,
            float(resolved["epsilon"]),
        )
    if mode == "dual_residual":
        return _dual_residual(
            a,
            b,
            resolved_alpha,
            float(resolved["epsilon"]),
            float(resolved["residual_exponent"]),
            float(resolved["energy_cap"]),
        )
    if mode not in V11_CALCULATION_MODES + V12_RESEARCH_CALCULATION_MODES:
        raise CalculationModeError(f"unsupported calculation mode: {mode}")
    difference = (a - b).abs()
    epsilon = float(resolved["epsilon"])
    weight = (resolved_alpha / difference.clamp_min(epsilon)).clamp(0.0, 1.0)
    return weight * b + (1.0 - weight) * a
