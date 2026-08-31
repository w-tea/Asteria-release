from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from threading import RLock
from typing import Any, Callable, Iterable

import torch

from .spatial_contract import validate_stage


_COMFY_GRADIENT_OPERATION_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class SpatialLatentGuidanceStep:
    updated_latent: torch.Tensor
    stage: str
    step_index: int
    sigma: float
    step_size: float
    loss: float
    gradient_norm: float
    gradient_max_abs: float
    gradient_normalization: str
    input_latent_sha256: str
    gradient_sha256: str
    updated_latent_sha256: str
    parameter_sha256_before: str
    parameter_sha256_after: str
    iteration_count: int = 1
    guided: bool = True


def identity_latent_guidance_step(
    latent: torch.Tensor,
    *,
    stage: str,
    step_index: int,
    sigma: float,
) -> SpatialLatentGuidanceStep:
    digest = _tensor_sha256(latent)
    empty_parameters = _parameter_sha256(())
    return SpatialLatentGuidanceStep(
        updated_latent=latent.detach(),
        stage=validate_stage(stage),
        step_index=int(step_index),
        sigma=float(sigma),
        step_size=0.0,
        loss=0.0,
        gradient_norm=0.0,
        gradient_max_abs=0.0,
        gradient_normalization="none",
        input_latent_sha256=digest,
        gradient_sha256=_tensor_sha256(torch.zeros_like(latent)),
        updated_latent_sha256=digest,
        parameter_sha256_before=empty_parameters,
        parameter_sha256_after=empty_parameters,
        iteration_count=0,
        guided=False,
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.device.type == "meta"
    ):
        raise ValueError("Asteria spatial latent tensor is unsupported")
    normalized = tensor.detach().to(device="cpu").contiguous()
    digest = sha256()
    digest.update(
        json.dumps(
            {"dtype": str(normalized.dtype), "shape": list(normalized.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(memoryview(normalized.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _parameter_sha256(parameters: tuple[torch.Tensor, ...]) -> str:
    digest = sha256()
    for index, parameter in enumerate(parameters):
        digest.update(str(index).encode("ascii"))
        digest.update(_tensor_sha256(parameter).encode("ascii"))
    return digest.hexdigest()


def _validated_parameters(parameters: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    normalized = tuple(parameters)
    for parameter in normalized:
        if not isinstance(parameter, torch.Tensor):
            raise ValueError("Asteria spatial model parameter is invalid")
        if parameter.requires_grad:
            raise ValueError("Asteria spatial model parameter requires_grad must remain disabled")
        if parameter.grad is not None:
            raise ValueError("Asteria spatial model parameter gradient must remain empty")
    return normalized


def _validated_update_mask(mask: torch.Tensor | None, latent: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Asteria spatial latent update mask is invalid")
    try:
        broadcast = torch.broadcast_to(mask.to(device=latent.device, dtype=latent.dtype), latent.shape)
    except RuntimeError as error:
        raise ValueError("Asteria spatial latent update mask shape is invalid") from error
    if not bool(torch.isfinite(broadcast).all()):
        raise ValueError("Asteria spatial latent update mask must be finite")
    if float(broadcast.min()) < 0.0 or float(broadcast.max()) > 1.0:
        raise ValueError("Asteria spatial latent update mask is outside its range")
    return broadcast


def gradient_rms_rope_split_half(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    rot_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-compatible equivalent of Comfy Kitchen's fused Anima operation."""

    if key_scale is None:
        key_scale = query_scale

    def normalize(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        variance = value.float().square().mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + float(epsilon)).to(value.dtype)
        return normalized * scale.to(device=value.device, dtype=value.dtype)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        dimension = int(rot_dim) if int(rot_dim) > 0 else int(value.shape[-1])
        if dimension <= 0 or dimension > value.shape[-1] or dimension % 2:
            raise ValueError("Asteria spatial rotary dimension is invalid")
        rotated_input = value[..., :dimension]
        paired = (
            rotated_input.reshape(*rotated_input.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(frequencies.dtype)
        )
        relevant_frequencies = frequencies[..., : dimension // 2, :, :]
        output = (
            relevant_frequencies[..., 0] * paired[..., 0]
            + relevant_frequencies[..., 1] * paired[..., 1]
        )
        output = output.movedim(-1, -2).reshape(*rotated_input.shape).to(value.dtype)
        if dimension == value.shape[-1]:
            return output
        return torch.cat((output, value[..., dimension:]), dim=-1)

    return rotate(normalize(query, query_scale)), rotate(normalize(key, key_scale))


@contextmanager
def gradient_compatible_comfy_operations():
    """Make Comfy's inference-time weight casts safe for an input-gradient pass."""

    comfy_ops = None
    original_cast_bias_weight = None
    original_sdpa_priority = None
    try:
        import comfy.ops as comfy_ops
    except ImportError:
        pass
    if comfy_ops is not None:
        original_cast_bias_weight = getattr(comfy_ops, "cast_bias_weight", None)
        original_sdpa_priority = getattr(comfy_ops, "SDPA_BACKEND_PRIORITY", None)

    quant_ops = None
    original_rope = None
    try:
        import comfy.quant_ops as quant_ops
    except ImportError:
        pass
    if quant_ops is not None:
        original_rope = getattr(quant_ops.ck, "rms_rope_split_half", None)

    if not callable(original_cast_bias_weight) and not callable(original_rope):
        yield
        return

    def gradient_cast_bias_weight(*args, **kwargs):
        result = original_cast_bias_weight(*args, **kwargs)
        if torch.is_inference_mode_enabled():
            return result
        values = list(result)
        for index in range(min(2, len(values))):
            value = values[index]
            if isinstance(value, torch.Tensor) and value.is_inference():
                values[index] = value.clone()
        return tuple(values)

    with _COMFY_GRADIENT_OPERATION_LOCK:
        if callable(original_cast_bias_weight):
            comfy_ops.cast_bias_weight = gradient_cast_bias_weight
        if isinstance(original_sdpa_priority, list):
            from torch.nn.attention import SDPBackend

            comfy_ops.SDPA_BACKEND_PRIORITY = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        if callable(original_rope):
            quant_ops.ck.rms_rope_split_half = gradient_rms_rope_split_half
        try:
            yield
        finally:
            if callable(original_cast_bias_weight):
                comfy_ops.cast_bias_weight = original_cast_bias_weight
            if isinstance(original_sdpa_priority, list):
                comfy_ops.SDPA_BACKEND_PRIORITY = original_sdpa_priority
            if callable(original_rope):
                quant_ops.ck.rms_rope_split_half = original_rope


@contextmanager
def gradient_stable_attention_backend():
    """Avoid the cuDNN SDPA backward path that can emit NaNs for SDXL fp16."""

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        set_priority=True,
    ):
        yield


def compute_latent_guidance_update(
    latent: torch.Tensor,
    *,
    forward_and_measure: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    stage: str,
    step_index: int,
    sigma: float,
    frozen_parameters: Iterable[torch.Tensor] = (),
    update_mask: torch.Tensor | None = None,
    backward_scale: float = 16.0,
    gradient_normalization: str = "none",
) -> SpatialLatentGuidanceStep:
    if not isinstance(latent, torch.Tensor) or latent.numel() < 1:
        raise ValueError("Asteria spatial latent is invalid")
    if latent.layout != torch.strided or latent.is_quantized or latent.device.type == "meta":
        raise ValueError("Asteria spatial latent tensor is unsupported")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Asteria spatial latent must be finite")
    normalized_step_size = float(step_size)
    if not math.isfinite(normalized_step_size) or normalized_step_size <= 0.0:
        raise ValueError("Asteria spatial latent step size must be positive and finite")
    normalized_step_index = int(step_index)
    if normalized_step_index < 0:
        raise ValueError("Asteria spatial latent step index is invalid")
    normalized_sigma = float(sigma)
    if not math.isfinite(normalized_sigma) or normalized_sigma < 0.0:
        raise ValueError("Asteria spatial latent sigma is invalid")
    normalized_stage = validate_stage(stage)
    normalized_backward_scale = float(backward_scale)
    if (
        not math.isfinite(normalized_backward_scale)
        or normalized_backward_scale <= 0.0
        or normalized_backward_scale > 65536.0
    ):
        raise ValueError("Asteria spatial backward scale is invalid")
    if gradient_normalization not in {"none", "linf"}:
        raise ValueError("Asteria spatial gradient normalization is invalid")
    parameters = _validated_parameters(frozen_parameters)
    parameter_sha256_before = _parameter_sha256(parameters)
    input_sha256 = _tensor_sha256(latent)

    with torch.inference_mode(False), torch.enable_grad():
        guidance_latent = latent.detach().clone().requires_grad_(True)
        with gradient_compatible_comfy_operations(), gradient_stable_attention_backend():
            loss = forward_and_measure(guidance_latent)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError("Asteria spatial latent loss must be a scalar tensor")
            if not bool(torch.isfinite(loss.detach()).all()):
                raise ValueError("Asteria spatial latent loss must be finite")
            (gradient,) = torch.autograd.grad(
                loss * normalized_backward_scale,
                guidance_latent,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        gradient = gradient / normalized_backward_scale
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("Asteria spatial latent gradient must be finite")
        gradient_norm_tensor = torch.linalg.vector_norm(gradient.float())
        gradient_norm = float(gradient_norm_tensor.detach().cpu())
        gradient_max_abs_tensor = gradient.detach().abs().max()
        gradient_max_abs = float(gradient_max_abs_tensor.cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise ValueError(
                "Asteria spatial latent gradient must be non-zero "
                f"(loss={float(loss.detach().cpu()):.8g}, "
                f"gradient_abs_max={float(gradient.detach().abs().max().cpu()):.8g})"
            )
        applied_gradient = gradient
        if gradient_normalization == "linf":
            if not math.isfinite(gradient_max_abs) or gradient_max_abs <= 0.0:
                raise ValueError("Asteria spatial latent gradient scale is invalid")
            applied_gradient = gradient / gradient_max_abs_tensor
        proposed = guidance_latent - normalized_step_size * applied_gradient
        mask = _validated_update_mask(update_mask, guidance_latent)
        if mask is not None:
            proposed = guidance_latent + (proposed - guidance_latent) * mask
        updated = proposed.detach()

    if not bool(torch.isfinite(updated).all()):
        raise ValueError("Asteria spatial updated latent must be finite")
    for parameter in parameters:
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError("Asteria spatial model parameter changed its gradient state")
    parameter_sha256_after = _parameter_sha256(parameters)
    if parameter_sha256_after != parameter_sha256_before:
        raise ValueError("Asteria spatial model parameter values changed")

    return SpatialLatentGuidanceStep(
        updated_latent=updated,
        stage=normalized_stage,
        step_index=normalized_step_index,
        sigma=normalized_sigma,
        step_size=normalized_step_size,
        loss=float(loss.detach().cpu()),
        gradient_norm=gradient_norm,
        gradient_max_abs=gradient_max_abs,
        gradient_normalization=gradient_normalization,
        input_latent_sha256=input_sha256,
        gradient_sha256=_tensor_sha256(gradient),
        updated_latent_sha256=_tensor_sha256(updated),
        parameter_sha256_before=parameter_sha256_before,
        parameter_sha256_after=parameter_sha256_after,
    )


def compute_spsa_latent_guidance_update(
    latent: torch.Tensor,
    *,
    forward_and_measure: Callable[[torch.Tensor], torch.Tensor],
    step_size: float,
    perturbation_scale: float,
    perturbation_seed: int,
    stage: str,
    step_index: int,
    sigma: float,
    frozen_parameters: Iterable[torch.Tensor] = (),
    update_mask: torch.Tensor | None = None,
    gradient_normalization: str = "none",
) -> SpatialLatentGuidanceStep:
    """Estimate the latent gradient without differentiating through the model.

    Anima's current CUDA execution path aborts inside model backward. SPSA keeps
    the approved full-text objective and latent update while requiring only two
    inference forwards, regardless of latent dimensionality.
    """

    if not isinstance(latent, torch.Tensor) or latent.numel() < 1:
        raise ValueError("Asteria spatial latent is invalid")
    if latent.layout != torch.strided or latent.is_quantized or latent.device.type == "meta":
        raise ValueError("Asteria spatial latent tensor is unsupported")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Asteria spatial latent must be finite")
    normalized_step_size = float(step_size)
    if not math.isfinite(normalized_step_size) or normalized_step_size <= 0.0:
        raise ValueError("Asteria spatial latent step size must be positive and finite")
    normalized_perturbation_scale = float(perturbation_scale)
    if (
        not math.isfinite(normalized_perturbation_scale)
        or normalized_perturbation_scale <= 0.0
        or normalized_perturbation_scale > 1.0
    ):
        raise ValueError("Asteria spatial perturbation scale is invalid")
    normalized_step_index = int(step_index)
    if normalized_step_index < 0:
        raise ValueError("Asteria spatial latent step index is invalid")
    normalized_sigma = float(sigma)
    if not math.isfinite(normalized_sigma) or normalized_sigma < 0.0:
        raise ValueError("Asteria spatial latent sigma is invalid")
    normalized_stage = validate_stage(stage)
    if gradient_normalization not in {"none", "linf"}:
        raise ValueError("Asteria spatial gradient normalization is invalid")
    parameters = _validated_parameters(frozen_parameters)
    parameter_sha256_before = _parameter_sha256(parameters)
    input_sha256 = _tensor_sha256(latent)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(perturbation_seed) & ((1 << 63) - 1))
    perturbation = torch.randint(
        0,
        2,
        latent.shape,
        generator=generator,
        device="cpu",
        dtype=torch.int8,
    )
    perturbation = perturbation.to(device=latent.device, dtype=latent.dtype)
    perturbation = perturbation.mul_(2).sub_(1)
    positive = (latent.detach() + normalized_perturbation_scale * perturbation).detach()
    negative = (latent.detach() - normalized_perturbation_scale * perturbation).detach()

    with torch.inference_mode():
        positive_loss = forward_and_measure(positive)
        negative_loss = forward_and_measure(negative)
    for value in (positive_loss, negative_loss):
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("Asteria spatial latent loss must be a scalar tensor")
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("Asteria spatial latent loss must be finite")

    loss_difference = (positive_loss.detach().float() - negative_loss.detach().float())
    gradient_scale = loss_difference / (2.0 * normalized_perturbation_scale)
    gradient = perturbation.float() * gradient_scale
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError("Asteria spatial latent gradient must be finite")
    gradient_norm_tensor = torch.linalg.vector_norm(gradient)
    gradient_norm = float(gradient_norm_tensor.cpu())
    gradient_max_abs_tensor = gradient.abs().max()
    gradient_max_abs = float(gradient_max_abs_tensor.cpu())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("Asteria spatial latent gradient must be non-zero")
    applied_gradient = gradient.to(dtype=latent.dtype)
    if gradient_normalization == "linf":
        if not math.isfinite(gradient_max_abs) or gradient_max_abs <= 0.0:
            raise ValueError("Asteria spatial latent gradient scale is invalid")
        applied_gradient = applied_gradient / gradient_max_abs_tensor.to(
            device=latent.device,
            dtype=latent.dtype,
        )
    proposed = latent.detach() - normalized_step_size * applied_gradient
    mask = _validated_update_mask(update_mask, latent)
    if mask is not None:
        proposed = latent.detach() + (proposed - latent.detach()) * mask
    updated = proposed.detach()
    if not bool(torch.isfinite(updated).all()):
        raise ValueError("Asteria spatial updated latent must be finite")

    for parameter in parameters:
        if parameter.requires_grad or parameter.grad is not None:
            raise ValueError("Asteria spatial model parameter changed its gradient state")
    parameter_sha256_after = _parameter_sha256(parameters)
    if parameter_sha256_after != parameter_sha256_before:
        raise ValueError("Asteria spatial model parameter values changed")
    measured_loss = (positive_loss.detach().float() + negative_loss.detach().float()) / 2.0

    return SpatialLatentGuidanceStep(
        updated_latent=updated,
        stage=normalized_stage,
        step_index=normalized_step_index,
        sigma=normalized_sigma,
        step_size=normalized_step_size,
        loss=float(measured_loss.cpu()),
        gradient_norm=gradient_norm,
        gradient_max_abs=gradient_max_abs,
        gradient_normalization=gradient_normalization,
        input_latent_sha256=input_sha256,
        gradient_sha256=_tensor_sha256(gradient),
        updated_latent_sha256=_tensor_sha256(updated),
        parameter_sha256_before=parameter_sha256_before,
        parameter_sha256_after=parameter_sha256_after,
    )
