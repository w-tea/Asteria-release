from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable

import torch

from .latent_guidance_runtime import SpatialLatentGuidanceStep, _tensor_sha256
from .guidance_loss_runtime import (
    SpatialLossWeights,
    loss_receipt_payload,
    spatial_loss_breakdown,
)
from .latent_guidance_runtime import (
    compute_latent_guidance_update,
    compute_spsa_latent_guidance_update,
    identity_latent_guidance_step,
)


@dataclass(frozen=True, slots=True)
class SpatialSolverStepReceipt:
    step_index: int
    input_latent_sha256: str
    guidance_input_latent_sha256: str
    updated_latent_sha256: str
    guidance_iteration_count: int
    guidance_loss: float
    guidance_gradient_norm: float
    guidance_gradient_max_abs: float
    guidance_gradient_normalization: str
    guidance_update_norm: float
    guidance_step_size: float
    sampler_input_latent_sha256: str
    sampler_output_latent_sha256: str
    next_step_input_latent_sha256: str | None
    sampler_consumption_count: int


@dataclass(frozen=True, slots=True)
class SpatialGuidedSolverResult:
    samples: torch.Tensor
    sampler_name: str
    steps: tuple[SpatialSolverStepReceipt, ...]
    sampler_attention_prior_count: int = 0


def _append_dims(value: torch.Tensor, dimensions: int) -> torch.Tensor:
    return value[(...,) + (None,) * (dimensions - value.ndim)]


def _to_d(current: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (current - denoised) / _append_dims(sigma, current.ndim)


def _ancestral_step(
    sigma_from: torch.Tensor,
    sigma_to: torch.Tensor,
    *,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not eta:
        return sigma_to, sigma_to.new_zeros(())
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (
            sigma_to.square()
            * (sigma_from.square() - sigma_to.square())
            / sigma_from.square()
        ).sqrt(),
    )
    return (sigma_to.square() - sigma_up.square()).sqrt(), sigma_up


def _receipt(
    *,
    step_index: int,
    input_latent: torch.Tensor,
    guidance: SpatialLatentGuidanceStep,
    sampler_input: torch.Tensor,
    sampler_output: torch.Tensor,
) -> SpatialSolverStepReceipt:
    input_digest = _tensor_sha256(input_latent)
    sampler_input_digest = _tensor_sha256(sampler_input)
    if guidance.input_latent_sha256 != input_digest:
        raise ValueError("Asteria spatial guidance consumed the wrong latent")
    if guidance.updated_latent_sha256 != sampler_input_digest:
        raise ValueError("Asteria spatial sampler did not consume the guided latent")
    return SpatialSolverStepReceipt(
        step_index=step_index,
        input_latent_sha256=input_digest,
        guidance_input_latent_sha256=guidance.input_latent_sha256,
        updated_latent_sha256=guidance.updated_latent_sha256,
        guidance_iteration_count=guidance.iteration_count if guidance.guided else 0,
        guidance_loss=guidance.loss,
        guidance_gradient_norm=guidance.gradient_norm,
        guidance_gradient_max_abs=guidance.gradient_max_abs,
        guidance_gradient_normalization=guidance.gradient_normalization,
        guidance_update_norm=float(
            torch.linalg.vector_norm((sampler_input - input_latent).float()).detach().cpu()
        ),
        guidance_step_size=guidance.step_size,
        sampler_input_latent_sha256=sampler_input_digest,
        sampler_output_latent_sha256=_tensor_sha256(sampler_output),
        next_step_input_latent_sha256=None,
        sampler_consumption_count=1,
    )

def _seal_next_step(
    receipts: list[SpatialSolverStepReceipt],
    next_input: torch.Tensor,
) -> None:
    if not receipts:
        return
    previous = receipts[-1]
    next_digest = _tensor_sha256(next_input)
    if previous.sampler_output_latent_sha256 != next_digest:
        raise ValueError("Asteria spatial solver trajectory is discontinuous")
    receipts[-1] = SpatialSolverStepReceipt(
        step_index=previous.step_index,
        input_latent_sha256=previous.input_latent_sha256,
        guidance_input_latent_sha256=previous.guidance_input_latent_sha256,
        updated_latent_sha256=previous.updated_latent_sha256,
        guidance_iteration_count=previous.guidance_iteration_count,
        guidance_loss=previous.guidance_loss,
        guidance_gradient_norm=previous.guidance_gradient_norm,
        guidance_gradient_max_abs=previous.guidance_gradient_max_abs,
        guidance_gradient_normalization=previous.guidance_gradient_normalization,
        guidance_update_norm=previous.guidance_update_norm,
        guidance_step_size=previous.guidance_step_size,
        sampler_input_latent_sha256=previous.sampler_input_latent_sha256,
        sampler_output_latent_sha256=previous.sampler_output_latent_sha256,
        next_step_input_latent_sha256=next_digest,
        sampler_consumption_count=previous.sampler_consumption_count,
    )


def run_guided_solver(
    sampler_name: str,
    *,
    model: Callable[..., torch.Tensor],
    initial_latent: torch.Tensor,
    sigmas: torch.Tensor,
    guide: Callable[[torch.Tensor, torch.Tensor, int], SpatialLatentGuidanceStep],
    extra_args: dict[str, Any] | None = None,
    noise_sampler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    flow_model: bool = False,
    flow_noise_scale: float = 1.0,
    er_lambdas: torch.Tensor | None = None,
    begin_sampler_forward: Callable[[int, int], None] | None = None,
    finish_sampler_forward: Callable[[], int] | None = None,
) -> SpatialGuidedSolverResult:
    if sampler_name not in {"euler_ancestral", "dpmpp_2m", "er_sde"}:
        raise ValueError("Asteria spatial sampler is unsupported")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("Asteria spatial sigma schedule is invalid")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("Asteria spatial sigma schedule must be finite")
    if sampler_name == "er_sde":
        if (
            not isinstance(er_lambdas, torch.Tensor)
            or er_lambdas.ndim != 1
            or len(er_lambdas) != len(sigmas)
            or not bool(torch.isfinite(er_lambdas).all())
        ):
            raise ValueError("Asteria spatial ER SDE schedule is invalid")
    if (begin_sampler_forward is None) != (finish_sampler_forward is None):
        raise ValueError("Asteria spatial sampler attention callbacks are incomplete")
    arguments = dict(extra_args or {})
    normalized_flow_noise_scale = float(flow_noise_scale)
    if not math.isfinite(normalized_flow_noise_scale) or normalized_flow_noise_scale < 0.0:
        raise ValueError("Asteria spatial flow noise scale is invalid")
    current = initial_latent.detach().clone()
    receipts: list[SpatialSolverStepReceipt] = []
    old_denoised: torch.Tensor | None = None
    old_er_denoised: torch.Tensor | None = None
    old_er_denoised_d: torch.Tensor | None = None
    if noise_sampler is None:
        noise_sampler = lambda _sigma, _next: torch.randn_like(current)
    s_in = current.new_ones((current.shape[0],))
    total_steps = len(sigmas) - 1
    sampler_attention_prior_count = 0

    with torch.no_grad():
        for index in range(total_steps):
            _seal_next_step(receipts, current)
            input_latent = current.detach().clone()
            guidance = guide(input_latent, sigmas[index], index)
            sampler_input = guidance.updated_latent.detach()
            if begin_sampler_forward is not None:
                begin_sampler_forward(index, total_steps)
            try:
                denoised = model(sampler_input, sigmas[index] * s_in, **arguments)
            finally:
                if finish_sampler_forward is not None:
                    count = finish_sampler_forward()
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError(
                            "Asteria spatial sampler attention prior count is invalid"
                        )
                    sampler_attention_prior_count += count
            if not isinstance(denoised, torch.Tensor) or denoised.shape != sampler_input.shape:
                raise ValueError("Asteria spatial sampler model output is invalid")
            if not bool(torch.isfinite(denoised).all()):
                raise ValueError("Asteria spatial sampler model output must be finite")
            if callback is not None:
                callback(
                    {
                        "x": sampler_input,
                        "i": index,
                        "sigma": sigmas[index],
                        "sigma_hat": sigmas[index],
                        "denoised": denoised,
                    }
                )

            if sampler_name == "euler_ancestral":
                if flow_model and float(sigmas[index + 1]) == 0.0:
                    current = denoised
                elif flow_model:
                    downstep_ratio = 1.0 + (
                        sigmas[index + 1] / sigmas[index] - 1.0
                    )
                    sigma_down = sigmas[index + 1] * downstep_ratio
                    alpha_next = 1.0 - sigmas[index + 1]
                    alpha_down = 1.0 - sigma_down
                    renoise = (
                        sigmas[index + 1].square()
                        - sigma_down.square()
                        * alpha_next.square()
                        / alpha_down.square()
                    ).clamp_min(0.0).sqrt()
                    ratio = sigma_down / sigmas[index]
                    current = ratio * sampler_input + (1.0 - ratio) * denoised
                    current = (
                        alpha_next / alpha_down * current
                        + noise_sampler(sigmas[index], sigmas[index + 1])
                        * normalized_flow_noise_scale
                        * renoise
                    )
                else:
                    sigma_down, sigma_up = _ancestral_step(
                        sigmas[index], sigmas[index + 1]
                    )
                    if float(sigma_down) == 0.0:
                        current = denoised
                    else:
                        derivative = _to_d(sampler_input, sigmas[index], denoised)
                        delta = sigma_down - sigmas[index]
                        current = (
                            sampler_input
                            + derivative * delta
                            + noise_sampler(sigmas[index], sigmas[index + 1]) * sigma_up
                        )
            elif sampler_name == "dpmpp_2m":
                t = -sigmas[index].log()
                t_next = -sigmas[index + 1].log()
                h = t_next - t
                if old_denoised is None or float(sigmas[index + 1]) == 0.0:
                    current = (
                        torch.exp(-t_next) / torch.exp(-t) * sampler_input
                        - torch.expm1(-h) * denoised
                    )
                else:
                    h_last = t - (-sigmas[index - 1].log())
                    ratio = h_last / h
                    denoised_derivative = (
                        (1 + 1 / (2 * ratio)) * denoised
                        - (1 / (2 * ratio)) * old_denoised
                    )
                    current = (
                        torch.exp(-t_next) / torch.exp(-t) * sampler_input
                        - torch.expm1(-h) * denoised_derivative
                    )
                old_denoised = denoised
            else:
                assert er_lambdas is not None
                if float(sigmas[index + 1]) == 0.0:
                    current = denoised
                else:
                    er_lambda_s = er_lambdas[index]
                    er_lambda_t = er_lambdas[index + 1]
                    alpha_s = sigmas[index] / er_lambda_s
                    alpha_t = sigmas[index + 1] / er_lambda_t

                    def noise_scale(value: torch.Tensor) -> torch.Tensor:
                        return value * (value.pow(0.3).exp() + 10.0)

                    ratio = noise_scale(er_lambda_t) / noise_scale(er_lambda_s)
                    current = (
                        alpha_t / alpha_s * ratio * sampler_input
                        + alpha_t * (1.0 - ratio) * denoised
                    )
                    stage_used = min(3, index + 1)
                    if stage_used >= 2:
                        integration_points = 200.0
                        point_indices = torch.arange(
                            0,
                            integration_points,
                            dtype=torch.float32,
                            device=sampler_input.device,
                        )
                        delta = er_lambda_t - er_lambda_s
                        lambda_step = -delta / integration_points
                        lambda_positions = er_lambda_t + point_indices * lambda_step
                        scaled_positions = noise_scale(lambda_positions)
                        integral = torch.sum(1.0 / scaled_positions) * lambda_step
                        assert old_er_denoised is not None
                        denoised_d = (denoised - old_er_denoised) / (
                            er_lambda_s - er_lambdas[index - 1]
                        )
                        current = current + alpha_t * (
                            delta + integral * noise_scale(er_lambda_t)
                        ) * denoised_d
                        if stage_used >= 3:
                            assert old_er_denoised_d is not None
                            integral_u = torch.sum(
                                (lambda_positions - er_lambda_s) / scaled_positions
                            ) * lambda_step
                            denoised_u = (denoised_d - old_er_denoised_d) / (
                                (er_lambda_s - er_lambdas[index - 2]) / 2.0
                            )
                            current = current + alpha_t * (
                                delta.square() / 2.0
                                + integral_u * noise_scale(er_lambda_t)
                            ) * denoised_u
                        old_er_denoised_d = denoised_d
                    if normalized_flow_noise_scale > 0.0:
                        variance = (
                            er_lambda_t.square()
                            - er_lambda_s.square() * ratio.square()
                        ).sqrt().nan_to_num(nan=0.0)
                        current = (
                            current
                            + alpha_t
                            * noise_sampler(sigmas[index], sigmas[index + 1])
                            * normalized_flow_noise_scale
                            * variance
                        )
                old_er_denoised = denoised
            if not bool(torch.isfinite(current).all()):
                raise ValueError("Asteria spatial solver output must be finite")
            receipts.append(
                _receipt(
                    step_index=index,
                    input_latent=input_latent,
                    guidance=guidance,
                    sampler_input=sampler_input,
                    sampler_output=current,
                )
            )

    return SpatialGuidedSolverResult(
        samples=current,
        sampler_name=sampler_name,
        steps=tuple(receipts),
        sampler_attention_prior_count=sampler_attention_prior_count,
    )


def _profile_value(profile: Any, name: str) -> Any:
    if not hasattr(profile, name):
        raise ValueError(f"Asteria spatial guidance profile is missing {name}")
    return getattr(profile, name)


def _positive_condition_forward(
    *,
    calc_cond_batch: Callable[..., Any],
    model: Any,
    positive: Any,
    current: torch.Tensor,
    sigma: torch.Tensor,
    model_options: dict[str, Any],
) -> torch.Tensor:
    """Run one logical positive full-text forward for layout guidance only."""

    outputs = calc_cond_batch(
        model,
        [positive],
        current,
        sigma,
        model_options,
    )
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("Asteria spatial positive guidance output is invalid")
    output = outputs[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("Asteria spatial positive guidance output is invalid")
    return output


def _select_initial_latent_candidate(
    candidates: tuple[tuple[int, torch.Tensor], ...],
    *,
    measure: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, int, int, tuple[float, ...]]:
    if not candidates:
        raise ValueError("Asteria spatial initial noise candidates are empty")
    losses: list[float] = []
    for seed, candidate in candidates:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Asteria spatial initial noise candidate seed is invalid")
        if not isinstance(candidate, torch.Tensor) or not bool(torch.isfinite(candidate).all()):
            raise ValueError("Asteria spatial initial noise candidate is invalid")
        measured = measure(candidate)
        if not isinstance(measured, torch.Tensor) or measured.numel() != 1:
            raise ValueError("Asteria spatial initial noise score is invalid")
        loss = float(measured.detach().cpu())
        if not math.isfinite(loss):
            raise ValueError("Asteria spatial initial noise score must be finite")
        losses.append(loss)
    selected_index = min(range(len(losses)), key=losses.__getitem__)
    selected_seed, selected = candidates[selected_index]
    return selected.detach(), selected_seed, selected_index, tuple(losses)


def _align_initial_noise_candidates(
    candidates: tuple[tuple[int, torch.Tensor], ...],
    *,
    reference: torch.Tensor,
) -> tuple[tuple[int, torch.Tensor], ...]:
    if not isinstance(reference, torch.Tensor):
        raise ValueError("Asteria spatial sampler noise reference is invalid")
    return tuple(
        (
            seed,
            candidate.to(device=reference.device, dtype=reference.dtype),
        )
        for seed, candidate in candidates
    )


def _normal_tensor_for_autograd(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None or not value.is_inference():
        return value
    with torch.inference_mode(False):
        return value.clone()


def _optimize_initial_latent_candidate(
    candidate: torch.Tensor,
    *,
    iterations: int,
    update: Callable[[torch.Tensor, int], SpatialLatentGuidanceStep],
) -> tuple[torch.Tensor, tuple[float, ...]]:
    normalized_iterations = int(iterations)
    if not 1 <= normalized_iterations <= 8:
        raise ValueError("Asteria spatial initial noise optimization count is invalid")
    current = candidate.detach()
    losses: list[float] = []
    for iteration_index in range(normalized_iterations):
        result = update(current, iteration_index)
        if not isinstance(result, SpatialLatentGuidanceStep):
            raise ValueError("Asteria spatial initial noise optimization result is invalid")
        if result.updated_latent.shape != current.shape:
            raise ValueError("Asteria spatial initial noise optimization shape changed")
        if not math.isfinite(result.loss):
            raise ValueError("Asteria spatial initial noise optimization loss is invalid")
        losses.append(result.loss)
        current = result.updated_latent.detach()
    return current, tuple(losses)


def _spatial_guide(
    *,
    model: Callable[..., torch.Tensor],
    guidance_state: Any,
    sigmas: torch.Tensor,
    extra_args: dict[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor, int], SpatialLatentGuidanceStep]:
    profile = guidance_state.profile
    start = float(_profile_value(profile, "guidance_start_percent"))
    end = float(_profile_value(profile, "guidance_end_percent"))
    iterations = int(_profile_value(profile, "guidance_iterations"))
    step_size = float(_profile_value(profile, "guidance_step_size"))
    retention_floor = float(_profile_value(profile, "global_retention_floor"))
    weights = _profile_value(profile, "loss_weights")
    if not isinstance(weights, SpatialLossWeights):
        raise ValueError("Asteria spatial guidance loss weights are invalid")
    if not (0.0 <= start < end <= 1.0) or not 1 <= iterations <= 8:
        raise ValueError("Asteria spatial guidance schedule is invalid")
    total_steps = max(1, len(sigmas) - 1)
    model_family = str(getattr(profile, "model_family", "sdxl"))
    stage = str(guidance_state.execution_body["stage"])
    gradient_estimator = (
        "spsa" if model_family == "anima" or stage == "hires" else "autograd"
    )

    def guide(current: torch.Tensor, sigma: torch.Tensor, step_index: int):
        progress = step_index / total_steps
        collector = getattr(guidance_state, "collector", None)
        activate_binding = getattr(collector, "activate_step_progress", None)
        if callable(activate_binding):
            active_binding = activate_binding(progress)
            guidance_state.execution_body.setdefault("active_binding_steps", []).append(
                {
                    "step_index": step_index,
                    "binding_sha256": active_binding.receipt_sha256,
                    "full_text_sha256": active_binding.full_text_sha256,
                }
            )
        if progress < start or progress >= end:
            return identity_latent_guidance_step(
                current,
                stage=guidance_state.execution_body["stage"],
                step_index=step_index,
                sigma=float(sigma),
            )
        schedule_progress = min(1.0, max(0.0, (progress - start) / (end - start)))
        effective_step_size = step_size * (1.0 - 0.5 * schedule_progress)
        guided = current
        first_input_sha256: str | None = None
        result: SpatialLatentGuidanceStep | None = None
        for iteration_index in range(iterations):
            captured: dict[str, Any] = {}

            def forward_and_measure(guidance_latent: torch.Tensor) -> torch.Tensor:
                guidance_state.collector.begin_iteration()
                try:
                    batch_sigma = sigma * guidance_latent.new_ones(
                        (guidance_latent.shape[0],)
                    )
                    model(guidance_latent, batch_sigma, **extra_args)
                    snapshot = guidance_state.collector.finish_iteration()
                except Exception:
                    guidance_state.collector.abort_iteration()
                    raise
                layout_fields = guidance_state.collector.layout_fields.to(
                    device=snapshot.target_attention.device,
                    dtype=snapshot.target_attention.dtype,
                )
                if guidance_state.global_attention_baseline is None:
                    guidance_state.global_attention_baseline = (
                        snapshot.global_attention.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                    )
                global_baseline = guidance_state.global_attention_baseline.to(
                    device=snapshot.global_attention.device,
                    dtype=snapshot.global_attention.dtype,
                )
                breakdown = spatial_loss_breakdown(
                    target_attention=snapshot.target_attention,
                    target_presence_attention=snapshot.target_presence_attention,
                    layout_fields=layout_fields,
                    global_attention=snapshot.global_attention,
                    global_baseline=global_baseline,
                    global_retention_floor=retention_floor,
                    self_relation=snapshot.self_relation,
                )
                captured["snapshot"] = snapshot
                captured["loss"] = loss_receipt_payload(breakdown, weights)
                return breakdown.weighted_total(weights)

            if gradient_estimator == "spsa":
                perturbation_seed = (
                    int(_tensor_sha256(guided)[:16], 16)
                    ^ (step_index << 16)
                    ^ iteration_index
                )
                result = compute_spsa_latent_guidance_update(
                    guided,
                    forward_and_measure=forward_and_measure,
                    step_size=effective_step_size,
                    perturbation_scale=0.01,
                    perturbation_seed=perturbation_seed,
                    stage=guidance_state.execution_body["stage"],
                    step_index=step_index,
                    sigma=float(sigma),
                    gradient_normalization="linf",
                )
            else:
                result = compute_latent_guidance_update(
                    guided,
                    forward_and_measure=forward_and_measure,
                    step_size=effective_step_size,
                    stage=guidance_state.execution_body["stage"],
                    step_index=step_index,
                    sigma=float(sigma),
                    gradient_normalization="linf",
                )
            if first_input_sha256 is None:
                first_input_sha256 = result.input_latent_sha256
            update_norm = float(
                torch.linalg.vector_norm(
                    (result.updated_latent - guided).float()
                ).detach().cpu()
            )
            snapshot = captured["snapshot"]
            guidance_state.loss_receipts = guidance_state.loss_receipts + (
                {
                    "step_index": step_index,
                    "iteration_index": iteration_index,
                    "sigma": float(sigma),
                    **captured["loss"],
                    "gradient_norm": result.gradient_norm,
                    "gradient_max_abs": result.gradient_max_abs,
                    "gradient_normalization": result.gradient_normalization,
                    "gradient_estimator": gradient_estimator,
                    "update_norm": update_norm,
                    "step_size": effective_step_size,
                    "cross_attention_call_count": snapshot.cross_attention_call_count,
                    "self_attention_call_count": snapshot.self_attention_call_count,
                    "guidance_prior_bias_count": getattr(
                        snapshot,
                        "guidance_prior_bias_count",
                        0,
                    ),
                },
            )
            guided = result.updated_latent
        if result is None or first_input_sha256 is None:
            raise ValueError("Asteria spatial guidance produced no latent update")
        return replace(
            result,
            input_latent_sha256=first_input_sha256,
            iteration_count=iterations,
        )

    return guide


class SpatialComfySampler:
    def __init__(
        self,
        *,
        guidance_state: Any,
        sampler_name: str,
        initial_noise_candidates: tuple[tuple[int, torch.Tensor], ...],
        preserve_initial_noise: bool = False,
    ) -> None:
        if sampler_name not in {"euler_ancestral", "dpmpp_2m", "er_sde"}:
            raise ValueError("Asteria spatial sampler is unsupported")
        self.guidance_state = guidance_state
        self.sampler_name = sampler_name
        self.initial_noise_candidates = initial_noise_candidates
        self.preserve_initial_noise = bool(preserve_initial_noise)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        del disable_pbar
        import comfy.model_sampling
        import comfy.samplers
        from comfy.k_diffusion.sampling import (
            default_noise_sampler,
            offset_first_sigma_for_snr,
            sigma_to_half_log_snr,
        )

        guidance_denoise_mask = _normal_tensor_for_autograd(denoise_mask)
        guidance_noise = _normal_tensor_for_autograd(noise)
        guidance_latent_image = _normal_tensor_for_autograd(latent_image)
        arguments = dict(extra_args)
        arguments["denoise_mask"] = guidance_denoise_mask
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        current = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0],
            noise,
            latent_image,
            self.max_denoise(model_wrap, sigmas),
        )
        flow_model = isinstance(
            model_wrap.inner_model.model_sampling,
            comfy.model_sampling.CONST,
        )
        flow_noise_scale = float(
            getattr(model_wrap.inner_model.model_sampling, "noise_scale", 1.0)
        )
        total_steps = len(sigmas) - 1

        def positive_model(current, sigma, **positive_arguments):
            denoise_mask = positive_arguments.get("denoise_mask")
            guidance_input = current
            if denoise_mask is not None:
                latent_mask = 1.0 - denoise_mask
                guidance_input = (
                    current * denoise_mask
                    + model_wrap.inner_model.scale_latent_inpaint(
                        x=current,
                        sigma=sigma,
                        noise=guidance_noise,
                        latent_image=guidance_latent_image,
                    )
                    * latent_mask
                )
            return _positive_condition_forward(
                calc_cond_batch=comfy.samplers.calc_cond_batch,
                model=model_wrap.inner_model,
                positive=model_wrap.conds["positive"],
                current=guidance_input,
                sigma=sigma,
                model_options=positive_arguments.get("model_options", {}),
            )

        def solver_callback(payload: dict[str, Any]) -> None:
            if callback is not None:
                callback(
                    payload["i"],
                    payload["denoised"],
                    payload["x"],
                    total_steps,
                )

        profile = self.guidance_state.profile
        loss_weights = _profile_value(profile, "loss_weights")
        if not isinstance(loss_weights, SpatialLossWeights):
            raise ValueError("Asteria spatial guidance loss weights are invalid")
        aligned_initial_noise_candidates = _align_initial_noise_candidates(
            self.initial_noise_candidates,
            reference=noise,
        )
        scaled_candidates = tuple(
            (
                candidate_seed,
                model_wrap.inner_model.model_sampling.noise_scaling(
                    sigmas[0],
                    candidate_noise,
                    latent_image,
                    self.max_denoise(model_wrap, sigmas),
                ),
            )
            for candidate_seed, candidate_noise in aligned_initial_noise_candidates
        )

        def measure_initial_candidate(candidate: torch.Tensor) -> torch.Tensor:
            self.guidance_state.collector.activate_step_progress(0.0)
            self.guidance_state.collector.begin_iteration()
            try:
                batch_sigma = sigmas[0] * candidate.new_ones((candidate.shape[0],))
                positive_model(candidate, batch_sigma, **arguments)
                snapshot = self.guidance_state.collector.finish_iteration()
            except Exception:
                self.guidance_state.collector.abort_iteration()
                raise
            layout_fields = self.guidance_state.collector.layout_fields.to(
                device=snapshot.target_attention.device,
                dtype=snapshot.target_attention.dtype,
            )
            breakdown = spatial_loss_breakdown(
                target_attention=snapshot.target_attention,
                target_presence_attention=snapshot.target_presence_attention,
                layout_fields=layout_fields,
                global_attention=snapshot.global_attention,
                global_baseline=snapshot.global_attention.detach(),
                global_retention_floor=float(
                    _profile_value(profile, "global_retention_floor")
                ),
                self_relation=snapshot.self_relation,
            )
            return breakdown.weighted_total(loss_weights)

        initial_noise_optimization_iterations = (
            0
            if (
                self.preserve_initial_noise
                or
                str(getattr(profile, "model_family", "sdxl")) == "anima"
                or self.guidance_state.execution_body["stage"] == "hires"
            )
            else min(3, int(_profile_value(profile, "guidance_iterations")))
        )
        optimized_candidates: list[tuple[int, torch.Tensor]] = []
        initial_noise_optimization_losses: list[list[float]] = []
        for candidate_seed, scaled_candidate in scaled_candidates:
            if initial_noise_optimization_iterations == 0:
                optimized_candidate = scaled_candidate.detach()
                optimization_losses: tuple[float, ...] = ()
            else:
                optimized_candidate, optimization_losses = _optimize_initial_latent_candidate(
                    scaled_candidate,
                    iterations=initial_noise_optimization_iterations,
                    update=lambda current_candidate, _iteration_index: (
                        compute_latent_guidance_update(
                            current_candidate,
                            forward_and_measure=measure_initial_candidate,
                            step_size=float(
                                _profile_value(profile, "guidance_step_size")
                            ),
                            stage=self.guidance_state.execution_body["stage"],
                            step_index=0,
                            sigma=float(sigmas[0]),
                            gradient_normalization="linf",
                        )
                    ),
                )
            optimized_candidates.append((candidate_seed, optimized_candidate))
            initial_noise_optimization_losses.append(list(optimization_losses))

        current, selected_seed, selected_index, candidate_losses = (
            _select_initial_latent_candidate(
                tuple(optimized_candidates),
                measure=measure_initial_candidate,
            )
        )
        noise = aligned_initial_noise_candidates[selected_index][1]
        guidance_noise = _normal_tensor_for_autograd(noise)
        model_k.noise = noise
        self.guidance_state.execution_body.update(
            {
                "initial_noise_candidate_count": len(scaled_candidates),
                "initial_noise_evaluation_forward_count": len(scaled_candidates)
                * (initial_noise_optimization_iterations + 1),
                "initial_noise_candidate_seeds": [
                    candidate_seed for candidate_seed, _candidate in scaled_candidates
                ],
                "initial_noise_optimization_iterations": (
                    initial_noise_optimization_iterations
                ),
                "initial_noise_optimization_losses": (
                    initial_noise_optimization_losses
                ),
                "initial_noise_candidate_losses": list(candidate_losses),
                "initial_noise_selected_index": selected_index,
                "initial_noise_selected_seed": selected_seed,
                "initial_noise_selected_latent_sha256": _tensor_sha256(current),
            }
        )
        guidance_start = float(_profile_value(profile, "guidance_start_percent"))
        guidance_end = float(_profile_value(profile, "guidance_end_percent"))

        def begin_sampler_forward(index: int, step_count: int) -> None:
            progress = index / max(1, step_count)
            self.guidance_state.collector.activate_step_progress(progress)
            if progress < guidance_start or progress >= guidance_end:
                multiplier = 0.0
            else:
                scheduled = (progress - guidance_start) / (
                    guidance_end - guidance_start
                )
                multiplier = 1.0 - 0.5 * min(1.0, max(0.0, scheduled))
            self.guidance_state.collector.begin_sampler_step(
                scale_multiplier=multiplier
            )

        solver_sigmas = sigmas
        er_lambdas = None
        if self.sampler_name == "er_sde":
            model_sampling = model_wrap.inner_model.model_sampling
            solver_sigmas = offset_first_sigma_for_snr(sigmas, model_sampling)
            er_lambdas = sigma_to_half_log_snr(
                solver_sigmas,
                model_sampling,
            ).neg().exp()

        result = run_guided_solver(
            self.sampler_name,
            model=model_k,
            initial_latent=current,
            sigmas=solver_sigmas,
            guide=_spatial_guide(
                model=positive_model,
                guidance_state=self.guidance_state,
                sigmas=solver_sigmas,
                extra_args=arguments,
            ),
            extra_args=arguments,
            noise_sampler=default_noise_sampler(
                current,
                seed=selected_seed,
            ),
            callback=solver_callback,
            flow_model=flow_model,
            flow_noise_scale=flow_noise_scale,
            er_lambdas=er_lambdas,
            begin_sampler_forward=begin_sampler_forward,
            finish_sampler_forward=self.guidance_state.collector.finish_sampler_step,
        )
        self.guidance_state.solver_receipts = result.steps
        self.guidance_state.execution_body["latent_update_count"] = sum(
            1 for receipt in self.guidance_state.loss_receipts
        )
        self.guidance_state.execution_body[
            "sampler_attention_prior_count"
        ] = result.sampler_attention_prior_count
        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1],
            result.samples,
        )

    @staticmethod
    def max_denoise(model_wrap, sigmas) -> bool:
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-5) or sigma > max_sigma


def sample_comfy_spatial_guidance(
    *,
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    denoise: float,
    guidance_state: Any,
    noise_opt: torch.Tensor | None = None,
    add_noise: str = "enable",
    start_at_step: int = 0,
    end_at_step: int = 10000,
    return_with_leftover_noise: str = "disable",
) -> tuple[dict[str, Any], Any]:
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
        raise ValueError("Asteria spatial sampler latent is invalid")
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    batch_indices = latent.get("batch_index")
    if add_noise not in {"enable", "disable"}:
        raise ValueError("Asteria spatial add-noise mode is invalid")
    if return_with_leftover_noise not in {"enable", "disable"}:
        raise ValueError("Asteria spatial leftover-noise mode is invalid")
    if (
        isinstance(start_at_step, bool)
        or not isinstance(start_at_step, int)
        or isinstance(end_at_step, bool)
        or not isinstance(end_at_step, int)
        or start_at_step < 0
        or end_at_step <= start_at_step
    ):
        raise ValueError("Asteria spatial sampler step range is invalid")
    max_seed = 0xFFFFFFFFFFFFFFFF
    preserve_initial_noise = add_noise == "disable" or noise_opt is not None
    if add_noise == "disable":
        noise = comfy.sample.prepare_empty_noise(latent_image)
        initial_noise_candidates = ((int(seed), noise),)
    elif noise_opt is not None:
        if (
            not isinstance(noise_opt, torch.Tensor)
            or noise_opt.shape != latent_image.shape
            or not bool(torch.isfinite(noise_opt).all())
        ):
            raise ValueError("Asteria spatial custom noise is invalid")
        noise = noise_opt.to(device=latent_image.device, dtype=latent_image.dtype)
        initial_noise_candidates = ((int(seed), noise),)
    else:
        candidate_seeds = (
            int(seed),
            (int(seed) - 1) & max_seed,
            (int(seed) + 1) & max_seed,
            (int(seed) + 2) & max_seed,
        )
        initial_noise_candidates = tuple(
            (
                candidate_seed,
                comfy.sample.prepare_noise(latent_image, candidate_seed, batch_indices),
            )
            for candidate_seed in candidate_seeds
        )
        noise = initial_noise_candidates[0][1]
    noise_mask = latent.get("noise_mask")
    schedule = comfy.samplers.KSampler(
        model,
        steps=int(steps),
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=float(denoise),
        model_options=model.model_options,
    ).sigmas
    if end_at_step < len(schedule) - 1:
        schedule = schedule[: end_at_step + 1].clone()
        if return_with_leftover_noise == "disable":
            schedule[-1] = 0
    if start_at_step < len(schedule) - 1:
        schedule = schedule[start_at_step:]
    else:
        raise ValueError("Asteria spatial sampler step range is outside the schedule")
    callback = latent_preview.prepare_callback(model, int(steps))
    samples = comfy.sample.sample_custom(
        model,
        noise,
        float(cfg),
        SpatialComfySampler(
            guidance_state=guidance_state,
            sampler_name=sampler_name,
            initial_noise_candidates=initial_noise_candidates,
            preserve_initial_noise=preserve_initial_noise,
        ),
        schedule,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )
    output = latent.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    guidance_state.execution_body.update(
        {
            "add_noise": add_noise,
            "custom_noise_consumption_count": int(noise_opt is not None),
            "sampler_start_at_step": start_at_step,
            "sampler_end_at_step": min(end_at_step, int(steps)),
            "return_with_leftover_noise": return_with_leftover_noise,
        }
    )
    return output, guidance_state
