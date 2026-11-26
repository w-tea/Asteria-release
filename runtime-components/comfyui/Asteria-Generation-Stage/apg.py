"""Adaptive Projected Guidance, applied to denoised predictions (Algorithm 1)."""
from contextvars import ContextVar

import torch


class Guidance:
    def __init__(self, eta, norm_threshold, momentum):
        self.eta = eta
        self.limit = norm_threshold
        self.momentum = momentum
        self._state = ContextVar("asteria_apg_history")

    def sample(self, executor, *args, **kwargs):
        # Node/model clones may be cached. The history belongs to a sampler
        # invocation, not to a cached node, a checkpoint, a seed, or a thread.
        token = self._state.set(None)
        try:
            return executor(*args, **kwargs)
        finally:
            self._state.reset(token)

    def correct(self, args):
        cfg, existing = args["cond_scale"], args["denoised"]
        if cfg == 1 or (self.eta == 1 and self.limit == 0 and self.momentum == 0):
            return existing
        dtype = torch.float64 if existing.dtype == torch.float64 else torch.float32
        cond = args["cond_denoised"].to(dtype)
        delta = cond - args["uncond_denoised"].to(dtype)
        update = delta
        if self.momentum:
            previous = self._state.get()
            if previous is not None:
                update = delta + self.momentum * previous
            self._state.set(update.detach())
        axes = tuple(range(1, cond.ndim))
        if self.limit > 0:
            length = torch.linalg.vector_norm(update, dim=axes, keepdim=True)
            update = update * (self.limit / length.clamp_min(self.limit))
        if self.eta != 1:
            length = torch.linalg.vector_norm(cond, dim=axes, keepdim=True)
            direction = cond / torch.where(length == 0, 1., length)
            parallel = (update * direction).sum(dim=axes, keepdim=True) * direction
            update = update + (self.eta - 1) * parallel
        # Ordinary CFG already contributes (cfg - 1) * delta. Add only the
        # difference so other post-CFG adjustments are not replaced.
        return (existing.to(dtype) + (cfg - 1) * (update - delta)).to(existing.dtype)


class AsteriaAdaptiveProjectedGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "eta": ("FLOAT", {"default": 0.}),
                             "norm_threshold": ("FLOAT", {"default": 0., "min": 0.}),
                             "momentum": ("FLOAT", {"default": 0.})}}

    FUNCTION = "attach"
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Asteria/internal"

    def attach(self, model, eta, norm_threshold, momentum):
        from comfy.patcher_extension import WrappersMP

        result = model.clone()
        guidance = Guidance(eta, norm_threshold, momentum)
        if momentum:
            result.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "asteria_apg", guidance.sample)
        result.set_model_sampler_post_cfg_function(guidance.correct)
        return (result,)


NODE_CLASS_MAPPINGS = {"AsteriaAdaptiveProjectedGuidance": AsteriaAdaptiveProjectedGuidance}
