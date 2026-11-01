"""Use the configured sampler and noise generator across an image-check boundary."""
from __future__ import annotations

import json
import inspect
from copy import copy
import torch


MAX_NOISE_BYTES = 256 * 1024 * 1024


def _whole_noise_shape(sample, info):
    if sample.ndim != 4 or sample.shape[0] != len(info["indices"]):
        raise ValueError("generation stage selected batch does not match the noise input")
    per_image_bytes = sample[0].numel() * max(sample.element_size(), 4)
    if per_image_bytes * (info["count"] + len(info["indices"])) > MAX_NOISE_BYTES:
        raise ValueError("generation stage noise byte limit exceeded")
    return torch.zeros_like(sample[:1]).expand(info["count"], -1, -1, -1)


def _selection(value):
    count, indices = value.get("count"), value.get("indices")
    if (type(count) is not int or not 1 <= count <= 64 or not isinstance(indices, list)
            or not indices or len(indices) > count or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < count for i in indices)):
        raise ValueError("generation stage noise selection is invalid")
    original = value.get("noise_indices")
    if original is not None and (not isinstance(original, list) or len(original) != count
                                or any(type(i) is not int or i < 0 for i in original)):
        raise ValueError("generation stage original noise indices are invalid")
    return {"count": count, "indices": list(indices),
            "noise_indices": None if original is None else list(original)}


class _SelectedNoise:
    def __init__(self, original, selection):
        self.original = original
        self.selection = _selection(selection)
        self.seed = original.seed

    def generate_noise(self, latent):
        sample = latent["samples"]
        info = self.selection
        # The expanded zero view only supplies shape/dtype. Rejected images never
        # reach VAE upscale or sampling. Preserve the generator's original batch.
        whole = {"samples": _whole_noise_shape(sample, info)}
        if info["noise_indices"] is not None:
            whole["batch_index"] = info["noise_indices"]
        generated = self.original.generate_noise(whole)
        return generated[info["indices"]]


def _selected_sampler(original, selection):
    """Inject the native full-batch noise stream through the standard sampler API."""
    from comfy.k_diffusion import sampling as native_noise

    info = _selection(selection)
    function = original.sampler_function
    accepts_noise = "noise_sampler" in inspect.signature(function).parameters
    random_inpaint = original.inpaint_options.get("random", False)
    if not accepts_noise and not random_inpaint:
        return original
    selected = copy(original)

    def sample(model, x, sigmas, **kwargs):
        whole = _whole_noise_shape(x, info)
        seed = kwargs.get("extra_args", {}).get("seed")
        if accepts_noise:
            if kwargs.get("noise_sampler") is not None:
                raise ValueError("generation stage custom intermediate noise cannot be reindexed")
            name = function.__name__
            if name.startswith("sample_dpmpp_") and "sde" in name:
                generator = native_noise.BrownianTreeNoiseSampler(
                    whole, sigmas[sigmas > 0].min(), sigmas.max(), seed=seed, cpu=not name.endswith("_gpu"))
            else:
                generator = native_noise.default_noise_sampler(whole, seed=seed)
            kwargs["noise_sampler"] = lambda sigma, sigma_next: generator(sigma, sigma_next)[info["indices"]]
        if random_inpaint:
            # DDIM's inpaint noise is separate from initial and ancestral noise.
            rng = torch.Generator(device="cpu").manual_seed((seed if seed is not None else 41) + 1)
            model.noise = torch.randn(whole.shape, generator=rng, device="cpu")[info["indices"]].to(x)
        return function(model, x, sigmas, **kwargs)

    selected.sampler_function = sample
    return selected


def _model_with_selected_sampler(model, selection):
    import comfy.patcher_extension

    info = _selection(selection)
    selected_model = model.clone()

    def outer(executor, noise, latent, sampler, sigmas, *args, **kwargs):
        return executor(noise, latent, _selected_sampler(sampler, info), sigmas, *args, **kwargs)

    selected_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                        "asteria_image_check_batch", outer)
    return selected_model


class AsteriaSelectedGenerationSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler": ("SAMPLER",), "selection": ("ASTERIA_BATCH_SELECTION",)}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "select"
    CATEGORY = "Asteria/internal"

    def select(self, sampler, selection):
        return (_selected_sampler(sampler, selection),)


class AsteriaSelectedGenerationNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "selection": ("ASTERIA_BATCH_SELECTION",)}}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "select"
    CATEGORY = "Asteria/internal"

    def select(self, noise, selection):
        return (_SelectedNoise(noise, selection),)


def _inspire_noise(parameters, registry):
    mode = parameters.get("noise_mode", "CPU")
    seed = parameters.get("noise_seed", parameters.get("seed"))
    return registry["RandomNoise //Inspire"]().get_noise(
        noise_seed=seed, noise_mode="CPU" if "cpu" in mode.lower() else "GPU(=A1111)",
        batch_seed_mode=parameters.get("batch_seed_mode", "comfy"),
        variation_seed=parameters.get("variation_seed", 0),
        variation_strength=parameters.get("variation_strength", 0.0),
        variation_method=parameters.get("variation_method", "linear"),
        internal_seed=parameters.get("internal_seed", seed),
    )[0]


class AsteriaGenerationStageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",), "sampling": ("STRING",),
        }, "optional": {"selection": ("ASTERIA_BATCH_SELECTION",), "noise_opt": ("NOISE_IMAGE",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("samples", "denoised_preview")
    FUNCTION = "sample"
    CATEGORY = "Asteria/internal"

    def sample(self, model, positive, negative, latent_image, sampling, selection=None, noise_opt=None):
        import nodes
        import comfy.sample
        import comfy.utils
        import latent_preview
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        parameters = json.loads(sampling)
        kind = parameters.pop("class_type")
        if kind not in {"KSampler", "KSamplerAdvanced", "KSampler //Inspire", "KSamplerAdvanced //Inspire"}:
            raise ValueError("generation stage sampler is not supported")
        if selection is not None:
            model = _model_with_selected_sampler(model, selection)
        latent = dict(latent_image)
        latent["samples"] = comfy.sample.fix_empty_latent_channels(
            model, latent["samples"], latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
        captured = {}
        callback = latent_preview.prepare_callback(model, parameters["steps"], captured)
        advanced = "Advanced" in kind
        seed = parameters.get("noise_seed", parameters.get("seed"))
        add_noise = parameters.get("add_noise", True) not in (False, "disable")
        leftover = parameters.get("return_with_leftover_noise", not advanced) in (True, "enable")
        noise = noise_opt
        if selection is not None and add_noise and noise is None:
            generator = (_inspire_noise(parameters, nodes.NODE_CLASS_MAPPINGS) if "Inspire" in kind
                         else Noise_RandomNoise(seed))
            noise = _SelectedNoise(generator, selection).generate_noise(latent)
        if "Inspire" in kind:
            # Delegate to the installed sampler. Do not reproduce Impact's
            # schedule/denoise implementation or patch process-global functions.
            sampled, _ = nodes.NODE_CLASS_MAPPINGS["KSamplerAdvanced //Inspire"].sample(
                model=model, positive=positive, negative=negative, latent_image=latent,
                noise_seed=seed, add_noise=add_noise, steps=parameters["steps"], cfg=parameters["cfg"],
                sampler_name=parameters["sampler_name"], scheduler=parameters["scheduler"],
                start_at_step=parameters.get("start_at_step"), end_at_step=parameters.get("end_at_step"),
                return_with_leftover_noise=leftover, denoise=parameters.get("denoise", 1.0),
                noise_mode=parameters["noise_mode"], batch_seed_mode=parameters.get("batch_seed_mode", "comfy"),
                variation_seed=parameters.get("variation_seed", 0),
                variation_strength=parameters.get("variation_strength", 0.0),
                variation_method=parameters.get("variation_method", "linear"),
                internal_seed=parameters.get("internal_seed"), noise_opt=noise, callback=callback,
            )
        else:
            if noise is None:
                noise = (comfy.sample.prepare_noise(latent["samples"], seed, latent.get("batch_index"))
                         if add_noise else comfy.sample.prepare_empty_noise(latent["samples"]))
            sampled = dict(latent)
            sampled["samples"] = comfy.sample.sample(
                model, noise, parameters["steps"], parameters["cfg"], parameters["sampler_name"],
                parameters["scheduler"], positive, negative, latent["samples"],
                denoise=parameters.get("denoise", 1.0), disable_noise=not add_noise,
                start_step=parameters.get("start_at_step"), last_step=parameters.get("end_at_step"),
                force_full_denoise=advanced and not leftover, noise_mask=latent.get("noise_mask"),
                callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
            )
            sampled.pop("downscale_ratio_spacial", None)
            sampled.pop("downscale_ratio_temporal", None)
        preview = dict(sampled)
        if "x0" in captured:
            preview["samples"] = model.model.process_latent_out(captured["x0"].cpu())
        elif leftover:
            raise ValueError("generation stage denoised preview was not captured")
        return sampled, preview


NODE_CLASS_MAPPINGS = {
    "AsteriaSelectedGenerationNoise": AsteriaSelectedGenerationNoise,
    "AsteriaGenerationStageSampler": AsteriaGenerationStageSampler,
    "AsteriaSelectedGenerationSampler": AsteriaSelectedGenerationSampler,
}
