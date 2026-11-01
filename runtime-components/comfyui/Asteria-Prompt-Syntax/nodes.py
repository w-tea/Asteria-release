from __future__ import annotations

from comfy.samplers import KSampler


class AsteriaPromptStepSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        samplers = tuple(getattr(KSampler, "SAMPLERS", ("euler",)))
        schedulers = tuple(getattr(KSampler, "SCHEDULERS", ("normal",)))
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning_a": ("CONDITIONING",),
                "conditioning_b": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "schedule"
    CATEGORY = "Asteria/conditioning"

    def schedule(
        self,
        model,
        conditioning_a,
        conditioning_b,
        steps,
        sampler_name,
        scheduler,
        denoise,
    ):
        if len(conditioning_a) != 1 or len(conditioning_b) != 1:
            raise ValueError("asteria_prompt_schedule_conditioning_entry_count")
        step_count = int(steps)
        if step_count < 1 or step_count > 150:
            raise ValueError("asteria_prompt_schedule_step_count")

        sampler = KSampler(
            model,
            step_count,
            model.load_device,
            sampler_name,
            scheduler,
            float(denoise),
            model.model_options,
        )
        sigmas = sampler.sigmas
        if hasattr(sigmas, "detach"):
            sigmas = sigmas.detach()
        if hasattr(sigmas, "cpu"):
            sigmas = sigmas.cpu()
        if len(sigmas) != step_count + 1:
            raise ValueError("asteria_prompt_schedule_sigma_count")
        sigma_values = [float(sigmas[index]) for index in range(step_count + 1)]
        if any(
            sigma_values[index] < sigma_values[index + 1]
            for index in range(step_count)
        ):
            raise ValueError("asteria_prompt_schedule_sigma_order")

        scheduled = []
        for index in range(step_count):
            source = conditioning_a[0] if index % 2 == 0 else conditioning_b[0]
            tensor, source_metadata = source
            metadata = dict(source_metadata)
            if index > 0:
                metadata["timestep_start"] = (
                    sigma_values[index - 1] + sigma_values[index]
                ) / 2.0
            if index < step_count - 1:
                metadata["timestep_end"] = (
                    sigma_values[index] + sigma_values[index + 1]
                ) / 2.0
            scheduled.append([tensor, metadata])
        return (scheduled,)


NODE_CLASS_MAPPINGS = {
    "AsteriaPromptStepSchedule": AsteriaPromptStepSchedule,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteriaPromptStepSchedule": "Asteria Prompt Step Schedule",
}
