from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingOption:
    label: str
    runtime_name: str


SAMPLERS: tuple[SamplingOption, ...] = (
    SamplingOption("DPM++ 2M", "dpmpp_2m"),
    SamplingOption("DPM++ SDE", "dpmpp_sde"),
    SamplingOption("DPM++ 2M SDE", "dpmpp_2m_sde"),
    SamplingOption("DPM++ 2M SDE GPU", "dpmpp_2m_sde_gpu"),
    SamplingOption("DPM++ 3M SDE", "dpmpp_3m_sde"),
    SamplingOption("DPM++ 2s a RF", "dpmpp_2s_ancestral"),
    SamplingOption("Flux Realistic", "dpmpp_2s_ancestral"),
    SamplingOption("Euler a", "euler_ancestral"),
    SamplingOption("Euler", "euler"),
    SamplingOption("ER SDE", "er_sde"),
    SamplingOption("LCM", "lcm"),
    SamplingOption("LMS", "lms"),
    SamplingOption("Heun", "heun"),
    SamplingOption("DPM2", "dpm_2"),
    SamplingOption("Res Multistep", "res_multistep"),
    SamplingOption("Kohaku LoNyu Yog", "kohaku_lonyu_yog"),
    SamplingOption("Restart", "restart"),
    SamplingOption("UniPC", "uni_pc"),
    SamplingOption("DDIM", "ddim"),
    SamplingOption("PLMS", "plms"),
    SamplingOption("DPM++ 2M CFG++", "dpmpp_2m_cfg_pp"),
    SamplingOption("Euler a CFG++", "euler_ancestral_cfg_pp"),
    SamplingOption("Euler CFG++", "euler_cfg_pp"),
)

SCHEDULERS: tuple[SamplingOption, ...] = (
    SamplingOption("Automatic", "normal"),
    SamplingOption("Karras", "karras"),
    SamplingOption("Exponential", "exponential"),
    SamplingOption("Polyexponential", "polyexponential"),
    SamplingOption("Normal", "normal"),
    SamplingOption("Simple", "simple"),
    SamplingOption("Uniform", "uniform"),
    SamplingOption("SGM Uniform", "sgm_uniform"),
    SamplingOption("Linear Quadratic", "linear_quadratic"),
    SamplingOption("KL Optimal", "kl_optimal"),
    SamplingOption("DDIM", "ddim"),
    SamplingOption("Align Your Steps", "align_your_steps"),
    SamplingOption("Beta", "beta"),
    SamplingOption("Turbo", "turbo"),
    SamplingOption("Bong Tangent", "bong_tangent"),
    SamplingOption("FlowMatchEulerDiscrete", "flow_match"),
    SamplingOption("Flux2", "flux2"),
)

_SAMPLER_MAP = {option.label: option.runtime_name for option in SAMPLERS}
_SCHEDULER_MAP = {option.label: option.runtime_name for option in SCHEDULERS}


def sampler_runtime_name(label: str) -> str:
    return _SAMPLER_MAP.get(label, label)


def scheduler_runtime_name(label: str) -> str:
    return _SCHEDULER_MAP.get(label, label.lower().replace(" ", "_"))


def sampler_labels() -> list[str]:
    return [option.label for option in SAMPLERS]


def scheduler_labels() -> list[str]:
    return [option.label for option in SCHEDULERS]
