from dataclasses import dataclass


@dataclass(frozen=True)
class UpscalerOption:
    label: str
    kind: str
    runtime_name: str
    model_name: str | None = None
    dependency_id: str | None = None


UPSCALERS: tuple[UpscalerOption, ...] = (
    UpscalerOption("Latent", "latent", "bilinear"),
    UpscalerOption("Latent (antialiased)", "latent", "bilinear"),
    UpscalerOption("Latent (bicubic)", "latent", "bicubic"),
    UpscalerOption("Latent (bicubic antialiased)", "latent", "bicubic"),
    UpscalerOption("Latent (nearest)", "latent", "nearest"),
    UpscalerOption("Latent (nearest-exact)", "latent", "nearest-exact"),
    UpscalerOption("Lanczos", "image", "lanczos"),
    UpscalerOption("Bicubic", "image", "bicubic"),
    UpscalerOption("Bilinear", "image", "bilinear"),
    UpscalerOption("Nearest", "image", "nearest-exact"),
    UpscalerOption("Area", "image", "area"),
    UpscalerOption("ESRGAN 4x", "model", "ESRGAN.pth", "ESRGAN.pth", "esrgan-4x"),
    UpscalerOption(
        "R-ESGAN4x",
        "model",
        "RealESRGAN_x4plus.safetensors",
        "RealESRGAN_x4plus.safetensors",
        "realesrgan-x4plus",
    ),
    UpscalerOption(
        "R-ESGAN4x Anime6B",
        "model",
        "RealESRGAN_x4plus_anime_6B.pth",
        "RealESRGAN_x4plus_anime_6B.pth",
        "realesrgan-x4plus-anime6b",
    ),
    UpscalerOption("Topaz Photo AI", "external", "topaz_photo_ai"),
)

_UPSCALER_MAP = {option.label: option for option in UPSCALERS}


def upscaler_labels() -> list[str]:
    return [option.label for option in UPSCALERS]


def upscaler_option(label: str | None) -> UpscalerOption:
    return _UPSCALER_MAP.get(str(label or "Latent"), _UPSCALER_MAP["Latent"])


def upscaler_model_name(label: str | None, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return upscaler_option(label).model_name
