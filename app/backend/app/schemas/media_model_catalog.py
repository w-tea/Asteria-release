from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MediaModelDomain(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"


class MediaModelComponentRole(StrEnum):
    DIFFUSION_MODEL = "diffusion_model"
    TEXT_ENCODER = "text_encoder"
    NATIVE_TEXT_ENCODER = "native_text_encoder"
    QWEN35_TEXT_ENCODER = "qwen35_text_encoder"
    PROGRESSIVE_ADAPTER = "progressive_adapter"
    VIDEO_VAE = "video_vae"
    AUDIO_VAE = "audio_vae"
    LORA = "lora"
    LATENT_UPSCALER = "latent_upscaler"
    UNKNOWN = "unknown"


class MediaModelCompatibility(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PROVISIONAL_UNSUPPORTED = "provisional_unsupported"


class MediaModelClassification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component_role: MediaModelComponentRole = MediaModelComponentRole.UNKNOWN
    architecture: str = "unknown"
    variant: str | None = None
    precision: str | None = None
    quantization: str | None = None
    supported_tasks: tuple[str, ...] = ()
    compatible_architecture_ids: tuple[str, ...] = ()
    compatible_variant_ids: tuple[str, ...] = ()
    required_component_roles: tuple[MediaModelComponentRole, ...] = ()
    compatible_profile_ids: tuple[str, ...] = ()
    compatibility: MediaModelCompatibility = MediaModelCompatibility.UNSUPPORTED
    reason_code: str | None = None


class MediaModelDescriptor(MediaModelClassification):
    stable_asset_id: str = Field(min_length=1)
    domain: MediaModelDomain
    display_name: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    source: str
    root_id: str
    fingerprint_algorithm: str | None = None
    fingerprint_version: int | None = None
    fingerprint_value: str | None = None


class MediaModelCatalogResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: MediaModelDomain
    items: tuple[MediaModelDescriptor, ...]
