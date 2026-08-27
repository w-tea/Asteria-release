from __future__ import annotations

import logging

import torch

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sd as comfy_sd

from .qwen2d import Qwen2DVAE, is_qwen2d_state_dict


_PATCH_MARKER = "_asteria_qwen2d_vae_patch_v1"


def _frame_count(shape: tuple[int, ...]) -> int:
    return shape[2] if len(shape) == 5 else 1


def _initialize_qwen2d_vae(
    vae: comfy_sd.VAE,
    state_dict: dict[str, torch.Tensor],
    *,
    device=None,
    dtype=None,
) -> None:
    latent_channels = state_dict["decoder.conv_in.weight"].shape[1]
    image_channels = state_dict["decoder.conv_out.weight"].shape[0]
    base_dim = state_dict["decoder.norm_out.gamma"].shape[0]

    vae.memory_used_encode = lambda shape, value_dtype: (
        1500
        * _frame_count(shape)
        * shape[-2]
        * shape[-1]
        * model_management.dtype_size(value_dtype)
    )
    vae.memory_used_decode = lambda shape, value_dtype: (
        2200
        * _frame_count(shape)
        * shape[-2]
        * shape[-1]
        * 64
        * model_management.dtype_size(value_dtype)
    )
    vae.downscale_ratio = (lambda frames: frames, 8, 8)
    vae.upscale_ratio = (lambda frames: frames, 8, 8)
    vae.downscale_index_formula = (1, 8, 8)
    vae.upscale_index_formula = (1, 8, 8)
    vae.latent_channels = latent_channels
    vae.latent_dim = 3
    vae.output_channels = image_channels
    vae.pad_channel_value = None
    vae.process_input = lambda image: image.mul(2.0).sub(1.0)
    vae.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
    vae.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    vae.disable_offload = False
    vae.not_video = True
    vae.size = None
    vae.extra_1d_channel = None
    vae.crop_input = True
    vae.handles_tiling = False
    vae.format_encoded = None
    vae.audio_sample_rate = 44100

    vae.first_stage_model = Qwen2DVAE(
        base_dim=base_dim,
        z_dim=latent_channels,
        image_channels=image_channels,
    ).eval()
    vae.device = device if device is not None else model_management.vae_device()
    offload_device = model_management.vae_offload_device()
    vae.vae_dtype = (
        dtype
        if dtype is not None
        else model_management.vae_dtype(vae.device, vae.working_dtypes)
    )
    vae.first_stage_model.to(vae.vae_dtype)
    model_management.archive_model_dtypes(vae.first_stage_model)
    vae.output_device = model_management.intermediate_device()
    vae.patcher = comfy.model_patcher.CoreModelPatcher(
        vae.first_stage_model,
        load_device=vae.device,
        offload_device=offload_device,
    )
    missing, unexpected = vae.first_stage_model.load_state_dict(
        state_dict,
        strict=False,
        assign=vae.patcher.is_dynamic(),
    )
    if missing or unexpected:
        raise RuntimeError(
            "Qwen2D VAE weights do not match the Asteria runtime contract: "
            f"missing={missing}, unexpected={unexpected}"
        )
    logging.info(
        "Loaded Qwen2D VAE on %s with offload device %s and dtype %s",
        vae.device,
        offload_device,
        vae.vae_dtype,
    )
    vae.model_size()


def install_qwen2d_vae_support() -> None:
    if getattr(comfy_sd.VAE, _PATCH_MARKER, False):
        return

    original_init = comfy_sd.VAE.__init__

    def qwen2d_aware_init(
        self,
        sd=None,
        device=None,
        config=None,
        dtype=None,
        metadata=None,
    ):
        if config is None and is_qwen2d_state_dict(sd):
            _initialize_qwen2d_vae(self, sd, device=device, dtype=dtype)
            return
        original_init(
            self,
            sd=sd,
            device=device,
            config=config,
            dtype=dtype,
            metadata=metadata,
        )

    comfy_sd.VAE.__init__ = qwen2d_aware_init
    setattr(comfy_sd.VAE, _PATCH_MARKER, True)
    logging.info("Asteria Qwen2D VAE support is active")
