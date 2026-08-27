# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Copyright 2026 Asteria contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import comfy.ops
from comfy.ldm.modules.diffusionmodules.model import vae_attention


ops = comfy.ops.disable_weight_init


def is_qwen2d_state_dict(state_dict: Mapping[str, torch.Tensor] | None) -> bool:
    if not state_dict:
        return False
    required = {
        "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.conv_in.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
    }
    if not required.issubset(state_dict):
        return False
    return state_dict["decoder.conv_in.weight"].ndim == 4


class RMSNorm2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels**0.5
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(tensor.float(), dim=1).to(tensor.dtype)
        return normalized * self.scale * self.gamma.to(tensor)


class Upsample2D(nn.Upsample):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().forward(tensor.float()).to(tensor.dtype)


class Resample2D(nn.Module):
    def __init__(self, channels: int, mode: str) -> None:
        super().__init__()
        if mode == "up":
            self.resample = nn.Sequential(
                Upsample2D(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                ops.Conv2d(channels, channels // 2, 3, padding=1),
            )
        elif mode == "down":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                ops.Conv2d(channels, channels, 3, stride=2),
            )
        else:
            raise ValueError(f"Unsupported Qwen2D resample mode: {mode}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.resample(tensor)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm2D(in_channels)
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = RMSNorm2D(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            ops.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = self.conv_shortcut(tensor)
        tensor = self.conv1(F.silu(self.norm1(tensor)))
        tensor = self.conv2(self.dropout(F.silu(self.norm2(tensor))))
        return tensor + residual


class AttentionBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm2D(channels)
        self.to_qkv = ops.Conv2d(channels, channels * 3, 1)
        self.proj = ops.Conv2d(channels, channels, 1)
        self._attention = vae_attention()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor
        query, key, value = self.to_qkv(self.norm(tensor)).chunk(3, dim=1)
        return self.proj(self._attention(query, key, value)) + residual


class MidBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResidualBlock2D(channels, channels, dropout),
                ResidualBlock2D(channels, channels, dropout),
            ]
        )
        self.attentions = nn.ModuleList([AttentionBlock2D(channels)])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.resnets[0](tensor)
        tensor = self.attentions[0](tensor)
        return self.resnets[1](tensor)


class Encoder2D(nn.Module):
    def __init__(
        self,
        base_dim: int,
        latent_moments: int,
        image_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [base_dim * item for item in (1, *dim_mult)]
        self.conv_in = ops.Conv2d(image_channels, dimensions[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock2D(in_channels, out_channels, dropout)
                )
                in_channels = out_channels
            if index != len(dim_mult) - 1:
                self.down_blocks.append(Resample2D(out_channels, "down"))
        self.mid_block = MidBlock2D(out_channels, dropout)
        self.norm_out = RMSNorm2D(out_channels)
        self.conv_out = ops.Conv2d(out_channels, latent_moments, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv_in(tensor)
        for layer in self.down_blocks:
            tensor = layer(tensor)
        tensor = self.mid_block(tensor)
        return self.conv_out(F.silu(self.norm_out(tensor)))


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float,
        *,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        for _ in range(num_res_blocks + 1):
            self.resnets.append(
                ResidualBlock2D(in_channels, out_channels, dropout)
            )
            in_channels = out_channels
        self.upsamplers = (
            nn.ModuleList([Resample2D(out_channels, "up")])
            if upsample
            else None
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.resnets:
            tensor = layer(tensor)
        if self.upsamplers is not None:
            tensor = self.upsamplers[0](tensor)
        return tensor


class Decoder2D(nn.Module):
    def __init__(
        self,
        base_dim: int,
        latent_channels: int,
        image_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [base_dim * item for item in (dim_mult[-1], *reversed(dim_mult))]
        self.conv_in = ops.Conv2d(latent_channels, dimensions[0], 3, padding=1)
        self.mid_block = MidBlock2D(dimensions[0], dropout)
        self.up_blocks = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            if index > 0:
                in_channels //= 2
            self.up_blocks.append(
                UpBlock2D(
                    in_channels,
                    out_channels,
                    num_res_blocks,
                    dropout,
                    upsample=index != len(dim_mult) - 1,
                )
            )
        self.norm_out = RMSNorm2D(out_channels)
        self.conv_out = ops.Conv2d(out_channels, image_channels, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv_in(tensor)
        tensor = self.mid_block(tensor)
        for layer in self.up_blocks:
            tensor = layer(tensor)
        return self.conv_out(F.silu(self.norm_out(tensor)))


class Qwen2DVAE(nn.Module):
    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        image_channels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder2D(
            base_dim,
            z_dim * 2,
            image_channels,
            dim_mult,
            num_res_blocks,
            dropout,
        )
        self.quant_conv = ops.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = ops.Conv2d(z_dim, z_dim, 1)
        self.decoder = Decoder2D(
            base_dim,
            z_dim,
            image_channels,
            dim_mult,
            num_res_blocks,
            dropout,
        )

    @staticmethod
    def _to_images(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        if tensor.ndim == 4:
            return tensor, None
        if tensor.ndim != 5:
            raise ValueError("Qwen2D VAE expects a 4D image batch or 5D frame batch")
        batch, channels, frames, height, width = tensor.shape
        images = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        return images, (batch, frames)

    @staticmethod
    def _from_images(
        tensor: torch.Tensor,
        frame_shape: tuple[int, int] | None,
    ) -> torch.Tensor:
        if frame_shape is None:
            return tensor
        batch, frames = frame_shape
        channels, height, width = tensor.shape[1:]
        return tensor.reshape(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        images, frame_shape = self._to_images(tensor)
        moments = self.quant_conv(self.encoder(images))
        mean, _ = moments.chunk(2, dim=1)
        return self._from_images(mean, frame_shape)

    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        latents, frame_shape = self._to_images(tensor)
        images = self.decoder(self.post_quant_conv(latents))
        return self._from_images(images, frame_shape)
