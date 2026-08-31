from __future__ import annotations

import torch


FIELD_EPSILON = float(torch.exp(torch.tensor(-60.0)))
OVERLAP_FLOOR = 0.05


def _tail(distance: torch.Tensor, blending: float) -> torch.Tensor:
    outside = torch.clamp(distance - 0.72, min=0.0)
    exponent = torch.clamp(0.5 * torch.square(outside / max(blending, 1e-4)), max=60.0)
    return torch.clamp(torch.exp(-exponent), min=FIELD_EPSILON)


def compile_layout_fields(
    targets: tuple[dict[str, object], ...],
    *,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    ordered = sorted(targets, key=lambda item: int(item["depth_index"]))
    raw_fields: list[torch.Tensor] = []
    for target in ordered:
        radius_x = max(float(target["width"]) / 2.0, 1e-6)
        radius_y = max(float(target["height"]) / 2.0, 1e-6)
        dx = torch.abs(grid_x - float(target["center_x"])) / radius_x
        dy = torch.abs(grid_y - float(target["center_y"])) / radius_y
        if target["shape"] == "ellipse":
            distance = torch.sqrt(torch.square(dx) + torch.square(dy))
        elif target["shape"] == "rectangle":
            distance = torch.logaddexp(dx * 12.0, dy * 12.0) / 12.0
        else:
            raise ValueError("spatial_layout_field_shape_unsupported")
        raw_fields.append(_tail(distance, float(target["blending"])))
    stack = torch.stack(raw_fields)
    overlap = torch.sum(stack >= OVERLAP_FLOOR, dim=0) >= 2
    priors = torch.tensor(
        [float(target["depth_prior"]) for target in ordered],
        device=device,
        dtype=dtype,
    )
    weighted = stack * priors[:, None, None]
    prioritized = torch.where(overlap[None, :, :], weighted, stack)
    normalized = prioritized / torch.clamp(prioritized.sum(dim=0), min=1.0)
    return {
        str(target["target_id"]): normalized[index]
        for index, target in enumerate(ordered)
    }
