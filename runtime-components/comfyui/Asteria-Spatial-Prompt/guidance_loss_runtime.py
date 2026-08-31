from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SpatialLossWeights:
    position: float
    size: float
    separation: float
    presence: float
    global_retention: float

    def __post_init__(self) -> None:
        for name in (
            "position",
            "size",
            "separation",
            "presence",
            "global_retention",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Asteria spatial {name} loss weight is invalid")


@dataclass(frozen=True, slots=True)
class SpatialTargetLossDiagnostics:
    center_y: torch.Tensor
    center_x: torch.Tensor
    own_region_mass: torch.Tensor
    presence_region_mass: torch.Tensor
    foreign_region_mass: torch.Tensor
    attention_overlap: torch.Tensor
    self_relation_overlap: torch.Tensor


@dataclass(frozen=True, slots=True)
class SpatialLossBreakdown:
    position: torch.Tensor
    size: torch.Tensor
    separation: torch.Tensor
    presence: torch.Tensor
    global_retention: torch.Tensor
    assignment_contrast: torch.Tensor | None = None
    target_diagnostics: tuple[SpatialTargetLossDiagnostics, ...] = ()

    def weighted_components(self, weights: SpatialLossWeights) -> dict[str, torch.Tensor]:
        return {
            "position": self.position * weights.position,
            "size": self.size * weights.size,
            "separation": self.separation * weights.separation,
            "presence": self.presence * weights.presence,
            "global_retention": self.global_retention * weights.global_retention,
        }

    def weighted_total(self, weights: SpatialLossWeights) -> torch.Tensor:
        components = self.weighted_components(weights)
        return torch.stack(tuple(components.values())).sum()


def _validate_map(value: torch.Tensor, name: str, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != dimensions or value.numel() < 1:
        raise ValueError(f"Asteria spatial {name} shape is invalid")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"Asteria spatial {name} must be finite")
    if bool((value.detach() < 0).any()):
        raise ValueError(f"Asteria spatial {name} must be non-negative")
    return value


def _coordinate_grid(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = reference.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=reference.device, dtype=reference.dtype)
    x = torch.linspace(0.0, 1.0, width, device=reference.device, dtype=reference.dtype)
    return torch.meshgrid(y, x, indexing="ij")


def _moments(
    distribution: torch.Tensor,
    y_grid: torch.Tensor,
    x_grid: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mass = distribution.sum().clamp_min(epsilon)
    center_y = (distribution * y_grid).sum() / mass
    center_x = (distribution * x_grid).sum() / mass
    variance_y = (distribution * (y_grid - center_y).square()).sum() / mass
    variance_x = (distribution * (x_grid - center_x).square()).sum() / mass
    return center_y, center_x, variance_y, variance_x


def _spatial_distribution(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return value / value.sum(dim=(-2, -1), keepdim=True).clamp_min(epsilon)


def _unit_field(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return value / value.amax(dim=(-2, -1), keepdim=True).clamp_min(epsilon)


def _cosine_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    numerator = (left * right).sum(dim=(-2, -1))
    denominator = (
        torch.linalg.vector_norm(left.flatten(start_dim=-2), dim=-1)
        * torch.linalg.vector_norm(right.flatten(start_dim=-2), dim=-1)
    ).clamp_min(epsilon)
    return numerator / denominator


def spatial_loss_breakdown(
    *,
    target_attention: torch.Tensor,
    target_presence_attention: torch.Tensor | None = None,
    layout_fields: torch.Tensor,
    global_attention: torch.Tensor,
    global_baseline: torch.Tensor,
    global_retention_floor: float,
    self_relation: torch.Tensor | None = None,
    presence_threshold: float = 0.01,
    epsilon: float = 1e-8,
) -> SpatialLossBreakdown:
    target_attention = _validate_map(target_attention, "target attention", 3)
    if target_presence_attention is None:
        target_presence_attention = target_attention
    else:
        target_presence_attention = _validate_map(
            target_presence_attention,
            "target presence attention",
            3,
        )
    layout_fields = _validate_map(layout_fields, "layout fields", 3)
    global_attention = _validate_map(global_attention, "global attention", 2)
    global_baseline = _validate_map(global_baseline, "global baseline", 2)
    if target_attention.shape != layout_fields.shape:
        raise ValueError("Asteria spatial target attention and layout field shapes differ")
    if target_presence_attention.shape != target_attention.shape:
        raise ValueError("Asteria spatial target presence attention shape differs")
    if target_attention.shape[0] < 1:
        raise ValueError("Asteria spatial target count must be positive")
    if self_relation is not None:
        self_relation = _validate_map(self_relation, "self relation", 3)
        if self_relation.shape != target_attention.shape:
            raise ValueError("Asteria spatial self relation shape is invalid")
    if global_attention.shape != target_attention.shape[-2:]:
        raise ValueError("Asteria spatial global attention shape is invalid")
    if global_baseline.shape != global_attention.shape:
        raise ValueError("Asteria spatial global baseline shape is invalid")
    retention_floor = float(global_retention_floor)
    threshold = float(presence_threshold)
    if not math.isfinite(retention_floor) or retention_floor < 0.0:
        raise ValueError("Asteria spatial global retention floor is invalid")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Asteria spatial presence threshold is invalid")
    y_grid, x_grid = _coordinate_grid(global_attention)

    attention_distributions = _spatial_distribution(target_attention, epsilon)
    presence_distributions = _spatial_distribution(target_presence_attention, epsilon)
    unit_fields = _unit_field(layout_fields, epsilon)
    assignment_mass = unit_fields.sum(dim=0)
    assignment_weight = assignment_mass.clamp(max=1.0)
    desired_assignment = unit_fields / assignment_mass.clamp_min(epsilon).unsqueeze(0)
    predicted_assignment = torch.softmax(
        torch.log(target_attention.clamp_min(epsilon)),
        dim=0,
    )
    assignment_contrast = (
        (predicted_assignment - desired_assignment).square()
        * assignment_weight.unsqueeze(0)
    ).sum() / (assignment_weight.sum().clamp_min(epsilon) * target_attention.shape[0])

    position_terms = []
    size_terms = []
    separation_terms = []
    presence_terms = []
    target_diagnostics = []
    target_count = target_attention.shape[0]
    for index in range(target_count):
        attention = target_attention[index]
        attention_distribution = attention_distributions[index]
        field = unit_fields[index]
        attention_moments = _moments(attention, y_grid, x_grid, epsilon)
        field_moments = _moments(layout_fields[index], y_grid, x_grid, epsilon)
        position_terms.append(
            (attention_moments[0] - field_moments[0]).square()
            + (attention_moments[1] - field_moments[1]).square()
        )
        size_terms.append(
            (attention_moments[2] - field_moments[2]).square()
            + (attention_moments[3] - field_moments[3]).square()
        )
        other_fields = torch.cat((unit_fields[:index], unit_fields[index + 1 :]), dim=0)
        if other_fields.numel() == 0:
            separation = attention.new_zeros(())
            foreign_leakage = attention.new_zeros(())
            target_overlap = attention.new_zeros(())
        else:
            foreign_region = other_fields.amax(dim=0)
            foreign_leakage = (attention_distribution * foreign_region).sum()
            other_attention = torch.cat(
                (
                    attention_distributions[:index],
                    attention_distributions[index + 1 :],
                ),
                dim=0,
            )
            target_overlap = _cosine_overlap(
                attention_distribution.unsqueeze(0),
                other_attention,
                epsilon,
            ).mean()
            separation = foreign_leakage + target_overlap
        self_overlap = attention.new_zeros(())
        if self_relation is not None and target_count > 1:
            normalized_relations = _spatial_distribution(self_relation, epsilon)
            relation = normalized_relations[index]
            other_relations = torch.cat(
                (
                    normalized_relations[:index],
                    normalized_relations[index + 1 :],
                ),
                dim=0,
            )
            other_target_attention = torch.cat(
                (
                    attention_distributions[:index],
                    attention_distributions[index + 1 :],
                ),
                dim=0,
            )
            self_cross_terms = []
            for other_index in range(other_relations.shape[0]):
                self_cross_terms.append(
                    torch.minimum(relation, other_target_attention[other_index]).sum()
                    + torch.minimum(attention_distribution, other_relations[other_index]).sum()
                )
            self_overlap = torch.stack(self_cross_terms).mean()
            separation = separation + self_overlap
        separation_terms.append(separation)
        own_region_mass = (attention_distribution * field).sum()
        presence_region_mass = (presence_distributions[index] * field).sum()
        missing_mass = torch.relu(
            attention.new_tensor(threshold) - target_presence_attention[index].mean()
        ).square()
        response_missing = (
            1.0 - target_presence_attention[index].amax().clamp(max=1.0)
        ).square()
        presence_terms.append(
            (1.0 - own_region_mass).square()
            + 0.5 * (1.0 - presence_region_mass).square()
            + missing_mass
            + response_missing
        )
        target_diagnostics.append(
            SpatialTargetLossDiagnostics(
                center_y=attention_moments[0],
                center_x=attention_moments[1],
                own_region_mass=own_region_mass,
                presence_region_mass=presence_region_mass,
                foreign_region_mass=foreign_leakage,
                attention_overlap=target_overlap,
                self_relation_overlap=self_overlap,
            )
        )

    region_weight = layout_fields.amax(dim=0)
    current_global_mass = (global_attention * region_weight).sum()
    baseline_global_mass = (global_baseline * region_weight).sum().clamp_min(epsilon)
    retention_ratio = current_global_mass / baseline_global_mass
    global_retention = torch.relu(
        global_attention.new_tensor(retention_floor) - retention_ratio
    ).square()
    breakdown = SpatialLossBreakdown(
        position=torch.stack(position_terms).mean(),
        size=torch.stack(size_terms).mean(),
        separation=torch.stack(separation_terms).mean() + assignment_contrast,
        presence=torch.stack(presence_terms).mean(),
        global_retention=global_retention,
        assignment_contrast=assignment_contrast,
        target_diagnostics=tuple(target_diagnostics),
    )
    for value in (
        breakdown.position,
        breakdown.size,
        breakdown.separation,
        breakdown.presence,
        breakdown.global_retention,
        breakdown.assignment_contrast,
    ):
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("Asteria spatial loss must be finite")
    return breakdown


def loss_receipt_payload(
    breakdown: SpatialLossBreakdown,
    weights: SpatialLossWeights,
) -> dict[str, Any]:
    raw = {
        name: float(getattr(breakdown, name).detach().cpu())
        for name in (
            "position",
            "size",
            "separation",
            "presence",
            "global_retention",
        )
    }
    weighted_tensors = breakdown.weighted_components(weights)
    weighted = {
        name: float(value.detach().cpu()) for name, value in weighted_tensors.items()
    }
    return {
        "raw": raw,
        "weights": {
            name: float(getattr(weights, name)) for name in raw
        },
        "weighted": weighted,
        "total": float(breakdown.weighted_total(weights).detach().cpu()),
        "assignment_contrast": (
            float(breakdown.assignment_contrast.detach().cpu())
            if breakdown.assignment_contrast is not None
            else None
        ),
        "targets": [
            {
                name: float(getattr(item, name).detach().cpu())
                for name in (
                    "center_y",
                    "center_x",
                    "own_region_mass",
                    "presence_region_mass",
                    "foreign_region_mass",
                    "attention_overlap",
                    "self_relation_overlap",
                )
            }
            for item in breakdown.target_diagnostics
        ],
    }
