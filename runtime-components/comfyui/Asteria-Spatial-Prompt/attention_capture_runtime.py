from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Callable

import torch
import torch.nn.functional as functional

from .spatial_contract import (
    SpatialEncoderTokenBinding,
    SpatialScheduledTokenBindingBundle,
    SpatialTokenBindingBundle,
)


_SELF_PRIOR_RELATION_CHUNK_ROWS = 64


@dataclass(frozen=True, slots=True)
class SpatialEncoderAttentionMaps:
    encoder_id: str
    target_attention: torch.Tensor
    target_presence_attention: torch.Tensor
    global_attention: torch.Tensor
    global_context_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SpatialAttentionSnapshot:
    encoders: tuple[SpatialEncoderAttentionMaps, ...]
    target_attention: torch.Tensor
    target_presence_attention: torch.Tensor
    global_attention: torch.Tensor
    self_relation: torch.Tensor
    cross_attention_call_count: int
    self_attention_call_count: int
    guidance_prior_bias_count: int = 0
    target_only_attention_forward_count: int = 0
    global_only_attention_forward_count: int = 0
    spatial_output_blend_count: int = 0
    background_substitution_count: int = 0


@dataclass(frozen=True, slots=True)
class _PendingAttentionCall:
    kind: str
    options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CrossRecord:
    encoders: tuple[SpatialEncoderAttentionMaps, ...]


@dataclass(frozen=True, slots=True)
class _SelfRecord:
    relation: torch.Tensor
    grid_height: int
    grid_width: int


def _factor_grid(tokens: int, source_height: int, source_width: int) -> tuple[int, int]:
    if tokens < 1:
        raise ValueError("Asteria spatial attention token count is invalid")
    source_ratio = source_width / max(source_height, 1)
    candidates: list[tuple[float, int, int]] = []
    for height in range(1, math.isqrt(tokens) + 1):
        if tokens % height:
            continue
        width = tokens // height
        for candidate_height, candidate_width in ((height, width), (width, height)):
            ratio = candidate_width / max(candidate_height, 1)
            candidates.append(
                (
                    abs(math.log(max(ratio, 1e-9) / source_ratio)),
                    candidate_height,
                    candidate_width,
                )
            )
    if not candidates:
        return 1, tokens
    _, height, width = min(
        candidates,
        key=lambda item: (item[0], abs(item[1] - item[2])),
    )
    return height, width


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    anima_protocol = "extra_options" in kwargs
    if anima_protocol:
        options = kwargs.get("extra_options")
    elif args and isinstance(args[0], dict):
        options = args[0]
    else:
        options = {}
    return dict(options) if isinstance(options, dict) else {}, anima_protocol


def _positive_rows(options: dict[str, Any], batch: int, device: torch.device) -> torch.Tensor:
    branches = options.get("cond_or_uncond")
    if not isinstance(branches, (list, tuple)) or not branches or batch % len(branches):
        return torch.ones(batch, dtype=torch.bool, device=device)
    rows_per_branch = batch // len(branches)
    return torch.tensor(
        [branch == 0 for branch in branches for _ in range(rows_per_branch)],
        dtype=torch.bool,
        device=device,
    )


def _reshape_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    heads: int,
    *,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(heads, bool) or int(heads) < 1:
        raise ValueError("Asteria spatial attention head count is invalid")
    head_count = int(heads)
    if q.ndim == 3:
        batch, query_tokens, inner = q.shape
        if inner % head_count:
            raise ValueError("Asteria spatial query width is incompatible with its heads")
        dimension = inner // head_count
        query = q.reshape(batch, query_tokens, head_count, dimension).permute(0, 2, 1, 3)
        if k.shape[0] != batch or k.ndim != 3 or k.shape[-1] % dimension:
            raise ValueError("Asteria spatial key shape is incompatible")
        key_heads = k.shape[-1] // dimension if enable_gqa else head_count
        if k.shape[-1] != key_heads * dimension:
            raise ValueError("Asteria spatial key width is incompatible with its heads")
        key = k.reshape(batch, k.shape[1], key_heads, dimension).permute(0, 2, 1, 3)
    elif q.ndim == 4:
        batch, query_heads, _query_tokens, dimension = q.shape
        if query_heads != head_count or k.ndim != 4 or k.shape[0] != batch:
            raise ValueError("Asteria spatial projected attention shape is incompatible")
        query = q
        key = k
        key_heads = int(key.shape[1])
    else:
        raise ValueError("Asteria spatial attention tensor shape is unsupported")
    if key_heads != head_count:
        if not enable_gqa or head_count % key_heads:
            raise ValueError("Asteria spatial grouped-query attention shape is incompatible")
        key = key.repeat_interleave(head_count // key_heads, dim=1)
    return query, key


def _normalized_mask(
    mask: torch.Tensor,
    *,
    batch: int,
    heads: int,
    query_tokens: int,
    key_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    current = mask.to(device=device)
    if current.ndim == 2:
        current = current.unsqueeze(0).unsqueeze(0)
    elif current.ndim == 3:
        current = current.unsqueeze(1)
    elif current.ndim != 4:
        raise ValueError("Asteria spatial attention mask shape is unsupported")
    try:
        return torch.broadcast_to(current, (batch, heads, query_tokens, key_tokens))
    except RuntimeError as error:
        raise ValueError("Asteria spatial attention mask shape is incompatible") from error


def _attention_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
    kwargs: dict[str, Any],
    *,
    source_height: int,
    source_width: int,
    max_query_tokens: int,
    pool_keys: bool,
) -> torch.Tensor:
    query, key = _reshape_heads(
        q,
        k,
        heads,
        enable_gqa=bool(kwargs.get("enable_gqa", False)),
    )
    query = _pool_spatial_attention_tokens(
        query,
        source_height=source_height,
        source_width=source_width,
        max_tokens=max_query_tokens,
    )
    if pool_keys:
        key = _pool_spatial_attention_tokens(
            key,
            source_height=source_height,
            source_width=source_width,
            max_tokens=max_query_tokens,
        )
    scale = float(kwargs.get("scale", query.shape[-1] ** -0.5))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Asteria spatial attention scale is invalid")
    logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if mask is not None:
        normalized = _normalized_mask(
            mask,
            batch=int(logits.shape[0]),
            heads=int(logits.shape[1]),
            query_tokens=int(logits.shape[2]),
            key_tokens=int(logits.shape[3]),
            device=logits.device,
        )
        if normalized.dtype == torch.bool:
            logits = logits.masked_fill(~normalized, -torch.finfo(logits.dtype).max)
        else:
            logits = logits + normalized.to(dtype=logits.dtype)
    return logits.softmax(dim=-1)


def _pool_spatial_attention_tokens(
    value: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    max_tokens: int,
) -> torch.Tensor:
    tokens = int(value.shape[-2])
    if tokens <= max_tokens:
        return value
    grid_height, grid_width = _factor_grid(tokens, source_height, source_width)
    scale = math.sqrt(max_tokens / tokens)
    target_height = max(1, int(math.floor(grid_height * scale)))
    target_width = max(1, int(math.floor(grid_width * scale)))
    while target_height * target_width > max_tokens:
        if target_width >= target_height and target_width > 1:
            target_width -= 1
        elif target_height > 1:
            target_height -= 1
        else:
            break
    batch, heads, _tokens, dimension = value.shape
    spatial = value.permute(0, 1, 3, 2).reshape(
        batch * heads,
        dimension,
        grid_height,
        grid_width,
    )
    pooled = functional.adaptive_avg_pool2d(
        spatial.float(),
        (target_height, target_width),
    ).to(dtype=value.dtype)
    return pooled.reshape(
        batch,
        heads,
        dimension,
        target_height * target_width,
    ).permute(0, 1, 3, 2)


def _validated_layer_selectors(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Asteria spatial {label} selectors are invalid")
    for value in values:
        parts = value.split(":")
        if value == "all":
            continue
        if parts[0] == "block" and len(parts) == 3:
            if parts[1] not in {"input", "middle", "output"} or not parts[2].isdigit():
                raise ValueError(f"Asteria spatial {label} selector is invalid")
        elif parts[0] == "block_index" and len(parts) == 2:
            if not parts[1].isdigit():
                raise ValueError(f"Asteria spatial {label} selector is invalid")
        else:
            raise ValueError(f"Asteria spatial {label} selector is invalid")
    return values


def _matches_layer_selector(options: dict[str, Any], selectors: tuple[str, ...]) -> bool:
    block = options.get("block")
    block_index = options.get("block_index")
    for selector in selectors:
        if selector == "all":
            return True
        parts = selector.split(":")
        if parts[0] == "block":
            if (
                isinstance(block, (tuple, list))
                and len(block) >= 2
                and str(block[0]) == parts[1]
                and int(block[1]) == int(parts[2])
            ):
                return True
        elif (
            parts[0] == "block_index"
            and isinstance(block_index, int)
            and not isinstance(block_index, bool)
            and block_index == int(parts[1])
        ):
            return True
    return False


def _resize_map(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if value.shape[-2:] == (height, width):
        return value
    if value.ndim == 2:
        return functional.interpolate(
            value.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if value.ndim == 3:
        return functional.interpolate(
            value.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
    raise ValueError("Asteria spatial attention map shape is unsupported")


def _otsu_high_response_mask(value: torch.Tensor) -> torch.Tensor:
    detached = value.detach().float().reshape(-1)
    minimum = detached.amin()
    maximum = detached.amax()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("Asteria spatial target attention contains non-finite values")
    if float(maximum - minimum) <= 1e-8:
        mask = torch.zeros_like(detached, dtype=torch.bool)
        mask[detached.argmax()] = True
        return mask.reshape(value.shape)
    bins = min(64, max(8, int(detached.numel()) * 2))
    histogram = torch.histc(detached, bins=bins, min=float(minimum), max=float(maximum))
    centers = torch.linspace(
        minimum,
        maximum,
        bins,
        device=detached.device,
        dtype=detached.dtype,
    )
    lower_weight = histogram.cumsum(dim=0)
    upper_weight = histogram.sum() - lower_weight
    lower_mean = (histogram * centers).cumsum(dim=0) / lower_weight.clamp_min(1.0)
    upper_total = (histogram * centers).sum()
    upper_mean = (
        upper_total - (histogram * centers).cumsum(dim=0)
    ) / upper_weight.clamp_min(1.0)
    between = lower_weight * upper_weight * (lower_mean - upper_mean).square()
    between[(lower_weight <= 0) | (upper_weight <= 0)] = -1.0
    threshold = centers[int(between.argmax())]
    mask = detached > threshold
    if not bool(mask.any()):
        mask[detached.argmax()] = True
    return mask.reshape(value.shape)


def _merge_attention_mask(
    existing: Any,
    routing: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if existing is None:
        return routing
    if not isinstance(existing, torch.Tensor):
        raise ValueError("Asteria spatial attention mask type is unsupported")
    current = existing.to(device=reference.device)
    if current.dtype == torch.bool:
        current = torch.where(
            current,
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.full(
                (),
                -torch.finfo(reference.dtype).max,
                device=reference.device,
                dtype=reference.dtype,
            ),
        )
    else:
        current = current.to(dtype=reference.dtype)
    if current.ndim == 4 and routing.ndim == 3:
        routing = routing.unsqueeze(1)
    return current + routing


class SpatialAttentionCollector:
    """Collect loss inputs from one unchanged full-text attention forward."""

    def __init__(
        self,
        *,
        token_binding: SpatialTokenBindingBundle | SpatialScheduledTokenBindingBundle,
        target_ids: tuple[str, ...],
        layout_fields: torch.Tensor,
        prior_attention_override: Callable[..., Any] | None = None,
        max_self_attention_tokens: int = 2048,
        max_cross_attention_tokens: int = 2048,
        attention_layers: tuple[str, ...] = ("all",),
        self_relation_layers: tuple[str, ...] = ("all",),
        attention_prior_scale: float = 0.0,
    ) -> None:
        if not isinstance(layout_fields, torch.Tensor) or layout_fields.ndim != 3:
            raise ValueError("Asteria spatial layout field shape is invalid")
        if not bool(torch.isfinite(layout_fields).all()) or bool((layout_fields < 0).any()):
            raise ValueError("Asteria spatial layout fields are invalid")
        if len(target_ids) < 1 or len(target_ids) != len(set(target_ids)):
            raise ValueError("Asteria spatial target ids are invalid")
        if int(layout_fields.shape[0]) != len(target_ids):
            raise ValueError("Asteria spatial layout field target count differs")
        if isinstance(max_self_attention_tokens, bool) or int(max_self_attention_tokens) < 1:
            raise ValueError("Asteria spatial self attention token budget is invalid")
        if isinstance(max_cross_attention_tokens, bool) or int(max_cross_attention_tokens) < 1:
            raise ValueError("Asteria spatial cross attention token budget is invalid")
        bindings = (
            token_binding.bindings
            if isinstance(token_binding, SpatialScheduledTokenBindingBundle)
            else (token_binding,)
        )
        if not bindings or any(not binding.encoders for binding in bindings):
            raise ValueError("Asteria spatial token binding has no encoders")
        normalized_prior_scale = float(attention_prior_scale)
        if (
            not math.isfinite(normalized_prior_scale)
            or normalized_prior_scale < 0.0
            or normalized_prior_scale > 5.0
        ):
            raise ValueError("Asteria spatial attention prior scale is invalid")
        context_encoder_schedules: list[tuple[SpatialEncoderTokenBinding, ...]] = []
        for binding in bindings:
            for encoder in binding.encoders:
                if tuple(target.target_id for target in encoder.targets) != target_ids:
                    raise ValueError("Asteria spatial token binding target ids differ")
            context_encoders = tuple(
                encoder
                for encoder in binding.encoders
                if all(target.adapter_context_indices for target in encoder.targets)
            )
            if not context_encoders:
                raise ValueError(
                    "Asteria spatial token binding has no attention context encoder"
                )
            context_encoder_schedules.append(context_encoders)
        self.token_binding = token_binding
        self.bindings = tuple(bindings)
        self.context_encoder_schedules = tuple(context_encoder_schedules)
        self.target_ids = target_ids
        self.layout_fields = layout_fields.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.prior_attention_override = prior_attention_override
        self.max_self_attention_tokens = int(max_self_attention_tokens)
        self.max_cross_attention_tokens = int(max_cross_attention_tokens)
        self.attention_layers = _validated_layer_selectors(
            tuple(attention_layers),
            "cross attention",
        )
        self.self_relation_layers = _validated_layer_selectors(
            tuple(self_relation_layers),
            "self attention",
        )
        self.attention_prior_scale = normalized_prior_scale
        self._local = threading.local()

    @property
    def context_encoders(self) -> tuple[SpatialEncoderTokenBinding, ...]:
        index = int(getattr(self._local, "binding_index", 0))
        return self.context_encoder_schedules[index]

    def activate_step_progress(self, progress: float) -> SpatialTokenBindingBundle:
        normalized = float(progress)
        if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
            raise ValueError("Asteria spatial binding progress is invalid")
        selected_index = len(self.bindings) - 1
        for index, binding in enumerate(self.bindings):
            if binding.step_start_percent <= normalized < binding.step_end_percent:
                selected_index = index
                break
        self._local.binding_index = selected_index
        return self.bindings[selected_index]

    def begin_iteration(self) -> None:
        if bool(getattr(self._local, "active", False)) or bool(
            getattr(self._local, "sampler_step_open", False)
        ):
            raise ValueError("Asteria spatial attention capture is already active")
        self._local.active = True
        self._local.call = None
        self._local.cross_records = []
        self._local.self_records = []
        self._local.guidance_prior_bias_count = 0

    def begin_sampler_step(self, *, scale_multiplier: float) -> None:
        if bool(getattr(self._local, "active", False)) or bool(
            getattr(self._local, "sampler_step_open", False)
        ):
            raise ValueError("Asteria spatial attention prior is already active")
        multiplier = float(scale_multiplier)
        if not math.isfinite(multiplier) or not 0.0 <= multiplier <= 1.0:
            raise ValueError("Asteria spatial attention prior multiplier is invalid")
        self._local.sampler_step_open = True
        self._local.sampler_bias_active = (
            self.attention_prior_scale > 0.0 and multiplier > 0.0
        )
        self._local.sampler_bias_scale = self.attention_prior_scale * multiplier
        self._local.sampler_prior_bias_count = 0
        self._local.call = None

    def finish_sampler_step(self) -> int:
        if not bool(getattr(self._local, "sampler_step_open", False)):
            raise ValueError("Asteria spatial attention prior is not active")
        count = int(getattr(self._local, "sampler_prior_bias_count", 0))
        self._local.sampler_step_open = False
        self._local.sampler_bias_active = False
        self._local.sampler_bias_scale = 0.0
        self._local.sampler_prior_bias_count = 0
        self._local.call = None
        return count

    def _finish_state(self) -> tuple[list[_CrossRecord], list[_SelfRecord], int]:
        if not bool(getattr(self._local, "active", False)):
            raise ValueError("Asteria spatial attention capture is not active")
        cross_records = list(getattr(self._local, "cross_records", ()))
        self_records = list(getattr(self._local, "self_records", ()))
        guidance_prior_bias_count = int(
            getattr(self._local, "guidance_prior_bias_count", 0)
        )
        self._local.active = False
        self._local.call = None
        self._local.cross_records = []
        self._local.self_records = []
        self._local.guidance_prior_bias_count = 0
        return cross_records, self_records, guidance_prior_bias_count

    def abort_iteration(self) -> None:
        if bool(getattr(self._local, "active", False)):
            self._finish_state()

    def finish_iteration(self) -> SpatialAttentionSnapshot:
        cross_records, self_records, guidance_prior_bias_count = self._finish_state()
        if not cross_records:
            raise ValueError("Asteria spatial cross attention was not captured")
        source_height, source_width = self.layout_fields.shape[-2:]
        encoder_maps: list[SpatialEncoderAttentionMaps] = []
        for encoder_index, encoder in enumerate(self.context_encoders):
            records = [record.encoders[encoder_index] for record in cross_records]
            encoder_maps.append(
                SpatialEncoderAttentionMaps(
                    encoder_id=encoder.encoder_id,
                    target_attention=torch.stack(
                        [item.target_attention for item in records]
                    ).mean(dim=0),
                    target_presence_attention=torch.stack(
                        [item.target_presence_attention for item in records]
                    ).mean(dim=0),
                    global_attention=torch.stack(
                        [item.global_attention for item in records]
                    ).mean(dim=0),
                    global_context_indices=records[0].global_context_indices,
                )
            )
        combined_targets = torch.stack(
            [item.target_attention for item in encoder_maps]
        ).mean(dim=0)
        combined_presence = torch.stack(
            [item.target_presence_attention for item in encoder_maps]
        ).mean(dim=0)
        combined_global = torch.stack(
            [item.global_attention for item in encoder_maps]
        ).mean(dim=0)
        if self_records:
            aggregated_self_records: list[torch.Tensor] = []
            for record in self_records:
                target_maps = _resize_map(
                    combined_targets,
                    record.grid_height,
                    record.grid_width,
                ).reshape(len(self.target_ids), -1)
                aggregated_targets: list[torch.Tensor] = []
                for target_map in target_maps:
                    selected = _otsu_high_response_mask(target_map)
                    weights = torch.where(selected, target_map, torch.zeros_like(target_map))
                    if float(weights.detach().sum()) <= 1e-8:
                        weights = torch.zeros_like(target_map)
                        weights[target_map.detach().argmax()] = 1.0
                    aggregated = torch.matmul(weights, record.relation) / weights.sum().clamp_min(1e-8)
                    aggregated_targets.append(
                        aggregated.reshape(record.grid_height, record.grid_width)
                    )
                aggregated_self_records.append(
                    _resize_map(
                        torch.stack(aggregated_targets),
                        source_height,
                        source_width,
                    )
                )
            self_relation = torch.stack(aggregated_self_records).mean(dim=0)
        else:
            self_relation = combined_targets.new_zeros(
                (len(self.target_ids), source_height, source_width)
            )
        return SpatialAttentionSnapshot(
            encoders=tuple(encoder_maps),
            target_attention=combined_targets,
            target_presence_attention=combined_presence,
            global_attention=combined_global,
            self_relation=self_relation,
            cross_attention_call_count=len(cross_records),
            self_attention_call_count=len(self_records),
            guidance_prior_bias_count=guidance_prior_bias_count,
        )
    def cross_patch(self, q, k, v, *args, **kwargs):
        options, anima_protocol = _options(args, kwargs)
        if bool(getattr(self._local, "active", False)) or bool(
            getattr(self._local, "sampler_bias_active", False)
        ):
            self._local.call = _PendingAttentionCall(kind="cross", options=options)
        if anima_protocol:
            return {"q": q, "k": k, "v": v}
        return q, k, v

    def self_patch(self, q, k, v, *args, **kwargs):
        options, anima_protocol = _options(args, kwargs)
        if bool(getattr(self._local, "active", False)) or bool(
            getattr(self._local, "sampler_bias_active", False)
        ):
            self._local.call = _PendingAttentionCall(kind="self", options=options)
        if anima_protocol:
            return {"q": q, "k": k, "v": v}
        return q, k, v

    def _global_indices(
        self,
        encoder: SpatialEncoderTokenBinding,
        key_tokens: int,
    ) -> tuple[int, ...]:
        excluded = set(encoder.special_token_indices) | set(encoder.padding_token_indices)
        excluded.update(
            index
            for target in encoder.targets
            for index in target.adapter_context_indices
        )
        indices = tuple(
            index
            for index in range(min(len(encoder.token_ids), key_tokens))
            if index not in excluded
        )
        if not indices:
            raise ValueError("Asteria spatial global token binding is empty")
        return indices

    def _cross_record(self, probabilities: torch.Tensor, options: dict[str, Any]) -> None:
        positive = _positive_rows(options, int(probabilities.shape[0]), probabilities.device)
        if not bool(positive.any()):
            raise ValueError("Asteria spatial positive attention row is missing")
        selected = probabilities[positive]
        mean_attention = selected.mean(dim=(0, 1))
        query_tokens, key_tokens = int(mean_attention.shape[0]), int(mean_attention.shape[1])
        source_height, source_width = self.layout_fields.shape[-2:]
        grid_height, grid_width = _factor_grid(query_tokens, source_height, source_width)
        encoder_maps: list[SpatialEncoderAttentionMaps] = []
        for encoder in self.context_encoders:
            token_id_counts: dict[int, int] = {}
            target_token_ids: list[tuple[int, ...]] = []
            for target in encoder.targets:
                if max(target.adapter_context_indices, default=-1) >= len(encoder.token_ids):
                    raise ValueError("Asteria spatial target token binding is outside token ids")
                token_ids = tuple(
                    int(encoder.token_ids[index])
                    for index in target.adapter_context_indices
                )
                target_token_ids.append(token_ids)
                for token_id in set(token_ids):
                    token_id_counts[token_id] = token_id_counts.get(token_id, 0) + 1
            shared_token_ids = {
                token_id for token_id, count in token_id_counts.items() if count > 1
            }
            target_maps: list[torch.Tensor] = []
            target_presence_maps: list[torch.Tensor] = []
            for target_index, target in enumerate(encoder.targets):
                indices = tuple(int(value) for value in target.adapter_context_indices)
                if not indices or min(indices) < 0 or max(indices) >= key_tokens:
                    raise ValueError("Asteria spatial target token binding is outside attention context")
                weights = mean_attention.new_tensor(target.token_weights)
                if weights.numel() != len(indices) or not bool(torch.isfinite(weights).all()):
                    raise ValueError("Asteria spatial target token weights are invalid")
                weights = weights.abs()
                if float(weights.detach().sum()) <= 0.0:
                    raise ValueError("Asteria spatial target token weights are empty")
                presence_vector = (
                    mean_attention[:, list(indices)] * weights.unsqueeze(0)
                ).sum(dim=-1) / weights.sum()
                discriminative_weights = weights.clone()
                unique_mask = discriminative_weights.new_tensor(
                    [
                        token_id not in shared_token_ids
                        for token_id in target_token_ids[target_index]
                    ],
                    dtype=torch.bool,
                )
                if bool(unique_mask.any()):
                    discriminative_weights = torch.where(
                        unique_mask,
                        discriminative_weights,
                        discriminative_weights * 0.1,
                    )
                identity_vector = (
                    mean_attention[:, list(indices)]
                    * discriminative_weights.unsqueeze(0)
                ).sum(dim=-1) / discriminative_weights.sum().clamp_min(1e-8)
                target_maps.append(identity_vector.reshape(grid_height, grid_width))
                target_presence_maps.append(
                    presence_vector.reshape(grid_height, grid_width)
                )
            global_indices = self._global_indices(encoder, key_tokens)
            global_map = mean_attention[:, list(global_indices)].mean(dim=-1).reshape(
                grid_height, grid_width
            )
            encoder_maps.append(
                SpatialEncoderAttentionMaps(
                    encoder_id=encoder.encoder_id,
                    target_attention=_resize_map(
                        torch.stack(target_maps), source_height, source_width
                    ),
                    target_presence_attention=_resize_map(
                        torch.stack(target_presence_maps), source_height, source_width
                    ),
                    global_attention=_resize_map(
                        global_map, source_height, source_width
                    ),
                    global_context_indices=global_indices,
                )
            )
        self._local.cross_records.append(_CrossRecord(encoders=tuple(encoder_maps)))

    def _self_record(self, probabilities: torch.Tensor, options: dict[str, Any]) -> None:
        positive = _positive_rows(options, int(probabilities.shape[0]), probabilities.device)
        if not bool(positive.any()):
            raise ValueError("Asteria spatial positive attention row is missing")
        relation = probabilities[positive].mean(dim=(0, 1))
        query_tokens, key_tokens = int(relation.shape[0]), int(relation.shape[1])
        if query_tokens != key_tokens:
            raise ValueError("Asteria spatial self attention shape is invalid")
        source_height, source_width = self.layout_fields.shape[-2:]
        grid_height, grid_width = _factor_grid(query_tokens, source_height, source_width)
        self._local.self_records.append(
            _SelfRecord(
                relation=relation,
                grid_height=grid_height,
                grid_width=grid_width,
            )
        )

    def _base_attention(self, original, *args, **kwargs):
        if self.prior_attention_override is not None:
            return self.prior_attention_override(original, *args, **kwargs)
        return original(*args, **kwargs)

    def _cross_prior_bias(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        options: dict[str, Any],
        *,
        heads: int,
        attention_kwargs: dict[str, Any],
        modulation_scale: float | None = None,
    ) -> torch.Tensor:
        batch = int(q.shape[0])
        query_tokens = int(q.shape[1] if q.ndim == 3 else q.shape[-2])
        key_tokens = int(k.shape[1] if k.ndim == 3 else k.shape[-2])
        source_height, source_width = self.layout_fields.shape[-2:]
        grid_height, grid_width = _factor_grid(
            query_tokens,
            source_height,
            source_width,
        )
        fields = _resize_map(
            self.layout_fields.to(device=q.device, dtype=torch.float32),
            grid_height,
            grid_width,
        ).reshape(len(self.target_ids), query_tokens)
        query, key = _reshape_heads(
            q,
            k,
            heads,
            enable_gqa=bool(attention_kwargs.get("enable_gqa", False)),
        )
        attention_scale = float(
            attention_kwargs.get("scale", query.shape[-1] ** -0.5)
        )
        if not math.isfinite(attention_scale) or attention_scale <= 0.0:
            raise ValueError("Asteria spatial attention scale is invalid")
        logits = torch.matmul(
            query.float(), key.float().transpose(-2, -1)
        ) * attention_scale
        score_min = logits.amin(dim=-1)
        score_max = logits.amax(dim=-1)
        bias = torch.zeros_like(logits)
        positive_rows = _positive_rows(options, batch, q.device)
        strength = (
            self.attention_prior_scale
            if modulation_scale is None
            else float(modulation_scale)
        )
        for target_index in range(len(self.target_ids)):
            indices = sorted(
                {
                    int(index)
                    for encoder in self.context_encoders
                    for index in encoder.targets[target_index].adapter_context_indices
                }
            )
            if not indices or min(indices) < 0 or max(indices) >= key_tokens:
                raise ValueError("Asteria spatial target token binding is outside attention context")
            field = fields[target_index].clamp(0.0, 1.0)
            area_adjustment = 1.0 - field.mean().clamp(0.0, 1.0)
            for index in indices:
                token_score = logits[..., index]
                inside = field.view(1, 1, -1) * (score_max - token_score)
                outside = (1.0 - field).view(1, 1, -1) * (
                    token_score - score_min
                )
                bias[positive_rows, :, :, index] += (
                    area_adjustment * strength * (inside - outside)
                )[positive_rows]
        return bias.to(dtype=q.dtype)

    def _self_prior_bias(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        options: dict[str, Any],
        *,
        heads: int,
        attention_kwargs: dict[str, Any],
        modulation_scale: float,
    ) -> torch.Tensor:
        batch = int(q.shape[0])
        query_tokens = int(q.shape[1] if q.ndim == 3 else q.shape[-2])
        key_tokens = int(k.shape[1] if k.ndim == 3 else k.shape[-2])
        if query_tokens != key_tokens:
            raise ValueError("Asteria spatial self attention shape is invalid")
        source_height, source_width = self.layout_fields.shape[-2:]
        grid_height, grid_width = _factor_grid(
            query_tokens,
            source_height,
            source_width,
        )
        fields = _resize_map(
            self.layout_fields.to(device=q.device, dtype=torch.float32),
            grid_height,
            grid_width,
        ).reshape(len(self.target_ids), query_tokens).clamp(0.0, 1.0)
        query_gate = fields.amax(dim=0).view(1, 1, query_tokens, 1)
        query, key = _reshape_heads(
            q,
            k,
            heads,
            enable_gqa=bool(attention_kwargs.get("enable_gqa", False)),
        )
        attention_scale = float(
            attention_kwargs.get("scale", query.shape[-1] ** -0.5)
        )
        if not math.isfinite(attention_scale) or attention_scale <= 0.0:
            raise ValueError("Asteria spatial attention scale is invalid")
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
        if not torch.is_grad_enabled():
            logits.mul_(attention_scale)
        else:
            logits = logits * attention_scale
        score_min = logits.amin(dim=-1, keepdim=True)
        score_max = logits.amax(dim=-1, keepdim=True)
        self_strength = float(modulation_scale) * 0.3
        if not torch.is_grad_enabled():
            # The original expression simplifies exactly to
            # min - logits + relation * (max - min). Reuse logits storage and
            # build relation rows in bounded chunks for high-resolution stages.
            score_range = score_max - score_min
            bias = logits.neg_().add_(score_min)
            chunk_rows = _SELF_PRIOR_RELATION_CHUNK_ROWS
            for start in range(0, query_tokens, chunk_rows):
                end = min(query_tokens, start + chunk_rows)
                relation_chunk = torch.matmul(
                    fields[:, start:end].transpose(0, 1),
                    fields,
                ).clamp_(0.0, 1.0)
                bias[..., start:end, :].addcmul_(
                    relation_chunk.view(1, 1, end - start, key_tokens),
                    score_range[..., start:end, :],
                )
            bias.mul_(query_gate).mul_(self_strength)
        else:
            relation = torch.einsum("tq,tk->qk", fields, fields).clamp(0.0, 1.0)
            relation = relation.view(1, 1, query_tokens, key_tokens)
            bias = query_gate * self_strength * (
                relation * (score_max - logits)
                - (1.0 - relation) * (logits - score_min)
            )
        positive_rows = _positive_rows(options, batch, q.device)
        bias[~positive_rows] = 0
        return bias.to(dtype=q.dtype)

    def attention_override(self, original, q, k, v, heads, mask=None, **kwargs):
        call = getattr(self._local, "call", None)
        self._local.call = None
        capture_active = bool(getattr(self._local, "active", False))
        sampler_bias_active = bool(
            getattr(self._local, "sampler_bias_active", False)
        )
        if (capture_active or sampler_bias_active) and isinstance(
            call, _PendingAttentionCall
        ):
            selectors = (
                ("all",)
                if sampler_bias_active and not capture_active
                else (
                    self.attention_layers
                    if call.kind == "cross"
                    else self.self_relation_layers
                )
            )
            if not _matches_layer_selector(call.options, selectors):
                return self._base_attention(
                    original,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    **kwargs,
                )
            if (
                call.kind == "cross"
                and sampler_bias_active
                and not capture_active
                and self.attention_prior_scale > 0.0
            ):
                prior_scale = float(getattr(self._local, "sampler_bias_scale", 0.0))
                mask = _merge_attention_mask(
                    mask,
                    self._cross_prior_bias(
                        q,
                        k,
                        call.options,
                        heads=heads,
                        attention_kwargs=kwargs,
                        modulation_scale=prior_scale,
                    ),
                    q,
                )
                self._local.sampler_prior_bias_count += 1
            elif (
                call.kind == "self"
                and sampler_bias_active
                and self.attention_prior_scale > 0.0
            ):
                mask = _merge_attention_mask(
                    mask,
                    self._self_prior_bias(
                        q,
                        k,
                        call.options,
                        heads=heads,
                        attention_kwargs=kwargs,
                        modulation_scale=float(
                            getattr(self._local, "sampler_bias_scale", 0.0)
                        ),
                    ),
                    q,
                )
                self._local.sampler_prior_bias_count += 1
            if not capture_active:
                return self._base_attention(
                    original,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    **kwargs,
                )
            source_height, source_width = self.layout_fields.shape[-2:]
            probabilities = _attention_probabilities(
                q,
                k,
                heads,
                mask,
                kwargs,
                source_height=source_height,
                source_width=source_width,
                max_query_tokens=(
                    self.max_cross_attention_tokens
                    if call.kind == "cross"
                    else self.max_self_attention_tokens
                ),
                pool_keys=call.kind == "self",
            )
            if call.kind == "cross":
                self._cross_record(probabilities, call.options)
            elif call.kind == "self":
                self._self_record(probabilities, call.options)
            else:
                raise ValueError("Asteria spatial attention capture kind is invalid")
        return self._base_attention(
            original,
            q,
            k,
            v,
            heads,
            mask=mask,
            **kwargs,
        )
