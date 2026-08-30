from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Callable

import torch
import torch.nn.functional as functional

from .profile_runtime import SpatialPhaseProfile


_BLOCKED_ATTENTION = -10_000.0


@dataclass(frozen=True, slots=True)
class _RouteTarget:
    target_id: str
    mask: torch.Tensor
    conditioning: tuple[tuple[torch.Tensor, dict[str, Any]], ...]
    strength: float
    blending: float


@dataclass(frozen=True, slots=True)
class _TargetTokenGroup:
    target_index: int
    start: int
    end: int
    route_scale: float


@dataclass(frozen=True, slots=True)
class _AttentionCall:
    kind: str
    options: dict[str, Any]
    progress: float
    base_key_tokens: int
    groups: tuple[_TargetTokenGroup, ...] = ()


def _smoothstep(edge_start: float, edge_end: float, value: float) -> float:
    if edge_end <= edge_start:
        return 1.0 if value >= edge_end else 0.0
    position = min(1.0, max(0.0, (value - edge_start) / (edge_end - edge_start)))
    return position * position * (3.0 - 2.0 * position)


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
            candidates.append((abs(math.log(max(ratio, 1e-9) / source_ratio)), candidate_height, candidate_width))
    if not candidates:
        return 1, tokens
    _, height, width = min(candidates, key=lambda item: (item[0], abs(item[1] - item[2])))
    return height, width


def _attention_shape(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.ndim == 3:
        return int(tensor.shape[0]), int(tensor.shape[1])
    if tensor.ndim == 4:
        return int(tensor.shape[0]), int(tensor.shape[-2])
    raise ValueError("Asteria spatial attention tensor shape is unsupported")


def _key_tokens(tensor: torch.Tensor) -> int:
    if tensor.ndim == 3:
        return int(tensor.shape[1])
    if tensor.ndim == 4:
        return int(tensor.shape[-2])
    raise ValueError("Asteria spatial attention tensor shape is unsupported")


def _branch_rows(options: dict[str, Any], batch: int, *, positive_only: bool) -> torch.Tensor:
    branches = options.get("cond_or_uncond")
    if not isinstance(branches, (list, tuple)) or not branches or batch % len(branches):
        return torch.ones(batch, dtype=torch.bool)
    rows_per_branch = batch // len(branches)
    values = [
        (not positive_only) or branch == 0
        for branch in branches
        for _ in range(rows_per_branch)
    ]
    return torch.tensor(values, dtype=torch.bool)


def _expand_context(tensor: torch.Tensor, batch: int, reference: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("Asteria spatial target conditioning shape is unsupported")
    converted = tensor.to(device=reference.device, dtype=reference.dtype)
    if converted.shape[0] == batch:
        return converted
    if converted.shape[0] == 1:
        return converted.expand(batch, -1, -1)
    if batch % converted.shape[0] == 0:
        return converted.repeat(batch // converted.shape[0], 1, 1)
    raise ValueError("Asteria spatial target conditioning batch is incompatible")


def _merge_attention_mask(existing: Any, routing: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if existing is None:
        return routing
    if not isinstance(existing, torch.Tensor):
        raise ValueError("Asteria spatial attention mask type is unsupported")
    current = existing.to(device=reference.device)
    if current.dtype == torch.bool:
        current = torch.where(
            current,
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.full((), _BLOCKED_ATTENTION, device=reference.device, dtype=reference.dtype),
        )
    else:
        current = current.to(dtype=reference.dtype)
    if current.ndim == 4 and routing.ndim == 3:
        routing = routing.unsqueeze(1)
    return current + routing


def _slice_key_tokens(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, start:end, :]
    if tensor.ndim == 4:
        return tensor[:, :, start:end, :]
    raise ValueError("Asteria spatial attention tensor shape is unsupported")


def _slice_key_mask(
    mask: Any,
    start: int,
    end: int,
    *,
    total_tokens: int,
) -> Any:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Asteria spatial attention mask type is unsupported")
    if int(mask.shape[-1]) == total_tokens:
        return mask[..., start:end]
    if start == 0 and int(mask.shape[-1]) == end:
        return mask
    return None


def _query_weight(output: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3:
        return vector.view(1, -1, 1)
    if output.ndim == 4:
        return vector.view(1, 1, -1, 1)
    raise ValueError("Asteria spatial attention output shape is unsupported")


class SpatialAttentionRouter:
    """Training-free layout routing for SDXL-family and Anima attention stacks."""

    def __init__(
        self,
        conditions,
        masks,
        profile: SpatialPhaseProfile,
        *,
        prior_attention_override: Callable[..., Any] | None = None,
        anima_text_preprocessor: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.profile = profile
        self.prior_attention_override = prior_attention_override
        self.anima_text_preprocessor = anima_text_preprocessor
        self.targets = tuple(
            _RouteTarget(
                target_id=condition_target.target_id,
                mask=mask_target.mask.detach().to(device="cpu", dtype=torch.float32).contiguous().clone(),
                conditioning=condition_target.conditioning,
                strength=float(mask_target.strength) / 100.0,
                blending=float(mask_target.blending) / 100.0,
            )
            for condition_target, mask_target in zip(
                conditions.targets,
                masks.targets,
                strict=True,
            )
        )
        self._local = threading.local()
        self._progress_cache: tuple[int, int, float, float] | None = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "cross_patch_calls": 0,
            "self_patch_calls": 0,
            "cross_attention_calls": 0,
            "self_attention_calls": 0,
            "cross_bias_applied": 0,
            "cross_output_blends": 0,
            "self_bias_applied": 0,
            "anima_context_preparations": 0,
            "cross_blocks": set(),
            "self_blocks": set(),
            "query_token_counts": set(),
            "target_routes": {target.target_id: 0 for target in self.targets},
            "skips": {
                "inactive_target_conditioning": 0,
                "self_attention_token_limit": 0,
                "self_attention_schedule": 0,
                "missing_patch_context": 0,
            },
        }

    @staticmethod
    def _block_key(options: dict[str, Any]) -> str:
        block = options.get("block")
        if isinstance(block, (list, tuple)):
            block_parts = [str(value) for value in block]
        elif block is None:
            block_parts = ["unknown"]
        else:
            block_parts = [str(block)]
        block_parts.append(str(options.get("block_index", "unknown")))
        return ":".join(block_parts)

    def _increment(self, field: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[field] += amount

    def _skip(self, reason: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats["skips"][reason] += amount

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "cross_patch_calls": self._stats["cross_patch_calls"],
                "self_patch_calls": self._stats["self_patch_calls"],
                "cross_attention_calls": self._stats["cross_attention_calls"],
                "self_attention_calls": self._stats["self_attention_calls"],
                "cross_bias_applied": self._stats["cross_bias_applied"],
                "cross_output_blends": self._stats["cross_output_blends"],
                "self_bias_applied": self._stats["self_bias_applied"],
                "anima_context_preparations": self._stats["anima_context_preparations"],
                "cross_blocks": sorted(self._stats["cross_blocks"]),
                "self_blocks": sorted(self._stats["self_blocks"]),
                "query_token_counts": sorted(self._stats["query_token_counts"]),
                "target_routes": dict(self._stats["target_routes"]),
                "skips": dict(self._stats["skips"]),
            }

    def prepare_anima_contexts(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.anima_text_preprocessor is None:
            raise ValueError("Asteria Anima spatial text preprocessor is unavailable")
        if not isinstance(payload, dict) or not isinstance(payload.get("img"), torch.Tensor):
            raise ValueError("Asteria Anima spatial post-input payload is invalid")
        reference = payload["img"]
        prepared: list[tuple[tuple[torch.Tensor, dict[str, Any]], ...]] = []
        preparation_count = 0
        for target in self.targets:
            entries: list[tuple[torch.Tensor, dict[str, Any]]] = []
            for tensor, metadata in target.conditioning:
                text_ids = metadata.get("t5xxl_ids")
                if not isinstance(text_ids, torch.Tensor):
                    raise ValueError("Asteria Anima spatial conditioning requires t5xxl_ids")
                if text_ids.ndim == 1:
                    text_ids = text_ids.unsqueeze(0)
                elif text_ids.ndim != 2:
                    raise ValueError("Asteria Anima spatial t5xxl_ids shape is invalid")
                weights = metadata.get("t5xxl_weights")
                if weights is not None:
                    if not isinstance(weights, torch.Tensor):
                        raise ValueError("Asteria Anima spatial t5xxl_weights are invalid")
                    if weights.ndim == 1:
                        weights = weights.unsqueeze(0).unsqueeze(-1)
                    elif weights.ndim == 2:
                        weights = weights.unsqueeze(-1)
                    elif weights.ndim != 3:
                        raise ValueError("Asteria Anima spatial t5xxl_weights shape is invalid")
                    weights = weights.to(device=reference.device, dtype=reference.dtype)
                converted = tensor.to(device=reference.device, dtype=reference.dtype)
                processed = self.anima_text_preprocessor(
                    converted,
                    text_ids.to(device=reference.device),
                    t5xxl_weights=weights,
                )
                if not isinstance(processed, torch.Tensor) or processed.ndim != 3:
                    raise ValueError("Asteria Anima spatial text preprocessing result is invalid")
                entries.append((processed, metadata))
                preparation_count += 1
            prepared.append(tuple(entries))
        self._local.prepared_anima_contexts = tuple(prepared)
        self._increment("anima_context_preparations", preparation_count)
        return payload

    def _progress(self, options: dict[str, Any]) -> float:
        sigmas = options.get("sigmas")
        sample_sigmas = options.get("sample_sigmas")
        if not isinstance(sigmas, torch.Tensor) or not isinstance(sample_sigmas, torch.Tensor):
            return 0.5
        if sigmas.numel() < 1 or sample_sigmas.numel() < 2:
            return 0.5
        current = sigmas.reshape(-1)[0].detach().to(
            device=sample_sigmas.device,
            dtype=sample_sigmas.dtype,
        )
        current_value = float(current.item())
        cache_key = (id(sample_sigmas), int(sample_sigmas.numel()), current_value)
        if self._progress_cache is not None and self._progress_cache[:3] == cache_key:
            return self._progress_cache[3]
        index = int(torch.argmin(torch.abs(sample_sigmas.reshape(-1) - current)).item())
        progress = index / max(int(sample_sigmas.numel()) - 1, 1)
        self._progress_cache = (*cache_key, progress)
        return progress

    def _isolation_scale(self, progress: float) -> float:
        smoothing = self.profile.schedule_smoothing
        rising = _smoothstep(
            self.profile.isolation_start - smoothing,
            self.profile.isolation_start + smoothing,
            progress,
        )
        falling = 1.0 - _smoothstep(
            self.profile.isolation_end - smoothing,
            self.profile.isolation_end + smoothing,
            progress,
        )
        # Layout and subject count are established during the first denoising
        # steps.  Keep isolation fully active before the profiled window, then
        # release it smoothly near ``isolation_end``.  Multiplying the two
        # ramps would disable isolation exactly when the composition is formed.
        return 1.0 - rising * (1.0 - falling)

    def _route_scale(self, target: _RouteTarget, progress: float) -> float:
        early = 1.0 + 0.25 * (
            1.0
            - _smoothstep(
                0.0,
                self.profile.early_guidance_end,
                progress,
            )
        )
        late = _smoothstep(
            self.profile.late_blending_start,
            1.0,
            progress,
        )
        late_scale = 1.0 - late * (1.0 - target.blending)
        return target.strength * early * late_scale

    @staticmethod
    def _entry_is_active(metadata: dict[str, Any], progress: float) -> bool:
        start = metadata.get("start_percent", 0.0)
        end = metadata.get("end_percent", 1.0)
        try:
            return float(start) <= progress <= float(end)
        except (TypeError, ValueError):
            return True

    def _target_context(
        self,
        target: _RouteTarget,
        target_index: int,
        *,
        progress: float,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        conditioning = target.conditioning
        if self.anima_text_preprocessor is not None:
            prepared = getattr(self._local, "prepared_anima_contexts", None)
            if not isinstance(prepared, tuple) or len(prepared) != len(self.targets):
                raise ValueError("Asteria Anima spatial conditioning was not preprocessed")
            conditioning = prepared[target_index]
        entries = [
            tensor
            for tensor, metadata in conditioning
            if self._entry_is_active(metadata, progress)
        ]
        if not entries:
            return None
        converted = [_expand_context(tensor, batch, reference) for tensor in entries]
        if any(tensor.shape[-1] != reference.shape[-1] for tensor in converted):
            raise ValueError("Asteria spatial target conditioning width is incompatible")
        return torch.cat(converted, dim=1)

    @staticmethod
    def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        anima_protocol = "extra_options" in kwargs
        if anima_protocol:
            options = kwargs.get("extra_options")
        elif args and isinstance(args[0], dict):
            options = args[0]
        else:
            options = {}
        return dict(options) if isinstance(options, dict) else {}, anima_protocol

    def cross_patch(self, q, k, v, *args, **kwargs):
        options, anima_protocol = self._options(args, kwargs)
        self._increment("cross_patch_calls")
        progress = self._progress(options)
        batch = int(k.shape[0])
        base_key_tokens = int(k.shape[1])
        key_parts = [k]
        value_parts = [v]
        groups: list[_TargetTokenGroup] = []
        cursor = base_key_tokens
        for target_index, target in enumerate(self.targets):
            route_scale = self._route_scale(target, progress)
            if route_scale <= 0.0:
                continue
            context = self._target_context(
                target,
                target_index,
                progress=progress,
                batch=batch,
                reference=k,
            )
            if context is None:
                self._skip("inactive_target_conditioning")
                continue
            key_parts.append(context)
            value_parts.append(context.to(device=v.device, dtype=v.dtype))
            groups.append(
                _TargetTokenGroup(
                    target_index=target_index,
                    start=cursor,
                    end=cursor + int(context.shape[1]),
                    route_scale=route_scale,
                )
            )
            cursor += int(context.shape[1])
        routed_k = torch.cat(key_parts, dim=1)
        routed_v = torch.cat(value_parts, dim=1)
        self._local.call = _AttentionCall(
            kind="cross",
            options=options,
            progress=progress,
            base_key_tokens=base_key_tokens,
            groups=tuple(groups),
        )
        if anima_protocol:
            return {"q": q, "k": routed_k, "v": routed_v}
        return q, routed_k, routed_v

    def self_patch(self, q, k, v, *args, **kwargs):
        options, anima_protocol = self._options(args, kwargs)
        self._increment("self_patch_calls")
        self._local.call = _AttentionCall(
            kind="self",
            options=options,
            progress=self._progress(options),
            base_key_tokens=int(k.shape[1]),
        )
        if anima_protocol:
            return {"q": q, "k": k, "v": v}
        return q, k, v

    def _mask_vectors(
        self,
        query_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        source = self.targets[0].mask
        source_height, source_width = int(source.shape[-2]), int(source.shape[-1])
        height, width = _factor_grid(query_tokens, source_height, source_width)
        vectors: list[torch.Tensor] = []
        for target in self.targets:
            mask = target.mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask[:1].unsqueeze(0).to(device=device, dtype=torch.float32)
            resized = functional.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
            vectors.append(resized.reshape(-1).to(dtype=dtype).clamp_(0.0, 1.0))
        return tuple(vectors)

    def _cross_bias(
        self,
        call: _AttentionCall,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_tokens = _attention_shape(q)
        key_tokens = _key_tokens(k)
        bias = torch.zeros((batch, query_tokens, key_tokens), device=q.device, dtype=q.dtype)
        active_rows = _branch_rows(call.options, batch, positive_only=True).to(device=q.device)
        vectors = self._mask_vectors(query_tokens, device=q.device, dtype=q.dtype)
        isolation = self._isolation_scale(call.progress)
        if call.base_key_tokens > 0 and self.profile.global_inside_penalty > 0.0:
            union = torch.stack(vectors).amax(dim=0)
            global_penalty = -self.profile.global_inside_penalty * isolation * union
            bias[active_rows, :, : call.base_key_tokens] += global_penalty[None, :, None]
        for group in call.groups:
            mask = vectors[group.target_index]
            target_bias = group.route_scale * (
                self.profile.cross_inside_bias * mask
                - self.profile.cross_outside_penalty * isolation * (1.0 - mask)
            )
            bias[active_rows, :, group.start : group.end] += target_bias[None, :, None]

        if call.groups:
            inactive_rows = ~active_rows
            if bool(inactive_rows.any()):
                first = min(group.start for group in call.groups)
                if bias.ndim == 3:
                    bias[inactive_rows, :, first:] = _BLOCKED_ATTENTION
                else:
                    bias[inactive_rows, :, :, first:] = _BLOCKED_ATTENTION
        return bias

    def _cross_output_blend(self, original, call: _AttentionCall, q, k, v, heads, mask, **kwargs):
        total_tokens = _key_tokens(k)
        global_k = _slice_key_tokens(k, 0, call.base_key_tokens)
        global_v = _slice_key_tokens(v, 0, call.base_key_tokens)
        global_output = self._base_attention(
            original,
            q,
            global_k,
            global_v,
            heads,
            mask=_slice_key_mask(
                mask,
                0,
                call.base_key_tokens,
                total_tokens=total_tokens,
            ),
            **kwargs,
        )
        if not call.groups:
            return global_output

        query_tokens = int(global_output.shape[1] if global_output.ndim == 3 else global_output.shape[-2])
        vectors = self._mask_vectors(
            query_tokens,
            device=global_output.device,
            dtype=global_output.dtype,
        )
        raw_weights = torch.stack(
            [
                vectors[group.target_index] * max(0.0, float(group.route_scale))
                for group in call.groups
            ],
            dim=0,
        )
        total_weight = raw_weights.sum(dim=0)
        normalized_weights = raw_weights / total_weight.clamp_min(1.0).unsqueeze(0)
        covered_weight = total_weight.clamp(0.0, 1.0)
        blended = global_output * _query_weight(global_output, 1.0 - covered_weight)

        for group, weight in zip(call.groups, normalized_weights, strict=True):
            target_output = self._base_attention(
                original,
                q,
                _slice_key_tokens(k, group.start, group.end),
                _slice_key_tokens(v, group.start, group.end),
                heads,
                mask=_slice_key_mask(
                    mask,
                    group.start,
                    group.end,
                    total_tokens=total_tokens,
                ),
                **kwargs,
            )
            if target_output.shape != global_output.shape:
                raise ValueError("Asteria spatial target attention output shape is incompatible")
            blended = blended + target_output * _query_weight(target_output, weight)

        active_rows = _branch_rows(
            call.options,
            int(global_output.shape[0]),
            positive_only=True,
        ).to(device=global_output.device)
        active_shape = [int(global_output.shape[0])] + [1] * (global_output.ndim - 1)
        return torch.where(active_rows.view(*active_shape), blended, global_output)

    def _self_bias(self, call: _AttentionCall, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor | None:
        batch, query_tokens = _attention_shape(q)
        if query_tokens != _key_tokens(k) or query_tokens > self.profile.self_attention_token_limit:
            self._skip("self_attention_token_limit")
            return None
        isolation = self._isolation_scale(call.progress)
        if isolation <= 0.0 or self.profile.self_leakage_penalty <= 0.0:
            self._skip("self_attention_schedule")
            return None
        vectors = self._mask_vectors(query_tokens, device=q.device, dtype=q.dtype)
        early_separation = 1.0 - _smoothstep(
            self.profile.early_guidance_end,
            self.profile.isolation_end,
            call.progress,
        )
        separation_penalty = self.profile.self_leakage_penalty + (
            max(self.profile.cross_outside_penalty, self.profile.self_leakage_penalty)
            * early_separation
        )
        single = torch.zeros((query_tokens, query_tokens), device=q.device, dtype=q.dtype)
        for left_index, left in enumerate(vectors):
            for right_index, right in enumerate(vectors):
                if left_index == right_index:
                    continue
                single -= (
                    separation_penalty
                    * isolation
                    * torch.outer(left, right)
                )
        return single.unsqueeze(0).expand(batch, -1, -1)

    def _base_attention(self, original, *args, **kwargs):
        if self.prior_attention_override is not None:
            return self.prior_attention_override(original, *args, **kwargs)
        return original(*args, **kwargs)

    def attention_override(self, original, q, k, v, heads, mask=None, **kwargs):
        call = getattr(self._local, "call", None)
        self._local.call = None
        routing = None
        if isinstance(call, _AttentionCall):
            batch, query_tokens = _attention_shape(q)
            del batch
            with self._stats_lock:
                self._stats["query_token_counts"].add(query_tokens)
            if call.kind == "cross":
                with self._stats_lock:
                    self._stats["cross_attention_calls"] += 1
                    self._stats["cross_blocks"].add(self._block_key(call.options))
                    for group in call.groups:
                        target_id = self.targets[group.target_index].target_id
                        self._stats["target_routes"][target_id] += 1
                output = self._cross_output_blend(
                    original,
                    call,
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    **kwargs,
                )
                # ``cross_bias_applied`` is kept in the V2 receipt for
                # compatibility with already generated development history.
                # New consumers use ``cross_output_blends`` to distinguish the
                # corrected global/local output routing from the old single
                # Softmax bias implementation.
                self._increment("cross_bias_applied")
                self._increment("cross_output_blends")
                return output
            elif call.kind == "self":
                with self._stats_lock:
                    self._stats["self_attention_calls"] += 1
                    self._stats["self_blocks"].add(self._block_key(call.options))
                routing = self._self_bias(call, q, k)
                if routing is not None:
                    self._increment("self_bias_applied")
        else:
            self._skip("missing_patch_context")
        if routing is not None:
            mask = _merge_attention_mask(mask, routing, q)
        return self._base_attention(
            original,
            q,
            k,
            v,
            heads,
            mask=mask,
            **kwargs,
        )
