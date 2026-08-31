from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .spatial_contract import (
    SpatialEncoderTokenBinding,
    SpatialTargetTokenBinding,
    SpatialTokenBindingBundle,
    SpatialScheduledTokenBindingBundle,
    SpatialTokenChunkRange,
    canonical_json_sha256,
    validate_target_id,
)


class SpatialTokenBindingError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class PreparedSpatialTokens:
    tokens: dict[str, Any]
    binding: SpatialTokenBindingBundle


def _tokenize(clip: Any, text: str) -> dict[str, Any]:
    tokenize = getattr(clip, "tokenize", None)
    if callable(tokenize):
        payload = tokenize(text, return_word_ids=True)
    else:
        tokenize_with_weights = getattr(clip, "tokenize_with_weights", None)
        if not callable(tokenize_with_weights):
            raise SpatialTokenBindingError("spatial_token_binding_tokenizer_unavailable")
        payload = tokenize_with_weights(text, return_word_ids=True)
    if not isinstance(payload, dict) or not payload:
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    return payload


def _tokenizer_root(clip: Any) -> Any:
    return getattr(clip, "tokenizer", clip)


def _encoder_tokenizer(clip: Any, encoder_id: str) -> Any:
    metadata = getattr(clip, "asteria_tokenizer_metadata", None)
    if isinstance(metadata, Mapping) and encoder_id in metadata:
        return metadata[encoder_id]
    root = _tokenizer_root(clip)
    aliases = (encoder_id, f"clip_{encoder_id}")
    for alias in aliases:
        candidate = getattr(root, alias, None)
        if candidate is not None:
            return candidate
    return root


def _metadata_value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _word_zero_is_special(tokenizer: Any) -> bool:
    explicit = _metadata_value(tokenizer, "word_zero_is_special")
    if explicit is not None:
        return bool(explicit)
    return any(
        _metadata_value(tokenizer, name) is not None
        for name in ("start_token", "end_token")
    )


def _vocabulary_fingerprint(tokenizer: Any, encoder_id: str) -> str:
    vocabulary: Mapping[Any, Any] | None = None
    direct = _metadata_value(tokenizer, "vocabulary")
    if isinstance(direct, Mapping):
        vocabulary = direct
    else:
        inner = _metadata_value(tokenizer, "tokenizer")
        get_vocab = getattr(inner, "get_vocab", None)
        if callable(get_vocab):
            candidate = get_vocab()
            if isinstance(candidate, Mapping):
                vocabulary = candidate
        if vocabulary is None:
            get_vocab = getattr(tokenizer, "get_vocab", None)
            if callable(get_vocab):
                candidate = get_vocab()
                if isinstance(candidate, Mapping):
                    vocabulary = candidate
    digest = sha256()
    digest.update(
        f"{encoder_id}|{type(tokenizer).__module__}.{type(tokenizer).__qualname__}|".encode(
            "utf-8"
        )
    )
    if vocabulary is not None:
        for token, token_id in sorted(
            vocabulary.items(), key=lambda item: (str(item[0]), str(item[1]))
        ):
            digest.update(str(token).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(token_id).encode("ascii", errors="backslashreplace"))
            digest.update(b"\n")
    return digest.hexdigest()


def _integer_token(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialTokenBindingError("spatial_token_binding_unsupported")
    return value


def _flatten_encoder(
    chunks: Any,
    *,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    word_zero_is_special = _word_zero_is_special(tokenizer)
    start_token = _metadata_value(tokenizer, "start_token")
    end_token = _metadata_value(tokenizer, "end_token")
    pad_token = _metadata_value(tokenizer, "pad_token")
    if pad_token is None:
        pad_token = _metadata_value(tokenizer, "pad_token_id")
    flattened: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    context_index = 0
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, (list, tuple)) or not chunk:
            raise SpatialTokenBindingError("spatial_token_binding_unresolved")
        seen_end = False
        for token_index, item in enumerate(chunk):
            if not isinstance(item, (list, tuple)) or len(item) not in {2, 3}:
                raise SpatialTokenBindingError("spatial_token_binding_unresolved")
            token_id = _integer_token(item[0])
            try:
                weight = float(item[1])
            except (TypeError, ValueError) as error:
                raise SpatialTokenBindingError("spatial_token_binding_unresolved") from error
            if not math.isfinite(weight):
                raise SpatialTokenBindingError("spatial_token_binding_unresolved")
            word_id = item[2] if len(item) == 3 else None
            special = word_zero_is_special and word_id == 0
            is_start = (
                special
                and token_index == 0
                and start_token is not None
                and token_id == int(start_token)
            )
            is_end = (
                special
                and not is_start
                and not seen_end
                and end_token is not None
                and token_id == int(end_token)
            )
            if is_end:
                seen_end = True
            padding = (
                special
                and not is_start
                and not is_end
                and pad_token is not None
                and token_id == int(pad_token)
            )
            record = {
                "token_id": token_id,
                "weight": weight,
                "word_id": word_id,
                "chunk_index": chunk_index,
                "token_index": token_index,
                "context_index": context_index,
                "special": special,
                "padding": padding,
            }
            flattened.append(record)
            if not special:
                semantic.append(record)
            context_index += 1
    if not semantic:
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    return flattened, semantic


def _encoder_payload(payload: Mapping[str, Any], encoder_id: str) -> Any:
    if encoder_id not in payload:
        raise SpatialTokenBindingError("spatial_token_binding_encoder_missing")
    return payload[encoder_id]


def _semantic_for_text(clip: Any, text: str, encoder_id: str, tokenizer: Any) -> list[dict[str, Any]]:
    payload = _tokenize(clip, text)
    _, semantic = _flatten_encoder(_encoder_payload(payload, encoder_id), tokenizer=tokenizer)
    return semantic


def _prefix_length(full: Sequence[dict[str, Any]], prefix: Sequence[dict[str, Any]]) -> int:
    if len(prefix) > len(full):
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    for index, item in enumerate(prefix):
        if item["token_id"] != full[index]["token_id"]:
            raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    return len(prefix)


def _truncation_start(
    clip: Any,
    full_text: str,
    encoder_id: str,
    tokenizer: Any,
    full_semantic: Sequence[dict[str, Any]],
) -> int | None:
    probe = _semantic_for_text(
        clip,
        f"{full_text} __ASTERIA_TOKEN_BOUNDARY_PROBE_7F91D3__",
        encoder_id,
        tokenizer,
    )
    if len(probe) > len(full_semantic):
        return None
    if [item["token_id"] for item in probe] != [
        item["token_id"] for item in full_semantic
    ]:
        return None
    low = 0
    high = len(full_text)
    while low < high:
        middle = (low + high) // 2
        candidate = _semantic_for_text(
            clip, full_text[:middle], encoder_id, tokenizer
        )
        if len(candidate) >= len(full_semantic):
            high = middle
        else:
            low = middle + 1
    return low


def _chunk_ranges(records: Sequence[dict[str, Any]]) -> tuple[SpatialTokenChunkRange, ...]:
    ranges: list[SpatialTokenChunkRange] = []
    for record in records:
        if (
            ranges
            and ranges[-1].chunk_index == record["chunk_index"]
            and ranges[-1].end == record["context_index"]
        ):
            previous = ranges[-1]
            ranges[-1] = SpatialTokenChunkRange(
                chunk_index=previous.chunk_index,
                start=previous.start,
                end=record["context_index"] + 1,
            )
        else:
            ranges.append(
                SpatialTokenChunkRange(
                    chunk_index=record["chunk_index"],
                    start=record["context_index"],
                    end=record["context_index"] + 1,
                )
            )
    return tuple(ranges)


def _adapter_role(adapter_kind: str, encoder_id: str) -> str:
    if adapter_kind == "identity":
        return "context_identity"
    if adapter_kind == "anima_native_t5_query_v1":
        return "context_query" if encoder_id == "t5xxl" else "semantic_source"
    if adapter_kind == "anima_progressive_qwen35_t5_query_v1":
        return "context_query" if encoder_id == "t5xxl" else "semantic_source"
    raise SpatialTokenBindingError("spatial_token_binding_unsupported")


def _adapter_indices(
    adapter_kind: str,
    encoder_id: str,
    context_indices: tuple[int, ...],
) -> tuple[int, ...]:
    role = _adapter_role(adapter_kind, encoder_id)
    if role in {"context_identity", "context_query"}:
        if adapter_kind.startswith("anima_") and any(index >= 512 for index in context_indices):
            raise SpatialTokenBindingError("spatial_token_binding_unsupported")
        return context_indices
    return ()


def _target_payload(binding: SpatialTargetTokenBinding) -> dict[str, Any]:
    payload = asdict(binding)
    payload["chunk_ranges"] = [asdict(item) for item in binding.chunk_ranges]
    return payload


def _encoder_receipt_payload(binding: SpatialEncoderTokenBinding) -> dict[str, Any]:
    payload = asdict(binding)
    payload["targets"] = [_target_payload(item) for item in binding.targets]
    return payload


def bind_spatial_tokens(
    clip: Any,
    *,
    full_text: str,
    character_ranges: Iterable[Mapping[str, Any]],
    required_encoders: Sequence[str],
    adapter_kind: str = "identity",
    step_start_percent: float = 0.0,
    step_end_percent: float = 1.0,
) -> SpatialTokenBindingBundle:
    return prepare_spatial_tokens(
        clip,
        full_text=full_text,
        character_ranges=character_ranges,
        required_encoders=required_encoders,
        adapter_kind=adapter_kind,
        step_start_percent=step_start_percent,
        step_end_percent=step_end_percent,
    ).binding


def prepare_spatial_tokens(
    clip: Any,
    *,
    full_text: str,
    character_ranges: Iterable[Mapping[str, Any]],
    required_encoders: Sequence[str],
    adapter_kind: str = "identity",
    step_start_percent: float = 0.0,
    step_end_percent: float = 1.0,
) -> PreparedSpatialTokens:
    if not isinstance(full_text, str) or not full_text:
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    if not (0.0 <= step_start_percent < step_end_percent <= 1.0):
        raise SpatialTokenBindingError("spatial_token_binding_step_range_invalid")
    normalized_ranges: list[tuple[str, int, int]] = []
    for item in character_ranges:
        if not isinstance(item, Mapping):
            raise SpatialTokenBindingError("spatial_token_binding_unresolved")
        target_id = validate_target_id(item.get("target_id"))
        start = item.get("character_start")
        end = item.get("character_end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise SpatialTokenBindingError("spatial_token_binding_unresolved")
        if start < 0 or end <= start or end > len(full_text):
            raise SpatialTokenBindingError("spatial_token_binding_unresolved")
        normalized_ranges.append((target_id, start, end))
    if not normalized_ranges or len({item[0] for item in normalized_ranges}) != len(
        normalized_ranges
    ):
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    ordered_ranges = sorted(normalized_ranges, key=lambda item: item[1])
    if any(
        current[1] < previous[2]
        for previous, current in zip(ordered_ranges, ordered_ranges[1:])
    ):
        raise SpatialTokenBindingError("spatial_token_binding_unresolved")
    if (
        not required_encoders
        or any(not isinstance(item, str) or not item for item in required_encoders)
        or len(set(required_encoders)) != len(required_encoders)
    ):
        raise SpatialTokenBindingError("spatial_token_binding_encoder_missing")
    expected_anima_encoders = {
        "anima_native_t5_query_v1": {"qwen3_06b", "t5xxl"},
        "anima_progressive_qwen35_t5_query_v1": {"qwen35_4b", "t5xxl"},
    }
    expected = expected_anima_encoders.get(adapter_kind)
    if expected is not None and set(required_encoders) != expected:
        raise SpatialTokenBindingError("spatial_token_binding_encoder_missing")

    full_payload = _tokenize(clip, full_text)
    full_text_sha = sha256(full_text.encode("utf-8")).hexdigest()
    encoders: list[SpatialEncoderTokenBinding] = []
    for encoder_id in required_encoders:
        tokenizer = _encoder_tokenizer(clip, encoder_id)
        flattened, semantic = _flatten_encoder(
            _encoder_payload(full_payload, encoder_id), tokenizer=tokenizer
        )
        truncation_start = _truncation_start(
            clip, full_text, encoder_id, tokenizer, semantic
        )
        boundary_cache: dict[int, list[dict[str, Any]]] = {
            0: [],
            len(full_text): semantic,
        }

        def semantic_prefix(character_index: int) -> list[dict[str, Any]]:
            cached = boundary_cache.get(character_index)
            if cached is not None:
                return cached
            resolved = _semantic_for_text(
                clip, full_text[:character_index], encoder_id, tokenizer
            )
            boundary_cache[character_index] = resolved
            return resolved

        target_bindings: list[SpatialTargetTokenBinding] = []
        for target_id, character_start, character_end in normalized_ranges:
            if truncation_start is not None and character_end > truncation_start:
                raise SpatialTokenBindingError("spatial_token_binding_unsupported")
            start_semantic = semantic_prefix(character_start)
            end_semantic = semantic_prefix(character_end)
            semantic_start = _prefix_length(semantic, start_semantic)
            semantic_end = _prefix_length(semantic, end_semantic)
            if semantic_end <= semantic_start:
                raise SpatialTokenBindingError("spatial_token_binding_unresolved")
            selected = semantic[semantic_start:semantic_end]
            context_indices = tuple(int(item["context_index"]) for item in selected)
            adapter_indices = _adapter_indices(
                adapter_kind, encoder_id, context_indices
            )
            target_bindings.append(
                SpatialTargetTokenBinding(
                    target_id=target_id,
                    character_start=character_start,
                    character_end=character_end,
                    semantic_start=semantic_start,
                    semantic_end=semantic_end,
                    context_indices=context_indices,
                    adapter_context_indices=adapter_indices,
                    chunk_ranges=_chunk_ranges(selected),
                    token_weights=tuple(float(item["weight"]) for item in selected),
                )
            )
        encoders.append(
            SpatialEncoderTokenBinding(
                encoder_id=encoder_id,
                tokenizer_fingerprint=_vocabulary_fingerprint(tokenizer, encoder_id),
                full_text_sha256=full_text_sha,
                token_ids=tuple(int(item["token_id"]) for item in flattened),
                special_token_indices=tuple(
                    int(item["context_index"])
                    for item in flattened
                    if item["special"]
                ),
                padding_token_indices=tuple(
                    int(item["context_index"])
                    for item in flattened
                    if item["padding"]
                ),
                overflow_chunk_indices=tuple(
                    sorted({int(item["chunk_index"]) for item in flattened if item["chunk_index"] > 0})
                ),
                truncated=truncation_start is not None,
                truncation_character_start=truncation_start,
                adapter_role=_adapter_role(adapter_kind, encoder_id),
                targets=tuple(target_bindings),
            )
        )

    if adapter_kind == "identity" and set(required_encoders) == {"l", "g"}:
        effective_length = min(len(item.token_ids) for item in encoders)
        for encoder in encoders:
            if any(
                index >= effective_length
                for target in encoder.targets
                for index in target.adapter_context_indices
            ):
                raise SpatialTokenBindingError("spatial_token_binding_unsupported")

    body = {
        "schema_version": "asteria.spatial-token-binding.v1",
        "full_text": full_text,
        "full_text_sha256": full_text_sha,
        "step_start_percent": step_start_percent,
        "step_end_percent": step_end_percent,
        "pooled_output_scope": "global_only",
        "encoders": [_encoder_receipt_payload(item) for item in encoders],
    }
    binding = SpatialTokenBindingBundle(
        schema_version=body["schema_version"],
        full_text=full_text,
        full_text_sha256=full_text_sha,
        step_start_percent=step_start_percent,
        step_end_percent=step_end_percent,
        pooled_output_scope="global_only",
        encoders=tuple(encoders),
        receipt_sha256=canonical_json_sha256(body),
    )
    return PreparedSpatialTokens(tokens=full_payload, binding=binding)


def token_binding_payload(bundle: SpatialTokenBindingBundle) -> dict[str, Any]:
    body = asdict(bundle)
    body["encoders"] = [_encoder_receipt_payload(item) for item in bundle.encoders]
    expected = canonical_json_sha256(
        {key: value for key, value in body.items() if key != "receipt_sha256"}
    )
    if expected != bundle.receipt_sha256:
        raise SpatialTokenBindingError("spatial_token_binding_receipt_mismatch")
    return body


def prepare_spatial_token_schedule(
    clip: Any,
    *,
    scheduled_bindings: Sequence[Mapping[str, Any]],
    required_encoders: Sequence[str],
    adapter_kind: str,
) -> SpatialScheduledTokenBindingBundle:
    if not scheduled_bindings:
        raise SpatialTokenBindingError("spatial_token_binding_schedule_empty")
    bindings: list[SpatialTokenBindingBundle] = []
    cursor = 0.0
    target_ids: tuple[str, ...] | None = None
    for item in scheduled_bindings:
        if not isinstance(item, Mapping):
            raise SpatialTokenBindingError("spatial_token_binding_schedule_invalid")
        start = item.get("start_percent")
        end = item.get("end_percent")
        full_text = item.get("full_text")
        ranges = item.get("character_bindings")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or isinstance(end, bool)
            or not isinstance(end, (int, float))
            or not isinstance(full_text, str)
            or not isinstance(ranges, list)
            or abs(float(start) - cursor) > 1e-9
        ):
            raise SpatialTokenBindingError("spatial_token_binding_schedule_invalid")
        character_ranges = [
            {
                "target_id": value.get("target_id"),
                "character_start": value.get("start"),
                "character_end": value.get("end"),
            }
            for value in ranges
            if isinstance(value, Mapping)
        ]
        if len(character_ranges) != len(ranges):
            raise SpatialTokenBindingError("spatial_token_binding_schedule_invalid")
        prepared = prepare_spatial_tokens(
            clip,
            full_text=full_text,
            character_ranges=character_ranges,
            required_encoders=required_encoders,
            adapter_kind=adapter_kind,
            step_start_percent=float(start),
            step_end_percent=float(end),
        )
        current_ids = tuple(
            target.target_id for target in prepared.binding.encoders[0].targets
        )
        if target_ids is None:
            target_ids = current_ids
        elif current_ids != target_ids:
            raise SpatialTokenBindingError("spatial_token_binding_schedule_targets_mismatch")
        bindings.append(prepared.binding)
        cursor = float(end)
    if abs(cursor - 1.0) > 1e-9:
        raise SpatialTokenBindingError("spatial_token_binding_schedule_invalid")
    body = {
        "schema_version": "asteria.spatial-token-binding-schedule.v1",
        "bindings": [token_binding_payload(item) for item in bindings],
    }
    return SpatialScheduledTokenBindingBundle(
        schema_version=body["schema_version"],
        bindings=tuple(bindings),
        receipt_sha256=canonical_json_sha256(body),
    )


def scheduled_token_binding_payload(
    bundle: SpatialScheduledTokenBindingBundle,
) -> dict[str, Any]:
    body = {
        "schema_version": bundle.schema_version,
        "bindings": [token_binding_payload(item) for item in bundle.bindings],
    }
    if canonical_json_sha256(body) != bundle.receipt_sha256:
        raise SpatialTokenBindingError("spatial_token_binding_receipt_mismatch")
    return {**body, "receipt_sha256": bundle.receipt_sha256}
