from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .spatial_contract import (
    CONTRACT_VERSION,
    canonical_json_sha256,
    finite_number,
    validate_sha256,
    validate_stage,
    validate_target_id,
)


@dataclass(frozen=True, slots=True)
class SpatialRoutingProfile:
    stage: str
    profile_id: str
    profile_revision: str
    model_family: str
    model_slot_id: str
    encoder_kind: str
    verified_structure_id: str
    layout_fingerprint: str
    structure_fingerprint: str
    profile_receipt_sha256: str
    profile_payload_sha256: str
    target_ids: tuple[str, ...]
    cross_start_percent: float
    cross_end_percent: float
    self_start_percent: float
    self_end_percent: float
    cross_penalty: float
    cross_penalty_cap: float
    self_penalty: float
    self_penalty_cap: float
    core_threshold: float
    ownership_margin: float
    cross_attention_layers: tuple[str, ...]
    self_attention_layers: tuple[str, ...]
    self_attention_token_counts: tuple[int, ...]
    self_attention_token_limit: int
    hires_mode: str


_RECEIPT_KEYS = {
    "schema_version",
    "stage",
    "profile_id",
    "profile_revision",
    "model_family",
    "model_slot_id",
    "encoder_kind",
    "verified_structure_id",
    "layout_fingerprint",
    "structure_fingerprint",
    "support_status",
    "selection_mode",
    "reason_code",
    "target_ids",
    "receipt_sha256",
}
_ROUTING_KEYS = {
    "cross_start_percent",
    "cross_end_percent",
    "cross_penalty",
    "cross_penalty_cap",
    "core_threshold",
    "ownership_margin",
    "cross_attention_layers",
    "hires_mode",
}


def _validated_receipt(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    receipt = payload.get("profile_receipt")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("Asteria spatial profile receipt is invalid")
    if receipt.get("schema_version") != "asteria.spatial-model-profile.v1":
        raise ValueError("Asteria spatial profile receipt is invalid")
    if receipt.get("support_status") != "supported":
        raise ValueError("Asteria spatial profile support is not confirmed")
    for field in (
        "profile_id",
        "profile_revision",
        "model_family",
        "model_slot_id",
        "encoder_kind",
        "verified_structure_id",
    ):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise ValueError("Asteria spatial profile receipt is invalid")
    validate_sha256(receipt.get("layout_fingerprint"), "layout fingerprint")
    validate_sha256(receipt.get("structure_fingerprint"), "structure fingerprint")
    if receipt.get("selection_mode") not in {"auto", "manual"}:
        raise ValueError("Asteria spatial profile receipt is invalid")
    if receipt.get("reason_code") is not None:
        raise ValueError("Asteria spatial supported profile cannot have a reason")
    expected = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    actual = validate_sha256(receipt.get("receipt_sha256"), "profile receipt")
    if actual != expected:
        raise ValueError("Asteria spatial profile receipt does not match")
    target_ids = tuple(
        validate_target_id(value) for value in receipt.get("target_ids", ())
    )
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("Asteria spatial profile target ids are invalid")
    return receipt, actual, target_ids


def load_phase_profile(profile_json: Any) -> SpatialRoutingProfile:
    raw = str(profile_json or "")
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Asteria spatial profile is too large")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Asteria spatial profile is invalid") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "contract_version",
        "profile_receipt",
        "routing",
        "profile_payload_sha256",
    }:
        raise ValueError("Asteria spatial routing profile contract is invalid")
    if payload.get("schema_version") != "asteria.spatial-node-profile.v6":
        raise ValueError("Asteria spatial routing profile contract is invalid")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Asteria spatial routing profile contract is invalid")
    routing = payload.get("routing")
    if not isinstance(routing, dict) or set(routing) != _ROUTING_KEYS:
        raise ValueError("Asteria spatial routing profile is invalid")
    receipt, actual_receipt, target_ids = _validated_receipt(payload)
    expected_payload = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "profile_payload_sha256"}
    )
    actual_payload = validate_sha256(
        payload.get("profile_payload_sha256"),
        "profile payload",
    )
    if actual_payload != expected_payload:
        raise ValueError("Asteria spatial routing profile payload does not match")

    cross_start = finite_number(
        routing.get("cross_start_percent"),
        "cross routing start percent",
        minimum=0.0,
        maximum=1.0,
    )
    cross_end = finite_number(
        routing.get("cross_end_percent"),
        "cross routing end percent",
        minimum=0.0,
        maximum=1.0,
    )
    cross_penalty = finite_number(
        routing.get("cross_penalty"),
        "cross routing penalty",
        minimum=0.0,
        maximum=32.0,
    )
    cross_penalty_cap = finite_number(
        routing.get("cross_penalty_cap"),
        "cross routing penalty cap",
        minimum=0.0,
        maximum=32.0,
    )
    core_threshold = finite_number(
        routing.get("core_threshold"),
        "core threshold",
        minimum=0.0,
        maximum=1.0,
    )
    ownership_margin = finite_number(
        routing.get("ownership_margin"),
        "ownership margin",
        minimum=0.0,
        maximum=1.0,
    )
    cross_layers = routing.get("cross_attention_layers")
    for layers, label in ((cross_layers, "cross"),):
        if (
            not isinstance(layers, list)
            or not layers
            or any(not isinstance(value, str) or not value.strip() for value in layers)
            or len(layers) != len(set(layers))
            or "all" in layers
        ):
            raise ValueError(f"Asteria spatial {label} attention layers are invalid")
    hires_mode = routing.get("hires_mode")
    if hires_mode != "base_only":
        raise ValueError("Asteria spatial Hires routing mode is invalid")
    if (
        cross_start >= cross_end
        or cross_penalty <= 0.0
        or cross_penalty > cross_penalty_cap
        or core_threshold <= 0.0
    ):
        raise ValueError("Asteria spatial routing profile range is invalid")

    return SpatialRoutingProfile(
        stage=validate_stage(receipt.get("stage")),
        profile_id=str(receipt.get("profile_id") or ""),
        profile_revision=str(receipt.get("profile_revision") or ""),
        model_family=str(receipt.get("model_family") or ""),
        model_slot_id=str(receipt.get("model_slot_id") or ""),
        encoder_kind=str(receipt.get("encoder_kind") or ""),
        verified_structure_id=str(receipt.get("verified_structure_id") or ""),
        layout_fingerprint=str(receipt.get("layout_fingerprint") or ""),
        structure_fingerprint=str(receipt.get("structure_fingerprint") or ""),
        profile_receipt_sha256=actual_receipt,
        profile_payload_sha256=actual_payload,
        target_ids=target_ids,
        cross_start_percent=cross_start,
        cross_end_percent=cross_end,
        self_start_percent=0.0,
        self_end_percent=0.0,
        cross_penalty=cross_penalty,
        cross_penalty_cap=cross_penalty_cap,
        self_penalty=0.0,
        self_penalty_cap=0.0,
        core_threshold=core_threshold,
        ownership_margin=ownership_margin,
        cross_attention_layers=tuple(cross_layers),
        self_attention_layers=(),
        self_attention_token_counts=(),
        self_attention_token_limit=0,
        hires_mode=hires_mode,
    )
