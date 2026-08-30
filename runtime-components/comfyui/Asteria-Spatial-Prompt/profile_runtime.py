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
class SpatialPhaseProfile:
    stage: str
    profile_id: str
    profile_revision: str
    model_family: str
    encoder_kind: str
    profile_receipt_sha256: str
    profile_payload_sha256: str
    target_ids: tuple[str, ...]
    early_guidance_end: float
    isolation_start: float
    isolation_end: float
    late_blending_start: float
    schedule_smoothing: float
    cross_inside_bias: float
    cross_outside_penalty: float
    global_inside_penalty: float
    self_leakage_penalty: float
    self_attention_token_limit: int


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
_COEFFICIENT_KEYS = {
    "early_guidance_end",
    "isolation_start",
    "isolation_end",
    "late_blending_start",
    "schedule_smoothing",
    "cross_inside_bias",
    "cross_outside_penalty",
    "global_inside_penalty",
    "self_leakage_penalty",
    "self_attention_token_limit",
}


def load_phase_profile(profile_json: Any) -> SpatialPhaseProfile:
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
        "coefficients",
        "profile_payload_sha256",
    }:
        raise ValueError("Asteria spatial profile contract is invalid")
    if payload.get("schema_version") != "asteria.spatial-node-profile.v2":
        raise ValueError("Asteria spatial profile contract is invalid")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Asteria spatial profile contract is invalid")
    receipt = payload.get("profile_receipt")
    coefficients = payload.get("coefficients")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("Asteria spatial profile receipt is invalid")
    if not isinstance(coefficients, dict) or set(coefficients) != _COEFFICIENT_KEYS:
        raise ValueError("Asteria spatial profile coefficients are invalid")
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
    expected_receipt = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    actual_receipt = validate_sha256(receipt.get("receipt_sha256"), "profile receipt")
    if actual_receipt != expected_receipt:
        raise ValueError("Asteria spatial profile receipt does not match")
    expected_payload = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "profile_payload_sha256"}
    )
    actual_payload = validate_sha256(
        payload.get("profile_payload_sha256"),
        "profile payload",
    )
    if actual_payload != expected_payload:
        raise ValueError("Asteria spatial profile payload does not match")
    target_ids = tuple(validate_target_id(value) for value in receipt.get("target_ids", ()))
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("Asteria spatial profile target ids are invalid")
    early = finite_number(
        coefficients.get("early_guidance_end"),
        "early guidance end",
        minimum=0.0,
        maximum=1.0,
    )
    isolation_start = finite_number(
        coefficients.get("isolation_start"),
        "isolation start",
        minimum=0.0,
        maximum=1.0,
    )
    isolation_end = finite_number(
        coefficients.get("isolation_end"),
        "isolation end",
        minimum=0.0,
        maximum=1.0,
    )
    late = finite_number(
        coefficients.get("late_blending_start"),
        "late blending start",
        minimum=0.0,
        maximum=1.0,
    )
    smoothing = finite_number(
        coefficients.get("schedule_smoothing"),
        "schedule smoothing",
        minimum=0.001,
        maximum=0.25,
    )
    cross_inside = finite_number(
        coefficients.get("cross_inside_bias"),
        "cross inside bias",
        minimum=0.0,
        maximum=32.0,
    )
    cross_outside = finite_number(
        coefficients.get("cross_outside_penalty"),
        "cross outside penalty",
        minimum=0.0,
        maximum=32.0,
    )
    global_inside = finite_number(
        coefficients.get("global_inside_penalty"),
        "global inside penalty",
        minimum=0.0,
        maximum=8.0,
    )
    self_leakage = finite_number(
        coefficients.get("self_leakage_penalty"),
        "self leakage penalty",
        minimum=0.0,
        maximum=32.0,
    )
    token_limit = coefficients.get("self_attention_token_limit")
    if (
        isinstance(token_limit, bool)
        or not isinstance(token_limit, int)
        or token_limit < 1
        or token_limit > 65536
    ):
        raise ValueError("Asteria spatial self attention token limit is invalid")
    if early <= 0.0 or isolation_start >= isolation_end or late >= 1.0:
        raise ValueError("Asteria spatial profile coefficient order is invalid")
    return SpatialPhaseProfile(
        stage=validate_stage(receipt.get("stage")),
        profile_id=str(receipt.get("profile_id") or ""),
        profile_revision=str(receipt.get("profile_revision") or ""),
        model_family=str(receipt.get("model_family") or ""),
        encoder_kind=str(receipt.get("encoder_kind") or ""),
        profile_receipt_sha256=actual_receipt,
        profile_payload_sha256=actual_payload,
        target_ids=target_ids,
        early_guidance_end=early,
        isolation_start=isolation_start,
        isolation_end=isolation_end,
        late_blending_start=late,
        schedule_smoothing=smoothing,
        cross_inside_bias=cross_inside,
        cross_outside_penalty=cross_outside,
        global_inside_penalty=global_inside,
        self_leakage_penalty=self_leakage,
        self_attention_token_limit=token_limit,
    )
