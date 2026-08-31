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
from .guidance_loss_runtime import SpatialLossWeights


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
    guidance_start_percent: float
    guidance_end_percent: float
    guidance_iterations: int
    guidance_step_size: float
    attention_prior_scale: float
    attention_layers: tuple[str, ...]
    self_relation_layers: tuple[str, ...]
    global_retention_floor: float
    loss_weights: SpatialLossWeights
    legacy_early_guidance_end: float = 0.35
    legacy_isolation_start: float = 0.15
    legacy_isolation_end: float = 0.85
    legacy_late_blending_start: float = 0.65
    legacy_schedule_smoothing: float = 0.08
    legacy_cross_inside_bias: float = 3.0
    legacy_cross_outside_penalty: float = 6.0
    legacy_global_inside_penalty: float = 0.4
    legacy_self_leakage_penalty: float = 2.0
    legacy_self_attention_token_limit: int = 4096

    # The legacy output-blending adapter remains importable only for historical
    # workflow inspection. New workflows never use these compatibility values.
    @property
    def early_guidance_end(self) -> float:
        return self.legacy_early_guidance_end

    @property
    def isolation_start(self) -> float:
        return self.legacy_isolation_start

    @property
    def isolation_end(self) -> float:
        return self.legacy_isolation_end

    @property
    def late_blending_start(self) -> float:
        return self.legacy_late_blending_start

    @property
    def schedule_smoothing(self) -> float:
        return self.legacy_schedule_smoothing

    @property
    def cross_inside_bias(self) -> float:
        return self.legacy_cross_inside_bias

    @property
    def cross_outside_penalty(self) -> float:
        return self.legacy_cross_outside_penalty

    @property
    def global_inside_penalty(self) -> float:
        return self.legacy_global_inside_penalty

    @property
    def self_leakage_penalty(self) -> float:
        return self.legacy_self_leakage_penalty

    @property
    def self_attention_token_limit(self) -> int:
        return self.legacy_self_attention_token_limit


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
_GUIDANCE_KEYS = {
    "start_percent",
    "end_percent",
    "iterations",
    "step_size",
    "attention_prior_scale",
    "attention_layers",
    "self_relation_layers",
    "global_retention_floor",
    "loss_weights",
}
_LOSS_WEIGHT_KEYS = {"position", "size", "separation", "presence", "global_retention"}
_LEGACY_COEFFICIENT_KEYS = {
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


def _validated_receipt(payload: dict[str, Any]) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    receipt = payload.get("profile_receipt")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("Asteria spatial profile receipt is invalid")
    if receipt.get("schema_version") != "asteria.spatial-model-profile.v1":
        raise ValueError("Asteria spatial profile receipt is invalid")
    if receipt.get("support_status") != "supported":
        raise ValueError("Asteria spatial profile support is not confirmed")
    for field in (
        "profile_id", "profile_revision", "model_family", "model_slot_id",
        "encoder_kind", "verified_structure_id",
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
    target_ids = tuple(validate_target_id(value) for value in receipt.get("target_ids", ()))
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("Asteria spatial profile target ids are invalid")
    return receipt, actual, target_ids


def _load_legacy_phase_profile(payload: dict[str, Any]) -> SpatialPhaseProfile:
    if set(payload) != {
        "schema_version", "contract_version", "profile_receipt", "coefficients",
        "profile_payload_sha256",
    } or payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Asteria spatial profile contract is invalid")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, dict) or set(coefficients) != _LEGACY_COEFFICIENT_KEYS:
        raise ValueError("Asteria spatial profile coefficients are invalid")
    receipt, actual_receipt, target_ids = _validated_receipt(payload)
    expected_payload = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "profile_payload_sha256"}
    )
    actual_payload = validate_sha256(payload.get("profile_payload_sha256"), "profile payload")
    if actual_payload != expected_payload:
        raise ValueError("Asteria spatial profile payload does not match")
    early = finite_number(coefficients.get("early_guidance_end"), "early guidance end", minimum=0.0, maximum=1.0)
    isolation_start = finite_number(coefficients.get("isolation_start"), "isolation start", minimum=0.0, maximum=1.0)
    isolation_end = finite_number(coefficients.get("isolation_end"), "isolation end", minimum=0.0, maximum=1.0)
    late = finite_number(coefficients.get("late_blending_start"), "late blending start", minimum=0.0, maximum=1.0)
    smoothing = finite_number(coefficients.get("schedule_smoothing"), "schedule smoothing", minimum=0.001, maximum=0.25)
    cross_inside = finite_number(coefficients.get("cross_inside_bias"), "cross inside bias", minimum=0.0, maximum=32.0)
    cross_outside = finite_number(coefficients.get("cross_outside_penalty"), "cross outside penalty", minimum=0.0, maximum=32.0)
    global_inside = finite_number(coefficients.get("global_inside_penalty"), "global inside penalty", minimum=0.0, maximum=8.0)
    self_leakage = finite_number(coefficients.get("self_leakage_penalty"), "self leakage penalty", minimum=0.0, maximum=32.0)
    token_limit = coefficients.get("self_attention_token_limit")
    if isinstance(token_limit, bool) or not isinstance(token_limit, int) or not 1 <= token_limit <= 65536:
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
        guidance_start_percent=0.0,
        guidance_end_percent=early,
        guidance_iterations=1,
        guidance_step_size=0.01,
        attention_prior_scale=0.0,
        attention_layers=("legacy",),
        self_relation_layers=("legacy",),
        global_retention_floor=0.5,
        loss_weights=SpatialLossWeights(1.0, 1.0, 1.0, 1.0, 1.0),
        legacy_early_guidance_end=early,
        legacy_isolation_start=isolation_start,
        legacy_isolation_end=isolation_end,
        legacy_late_blending_start=late,
        legacy_schedule_smoothing=smoothing,
        legacy_cross_inside_bias=cross_inside,
        legacy_cross_outside_penalty=cross_outside,
        legacy_global_inside_penalty=global_inside,
        legacy_self_leakage_penalty=self_leakage,
        legacy_self_attention_token_limit=token_limit,
    )


def load_phase_profile(profile_json: Any) -> SpatialPhaseProfile:
    raw = str(profile_json or "")
    if len(raw.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Asteria spatial profile is too large")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Asteria spatial profile is invalid") from error
    if isinstance(payload, dict) and payload.get("schema_version") == "asteria.spatial-node-profile.v2":
        return _load_legacy_phase_profile(payload)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "contract_version",
        "profile_receipt",
        "guidance",
        "profile_payload_sha256",
    }:
        raise ValueError("Asteria spatial profile contract is invalid")
    if payload.get("schema_version") != "asteria.spatial-node-profile.v3":
        raise ValueError("Asteria spatial profile contract is invalid")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Asteria spatial profile contract is invalid")
    receipt = payload.get("profile_receipt")
    guidance = payload.get("guidance")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("Asteria spatial profile receipt is invalid")
    if not isinstance(guidance, dict) or set(guidance) != _GUIDANCE_KEYS:
        raise ValueError("Asteria spatial guidance profile is invalid")
    loss_weights = guidance.get("loss_weights")
    if not isinstance(loss_weights, dict) or set(loss_weights) != _LOSS_WEIGHT_KEYS:
        raise ValueError("Asteria spatial guidance loss weights are invalid")
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
    start = finite_number(
        guidance.get("start_percent"),
        "guidance start percent",
        minimum=0.0,
        maximum=1.0,
    )
    end = finite_number(
        guidance.get("end_percent"),
        "guidance end percent",
        minimum=0.0,
        maximum=1.0,
    )
    step_size = finite_number(
        guidance.get("step_size"),
        "guidance step size",
        minimum=0.0,
        maximum=1.0,
    )
    attention_prior_scale = finite_number(
        guidance.get("attention_prior_scale"),
        "attention prior scale",
        minimum=0.0,
        maximum=5.0,
    )
    retention_floor = finite_number(
        guidance.get("global_retention_floor"),
        "global retention floor",
        minimum=0.0,
        maximum=1.0,
    )
    iterations = guidance.get("iterations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 8
    ):
        raise ValueError("Asteria spatial guidance iterations are invalid")
    attention_layers = guidance.get("attention_layers")
    self_relation_layers = guidance.get("self_relation_layers")
    for values, label in (
        (attention_layers, "attention layers"),
        (self_relation_layers, "self relation layers"),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"Asteria spatial {label} are invalid")
    if start >= end or step_size <= 0.0 or retention_floor <= 0.0:
        raise ValueError("Asteria spatial guidance profile range is invalid")
    weights = SpatialLossWeights(
        position=loss_weights["position"],
        size=loss_weights["size"],
        separation=loss_weights["separation"],
        presence=loss_weights["presence"],
        global_retention=loss_weights["global_retention"],
    )
    return SpatialPhaseProfile(
        stage=validate_stage(receipt.get("stage")),
        profile_id=str(receipt.get("profile_id") or ""),
        profile_revision=str(receipt.get("profile_revision") or ""),
        model_family=str(receipt.get("model_family") or ""),
        encoder_kind=str(receipt.get("encoder_kind") or ""),
        profile_receipt_sha256=actual_receipt,
        profile_payload_sha256=actual_payload,
        target_ids=target_ids,
        guidance_start_percent=start,
        guidance_end_percent=end,
        guidance_iterations=iterations,
        guidance_step_size=step_size,
        attention_prior_scale=attention_prior_scale,
        attention_layers=tuple(attention_layers),
        self_relation_layers=tuple(self_relation_layers),
        global_retention_floor=retention_floor,
        loss_weights=weights,
    )
