from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
import hashlib
import json
import math
from threading import RLock
from typing import Callable, Iterable, Literal

from app.schemas.provider_capabilities import ProviderCapabilityDescriptor
from app.schemas.resource_advisory import (
    ResourceAdvisory,
    ResourceAdvisoryAcknowledgement,
    ResourceAdvisoryAuthorization,
    ResourceAdvisoryEvidence,
    ResourceEstimate,
    ResourceEstimateConfidence,
    ResourceObservationSnapshot,
    ResourceWorkload,
)


_MIB = 1024**2


class ResourceAdvisoryError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ResourceEstimateProfile:
    adapter_id: str
    model_profile: str
    revision: str
    baseline_vram_bytes: int
    baseline_ram_bytes: int
    vram_bytes_per_work_unit: int
    ram_bytes_per_work_unit: int
    temp_bytes_per_work_unit: int
    output_bytes_per_work_unit: int
    normal_work_units: float
    confidence: Literal["high", "medium", "low"]

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.model_profile or not self.revision:
            raise ValueError("resource estimate profile identifiers are required")
        numeric = (
            self.baseline_vram_bytes,
            self.baseline_ram_bytes,
            self.vram_bytes_per_work_unit,
            self.ram_bytes_per_work_unit,
            self.temp_bytes_per_work_unit,
            self.output_bytes_per_work_unit,
        )
        if any(isinstance(value, bool) or value < 0 for value in numeric):
            raise ValueError("resource estimate profile values must be non-negative")
        if not math.isfinite(self.normal_work_units) or self.normal_work_units <= 0:
            raise ValueError("resource estimate profile normal workload must be positive")


class ResourceAdvisor:
    def __init__(self, profiles: Iterable[ResourceEstimateProfile] = ()) -> None:
        self._profiles: dict[tuple[str, str], ResourceEstimateProfile] = {}
        for profile in profiles:
            key = (profile.adapter_id, profile.model_profile)
            if key in self._profiles:
                raise ValueError("resource estimate profile is duplicated")
            self._profiles[key] = profile

    def advise(
        self,
        workload: ResourceWorkload,
        capability: ProviderCapabilityDescriptor,
        observation: ResourceObservationSnapshot,
    ) -> ResourceAdvisory:
        self._validate_identity(workload, capability)
        if not workload.provider_id.startswith("local_"):
            return self._unknown(
                workload,
                observation,
                applicable=False,
                evidence_code="resource_advisory_local_only",
            )
        profile = self._profiles.get((workload.adapter_id, workload.model_profile))
        if profile is None:
            return self._unknown(
                workload,
                observation,
                applicable=True,
                evidence_code="resource_estimate_profile_unavailable",
            )
        if (
            observation.vram is None
            or observation.ram is None
            or observation.temp_free_bytes is None
            or observation.output_free_bytes is None
        ):
            return self._unknown(
                workload,
                observation,
                applicable=True,
                evidence_code="resource_telemetry_unavailable",
                profile_revision=profile.revision,
            )

        units = _work_units(workload)
        precision_factor = {
            "auto": Decimal("1"),
            "bf16": Decimal("1"),
            "fp16": Decimal("1"),
            "fp32": Decimal("1.5"),
        }[workload.precision]
        vram_offload = {
            "auto": Decimal("0.8"),
            "none": Decimal("1"),
            "balanced": Decimal("0.7"),
            "maximum": Decimal("0.45"),
            "layer_streaming": Decimal("0.35"),
        }[workload.offload]
        ram_offload = {
            "auto": Decimal("1.1"),
            "none": Decimal("1"),
            "balanced": Decimal("1.2"),
            "maximum": Decimal("1.45"),
            "layer_streaming": Decimal("1.6"),
        }[workload.offload]
        peak_vram = _rounded_bytes(
            profile.baseline_vram_bytes + profile.vram_bytes_per_work_unit * units * precision_factor * vram_offload
        )
        peak_ram = _rounded_bytes(
            profile.baseline_ram_bytes + profile.ram_bytes_per_work_unit * units * precision_factor * ram_offload
        )
        temporary = _rounded_bytes(profile.temp_bytes_per_work_unit * units)
        output = _rounded_bytes(profile.output_bytes_per_work_unit * units)
        estimate = ResourceEstimate(
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
            temporary_disk_bytes=temporary,
            output_bytes=output,
            relative_execution_class=_execution_class(units, profile.normal_work_units),
        )
        pressures = (
            ("resource_estimate_vram", peak_vram, observation.vram.available_bytes),
            ("resource_estimate_ram", peak_ram, observation.ram.available_bytes),
            ("resource_estimate_temp", temporary, observation.temp_free_bytes),
            ("resource_estimate_output", output, observation.output_free_bytes),
        )
        ratios = [
            Decimal(estimate_bytes) / Decimal(max(available_bytes, 1))
            for _, estimate_bytes, available_bytes in pressures
        ]
        maximum_pressure = max(ratios)
        level: Literal["normal", "caution", "high_risk"]
        if maximum_pressure >= Decimal("0.9"):
            level = "high_risk"
        elif maximum_pressure >= Decimal("0.7") or _outside_validated_range(workload, capability):
            level = "caution"
        else:
            level = "normal"
        evidence = tuple(
            ResourceAdvisoryEvidence(
                code=code,
                estimate_bytes=estimate_bytes,
                available_bytes=available_bytes,
            )
            for code, estimate_bytes, available_bytes in pressures
        )
        if _outside_validated_range(workload, capability):
            evidence += (ResourceAdvisoryEvidence(code="resource_outside_validated_range"),)
        mitigations = _mitigations(workload, pressures, level)
        return _build_advisory(
            applicable=True,
            level=level,
            confidence=profile.confidence,
            workload=workload,
            observation=observation,
            estimate=estimate,
            evidence=evidence,
            mitigation_codes=mitigations,
            profile_revision=profile.revision,
        )

    def authorize(
        self,
        advisory: ResourceAdvisory,
        acknowledgement: ResourceAdvisoryAcknowledgement | None,
    ) -> ResourceAdvisoryAuthorization:
        if advisory.acknowledgement_required:
            if acknowledgement is None:
                raise ResourceAdvisoryError("resource_advisory_acknowledgement_required")
            if acknowledgement.advisory_fingerprint != advisory.fingerprint:
                raise ResourceAdvisoryError("resource_advisory_acknowledgement_mismatch")
        return ResourceAdvisoryAuthorization(
            advisory=advisory,
            acknowledgement=acknowledgement if advisory.acknowledgement_required else None,
        )

    @staticmethod
    def _validate_identity(
        workload: ResourceWorkload,
        capability: ProviderCapabilityDescriptor,
    ) -> None:
        if workload.domain != capability.domain or workload.provider_id != capability.provider_id:
            raise ResourceAdvisoryError("resource_advisory_provider_mismatch")
        if workload.adapter_id != capability.adapter_id or workload.model_profile != capability.model_profile:
            raise ResourceAdvisoryError("resource_advisory_profile_mismatch")
        task = next((item for item in capability.tasks if item.task_id == workload.task_id), None)
        if task is None or not task.supported:
            raise ResourceAdvisoryError("resource_advisory_task_unavailable")

    def _unknown(
        self,
        workload: ResourceWorkload,
        observation: ResourceObservationSnapshot,
        *,
        applicable: bool,
        evidence_code: str,
        profile_revision: str | None = None,
    ) -> ResourceAdvisory:
        return _build_advisory(
            applicable=applicable,
            level="unknown",
            confidence="unknown",
            workload=workload,
            observation=observation,
            estimate=ResourceEstimate(relative_execution_class="unknown"),
            evidence=(ResourceAdvisoryEvidence(code=evidence_code),),
            mitigation_codes=(),
            profile_revision=profile_revision,
        )


def _work_units(workload: ResourceWorkload) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        duration = Decimal(str(workload.duration_seconds or 1.0))
        if workload.width is None or workload.height is None:
            media_units = max(duration, Decimal("1"))
        else:
            megapixels = Decimal(workload.width * workload.height) / Decimal(1_000_000)
            fps = Decimal(str(workload.fps or 24.0))
            media_units = megapixels * duration * (fps / Decimal(24))
        reference_factor = (
            Decimal("1")
            + Decimal(workload.guide_count) * Decimal("0.08")
            + Decimal(str(workload.guide_duration_seconds)) * Decimal("0.02")
        )
        media_factor = (Decimal("1.1") if workload.audio else Decimal("1")) * (
            Decimal("1.25") if workload.sequence_context else Decimal("1")
        )
        return media_units * Decimal(workload.batch) * reference_factor * media_factor


def _rounded_bytes(value: Decimal) -> int:
    if not value.is_finite() or value < 0:
        raise ResourceAdvisoryError("resource_estimate_invalid")
    try:
        units = (value / Decimal(_MIB)).to_integral_value(rounding=ROUND_CEILING)
        return int(units) * _MIB
    except (InvalidOperation, OverflowError, ValueError) as error:
        raise ResourceAdvisoryError("resource_estimate_invalid") from error


def _execution_class(units: Decimal, normal_units: float) -> Literal["light", "moderate", "heavy", "extreme"]:
    relative = units / Decimal(str(normal_units))
    if relative <= Decimal("0.5"):
        return "light"
    if relative <= Decimal("1"):
        return "moderate"
    if relative <= Decimal("4"):
        return "heavy"
    return "extreme"


def _outside_validated_range(
    workload: ResourceWorkload,
    capability: ProviderCapabilityDescriptor,
) -> bool:
    duration_range = capability.duration.validated_range
    if workload.duration_seconds is not None and duration_range is not None:
        if duration_range.minimum is not None and workload.duration_seconds < duration_range.minimum:
            return True
        if duration_range.maximum is not None and workload.duration_seconds > duration_range.maximum:
            return True
    if workload.width is not None and workload.height is not None and capability.size is not None:
        megapixel_range = capability.size.validated_megapixels
        megapixels = workload.width * workload.height / 1_000_000
        if megapixel_range is not None:
            if megapixel_range.minimum is not None and megapixels < megapixel_range.minimum:
                return True
            if megapixel_range.maximum is not None and megapixels > megapixel_range.maximum:
                return True
    return False


def _mitigations(
    workload: ResourceWorkload,
    pressures: tuple[tuple[str, int, int], ...],
    level: str,
) -> tuple[str, ...]:
    if level == "normal":
        return ()
    codes: list[str] = []
    if any(
        code == "resource_estimate_vram"
        and Decimal(estimate) / Decimal(max(available, 1)) >= Decimal("0.7")
        for code, estimate, available in pressures
    ):
        codes.extend(("resource_mitigation_increase_offload", "resource_mitigation_lower_precision"))
    if workload.width is not None:
        codes.append("resource_mitigation_reduce_resolution")
    if workload.duration_seconds is not None:
        codes.append("resource_mitigation_reduce_duration")
    if workload.guide_count:
        codes.append("resource_mitigation_reduce_references")
    return tuple(dict.fromkeys(codes))


def _build_advisory(
    *,
    applicable: bool,
    level: Literal["normal", "caution", "high_risk", "unknown"],
    confidence: ResourceEstimateConfidence,
    workload: ResourceWorkload,
    observation: ResourceObservationSnapshot,
    estimate: ResourceEstimate,
    evidence: tuple[ResourceAdvisoryEvidence, ...],
    mitigation_codes: tuple[str, ...],
    profile_revision: str | None,
) -> ResourceAdvisory:
    body = {
        "applicable": applicable,
        "level": level,
        "confidence": confidence,
        "provider_id": workload.provider_id,
        "adapter_id": workload.adapter_id,
        "model_profile": workload.model_profile,
        "profile_revision": profile_revision,
        "workload": workload.model_dump(mode="json"),
        "observation": observation.model_dump(mode="json"),
        "estimate": estimate.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "mitigation_codes": list(mitigation_codes),
    }
    fingerprint = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResourceAdvisory(
        **body,
        can_run=True,
        acknowledgement_required=applicable and level == "high_risk",
        fingerprint=fingerprint,
    )


def get_resource_advisor() -> ResourceAdvisor:
    # Task 5 registers verified model/runtime calibration profiles. Until then,
    # missing calibration is represented honestly as an unknown advisory.
    return ResourceAdvisor()


class ResourceCapabilityRegistry:
    def __init__(
        self,
        factories: Iterable[Callable[[], ProviderCapabilityDescriptor]] = (),
    ) -> None:
        self._lock = RLock()
        self._factories: dict[
            tuple[str, str, str, str],
            Callable[[], ProviderCapabilityDescriptor],
        ] = {}
        for factory in factories:
            self.register(factory)

    def register(self, factory: Callable[[], ProviderCapabilityDescriptor]) -> None:
        capability = factory()
        key = _capability_key(capability)
        with self._lock:
            if key in self._factories:
                raise ValueError("resource capability provider is duplicated")
            self._factories[key] = factory

    def snapshot(self) -> tuple[ProviderCapabilityDescriptor, ...]:
        with self._lock:
            items = tuple(self._factories.items())
        capabilities: list[ProviderCapabilityDescriptor] = []
        for expected_key, factory in items:
            capability = factory()
            if _capability_key(capability) != expected_key:
                raise ResourceAdvisoryError("resource_capability_identity_changed")
            capabilities.append(capability)
        return tuple(sorted(capabilities, key=_capability_key))


def _capability_key(
    capability: ProviderCapabilityDescriptor,
) -> tuple[str, str, str, str]:
    return (
        capability.domain,
        capability.provider_id,
        capability.adapter_id,
        capability.model_profile,
    )
