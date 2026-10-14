from __future__ import annotations

import json

from pydantic import ValidationError

from app.schemas.image_check import ImageCheckAssessment, ImageCheckDecision, ImageCheckSettings


def remap_checked_image_paths(settings: dict, before: list[str], after: list[str]) -> None:
    check = settings.get("image_check")
    if not isinstance(check, dict) or len(before) != len(after):
        return
    mapping = dict(zip(before, after))
    for result in check.get("results", []):
        if result.get("saved_path") in mapping:
            result["saved_path"] = mapping[result["saved_path"]]


def decide_image_check(
    settings: ImageCheckSettings,
    assessment: ImageCheckAssessment | None,
    *,
    auto_repeat: bool = False,
) -> ImageCheckDecision:
    if not settings.enabled:
        return ImageCheckDecision(status="disabled", continue_postprocessing=True)
    if assessment is None:
        return ImageCheckDecision(status="unknown", pause=True, reason_code="image_check_result_missing")
    if assessment.verdict == "pass":
        return ImageCheckDecision(status="pass", continue_postprocessing=True)
    if assessment.verdict == "reject":
        return ImageCheckDecision(
            status="reject",
            save_rejected_image=settings.save_rejected,
            continue_repeat=auto_repeat,
        )
    return ImageCheckDecision(status="unknown", pause=True, reason_code=assessment.reason_code)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError("image_check_duplicate_response_field")
        values[key] = value
    return values


def parse_image_check_response(text: str) -> ImageCheckAssessment:
    try:
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 16 * 1024:
            raise ValueError("image_check_response_size_invalid")
        payload = json.loads(text, object_pairs_hook=_unique_json_object)
        if not isinstance(payload, dict):
            raise ValueError("image_check_response_object_required")
        return ImageCheckAssessment.model_validate_json(json.dumps(payload, allow_nan=False))
    except (ValidationError, ValueError, TypeError, RecursionError):
        # Provider text and validation errors can contain untrusted content.
        return ImageCheckAssessment(verdict="unknown", reason_code="image_check_response_invalid")
