from dataclasses import asdict, dataclass, field, is_dataclass
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.services.user_settings import UserSettingsRepository, resolve_output_root


logger = logging.getLogger(__name__)
_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)


@dataclass
class ArtifactDeletionResult:
    deleted_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    protected_files: list[str] = field(default_factory=list)


def collect_owned_artifact_paths(items: list[dict]) -> list[Any]:
    candidates: list[Any] = []
    seen: set[str] = set()

    def append(value: Any) -> None:
        key = _raw_identity(value)
        if key in seen:
            return
        seen.add(key)
        candidates.append(value)

    for item in items:
        output_paths = item.get("output_paths")
        if isinstance(output_paths, list):
            for value in output_paths:
                append(value)
        settings = item.get("settings")
        if not isinstance(settings, dict):
            continue
        controlnet_artifacts = settings.get("controlnet_artifacts")
        if isinstance(controlnet_artifacts, list):
            for artifact in controlnet_artifacts:
                if isinstance(artifact, dict) and artifact.get("processed_image_path"):
                    append(artifact["processed_image_path"])
        derived_outputs = settings.get("derived_outputs")
        if isinstance(derived_outputs, list):
            for output in derived_outputs:
                if not isinstance(output, dict):
                    continue
                for key in ("path", "metadata_path"):
                    if output.get(key):
                        append(output[key])
    return candidates


def _raw_identity(value: Any) -> str:
    return os.path.normcase(str(value).replace("/", os.sep).replace("\\", os.sep))


def _resolved_identity(value: Any) -> tuple[str, str, Path] | None:
    try:
        resolved = Path(value).resolve()
    except _PATH_ACCESS_ERRORS:
        return None
    display = str(resolved)
    return os.path.normcase(display), display, resolved


def _walk_reference_strings(value: Any):
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_reference_strings(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _walk_reference_strings(nested)


def _reference_identities(values: list[Any], candidate_identities: set[str]) -> set[str]:
    references: set[str] = set()
    for value in values:
        for raw_reference in _walk_reference_strings(value):
            resolved = _resolved_identity(raw_reference)
            identity = resolved[0] if resolved else _raw_identity(raw_reference)
            if identity in candidate_identities:
                references.add(identity)
    return references


def _safe_roots(data_dir: Path, items: list[dict]) -> list[Path]:
    roots: list[Path] = []

    def append(value: Any) -> None:
        resolved = _resolved_identity(value)
        if resolved and resolved[2] not in roots:
            roots.append(resolved[2])

    try:
        output_settings = UserSettingsRepository(data_dir / "user-settings.json").load().output
        append(resolve_output_root(data_dir, output_settings))
    except _PATH_ACCESS_ERRORS:
        pass
    for item in items:
        settings = item.get("settings")
        if not isinstance(settings, dict):
            continue
        for key in ("output_directory", "output_root"):
            value = settings.get(key)
            if value:
                append(value)
    return roots


def _normalized_candidates(items: list[dict]) -> tuple[list[Any], list[tuple[str, str, Path] | None]]:
    candidates: list[Any] = []
    resolved_candidates: list[tuple[str, str, Path] | None] = []
    seen_identities: set[str] = set()
    for value in collect_owned_artifact_paths(items):
        resolved = _resolved_identity(value)
        identity = resolved[0] if resolved else _raw_identity(value)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        candidates.append(value)
        resolved_candidates.append(resolved)
    return candidates, resolved_candidates


def skip_generation_artifacts(items: list[dict]) -> ArtifactDeletionResult:
    result = ArtifactDeletionResult()
    try:
        candidates, resolved_candidates = _normalized_candidates(items)
    except Exception:
        logger.exception("Could not classify generation artifacts after file-phase failure")
        return result
    for value, resolved in zip(candidates, resolved_candidates):
        result.skipped_files.append(resolved[1] if resolved else str(value))
    return result


def delete_generation_artifacts(
    items: list[dict],
    data_dir: Path,
    *,
    remaining_items: list[dict] | None,
    lora_datasets: list[Any] | None,
) -> ArtifactDeletionResult:
    result = ArtifactDeletionResult()
    candidates, resolved_candidates = _normalized_candidates(items)
    if remaining_items is None or lora_datasets is None:
        for value, resolved in zip(candidates, resolved_candidates):
            result.skipped_files.append(resolved[1] if resolved else str(value))
        return result

    candidate_identities = {resolved[0] for resolved in resolved_candidates if resolved}
    try:
        references = _reference_identities(
            [*remaining_items, *lora_datasets],
            candidate_identities,
        )
    except Exception:
        logger.exception("Could not build generation artifact reference index")
        for value, resolved in zip(candidates, resolved_candidates):
            result.skipped_files.append(resolved[1] if resolved else str(value))
        return result
    safe_roots = _safe_roots(data_dir, items)

    for value, resolved in zip(candidates, resolved_candidates):
        if resolved is None:
            result.skipped_files.append(str(value))
            continue
        identity, display, path = resolved
        if identity in references:
            result.protected_files.append(display)
            continue
        if not any(path == safe_root or safe_root in path.parents for safe_root in safe_roots):
            result.skipped_files.append(display)
            continue
        try:
            exists = path.exists()
        except Exception:
            logger.exception("Could not check generation artifact existence")
            result.skipped_files.append(display)
            continue
        if not exists:
            result.missing_files.append(display)
            continue
        try:
            is_file = path.is_file()
        except Exception:
            logger.exception("Could not inspect generation artifact")
            result.skipped_files.append(display)
            continue
        if not is_file:
            result.skipped_files.append(display)
            continue
        try:
            path.unlink()
        except Exception:
            logger.exception("Could not delete generation artifact")
            result.skipped_files.append(display)
            continue
        result.deleted_files.append(display)
    return result
