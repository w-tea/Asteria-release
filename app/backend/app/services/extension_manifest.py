from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.extensions import ASTERIA_EXTENSION_SCHEMA_ID, ExtensionManifest, compare_semver


ASTERIA_VERSION = "1.2.0"


class ExtensionManifestError(ValueError):
    def __init__(self, message: str, *, code: str = "extension_manifest_invalid") -> None:
        super().__init__(message)
        self.code = code


def validate_extension_manifest(
    payload: Mapping[str, Any],
    *,
    asteria_version: str = ASTERIA_VERSION,
) -> ExtensionManifest:
    # Validate with JSON input semantics so strict frozen tuple fields accept
    # JSON arrays without permitting Python-side coercion of scalar values.
    manifest = ExtensionManifest.model_validate_json(json.dumps(payload))
    if (
        compare_semver(asteria_version, manifest.asteria.minimum) < 0
        or compare_semver(asteria_version, manifest.asteria.maximum_exclusive) >= 0
    ):
        raise ExtensionManifestError(
            f"extension is not compatible with Asteria {asteria_version}",
            code="extension_manifest_incompatible",
        )
    return manifest


def load_extension_manifest(
    path: Path,
    *,
    asteria_version: str = ASTERIA_VERSION,
) -> ExtensionManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtensionManifestError("extension manifest could not be read") from exc
    if not isinstance(payload, Mapping):
        raise ExtensionManifestError("extension manifest is invalid")
    try:
        return validate_extension_manifest(payload, asteria_version=asteria_version)
    except ExtensionManifestError:
        raise
    except ValidationError as exc:
        raise ExtensionManifestError("extension manifest is invalid") from exc


def extension_manifest_json_schema() -> dict[str, Any]:
    schema = ExtensionManifest.model_json_schema()
    return {"$id": ASTERIA_EXTENSION_SCHEMA_ID, **schema}
