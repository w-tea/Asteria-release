from pathlib import Path
from typing import Any

import yaml


class ParameterMapper:
    def __init__(self, schema_path: Path):
        try:
            schema = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raise ValueError("Generation schema could not be loaded") from None
        if not isinstance(schema, dict):
            raise ValueError("Generation schema must be an object")
        if not isinstance(schema.get("fields"), dict):
            raise ValueError("Generation schema fields must be an object")
        schema["fields"] = self._valid_fields(schema["fields"])
        self.schema = schema

    def map(self, values: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field_id, spec in self.schema["fields"].items():
            if field_id in values:
                value = values[field_id]
            elif "default" in spec:
                value = spec["default"]
            elif spec.get("required"):
                raise ValueError(f"Missing required field: {field_id}")
            else:
                continue

            self._validate(field_id, value, spec)
            backend_key = spec.get("backend_key", field_id)
            output[backend_key] = value
        return output

    def _valid_fields(self, fields: dict[Any, Any]) -> dict[str, dict[str, Any]]:
        valid: dict[str, dict[str, Any]] = {}
        for field_id, spec in fields.items():
            if not isinstance(field_id, str) or not isinstance(spec, dict):
                continue
            if "backend_key" in spec and not isinstance(spec.get("backend_key"), str):
                continue
            if "required" in spec and not isinstance(spec.get("required"), bool):
                continue
            valid[field_id] = spec
        return valid

    def _validate(self, field_id: str, value: Any, spec: dict[str, Any]) -> None:
        kind = spec.get("type")
        if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{field_id} must be integer")
        if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{field_id} must be number")
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{field_id} must be string")
        if kind == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{field_id} must be boolean")
        if kind == "enum" and "values" in spec and value not in spec["values"]:
            raise ValueError(f"{field_id} must be one of {spec['values']}")
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"{field_id} is below min")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"{field_id} is above max")
