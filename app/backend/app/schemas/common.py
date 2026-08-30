from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer, model_validator


class LoraSelection(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=-10, le=10)
    clip_weight: float | None = Field(default=None, ge=-10, le=10)
    block_weight: str = ""
    block_weight_preset: str = ""
    hierarchy_mode: str = "chain"
    stable_asset_id: str | None = None
    fingerprint_algorithm: str | None = None
    fingerprint_version: int | None = Field(default=None, ge=1)
    fingerprint_value: str | None = None
    compatibility_mode: Literal["preserve", "expand"] | None = None

    @model_serializer(mode="wrap")
    def serialize_without_empty_catalog_receipt(self, handler) -> dict[str, Any]:
        payload = handler(self)
        for field in (
            "stable_asset_id",
            "fingerprint_algorithm",
            "fingerprint_version",
            "fingerprint_value",
            "compatibility_mode",
        ):
            if payload.get(field) is None:
                payload.pop(field, None)
        return payload

    @model_validator(mode="after")
    def validate_catalog_receipt(self) -> "LoraSelection":
        receipt = (
            self.stable_asset_id,
            self.fingerprint_algorithm,
            self.fingerprint_version,
            self.fingerprint_value,
        )
        if any(value is not None for value in receipt) and not all(
            value is not None and (not isinstance(value, str) or bool(value.strip()))
            for value in receipt
        ):
            raise ValueError("LoRA catalog receipt must be complete")
        if self.compatibility_mode is not None and self.stable_asset_id is None:
            raise ValueError("LoRA compatibility mode requires a catalog receipt")
        return self
