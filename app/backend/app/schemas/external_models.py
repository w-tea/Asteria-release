from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ExternalModelLayout = Literal["asteria_comfy", "forge_a1111", "custom"]
ExternalModelRootStatus = Literal["online", "offline", "disabled", "conflict"]


class ExternalModelRootCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1)
    layout: ExternalModelLayout
    mappings: dict[str, list[str]] | None = None
    priority: int = Field(default=0, ge=-1000, le=1000)

    @field_validator("display_name", "path")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value is required")
        return stripped


class ExternalModelRootUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=-1000, le=1000)
    layout: ExternalModelLayout | None = None
    mappings: dict[str, list[str]] | None = None


class ExternalModelRootRebind(BaseModel):
    path: str = Field(min_length=1)


class ExternalModelRootPreviewRequest(BaseModel):
    path: str = Field(min_length=1)
    exclude_root_id: str | None = None


class ExternalModelRootPreviewResponse(BaseModel):
    path: str
    path_hint: str
    suggested_name: str
    detected_layout: Literal["asteria_comfy", "forge_a1111"] | None
    counts: dict[str, int]
    total_items: int
    truncated: bool


class ExternalModelOwnedLinkRequest(BaseModel):
    model_type: str = Field(min_length=1, max_length=40)
    link_name: str = Field(min_length=1, max_length=120)


class ExternalModelRootResponse(BaseModel):
    root_id: str
    display_name: str
    layout: ExternalModelLayout
    mappings: dict[str, list[str]]
    priority: int
    enabled: bool
    status: ExternalModelRootStatus
    path_hint: str
    revision: int


class ExternalModelLibraryItemResponse(BaseModel):
    stable_model_id: str
    root_id: str
    model_type: str
    logical_name: str
    normalized_relative_path: str
    size_bytes: int
    fingerprint_algorithm: str
    fingerprint_version: int
    fingerprint_value: str


class ExternalModelCollisionResponse(BaseModel):
    model_type: str
    logical_name: str
    winner_root_id: str
    shadowed_root_ids: list[str]


class ExternalModelRegistryResponse(BaseModel):
    roots: list[ExternalModelRootResponse]
    items: list[ExternalModelLibraryItemResponse] = Field(default_factory=list)
    collisions: list[ExternalModelCollisionResponse] = Field(default_factory=list)
    truncated: bool = False
