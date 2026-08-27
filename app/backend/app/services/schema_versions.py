from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchemaVersions:
    application: str
    settings: int
    database: int
    generation: int
    merge_recipe: int
    calculation: int
    component_manifest: int
    media: int
    music: int
    video_sequence: int
    extension_manifest: int


CURRENT_SCHEMA_VERSIONS = SchemaVersions(
    application="1.2.0",
    settings=3,
    database=9,
    generation=1,
    merge_recipe=2,
    calculation=2,
    component_manifest=1,
    media=1,
    music=1,
    video_sequence=1,
    extension_manifest=1,
)
