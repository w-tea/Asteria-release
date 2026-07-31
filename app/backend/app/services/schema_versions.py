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


CURRENT_SCHEMA_VERSIONS = SchemaVersions(
    application="1.1.0",
    settings=3,
    database=6,
    generation=1,
    merge_recipe=2,
    calculation=1,
    component_manifest=1,
)
