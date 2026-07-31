from __future__ import annotations

from pydantic import BaseModel

from app.services.schema_versions import CURRENT_SCHEMA_VERSIONS, SchemaVersions


class MaintenanceSchemaVersions(BaseModel):
    application: str = CURRENT_SCHEMA_VERSIONS.application
    settings: int = CURRENT_SCHEMA_VERSIONS.settings
    database: int = CURRENT_SCHEMA_VERSIONS.database
    generation: int = CURRENT_SCHEMA_VERSIONS.generation
    merge_recipe: int = CURRENT_SCHEMA_VERSIONS.merge_recipe
    calculation: int = CURRENT_SCHEMA_VERSIONS.calculation
    component_manifest: int = CURRENT_SCHEMA_VERSIONS.component_manifest

    def as_registry(self) -> SchemaVersions:
        return SchemaVersions(**self.model_dump())


class MaintenanceMigrationResponse(BaseModel):
    from_source_version: str
    versions: MaintenanceSchemaVersions
    applied_steps: list[str]


class MaintenanceBackupSummaryResponse(BaseModel):
    backup_id: str
    created_at: str
    file_count: int
    total_bytes: int
    status: str


class MaintenanceBackupListResponse(BaseModel):
    items: list[MaintenanceBackupSummaryResponse]


class MaintenanceRestoreRequest(BaseModel):
    backup_id: str
    confirmed: bool = False


class MaintenanceRestorePreviewResponse(BaseModel):
    backup_id: str
    created_at: str
    file_count: int
    total_bytes: int
    valid: bool


class MaintenanceStatusResponse(BaseModel):
    schemas: MaintenanceSchemaVersions
    last_operation: dict[str, str | int] | None = None
