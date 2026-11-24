"""Library integrations use saved image IDs, not paths or connection details."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.action_contracts import MediaSourceTarget
from app.schemas.operations import OperationStatus
from app.services.topaz import TopazSettings


class LibraryEagleSend(MediaSourceTarget):
    folder_id: str | None = None


class LibraryEagleResult(MediaSourceTarget):
    status: Literal['success', 'unconfirmed']
    folder_id: str | None = None
    folder_applied: bool | None = None
    error_code: str | None = None


class LibraryTopazSettings(TopazSettings):
    model_config = ConfigDict(extra='forbid')
    enabled: bool = True


class LibraryTopazStart(MediaSourceTarget):
    settings: LibraryTopazSettings = Field(default_factory=LibraryTopazSettings)


class LibraryTopazJob(BaseModel):
    job_id: str
    status: OperationStatus
    source: MediaSourceTarget
    output: MediaSourceTarget | None = None
    error_code: str | None = None
    warning_codes: tuple[str, ...] = ()
