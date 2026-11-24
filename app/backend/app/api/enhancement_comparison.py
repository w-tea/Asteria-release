from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import AppConfig
from app.services.enhancement_comparison import compare_stages
from app.services.history import HistoryRepository
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix='/api/enhancement-comparison', tags=['generation'])


class StageComparisonRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    generation_id: int = Field(ge=1)
    before: str = Field(min_length=1, max_length=4096)
    after: str = Field(min_length=1, max_length=4096)


@router.post('')
def compare(body: StageComparisonRequest):
    config = AppConfig()
    item = HistoryRepository(config.data_dir / 'asteria.db').get(body.generation_id)
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json').load()
    try:
        return compare_stages(item, body.before, body.after, resolve_output_root(config.data_dir, settings.output))
    except (ValueError, OSError, KeyError):
        raise HTTPException(409, detail={'code': 'enhancement_comparison_unavailable'}) from None
