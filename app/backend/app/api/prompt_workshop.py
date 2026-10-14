from fastapi import APIRouter, HTTPException

from app.core.config import AppConfig
from app.schemas.prompt_workshop import SceneComposeRequest, SemanticReviewRequest
from app.services.prompt_workshop import compose_scene, review_semantics
from app.services.user_settings import UserSettingsRepository

router = APIRouter(prefix='/api/prompt-workshop', tags=['prompt-workshop'])


def _repository():
    return UserSettingsRepository(AppConfig().data_dir / 'user-settings.json')


def _run(request, callback):
    try:
        return callback(request, _repository().load().ai.providers)
    except ValueError as exc:
        code = str(exc).split(':', 1)[0]
        if not code.startswith(('prompt_workshop_', 'prompt_assist_provider_')):
            code = 'prompt_workshop_response_invalid'
        status = 422 if code == 'prompt_workshop_provider_unavailable' or code.endswith('_required') else 502
        raise HTTPException(status, detail={'code': code}) from exc


@router.post('/compose')
def compose(request: SceneComposeRequest):
    return _run(request, compose_scene)


@router.post('/review')
def review(request: SemanticReviewRequest):
    return _run(request, review_semantics)
