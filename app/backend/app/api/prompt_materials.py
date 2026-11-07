from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from app.adapters.image_check_image import MAX_SOURCE_BYTES
from app.adapters.material_vision import propose_material
from app.core.config import AppConfig
from app.schemas.prompt_materials import MaterialKind, PromptMaterialUpdate, PromptMaterialWrite
from app.services.image_assets import ImageAssetError
from app.services.prompt_materials import PromptMaterialRepository
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix='/api/prompt-materials', tags=['prompt-materials'])


def repository():
    return PromptMaterialRepository(AppConfig().data_dir / 'asteria.db')


@router.get('')
def list_materials():
    return {'items': repository().list()}


@router.post('')
def create_material(request: PromptMaterialWrite):
    return repository().create(request)


@router.put('/{material_id}')
def update_material(material_id: str, request: PromptMaterialUpdate):
    try:
        return repository().update(material_id, PromptMaterialWrite.model_validate(request.model_dump(exclude={'revision'})), request.revision)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404 if isinstance(exc, KeyError) else 409, detail={'code': exc.args[0]}) from exc


@router.delete('/{material_id}')
def delete_material(material_id: str, revision: int = Query(ge=1)):
    try:
        repository().delete(material_id, revision)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404 if isinstance(exc, KeyError) else 409, detail={'code': exc.args[0]}) from exc
    return {'deleted': True}


@router.post('/propose-image')
async def propose_from_image(request: Request, provider_id: str = Query(min_length=1, max_length=128), kind: MaterialKind = Query(), filename: str = Query(max_length=255), language: Literal['en', 'ja'] = Query('ja')):
    config = AppConfig()
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json').load()
    provider = next((item for item in settings.ai.providers if item.id == provider_id), None)
    if provider is None:
        raise HTTPException(422, detail={'code': 'material_vision_provider_unavailable'})
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_SOURCE_BYTES:
            raise HTTPException(413, detail={'code': 'material_image_too_large'})
        content.extend(chunk)
    try:
        return await run_in_threadpool(propose_material, bytes(content), filename, kind=kind, provider=provider,
                                       output_root=resolve_output_root(config.data_dir, settings.output).resolve(), language=language)
    except (ValueError, OSError, ImageAssetError) as exc:
        raise HTTPException(502, detail={'code': 'material_image_proposal_failed'}) from exc
