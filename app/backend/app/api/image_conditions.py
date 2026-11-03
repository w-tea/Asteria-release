import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Query, Request
import httpx
from starlette.concurrency import run_in_threadpool

from app.adapters.image_conditions import assess_conditions
from app.core.config import AppConfig
from app.schemas.image_conditions import ConditionCorrection, ConditionRequest
from app.services.history import HistoryRepository
from app.services.image_conditions import ImageConditionRepository, prepare_registered_image, registered_output_path
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix='/api/image-conditions', tags=['image-conditions'])


def context():
    config = AppConfig()
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json').load()
    return ImageConditionRepository(config.data_dir / 'asteria.db'), settings, resolve_output_root(config.data_dir, settings.output)


def history_item(repo, generation_id):
    return HistoryRepository(repo.db_path).get(generation_id)


def prepare(repo, generation_id, path, root):
    try:
        return prepare_registered_image(history_item(repo, generation_id), path, root)
    except (ValueError, OSError):
        raise HTTPException(404, detail={'code': 'condition_image_unavailable'}) from None


@router.get('')
def list_checks(generation_id: int = Query(ge=1), output_path: str = Query(min_length=1, max_length=4096)):
    repo, _, root = context()
    try:
        output_path = registered_output_path(history_item(repo, generation_id), output_path)
    except ValueError:
        pass
    items = repo.list(generation_id, output_path)
    try:
        image = prepare(repo, generation_id, output_path, root)
        digest = image.receipt.source_sha256
    except HTTPException:
        digest = None
    return {'items': [{**item, 'source_current': bool(digest and item['receipt'].get('image', {}).get('source_sha256') == digest)} for item in items]}


@router.post('')
async def check_conditions(body: ConditionRequest, request: Request):
    repo, settings, root = context()
    providers = [item for item in settings.ai.providers if item.id == body.provider_id]
    if len(providers) != 1:
        raise HTTPException(422, detail={'code': 'condition_provider_unavailable'})
    image = await run_in_threadpool(prepare, repo, body.generation_id, body.output_path, root)
    body = body.model_copy(update={'output_path': registered_output_path(history_item(repo, body.generation_id), body.output_path)})
    task = asyncio.create_task(assess_conditions(body, providers[0], image))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.2)
            if await request.is_disconnected():
                raise HTTPException(499, detail={'code': 'condition_cancelled'})
        assessment, receipt = await task
        document = await run_in_threadpool(repo.create, body, assessment, receipt)
        try:
            current = await run_in_threadpool(prepare, repo, body.generation_id, body.output_path, root)
            source_current = current.receipt.source_sha256 == image.receipt.source_sha256
        except HTTPException:
            source_current = False
        return {**document, 'source_current': source_current}
    except (ValueError, httpx.HTTPError, TimeoutError):
        raise HTTPException(502, detail={'code': 'condition_check_failed'}) from None
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, ValueError, httpx.HTTPError, TimeoutError):
            await task


@router.patch('/{check_id}')
def correct_check(check_id: str, body: ConditionCorrection):
    repo, _, root = context()
    try:
        document = repo.get(check_id)
        image = prepare(repo, document['generation_id'], document['output_path'], root)
        if image.receipt.source_sha256 != document['receipt'].get('image', {}).get('source_sha256'):
            raise HTTPException(409, detail={'code': 'condition_image_changed'})
        return {**repo.correct(check_id, body), 'source_current': True}
    except KeyError:
        raise HTTPException(404, detail={'code': 'condition_not_found'}) from None
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from None
