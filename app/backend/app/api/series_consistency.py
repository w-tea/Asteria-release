import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
import httpx
from starlette.concurrency import run_in_threadpool

from app.adapters.image_conditions import assess_conditions
from app.core.config import AppConfig
from app.schemas.image_conditions import ConditionCorrection, ConditionRequest
from app.schemas.series_consistency import SeriesRequest
from app.services.history import HistoryRepository
from app.services.image_conditions import prepare_registered_image, registered_output_path
from app.services.series_consistency import SeriesRepository
from app.services.user_settings import UserSettingsRepository, resolve_output_root

router = APIRouter(prefix='/api/series-checks', tags=['series-checks'])


def context():
    config = AppConfig()
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json').load()
    return SeriesRepository(config.data_dir / 'asteria.db'), settings, resolve_output_root(config.data_dir, settings.output)


def history_item(repo, generation_id):
    return HistoryRepository(repo.db_path).get(generation_id)


def prepare(repo, image, root):
    try:
        return prepare_registered_image(history_item(repo, image['generation_id']), image['output_path'], root)
    except (ValueError, OSError):
        raise HTTPException(404, detail={'code': 'condition_image_unavailable'}) from None


def load(repo, run_id):
    try:
        return repo.get(run_id)
    except KeyError:
        raise HTTPException(404, detail={'code': 'series_not_found'}) from None


def annotate_sources(repo, document, root):
    def current(image):
        try:
            return prepare(repo, image, root).receipt.source_sha256 == image['source_sha256']
        except HTTPException:
            return False
    reference_current = current(document['reference'])
    return {**document, 'reference_current': reference_current, 'members': [
        {**item, 'source_current': reference_current and current(item)} for item in document['members']]}


def provider_for(settings, provider_id):
    values = [p for p in settings.ai.providers if p.id == provider_id and p.enabled and p.model
              and p.provider_mode in {'openai_compatible', 'anthropic'}]
    if len(values) != 1:
        raise HTTPException(422, detail={'code': 'condition_provider_unavailable'})
    return values[0].model_copy(deep=True)


@router.get('')
def list_checks():
    repo, _, _ = context()
    # List summaries without decoding every image in every old run.
    return {'items': [{key: item[key] for key in ['id', 'created_at', 'reference', 'provider_id']}
                      | {'count': len(item['members'])} for item in repo.list()]}


@router.get('/{run_id}')
def get_check(run_id: str):
    repo, _, root = context()
    return annotate_sources(repo, load(repo, run_id), root)


@router.post('')
def create_check(body: SeriesRequest):
    repo, settings, root = context()
    provider = provider_for(settings, body.provider_id)
    hashes = []
    for image in [body.reference, *body.candidates]:
        prepared = prepare(repo, image.model_dump(), root)
        image.output_path = registered_output_path(history_item(repo, image.generation_id), image.output_path)
        hashes.append(prepared.receipt.source_sha256)
    return annotate_sources(repo, repo.create(body, hashes, provider.model_dump()), root)


@router.post('/{run_id}/cancel')
def cancel_check(run_id: str):
    repo, _, _ = context()
    load(repo, run_id)
    repo.cancel(run_id)
    return {'cancelled': True}


@router.post('/{run_id}/members/{index}')
async def check_member(run_id: str, index: int, request: Request):
    repo, settings, root = context()
    document = load(repo, run_id)
    if not 0 <= index < len(document['members']):
        raise HTTPException(404, detail={'code': 'series_member_unavailable'})
    provider = provider_for(settings, document['provider_id'])
    if provider.model_dump() != document['provider']:
        raise HTTPException(409, detail={'code': 'series_provider_changed'})
    member = document['members'][index]
    reference = await run_in_threadpool(prepare, repo, document['reference'], root)
    image = await run_in_threadpool(prepare, repo, member, root)
    if reference.receipt.source_sha256 != document['reference']['source_sha256'] or image.receipt.source_sha256 != member['source_sha256']:
        raise HTTPException(409, detail={'code': 'condition_image_changed'})
    try:
        await run_in_threadpool(repo.claim, run_id, index)
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from None
    query = ConditionRequest(generation_id=member['generation_id'], output_path=member['output_path'],
                             provider_id=provider.id, criteria=document['criteria'], language=document['language'])
    task = asyncio.create_task(assess_conditions(query, provider, image, reference=reference))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.2)
            if await request.is_disconnected() or (await run_in_threadpool(repo.get, run_id)).get('cancelled_at'):
                raise asyncio.CancelledError()
        assessment, receipt = await task
        document = await run_in_threadpool(repo.finish, run_id, index, assessment, receipt)
        return await run_in_threadpool(annotate_sources, repo, document, root)
    except (ValueError, httpx.HTTPError, TimeoutError):
        await run_in_threadpool(repo.finish, run_id, index, error='check_failed')
        raise HTTPException(502, detail={'code': 'condition_check_failed'}) from None
    except asyncio.CancelledError:
        await run_in_threadpool(repo.finish, run_id, index, error='cancelled')
        if (await run_in_threadpool(repo.get, run_id)).get('cancelled_at'):
            raise HTTPException(409, detail={'code': 'series_cancelled'}) from None
        raise
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, ValueError, httpx.HTTPError, TimeoutError):
            await task


@router.patch('/{run_id}/members/{index}')
def correct_member(run_id: str, index: int, body: ConditionCorrection):
    repo, _, root = context()
    document = annotate_sources(repo, load(repo, run_id), root)
    if not 0 <= index < len(document['members']) or not document['members'][index]['source_current']:
        raise HTTPException(409, detail={'code': 'condition_image_changed'})
    try:
        return annotate_sources(repo, repo.correct(run_id, index, body), root)
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from None
