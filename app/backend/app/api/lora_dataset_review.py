import asyncio
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
import httpx
from starlette.concurrency import run_in_threadpool

from app.adapters.lora_dataset_review import assess_dataset_item
from app.core.config import AppConfig
from app.schemas.lora_dataset_review import ReviewSave, ReviewStart
from app.services.lora_dataset_review import DatasetReviewRepository, inspect_dataset, prepare_item
from app.services.lora_training import LoRATrainingDatasetRepository
from app.services.user_settings import UserSettingsRepository

router = APIRouter(prefix='/api/lora-training/datasets/{dataset_id}/review', tags=['lora-training'])


def context():
    config = AppConfig()
    return (LoRATrainingDatasetRepository(config.data_dir / 'lora-training' / 'datasets.json'),
            DatasetReviewRepository(config.data_dir / 'asteria.db'),
            UserSettingsRepository(config.data_dir / 'user-settings.json').load(), config.root_dir)


def dataset_for(datasets, dataset_id):
    dataset = datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(404, detail={'code': 'review_dataset_unavailable'})
    return dataset


def prepare(dataset, item_id, root):
    try:
        return prepare_item(dataset, item_id, root)
    except (ValueError, OSError):
        raise HTTPException(404, detail={'code': 'review_item_unavailable'}) from None


def load(repo, dataset_id, review_id):
    try:
        doc = repo.get(review_id)
    except KeyError:
        raise HTTPException(404, detail={'code': 'review_not_found'}) from None
    if doc['dataset_id'] != dataset_id:
        raise HTTPException(404, detail={'code': 'review_not_found'})
    return doc


def provider_for(settings, provider_id):
    provider = next((x for x in settings.ai.providers if x.id == provider_id and x.enabled and x.model
                     and x.provider_mode in {'openai_compatible', 'anthropic'}), None)
    if provider is None:
        raise HTTPException(422, detail={'code': 'condition_provider_unavailable'})
    return provider.model_copy(deep=True)


def apply_change(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from None


@router.get('')
def overview(dataset_id: str):
    datasets, repo, _, root = context()
    return apply_change(lambda: inspect_dataset(dataset_for(datasets, dataset_id), repo, root))


@router.get('/items/{item_id}/history')
def item_history(dataset_id: str, item_id: str):
    datasets, repo, _, _ = context()
    dataset = dataset_for(datasets, dataset_id)
    if not any(x.id == item_id for x in dataset.items):
        raise HTTPException(404, detail={'code': 'review_item_unavailable'})
    return {'items': repo.history(dataset_id, item_id)}


@router.put('/items/{item_id}')
def confirm(dataset_id: str, item_id: str, body: ReviewSave):
    datasets, repo, _, root = context()
    snapshot, _ = prepare(dataset_for(datasets, dataset_id), item_id, root)
    return apply_change(lambda: repo.confirm(dataset_id, item_id, snapshot, body))


@router.post('/items/{item_id}/start')
def start(dataset_id: str, item_id: str, body: ReviewStart):
    datasets, repo, settings, root = context()
    provider = provider_for(settings, body.provider_id)
    snapshot, _ = prepare(dataset_for(datasets, dataset_id), item_id, root)
    if snapshot['fingerprint'] != body.fingerprint:
        raise HTTPException(409, detail={'code': 'review_source_changed'})
    return apply_change(lambda: repo.start(dataset_id, item_id, snapshot, provider.id, body.previous_id,
                                           language=body.language, provider=provider.model_dump()))


@router.post('/{review_id}/cancel')
def cancel(dataset_id: str, review_id: str):
    _, repo, _, _ = context()
    load(repo, dataset_id, review_id)
    return repo.cancel(review_id)


@router.post('/{review_id}/analyze')
async def analyze(dataset_id: str, review_id: str, request: Request):
    datasets, repo, settings, root = context()
    doc = load(repo, dataset_id, review_id)
    provider = provider_for(settings, doc['provider_id'])
    if provider.model_dump() != doc['provider']:
        raise HTTPException(409, detail={'code': 'review_provider_changed'})
    dataset = dataset_for(datasets, dataset_id)
    snapshot, image = await run_in_threadpool(prepare, dataset, doc['item_id'], root)
    if snapshot['fingerprint'] != doc['snapshot']['fingerprint']:
        raise HTTPException(409, detail={'code': 'review_source_changed'})
    apply_change(lambda: repo.claim(review_id))
    task = asyncio.create_task(assess_dataset_item(provider, image, snapshot['caption'], dataset.character_caption_policy.model_dump(), doc['language']))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.2)
            if await request.is_disconnected() or (await run_in_threadpool(repo.get, review_id))['status'] == 'cancelled':
                raise asyncio.CancelledError()
        annotation, receipt = await task
        current, _ = await run_in_threadpool(prepare, dataset_for(datasets, dataset_id), doc['item_id'], root)
        if current['fingerprint'] != snapshot['fingerprint']:
            await run_in_threadpool(repo.finish, review_id, None, error='source_changed')
            raise HTTPException(409, detail={'code': 'review_source_changed'})
        return await run_in_threadpool(repo.finish, review_id, annotation, receipt)
    except (ValueError, httpx.HTTPError, TimeoutError):
        with suppress(ValueError):
            await run_in_threadpool(repo.finish, review_id, None, error='check_failed')
        raise HTTPException(502, detail={'code': 'review_check_failed'}) from None
    except asyncio.CancelledError:
        await run_in_threadpool(repo.cancel, review_id)
        raise HTTPException(409, detail={'code': 'review_cancelled'}) from None
    except HTTPException:
        with suppress(ValueError):
            await run_in_threadpool(repo.finish, review_id, None, error='source_unavailable')
        raise
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, ValueError, httpx.HTTPError, TimeoutError):
            await task
