from uuid import uuid4
from threading import Lock

from fastapi import APIRouter, HTTPException

from app.api import generation
from app.core.config import AppConfig
from app.schemas.prompt_comparison import PromptComparisonRating, PromptComparisonRequest, REQUEST_TYPES
from app.services.operation_repository import OperationNotFoundError
from app.services.history import HistoryRepository
from app.services.prompt_comparison import prepare_comparison
from app.services.prompt_comparison_repository import PromptComparisonRepository

router = APIRouter(prefix='/api/prompt-comparisons', tags=['prompt-comparisons'])
# The launcher owns one backend process. Readers must not observe a claim before
# the matching operation has been registered in the second repository.
_scheduling_lock = Lock()


def repository() -> PromptComparisonRepository:
    return PromptComparisonRepository(generation._data_dir() / 'asteria.db')


def _get(comparison_id: str) -> dict:
    with _scheduling_lock:
        return _get_registered(comparison_id)


def _get_registered(comparison_id: str) -> dict:
    try:
        plan = repository().get(comparison_id)
    except KeyError as exc:
        raise HTTPException(404, detail={'code': 'comparison_not_found'}) from exc
    if plan['operation_id'] and plan['status'] in {'queued', 'running'}:
        try:
            operation = generation._generation_operation_coordinator().get(plan['operation_id'])
        except OperationNotFoundError:
            return repository().set_status(comparison_id, 'interrupted', 'comparison_interrupted')
        if operation.status in {'cancelled', 'interrupted', 'failed'}:
            plan = repository().set_status(comparison_id, operation.status, operation.error_code or '')
        plan['progress'] = {'current': operation.progress_current, 'total': operation.progress_total, 'phase': operation.phase}
    return plan


@router.get('')
def list_comparisons():
    return {'items': repository().list_recent()}


@router.post('/prepare')
def prepare(request: PromptComparisonRequest):
    config = AppConfig()
    base = request.base.model_copy(deep=True)
    base.model_slot_overrides = generation._model_slots_for_request(base, config)
    base.runtime = generation._selected_runtime_id(config, base.runtime)
    request = request.model_copy(update={'base': base})
    try:
        plan = prepare_comparison(request, wildcard_dir=config.wildcard_dir)
    except ValueError as exc:
        raise HTTPException(422, detail={'code': str(exc)}) from exc
    return repository().create(plan)


@router.get('/{comparison_id}')
def get_comparison(comparison_id: str):
    return _get(comparison_id)


@router.post('/{comparison_id}/start')
def start(comparison_id: str):
    plan = _get(comparison_id)
    if plan['status'] != 'prepared':
        raise HTTPException(409, detail={'code': 'comparison_already_started'})
    if plan.get('lora_test'):
        from app.api.lora_training import _run_repository
        from app.services.lora_practical_test import verify_practical_sources
        try:
            verify_practical_sources(plan, _run_repository().get(plan['lora_test']['run_id']), AppConfig().root_dir)
        except ValueError as exc:
            raise HTTPException(409, detail={'code': str(exc)}) from None
    for cell in plan['cells']:
        request = REQUEST_TYPES[plan['mode']].model_validate(cell['request'])
        if plan['mode'] == 'img2img':
            generation._ensure_supported_img2img_profile(request.profile)
        elif plan['mode'] == 'inpaint':
            generation._ensure_supported_inpaint_request(request)
        generation._preflight_profile_prompt_request(request)
        generation._preflight_controlnet_mode_request(request, plan['mode'])
    snapshot = generation._generation_operation_snapshot(request)
    operation_id = uuid4().hex
    with _scheduling_lock:
        try:
            repository().claim_start(comparison_id, operation_id)
        except ValueError as exc:
            raise HTTPException(409, detail={'code': str(exc)}) from exc
        try:
            generation._generation_operation_coordinator().enqueue(
                kind='generation.prompt_comparison', source_actor='local', operation_id=operation_id,
                request_snapshot={'comparison_id': comparison_id, 'mode': plan['mode'], 'generation_operation_snapshot': snapshot},
                progress_total=len(plan['cells']), progress_unit='images',
            )
        except Exception:
            repository().set_status(comparison_id, 'failed', 'comparison_schedule_failed')
            raise
    return _get(comparison_id)


@router.post('/{comparison_id}/cancel')
def cancel(comparison_id: str):
    plan = _get(comparison_id)
    if plan['operation_id']:
        generation.cancel_generation_operation(plan['operation_id'])
    return _get(comparison_id)


@router.put('/{comparison_id}/rating')
def rate(comparison_id: str, rating: PromptComparisonRating):
    _get(comparison_id)
    try:
        return repository().rate(comparison_id, rating)
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from exc


@router.get('/{comparison_id}/pairs/{pair}')
def pair_history(comparison_id: str, pair: int):
    plan = _get(comparison_id)
    if pair < 0 or pair >= len(plan['seeds']):
        raise HTTPException(404, detail={'code': 'comparison_pair_incomplete'})
    cells = [next(cell for cell in plan['cells'] if cell['pair'] == pair and cell['arm'] == arm) for arm in plan['display_orders'][pair]]
    if any(cell['status'] != 'completed' for cell in cells):
        raise HTTPException(409, detail={'code': 'comparison_pair_incomplete'})
    history = HistoryRepository(generation._data_dir() / 'asteria.db')
    items = [history.get(cell['generation_id']) for cell in cells]
    if any(item is None for item in items):
        raise HTTPException(404, detail={'code': 'comparison_item_missing'})
    return {'items': items}
