"""GUI adapter for the same core repeat service used by Assistant and MCP."""
from threading import RLock

from fastapi import APIRouter, HTTPException

from app.api import generation
from app.core.config import AppConfig
from app.schemas.action_contracts import ActionResult, GenerationDraftDocument
from app.schemas.generation_repeat import RepeatInputUpdate, RepeatStopRequest, RepeatSubmit, GenerationRepeatRecord
from app.services.action_receipts import ActionRequestConflict
from app.services.generation_repeat import GenerationRepeatService
from app.services.operation_repository import OperationConflictError, OperationNotFoundError

router = APIRouter(prefix='/api/generations/repeats', tags=['generation'])
_services: dict[str, GenerationRepeatService] = {}
_lock = RLock()


def get_repeat_service() -> GenerationRepeatService:
    coordinator = generation._generation_operation_coordinator()
    key = str(coordinator.repository.db_path.resolve())
    with _lock:
        service = _services.get(key)
        if service is None or service.coordinator is not coordinator or service.closed:
            if service is not None:
                service.close()
            def enqueue(document, parent_id, request_id, on_accept):
                job, _ = generation.enqueue_action_generation(document.mode, document.input,
                    request_id=request_id, arguments={'parent_operation_id': parent_id, 'document': document.model_dump(mode='json')},
                    adopted_draft={}, parent_operation_id=parent_id, on_accept=on_accept)
                return job.id
            service = GenerationRepeatService(coordinator, enqueue=enqueue,
                get_job=lambda identity: generation.get_job(identity).model_dump(mode='json'),
                cancel_job=generation.cancel_job)
            _services[key] = service
        return service


def prepare_repeat_input(document: GenerationDraftDocument) -> GenerationDraftDocument:
    request, mode = document.input, document.mode
    if mode == 'img2img':
        generation._ensure_supported_img2img_profile(request.profile)
    elif mode == 'inpaint':
        generation._ensure_supported_inpaint_request(request)
    generation._preflight_controlnet_mode_request(request, mode)
    selected = generation._model_slots_for_request(request, AppConfig())
    return document.model_copy(update={'input': request.model_copy(update={'model_slot_overrides': {
        **selected, **request.model_slot_overrides}})})


def _error(exc):
    if isinstance(exc, OperationNotFoundError):
        return HTTPException(404, detail={'code': str(exc)})
    return HTTPException(409, detail={'code': str(exc)})


@router.post('', response_model=ActionResult)
def start_repeat(request: RepeatSubmit):
    service = get_repeat_service()
    arguments = request.model_dump(mode='json', exclude={'request_id'})
    try:
        accepted = service.coordinator.repository.get_action_operation(request.request_id, 'generation.repeats.start', arguments)
        if accepted is not None:
            return ActionResult(data=service.get(accepted.operation_id).model_dump(mode='json'), replayed=True)
        record, replayed = service.start(request.request_id, prepare_repeat_input(request.document), request.limits,
                                          identity_arguments=arguments)
        return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)
    except (OperationConflictError, OperationNotFoundError, ActionRequestConflict, ValueError) as exc:
        raise _error(exc) from exc


@router.get('/by-request/{request_id}', response_model=GenerationRepeatRecord)
def repeat_by_request(request_id: str):
    try:
        return get_repeat_service().by_request(request_id)
    except OperationNotFoundError as exc:
        raise _error(exc) from exc


@router.get('/{repeat_id}', response_model=GenerationRepeatRecord)
def get_repeat(repeat_id: str):
    try:
        return get_repeat_service().get(repeat_id)
    except OperationNotFoundError as exc:
        raise _error(exc) from exc


@router.put('/{repeat_id}', response_model=ActionResult)
def update_repeat(repeat_id: str, request: RepeatInputUpdate):
    try:
        service = get_repeat_service()
        arguments = {'repeat_id': repeat_id, **request.model_dump(mode='json', exclude={'request_id'})}
        accepted = service.update_receipt(request.request_id, arguments)
        if accepted is not None:
            return ActionResult(data=accepted.model_dump(mode='json'), replayed=True)
        record, replayed = service.update(request.request_id, repeat_id, request.expected_revision,
            prepare_repeat_input(request.document), identity_arguments=arguments)
        return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)
    except (OperationConflictError, OperationNotFoundError, ActionRequestConflict) as exc:
        raise _error(exc) from exc


@router.post('/{repeat_id}/stop', response_model=GenerationRepeatRecord)
def stop_repeat(repeat_id: str, request: RepeatStopRequest):
    return stop_repeat_action(repeat_id, request).data


def stop_repeat_action(repeat_id: str, request: RepeatStopRequest) -> ActionResult:
    try:
        service = get_repeat_service()
        # Bind the target first, then reapply stop even after a lost response.
        service.get(repeat_id)
        _, created = service.coordinator.repository.bind_action_operation(request.request_id, 'generation.repeats.stop',
            {'repeat_id': repeat_id, 'stop_current': request.stop_current}, repeat_id)
        record = service.stop(repeat_id, stop_current=request.stop_current)
        return ActionResult(data=record.model_dump(mode='json'), replayed=not created)
    except (OperationConflictError, OperationNotFoundError, ActionRequestConflict) as exc:
        raise _error(exc) from exc
