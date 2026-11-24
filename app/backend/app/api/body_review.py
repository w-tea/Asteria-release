from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from app.core.config import AppConfig
from app.schemas.body_review import ReviewRequest, ReviewSelection, ReviewEdits
from app.services.body_review import BodyReviewService, mask_path, mask_union_png
from app.services.user_settings import resolve_current_output_root

router = APIRouter(prefix='/api/body-review', tags=['body-review'])


def service():
    config = AppConfig()
    return BodyReviewService(config.data_dir, config.model_dir)


def source_path(value: str) -> Path:
    from app.api.assets import _is_registered_output_path, _is_asteria_init_image_path
    path = Path(value).resolve(strict=True)
    if not path.is_file() or not (_is_registered_output_path(path) or _is_asteria_init_image_path(path)):
        raise ValueError('body_review_source_unavailable')
    return path


def failed(exc):
    code = str(exc) if str(exc).startswith('body_review_') else 'body_review_unavailable'
    return HTTPException(409, detail={'code': code})


@router.get('/capabilities')
def capabilities():
    return service().capabilities()


@router.get('/latest')
def latest(path: str):
    try:
        return {'review': service().latest(source_path(path))}
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.post('/detect')
def detect(request: ReviewRequest):
    try:
        adapter = service()
        if not adapter.capabilities()['detector']:
            raise ValueError('body_review_model_unavailable')
        root = resolve_current_output_root(AppConfig().data_dir)
        row = adapter.create(source_path(request.source_path), root, request.operation_id)
        row['parts'] = request.parts
        return adapter.run(row, 'detect', request.operation_id)
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.post('/cancel/{operation_id}')
def cancel(operation_id: str):
    service().cancel(operation_id)
    return {'status': 'cancel_requested'}


@router.post('/{review_id}/selection')
def save_selection(review_id: str, edits: ReviewEdits):
    try:
        return service().save_selection(review_id, edits)
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.post('/{review_id}/masks')
def masks(review_id: str, selection: ReviewSelection):
    try:
        adapter = service()
        row = adapter.checked(review_id)
        allowed = {value['id'] for value in row['candidates']}
        if len({value.id for value in selection.candidates}) != len(selection.candidates) or any(
                value.id not in allowed for value in selection.candidates) or not selection.operation_id:
            raise ValueError('body_review_selection_invalid')
        return adapter.run(row, 'mask', selection.operation_id,
                           [value.model_dump() for value in selection.candidates])
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.get('/{review_id}/source')
def review_source(review_id: str):
    try:
        row = service().checked(review_id)
        return FileResponse(Path(row['directory']) / 'source.png', headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.get('/{review_id}/mask')
def review_mask(review_id: str, file: str):
    try:
        row = service().checked(review_id)
        return FileResponse(mask_path(row, file), headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc


@router.post('/{review_id}/mask-union')
def mask_union(review_id: str, selection: ReviewSelection):
    try:
        row = service().checked(review_id)
        data = mask_union_png(row, [value.model_dump() for value in selection.candidates])
        return Response(data, media_type='image/png', headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError, KeyError) as exc:
        raise failed(exc) from exc
