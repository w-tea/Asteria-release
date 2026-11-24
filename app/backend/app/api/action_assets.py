"""Binary transfers for GUI and authenticated MCP clients, not LLM args."""
from hashlib import sha256
import logging
from typing import Literal

import anyio
from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas.action_contracts import ActionResult, ImageUploadMetadata
from app.services.action_images import ActionImageRepository
from app.services.action_media_sources import MediaSourceUpload
from app.services.action_receipts import ActionRequestConflict
from app.services.image_validation import DEFAULT_MAX_IMAGE_BYTES, InvalidImageContentError
from app.services.media_probe import MediaProbeError
from app.services.resource_lifecycle import resource_admission, ResourceAdmissionTimeoutError
from starlette.requests import ClientDisconnect


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/images', response_model=ActionResult)
async def upload_image(request: Request, filename: str,
                       role: Literal['init', 'mask', 'control'] = 'init',
                       request_id: str = Query(min_length=1)) -> ActionResult:
    # Use the same raw-body upload and byte validation as the existing editor.
    # The binary does not enter tool arguments, receipts or conversation text.
    data = await request.body()
    metadata = ImageUploadMetadata(filename=filename, role=role)

    def save():
        return ActionImageRepository(request.app.state.config.data_dir).save(metadata, request_id, data)

    try:
        record, replayed = await anyio.to_thread.run_sync(save)
    except ActionRequestConflict as exc:
        raise HTTPException(409, detail={'code': 'action_request_conflict'}) from exc
    return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)


@router.post('/media', response_model=ActionResult)
async def upload_media(request: Request, filename: str,
                       request_id: str = Query(min_length=1)) -> ActionResult:
    data_dir = request.app.state.config.data_dir
    stage = None
    try:
        stage = await anyio.to_thread.run_sync(MediaSourceUpload.create, data_dir, filename)
        digest = sha256()
        size = 0
        async with await anyio.open_file(stage.path, 'xb') as file:
            async for chunk in request.stream():
                size += len(chunk)
                if stage.kind == 'image' and size > DEFAULT_MAX_IMAGE_BYTES:
                    raise HTTPException(413, {'code': 'assistant_media_image_too_large'})
                digest.update(chunk)
                await file.write(chunk)
        def register():
            with resource_admission.acquire((), operation_id='media.import', timeout_seconds=0):
                return stage.register(data_dir, request_id, digest.hexdigest())

        record, replayed = await anyio.to_thread.run_sync(register)
    except ActionRequestConflict as exc:
        raise HTTPException(409, detail={'code': 'action_request_conflict'}) from exc
    except (InvalidImageContentError, MediaProbeError) as exc:
        raise HTTPException(422, {'code': 'assistant_media_invalid'}) from exc
    except ClientDisconnect as exc:
        raise HTTPException(400, {'code': 'assistant_media_upload_interrupted'}) from exc
    except ResourceAdmissionTimeoutError as exc:
        raise HTTPException(409, {'code': 'maintenance_restore_busy'}) from exc
    except OSError as exc:
        logger.warning('Could not stage a media upload', exc_info=True)
        raise HTTPException(409, {'code': 'assistant_media_import_failed'}) from exc
    finally:
        if stage is not None:
            with anyio.CancelScope(shield=True):
                try:
                    await anyio.to_thread.run_sync(stage.cleanup)
                except OSError:
                    logger.warning('Could not remove an unregistered media upload', exc_info=True)
    return ActionResult(data=record.model_dump(mode='json'), replayed=replayed)
