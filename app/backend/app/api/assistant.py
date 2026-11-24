"""Chat persistence and resumable event delivery, not a new execution engine."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from app.schemas.assistant import AssistantDraftAttach, AssistantTurnCreate, ConversationCreate, ConversationUpdate
from app.services.assistant_drafts import DraftConflict, DraftNotFound
from app.services.action_receipts import ActionRequestConflict
from app.services.assistant_exports import conversation_markdown, project_conversation
from app.services.assistant_sessions import ConversationAnswerActive, ConversationConflict, ConversationNotFound


router = APIRouter(prefix='/api/assistant', tags=['assistant'])


def runtime(request: Request):
    return request.app.state.assistant_runtime


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ConversationNotFound as exc:
        raise HTTPException(404, {'code': 'assistant_record_not_found'}) from exc
    except ConversationConflict as exc:
        raise HTTPException(409, {'code': str(exc), 'current': exc.current.model_dump()}) from exc
    except ConversationAnswerActive as exc:
        raise HTTPException(409, {'code': 'assistant_conversation_answer_active'}) from exc
    except ActionRequestConflict as exc:
        raise HTTPException(409, {'code': 'action_request_conflict'}) from exc
    except DraftNotFound as exc:
        raise HTTPException(404, {'code': 'assistant_draft_not_found'}) from exc
    except DraftConflict as exc:
        raise HTTPException(409, {'code': exc.code, 'current': exc.current.model_dump() if exc.current else None}) from exc


@router.get('/providers')
def providers(request: Request):
    return {'items': [{'id': provider.id, 'name': provider.name, 'model': provider.model,
                       'provider_mode': provider.provider_mode, 'configured': bool(provider.endpoint and provider.model)}
                      for provider in runtime(request).providers() if provider.enabled]}


@router.get('/conversations')
def conversations(request: Request, query: str = ''):
    return {'items': runtime(request).repository.list(query)}


@router.post('/conversations')
def create_conversation(body: ConversationCreate, request: Request):
    with runtime(request).write_lease():
        conversation, replayed = _call(runtime(request).repository.create, **body.model_dump())
    return {'conversation': conversation, 'replayed': replayed}


@router.get('/conversations/{conversation_id}')
def get_conversation(conversation_id: str, request: Request):
    return _call(runtime(request).conversation, conversation_id)


@router.put('/conversations/{conversation_id}')
def update_conversation(conversation_id: str, body: ConversationUpdate, request: Request):
    with runtime(request).write_lease():
        return _call(runtime(request).repository.update, conversation_id, **body.model_dump())


@router.delete('/conversations/{conversation_id}', status_code=204)
def delete_conversation(conversation_id: str, request: Request, revision: int = Query(ge=1)):
    current = runtime(request)
    with current.write_lease():
        _call(current.repository.delete, conversation_id, revision, stream_pending=current.stream_pending)
    return Response(status_code=204)


@router.get('/conversations/{conversation_id}/export')
def export_conversation(conversation_id: str, request: Request, format: Literal['json', 'markdown'] = 'json'):
    record = _call(runtime(request).repository.export, conversation_id)
    document = project_conversation(record)
    suffix = 'md' if format == 'markdown' else 'json'
    headers = {'Cache-Control': 'no-store', 'Content-Disposition': f'attachment; filename="asteria-chat-{conversation_id}.{suffix}"'}
    if format == 'markdown':
        return Response(conversation_markdown(document), media_type='text/markdown', headers=headers)
    return JSONResponse(document, headers=headers)


@router.get('/attachments/images/{asset_id}/download')
def download_image_attachment(asset_id: str, request: Request):
    record, content = runtime(request).images.read_bytes(asset_id)
    filename = re.sub(r'^[a-f0-9]{32}_', '', record.filename)
    return Response(content, media_type=record.mime_type, headers={
        'Cache-Control': 'no-store',
        'Content-Disposition': f"attachment; filename*=UTF-8''{quote(filename, safe='')}",
        'X-Content-Type-Options': 'nosniff',
    })


@router.post('/conversations/{conversation_id}/turns', status_code=202)
async def submit_turn(conversation_id: str, body: AssistantTurnCreate, request: Request):
    turn, replayed = _call(runtime(request).submit, conversation_id, body)
    return {'turn': {**turn.model_dump(), 'stream_pending': runtime(request).stream_pending(turn.id),
                     'events': runtime(request).repository.events(turn.id)}, 'replayed': replayed}


@router.post('/turns/{turn_id}/stop')
async def stop_answer(turn_id: str, request: Request):
    with runtime(request).write_lease():
        return _call(runtime(request).runner.cancel, turn_id)


@router.post('/turns/{turn_id}/drafts')
def attach_draft(turn_id: str, body: AssistantDraftAttach, request: Request):
    with runtime(request).write_lease():
        event, replayed = _call(runtime(request).repository.attach_draft, turn_id, **body.model_dump())
    return {'event': event, 'replayed': replayed}


@router.get('/turns/{turn_id}/events')
async def turn_events(turn_id: str, request: Request, after: int = Query(default=0, ge=0)):
    repo = runtime(request).repository
    _call(repo.get_turn, turn_id)  # Return 404 before committing streaming headers.
    last_event = request.headers.get('last-event-id', '')
    if last_event.isdecimal():
        after = max(after, int(last_event))

    async def stream():
        cursor = after
        while not await request.is_disconnected():
            events = await asyncio.to_thread(repo.events, turn_id, after=cursor)
            for event in events:
                cursor = event.sequence
                yield f'id: {cursor}\nevent: {event.kind}\ndata: {event.model_dump_json()}\n\n'
            turn = await asyncio.to_thread(repo.get_turn, turn_id)
            if turn.status != 'running' and not runtime(request).stream_pending(turn_id):
                # Drain once after observing terminal status: the final write
                # could have happened between the two reads above.
                for event in await asyncio.to_thread(repo.events, turn_id, after=cursor):
                    yield f'id: {event.sequence}\nevent: {event.kind}\ndata: {event.model_dump_json()}\n\n'
                yield 'event: terminal\ndata: ' + turn.model_dump_json() + '\n\n'
                return
            yield ': keep-alive\n\n'
            await asyncio.sleep(0.25)
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
