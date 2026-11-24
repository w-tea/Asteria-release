"""LAN controls belong to the ordinary GUI, never to the AI action catalog."""
from fastapi import APIRouter, HTTPException, Request

from app.services.lan_access import LanAccessError, LanListenerSettings
from app.services.lan_manager import LanManagerError


router = APIRouter(prefix='/api/lan', tags=['lan'])


def call(request: Request, method: str, *args):
    try:
        return getattr(request.app.state.lan_manager, method)(*args)
    except (LanManagerError, LanAccessError) as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from exc


@router.get('/settings')
def settings(request: Request):
    return call(request, 'status')


@router.post('/start')
def start(body: LanListenerSettings, request: Request):
    return call(request, 'start', body)


@router.post('/stop')
def stop(request: Request):
    return call(request, 'stop')


@router.post('/pairings')
def pairing(request: Request):
    return call(request, 'create_pairing')


@router.delete('/devices/{device_id}')
def revoke(device_id: str, request: Request):
    call(request, 'revoke', device_id)
    return call(request, 'status')
