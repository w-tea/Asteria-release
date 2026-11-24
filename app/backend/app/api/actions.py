"""Thin HTTP access to the core catalog; Assistant and MCP use the same service."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.schemas.action_contracts import ActionCall, ActionDescription, ActionResult
from app.services.action_dispatch import ActionDispatcher, ActionError
from app.api.action_assets import router as upload_router


router = APIRouter(prefix='/api/actions', tags=['actions'])
router.include_router(upload_router, prefix='/uploads')


@router.get('')
def action_catalog(request: Request) -> dict[str, list[ActionDescription]]:
    return {'items': ActionDispatcher(request.app.state.config).catalog()}


@router.post('/{action_name}', response_model=ActionResult)
def invoke_action(action_name: str, call: ActionCall, request: Request) -> ActionResult:
    try:
        return ActionDispatcher(request.app.state.config).invoke(action_name, call)
    except ActionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ValidationError as exc:
        errors = [{**error, 'loc': ('body', 'arguments', *error['loc'])} for error in exc.errors()]
        raise RequestValidationError(errors, body=call.model_dump(mode='json')) from exc
