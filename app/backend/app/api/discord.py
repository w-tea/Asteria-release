from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import AppConfig
from app.schemas.discord import (
    DiscordConnectionStatus,
    DiscordPolicy,
    DiscordProfileStatus,
    DiscordPublishRequest,
    DiscordPublishedProfile,
    DiscordSecretStatus,
    DiscordUnpublishRequest,
)
from app.services.discord_bot import (
    DiscordBotRuntimeManager,
    DiscordOperationDelivery,
)
from app.services.discord_delivery import DiscordDeliveryService
from app.services.discord_gateway import DiscordGatewayService, DiscordPolicyRepository
from app.services.discord_overrides import (
    DiscordOverrideRepository,
    DiscordOverrideScope,
    DiscordOverrideValidationError,
    DiscordSessionOverride,
)
from app.services.discord_profiles import (
    DiscordProfileConflictError,
    DiscordProfileNotPublishedError,
    DiscordProfileRepository,
    DiscordProfileService,
    DiscordProfileValidationError,
)
from app.services.model_library_cache import get_model_library
from app.services.model_storage import resolve_model_root
from app.services.history import HistoryRepository
from app.services.discord_receipts import DiscordReceiptRepository
from app.services.presets import PresetRepository
from app.services.windows_secrets import (
    SecretDecryptionError,
    SecretUnavailableError,
    WindowsSecretStore,
)


router = APIRouter(prefix="/api/discord", tags=["discord"])
_service_lock = RLock()
_configured_secret_store: WindowsSecretStore | None = None
_configured_profile_service: DiscordProfileService | None = None
_configured_override_repository: DiscordOverrideRepository | None = None
_runtime_manager_instance: DiscordBotRuntimeManager | None = None


class DiscordConfigurationRequest(BaseModel):
    token: str = Field(default="", max_length=4096, exclude=True, repr=False)
    policy: DiscordPolicy


class DiscordApiStatus(BaseModel):
    policy: DiscordPolicy
    secret: DiscordSecretStatus
    profile: DiscordProfileStatus
    connection: DiscordConnectionStatus


class DiscordOverrideWriteRequest(BaseModel):
    scope: DiscordOverrideScope
    fields: dict[str, Any]
    ttl_seconds: int = Field(default=3600, ge=300, le=86_400)


def configure_discord_api_services(
    *,
    secret_store: WindowsSecretStore | None = None,
    profile_service: DiscordProfileService | None = None,
    override_repository: DiscordOverrideRepository | None = None,
) -> None:
    global _configured_secret_store, _configured_profile_service
    global _configured_override_repository, _runtime_manager_instance
    with _service_lock:
        previous_manager = _runtime_manager_instance
        _configured_secret_store = secret_store
        _configured_profile_service = profile_service
        _configured_override_repository = override_repository
        _runtime_manager_instance = None
    if previous_manager is not None:
        previous_manager.stop()


def _secret_store() -> WindowsSecretStore:
    with _service_lock:
        configured = _configured_secret_store
    if configured is not None:
        return configured
    return WindowsSecretStore(AppConfig().secret_dir)


def _model_exists(stable_model_id: str) -> bool:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    return any(
        item.stable_model_id == stable_model_id
        for item in get_model_library(model_root).items
    )


def _resolve_model(stable_model_id: str) -> str | None:
    config = AppConfig()
    model_root = resolve_model_root(data_dir=config.data_dir, fallback=config.model_dir)
    matches = [
        item
        for item in get_model_library(model_root).items
        if item.stable_model_id == stable_model_id
    ]
    if len(matches) != 1:
        return None
    return str(matches[0].path)


def _preset_exists(preset_id: str) -> bool:
    return any(
        item.id == preset_id
        for item in PresetRepository(AppConfig().data_dir / "presets.json").list("generation")
    )


def _load_preset(preset_id: str) -> dict[str, Any] | None:
    item = next(
        (
            preset
            for preset in PresetRepository(
                AppConfig().data_dir / "presets.json"
            ).list("generation")
            if preset.id == preset_id
        ),
        None,
    )
    return item.model_dump(mode="json") if item is not None else None


def _profile_service() -> DiscordProfileService:
    with _service_lock:
        configured = _configured_profile_service
    if configured is not None:
        return configured
    config = AppConfig()
    return DiscordProfileService(
        DiscordProfileRepository(config.data_dir / "asteria.db"),
        model_exists=_model_exists,
        preset_exists=_preset_exists,
    )


def _override_repository() -> DiscordOverrideRepository:
    with _service_lock:
        configured = _configured_override_repository
    if configured is not None:
        return configured
    return DiscordOverrideRepository(AppConfig().data_dir / "asteria.db")


def _policy_repository() -> DiscordPolicyRepository:
    return DiscordPolicyRepository(AppConfig().data_dir / "config" / "discord.json")


def _connection_status(
    policy: DiscordPolicy,
    secret: DiscordSecretStatus,
) -> DiscordConnectionStatus:
    return DiscordConnectionStatus(
        configured=bool(secret.configured and policy.application_id),
        connected=False,
        commands_registered=False,
        application_id=policy.application_id,
    )


def _gateway_service(policy: DiscordPolicy) -> DiscordGatewayService:
    from app.api.generation import cancel_generation_operation, enqueue_discord_txt2img
    from app.api.operations import get_operation_coordinator

    config = AppConfig()
    coordinator = get_operation_coordinator()
    receipts = DiscordReceiptRepository(config.data_dir / "asteria.db")

    def active_counts() -> tuple[int, int]:
        operations = coordinator.list(limit=1000)
        running = sum(
            item.status in {"preparing", "running", "cancelling"}
            and item.source_actor == "discord"
            for item in operations
        )
        queued = sum(
            item.status == "queued" and item.source_actor == "discord"
            for item in operations
        )
        return running, queued

    return DiscordGatewayService(
        policy=policy,
        profiles=_profile_service(),
        overrides=_override_repository(),
        receipts=receipts,
        enqueue=lambda request, snapshot, identity, context: enqueue_discord_txt2img(
            request,
            discord_snapshot=snapshot,
            identity=identity,
            requester_id=context.requester_id,
            guild_id=context.guild_id,
            channel_id=context.channel_id,
        ),
        cancel_operation=cancel_generation_operation,
        preset_loader=_load_preset,
        model_resolver=_resolve_model,
        active_counts=active_counts,
    )


def get_discord_runtime_manager() -> DiscordBotRuntimeManager:
    global _runtime_manager_instance
    with _service_lock:
        manager = _runtime_manager_instance
        if manager is not None:
            return manager
        config = AppConfig()
        from app.api.operations import get_operation_coordinator

        operations = get_operation_coordinator().repository
        manager = DiscordBotRuntimeManager(
            policy_loader=_policy_repository().load,
            secret_store=_secret_store(),
            gateway_factory=_gateway_service,
            delivery=DiscordOperationDelivery(
                operations=operations,
                receipts=DiscordReceiptRepository(config.data_dir / "asteria.db"),
                history=HistoryRepository(config.data_dir / "asteria.db"),
                artifacts=DiscordDeliveryService(
                    config.data_dir / "cache" / "discord-delivery"
                ),
            ),
            lock_path=config.data_dir / "cache" / "discord-gateway.lock",
        )
        _runtime_manager_instance = manager
        return manager


def start_discord_runtime() -> DiscordConnectionStatus:
    return get_discord_runtime_manager().start()


def stop_discord_runtime() -> DiscordConnectionStatus:
    with _service_lock:
        manager = _runtime_manager_instance
    if manager is None:
        return DiscordConnectionStatus(
            configured=False,
            connected=False,
            commands_registered=False,
        )
    return manager.stop()


def _status() -> DiscordApiStatus:
    policy = _policy_repository().load()
    secret = _secret_store().status()
    return DiscordApiStatus(
        policy=policy,
        secret=secret,
        profile=_profile_service().get_status(),
        connection=(
            _runtime_manager_instance.status()
            if _runtime_manager_instance is not None
            else _connection_status(policy, secret)
        ),
    )


@router.get("/status", response_model=DiscordApiStatus)
def get_discord_status() -> DiscordApiStatus:
    return _status()


@router.put("/configuration", response_model=DiscordApiStatus)
def update_discord_configuration(
    request: DiscordConfigurationRequest,
) -> DiscordApiStatus:
    if request.policy.enabled:
        if not request.policy.application_id:
            raise HTTPException(
                status_code=422,
                detail={"code": "discord_application_id_required"},
            )
        if not any(
            (
                request.policy.allowed_guild_ids,
                request.policy.allowed_channel_ids,
                request.policy.allowed_user_ids,
                request.policy.allowed_role_ids,
                request.policy.admin_user_ids,
            )
        ):
            raise HTTPException(
                status_code=422,
                detail={"code": "discord_allowlist_required"},
            )
    try:
        if request.token:
            _secret_store().save_discord_token(request.token)
        _policy_repository().save(request.policy)
    except (SecretUnavailableError, ValueError) as exc:
        code = str(exc) if str(exc).startswith("discord_") else "discord_secret_save_failed"
        raise HTTPException(status_code=409, detail={"code": code}) from None
    if _runtime_manager_instance is not None:
        _runtime_manager_instance.restart()
    return _status()


@router.delete("/token", response_model=DiscordApiStatus)
def delete_discord_token() -> DiscordApiStatus:
    try:
        _secret_store().delete_discord_token()
    except SecretUnavailableError:
        raise HTTPException(
            status_code=409,
            detail={"code": "discord_secret_delete_failed"},
        ) from None
    if _runtime_manager_instance is not None:
        _runtime_manager_instance.restart()
    return _status()


@router.post("/connection/restart", response_model=DiscordApiStatus)
def restart_discord_connection() -> DiscordApiStatus:
    get_discord_runtime_manager().restart()
    return _status()


@router.post("/profiles/publish", response_model=DiscordProfileStatus)
def publish_discord_profile(request: DiscordPublishRequest) -> DiscordProfileStatus:
    try:
        return _profile_service().publish(request)
    except DiscordProfileConflictError:
        raise HTTPException(
            status_code=409,
            detail={"code": "discord_publish_revision_conflict"},
        ) from None
    except DiscordProfileValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": str(exc)}) from None


@router.post("/profiles/unpublish", response_model=DiscordProfileStatus)
def unpublish_discord_profile(
    request: DiscordUnpublishRequest,
) -> DiscordProfileStatus:
    try:
        return _profile_service().unpublish(expected_revision=request.expected_revision)
    except DiscordProfileConflictError:
        raise HTTPException(
            status_code=409,
            detail={"code": "discord_publish_revision_conflict"},
        ) from None


@router.get("/profiles/{revision}", response_model=DiscordPublishedProfile)
def get_discord_profile_revision(revision: int) -> DiscordPublishedProfile:
    try:
        return _profile_service().repository.get_revision(revision)
    except DiscordProfileNotPublishedError:
        raise HTTPException(
            status_code=404,
            detail={"code": "discord_profile_revision_not_found"},
        ) from None


@router.put("/overrides", response_model=DiscordSessionOverride)
def set_discord_override(
    request: DiscordOverrideWriteRequest,
) -> DiscordSessionOverride:
    try:
        return _override_repository().set(
            request.scope,
            request.fields,
            ttl=timedelta(seconds=request.ttl_seconds),
        )
    except DiscordOverrideValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": str(exc)}) from None


@router.post("/overrides/show", response_model=DiscordSessionOverride | None)
def show_discord_override(
    scope: DiscordOverrideScope,
) -> DiscordSessionOverride | None:
    return _override_repository().show(scope)


@router.delete("/overrides")
def clear_discord_override(scope: DiscordOverrideScope) -> dict[str, bool]:
    return {"cleared": _override_repository().clear(scope)}
