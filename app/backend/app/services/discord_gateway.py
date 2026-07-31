from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.schemas.discord import (
    DiscordGenerateCommand,
    DiscordInteractionContext,
    DiscordPolicy,
)
from app.schemas.generation import Txt2ImgRequest
from app.schemas.operations import OperationRecord
from app.services.atomic_json import write_json_atomic
from app.services.discord_overrides import (
    DiscordOverrideRepository,
    compose_discord_generation_request,
)
from app.services.discord_profiles import (
    DiscordProfileNotPublishedError,
    DiscordProfileService,
)
from app.services.discord_receipts import (
    DiscordOperationClaim,
    DiscordReceiptIdentity,
    DiscordReceiptRepository,
)


class DiscordGatewayError(RuntimeError):
    pass


class DiscordGatewaySubmission(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: str
    created: bool
    published_revision: int


class DiscordPolicyRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()

    def load(self) -> DiscordPolicy:
        with self._lock:
            try:
                if not self.path.is_file():
                    return DiscordPolicy()
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                return DiscordPolicy.model_validate(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                return DiscordPolicy()

    def save(self, policy: DiscordPolicy) -> DiscordPolicy:
        with self._lock:
            write_json_atomic(self.path, policy.model_dump(mode="json"))
        return policy


class DiscordGatewayService:
    def __init__(
        self,
        *,
        policy: DiscordPolicy,
        profiles: DiscordProfileService,
        overrides: DiscordOverrideRepository,
        receipts: DiscordReceiptRepository,
        enqueue: Callable[
            [
                Txt2ImgRequest,
                dict[str, Any],
                DiscordReceiptIdentity,
                DiscordInteractionContext,
            ],
            DiscordOperationClaim,
        ],
        cancel_operation: Callable[[str], OperationRecord],
        preset_loader: Callable[[str], dict[str, Any] | None],
        model_resolver: Callable[[str], str | None],
        active_counts: Callable[[], tuple[int, int]],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy
        self.profiles = profiles
        self.overrides = overrides
        self.receipts = receipts
        self.enqueue = enqueue
        self.cancel_operation = cancel_operation
        self.preset_loader = preset_loader
        self.model_resolver = model_resolver
        self.active_counts = active_counts
        self.clock = clock or (lambda: datetime.now(UTC))
        self._cooldowns: dict[str, datetime] = {}
        self._lock = RLock()

    def submit(
        self,
        context: DiscordInteractionContext,
        command: DiscordGenerateCommand,
    ) -> DiscordGatewaySubmission:
        self._authorize(context)
        identity = DiscordReceiptIdentity(
            provider="discord",
            application_id=context.application_id,
            interaction_id=context.interaction_id,
        )
        try:
            existing = self.receipts.get(identity)
        except LookupError:
            existing = None
        if existing is not None:
            try:
                revision = self.profiles.require_published().revision
            except DiscordProfileNotPublishedError:
                revision = 1
            return DiscordGatewaySubmission(
                operation_id=existing.operation_id,
                created=False,
                published_revision=revision,
            )

        with self._lock:
            self._enforce_cooldown(context.requester_id)
            _running, queued = self.active_counts()
            if queued >= self.policy.queue_limit:
                raise DiscordGatewayError("discord_queue_full")

            try:
                published = self.profiles.require_published()
            except DiscordProfileNotPublishedError as exc:
                raise DiscordGatewayError("discord_profile_not_published") from exc

            selected_preset_id = command.preset_id or published.preset_id
            preset = None
            if selected_preset_id:
                if selected_preset_id not in self.policy.allowed_preset_ids:
                    raise DiscordGatewayError("discord_preset_not_allowed")
                preset = self.preset_loader(selected_preset_id)
                if preset is None:
                    raise DiscordGatewayError("discord_preset_unresolved")

            model_references = dict(published.model_references)
            if command.model_id:
                if command.model_id not in self.policy.allowed_model_ids:
                    raise DiscordGatewayError("discord_model_not_allowed")
                model_references["checkpoint"] = command.model_id
            resolved_model_slots: dict[str, str] = {}
            for slot_id, stable_model_id in model_references.items():
                resolved = self.model_resolver(stable_model_id)
                if not resolved:
                    raise DiscordGatewayError("discord_model_unresolved")
                resolved_model_slots[slot_id] = resolved

            session_overrides = self.overrides.resolve(
                guild_id=context.guild_id,
                channel_id=context.channel_id,
                user_id=context.requester_id,
            )
            composition = compose_discord_generation_request(
                published=published,
                preset=preset,
                session_overrides=session_overrides,
                command_overrides=dict(command.overrides),
                prompt=command.prompt,
                negative_prompt=command.negative_prompt,
                resolved_model_slots=resolved_model_slots,
            )
            snapshot = {
                **composition.snapshot,
                "model_references": model_references,
                "provider": "discord",
                "application_id": context.application_id,
                "interaction_id": context.interaction_id,
                "requester_id": context.requester_id,
                "guild_id": context.guild_id,
                "channel_id": context.channel_id,
            }
            claim = self.enqueue(composition.request, snapshot, identity, context)
            if claim.created:
                self._cooldowns[context.requester_id] = self.clock()
            return DiscordGatewaySubmission(
                operation_id=claim.operation.operation_id,
                created=claim.created,
                published_revision=published.revision,
            )

    def cancel(self, operation_id: str, *, requester_id: str) -> OperationRecord:
        try:
            receipt = self.receipts.get_by_operation(operation_id)
        except LookupError as exc:
            raise DiscordGatewayError("discord_operation_not_found") from exc
        if (
            requester_id != receipt.requester_id
            and requester_id not in self.policy.admin_user_ids
        ):
            raise DiscordGatewayError("discord_cancel_not_allowed")
        return self.cancel_operation(operation_id)

    def authorize(self, context: DiscordInteractionContext) -> None:
        self._authorize(context)

    def autocomplete(self, kind: str, query: str) -> list[dict[str, str]]:
        if kind == "model":
            values = self.policy.allowed_model_ids
        elif kind == "preset":
            values = self.policy.allowed_preset_ids
        else:
            raise DiscordGatewayError("discord_autocomplete_kind_invalid")
        needle = query.strip().casefold()
        matched = [value for value in values if needle in value.casefold()]
        return [{"name": value[:100], "value": value} for value in matched[:25]]

    def _authorize(self, context: DiscordInteractionContext) -> None:
        if not self.policy.enabled:
            raise DiscordGatewayError("discord_gateway_disabled")
        if context.application_id != self.policy.application_id:
            raise DiscordGatewayError("discord_application_mismatch")
        if context.has_attachment:
            raise DiscordGatewayError("discord_txt2img_attachments_unsupported")
        if context.guild_id is None:
            if not self.policy.allow_dms:
                raise DiscordGatewayError("discord_dm_not_allowed")
        else:
            if (
                self.policy.allowed_guild_ids
                and context.guild_id not in self.policy.allowed_guild_ids
            ):
                raise DiscordGatewayError("discord_guild_not_allowed")
            if (
                self.policy.allowed_channel_ids
                and context.channel_id not in self.policy.allowed_channel_ids
            ):
                raise DiscordGatewayError("discord_channel_not_allowed")
        if context.channel_nsfw and not self.policy.allow_nsfw_channels:
            raise DiscordGatewayError("discord_nsfw_not_allowed")

        requester_allowed = context.requester_id in self.policy.allowed_user_ids
        role_allowed = bool(set(context.role_ids) & set(self.policy.allowed_role_ids))
        admin_allowed = context.requester_id in self.policy.admin_user_ids
        if (
            not requester_allowed
            and not role_allowed
            and not admin_allowed
        ):
            raise DiscordGatewayError("discord_requester_not_allowed")
        if not any(
            (
                self.policy.allowed_guild_ids,
                self.policy.allowed_channel_ids,
                self.policy.allowed_user_ids,
                self.policy.allowed_role_ids,
                self.policy.admin_user_ids,
            )
        ):
            raise DiscordGatewayError("discord_allowlist_required")

    def _enforce_cooldown(self, requester_id: str) -> None:
        if self.policy.cooldown_seconds == 0:
            return
        last = self._cooldowns.get(requester_id)
        if last is None:
            return
        elapsed = (self.clock() - last).total_seconds()
        if elapsed < self.policy.cooldown_seconds:
            raise DiscordGatewayError("discord_user_cooldown")
