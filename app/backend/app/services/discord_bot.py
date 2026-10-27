from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import os
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

import discord
from discord import app_commands
import psutil

from app.schemas.discord import (
    DiscordConnectionStatus,
    DiscordGenerateCommand,
    DiscordInteractionContext,
    DiscordPolicy,
)
from app.schemas.operations import TERMINAL_OPERATION_STATUSES, OperationRecord
from app.services.discord_delivery import DiscordDeliveryArtifact, DiscordDeliveryService
from app.services.discord_gateway import (
    DiscordGatewayError,
    DiscordGatewayService,
    DiscordGatewaySubmission,
)
from app.services.discord_receipts import DiscordReceiptRepository
from app.services.history import HistoryRepository
from app.services.operation_repository import OperationRepository
from app.services.windows_secrets import (
    SecretDecryptionError,
    SecretUnavailableError,
    WindowsSecretStore,
)


class DiscordBotRuntimeError(RuntimeError):
    pass


_DELIVERABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


async def bounded_discord_retry(
    action: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    maximum_retry_after: float = 30.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    if attempts < 1 or attempts > 5 or maximum_retry_after <= 0:
        raise ValueError("discord_retry_policy_invalid")
    for attempt in range(attempts):
        try:
            return await action()
        except Exception as exc:
            retry_after = _retry_after(exc)
            if retry_after is None or attempt + 1 >= attempts:
                raise
            await sleep(min(maximum_retry_after, max(0.0, retry_after)))
    raise AssertionError("unreachable")


def _retry_after(exc: Exception) -> float | None:
    status = getattr(exc, "status", None)
    if status != 429:
        return None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "retry_after", None)
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        return None


class DiscordOperationDelivery:
    def __init__(
        self,
        *,
        operations: OperationRepository,
        receipts: DiscordReceiptRepository,
        history: HistoryRepository,
        artifacts: DiscordDeliveryService,
        poll_interval: float = 1.0,
        timeout_seconds: float = 15 * 60,
    ) -> None:
        self.operations = operations
        self.receipts = receipts
        self.history = history
        self.artifacts = artifacts
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds

    async def wait_and_deliver(
        self,
        submission: DiscordGatewaySubmission,
        *,
        followup: Callable[[str, tuple[Path, ...]], Awaitable[None]],
        fallback: Callable[[str, tuple[Path, ...]], Awaitable[None]] | None,
    ) -> None:
        operation, interaction_expired = await self._wait_for_terminal(
            submission.operation_id
        )
        started_at = datetime.now(UTC)
        prepared: tuple[DiscordDeliveryArtifact, ...] = ()
        warning_codes: tuple[str, ...] = ()
        error_code: str | None = None
        try:
            message = _operation_message(operation)
            if operation.status in {"completed", "completed_with_warnings"}:
                prepared = tuple(
                    self.artifacts.prepare(
                        source,
                        operation_id=operation.operation_id,
                        output_index=index,
                    )
                    for index, source in enumerate(self._history_output_paths(operation))
                )
                warning_codes = tuple(
                    dict.fromkeys(
                        warning
                        for artifact in prepared
                        for warning in artifact.warning_codes
                    )
                )
            paths = tuple(artifact.path for artifact in prepared)
            if interaction_expired:
                if fallback is None:
                    raise DiscordBotRuntimeError("discord_interaction_expired")
                await bounded_discord_retry(lambda: fallback(message, paths))
            else:
                try:
                    await bounded_discord_retry(lambda: followup(message, paths))
                except Exception:
                    if fallback is None:
                        raise
                    await bounded_discord_retry(lambda: fallback(message, paths))
            receipt = self.receipts.get_by_operation(operation.operation_id)
            self.receipts.update_delivery(receipt.identity, status="delivered")
            self.operations.append_terminal_child_summary(
                operation.operation_id,
                {
                    "child_summary_id": "discord-delivery",
                    "ordinal": len(operation.child_summaries),
                    "kind": "discord.delivery",
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC),
                    "artifact_refs": operation.artifact_refs,
                    "warning_codes": warning_codes,
                },
            )
        except Exception as exc:
            error_code = _delivery_error_code(exc)
            receipt = self.receipts.get_by_operation(operation.operation_id)
            self.receipts.update_delivery(
                receipt.identity,
                status="delivery_failed",
                error_code=error_code,
            )
            self.operations.append_terminal_child_summary(
                operation.operation_id,
                {
                    "child_summary_id": "discord-delivery",
                    "ordinal": len(operation.child_summaries),
                    "kind": "discord.delivery",
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC),
                    "artifact_refs": operation.artifact_refs,
                    "warning_codes": ("discord_delivery_failed",),
                    "error_code": error_code,
                },
            )
        finally:
            for artifact in prepared:
                artifact.path.unlink(missing_ok=True)

    async def _wait_for_terminal(
        self,
        operation_id: str,
    ) -> tuple[OperationRecord, bool]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        interaction_expired = False
        while True:
            operation = await asyncio.to_thread(self.operations.get, operation_id)
            if operation.status in TERMINAL_OPERATION_STATUSES:
                return operation, interaction_expired
            interaction_expired = interaction_expired or loop.time() >= deadline
            await asyncio.sleep(self.poll_interval)

    def _history_output_paths(self, operation: OperationRecord) -> tuple[Path, ...]:
        paths: list[Path] = []
        for artifact_ref in operation.artifact_refs:
            if not artifact_ref.startswith("history:"):
                continue
            try:
                generation_id = int(artifact_ref.partition(":")[2])
            except ValueError:
                continue
            item = self.history.get(generation_id)
            if item is None:
                continue
            paths.extend(
                Path(value)
                for value in item.get("output_paths", [])
                if (
                    isinstance(value, str)
                    and value
                    and Path(value).suffix.casefold() in _DELIVERABLE_IMAGE_SUFFIXES
                )
            )
        if not paths:
            raise DiscordBotRuntimeError("discord_delivery_artifact_missing")
        return tuple(paths)


def _operation_message(operation: OperationRecord) -> str:
    if operation.status in {"completed", "completed_with_warnings"}:
        return f"Asteria generation completed (`{operation.operation_id}`)."
    if operation.status == "cancelled":
        return f"Asteria generation was cancelled (`{operation.operation_id}`)."
    code = operation.error_code or operation.status
    if code == "generation_runtime_unreachable":
        return (
            "Asteria generation failed because the runtime is unavailable "
            f"(`{operation.operation_id}`). Start or repair the runtime in "
            "Asteria Settings, then try again."
        )
    if code in {"missing_detailer_runtime_nodes", "missing_runtime_nodes"}:
        return (
            "Asteria generation failed because required runtime components are unavailable "
            f"(`{operation.operation_id}`). Repair runtime dependencies in "
            "Asteria Settings, then try again."
        )
    return (
        f"Asteria generation failed (`{operation.operation_id}`). "
        "Check the operation in Asteria and retry after resolving the reported issue."
    )


def _delivery_error_code(exc: Exception) -> str:
    text = str(exc)
    if text.startswith("discord_") and len(text) <= 128:
        return text
    if isinstance(exc, discord.NotFound):
        return "discord_interaction_expired"
    if isinstance(exc, discord.Forbidden):
        return "discord_delivery_forbidden"
    return "discord_delivery_failed"


class _DiscordInstanceLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._held = False

    def acquire(self, application_id: str) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{os.getpid()}:{application_id}"
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if self._stale():
                self.path.unlink(missing_ok=True)
                return self.acquire(application_id)
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            target.write(payload)
        self._held = True
        return True

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def _stale(self) -> bool:
        try:
            pid_text = self.path.read_text(encoding="utf-8").partition(":")[0]
            pid = int(pid_text)
            if pid <= 0:
                return True
            return not psutil.pid_exists(pid)
        except (OSError, ValueError):
            return True


class DiscordBotRuntimeManager:
    def __init__(
        self,
        *,
        policy_loader: Callable[[], DiscordPolicy],
        secret_store: WindowsSecretStore,
        gateway_factory: Callable[[DiscordPolicy], DiscordGatewayService],
        delivery: DiscordOperationDelivery,
        lock_path: Path,
    ) -> None:
        self.policy_loader = policy_loader
        self.secret_store = secret_store
        self.gateway_factory = gateway_factory
        self.delivery = delivery
        self._lease = _DiscordInstanceLease(lock_path)
        self._lock = RLock()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: discord.Client | None = None
        self._stopping = Event()
        self._status = DiscordConnectionStatus(
            configured=False,
            connected=False,
            commands_registered=False,
        )

    def status(self) -> DiscordConnectionStatus:
        with self._lock:
            return self._status.model_copy()

    def start(self) -> DiscordConnectionStatus:
        policy = self.policy_loader()
        try:
            token = self.secret_store.load_discord_token()
        except (SecretUnavailableError, SecretDecryptionError):
            token = ""
        configured = bool(policy.application_id and token)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._status
            self._status = DiscordConnectionStatus(
                configured=configured,
                connected=False,
                commands_registered=False,
                application_id=policy.application_id,
            )
        if not policy.enabled or not configured:
            return self.status()
        if not self._lease.acquire(policy.application_id):
            with self._lock:
                self._status = self._status.model_copy(
                    update={
                        "last_error_code": "discord_multi_instance_detected",
                        "multi_instance_warning": True,
                    }
                )
            return self.status()
        self._stopping.clear()
        self._thread = Thread(
            target=self._thread_main,
            args=(policy, token),
            name="asteria-discord-gateway",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def stop(self) -> DiscordConnectionStatus:
        self._stopping.set()
        with self._lock:
            loop = self._loop
            client = self._client
            thread = self._thread
        if loop is not None and client is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(client.close(), loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        self._lease.release()
        with self._lock:
            self._thread = None
            self._loop = None
            self._client = None
            self._status = self._status.model_copy(
                update={"connected": False, "commands_registered": False}
            )
            return self._status

    def restart(self) -> DiscordConnectionStatus:
        self.stop()
        return self.start()

    def _thread_main(self, policy: DiscordPolicy, token: str) -> None:
        try:
            asyncio.run(self._serve(policy, token))
        except discord.LoginFailure:
            self._set_error("discord_token_invalid")
        except Exception:
            self._set_error("discord_gateway_connection_failed")
        finally:
            self._lease.release()
            with self._lock:
                self._loop = None
                self._client = None
                self._status = self._status.model_copy(
                    update={"connected": False}
                )

    async def _serve(self, policy: DiscordPolicy, token: str) -> None:
        runtime = self
        gateway = self.gateway_factory(policy)

        class AsteriaDiscordClient(discord.Client):
            def __init__(self) -> None:
                super().__init__(intents=discord.Intents.none())
                self.tree = app_commands.CommandTree(self)

            async def setup_hook(self) -> None:
                runtime._register_commands(self.tree, gateway, policy)
                if policy.registration_guild_id:
                    guild = discord.Object(id=int(policy.registration_guild_id))
                    self.tree.copy_global_to(guild=guild)
                    await self.tree.sync(guild=guild)
                else:
                    await self.tree.sync()
                runtime._mark_commands_registered()

            async def on_ready(self) -> None:
                runtime._mark_connected(self)

        client = AsteriaDiscordClient()
        with self._lock:
            self._loop = asyncio.get_running_loop()
            self._client = client
        await client.start(token, reconnect=True)

    def _register_commands(
        self,
        tree: app_commands.CommandTree,
        gateway: DiscordGatewayService,
        policy: DiscordPolicy,
    ) -> None:
        @tree.command(name="generate", description="Generate a txt2img image with Asteria")
        @app_commands.describe(
            prompt="Positive prompt",
            negative_prompt="Negative prompt",
            preset="Allowed Asteria preset ID",
            model="Allowed Asteria stable model ID",
            steps="Sampling steps (1-80)",
            width="Width (256-2048, multiple of 64)",
            height="Height (256-2048, multiple of 64)",
            batch_count="Batch count (1-4)",
        )
        async def generate(
            interaction: discord.Interaction,
            prompt: str,
            negative_prompt: str = "",
            preset: str = "",
            model: str = "",
            steps: app_commands.Range[int, 1, 80] | None = None,
            width: app_commands.Range[int, 256, 2048] | None = None,
            height: app_commands.Range[int, 256, 2048] | None = None,
            batch_count: app_commands.Range[int, 1, 4] | None = None,
        ) -> None:
            await interaction.response.defer(thinking=True)
            overrides = {
                key: value
                for key, value in {
                    "steps": steps,
                    "width": width,
                    "height": height,
                    "batch_count": batch_count,
                }.items()
                if value is not None
            }
            try:
                context = _interaction_context(interaction, policy.application_id)
                submission = await asyncio.to_thread(
                    gateway.submit,
                    context,
                    DiscordGenerateCommand(
                        prompt=prompt,
                        negative_prompt=negative_prompt or None,
                        preset_id=preset or None,
                        model_id=model or None,
                        overrides=overrides,
                    ),
                )
                await interaction.edit_original_response(
                    content=_operation_running_message(submission.operation_id),
                )
                asyncio.create_task(
                    self.delivery.wait_and_deliver(
                        submission,
                        followup=lambda message, paths: _send_interaction_files(
                            interaction,
                            message,
                            paths,
                        ),
                        fallback=(
                            (
                                lambda message, paths: self._send_fallback(
                                    policy.fallback_channel_id,
                                    message,
                                    paths,
                                )
                            )
                            if policy.fallback_channel_id
                            else None
                        ),
                    )
                )
            except (DiscordGatewayError, ValueError) as exc:
                await interaction.followup.send(
                    _actionable_error(str(exc)),
                    ephemeral=True,
                )

        @generate.autocomplete("model")
        async def model_autocomplete(
            _interaction: discord.Interaction,
            current: str,
        ) -> list[app_commands.Choice[str]]:
            return [
                app_commands.Choice(name=item["name"], value=item["value"])
                for item in gateway.autocomplete("model", current)
            ]

        @generate.autocomplete("preset")
        async def preset_autocomplete(
            _interaction: discord.Interaction,
            current: str,
        ) -> list[app_commands.Choice[str]]:
            return [
                app_commands.Choice(name=item["name"], value=item["value"])
                for item in gateway.autocomplete("preset", current)
            ]

        @tree.command(name="status", description="Show an Asteria operation status")
        async def status(interaction: discord.Interaction, operation_id: str) -> None:
            try:
                context = _interaction_context(interaction, policy.application_id)
                gateway.authorize(context)
                _authorize_operation_access(
                    gateway,
                    operation_id,
                    requester_id=context.requester_id,
                )
                operation = await asyncio.to_thread(
                    self.delivery.operations.get,
                    operation_id,
                )
                await interaction.response.send_message(
                    f"`{operation.operation_id}`: **{operation.status}** / {operation.phase}",
                    ephemeral=True,
                )
            except Exception:
                await interaction.response.send_message(
                    "The Asteria operation was not found.",
                    ephemeral=True,
                )

        @tree.command(name="cancel", description="Cancel your Asteria operation")
        async def cancel(interaction: discord.Interaction, operation_id: str) -> None:
            try:
                context = _interaction_context(interaction, policy.application_id)
                gateway.authorize(context)
                operation = await asyncio.to_thread(
                    gateway.cancel,
                    operation_id,
                    requester_id=str(interaction.user.id),
                )
                await interaction.response.send_message(
                    f"`{operation.operation_id}`: **{operation.status}**",
                    ephemeral=True,
                )
            except (DiscordGatewayError, ValueError) as exc:
                await interaction.response.send_message(
                    _actionable_error(str(exc)),
                    ephemeral=True,
                )

    async def _send_fallback(
        self,
        channel_id: str,
        message: str,
        paths: tuple[Path, ...],
    ) -> None:
        with self._lock:
            client = self._client
        if client is None:
            raise DiscordBotRuntimeError("discord_fallback_unavailable")
        channel = client.get_channel(int(channel_id))
        if channel is None:
            channel = await client.fetch_channel(int(channel_id))
        if not hasattr(channel, "send"):
            raise DiscordBotRuntimeError("discord_fallback_unavailable")
        if not paths:
            await channel.send(message)
            return
        for index in range(0, len(paths), 10):
            chunk = paths[index : index + 10]
            files = [discord.File(path, filename=path.name) for path in chunk]
            await channel.send(message if index == 0 else "Additional Asteria outputs.", files=files)

    def _mark_commands_registered(self) -> None:
        with self._lock:
            self._status = self._status.model_copy(
                update={
                    "commands_registered": True,
                    "validated_at": datetime.now(UTC),
                    "last_error_code": "",
                }
            )

    def _mark_connected(self, client: discord.Client) -> None:
        user = client.user
        with self._lock:
            self._status = self._status.model_copy(
                update={
                    "connected": True,
                    "bot_user_id": str(user.id) if user else "",
                    "connected_at": datetime.now(UTC),
                    "last_error_code": "",
                }
            )

    def _set_error(self, code: str) -> None:
        with self._lock:
            self._status = self._status.model_copy(
                update={
                    "connected": False,
                    "commands_registered": False,
                    "last_error_code": code,
                }
            )


def _interaction_context(
    interaction: discord.Interaction,
    application_id: str,
) -> DiscordInteractionContext:
    guild = interaction.guild
    channel = interaction.channel
    role_ids = tuple(
        str(role.id)
        for role in getattr(interaction.user, "roles", ())
    )
    return DiscordInteractionContext(
        application_id=application_id,
        interaction_id=str(interaction.id),
        requester_id=str(interaction.user.id),
        guild_id=str(guild.id) if guild is not None else None,
        channel_id=str(interaction.channel_id) if interaction.channel_id else None,
        role_ids=role_ids,
        channel_nsfw=bool(
            channel is not None
            and hasattr(channel, "is_nsfw")
            and channel.is_nsfw()
        ),
        has_attachment=bool(
            interaction.data
            and isinstance(interaction.data, dict)
            and interaction.data.get("resolved", {}).get("attachments")
        ),
    )


async def _send_interaction_files(
    interaction: discord.Interaction,
    message: str,
    paths: tuple[Path, ...],
) -> None:
    if not paths:
        await interaction.edit_original_response(
            content=message,
            attachments=[],
        )
        return
    first_chunk = paths[:10]
    await interaction.edit_original_response(
        content=message,
        attachments=[
            discord.File(path, filename=path.name)
            for path in first_chunk
        ],
    )
    for index in range(10, len(paths), 10):
        chunk = paths[index : index + 10]
        files = [discord.File(path, filename=path.name) for path in chunk]
        await interaction.followup.send(
            "Additional Asteria outputs.",
            files=files,
        )


def _operation_running_message(operation_id: str) -> str:
    return (
        f"Asteria is generating (`{operation_id}`). "
        f"Use `/status operation_id:{operation_id}` to check progress."
    )


def _authorize_operation_access(
    gateway: DiscordGatewayService,
    operation_id: str,
    *,
    requester_id: str,
) -> None:
    try:
        receipt = gateway.receipts.get_by_operation(operation_id)
    except LookupError as exc:
        raise DiscordGatewayError("discord_operation_not_found") from exc
    if (
        requester_id != receipt.requester_id
        and requester_id not in gateway.policy.admin_user_ids
    ):
        raise DiscordGatewayError("discord_operation_not_allowed")


def _actionable_error(code: str) -> str:
    messages = {
        "discord_profile_not_published": "Discord generation is not published in Asteria Settings.",
        "discord_gateway_disabled": "The Discord gateway is disabled in Asteria Settings.",
        "discord_queue_full": "The Asteria Discord queue is full. Try again later.",
        "discord_user_cooldown": "Please wait for the Discord generation cooldown.",
        "discord_model_unresolved": "The published model is unavailable. Publish the Generate settings again.",
        "discord_cancel_not_allowed": "Only the requester or an Asteria Discord admin can cancel this operation.",
        "discord_requester_not_allowed": "This Discord user or role is not allowed to use Asteria.",
        "discord_channel_not_allowed": "This Discord channel is not allowed to use Asteria.",
        "discord_guild_not_allowed": "This Discord server is not allowed to use Asteria.",
    }
    return messages.get(code, f"Asteria rejected the request: `{code}`.")
