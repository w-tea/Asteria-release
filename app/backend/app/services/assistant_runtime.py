"""Own live answer tasks only; generation jobs stay in the core coordinator."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import HTTPException

from app.adapters.assistant_llm import AssistantLLM
from app.services.assistant_sessions import AssistantSessionRepository
from app.services.assistant_turns import AssistantTurnRunner
from app.services.action_images import ActionImageRepository
from app.services.action_media_sources import get_media_source
from app.services.history import HistoryRepository
from app.services.resource_lifecycle import resource_admission, ResourceAdmissionTimeoutError


class AssistantRuntime:
    def __init__(self, db_path: Path, *, dispatcher, providers, llm_factory=AssistantLLM):
        self.repository = AssistantSessionRepository(db_path)
        self.images = ActionImageRepository(db_path.parent)
        self.data_dir = db_path.parent
        self.media_history: HistoryRepository | None = None
        self.runner = AssistantTurnRunner(self.repository, dispatcher)
        self.providers, self.llm_factory = providers, llm_factory
        self.tasks: dict[str, asyncio.Task] = {}

    def recover(self):
        return self.repository.recover_interrupted()

    def write_lease(self):
        # No GPU/CPU reservation: only database replacement must exclude chat
        # writes, just as it already excludes other core operations.
        try:
            return resource_admission.acquire((), operation_id='assistant.write', timeout_seconds=0)
        except ResourceAdmissionTimeoutError as exc:
            raise HTTPException(409, {'code': 'maintenance_restore_busy'}) from exc

    def submit(self, conversation_id, request):
        llm = None
        def prepare():
            nonlocal llm
            provider = next((item for item in self.providers() if item.id == request.provider_id and item.enabled), None)
            if provider is None:
                raise HTTPException(409, {'code': 'assistant_provider_not_available'})
            if not provider.endpoint or not request.model.strip():
                raise HTTPException(409, {'code': 'assistant_provider_setup_required'})
            llm = self.llm_factory(provider.model_copy(update={'model': request.model}))
            return [self.images.get(identifier).model_dump(exclude={'image'}) for identifier in request.attachments]
        # Re-delivery returns the accepted turn even if its provider was later
        # removed. Only a new request needs a currently configured connection.
        lease = self.write_lease()
        try:
            if request.media_attachments and self.media_history is None:
                self.media_history = HistoryRepository(self.data_dir / 'asteria.db')

            def prepare_media():
                return [get_media_source(self.media_history, self.data_dir, target).model_dump(mode='json')
                        for target in request.media_attachments]

            turn, replayed = self.repository.start_turn(conversation_id, request.request_id, request.prompt,
                request.provider_id, request.model, attachments=request.attachments, before_create=prepare,
                media_attachments=[item.model_dump(mode='json') for item in request.media_attachments],
                before_create_media=prepare_media)
            if not replayed:
                task = asyncio.create_task(self.runner.run(turn.id, llm))
                self.tasks[turn.id] = task
                def finished(_):
                    self.tasks.pop(turn.id, None)
                    lease.release()
                # Includes the final receipt of an already admitted core action,
                # even when the user stops the answer while admission is pending.
                task.add_done_callback(finished)
            else:
                lease.release()
        except BaseException:
            lease.release()
            raise
        return turn, replayed

    def stream_pending(self, turn_id):
        task = self.tasks.get(turn_id)
        return task is not None and not task.done()

    def conversation(self, conversation_id):
        record = self.repository.export(conversation_id)
        for turn in record['turns']:
            turn['stream_pending'] = self.stream_pending(turn['id'])
        return record

    async def close(self):
        tasks = list(self.tasks.items())
        for identifier, _ in tasks:
            self.runner.cancel(identifier)
        # Do not interrupt an already admitted core action in a worker thread.
        if tasks:
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)


def create_assistant_runtime(config):
    from app.services.action_dispatch import ActionDispatcher
    from app.services.user_settings import UserSettingsRepository
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json')
    return AssistantRuntime(config.data_dir / 'asteria.db', dispatcher=ActionDispatcher(config),
                            providers=lambda: settings.load().ai.providers)
