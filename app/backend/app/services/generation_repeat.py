"""Persistent repeat progression around existing image jobs, not another GPU queue.

The small scheduler never waits for a child inside an Operation worker. Startup
recovery interrupts old parents; only this process's explicit starts are advanced.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import UTC, datetime
import json
import logging
import sqlite3
from threading import Event, RLock, Thread
from typing import Callable

from pydantic import TypeAdapter

from app.schemas.action_contracts import GenerationDraftDocument
from app.schemas.generation_repeat import GenerationRepeatRecord, RepeatLimits
from app.schemas.operations import TERMINAL_OPERATION_STATUSES
from app.services.action_receipts import read_action_receipt, save_action_receipt
from app.services.operation_coordinator import OperationCoordinator
from app.services.operation_repository import OperationConflictError, OperationNotFoundError, canonical_operation_json
from app.services.reference_mutation import reference_mutation_coordinator


logger = logging.getLogger(__name__)
_document = TypeAdapter(GenerationDraftDocument)


class GenerationRepeatService:
    def __init__(self, coordinator: OperationCoordinator, *, enqueue: Callable, get_job: Callable,
                 cancel_job: Callable, clock: Callable[[], datetime] | None = None):
        self.coordinator = coordinator
        self.db_path = coordinator.repository.db_path
        self.enqueue, self.get_job, self.cancel_job = enqueue, get_job, cancel_job
        self.clock = clock or (lambda: datetime.now(UTC))
        self._lock = RLock()
        self._active: set[str] = set()
        self._closed = Event()
        self._thread: Thread | None = None
        with closing(self._connect()) as con, con:
            con.execute('''create table if not exists generation_repeats (
                operation_id text primary key, input_revision integer not null,
                document_json text not null, limits_json text not null, state_json text not null
            )''')

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @contextmanager
    def _transaction(self):
        with closing(self._connect()) as con, con:
            con.execute('begin immediate')
            yield con

    def _row(self, con, identity):
        row = con.execute('select * from generation_repeats where operation_id=?', (identity,)).fetchone()
        if row is None:
            raise OperationNotFoundError('generation_repeat_not_found')
        return row

    def get(self, identity: str) -> GenerationRepeatRecord:
        with self._lock, closing(self._connect()) as con:
            row = self._row(con, identity)
            operation = self.coordinator.get(identity)
            end = operation.completed_at or self.clock()
            return GenerationRepeatRecord(
                id=identity, status=operation.status, input_revision=row['input_revision'],
                document=json.loads(row['document_json']), limits=json.loads(row['limits_json']),
                elapsed_ms=max(0, int((end - operation.created_at).total_seconds() * 1000)),
                **json.loads(row['state_json']),
            )

    def by_request(self, request_id: str) -> GenerationRepeatRecord:
        with self._lock, closing(self._connect()) as con:
            row = con.execute('select action, result_json from action_receipts where request_id=?', (request_id,)).fetchone()
            if row is None or row['action'] != 'generation.repeats.start':
                raise OperationNotFoundError('generation_repeat_not_found')
            return self.get(json.loads(row['result_json'])['operation_id'])

    def start(self, request_id: str, document, limits: RepeatLimits, *, identity_arguments=None):
        document = _document.validate_python(document)
        if limits.pass_target and document.input.image_check_enabled is False:
            raise ValueError('generation_repeat_checker_required')
        arguments = {'document': document.model_dump(mode='json'), 'limits': limits.model_dump()}
        with self._lock, reference_mutation_coordinator.acquire_publisher(), self.coordinator.submission(), self._transaction() as con:
            if self._closed.is_set():
                raise OperationConflictError('generation_repeat_service_closed')
            operation, created = self.coordinator.enqueue_with_action_receipt(
                request_id=request_id, action='generation.repeats.start', arguments=identity_arguments or arguments,
                kind='generation.repeat', source_actor='local', request_snapshot=arguments,
                progress_total=limits.max_attempts or None, progress_unit='images', connection=con)
            if created:
                con.execute('insert into generation_repeats values (?, 1, ?, ?, ?)', (
                    operation.operation_id, canonical_operation_json(arguments['document']),
                    canonical_operation_json(arguments['limits']), '{}'))
        with self._lock:
            if created:
                self._active.add(operation.operation_id)
        return self.get(operation.operation_id), not created

    def update_receipt(self, request_id: str, arguments: dict):
        with self._lock, closing(self._connect()) as con:
            receipt = read_action_receipt(con, request_id, 'generation.repeats.update', canonical_operation_json(arguments))
            return GenerationRepeatRecord.model_validate(receipt) if receipt is not None else None

    def update(self, request_id: str, identity: str, expected_revision: int, document, *, identity_arguments=None):
        document = _document.validate_python(document)
        arguments = canonical_operation_json(identity_arguments or {'repeat_id': identity, 'expected_revision': expected_revision,
                                               'document': document.model_dump(mode='json')})
        with self._lock, reference_mutation_coordinator.acquire_publisher(), self.coordinator.submission(), self._transaction() as con:
            receipt = read_action_receipt(con, request_id, 'generation.repeats.update', arguments)
            if receipt is not None:
                return GenerationRepeatRecord.model_validate(receipt), True
            row = self._row(con, identity)
            operation = self.coordinator.get(identity)
            if operation.status in TERMINAL_OPERATION_STATUSES or operation.cancel_requested_at:
                raise OperationConflictError('generation_repeat_stopped')
            if row['input_revision'] != expected_revision:
                raise OperationConflictError('repeat_input_revision_conflict')
            con.execute('update generation_repeats set input_revision=input_revision+1, document_json=? where operation_id=?',
                        (canonical_operation_json(document.model_dump(mode='json')), identity))
            # Build the new response from this transaction; another connection
            # must not read the uncommitted version or wait on its own write lock.
            current = self.get(identity).model_copy(update={'input_revision': expected_revision + 1, 'document': document})
            save_action_receipt(con, request_id, 'generation.repeats.update', arguments, current.model_dump_json())
            return current, False

    def _save_state(self, record: GenerationRepeatRecord, *, connection=None, **patch):
        fields = ('current_job_id', 'latest_job_id', 'attempted', 'passed', 'completed_cycles', 'stop_reason', 'error')
        state = {name: getattr(record, name) for name in fields}
        state.update(patch)
        def write(con):
            con.execute('update generation_repeats set state_json=? where operation_id=?', (canonical_operation_json(state), record.id))
        if connection is not None:
            write(connection)
        else:
            with self._transaction() as con:
                write(con)

    def _transition(self, identity, status, phase, **values):
        with self.coordinator.submission():
            operation = self.coordinator.get(identity)
            if operation.status in TERMINAL_OPERATION_STATUSES:
                return operation
            return self.coordinator.repository.transition(identity, expected_revision=operation.row_revision,
                target_status=status, phase=phase, **values)

    def stop(self, identity: str, *, stop_current: bool):
        with self._lock, reference_mutation_coordinator.acquire_publisher(), self.coordinator.submission():
            record = self.get(identity)
            if record.status in TERMINAL_OPERATION_STATUSES:
                return record
            operation = self.coordinator.request_cancel(identity)
            # Queued repeats can end immediately. A running one is still
            # stopping until its current child reaches a terminal state.
            if operation.status == 'cancelled':
                self._save_state(record, stop_reason='stopped')
            if stop_current and record.current_job_id:
                self.cancel_job(record.current_job_id)
            return self.get(identity)

    @staticmethod
    def _limit(record):
        if record.limits.max_attempts and record.attempted >= record.limits.max_attempts:
            return 'attempts'
        if record.limits.max_minutes and record.elapsed_ms >= record.limits.max_minutes * 60_000:
            return 'time'
        if record.limits.pass_target and record.passed >= record.limits.pass_target:
            return 'passes'
        return ''

    def _finish(self, record, reason, status='completed', error=None):
        self._save_state(record, current_job_id=None, stop_reason=reason, error=error)
        self._transition(record.id, status, f'repeat.{reason}', progress_current=record.attempted)
        self._active.discard(record.id)

    def tick(self):
        with self._lock:
            if self._closed.is_set():
                return
            for identity in tuple(self._active):
                with reference_mutation_coordinator.acquire_publisher():
                    try:
                        self._advance(identity)
                    except OperationConflictError as exc:
                        if str(exc) == 'operation_parent_stopping':
                            continue
                        self._finish(self.get(identity), 'error', 'failed', str(exc))
                    except Exception as exc:
                        # Preserve core HTTP details, never fabricate a completed
                        # result or retry generation after an uncertain failure.
                        detail = getattr(exc, 'detail', 'generation_repeat_failed')
                        self._finish(self.get(identity), 'error', 'failed', detail)
                        logger.warning('Generation repeat progression failed: %s', type(exc).__name__)

    def _advance(self, identity):
        record = self.get(identity)
        if record.status in TERMINAL_OPERATION_STATUSES:
            self._active.discard(identity)
            return
        if record.current_job_id:
            job = self.get_job(record.current_job_id)
            if job['status'] not in TERMINAL_OPERATION_STATUSES:
                return
            if job['status'] not in {'completed', 'completed_with_warnings'}:
                self._finish(record, 'stopped' if job['status'] == 'cancelled' else 'error', job['status'], job.get('error'))
                return
            check = ((job.get('generation') or {}).get('settings') or {}).get('image_check')
            check = check if isinstance(check, dict) and check.get('status') in {'pass', 'reject', 'unknown'} else None
            previous = self.coordinator.get(record.current_job_id).request_snapshot
            previous_input = previous.get('request') or previous.get('input') or {}
            count = int(previous_input.get('batch_size', 1)) * int(previous_input.get('batch_count', 1))
            passed = max(0, min(count, int((check or {}).get('passed') or 0)))
            self._save_state(record, current_job_id=None, passed=record.passed + passed, completed_cycles=record.completed_cycles + 1)
            record = self.get(identity)
            if check and check.get('pause') is True:
                self._finish(record, 'checker_pause', 'completed_with_warnings')
                return
            if record.limits.pass_target and check is None:
                self._finish(record, 'checker_required', 'failed', 'generation_repeat_checker_required')
                return
            # Account the completed image separately from the next acceptance.
            # UI input changes and stop requests may arrive between these ticks.
            if record.status == 'cancelling':
                self._finish(record, 'stopped', 'cancelled')
            elif reason := self._limit(record):
                self._finish(record, reason)
            return
        if record.status == 'cancelling':
            self._finish(record, 'stopped', 'cancelled')
            return
        if reason := self._limit(record):
            self._finish(record, reason)
            return
        document = record.document.model_copy(deep=True)
        size, batches = document.input.batch_size, document.input.batch_count
        if record.limits.max_attempts:
            remaining = record.limits.max_attempts - record.attempted
            size = min(size, remaining)
            batches = min(batches, remaining // size)
        document.input = document.input.model_copy(update={'batch_size': size, 'batch_count': batches})
        with self.coordinator.submission(identity):
            self._transition(identity, 'running', 'repeat.running')
            # The child, its redelivery receipt and this parent link commit in
            # one transaction, before the ordinary worker can start.
            def accepted(child, connection):
                self._save_state(record, connection=connection, current_job_id=child.operation_id,
                                 latest_job_id=child.operation_id, attempted=record.attempted + size * batches)
            self.enqueue(document, identity, f'repeat:{identity}:{record.completed_cycles}', accepted)
            self._transition(identity, 'running', 'repeat.running', progress_current=record.attempted + size * batches)

    def run_in_background(self):
        with self._lock:
            if self._thread is not None or self._closed.is_set():
                return
            def run():
                while not self._closed.wait(.35):
                    try:
                        self.tick()
                    except Exception:
                        logger.warning('Generation repeat storage unavailable; accepted work retained')
            self._thread = Thread(target=run, daemon=True, name='asteria-generation-repeat')
            self._thread.start()

    def close(self):
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()
