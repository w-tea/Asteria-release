"""Run fixed pairs through the normal image generator under one GPU claim."""
from hashlib import sha256
import json
import logging
from pathlib import Path

from fastapi import HTTPException

from app.schemas.prompt_comparison import REQUEST_TYPES
from app.services.operation_executor import OperationCancelled, OperationExecutionError, OperationExecutionResult
from app.services.prompt_comparison_repository import PromptComparisonRepository

logger = logging.getLogger(__name__)


class PromptComparisonExecutor:
    version = 'prompt-comparison-v1'

    def __init__(self, db_path: Path):
        self.repository = PromptComparisonRepository(db_path)

    def resource_claims(self, _snapshot: dict) -> tuple[str, ...]:
        return ('comfy_gpu_exclusive',)

    def execute(self, context) -> OperationExecutionResult:
        from app.api import generation

        comparison_id = context.request_snapshot['comparison_id']
        plan = self.repository.get(comparison_id)
        job_id = context.operation.operation_id
        if plan['status'] != 'queued' or plan['operation_id'] != job_id:
            raise OperationExecutionError('comparison_already_started')
        total = len(plan['cells'])
        artifacts = []
        active_index = None
        first = REQUEST_TYPES[plan['mode']].model_validate(plan['cells'][0]['request'])
        with generation._jobs_lock:
            generation._jobs[job_id] = {
                **generation._new_job_cache(plan['mode'], first),
                'generation_operation_snapshot': context.request_snapshot['generation_operation_snapshot'],
                'comparison_total': total, 'comparison_index': 0,
                'cancelled': context.cancel_requested,
            }
        lease = generation.reference_mutation_coordinator.acquire_publisher()
        try:
            if plan.get('lora_test'):
                from app.api.lora_training import _run_repository
                from app.core.config import AppConfig
                from app.services.lora_practical_test import verify_practical_sources
                try:
                    verify_practical_sources(plan, _run_repository().get(plan['lora_test']['run_id']), AppConfig().root_dir)
                except ValueError as exc:
                    raise OperationExecutionError(str(exc)) from exc
            self.repository.set_status(comparison_id, 'running')
            for cell in plan['cells']:
                context.raise_if_cancelled()
                active_index = cell['index']
                request = REQUEST_TYPES[plan['mode']].model_validate(cell['request'])
                generation._set_job(job_id, request=request, comparison_index=active_index, runtime_job_id='', current=0, total=1, phase='preparing')
                self.repository.update_cell(comparison_id, active_index, status='running')
                metadata = {'comparison_id': comparison_id, 'pair': cell['pair'], 'arm': cell['arm'], 'seed': request.seed,
                            'change': plan['change'], 'input_sha256': sha256(json.dumps(cell['request'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
                if trial := plan.get('lora_test'):
                    metadata['lora_test'] = {'run_id': trial['run_id'], 'training_fingerprint': trial['training']['fingerprint'],
                        'scene': trial['scenes'][trial['pairs'][cell['pair']]['scene']], 'learned_features': trial['learned_features'],
                        'model_sha256': next((asset['sha256'] for asset in trial['assets'] if asset.get('role') == 'lora:0'), None)}
                result = generation._generate(request, plan['mode'], job_id=job_id, comparison_metadata=metadata)
                artifacts.extend(str(path) for path in result.output_paths)
                self.repository.update_cell(comparison_id, active_index, status=result.status, generation_id=result.id,
                                            output_paths=[str(item.path) for item in getattr(result, 'primary_outputs', [])]
                                            or [str(path) for path in result.output_paths])
                if result.status == 'cancelled':
                    context.acknowledge_cancel()
                    raise OperationCancelled(artifact_refs=tuple(artifacts))
                check = result.settings.get('image_check', {})
                if result.status != 'completed' or not result.output_paths or (isinstance(check, dict) and check.get('status') in {'reject', 'unknown'}):
                    raise OperationExecutionError('comparison_generation_not_completed', artifact_refs=tuple(artifacts))
                context.raise_if_cancelled()
                context.report_progress(active_index + 1, total, 'images', phase='running')
            self.repository.set_status(comparison_id, 'completed')
            return OperationExecutionResult(artifact_refs=tuple(artifacts))
        except OperationCancelled:
            self.repository.set_status(comparison_id, 'cancelled')
            if active_index is not None and self.repository.get(comparison_id)['cells'][active_index]['status'] == 'running':
                self.repository.update_cell(comparison_id, active_index, status='cancelled')
            raise
        except Exception as exc:
            if isinstance(exc, OperationExecutionError):
                code = exc.error_code
            elif isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
                code = str(exc.detail.get('code', 'comparison_generation_failed'))
            else:
                code = 'comparison_generation_failed'
            logger.exception('Prompt comparison %s stopped at %s', comparison_id, active_index)
            self.repository.set_status(comparison_id, 'failed', code)
            if active_index is not None:
                self.repository.update_cell(comparison_id, active_index, status='failed', error=code)
            raise OperationExecutionError(code, artifact_refs=tuple(artifacts)) from exc
        finally:
            try:
                generation.PreviewRegistry(self.repository.db_path).mark_terminal(job_id)
            except (OSError, generation.sqlite3.Error):
                logger.warning('Could not mark comparison previews terminal', exc_info=True)
            lease.release()
