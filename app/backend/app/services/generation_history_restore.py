"""Recover a saved image request without reading other clients' transient input."""
from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.schemas.action_contracts import GenerationDraftDocument
from app.schemas.generation import Img2ImgRequest, InpaintRequest, Txt2ImgRequest
from app.services.operation_repository import OperationRepository

_MODES = {'txt2img': Txt2ImgRequest, 'img2img': Img2ImgRequest, 'inpaint': InpaintRequest}


class HistoryRestoreError(ValueError):
    def __init__(self, code: str, status_code: int = 409, **detail):
        super().__init__(code)
        self.status_code, self.detail = status_code, detail


def capture_history_request(request: Txt2ImgRequest, *, image_check_enabled: bool) -> dict[str, Any]:
    # Keep the pre-adapter input paths, not the filenames written by ComfyUI.
    # Provider credentials and the resolved output root are not draft inputs.
    return {**request.model_dump(mode='json', exclude_unset=True), 'image_check_enabled': image_check_enabled}


def _original_input(item: dict, operations: OperationRepository, mode: str) -> dict:
    stored = item['settings'].get('request_snapshot')
    if isinstance(stored, dict):
        return stored
    source = operations.for_artifact(f"history:{item['id']}", kind=f'generation.{mode}')
    if len(source) != 1:
        raise HistoryRestoreError('generation_history_input_unavailable')
    saved = source[0].request_snapshot
    if saved.get('mode') != mode or not isinstance(saved.get('request'), dict):
        raise HistoryRestoreError('generation_history_input_unavailable')
    explicit = saved.get('request_fields_set')
    inputs = {key: value for key, value in saved['request'].items()
              if not isinstance(explicit, list) or key in explicit}
    operation = saved.get('generation_operation_snapshot')
    operation_settings = operation.get('settings') if isinstance(operation, dict) else None
    if isinstance(operation_settings, dict) and type(operation_settings.get('enabled')) is bool:
        inputs['image_check_enabled'] = operation_settings['enabled']
    return inputs


def _recorded_seed(item: dict, output_index: int | None) -> int:
    settings = item['settings']
    if output_index is None:
        generated = settings.get('generated_seeds')
        seed = generated[0] if isinstance(generated, list) and generated else settings.get('seed')
    else:
        manifest = settings.get('primary_outputs')
        matches = [output for output in manifest if isinstance(output, dict)
                   and type(output.get('output_index')) is int and output['output_index'] == output_index
                   and isinstance(output.get('path'), str) and output['path'] in item['output_paths']
                   ] if isinstance(manifest, list) else []
        if len(matches) != 1:
            raise HistoryRestoreError('output_image_not_found', 404, output_index=output_index)
        seed = matches[0].get('seed')
    if type(seed) is not int or seed < 0:
        raise HistoryRestoreError('generation_history_seed_unavailable')
    return seed


def restore_image_generation(item: dict | None, operations: OperationRepository, *, output_index: int | None = None) -> dict:
    from app.api.generation import _missing_generation_image_inputs

    if item is None:
        raise HistoryRestoreError('generation_not_found', 404)
    if item['media_kind'] != 'image':
        raise HistoryRestoreError('generation_history_image_required')
    settings = item['settings']
    mode = settings.get('mode')
    if not isinstance(mode, str) or mode not in _MODES or settings.get('xyz_lora_restore'):
        raise HistoryRestoreError('generation_history_input_unavailable')
    request = _original_input(item, operations, mode)
    if not isinstance(request.get('prompt'), str) or not isinstance(item.get('model_slots'), dict) or not item['model_slots']:
        raise HistoryRestoreError('generation_history_input_unavailable')

    seed = _recorded_seed(item, output_index)
    inputs = {key: settings[key] for key in _MODES[mode].model_fields.keys() & settings.keys()
              if key not in {'init_image_path', 'mask_image_path', 'record_branch_step', 'model_slot_overrides'}}
    inputs.update(request)
    slots = dict(item['model_slots'])
    if item['profile'] == 'sdxl':
        # A saved built-in VAE must not pick up a later global external selection.
        slots.setdefault('vae', 'Automatic')
    inputs.update(profile=item['profile'], seed=seed, model_slot_overrides=slots)
    if isinstance(settings.get('runtime_adapter'), str):
        inputs['runtime'] = settings['runtime_adapter']
    missing = _missing_generation_image_inputs({**inputs, 'mode': mode})
    if missing:
        raise HistoryRestoreError('generation_history_assets_missing', fields=missing)
    try:
        return TypeAdapter(GenerationDraftDocument).validate_python(dict(mode=mode, input=inputs)).model_dump(mode='json')
    except ValidationError as exc:
        if any('spatial_prompt_not_in_release' in error['msg'] for error in exc.errors()):
            raise HistoryRestoreError('spatial_prompt_not_in_release') from exc
        raise HistoryRestoreError('generation_history_input_unavailable') from exc
