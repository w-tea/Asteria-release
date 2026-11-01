"""Registered intermediate sampler checkpoints under the common output root."""
from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from PIL import Image

from app.schemas.generation import Txt2ImgRequest
from app.services.atomic_json import write_json_atomic
from app.services.media_artifacts import file_sha256

CONTRACT = 'asteria.generation-branch.v1'
MAX_BYTES = 128 * 1024 * 1024
MAX_METADATA = 256 * 1024
FILES = ('owner.json', 'state.json', 'state.safetensors', 'replay.json')
SAMPLERS = {'Euler', 'Euler a', 'DPM++ 2M', 'DPM++ 2M SDE', 'DPM++ 3M SDE', 'ER SDE'}


def control_input_refs(values: dict):
    for group in ('controlnet_units', 'anima_lllite_units'):
        units = [unit for unit in values.get(group, []) if unit.get('enabled', True)]
        for index, unit in enumerate(units):
            for field in ('image_path', 'mask_image_path'):
                if unit.get(field):
                    yield group, index, field, str(unit[field])


def check_control_inputs(saved: dict, directory: Path) -> None:
    refs = list(control_input_refs(saved.get('request', {})))
    records = saved.get('control_inputs') or []
    expected = {(group, index, field): path for group, index, field, path in refs}
    if len(records) != len(expected):
        raise ValueError('branch_inputs_unavailable')
    seen = set()
    for record in records:
        key = record['group'], record['index'], record['field']
        original = Path(record['path'])
        if (key in seen or expected.get(key) != str(original) or original.is_symlink()
                or original.resolve().parent != directory or not original.is_file()
                or original.stat().st_size != record['size_bytes']
                or file_sha256(original) != record['sha256']):
            raise ValueError('branch_inputs_changed')
        seen.add(key)


def executed_seed(settings: dict) -> int:
    generated = settings.get('generated_seeds') or []
    seed = generated[0] if len(generated) == 1 else settings.get('seed')
    if type(seed) is not int or seed < 0:
        raise ValueError('branch_seed_unavailable')
    return seed


def read_json(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > MAX_METADATA:
        raise ValueError('branch_file_invalid')
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('branch_file_invalid')
    return result


def validate_capture(request: Txt2ImgRequest, mode: str, runtime: str) -> None:
    if request.record_branch_step is None and request.branch_source_id is None:
        return
    if (mode != 'txt2img' or request.profile not in {'sdxl', 'anima'} or runtime != 'comfy_workflow'
            or request.batch_size != 1 or request.batch_count != 1 or request.sampler not in SAMPLERS):
        raise ValueError('branch_settings_not_supported')
    if request.record_branch_step is not None:
        limit = (int(request.steps * request.refiner_switch_at)
                 if request.refiner_enabled and request.refiner_checkpoint not in {'', 'None'} else request.steps)
        if request.record_branch_step >= limit or request.branch_source_id is not None:
            raise ValueError('branch_step_not_available')


def branch_request(item: dict, changes: dict) -> Txt2ImgRequest:
    saved = (item.get('settings') or {}).get('generation_branch') or {}
    if saved.get('mode') != 'record' or saved.get('status') != 'completed':
        raise ValueError('branch_not_available')
    if set(changes) - {'prompt', 'negative_prompt', 'cfg_scale'}:
        raise ValueError('branch_changes_not_supported')
    values = deepcopy(saved['request'])
    if values.get('seed', -1) < 0:
        values['seed'] = executed_seed(item.get('settings') or {})
    values.update(prompt=saved['prompt'], negative_prompt=saved['negative_prompt'],
                  prompt_input=None, prompt_pins=None, advanced_styles=[], record_branch_step=None,
                  branch_source_id=item['id'])
    values.update(changes)
    return Txt2ImgRequest.model_validate(values)


def resolve_source(item: dict) -> Path:
    settings = item.get('settings') or {}
    saved = settings.get('generation_branch') or {}
    if item.get('status') != 'completed' or saved.get('mode') != 'record' or saved.get('status') != 'completed':
        raise ValueError('branch_not_available')
    original = Path(saved['directory'])
    directory = original.resolve(strict=True)
    root = Path(settings['output_root']).resolve(strict=True)
    if (original.is_symlink() or directory.parent != root / '_branches'
            or not re.fullmatch(r'[0-9a-f]{32}', directory.name)):
        raise ValueError('branch_path_invalid')
    owner = read_json(directory / 'owner.json')
    if owner != {'contract': CONTRACT, 'output_root': str(root), 'operation_id': directory.name}:
        raise ValueError('branch_owner_changed')
    manifest = directory / 'state.json'
    archive = directory / 'state.safetensors'
    registered = {str(Path(path).resolve()) for path in item.get('output_paths', [])}
    if str(manifest) not in registered or str(archive) not in registered:
        raise ValueError('branch_not_registered')
    if file_sha256(manifest) != saved['manifest_sha256']:
        raise ValueError('branch_state_changed')
    state = read_json(manifest)
    if (state.get('contract') != CONTRACT or state.get('status') != 'completed' or archive.is_symlink()
            or archive.stat().st_size != state.get('archive_bytes') or archive.stat().st_size > MAX_BYTES + MAX_METADATA
            or file_sha256(archive) != state.get('archive_sha256')):
        raise ValueError('branch_state_changed')
    check_control_inputs(saved, directory)
    return directory


def check_asset_identity(saved: dict, current: dict) -> None:
    def records(snapshot):
        assets = snapshot.get('assets', [])
        selected = assets
        if not selected or any(a.get('status') != 'recorded' or not a.get('sha256') for a in selected):
            raise ValueError('branch_assets_unavailable')
        return sorted((a['role'], a['type'], a['sha256']) for a in selected)
    if records(saved) != records(current):
        raise ValueError('branch_assets_changed')


class BranchSession:
    def __init__(self):
        self.directory: Path | None = None
        self.root: Path | None = None
        self.source: dict | None = None
        self.committed = False
        self.control_inputs: list[dict] = []
        self.input_paths: list[Path] = []

    def __enter__(self): return self

    def __exit__(self, *_):
        if self.directory is None or self.committed:
            return
        if self.directory.parent != self.root / '_branches' or self.directory.is_symlink():
            return
        with suppress(OSError):
            for name in (*FILES, *(path.name for path in self.input_paths)):
                for suffix in ('', '.tmp'):
                    (self.directory / (name + suffix)).unlink(missing_ok=True)
            self.directory.rmdir()

    def begin(self, root: Path, request: Txt2ImgRequest, source: dict | None) -> dict:
        self.root = Path(root).resolve()
        parent = self.root / '_branches'
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError('branch_path_invalid')
        self.directory = parent / uuid4().hex
        self.directory.mkdir()
        write_json_atomic(self.directory / 'owner.json', {
            'contract': CONTRACT, 'output_root': str(self.root), 'operation_id': self.directory.name})
        self.source = source
        operation = {'directory': str(self.directory), 'mode': 'replay' if source else 'record',
                     'cut_step': request.record_branch_step or 1}
        if source:
            operation.update(source_directory=str(resolve_source(source)),
                             source_sha256=source['settings']['generation_branch']['manifest_sha256'])
        values = request.model_dump(mode='json')
        total_bytes = 0
        for group, index, field, source_path in control_input_refs(values):
            path = Path(source_path)
            before = path.stat()
            total_bytes += before.st_size
            if total_bytes > MAX_BYTES:
                raise ValueError('branch_input_limit')
            with Image.open(path) as image:
                image.verify()
            digest = file_sha256(path)
            target = self.directory / f'input-{self.directory.name}-{len(self.input_paths)}{path.suffix or ".png"}'
            self.input_paths.append(target)
            shutil.copyfile(path, target)
            after = path.stat()
            identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if identity(before) != identity(after) or file_sha256(target) != digest:
                raise ValueError('branch_input_changed')
            self.control_inputs.append({'group': group, 'index': index, 'field': field,
                'source_path': source_path, 'path': str(target),
                'size_bytes': target.stat().st_size, 'sha256': digest})
        return operation

    def prepare_control_inputs(self, settings: dict) -> None:
        for record in self.control_inputs:
            units = [unit for unit in settings.get(record['group'], []) if unit.get('enabled', True)]
            units[record['index']][record['field']] = record['path']

    def finish(self, request: Txt2ImgRequest, settings: dict, slots: dict, prompt: str, negative: str,
               status: str) -> tuple[dict, list[str]]:
        retain_inputs = False
        try:
            receipt, paths = self._finish(request, settings, slots, prompt, negative, status)
            retain_inputs = receipt.get('mode') == 'record' and receipt.get('status') == 'completed'
            return receipt, paths
        finally:
            if not retain_inputs:
                for record in self.control_inputs:
                    units = [unit for unit in settings.get(record['group'], []) if unit.get('enabled', True)]
                    if record['index'] < len(units):
                        unit = units[record['index']]
                        unit[record['field']] = record['source_path']
                        field = 'source_image_path' if record['field'] == 'image_path' else 'mask_source_image_path'
                        unit[field] = record['source_path']

    def _finish(self, request: Txt2ImgRequest, settings: dict, slots: dict, prompt: str, negative: str,
                status: str) -> tuple[dict, list[str]]:
        check = settings.get('image_check') or {}
        if (status != 'completed' or check.get('status') in {'reject', 'unknown'}
                or any(item.get('status', item.get('verdict')) in {'reject', 'unknown'} for item in check.get('results', []))):
            return {'status': 'not_saved', 'reason': 'generation_incomplete'}, []
        if self.source:
            receipt = read_json(self.directory / 'replay.json')
            if receipt.get('boundary_exact') is not True or receipt.get('prefix_model_calls') != 0:
                raise ValueError('branch_boundary_mismatch')
            saved = self.source['settings']['generation_branch']
            return {'mode': 'replay', 'status': 'completed', 'source_generation_id': self.source['id'],
                    'source_manifest_sha256': saved['manifest_sha256'], 'cut_step': saved['cut_step'], **receipt}, []
        try:
            assets = settings.get('generation_assets', {})
            try:
                check_asset_identity(assets, assets)
            except ValueError:
                return {'status': 'not_saved', 'reason': 'branch_assets_unavailable'}, []
            manifest = self.directory / 'state.json'
            state = read_json(manifest)
            if state.get('status') != 'completed':
                return {'status': 'not_saved', 'reason': state.get('reason', 'recording_failed')}, []
            input_bytes = sum(record['size_bytes'] for record in self.control_inputs)
            if state.get('archive_bytes', state['byte_count']) + input_bytes > MAX_BYTES:
                return {'status': 'not_saved', 'reason': 'branch_input_limit'}, []
            frozen = request.model_dump(mode='json')
            frozen.update(seed=executed_seed(settings), model_slot_overrides=slots)
            self.prepare_control_inputs(frozen)
            paths = [str(self.directory / name) for name in ('owner.json', 'state.json', 'state.safetensors')]
            return {'mode': 'record', 'status': 'completed', 'directory': str(self.directory),
                    'manifest_sha256': file_sha256(manifest), 'cut_step': state['cut_step'],
                    'prefix_count': state['prefix_count'], 'bytes': state['byte_count'] + input_bytes,
                    'request': frozen, 'prompt': prompt, 'negative_prompt': negative,
                    'control_inputs': deepcopy(self.control_inputs),
                    'generation_assets': settings.get('generation_assets', {})}, paths
        except (OSError, ValueError, KeyError):
            return {'status': 'not_saved', 'reason': 'recording_failed'}, []

    def commit(self):
        # Replay receipts are embedded in the new history; their private files are temporary.
        if self.source is None:
            self.committed = True
