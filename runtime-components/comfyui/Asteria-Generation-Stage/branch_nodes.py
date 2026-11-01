"""Asteria-owned model wrapper using the installed native sampler unchanged."""
from copy import copy
from contextlib import suppress
import json
import os
from pathlib import Path
import re

from .branching import CONTRACT, MAX_BYTES, MAX_METADATA, _file_hash, load_tape, record, replay, save_tape


def _directory(value):
    original = Path(value)
    path = original.resolve(strict=True)
    owner_path = path / 'owner.json'
    if original.is_symlink() or owner_path.is_symlink() or owner_path.stat().st_size > MAX_METADATA:
        raise ValueError('branch owner is invalid')
    owner = json.loads(owner_path.read_text(encoding='utf-8'))
    root = Path(owner['output_root']).resolve(strict=True)
    if (owner.get('contract') != CONTRACT or not re.fullmatch(r'[0-9a-f]{32}', path.name)
            or owner.get('operation_id') != path.name or path.parent != root / '_branches'):
        raise ValueError('branch directory is invalid')
    return path


def _write(directory, name, value):
    path = directory / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, sort_keys=True), encoding='utf-8')
    os.replace(temporary, path)


class AsteriaGenerationBranchModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'operation': ('STRING',)}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'attach'
    CATEGORY = 'Asteria/internal'

    def attach(self, model, operation):
        import comfy.patcher_extension
        import psutil

        config = json.loads(operation)
        directory = _directory(config['directory'])
        if config['mode'] not in {'record', 'replay'}:
            raise ValueError('branch mode is invalid')
        result = model.clone()

        def outer(executor, noise, latent, sampler, sigmas, *args, **kwargs):
            selected = copy(sampler)
            native = sampler.sampler_function
            saved_tape = None
            if config['mode'] == 'replay':
                if psutil.virtual_memory().available < MAX_BYTES * 2:
                    raise ValueError('branch needs more free system memory')
                source = _directory(config['source_directory'])
                if _file_hash(source / 'state.json') != config['source_sha256']:
                    raise ValueError('branch source changed')
                saved_tape = load_tape(source)
                restored = saved_tape['tensors']['sigmas']
                if restored.shape != sigmas.shape or restored.dtype != sigmas.dtype:
                    raise ValueError('branch schedule contract changed')
                # CPU and GPU interpolation can differ after the model is loaded.
                # Resume the saved schedule before noise scaling, not a recomputation.
                sigmas = restored.to(device=sigmas.device)

            def sample(denoiser, x, schedule, **options):
                if psutil.virtual_memory().available < MAX_BYTES * 2:
                    if config['mode'] == 'replay':
                        raise ValueError('branch needs more free system memory')
                    _write(directory, 'state.json', {'status': 'not_saved', 'reason': 'memory_limit'})
                    return native(denoiser, x, schedule, **options)
                if config['mode'] == 'record':
                    output, tape = record(native, denoiser, x, schedule, cut_step=config['cut_step'], **options)
                    try:
                        if tape['status'] == 'completed':
                            save_tape(tape, directory)
                        else:
                            _write(directory, 'state.json', {'status': 'not_saved', 'reason': tape.get('reason')})
                    except (OSError, ValueError):
                        # A failed optional save must not discard the finished image.
                        with suppress(OSError):
                            _write(directory, 'state.json', {'status': 'not_saved', 'reason': 'recording_failed'})
                    return output
                output, receipt = replay(native, denoiser, x, schedule, saved_tape, **options)
                _write(directory, 'replay.json', receipt)
                return output

            selected.sampler_function = sample
            return executor(noise, latent, selected, sigmas, *args, **kwargs)

        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, 'asteria_generation_branch', outer)
        return (result,)


NODE_CLASS_MAPPINGS = {'AsteriaGenerationBranchModel': AsteriaGenerationBranchModel}
