"""Pair-local sampling inputs, applied before native noise scaling; ordinary jobs are untouched."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from safetensors.torch import load_file, save_file
import torch

CONTRACT = 'asteria.comparison-sampling.v1'
MAX_BYTES = 128 * 1024 * 1024
MAX_METADATA = 256 * 1024


def _json(path):
    if path.is_symlink() or path.stat().st_size > MAX_METADATA:
        raise ValueError('comparison metadata limit')
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('comparison metadata invalid')
    return result


def _directory(value):
    original = Path(value)
    path = original.resolve(strict=True)
    owner = _json(path / 'owner.json')
    root = Path(owner['output_root']).resolve(strict=True)
    if (original.is_symlink() or owner.get('contract') != CONTRACT
            or not re.fullmatch('[0-9a-f]{32}', path.name)
            or owner.get('operation_id') != path.name or path.parent != root / '_comparisons'):
        raise ValueError('comparison owner invalid')
    return path


def _write(path, value):
    if path.exists():
        raise ValueError('comparison record already exists')
    body = json.dumps(value, allow_nan=False, sort_keys=True).encode('utf-8')
    if len(body) > MAX_METADATA:
        raise ValueError('comparison metadata limit')
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('xb') as stream:
        stream.write(body)
    os.replace(temporary, path)


def _cpu(tensor):
    if not torch.is_tensor(tensor) or tensor.is_nested:
        raise ValueError('comparison tensor invalid')
    return tensor.detach().to(device='cpu', copy=True).contiguous()


def _identity(tensor):
    value = _cpu(tensor)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def sample_pair(executor, config, index, noise, latent, sampler, sigmas, *args, **kwargs):
    directory = _directory(config['directory'])
    arm, stage = config['arm'], config['stage']
    if arm not in {'A', 'B'} or not re.fullmatch('[a-z]+-[0-9]+', stage) or not 0 <= index < 256:
        raise ValueError('comparison stage invalid')
    tensors = {'noise': noise, 'sigmas': sigmas}
    byte_count = sum(t.numel() * t.element_size() for t in tensors.values())
    # Include saved archives and bounded CPU copies; do not allocate an oversized pair.
    saved_bytes = sum(p.stat().st_size for p in directory.iterdir() if p.suffix == '.safetensors')
    if byte_count * 3 + saved_bytes > MAX_BYTES:
        raise ValueError('comparison tensor byte limit')
    seed = args[3] if len(args) > 3 else kwargs.get('seed')
    mask = args[0] if args else kwargs.get('denoise_mask')
    function = sampler.sampler_function
    owner = getattr(executor, 'class_obj', None)
    contract = {'seed': seed, 'cfg': getattr(owner, 'cfg', None),
        'sampler': [function.__module__, function.__qualname__],
        'extra_options': sampler.extra_options, 'inpaint_options': sampler.inpaint_options,
        'mask': None if mask is None else _identity(mask)}
    incoming = {key: _identity(t) for key, t in tensors.items()}
    latent_id = _identity(latent)
    name = f'{stage}-{index}'
    archive = directory / f'A-{name}.safetensors'
    source = None
    if arm == 'A':
        if archive.exists():
            raise ValueError('comparison anchor already exists')
        temporary = archive.with_suffix('.safetensors.tmp')
        save_file({key: _cpu(t) for key, t in tensors.items()}, str(temporary))
        os.replace(temporary, archive)
    else:
        source = _json(directory / f'A-{name}.json')
        if (source.get('status') != 'completed' or source['contract'] != contract
                or archive.is_symlink() or archive.stat().st_size > MAX_BYTES
                or sha256(archive.read_bytes()).hexdigest() != source['archive_sha256']):
            raise ValueError('comparison sampling contract changed')
        restored = load_file(str(archive), device='cpu')
        for key, value in restored.items():
            if (key not in tensors or value.shape != tensors[key].shape or value.dtype != tensors[key].dtype
                    or _identity(value) != source['used'][key]):
                raise ValueError('comparison noise contract changed')
        if set(restored) != {'noise', 'sigmas'}:
            raise ValueError('comparison tensors missing')
        if stage == 'base-0' and index == 0 and latent_id != source['latent']:
            raise ValueError('comparison initial latent changed')
        noise = restored['noise'].to(device=noise.device)
        sigmas = restored['sigmas'].to(device=sigmas.device)
    used = {'noise': _identity(noise), 'sigmas': _identity(sigmas)}
    # The native executor still owns sampling, RNG for intermediate draws and callbacks.
    result = executor(noise, latent, sampler, sigmas, *args, **kwargs)
    _write(directory / f'{arm}-{name}.json', {'status': 'completed', 'contract': contract,
        'stage': stage, 'index': index, 'incoming': incoming, 'used': used, 'latent': latent_id,
        'source': source['used'] if source else used,
        'archive_sha256': sha256(archive.read_bytes()).hexdigest(),
        'noise_replaced': incoming['noise'] != used['noise'],
        'sigmas_replaced': incoming['sigmas'] != used['sigmas']})
    return result


class AsteriaComparisonSamplingModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'operation': ('STRING',)}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'attach'
    CATEGORY = 'Asteria/internal'

    def attach(self, model, operation):
        import comfy.patcher_extension
        config = json.loads(operation)
        _directory(config['directory'])
        result = model.clone()
        calls = 0
        def outer(executor, noise, latent, sampler, sigmas, *args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            return sample_pair(executor, config, index, noise, latent, sampler, sigmas, *args, **kwargs)
        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            'asteria_comparison_sampling', outer)
        return (result,)


NODE_CLASS_MAPPINGS = {'AsteriaComparisonSamplingModel': AsteriaComparisonSamplingModel}
