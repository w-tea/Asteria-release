"""Restore native solver state by replaying saved predictions, not model forwards."""
from __future__ import annotations

from hashlib import sha256
import inspect
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

CONTRACT = 'asteria.generation-branch.v1'
MAX_BYTES = 128 * 1024 * 1024
MAX_METADATA = 256 * 1024
SUPPORTED = {'sample_euler', 'sample_euler_ancestral', 'sample_dpmpp_2m',
             'sample_dpmpp_2m_sde', 'sample_dpmpp_3m_sde', 'sample_er_sde'}


class _TapeLimit(Exception):
    pass


def _file_hash(path):
    result = sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1024 * 1024):
            result.update(part)
    return result.hexdigest()


def _cpu_copy(value):
    # A one-element expanded sigma can be "contiguous" while retaining stride 0.
    return torch.empty_like(value, device='cpu', memory_format=torch.contiguous_format).copy_(value.detach())


def _tensor_hash(value):
    item = _cpu_copy(value)
    description = f'{item.dtype}:{tuple(item.shape)}:'.encode()
    return sha256(description + item.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _signature(function, kwargs):
    name = function.__name__
    wrapper = None
    if name == 'sampler_function_wrapper' and function.__module__.endswith('impact_sampling'):
        closure = inspect.getclosurevars(function).nonlocals
        original = closure.get('orig_sampler_function')
        sampler_name = closure.get('sampler_name')
        total = closure.get('total_sigmas')
        if (sampler_name not in {'dpmpp_2m_sde', 'dpmpp_3m_sde'}
                or getattr(original, '__name__', '') != 'sample_' + sampler_name
                or not isinstance(total, torch.Tensor)):
            raise ValueError('branch sampler wrapper is not supported')
        # Impact's wrapper only uses full-schedule extrema to seed Brownian noise.
        wrapper = {'source_sha256': _file_hash(inspect.getsourcefile(function)),
                   'noise_min': float(total[total > 0].min()), 'noise_max': float(total.max())}
        function, name = original, original.__name__
    if name not in SUPPORTED:
        raise ValueError('branch sampler is not supported')
    options = {key: value for key, value in kwargs.items() if key not in {'extra_args', 'callback', 'disable'}}
    # A callable noise generator or custom solver option cannot be reconstructed.
    try:
        encoded = json.dumps(options, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError('branch sampler options are not supported') from error
    result = {'function': name, 'source_sha256': _file_hash(inspect.getsourcefile(inspect.unwrap(function))),
              'options': encoded, 'seed': (kwargs.get('extra_args') or {}).get('seed')}
    if wrapper:
        result['wrapper'] = wrapper
    return result


def _rng(device):
    result = {'cpu': torch.get_rng_state().clone()}
    if device.type == 'cuda':
        result['device'] = torch.cuda.get_rng_state(device).clone()
    return result


def _restore_rng(values, device):
    torch.set_rng_state(values['cpu'])
    if 'device' in values:
        if device.type != 'cuda':
            raise ValueError('branch device changed')
        torch.cuda.set_rng_state(values['device'], device)


class _ModelProxy:
    def __init__(self, model, call):
        self.model, self.call = model, call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def record(function, model, x, sigmas, *, cut_step, max_bytes=MAX_BYTES, **kwargs):
    signature = _signature(function, kwargs)
    if type(cut_step) is not int or cut_step < 1:
        raise ValueError('branch step is invalid')
    tape = {'contract': CONTRACT, 'status': 'recording', 'signature': signature,
            'device_type': x.device.type, 'cut_step': cut_step, 'prefix_count': 0,
            'tensors': {}, 'inputs': [], 'byte_count': 0}
    reached, sealed = False, False
    callback = kwargs.get('callback')

    def abandon(reason):
        tape.update(status='not_saved', reason=reason, tensors={}, inputs=[], byte_count=0)

    def put(name, value):
        size = value.numel() * value.element_size()
        if tape['byte_count'] + size > max_bytes:
            raise _TapeLimit('byte_limit')
        tape['tensors'][name] = _cpu_copy(value)
        tape['byte_count'] += size

    def put_rng(prefix):
        for key, value in _rng(x.device).items():
            put(f'{prefix}_{key}', value)

    try:
        put('initial', x); put('sigmas', sigmas); put_rng('initial_rng')
    except _TapeLimit:
        abandon('byte_limit')

    def call(value, sigma, **extra):
        nonlocal sealed
        if tape['status'] == 'not_saved' or sealed:
            return model(value, sigma, **extra)
        try:
            if reached:
                put('boundary', value); put('boundary_sigma', sigma)
                put_rng('boundary_rng')
                tape['status'] = 'completed'
                sealed = True
                return model(value, sigma, **extra)
            index = tape['prefix_count']
            input_record = {'x': _tensor_hash(value), 'sigma': _tensor_hash(sigma)}
            prediction = model(value, sigma, **extra)
            put(f'prediction_{index}', prediction)
            put_rng(f'rng_{index}')
            tape['inputs'].append(input_record)
            tape['prefix_count'] += 1
            return prediction
        except _TapeLimit:
            # A completed model call must never be executed a second time.
            abandon('byte_limit')
            return prediction if 'prediction' in locals() else model(value, sigma, **extra)

    def progress(event):
        nonlocal reached
        if int(event['i']) + 1 >= cut_step:
            reached = True
        if callback:
            callback(event)

    result = function(_ModelProxy(model, call), x, sigmas, **{**kwargs, 'callback': progress})
    if tape['status'] == 'recording':
        abandon('boundary_not_reached')
    if tape['status'] == 'completed':
        tape['final_sha256'] = _tensor_hash(result)
    return result, tape


def replay(function, model, x, sigmas, tape, **kwargs):
    if (tape.get('contract') != CONTRACT or tape.get('status') != 'completed'
            or tape.get('signature') != _signature(function, kwargs) or tape.get('device_type') != x.device.type):
        raise ValueError('branch sampler contract mismatch')
    values = tape['tensors']
    if _tensor_hash(x) != _tensor_hash(values['initial']) or _tensor_hash(sigmas) != _tensor_hash(values['sigmas']):
        raise ValueError('branch initial state mismatch')
    prefix_count = tape['prefix_count']
    if not 1 <= prefix_count <= 1000 or len(tape['inputs']) != prefix_count:
        raise ValueError('branch prefix is invalid')
    count, checked = 0, False
    callback = kwargs.get('callback')
    caller_rng = _rng(x.device)

    def restore(prefix):
        states = {key: values[f'{prefix}_{key}'] for key in ('cpu', 'device') if f'{prefix}_{key}' in values}
        _restore_rng(states, x.device)

    def call(value, sigma, **extra):
        nonlocal count, checked
        if count < prefix_count:
            entry = tape['inputs'][count]
            if entry != {'x': _tensor_hash(value), 'sigma': _tensor_hash(sigma)}:
                raise ValueError('branch prefix input mismatch')
            prediction = values[f'prediction_{count}'].to(device=value.device)
            restore(f'rng_{count}')
            count += 1
            return prediction
        if not checked:
            if (_tensor_hash(value) != _tensor_hash(values['boundary'])
                    or _tensor_hash(sigma) != _tensor_hash(values['boundary_sigma'])):
                raise ValueError('branch boundary mismatch')
            restore('boundary_rng')
            checked = True
        count += 1
        return model(value, sigma, **extra)

    def progress(event):
        if checked and callback:
            callback(event)

    try:
        restore('initial_rng')
        result = function(_ModelProxy(model, call), x, sigmas, **{**kwargs, 'callback': progress})
        if not checked:
            raise ValueError('branch boundary was not reached')
    except BaseException:
        _restore_rng(caller_rng, x.device)
        raise
    return result, {'boundary_exact': True, 'prefix_model_calls': 0, 'replayed_predictions': prefix_count,
                    'remaining_model_calls': count - prefix_count, 'final_sha256': _tensor_hash(result),
                    'unchanged_final': _tensor_hash(result) == tape['final_sha256']}


def save_tape(tape, directory):
    directory = Path(directory)
    if tape.get('status') != 'completed' or tape['byte_count'] > MAX_BYTES:
        raise ValueError('branch tape cannot be saved')
    tensors = tape['tensors']
    metadata = {key: value for key, value in tape.items() if key != 'tensors'}
    if len(json.dumps(metadata).encode()) > MAX_METADATA // 2:
        raise ValueError('branch metadata limit exceeded')
    temporary = directory / 'state.safetensors.tmp'
    target = directory / 'state.safetensors'
    save_file(tensors, str(temporary), metadata={'contract': CONTRACT})
    os.replace(temporary, target)
    metadata.update(archive_sha256=_file_hash(target), archive_bytes=target.stat().st_size)
    manifest = directory / 'state.json.tmp'
    manifest.write_text(json.dumps(metadata, allow_nan=False, sort_keys=True), encoding='utf-8')
    os.replace(manifest, directory / 'state.json')


def load_tape(directory):
    directory = Path(directory)
    manifest, archive = directory / 'state.json', directory / 'state.safetensors'
    if manifest.is_symlink() or archive.is_symlink() or manifest.stat().st_size > MAX_METADATA:
        raise ValueError('branch file is invalid')
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if (metadata.get('contract') != CONTRACT or archive.stat().st_size > MAX_BYTES + MAX_METADATA
            or archive.stat().st_size != metadata.get('archive_bytes')
            or _file_hash(archive) != metadata.get('archive_sha256')):
        raise ValueError('branch archive changed')
    with safe_open(str(archive), framework='pt', device='cpu') as stream:
        if (stream.metadata() or {}).get('contract') != CONTRACT or len(stream.keys()) > 3100:
            raise ValueError('branch archive contract mismatch')
    metadata['tensors'] = load_file(str(archive), device='cpu')
    size = sum(value.numel() * value.element_size() for value in metadata['tensors'].values())
    if size != metadata.get('byte_count') or size > MAX_BYTES:
        raise ValueError('branch tensor byte count mismatch')
    return metadata
