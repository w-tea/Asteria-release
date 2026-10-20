"""Read-only, bounded sampling observations. Native sampling owns every calculation."""
from copy import deepcopy
from hashlib import sha256
import json
from threading import local
from time import perf_counter
from uuid import uuid4

import torch

MAX_TENSOR_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 512 * 1024
MAX_RECORDS = 32


def execution_identity():
    try:
        from comfy_execution.utils import get_executing_context
        context = get_executing_context()
        return context.prompt_id if context is not None else None
    except Exception:
        return None


def tensor_identity(tensor):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor) or tensor.is_nested or tensor.layout != torch.strided:
        return {'status': 'unrecorded', 'reason': 'unsupported_tensor'}
    identity = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
    if tensor.numel() * tensor.element_size() > MAX_TENSOR_BYTES:
        return {**identity, 'status': 'unrecorded', 'reason': 'tensor_byte_limit'}
    # detach alone shares storage. Copy before reading; never cast or modify input.
    value = tensor.detach().to(device='cpu', copy=True).contiguous()
    identity['sha256'] = sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()
    return identity


class SamplingCapture:
    def __init__(self):
        self.pending = []
        self.previous = []
        self.byte_count = 0
        self.truncated = False
        self.previous_truncated = False
        self.attempted = False
        self.pending_execution = None
        self.current = local()

    def seal(self):
        self.previous = self.pending
        self.previous_truncated = self.truncated
        self.pending = []
        self.byte_count = 0
        self.truncated = False
        self.attempted = False

    def sample(self, executor, noise, latent, sampler, sigmas, *args, **kwargs):
        started = perf_counter()
        prompt_id = execution_identity()
        if self.attempted and prompt_id != self.pending_execution:
            self.seal()
        self.pending_execution = prompt_id
        seed = args[3] if len(args) > 3 else kwargs.get('seed')
        try:
            function = sampler.sampler_function
            owner = getattr(executor, 'class_obj', None)
            mask = args[0] if args else kwargs.get('denoise_mask')
            row = {'status': 'completed' if prompt_id is not None else 'partial',
                'source_prompt_id': prompt_id, 'capture_id': uuid4().hex,
                'contract': {'seed': seed, 'cfg': getattr(owner, 'cfg', None),
                    'sampler': [function.__module__, function.__qualname__],
                    'extra_options': sampler.extra_options, 'inpaint_options': sampler.inpaint_options,
                    'mask': tensor_identity(mask)},
                'used': {'noise': tensor_identity(noise), 'sigmas': tensor_identity(sigmas)},
                'latent': tensor_identity(latent),
                'devices': {'noise': str(noise.device), 'latent': str(latent.device), 'sigmas': str(sigmas.device)}}
            if sigmas.ndim == 1 and sigmas.numel() <= 10001:
                row['sigma_values'] = sigmas.detach().to(device='cpu', copy=True).tolist()
            row['observation_ms'] = (perf_counter() - started) * 1000
        except Exception:
            # Observation failure must not reject an otherwise valid generation.
            row = {'status': 'partial', 'reason': 'observation_unavailable'}
        previous = getattr(self.current, 'row', None)
        self.current.row = row
        try:
            result = executor(noise, latent, sampler, sigmas, *args, **kwargs)
        finally:
            self.current.row = previous
        # Only a successful native return can yield a completed execution receipt.
        self.attempted = True
        try:
            body = json.dumps(row, allow_nan=False).encode('utf-8')
            if (len(self.pending) >= MAX_RECORDS or len(body) > MAX_RECORD_BYTES
                    or self.byte_count + len(body) > MAX_TOTAL_BYTES):
                self.truncated = True
            else:
                self.pending.append(json.loads(body))
                self.byte_count += len(body)
        except (TypeError, ValueError):
            self.truncated = True
        return result

    def compute(self, executor, model, sigmas, extra_args, callback, noise, latent=None, *args, **kwargs):
        row = getattr(self.current, 'row', None)
        if row is not None:
            try:
                row['compute_devices'] = {'noise': str(noise.device), 'sigmas': str(sigmas.device),
                                          'latent': None if latent is None else str(latent.device)}
            except Exception:
                row['compute_devices_unavailable'] = True
        return executor(model, sigmas, extra_args, callback, noise, latent, *args, **kwargs)

    def take(self):
        # A failed downstream node may leave observations that were never collected.
        # Native cache reuse in the next prompt is still reuse, not a fresh execution.
        reused = not self.attempted or self.pending_execution != execution_identity()
        if self.attempted:
            self.seal()
        rows = deepcopy(self.previous)
        status = ('partial' if self.previous_truncated or any(r.get('status') != 'completed' for r in rows)
                  else 'recorded' if rows else 'unavailable')
        return {'version': 1, 'status': status, 'records': rows, 'reused': reused,
                'truncated': self.previous_truncated}


class AsteriaSamplingReceiptModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}}

    RETURN_TYPES = ('MODEL', 'ASTERIA_SAMPLING_CAPTURE')
    FUNCTION = 'attach'
    CATEGORY = 'Asteria/internal'

    def attach(self, model):
        import comfy.patcher_extension
        capture = SamplingCapture()
        result = model.clone()
        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                    'asteria_sampling_receipt', capture.sample)
        result.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                                    'asteria_sampling_receipt_device', capture.compute)
        return result, capture


class AsteriaSamplingReceiptOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'after': ('*',), 'capture': ('ASTERIA_SAMPLING_CAPTURE',)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Refresh only this output leaf. A cached native sampler stays cached.
        return float('nan')

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'record'
    CATEGORY = 'Asteria/internal'

    def record(self, after, capture):
        return {'ui': {'asteria_sampling_receipt': [capture.take()]}, 'result': ()}


NODE_CLASS_MAPPINGS = {'AsteriaSamplingReceiptModel': AsteriaSamplingReceiptModel,
                       'AsteriaSamplingReceiptOutput': AsteriaSamplingReceiptOutput}
