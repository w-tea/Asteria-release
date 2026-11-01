"""Use native prefetch without CUDA capture only within the fixed Music3 AR call."""
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
import logging
from pathlib import Path

PREFETCH_SHA256 = '2a1da9c539de1404aa530a9c456102731c9c3392adab35c45d928fa305448409'
AR_SHA256 = '25927897f219685bc3a80341c97285065070842c5a161f2caa6ef123be9b6671'


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def install(prefetch, ar, *, prefetch_sha256, ar_sha256):
    original_pop = prefetch.prefetch_queue_pop
    original_generate = ar.MiniMaxMusic3AR.generate
    if (prefetch_sha256 != PREFETCH_SHA256 or ar_sha256 != AR_SHA256
            or getattr(original_pop, '_asteria_music_native_prefetch', False)
            or getattr(original_generate, '_asteria_music_native_prefetch', False)):
        return False
    scope = ContextVar('asteria_music3_native_prefetch', default=None)

    @wraps(original_pop)
    def pop(queue, device, module, dtype=None, core=None, enable_graph=False, generator=None):
        current = scope.get()
        if current is not None and enable_graph:
            current['bypassed'] += 1
            enable_graph = False
        return original_pop(queue, device, module, dtype=dtype, core=core,
                            enable_graph=enable_graph, generator=generator)

    @wraps(original_generate)
    def generate(self, *args, **kwargs):
        current = {'bypassed': 0}
        token = scope.set(current)
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            scope.reset(token)
            logging.info('Asteria Music3 native prefetch: graph requests bypassed=%d', current['bypassed'])

    pop._asteria_music_native_prefetch = True
    generate._asteria_music_native_prefetch = True
    prefetch.prefetch_queue_pop = pop
    ar.MiniMaxMusic3AR.generate = generate
    return True


def install_runtime():
    try:
        import comfy.model_prefetch as prefetch
        import comfy.ldm.minimax_music.ar as ar
        installed = install(prefetch, ar, prefetch_sha256=file_sha256(prefetch.__file__),
                            ar_sha256=file_sha256(ar.__file__))
    except (ImportError, OSError, AttributeError):
        logging.warning('Asteria Music3 native prefetch: not installed; runtime source unavailable')
        return False
    logging.info('Asteria Music3 native prefetch: %s',
                 'installed for fixed runtime' if installed else 'not installed for this runtime or already installed')
    return installed
