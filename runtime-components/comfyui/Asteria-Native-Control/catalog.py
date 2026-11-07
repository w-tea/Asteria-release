"""Discover control adapters by their tensor contract, never their filename."""
import json
import struct
from functools import lru_cache


def control_profile(header):
    header = {key.removeprefix('diffusion_model.'): value for key, value in header.items()}
    def shape(name):
        return header.get(name, {}).get('shape', [])
    if shape('first.weight') == [6144, 128] and any(
        k.startswith('blocks.') and k.endswith(('.A', '.lora_A.weight', '.lora_down.weight')) for k in header
    ):
        return 'krea2'
    if (shape('video_embedder.weight') == [3072, 384] or shape('video_embedder.set_weight') == [3072, 384]) and any(
        k.startswith('layers.') and '.lora_' in k for k in header
    ):
        return 'moss_image'
    return None


@lru_cache(maxsize=256)
def _inspect(path, size, modified):
    try:
        with open(path, 'rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack('<Q', raw)[0]
            if not 0 < length <= min(size - 8, 16 * 1024**2):
                return None
            return control_profile(json.loads(stream.read(length)))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def control_models(profile):
    import folder_paths
    from pathlib import Path
    result = []
    for name in folder_paths.get_filename_list('loras'):
        path = folder_paths.get_full_path('loras', name)
        if not path:
            continue
        try:
            stat = Path(path).stat()
            if _inspect(path, stat.st_size, stat.st_mtime_ns) == profile:
                result.append(name)
        except OSError:
            continue
    return result
