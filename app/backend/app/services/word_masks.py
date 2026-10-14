"""Image-bound, registered cross-attention maps. No inference at editing time."""
from __future__ import annotations

from contextlib import suppress
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from safetensors import safe_open

from app.services.atomic_json import write_json_atomic
from app.services.media_artifacts import file_sha256

CONTRACT = 'asteria.word-maps.v1'
MAX_BYTES = 64 * 1024 * 1024
MAX_METADATA = 4 * 1024 * 1024


def image_identity(path: Path) -> dict:
    # PNG metadata is finalized after this receipt. Bind the actual pixels,
    # not the container bytes that will include this receipt's own hash.
    try:
        with Image.open(path) as source:
            if source.width * source.height > 64_000_000:
                raise ValueError('word_map_image_too_large')
            pixels = ImageOps.exif_transpose(source).convert('RGBA')
            return {'width': pixels.width, 'height': pixels.height,
                    'pixel_sha256': sha256(pixels.tobytes()).hexdigest()}
    except OSError as exc:
        raise ValueError('word_map_source_changed') from exc


def read_json(path: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > MAX_METADATA:
        raise ValueError('word_map_file_invalid')
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('word_map_file_invalid')
    return result


def check_archive(directory: Path, record: dict) -> tuple[Path, list[int]]:
    name = record['archive']
    if not re.fullmatch(r'\d+\.safetensors', name):
        raise ValueError('word_map_file_invalid')
    archive = directory / name
    if (archive.is_symlink() or archive.resolve().parent != directory
            or archive.stat().st_size != record['archive_bytes'] or archive.stat().st_size > MAX_BYTES
            or file_sha256(archive) != record['archive_sha256']):
        raise ValueError('word_map_archive_changed')
    with safe_open(str(archive), framework='numpy') as tensors:
        if list(tensors.keys()) != ['maps']:
            raise ValueError('word_map_tensor_invalid')
        view = tensors.get_slice('maps')
        shape = view.get_shape()
        if (len(shape) != 4 or any(v < 1 for v in shape) or shape[2] > 96 or shape[3] > 96
                or math.prod(shape)*4 > MAX_BYTES or view.get_dtype() != 'F32'
                or shape[1] != len(record['columns'])):
            raise ValueError('word_map_tensor_invalid')
    return archive, shape


class WordMapSession:
    def __init__(self):
        self.directory: Path | None = None
        self.root: Path | None = None
        self.committed = False

    def __enter__(self): return self

    def __exit__(self, *_):
        if self.directory is None or self.committed:
            return
        if self.directory.is_symlink() or self.directory.resolve().parent != self.root / '_word_maps':
            return
        with suppress(OSError):
            for path in self.directory.iterdir():
                if re.fullmatch(r'(owner|images)\.json|\d+\.(json|safetensors)(\.tmp)?', path.name):
                    path.unlink(missing_ok=True)
            self.directory.rmdir()

    def begin(self, root: Path, profile: str) -> dict:
        self.root = Path(root).resolve()
        parent = self.root / '_word_maps'
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError('word_map_path_invalid')
        self.directory = parent / uuid4().hex
        self.directory.mkdir()
        write_json_atomic(self.directory / 'owner.json', {
            'contract': CONTRACT, 'output_root': str(self.root), 'operation_id': self.directory.name})
        return {'directory': str(self.directory), 'profile': profile}

    def finish(self, outputs: list[dict], status: str) -> tuple[dict, list[str]]:
        if status != 'completed' or not outputs or self.directory is None:
            return {'status': 'not_saved', 'reason': 'generation_incomplete'}, []
        try:
            images, paths, seen = [], [str(self.directory / 'owner.json')], {}
            selected_rows, implicit_seeds = {}, set()
            for output in outputs:
                seed = output['seed']
                if type(seed) is not int or seed < 0:
                    raise ValueError('word_map_seed_missing')
                record_path = self.directory / f'{seed}.json'
                record = read_json(record_path)
                if record.get('status') != 'recorded':
                    raise ValueError(record.get('reason', 'word_map_not_recorded'))
                archive, shape = check_archive(self.directory, record)
                row = output.get('image_index', seen.get(seed, 0)); seen[seed] = seen.get(seed, 0) + 1
                if 'image_index' not in output:
                    implicit_seeds.add(seed)
                rows = selected_rows.setdefault(seed, set())
                if type(row) is not int or not 0 <= row < shape[0] or row in rows:
                    raise ValueError('word_map_batch_ambiguous')
                rows.add(row)
                image = Path(output['path']).resolve(strict=True)
                if not image.is_relative_to(self.root):
                    raise ValueError('word_map_image_outside_root')
                images.append({'path': str(image), **image_identity(image),
                               'seed': seed, 'row': row, 'record': record_path.name,
                               'record_sha256': file_sha256(record_path)})
                paths.extend([str(record_path), str(archive)])
            # Explicit original indices can select a checked batch's passing subset.
            # Without them, a missing row cannot be assigned safely.
            for seed in implicit_seeds:
                _, shape = check_archive(self.directory, read_json(self.directory / f'{seed}.json'))
                if shape[0] != seen[seed]:
                    raise ValueError('word_map_batch_ambiguous')
            manifest = self.directory / 'images.json'
            write_json_atomic(manifest, {'contract': CONTRACT, 'images': images})
            paths.append(str(manifest))
            return {'status': 'recorded', 'directory': str(self.directory), 'manifest_sha256': file_sha256(manifest),
                    'image_count': len(images), 'stage': 'base'}, list(dict.fromkeys(paths))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {'status': 'not_saved', 'reason': str(exc)[:256]}, []

    def commit(self): self.committed = True


def resolve_maps(item: dict) -> tuple[Path, list[dict], set[str]]:
    settings = item.get('settings') or {}
    saved = settings.get('word_masks') or {}
    if item.get('status') != 'completed' or saved.get('status') != 'recorded':
        raise ValueError('word_map_not_available')
    original = Path(saved['directory'])
    directory = original.resolve(strict=True)
    root = Path(settings['output_root']).resolve(strict=True)
    if (original.is_symlink() or directory.parent != root / '_word_maps'
            or not re.fullmatch(r'[0-9a-f]{32}', directory.name)):
        raise ValueError('word_map_path_invalid')
    if read_json(directory / 'owner.json') != {
            'contract': CONTRACT, 'output_root': str(root), 'operation_id': directory.name}:
        raise ValueError('word_map_owner_changed')
    registered = {str(Path(path).resolve()) for path in item.get('output_paths', [])}
    manifest = directory / 'images.json'
    if str(manifest) not in registered:
        raise ValueError('word_map_not_registered')
    if file_sha256(manifest) != saved['manifest_sha256']:
        raise ValueError('word_map_manifest_changed')
    contents = read_json(manifest)
    if contents.get('contract') != CONTRACT:
        raise ValueError('word_map_file_invalid')
    return directory, contents['images'], registered


def checked_image(directory: Path, image: dict, registered: set[str]) -> tuple[dict, Path]:
    if not re.fullmatch(r'\d+\.json', image['record']):
        raise ValueError('word_map_file_invalid')
    path = directory / image['record']
    source = Path(image['path'])
    if str(path) not in registered or str(source.resolve()) not in registered:
        raise ValueError('word_map_not_registered')
    if (source.is_symlink() or image_identity(source) != {
            key: image[key] for key in ('width', 'height', 'pixel_sha256')}
            or file_sha256(path) != image['record_sha256']):
        raise ValueError('word_map_source_changed')
    record = read_json(path)
    archive, shape = check_archive(directory, record)
    if str(archive) not in registered:
        raise ValueError('word_map_not_registered')
    if not 0 <= image['row'] < shape[0]:
        raise ValueError('word_map_batch_ambiguous')
    return record, archive


def word_mask_details(item: dict) -> dict:
    directory, images, registered = resolve_maps(item)
    results = []
    for index, image in enumerate(images):
        record, _ = checked_image(directory, image, registered)
        columns = [c for c in record['contract']['columns'] if c['position'] in record['columns']]
        results.append({**{k: image[k] for k in ('path', 'width', 'height', 'seed')},
                        'index': index, 'columns': columns})
    return {'images': results, 'stage': 'base'}


def make_word_mask(item: dict, index: int, columns: list[int], threshold: float, expand: float) -> Image.Image:
    directory, images, registered = resolve_maps(item)
    if (type(index) is not int or not 0 <= index < len(images) or not columns or len(columns) > 256
            or not math.isfinite(threshold) or not .01 <= threshold <= .99
            or not math.isfinite(expand) or not -10 <= expand <= 10):
        raise ValueError('word_map_selection_invalid')
    image = images[index]; record, archive = checked_image(directory, image, registered)
    if any(type(c) is not int or c not in record['columns'] for c in columns):
        raise ValueError('word_map_selection_invalid')
    with safe_open(str(archive), framework='numpy') as tensors:
        view = tensors.get_slice('maps')
        parts = [view[image['row'], record['columns'].index(c)] for c in set(columns)]
        heat = np.mean(parts, axis=0)
    if not np.isfinite(heat).all() or heat.max() <= heat.min():
        raise ValueError('word_map_no_range')
    heat = (heat - heat.min()) / (heat.max() - heat.min())
    mask = Image.fromarray(np.uint8(heat >= threshold)*255)
    radius = round(abs(expand)*max(mask.size)/100)
    if radius:
        mask = mask.filter((ImageFilter.MaxFilter if expand > 0 else ImageFilter.MinFilter)(radius*2+1))
    width, height = image['width'], image['height']
    if width < 1 or height < 1 or width*height > 64*1024*1024:
        raise ValueError('word_map_image_limit')
    return mask.resize((width, height), Image.Resampling.BILINEAR)
