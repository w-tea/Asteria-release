"""Stage receipts and bounded pixel differences; not semantic judgements."""
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


def stage_receipt(path: Path, kind: str, *, seed: int, batch_index: int, image_index: int,
                  runtime_job_id: str, ordinal: int) -> dict:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        digest = sha256(image.tobytes()).hexdigest()
        width, height = image.size
    return {'path': str(path), 'kind': kind, 'seed': seed, 'batch_index': batch_index,
            'image_index': image_index, 'runtime_job_id': runtime_job_id, 'ordinal': ordinal,
            'width': width, 'height': height, 'pixel_sha256': digest}


def copy_stage(source: Path, output_dir: Path, kind: str, **identity) -> dict:
    directory = output_dir / '_stages'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{uuid4().hex}_{kind}.png'
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).convert('RGB').save(path, format='PNG')
    return stage_receipt(path, kind, **identity)


def finalize_stages(settings: dict, recorded: list[dict], primary: list[dict], output_paths: list[str]) -> None:
    if not settings.get('record_enhancement_stages'):
        return
    stages = list(recorded)
    batch_size = max(1, int(settings.get('batch_size', 1)))
    if settings.get('hires_upscaler') == 'Topaz Photo AI':
        batch_size = 1
    for index, item in enumerate(primary):
        batch_index = item.get('batch_index', index // batch_size)
        image_index = item.get('image_index', index % batch_size)
        stages.append(stage_receipt(Path(item['path']), 'final', seed=item.get('seed', settings.get('seed', -1)),
                      batch_index=batch_index, image_index=image_index, runtime_job_id=item.get('runtime_job_id', ''),
                      ordinal=10000))
    for item in stages:
        if item['path'] not in output_paths:
            output_paths.append(item['path'])
    settings['enhancement_stages'] = stages
