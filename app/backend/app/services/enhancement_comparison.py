import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

from app.adapters.image_check_image import _read_source, MAX_SOURCE_PIXELS
from app.services.image_conditions import registered_output_path


def compare_stages(item: dict, before: str, after: str, current_root: Path) -> dict:
    before = registered_output_path(item, before)
    after = registered_output_path(item, after)
    settings = item.get('settings', {})
    records = {entry['path']: entry for entry in settings.get('enhancement_stages', [])}
    if before not in records or after not in records:
        raise ValueError('stage_image_unavailable')
    left, right = records[before], records[after]
    if any(left[key] != right[key] for key in ('seed', 'batch_index', 'image_index')):
        raise ValueError('stage_pair_mismatch')
    root = Path(settings.get('output_root') or current_root)
    images, sizes = [], []
    for record in (left, right):
        _, _, content = _read_source(record['path'], root)
        with Image.open(BytesIO(content)) as source:
            if source.width * source.height > MAX_SOURCE_PIXELS:
                raise ValueError('stage_image_too_large')
            image = ImageOps.exif_transpose(source).convert('RGB')
        if sha256(image.tobytes()).hexdigest() != record['pixel_sha256']:
            raise ValueError('stage_image_changed')
        sizes.append(list(image.size))
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        images.append(image)
    size = images[0].size
    diff = ImageChops.difference(images[0], images[1].resize(size, Image.Resampling.LANCZOS))
    mean = sum(ImageStat.Stat(diff).mean) / (3 * 255)
    maximum = ImageChops.lighter(ImageChops.lighter(*diff.split()[:2]), diff.split()[2])
    histogram = maximum.histogram()
    changed = sum(histogram[9:]) / (size[0] * size[1])
    data = BytesIO(); diff.save(data, format='PNG')
    return {'preview': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode('ascii'),
            'mean_absolute_difference': mean, 'changed_fraction': changed, 'channel_threshold': 8,
            'comparison_size': list(size), 'before_size': sizes[0], 'after_size': sizes[1],
            'aspect_changed': abs(sizes[0][0] / sizes[0][1] - sizes[1][0] / sizes[1][1]) > .001}
