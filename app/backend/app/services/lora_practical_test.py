"""Held-out LoRA evaluation using the existing fixed-pair generation executor."""
from hashlib import sha256
import json
from pathlib import Path
import re
import secrets
import tomllib

from app.schemas.common import LoraSelection
from app.schemas.lora_practical_test import PracticalTestRequest
from app.schemas.prompt_comparison import PromptComparisonChange, PromptComparisonRequest
from app.services.prompt_comparison import prepare_comparison
from app.services.prompt_pipeline import extract_loras

_IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def file_hash(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('practical_source_changed')
    return digest.hexdigest()


def _path(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def training_directory(run, root: Path) -> Path:
    if not run.image_dir:
        raise ValueError('practical_training_unavailable')
    return _path(run.image_dir, root)


def training_snapshot(run, project_root: Path) -> dict:
    image_dir = training_directory(run, project_root)
    extension, fallback = '.txt', ''
    config_path = getattr(run, 'dataset_config_path', '')
    if config_path:
        config = tomllib.loads(_path(config_path, project_root).read_text(encoding='utf-8-sig'))
        general = config.get('general', {})
        for dataset in config.get('datasets', []):
            for subset in dataset.get('subsets', []):
                if _path(subset.get('image_dir', ''), project_root) == image_dir:
                    extension = subset.get('caption_extension', dataset.get('caption_extension', general.get('caption_extension', '.txt')))
                    fallback = subset.get('class_tokens', '')
    if not re.fullmatch(r'\.[a-zA-Z0-9]+', extension):
        raise ValueError('practical_training_unavailable')
    files = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGES), key=lambda p: p.name)
    if not files or len(files) > 512:
        raise ValueError('practical_training_unavailable')
    items = []
    for image in files:
        caption = image.with_suffix(extension)
        if caption.is_file() and caption.stat().st_size > 1024 * 1024:
            raise ValueError('practical_training_unavailable')
        items.append({'name': image.name, 'image_sha256': file_hash(image),
                      'caption': caption.read_text(encoding='utf-8-sig') if caption.is_file() else fallback,
                      'caption_sha256': file_hash(caption) if caption.is_file() else None,
                      'caption_source': 'file' if caption.is_file() else 'class_tokens'})
    content = {'origin': 'current_materialized_files', 'run_id': run.id, 'dataset_id': run.dataset_id,
               'image_dir': str(image_dir), 'caption_extension': extension, 'items': items}
    return {**content, 'fingerprint': sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def training_overlaps(scenes, training: dict) -> list[dict]:
    overlaps = []
    normalize = lambda text: re.sub(r'\s+', ' ', text.replace('_', ' ').casefold()).strip()
    for index, scene in enumerate(scenes):
        for field in ('clothing', 'background', 'pose'):
            value = normalize(getattr(scene, field))
            matches = [item['name'] for item in training['items']
                       if re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', normalize(item['caption']))]
            if matches:
                overlaps.append({'scene': index, 'field': field, 'value': getattr(scene, field), 'items': matches})
    return overlaps


def prepare_practical_test(request: PracticalTestRequest, run, training: dict, *, wildcard_dir: Path, assets: list) -> dict:
    if run.id != request.run_id or run.status != 'completed' or request.model_index >= len(run.registered_model_paths):
        raise ValueError('practical_run_unavailable')
    if request.base.profile != run.profile:
        raise ValueError('practical_profile_mismatch')
    if request.training_fingerprint != training['fingerprint']:
        raise ValueError('practical_source_changed')
    if training_overlaps(request.scenes, training):
        raise ValueError('practical_training_overlap')
    seeds = []
    while len(seeds) < request.seed_count:
        seed = secrets.randbits(32)
        if seed not in seeds:
            seeds.append(seed)
    base = request.base.model_copy(deep=True)
    # Variation noise is also shared across scenes, not redrawn per scene.
    if base.subseed < 0:
        base.subseed = secrets.randbits(32)
    model = run.registered_model_paths[request.model_index]
    base.loras = [LoraSelection(name=model, weight=0, clip_weight=None)]
    base.prompt_input = base.prompt_layout = base.prompt_pins = None
    base.advanced_styles = []
    base.hires_prompt = ''
    plan = None
    mapping = []
    for scene_index, scene in enumerate(request.scenes):
        base.prompt = ', '.join(part.strip(' ,\n') for part in (request.common_prompt, scene.clothing, scene.background, scene.pose))
        part = prepare_comparison(PromptComparisonRequest(name=request.name, base=base, pairs=request.seed_count,
            seeds=seeds, change=PromptComparisonChange(kind='lora', index=0, weight=request.weight)), wildcard_dir=wildcard_dir)
        offset = scene_index * request.seed_count
        for cell in part['cells']:
            if extract_loras(cell['request']['prompt']):
                raise ValueError('practical_inline_lora')
            cell['pair'] += offset
            cell['index'] += offset * 2
        mapping.extend({'scene': scene_index, 'seed': seed} for seed in seeds)
        if plan is None:
            plan = part
        else:
            for key in ('cells', 'seeds', 'display_orders'):
                plan[key].extend(part[key])
    plan['lora_test'] = {'version': 1, 'run_id': run.id, 'dataset_id': run.dataset_id, 'dataset_name': run.dataset_name,
                        'model': model, 'model_index': request.model_index, 'assets': assets, 'training': training,
                        'common_prompt': request.common_prompt, 'learned_features': request.learned_features,
                        'scenes': [scene.model_dump() for scene in request.scenes], 'pairs': mapping,
                        'reviewed_training': True, 'ratings': {}, 'rating_history': []}
    return plan


def verify_practical_sources(plan: dict, run, project_root: Path) -> None:
    saved = plan.get('lora_test')
    if not saved:
        return
    try:
        if (run is None or run.id != saved['run_id'] or run.status != 'completed'
                or saved['model'] not in run.registered_model_paths
                or training_snapshot(run, project_root)['fingerprint'] != saved['training']['fingerprint']):
            raise ValueError('practical_source_changed')
        for asset in saved['assets']:
            if file_hash(Path(asset['path'])) != asset['sha256']:
                raise ValueError('practical_source_changed')
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError('practical_source_changed') from exc


def practical_summary(plan: dict) -> dict:
    count = len(plan['seeds'])
    complete = sum(all(cell['status'] == 'completed' for cell in plan['cells'] if cell['pair'] == pair) for pair in range(count))
    return {'planned_pairs': count, 'complete_pairs': complete, 'incomplete_pairs': count - complete,
            'rated_pairs': len(plan['lora_test']['ratings'])}
