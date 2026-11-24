from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api import generation, lora_training, prompt_comparisons
from app.core.config import AppConfig
from app.schemas.lora_practical_test import PracticalTestRating, PracticalTestRequest
from app.schemas.prompt_comparison import REQUEST_TYPES
from app.services.generation_reproduction import capture_generation_assets
from app.services.lora_practical_test import prepare_practical_test, training_snapshot, training_overlaps, training_directory
from app.services.model_fingerprints import ModelFingerprintService
from app.services.model_library_cache import get_model_library
from app.services.model_profiles import ModelProfileRepository
from app.services.prompt_pipeline import extract_embeddings

router = APIRouter(prefix='/api/lora-practical-tests', tags=['lora-training'])


def get_run(run_id: str):
    run = lora_training._run_repository().get(run_id)
    if run is None or run.status != 'completed' or not run.registered_model_paths:
        raise HTTPException(404, detail={'code': 'practical_run_unavailable'})
    return run


def snapshot(run, config):
    try:
        return training_snapshot(run, config.root_dir)
    except (OSError, ValueError):
        raise HTTPException(409, detail={'code': 'practical_training_unavailable'}) from None


@router.get('/runs/{run_id}')
def training(run_id: str):
    return snapshot(get_run(run_id), AppConfig())


@router.get('/runs/{run_id}/images/{name}')
def training_image(run_id: str, name: str):
    directory = training_directory(get_run(run_id), AppConfig().root_dir)
    path = directory / name
    if (path.name != name or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            or path.resolve().parent != directory or not path.is_file()):
        raise HTTPException(404, detail={'code': 'practical_training_unavailable'})
    return FileResponse(path)


@router.get('/runs/{run_id}/comparisons')
def list_tests(run_id: str):
    # Saved tests remain accessible after their training run is removed.
    return {'items': prompt_comparisons.repository().list_practical(run_id)}


def capture_assets(plan, config):
    library = get_model_library(generation._model_root(config), config=config)
    hashes = ModelFingerprintService(config.data_dir / 'config' / 'model-fingerprints.json')
    known = [item.path.stem for item in library.by_type('embedding')]
    profile = next((p for p in ModelProfileRepository(config.model_profiles_path).list() if p.id == plan['cells'][0]['request']['profile']), None)
    found = {}
    for cell in plan['cells'][::2]:
        request = REQUEST_TYPES['txt2img'].model_validate(cell['request'])
        settings, _ = generation._settings_from_request(request, config, 'txt2img')
        slots = dict(request.model_slot_overrides)
        if request.vae.casefold() not in {'', 'automatic', 'none'}:
            slots['vae'] = request.vae
        embeddings = sorted(set(extract_embeddings(request.prompt, known) + extract_embeddings(request.negative_prompt, known)))
        receipt = capture_generation_assets(slots, settings, [lora.model_dump() for lora in request.loras], embeddings,
            library, hashes, slot_types={slot.id: slot.type for slot in profile.model_slots} if profile else {},
            allow_identical_aliases=True)
        if any(asset['status'] != 'recorded' for asset in receipt['assets']):
            raise ValueError('practical_asset_unavailable')
        for asset in receipt['assets']:
            found.setdefault(asset['path'], asset)
    return list(found.values())


@router.post('/prepare')
def prepare(request: PracticalTestRequest):
    config = AppConfig()
    run = get_run(request.run_id)
    training = snapshot(run, config)
    overlaps = training_overlaps(request.scenes, training)
    if overlaps:
        raise HTTPException(422, detail={'code': 'practical_training_overlap', 'overlaps': overlaps})
    base = request.base.model_copy(deep=True)
    base.model_slot_overrides = generation._model_slots_for_request(base, config)
    base.runtime = generation._selected_runtime_id(config, base.runtime)
    try:
        plan = prepare_practical_test(request.model_copy(update={'base': base}), run, training, wildcard_dir=config.wildcard_dir, assets=[])
        plan['lora_test']['assets'] = capture_assets(plan, config)
    except (OSError, ValueError) as exc:
        code = str(exc) if isinstance(exc, ValueError) else 'practical_asset_unavailable'
        raise HTTPException(422, detail={'code': code}) from None
    return prompt_comparisons.repository().create(plan)


@router.put('/{comparison_id}/rating')
def rate(comparison_id: str, rating: PracticalTestRating):
    prompt_comparisons._get(comparison_id)
    try:
        return prompt_comparisons.repository().rate_practical(comparison_id, rating)
    except ValueError as exc:
        raise HTTPException(409, detail={'code': str(exc)}) from None
