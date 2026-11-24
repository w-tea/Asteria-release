"""Compare saved Library conditions, not the appearance or quality of a result."""
from __future__ import annotations

from app.schemas.action_contracts import LibraryComparisonItem, LibraryComparisonResult


_CONDITION_FIELDS = (
    'mode', 'width', 'height', 'size', 'sampler', 'scheduler', 'steps', 'cfg', 'cfg_scale',
    'denoise', 'denoise_strength', 'seed', 'fps', 'duration_seconds', 'duration',
    'frames', 'model_profile', 'model', 'prompt', 'negative_prompt', 'description', 'lyrics',
    'top_k', 'tiled_decode', 'hires_enabled',
)
_AV_SAMPLING_FIELDS = (
    'steps', 'sampler', 'scheduler', 'video_shift', 'audio_shift', 'batch_count', 'seed_policy',
    'text_guidance', 'top_k', 'denoise', 'tiled_decode', 'execution_profile',
)
_AV_REQUEST_GROUPS = {
    'size': ('width', 'height', 'aspect_ratio', 'resolution_intent'),
    'sampling': _AV_SAMPLING_FIELDS,
    'audio': ('enabled', 'prompt'),
}
_FINISHED = {'completed', 'completed_with_warnings'}
_NESTED_FIELDS = ('model_slots', 'recorded_conditions', 'requested_conditions')
_IGNORED_FIELDS = {'generation_id', 'artifacts'}
_MISSING = object()


def _scalars(document: dict, fields: tuple[str, ...]) -> dict:
    return {key: document[key] for key in fields
            if key in document and type(document[key]) in {str, int, float, bool, type(None)}}


def _group(values: dict, document: dict, name: str, fields: tuple[str, ...]) -> None:
    group = document.get(name)
    if isinstance(group, dict):
        values.update((f'{name}.{key}', value) for key, value in _scalars(group, fields).items())


def _conditions(settings: dict, *, recorded: bool) -> dict:
    values = _scalars(settings, _CONDITION_FIELDS)
    seed = settings.get('resolved_seed') if recorded else None
    if type(seed) is not int:
        seed = values.get('seed')
    if type(seed) is int:
        values['seed'] = str(seed)
    return values


def compare_library_items(rows: list[dict]) -> LibraryComparisonResult:
    items = []
    for row in rows:
        settings = row['settings']
        request = settings.get('request_snapshot')
        recorded = _conditions(settings, recorded=True)
        requested = _conditions(request, recorded=False) if isinstance(request, dict) else {}
        kind = row['media_kind']
        if isinstance(request, dict) and kind in {'video', 'audio'}:
            requested.update(_scalars(request, ('duration_intent', 'language_intent')))
            for name, fields in _AV_REQUEST_GROUPS.items():
                _group(requested, request, name, fields)
        # Initial video history can already contain the normalized frame
        # count. Do not report it as an executed condition before completion.
        if row['status'] in _FINISHED:
            if kind == 'video':
                recorded.update(_scalars(settings, ('actual_duration_seconds', 'actual_frame_count')))
            elif kind == 'audio':
                recorded.update(_scalars(settings, ('effective_duration_seconds', 'effective_audio_frames', 'execution_profile')))
                _group(recorded, settings, 'sampling', (*_AV_SAMPLING_FIELDS, 'ksampler_cfg'))
        items.append(LibraryComparisonItem(
            generation_id=row['id'], media_kind=kind, profile=row['profile'],
            prompt=row['prompt'], expanded_prompt=row['expanded_prompt'], negative_prompt=row['negative_prompt'],
            model_slots=row['model_slots'], loras=row['loras'],
            recorded_conditions=recorded, requested_conditions=requested,
            artifacts=[artifact for artifact in row['artifacts'] if artifact['kind'] in {'image', 'video', 'audio'}],
        ))
    documents = [item.model_dump() for item in items]
    baseline, *remaining = documents
    different_fields = [
        key for key, value in baseline.items()
        if key not in _IGNORED_FIELDS and key not in _NESTED_FIELDS and any(other[key] != value for other in remaining)
    ]
    for field in _NESTED_FIELDS:
        keys = sorted({key for document in documents for key in document[field]})
        different_fields.extend(f'{field}.{key}' for key in keys if any(
            other[field].get(key, _MISSING) != baseline[field].get(key, _MISSING) for other in remaining
        ))
    return LibraryComparisonResult(items=items, different_fields=sorted(different_fields))
