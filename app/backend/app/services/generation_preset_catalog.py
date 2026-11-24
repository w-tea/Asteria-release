"""Find inputs in the same saved collections used by Generate."""
from __future__ import annotations

from app.core.config import AppConfig
from app.schemas.action_contracts import GenerationPresetQuery, SavedGenerationPreset, SavedGenerationPresetList
from app.schemas.generation import InpaintRequest
from app.services.advanced_styles import AdvancedStylesRepository
from app.services.presets import PresetRepository
from app.services.prompt_pipeline import list_wildcards
from app.services.prompt_presets import PromptPresetRepository


# Preset.data is flexible for the desktop UI. Only generation inputs and its
# saved aliases are relevant to these callers, not arbitrary extra keys.
_GENERATION_FIELDS = InpaintRequest.model_fields.keys() | {'mode', 'runtime_adapter', 'model_slots'}


def read_generation_presets(config: AppConfig, request: GenerationPresetQuery) -> SavedGenerationPresetList:
    include_values = bool(request.id)
    if request.kind == 'generation':
        items = [SavedGenerationPreset(id=item.id, name=item.name, data={
            key: value for key, value in item.data.items() if key in _GENERATION_FIELDS
        } if include_values and item.id == request.id else None)
            for item in PresetRepository(config.data_dir / 'presets.json').list('generation', profile=request.profile)]
    elif request.kind == 'prompt':
        from app.api.prompt_presets import prompt_preset_to_item

        items = [SavedGenerationPreset(id=item.id, name=item.name,
            data=prompt_preset_to_item(item).model_dump(mode='json') if include_values and item.id == request.id else None)
            for item in PromptPresetRepository(config.data_dir / 'prompt-presets.json').list()]
    elif request.kind == 'style':
        items = [SavedGenerationPreset(id=item.id, name=item.name, data={
            'source_type': 'style', 'id': item.id, 'group': item.group, 'name': item.name,
            'prompt': item.prompt, 'negative': item.negative,
        } if include_values and item.id == request.id else None)
            for item in AdvancedStylesRepository(config.data_dir).list().styles]
    else:
        items = [SavedGenerationPreset(id=item.name, name=item.name, data={
            'token': f'__{item.name}__', 'option_count': item.option_count,
        } if include_values and item.name == request.id else None)
            for item in list_wildcards(config.wildcard_dir)]

    query = request.query.strip().casefold()
    matches = [item for item in items if (not request.id or item.id == request.id)
               and (not query or query in item.name.casefold() or query in item.id.casefold())]
    end = request.offset + request.limit
    return SavedGenerationPresetList(items=matches[request.offset:end], next_offset=end if end < len(matches) else None)
