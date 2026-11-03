"""Freeze controlled image pairs before execution, independently of presentation order."""
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import secrets
from uuid import uuid4

from app.schemas.generation import AdvancedStyleSelection
from app.schemas.prompt_comparison import PromptComparisonRequest, REQUEST_TYPES
from app.schemas.prompt_input import join_prompt_contract_fragments
from app.services.prompt_pipeline import expand_dynamic_prompt
from app.services.prompt_styles import compose_with_advanced_styles


def _component_seed(seed: int, name: str) -> int:
    return int.from_bytes(sha256(f'{seed}/{name}'.encode()).digest()[:4], 'big')


def _component_prompts(base, change, seed, wildcard_dir):
    if base.prompt_pins and base.prompt_pins.items:
        raise ValueError('comparison_component_pins')

    def freeze(text, name):
        # Preserve the style placeholder; it is not a one-element random choice.
        return '{prompt}'.join(expand_dynamic_prompt(part, _component_seed(seed, f'{name}/{i}'), wildcard_dir).expanded
                               for i, part in enumerate(text.split('{prompt}')))

    styles = [AdvancedStyleSelection.model_validate({**style.model_dump(),
              'prompt': freeze(style.prompt, f'style/{i}/positive'), 'negative': freeze(style.negative, f'style/{i}/negative')})
              for i, style in enumerate(base.advanced_styles)]
    negative = freeze(base.negative_prompt, 'negative')
    if change.kind == 'section':
        if not base.prompt_input or base.prompt_input.mode != 'sections':
            raise ValueError('comparison_sections_required')
        sections = sorted(base.prompt_input.sections, key=lambda item: item.order)
        texts = [freeze(getattr(section, 'text', getattr(getattr(section, 'content', None), 'text', '')), f'section/{section.id}') for section in sections]
        changed = list(texts)
        selected = next(i for i, section in enumerate(sections) if section.id == change.section)
        changed[selected] = freeze(change.after, f'section/{change.section}')
        return [compose_with_advanced_styles(join_prompt_contract_fragments(parts), negative, styles) for parts in (texts, changed)]
    if change.index >= len(styles):
        raise ValueError('comparison_item_missing')
    positive = freeze(base.prompt, 'positive')
    return [compose_with_advanced_styles(positive, negative, chosen) for chosen in (styles, [s for i, s in enumerate(styles) if i != change.index])]


def prepare_comparison(request: PromptComparisonRequest, *, wildcard_dir: Path) -> dict:
    base, change = request.base, request.change
    seeds = list(request.seeds or [])
    if not seeds and base.seed >= 0:
        seeds = [(base.seed + i) % 4294967296 for i in range(request.pairs)]
    while len(seeds) < request.pairs:
        seed = secrets.randbits(32)
        if seed not in seeds:
            seeds.append(seed)
    cells = []
    for pair, seed in enumerate(seeds):
        common = base.model_dump(mode='json')
        common.update(seed=seed, subseed=base.subseed if base.subseed >= 0 else secrets.randbits(32),
                      batch_size=1, batch_count=1, prompt_input=None, prompt_layout=None, prompt_pins=None, advanced_styles=[])
        for key in ('hires_prompt', 'hires_negative_prompt'):
            common[key] = expand_dynamic_prompt(common.get(key, ''), seed, wildcard_dir).expanded
        if change.kind in {'section', 'style'}:
            prompts = _component_prompts(base, change, seed, wildcard_dir)
        else:
            positive, negative = compose_with_advanced_styles(base.prompt, base.negative_prompt, base.advanced_styles)
            prompts = [(expand_dynamic_prompt(positive, seed, wildcard_dir, pins=base.prompt_pins).expanded,
                        expand_dynamic_prompt(negative, seed, wildcard_dir).expanded)] * 2
        left, right = [{**deepcopy(common), 'prompt': positive, 'negative_prompt': negative} for positive, negative in prompts]
        if change.kind in {'positive', 'negative', 'embedding'}:
            field = 'negative_prompt' if change.kind == 'negative' or (change.kind == 'embedding' and change.polarity == 'negative') else 'prompt'
            if not change.before or left[field].count(change.before) != 1:
                raise ValueError('comparison_phrase_not_unique')
            # Only the edited fragment gets expanded now; the common content is frozen.
            replacement = expand_dynamic_prompt(change.after, seed, wildcard_dir).expanded
            right[field] = left[field].replace(change.before, replacement, 1)
        elif change.kind == 'lora':
            if change.index >= len(right['loras']):
                raise ValueError('comparison_item_missing')
            right['loras'][change.index]['weight'] = change.weight
        if left == right:
            raise ValueError('comparison_no_difference')
        for arm, payload in zip(('A', 'B'), (left, right)):
            # Validate every cell, not just the first, before any generation.
            validated = REQUEST_TYPES[request.mode].model_validate(payload)
            cells.append({'index': len(cells), 'pair': pair, 'arm': arm, 'request': validated.model_dump(mode='json'),
                          'status': 'pending', 'generation_id': None, 'error': ''})
    return {'version': 1, 'id': uuid4().hex, 'name': request.name.strip(), 'created_at': datetime.now(UTC).isoformat(),
            'mode': request.mode, 'change': change.model_dump(mode='json'), 'source': base.model_dump(mode='json'),
            'seeds': seeds, 'cells': cells, 'display_orders': [secrets.choice([['A', 'B'], ['B', 'A']]) for _ in seeds],
            'ratings': {}, 'status': 'prepared', 'operation_id': None, 'error': ''}
