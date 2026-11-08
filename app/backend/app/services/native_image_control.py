"""Asteria ControlNet units mapped to native latent-control adapters."""
from __future__ import annotations


def control_is_active(unit: dict) -> bool:
    return (unit.get('enabled', True) and unit.get('weight', 1) != 0
            and unit.get('guidance_start', 0) != unit.get('guidance_end', 1))


def active_controls(settings: dict) -> list[dict]:
    return [u for u in settings.get('controlnet_units', ()) if control_is_active(u)]


def native_control_capability(mapping, object_info):
    classes = {value for key, value in mapping.class_types.items() if key.startswith('control_')}
    missing = sorted(classes - object_info.keys())
    loader = mapping.class_type('control_loader', '')
    descriptor = object_info.get(loader, {}).get('input', {}).get('required', {}).get('control_name', ())
    models = descriptor[0] if descriptor and isinstance(descriptor[0], list) else []
    return {'available': not missing, 'runtime': 'native_image_control', 'missing_nodes': missing,
            'controlnet_models': models,
            'control_mode': {'available': True, 'runtime': 'native_image_control',
                             'native_node_input': False, 'modes': ['Balanced'], 'message': ''},
            'message': ''}


def validate_native_controls(profile: str, settings: dict) -> list[dict]:
    units = active_controls(settings)
    models = set()
    for unit in units:
        kind = unit.get('control_type', 'depth')
        if kind not in ({'depth'} if profile == 'krea2' else {'color', 'depth'}):
            raise ValueError('controlnet_units.control_type is not supported by this adapter')
        if not unit.get('model') or not unit.get('image_path'):
            raise ValueError('controlnet_units requires a control model and image')
        if unit.get('control_mode', 'Balanced') != 'Balanced':
            raise ValueError('controlnet_units.control_mode priority is not implemented by latent control')
        if unit.get('preprocessor', 'none') not in ({'none', 'depth'} if kind == 'depth' else {'none'}):
            raise ValueError('controlnet_units.preprocessor must produce the selected control input')
        if float(unit.get('guidance_start', 0)) > float(unit.get('guidance_end', 1)):
            raise ValueError('controlnet_units.guidance_start must not exceed guidance_end')
        models.add(unit['model'])
    if len(models) > 1:
        # Each adapter owns one replacement input projection, not an independent
        # ControlNet branch. Different projections cannot be stacked as SDXL units.
        raise ValueError('controlnet_units must use one shared input-projection adapter')
    return units


def append_native_controls(workflow, request, mapping, model_name, model, vae, positive, negative,
                           preprocessor_node):
    settings, profile = request['settings'], request['profile']
    units = validate_native_controls(profile, settings)
    if not units:
        return model, positive, negative
    next_id = max([int(k) for k in workflow] + [int(k) for k in mapping.nodes.values()]) + 1

    def add(key, inputs):
        nonlocal next_id
        node_id = str(next_id)
        next_id += 1
        workflow[node_id] = {'class_type': mapping.class_type(key, ''), 'inputs': inputs}
        return [node_id, 0]

    if profile == 'moss_image':
        model = add('control_loader', {'model': model, 'control_name': model_name(units[0]['model'], 'loras')})
    maps = None
    for unit in units:
        image = add('control_image', {'image': unit['image_path']})
        prepared = preprocessor_node(unit, image, settings)
        if prepared:
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = prepared
            image = [node_id, 0]
        inputs = {'vae': vae, 'image': image, 'width': settings['width'], 'height': settings['height'],
                  'strength': unit.get('weight', 1), 'start_percent': unit.get('guidance_start', 0),
                  'end_percent': unit.get('guidance_end', 1)}
        if profile == 'krea2':
            model = add('control_loader', {**inputs, 'model': model,
                        'control_name': model_name(unit['model'], 'loras')})
        else:
            maps = add('control_map', {**inputs, 'kind': unit.get('control_type', 'depth'),
                                      **({'previous': maps} if maps else {})})
    if profile == 'moss_image':
        positive = add('control_apply', {'positive': positive, 'negative': negative, 'maps': maps})
        negative = [positive[0], 1]
    settings['controlnet_runtime'] = 'native_image_control'
    return model, positive, negative
