"""Record actual workflow image edges without altering the sampling graph."""
from copy import deepcopy


def mark_stage(settings: dict, kind: str, *, images=None, samples=None, vae=None) -> None:
    if not settings.get('record_enhancement_stages'):
        return
    settings.setdefault('_enhancement_points', []).append({
        'kind': kind, 'images': deepcopy(images), 'samples': deepcopy(samples), 'vae': deepcopy(vae),
    })


def decoded_stage_kind(settings: dict) -> str:
    hires = bool((settings.get('hires_enabled') and settings.get('hires_upscaler') != 'Topaz Photo AI')
                 or settings.get('_prompt_trace_stage') == 'hires')
    refiner = bool(settings.get('refiner_enabled') and settings.get('refiner_checkpoint')
                   and settings.get('refiner_checkpoint') != 'None')
    return 'hires_refiner' if hires and refiner else 'hires' if hires else 'refiner' if refiner else 'base'


def append_stage_outputs(workflow: dict, settings: dict, mapping) -> None:
    points = settings.pop('_enhancement_points', [])
    settings.pop('_enhancement_output_nodes', None)
    if not settings.get('record_enhancement_stages'):
        return
    output_nodes = {}
    next_id = max((int(key) for key in workflow if key.isdecimal()), default=0) + 1
    decode_class = mapping.class_type('vae_decode', 'VAEDecode')
    for ordinal, point in enumerate(points):
        images = point['images']
        if images is None:
            # Share an existing decode when possible; never add another sampler.
            images = next(([key, 0] for key, node in workflow.items()
                           if node['class_type'] == decode_class
                           and node['inputs'].get('samples') == point['samples']
                           and node['inputs'].get('vae') == point['vae']), None)
            if images is None:
                key = str(next_id); next_id += 1
                workflow[key] = {'class_type': decode_class, 'inputs': {'samples': point['samples'], 'vae': point['vae']}}
                images = [key, 0]
        key = str(next_id); next_id += 1
        workflow[key] = {'class_type': mapping.class_type('save_image', 'SaveImage'),
                         'inputs': {'images': images, 'filename_prefix': f'Asteria_stage_{ordinal}'}}
        output_nodes[key] = {'kind': point['kind'], 'ordinal': ordinal}
    settings['_enhancement_output_nodes'] = output_nodes
