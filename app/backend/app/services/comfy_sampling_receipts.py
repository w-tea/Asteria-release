"""Observe executed sampling inputs; never reconstruct noise or schedules."""
from collections import Counter
from copy import deepcopy
import json

MAX_BYTES = 512 * 1024


def instrument_sampling_inputs(workflow: dict, mapping=None) -> None:
    next_id = max((int(key) for key in workflow if key.isdecimal()), default=0) + 1
    ordinals = Counter()
    class_type = mapping.class_type if mapping else lambda key, fallback: fallback
    for node_id, node in list(workflow.items()):
        inputs = node.get('inputs', {})
        custom = all(k in inputs for k in ('guider', 'sigmas', 'sampler'))
        standard = 'model' in inputs and ('sampler_name' in inputs or node.get('class_type') == 'AsteriaGenerationStageSampler')
        if not (custom or standard):
            continue
        stage = node.get('_meta', {}).get('asteria_stage', 'base')
        name = f'{stage}-{ordinals[stage]}'
        ordinals[stage] += 1
        if custom:
            source = workflow.get(str(inputs['guider'][0]), {})
            if 'model' not in source.get('inputs', {}):
                continue
            guider = str(next_id); next_id += 1
            workflow[guider] = deepcopy(source)
            inputs['guider'] = [guider, 0]
            owner = workflow[guider]['inputs']
        else:
            owner = inputs
        wrapper, output = str(next_id), str(next_id + 1)
        next_id += 2
        workflow[wrapper] = {'class_type': class_type('sampling_receipt_model', 'AsteriaSamplingReceiptModel'),
                             'inputs': {'model': owner['model']}}
        owner['model'] = [wrapper, 0]
        workflow[output] = {'class_type': class_type('sampling_receipt_output', 'AsteriaSamplingReceiptOutput'),
            'inputs': {'after': [node_id, 0], 'capture': [wrapper, 1]},
            '_meta': {'asteria_sampling_stage': name, 'asteria_sampling_source': node_id}}


def record_sampling_execution(request: dict, workflow: dict, history: dict, prompt_id: str, *, batch_index=0) -> None:
    settings = request.setdefault('settings', {})
    summary = settings.setdefault('sampling_execution', {'version': 1, 'jobs': [], 'truncated': False})
    outputs = history.get(prompt_id, {}).get('outputs', {})
    rows, missing = [], []
    for node_id, node in workflow.items():
        stage = node.get('_meta', {}).get('asteria_sampling_stage')
        if stage is None:
            continue
        values = outputs.get(node_id, {}).get('asteria_sampling_receipt', [])
        if not values:
            missing.append(node_id)
        for value in values:
            if not isinstance(value, dict) or value.get('version') != 1:
                missing.append(node_id)
                continue
            if value.get('status') != 'recorded' or value.get('truncated'):
                missing.append(node_id)
            for index, record in enumerate(value.get('records', [])):
                rows.append({**deepcopy(record), 'stage': stage, 'index': index,
                    'node_id': node.get('_meta', {}).get('asteria_sampling_source'),
                    'reused': bool(value.get('reused'))})
    job = {'runtime_job_id': prompt_id, 'batch_index': batch_index, 'stages': rows,
           'status': 'partial' if missing else 'recorded' if rows else 'unavailable',
           'missing': list(dict.fromkeys(missing))}
    # Count the real serialized total, including framing and missing-stage metadata.
    while rows and len(json.dumps({**summary, 'jobs': [*summary['jobs'], job]}, ensure_ascii=False).encode('utf-8')) > MAX_BYTES:
        rows.pop()
        summary['truncated'] = True
        job['status'] = 'partial'
    if len(json.dumps({**summary, 'jobs': [*summary['jobs'], job]}, ensure_ascii=False).encode('utf-8')) <= MAX_BYTES:
        summary['jobs'].append(job)
    else:
        summary['truncated'] = True
