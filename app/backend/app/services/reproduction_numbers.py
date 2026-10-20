"""Compare recorded values only. Never regenerate missing sampling inputs."""
from __future__ import annotations


def recorded_values(item: dict) -> dict:
    settings = item.get('settings') or {}
    values = {}
    for key in ('seed', 'sampler', 'scheduler', 'steps', 'cfg_scale', 'width', 'height',
                'batch_size', 'batch_count', 'clip_skip', 'denoising_strength'):
        if key in settings and settings[key] is not None:
            values[key] = settings[key]
    for key in ('profile', 'expanded_prompt', 'negative_prompt', 'model_slots', 'loras', 'embeddings'):
        if key in item and item[key] is not None:
            values[key] = item[key]
    assets = settings.get('generation_assets') or {}
    for asset in assets.get('assets', []):
        if asset.get('status') == 'recorded' and asset.get('sha256'):
            values['asset/' + str(asset['role'])] = asset['sha256']
    sampling = (settings.get('prompt_comparison') or {}).get('sampling') or {}
    stages = [('', row) for row in sampling.get('stages', [])]
    execution = settings.get('sampling_execution') or {}
    if execution.get('version') == 1:
        for job in execution.get('jobs', []):
            batch = job.get('batch_index', 0)
            stages.extend((f'batch-{batch}/' if batch else '', row) for row in job.get('stages', []))
    for prefix, row in stages:
        if row.get('status') != 'completed':
            continue
        key = f"{prefix}{row.get('stage', 'unknown')}/{row.get('index', 0)}"
        for tensor in ('noise', 'sigmas'):
            actual = row.get('used', {}).get(tensor)
            if actual and actual.get('sha256') and actual.get('status') != 'unrecorded':
                values[f'{key}/{tensor}'] = actual
        for field in ('contract', 'latent', 'sigma_values', 'devices', 'compute_devices'):
            if row.get(field) is not None and not (isinstance(row[field], dict) and row[field].get('status') == 'unrecorded'):
                values[f'{key}/{field}'] = row[field]
    return values


def sampling_origin(item: dict) -> dict:
    execution = item.get('settings', {}).get('sampling_execution') or {}
    rows = [row for job in execution.get('jobs', []) for row in job.get('stages', [])
            if row.get('status') == 'completed'] if execution.get('version') == 1 else []
    return {'fresh': sum(not row.get('reused') for row in rows), 'reused': sum(bool(row.get('reused')) for row in rows)}


def compare_numbers(left: dict, right: dict) -> dict:
    a, b = recorded_values(left), recorded_values(right)
    keys = list(dict.fromkeys([*a, *b]))
    if not any('/noise' in key or '/sigmas' in key for key in keys):
        keys.append('sampling')
    rows = []
    for key in keys:
        recorded = key in a and key in b
        rows.append(dict(key=key, left=a.get(key), right=b.get(key),
            left_recorded=key in a, right_recorded=key in b,
            status=('same' if a[key] == b[key] else 'different') if recorded else 'unrecorded'))
    # This report does not claim pixel equality or infer the cause of a difference.
    return dict(left_id=left['id'], right_id=right['id'], rows=rows,
                sampling_origins={'left': sampling_origin(left), 'right': sampling_origin(right)},
                basis='saved_values_only', pixel_identity_guaranteed=False)
