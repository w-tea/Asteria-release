"""Keep numerical controls private to one comparison, under its captured output root."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

from app.services.atomic_json import write_json_atomic
from app.services.media_artifacts import file_sha256

CONTRACT = 'asteria.comparison-sampling.v1'


def attach_comparison_sampling(workflow, operation, mapping):
    if operation is None:
        return
    next_id = max((int(key) for key in workflow if key.isdecimal()), default=0) + 1
    ordinals = Counter()
    for node in list(workflow.values()):
        inputs = node['inputs']
        custom = 'guider' in inputs and 'sigmas' in inputs and 'sampler' in inputs
        standard = 'model' in inputs and 'sampler_name' in inputs and 'steps' in inputs
        if not (custom or standard):
            continue
        stage = node.get('_meta', {}).get('asteria_stage', 'base')
        key = f'{stage}-{ordinals[stage]}'
        ordinals[stage] += 1
        if custom:
            guider = str(next_id); next_id += 1
            workflow[guider] = deepcopy(workflow[inputs['guider'][0]])
            inputs['guider'] = [guider, 0]
            owner = workflow[guider]['inputs']
        else:
            owner = inputs
        wrapper = str(next_id); next_id += 1
        workflow[wrapper] = {'class_type': mapping.class_type('comparison_sampling_model', 'AsteriaComparisonSamplingModel'),
            'inputs': {'model': owner['model'], 'operation': json.dumps({**operation, 'stage': key}, sort_keys=True)}}
        owner['model'] = [wrapper, 0]
    if not ordinals:
        raise ValueError('comparison_sampling_unavailable')


class ComparisonSamplingSession:
    def __init__(self):
        self.directories = {}
        self.assets = {}
        self.common_assets = {}
        self.output_root = None
        self.directory = None
        self.arm = None

    def _verify_assets(self):
        for path, digest in self.assets.values():
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError('comparison_assets_changed')

    def prepare(self, root, metadata, request):
        self._verify_assets()
        root = Path(root).resolve()
        if self.output_root is not None and root != self.output_root:
            raise ValueError('comparison_directory_invalid')
        pair, self.arm = metadata['pair'], metadata['arm']
        records = request['settings'].get('generation_assets', {}).get('assets', [])
        if not records or any(a.get('status') != 'recorded' or not a.get('sha256') for a in records):
            raise ValueError('comparison_assets_unavailable')
        # Prompt edits can intentionally add/remove Embeddings. All other selected
        # assets must remain identical, even if a global model selection changes.
        common = {(a['role'], a['type'], a['sha256']) for a in records if a['type'] != 'embedding'}
        if pair in self.common_assets and common != self.common_assets[pair]:
            raise ValueError('comparison_assets_changed')
        for asset in records:
            key = (asset['type'], asset['reference'], str(Path(asset['path']).resolve()))
            value = (Path(asset['path']), asset['sha256'])
            if key in self.assets and self.assets[key][1] != value[1]:
                raise ValueError('comparison_assets_changed')
            self.assets[key] = value
        self._verify_assets()
        if pair not in self.directories:
            if self.arm != 'A':
                raise ValueError('comparison_anchor_missing')
            parent = root / '_comparisons'
            parent.mkdir(parents=True, exist_ok=True)
            if parent.resolve() != parent:
                raise ValueError('comparison_directory_invalid')
            directory = parent / uuid4().hex
            directory.mkdir()
            write_json_atomic(directory / 'owner.json', {'contract': CONTRACT,
                'output_root': str(root), 'operation_id': directory.name})
            self.directories[pair] = directory
            self.common_assets[pair] = common
            self.output_root = root
        self.directory = self.directories[pair]
        request['_comparison_sampling'] = {'directory': str(self.directory), 'arm': self.arm}

    def finish(self, settings):
        self._verify_assets()
        def read(arm):
            result = {}
            for path in sorted(self.directory.glob(f'{arm}-*.json')):
                if path.is_symlink() or path.stat().st_size > 256 * 1024:
                    raise ValueError('comparison_receipt_invalid')
                value = json.loads(path.read_text(encoding='utf-8'))
                if value.get('status') != 'completed':
                    raise ValueError('comparison_sampling_incomplete')
                result[path.name[2:]] = value
            return result
        rows = read(self.arm)
        if not rows:
            raise ValueError('comparison_sampling_missing')
        if self.arm == 'B':
            baseline = read('A')
            if (rows.keys() != baseline.keys() or any(row['used'] != baseline[key]['used']
                    or row['contract'] != baseline[key]['contract'] for key, row in rows.items())):
                raise ValueError('comparison_sampling_mismatch')
        return {'contract': CONTRACT, 'status': 'matched' if self.arm == 'B' else 'recorded',
            'directory': str(self.directory), 'stages': list(rows.values()),
            'assets': deepcopy(settings['generation_assets'])}
