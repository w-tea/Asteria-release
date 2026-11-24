"""Adapter-owned insertion on the base sampler edge; enhancement models stay unpatched."""
from copy import deepcopy
import json


def attach_branch(workflow: dict, sample_node: str, operation: dict, mapping) -> None:
    sample = workflow[sample_node]['inputs']
    next_id = max((int(key) for key in workflow if key.isdecimal()), default=0) + 1
    if 'model' in sample:
        owner = sample
    elif 'guider' in sample:
        # A guider may also feed Hires: never modify that shared owner.
        key = str(next_id); next_id += 1
        workflow[key] = deepcopy(workflow[sample['guider'][0]])
        sample['guider'] = [key, 0]
        owner = workflow[key]['inputs']
    else:
        raise ValueError('branch_sampler_not_supported')
    key = str(next_id)
    workflow[key] = {'class_type': mapping.class_type('generation_branch_model', 'AsteriaGenerationBranchModel'),
                     'inputs': {'model': owner['model'], 'operation': json.dumps(operation, sort_keys=True)}}
    owner['model'] = [key, 0]
