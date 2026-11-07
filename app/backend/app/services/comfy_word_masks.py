"""Adapter-owned read-only observation on the base sampler's model edge."""
from copy import deepcopy
import json


def attach_word_capture(workflow, sample_node, operation, mapping):
    sample=workflow[sample_node]['inputs']
    next_id=max((int(key) for key in workflow if key.isdecimal()),default=0)+1
    if 'model' in sample:
        owner=sample
    elif 'guider' in sample:
        key=str(next_id);next_id+=1
        workflow[key]=deepcopy(workflow[sample['guider'][0]])
        sample['guider']=[key,0];owner=workflow[key]['inputs']
    else: raise ValueError('word_capture_sampler_not_supported')
    positive=owner.get('positive',owner.get('conditioning'))
    if positive is None: raise ValueError('word_capture_positive_missing')
    key=str(next_id)
    workflow[key]={'class_type':mapping.class_type('word_capture_model','AsteriaWordCaptureModel'),
                   'inputs':{'model':owner['model'],'positive':positive,
                             'operation':json.dumps(operation,sort_keys=True)}}
    owner['model']=[key,0]
