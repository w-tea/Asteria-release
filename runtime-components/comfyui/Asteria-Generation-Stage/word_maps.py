"""Read-only, bounded attribution from the native cross-attention Q/K."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

MAX_TEMP_BYTES = 128 * 1024 * 1024
MAX_CPU_BYTES = 64 * 1024 * 1024
MAX_METADATA = 4 * 1024 * 1024


def token_contract(positive, profile):
    if len(positive) != 1:
        raise ValueError('multiple_positive_contexts')
    context, metadata = positive[0]
    streams = metadata.get('_asteria_word_streams') or []
    selected = [s for s in streams if s['encoder'] in ({'t5xxl'} if profile == 'anima' else {'l', 'g'})]
    if not selected:
        raise ValueError('token_record_missing')
    rows = {s['encoder']: [row for chunk in s['chunks'] for row in chunk] for s in selected}
    if any(sum(s['chunk_lengths']) != len(rows[s['encoder']]) for s in selected):
        raise ValueError('token_record_partial')
    if profile == 'anima':
        actual = metadata.get('t5xxl_ids')
        expected = [row.get('id') for row in rows['t5xxl']]
        expanded = metadata.get('qwen35_expanded_architecture') == 'anima_progressive_qwen35_cross_adapter_v1'
        actual = actual.flatten().tolist() if actual is not None else metadata.get('_asteria_word_t5_ids') if expanded else None
        if actual != expected:
            raise ValueError('token_id_mismatch')
        if expanded:
            rows['t5xxl'] = rows['t5xxl'][:512]
            selected = [{**selected[0], 'chunk_lengths':[len(rows['t5xxl'])], 'chunks':[rows['t5xxl']]}]
            length = 512
            if context.shape[1] != length: raise ValueError('expanded_context_length_mismatch')
        else:
            length = max(512, len(expected))
    else:
        length = context.shape[1]
        if set(rows) != {'l', 'g'} or any(len(r) != length for r in rows.values()):
            raise ValueError('token_length_mismatch')
    columns = []
    for index in range(max(map(len, rows.values()))):
        labels = {encoder: values[index].get('text', '').strip() for encoder, values in rows.items()
                  if index < len(values) and not values[index].get('role')
                  and values[index].get('kind') == 'token' and values[index].get('text', '').strip()}
        columns.append({'position': index, 'labels': labels})
    return {'context_length': length, 'columns': columns,
            'streams': selected, 'profile': profile}


def positive_rows(batch, flags):
    if not flags or flags.count(0) != 1 or batch % len(flags):
        raise ValueError('cfg_context_ambiguous')
    count = batch // len(flags)
    start = flags.index(0) * count
    return list(range(start, start + count))


def fold_context(probabilities, context_length):
    length = probabilities.shape[-1]
    if length % context_length:
        raise ValueError('context_length_mismatch')
    return probabilities.reshape(*probabilities.shape[:-1], length // context_length, context_length).sum(-2)


def probability_maps(q, k, columns, *, grid, output_grid, context_length=None, query_chunk=128):
    """Inputs are B,H,Q/K,D after the model's projections and normalization."""
    b, heads, queries, dimension = q.shape
    length = k.shape[-2]
    context_length = context_length or length
    if (k.shape[:2] != (b, heads) or k.shape[-1] != dimension or math.prod(grid) != queries
            or length % context_length or not columns or min(columns) < 0 or max(columns) >= context_length):
        raise ValueError('attention_shape_mismatch')
    # Original model tensors are not copied wholesale. Account for key cast,
    # query cast, logits, probabilities, selected reduction, and CPU resize.
    chunk = min(queries, query_chunk)
    temp = 4 * (heads * length * dimension + heads * chunk * dimension
                + 3 * heads * chunk * length + 2 * chunk * len(columns))
    cpu = 4 * b * len(columns) * (queries + 2 * math.prod(output_grid))
    if temp > MAX_TEMP_BYTES or cpu > MAX_CPU_BYTES:
        raise ValueError('attention_memory_limit')
    result = torch.empty(b, len(columns), queries, dtype=torch.float32, device='cpu')
    for batch in range(b):
        keys = k[batch].detach().float().transpose(-1, -2)
        for start in range(0, queries, chunk):
            logits = (q[batch, :, start:start+chunk].detach().float() @ keys) * dimension ** -.5
            probabilities = logits.softmax(-1).mean(0)
            folded = fold_context(probabilities, context_length)
            values = folded[:, columns].transpose(0, 1)
            result[batch, :, start:start+chunk].copy_(values, non_blocking=False)
    result = result.reshape(b, len(columns), *grid)
    if grid != output_grid:
        result = F.interpolate(result, size=output_grid, mode='bilinear', align_corners=False)
    if not torch.isfinite(result).all():
        raise ValueError('attention_nonfinite')
    return result


class AttentionTap:
    """Return the exact native output even if optional observation cannot proceed."""
    def __init__(self, observe):
        self.observe = observe
        self.error = None

    def __call__(self, native, *args, **kwargs):
        output = native(*args, **kwargs)
        if self.error is None:
            try:
                self.observe(*args, **kwargs)
            except (ValueError, RuntimeError, OSError, KeyError, IndexError, TypeError) as exc:
                self.error = str(exc)[:256]
        return output
