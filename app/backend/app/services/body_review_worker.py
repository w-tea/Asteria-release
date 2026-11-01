"""Offline CPU adapter. Invoked only by BodyReviewService in a bounded process."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def sam_chunked_attention(self, x, chunk=64):
    """SAM attention with a bounded query working set; all keys/values are retained.

    Adapted from Asteria's verified CPU diagnostic. Equations follow Meta SAM
    (Apache-2.0); see docs/BODY_REVIEW_DEPENDENCIES.md for attribution.
    """
    import torch
    from segment_anything.modeling.image_encoder import get_rel_pos
    batch, height, width, channels = x.shape
    count = height * width
    values = self.qkv(x).reshape(batch, count, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = values.reshape(3, batch*self.num_heads, count, -1).unbind(0)
    result = torch.empty_like(q)
    rh = get_rel_pos(height, height, self.rel_pos_h) if self.use_rel_pos else None
    rw = get_rel_pos(width, width, self.rel_pos_w) if self.use_rel_pos else None
    for start in range(0, count, chunk):
        stop = min(count, start+chunk)
        query = q[:, start:stop]
        score = (query*self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            indexes = torch.arange(start, stop, device=x.device)
            bias_h = torch.einsum('bqd,qkd->bqk', query, rh[indexes//width])
            bias_w = torch.einsum('bqd,qkd->bqk', query, rw[indexes%width])
            score = score.reshape(batch*self.num_heads, stop-start, height, width)
            score.add_(bias_h[:, :, :, None]).add_(bias_w[:, :, None, :])
            score = score.reshape(batch*self.num_heads, stop-start, count)
        result[:, start:stop] = score.softmax(dim=-1) @ v
    result = result.view(batch, self.num_heads, height, width, -1).permute(0, 2, 3, 1, 4)
    return self.proj(result.reshape(batch, height, width, channels))


def main():
    # Environment/device restrictions are set by the parent before this import.
    import numpy as np
    from PIL import Image
    import torch
    from app.schemas.body_review import CandidateBox
    from app.services.atomic_json import write_json_atomic
    from app.services.body_review import DETECTOR_SHA, SAM_SHA, digest, review_candidates
    torch.set_num_threads(4)
    spec = json.loads(Path(sys.argv[1]).read_text('utf-8'))
    image = Image.open(spec['source']).convert('RGB')
    output = Path(spec['output'])
    if spec['stage'] == 'detect':
        from transformers import AutoProcessor, AutoConfig, AutoModelForZeroShotObjectDetection
        model_path = Path(spec['detector'])
        if digest(model_path / 'model.safetensors') != DETECTOR_SHA:
            raise ValueError('detector_weight_changed')
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        config.disable_custom_kernels = True
        model = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path), config=config,
                    local_files_only=True, use_safetensors=True).eval().to('cpu')
        parts = list(dict.fromkeys(spec['parts']))
        caption = '. '.join(parts) + '.'
        offsets = processor.tokenizer(caption, return_offsets_mapping=True)['offset_mapping']
        positions = [[i for i, (a, b) in enumerate(offsets) if b > caption.index(word)
                      and a < caption.index(word)+len(word)] for word in parts]
        inputs = processor(images=image, text=caption, return_tensors='pt')
        with torch.inference_mode():
            result = model(**inputs)
        scores = torch.stack([result.logits[0, :, indexes].sigmoid().amax(-1) for indexes in positions], -1)
        cx, cy, bw, bh = result.pred_boxes[0].unbind(-1)
        boxes = torch.stack(((cx-bw/2)*image.width, (cy-bh/2)*image.height,
                             (cx+bw/2)*image.width, (cy+bh/2)*image.height), -1)
        candidates = review_candidates(boxes.tolist(), scores.tolist(), parts, *image.size)
        write_json_atomic(output, dict(candidates=candidates[:96], candidate_count=len(candidates), parts=parts,
                                       models={'detector_sha256': DETECTOR_SHA}))
    elif spec['stage'] == 'mask':
        from types import MethodType
        from segment_anything import sam_model_registry, SamPredictor
        if digest(Path(spec['sam'])) != SAM_SHA:
            raise ValueError('sam_weight_changed')
        model = sam_model_registry['vit_b'](checkpoint=None).eval()
        model.load_state_dict(torch.load(spec['sam'], map_location='cpu', weights_only=True), strict=True)
        for index in (2, 5, 8, 11):
            model.image_encoder.blocks[index].attn.forward = MethodType(sam_chunked_attention, model.image_encoder.blocks[index].attn)
        predictor = SamPredictor(model)
        masks = []
        with torch.inference_mode():
            predictor.set_image(np.asarray(image))
            for index, value in enumerate(spec['candidates']):
                candidate = CandidateBox.model_validate(value)
                box = np.asarray(candidate.box) * np.array([*image.size, *image.size])
                mask, quality, _ = predictor.predict(box=box.astype('float32'), multimask_output=False)
                name = f"{spec['mask_prefix']}-{index}.png"
                Image.fromarray(mask[0].astype('uint8')*255).save(output.parent / name)
                masks.append(dict(id=candidate.id, file=name, quality=float(quality[0])))
        write_json_atomic(output, dict(masks=masks, models={'sam_sha256': SAM_SHA}))
    else:
        raise ValueError('invalid stage')
    if torch.cuda.is_initialized():
        raise ValueError('unexpected CUDA initialization')


if __name__ == '__main__':
    main()
