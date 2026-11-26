"""Self-Swap Guidance (arXiv:2604.08048): perturb only the comparison prediction."""
import re

import torch
from torch.nn.functional import normalize


def swap_rows(x, ratio, *, max_scores=1 << 20):
    """Exchange dissimilar pairs, per image, without building a full attention map.

    Each row participates at most once. If two low-scoring candidates overlap,
    keep the first available pair and continue among the remaining rows.
    """
    n = x.shape[0]
    needed = int(n * ratio / 2)
    if not needed:
        return x
    unit = normalize(x.float(), dim=-1)
    available = torch.arange(n, device=x.device)
    swaps = []
    while needed:
        vectors = unit.index_select(0, available)
        m = len(available)
        columns = torch.arange(m, device=x.device)
        pool = min(m*(m-1)//2, max(64, needed*8))
        vals = x.new_empty(0, dtype=torch.float32)
        ids = available.new_empty(0)
        rows = max(1, max_scores // m)
        for start in range(0, m, rows):
            end = min(m, start + rows)
            scores = vectors[start:end] @ vectors.T
            scores.masked_fill_(columns[None, :] <= columns[start:end, None], torch.inf)
            block = scores.flatten()
            v, i = block.topk(min(pool, block.numel()), largest=False, sorted=False)
            vals = torch.cat((vals, v))
            ids = torch.cat((ids, i + start*m))
            if len(ids) > pool:
                vals, keep = vals.topk(pool, largest=False, sorted=False)
                ids = ids[keep]
        # Resolve conflicts on the small candidate set, not on the feature tensor.
        live = available.tolist()
        seen = set()
        for _, flat in sorted(zip(vals.tolist(), ids.tolist())):
            a, b = divmod(flat, m)
            if a >= b or a in seen or b in seen:
                continue
            swaps.append((live[a], live[b]))
            seen.update((a, b))
            needed -= 1
            if not needed:
                break
        if not seen:
            raise ValueError("SSG: no finite pair of image features could be selected")
        if needed:
            available = available[[i for i in range(m) if i not in seen]]
    left, right = torch.tensor(swaps, device=x.device).T
    perm = torch.arange(n, device=x.device)
    perm[left], perm[right] = right, left
    return x.index_select(0, perm)


def swap_features(x, ratio):
    if not ratio:
        return x
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    return torch.stack([swap_rows(swap_rows(image, ratio).T, ratio).T for image in flat]).reshape_as(x)


def correct(guided, clean_positive, perturbed_positive, strength):
    return guided + strength * (clean_positive - perturbed_positive)


def swap_image_prefix(x, count, ratio):
    return torch.cat((swap_features(x[:, :count], ratio), x[:, count:]), dim=1)


class AsteriaSelfSwapGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "strength": ("FLOAT", {"default": 3.}),
                             "ratio": ("FLOAT", {"default": .2, "min": 0., "max": 1.})}}

    FUNCTION = "attach"
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Asteria/internal"

    def attach(self, model, strength, ratio):
        from comfy.samplers import calc_cond_batch

        clone = model.clone()
        if not strength or not ratio:
            return (clone,)
        net = clone.get_model_object("diffusion_model")
        names = [name for name, _ in net.named_modules()
                 if re.fullmatch(r"output_blocks\.\d+\.\d+\.transformer_blocks\.\d+", name)][:6]
        if not names and hasattr(net, "blocks") and len(net.blocks) and hasattr(net.blocks[0], "cross_attn"):
            names = [f"blocks.{(len(net.blocks)-1)//2}"]
        key = "asteria_ssg_swap"
        if not names and hasattr(net, "double_blocks") and len(net.double_blocks):
            from .flux_guidance import middle_image_block
            def modify(img, opts, text_len, images):
                return (swap_image_prefix(img, images, ratio) if opts.get(key) else img), opts
            middle_image_block(clone, net, modify)
        elif not names:
            from . import moss_guidance, krea_guidance
            if krea_guidance.is_krea(net):
                def modify(x, opts, start, end):
                    if opts.get(key):
                        x = torch.cat((x[:, :start], swap_features(x[:, start:end], ratio), x[:, end:]), dim=1)
                    return x, opts
                krea_guidance.wrap_blocks(clone, [(len(net.blocks)-1)//2], modify)
            elif moss_guidance.is_moss(net):
                def modify(x, opts, images, texts):
                    return (swap_image_prefix(x, images, ratio) if opts.get(key) else x), opts
                moss_guidance.wrap_layers(clone, "ssg", [(len(net.layers)-1)//2], modify)
            else:
                raise ValueError("SSG: cannot locate the image transformer blocks")

        def wrap(original):
            def forward(x, *args, transformer_options=None, **kwargs):
                opts = transformer_options or {}
                if opts.get(key):
                    x = swap_features(x, ratio)
                return original(x, *args, transformer_options=opts, **kwargs)
            return forward

        for name in names:
            path = "diffusion_model." + name + ".forward"
            clone.add_object_patch(path, wrap(clone.get_model_object(path)))

        def guide(args):
            options = args["model_options"]
            swapped = {**options, "transformer_options": {**options.get("transformer_options", {}), key: True}}
            pred, = calc_cond_batch(args["model"], [args["cond"]], args["input"], args["sigma"], swapped)
            return correct(args["denoised"], args["cond_denoised"], pred, strength)

        clone.set_model_sampler_post_cfg_function(guide)
        return (clone,)


NODE_CLASS_MAPPINGS = {"AsteriaSelfSwapGuidance": AsteriaSelfSwapGuidance}
