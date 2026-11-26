"""Retrieval-Gated Attention Guidance, equations 11–12 (arXiv:2603.02531v4)."""
import math

import torch


def _entmax15(scores):
    # Solve sum(max(x - tau, 0)^2) = 1 on the sorted active support.
    # After subtracting max(x), tau >= -1, so ignored keys need not
    # participate in infinite or very large prefix-sum arithmetic.
    x = ((scores - scores.amax(-1, keepdim=True)) * .5).clamp_min(-1.)
    ordered = x.sort(dim=-1, descending=True).values
    count = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / count
    variance_sum = (ordered.square().cumsum(-1) - count * mean.square()).clamp_min(0.)
    taus = mean - ((1. - variance_sum).clamp_min(0.) / count).sqrt()
    support = (ordered >= taus).sum(-1, keepdim=True).clamp_min(1)
    return (x - taus.gather(-1, support - 1)).clamp_min(0.).square()


def cross_attention(q, k, v, gain, *, cap=15., chunk=None, mask=None, scale=None):
    """Input [batch, heads, tokens, channels]. One projection per sample and layer.

    Accumulate over every query and head before scaling the sparse readout.
    Splitting queries bounds intermediate memory; it must not change the clamp.
    """
    b, h, n, d = q.shape
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    step = chunk or max(1, (1 << 19) // (b * h * k.shape[-2]))
    kt = k.to(dtype).transpose(-2, -1)
    values = v.to(dtype)
    out = torch.empty((b, h, n, v.shape[-1]), device=v.device, dtype=dtype)
    stats = q.new_zeros((b, 3), dtype=dtype)
    if mask is not None:
        mask = torch.broadcast_to(mask, (b, h, n, k.shape[-2]))
    for start in range(0, n, step):
        rows = slice(start, start + step)
        logits = (q[:, :, rows].to(dtype) @ kt) * (d ** -.5 if scale is None else scale)
        if mask is not None:
            m = mask[:, :, rows]
            logits = logits.masked_fill(~m, -torch.inf) if m.dtype == torch.bool else logits + m
            empty = torch.isneginf(logits).all(-1, keepdim=True)
            logits = torch.where(empty, 0., logits)
        p = _entmax15(logits)
        delta = p - logits.softmax(-1)
        if mask is not None:
            p = p.masked_fill(empty, 0.)
            delta = delta.masked_fill(empty, 0.)
        out[:, :, rows] = p @ values
        axes = (1, 2, 3)
        stats += torch.stack(((delta*p).sum(axes), p.square().sum(axes), delta.square().sum(axes)), dim=1)
    dot, norm2, dist2 = stats.unbind(1)
    coeff = dot / torch.where(norm2 > 0, norm2, 1.)
    clamp = cap / dist2.sqrt().clamp_min(cap)
    return out.mul_(1. + (gain * coeff * clamp).view(-1, 1, 1, 1)).to(v.dtype)


def in_window(sigmas, sigma, fraction):
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    steps = len(sigmas) - 1
    end = min(steps, math.ceil(steps * fraction - 1e-10))
    return bool(sigma.flatten()[0] > sigmas[end])


def _kernel(gain, q, k, v, heads, mask=None, **options):
    b = q.shape[0]
    if not options.get("skip_reshape", False):
        q, k, v = (t.reshape(b, -1, heads, t.shape[-1] // heads).transpose(1, 2) for t in (q, k, v))
    if mask is not None and mask.ndim != 4:
        if mask.dtype == torch.bool and mask.ndim == 2 and mask.shape[0] == b:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
    out = cross_attention(q, k, v, gain, mask=mask, scale=options.get("scale"))
    return out if options.get("skip_output_reshape", False) else out.transpose(1, 2).reshape(b, q.shape[-2], -1)


def joint_text_guidance(base, q, k, v, text_len, gain, mask=None, scale=None, chunk=None,
                        *, image_start=None, image_len=None, text_start=0):
    """Replace only image→text readout, weighted by the original joint softmax text mass."""
    b, h, n, d = q.shape
    scale = d ** -.5 if scale is None else scale
    image_start = text_len if image_start is None else image_start
    image_end = n if image_len is None else image_start + image_len
    images, texts = slice(image_start, image_end), slice(text_start, text_start + text_len)
    iq, tk, tv = q[:, :, images], k[:, :, texts], v[:, :, texts]
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    if mask is not None:
        mask = torch.broadcast_to(mask, (b, h, n, k.shape[2]))
    sparse = cross_attention(iq, tk, tv, gain, scale=scale, mask=None if mask is None else mask[:, :, images, texts])
    result = base.clone()
    step = chunk or max(1, (1 << 21) // (b*h*k.shape[2]))
    kt = k.to(dtype).transpose(-2, -1)
    for start in range(image_start, image_end, step):
        end = min(image_end, start + step)
        logits = (q[:, :, start:end].to(dtype) @ kt) * scale
        if mask is not None:
            m = mask[:, :, start:end]
            logits = logits.masked_fill(~m, -torch.inf) if m.dtype == torch.bool else logits + m
        weights = logits.softmax(-1).nan_to_num()[:, :, :, texts]
        new = sparse[:, :, start-image_start:end-image_start].to(dtype) * weights.sum(-1, keepdim=True)
        result[:, :, start:end] = (base[:, :, start:end].to(dtype) + new - weights @ tv.to(dtype)).to(base.dtype)
    return result


class AsteriaRetrievalGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "gain": ("FLOAT", {"default": 10.}),
                             "window": ("FLOAT", {"default": .52, "min": 0., "max": 1.})}}

    FUNCTION = "attach"
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Asteria/internal"

    def attach(self, model, gain, window):
        from comfy.patcher_extension import WrappersMP

        result = model.clone()
        if window == 0:
            return (result,)
        net = result.get_model_object("diffusion_model")
        paths = [name for name, _ in net.named_modules()
                 if (name.endswith(".attn2") and ".transformer_blocks." in name)
                 or (name.startswith("blocks.") and name.endswith(".cross_attn"))]
        joint = not paths and hasattr(net, "double_blocks")
        if joint:
            paths = [f"double_blocks.{i}" for i in range(len(net.double_blocks))]
        if not paths:
            from . import moss_guidance, krea_guidance
            if moss_guidance.is_moss(net):
                def modify(x, opts, images, texts):
                    if not opts.get("asteria_rega_active"):
                        return x, opts
                    def apply(base, q, k, v, mask):
                        shaped = base.unflatten(-1, (q.shape[1], -1)).transpose(1, 2)
                        out = joint_text_guidance(shaped, q, k, v, texts, gain, mask=mask,
                                                  image_start=0, image_len=images, text_start=images)
                        return out.transpose(1, 2).reshape_as(base)
                    return x, moss_guidance.attention_options(opts, apply)
                moss_guidance.wrap_layers(result, "rega", range(len(net.layers)), modify)
            elif krea_guidance.is_krea(net):
                def modify(x, opts, start, end):
                    if not opts.get("asteria_rega_active"):
                        return x, opts
                    def apply(base, q, k, v, mask):
                        shaped = base.unflatten(-1, (q.shape[1], -1)).transpose(1, 2)
                        out = joint_text_guidance(shaped, q, k, v, start, gain, mask=mask,
                                                  image_start=start, image_len=end-start)
                        return out.transpose(1, 2).reshape_as(base)
                    return x, krea_guidance.attention_options(opts, apply)
                krea_guidance.wrap_blocks(result, range(len(net.blocks)), modify)
            else:
                raise ValueError("ReGA: no supported image-to-text attention path")

        def gate(executor, x, timestep, model_options, seed=None):
            opts = model_options["transformer_options"]
            use = in_window(opts["sample_sigmas"], timestep, window)
            if use:
                model_options = {**model_options, "transformer_options": {**opts, "asteria_rega_active": True}}
            return executor(x, timestep, model_options, seed)

        def wrap(original):
            def forward(*args, transformer_options=None, **kwargs):
                opts = transformer_options or {}
                if opts.get("asteria_rega_active"):
                    previous = opts.get("optimized_attention_override")

                    def apply(native, *a, **kw):
                        def guided(*args_, **kwargs_):
                            if joint:
                                q, k, v = args_[:3]
                                text = kwargs["txt"] if "txt" in kwargs else args[1]
                                base = native(*args_, **kwargs_)
                                unfolded = kwargs_.get("skip_output_reshape", False)
                                shaped = base if unfolded else base.reshape(q.shape[0], q.shape[2], q.shape[1], -1).transpose(1, 2)
                                out = joint_text_guidance(shaped, q, k, v, text.shape[1], gain,
                                                         kwargs_.get("mask"), kwargs_.get("scale"))
                                return out if unfolded else out.transpose(1, 2).reshape_as(base)
                            return _kernel(gain, *args_, **kwargs_)
                        return previous(guided, *a, **kw) if previous else guided(*a, **kw)
                    opts = {**opts, "optimized_attention_override": apply}
                return original(*args, transformer_options=opts, **kwargs)
            return forward

        for name in paths:
            path = "diffusion_model." + name + ".forward"
            result.add_object_patch(path, wrap(result.get_model_object(path)))
        result.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "asteria_rega", gate)
        return (result,)


NODE_CLASS_MAPPINGS = {"AsteriaRetrievalGuidance": AsteriaRetrievalGuidance}
