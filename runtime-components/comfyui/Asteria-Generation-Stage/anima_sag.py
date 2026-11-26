"""Spatial self-attention guidance for Anima and MOSS flow models."""
import math

import torch
import torch.nn.functional as F


def unconditional_scores(q, k, flags, *, chunk=None, image_tokens=None, image_start=0, mask=None):
    """Sum probability arriving at each spatial patch (mean over heads).

    Query/key here include the native head normalization and rotary positions.
    Conditional evaluations keep their native attention and need no score matrix.
    """
    if 1 not in flags:
        return []
    if len(q) % len(flags):
        raise ValueError("SAG: inconsistent conditioning batch")
    batch = len(q) // len(flags)
    results = []
    for index, flag in enumerate(flags):
        if flag != 1:
            continue
        subset = slice(index*batch,(index+1)*batch)
        queries,keys = (t[subset].reshape(batch,-1,*t.shape[-2:]).transpose(1,2).float() for t in (q,k))
        if image_tokens is not None:
            queries = queries[:, :, image_start:image_start + image_tokens]
        _, heads, tokens, dim = queries.shape
        totals = queries.new_zeros((batch,tokens))
        size = chunk or max(1, (4 << 20) // (batch*heads*keys.shape[-2]))
        kt = keys.transpose(-2,-1)
        bias = None if mask is None else torch.broadcast_to(mask,(len(q),heads,q.shape[1],keys.shape[-2]))[subset]
        for start in range(0,tokens,size):
            end = min(tokens,start+size)
            logits = (queries[:,:,start:end] @ kt).mul_(dim**-.5)
            if bias is not None:
                part = bias[:,:,image_start+start:image_start+end]
                logits = logits.masked_fill(~part,-torch.inf) if part.dtype == torch.bool else logits + part
            p = logits.softmax(-1)[..., image_start:image_start+tokens]
            totals.add_(p.sum(dim=(-3,-2)))
        results.append(totals.div_(heads))
    return results


def spatial_degrade(x, scores, patch, sigma):
    framed = x if x.ndim == 5 else x.unsqueeze(2)
    b,c,t,h,w = framed.shape
    rows,cols = math.ceil(h/patch),math.ceil(w/patch)
    if scores.shape != (b,t*rows*cols):
        raise ValueError("SAG: attention map does not cover the input image")
    mask = (scores > 1.).reshape(b,1,t,rows,cols)
    mask = mask.repeat_interleave(patch,-2).repeat_interleave(patch,-1)[...,:h,:w]
    # Work on spatial planes only. Anima images have a single temporal plane.
    planes = framed.transpose(1,2).reshape(b*t,c,h,w)
    radius = 4
    offsets = torch.arange(-radius,radius+1,dtype=torch.float32,device=x.device)
    weights = (-(offsets/sigma).square()*.5).softmax(0)
    kernel = torch.outer(weights,weights).expand(c,1,9,9)
    padded = F.pad(planes.float(),(radius,)*4,mode="reflect" if min(h,w)>radius else "replicate")
    low = F.conv2d(padded,kernel,groups=c).to(x.dtype).reshape(b,t,c,h,w).transpose(1,2)
    result = torch.where(mask,low,framed)
    return result if x.ndim == 5 else result.squeeze(2)


def flow_perturbation(x, denoised, blurred, sigma):
    # Keep the same noise in x_t=(1-s)*x_0+s*noise. Convert both
    # velocity predictions back to the same input before taking their difference.
    fraction = sigma.reshape(-1,*([1]*(x.ndim-1))).to(x)
    shift = (1.-fraction)*(blurred-denoised)
    return x+shift, denoised+shift


class _Capture:
    def __init__(self):
        self.maps = []


class AsteriaAnimaSelfAttentionGuidance:
    moss = False
    krea = False
    flux = False
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"model":("MODEL",),"scale":("FLOAT",{"default":.5}),
                            "blur_sigma":("FLOAT",{"default":2.})}}

    FUNCTION = "attach"
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Asteria/internal"

    def attach(self,model,scale,blur_sigma):
        from comfy.patcher_extension import WrappersMP
        from comfy.samplers import calc_cond_batch

        clone = model.clone()
        if scale == 0:
            return (clone,)
        if blur_sigma <= 0:
            raise ValueError("SAG requires a positive blur radius")
        net = clone.get_model_object("diffusion_model")
        key = "asteria_sag_capture"
        if self.moss:
            from . import moss_guidance
            def modify(x, opts, images, texts):
                capture = opts.get(key)
                if capture is None or 1 not in opts["cond_or_uncond"]:
                    return x, opts
                def apply(base, q, k, v, mask):
                    capture.maps.extend(unconditional_scores(q.transpose(1,2), k.transpose(1,2),
                                        opts["cond_or_uncond"], image_tokens=images, mask=mask))
                    return base
                return x, moss_guidance.attention_options(opts, apply)
            moss_guidance.wrap_layers(clone, "sag", [(len(net.layers)-1)//2], modify)
        elif self.flux:
            from .flux_guidance import middle_image_block
            from .krea_guidance import attention_options
            def modify(img, opts, text_len, images):
                capture = opts.get(key)
                if capture is None or 1 not in opts["cond_or_uncond"]:
                    return img, opts
                def apply(base, q, k, v, mask):
                    capture.maps.extend(unconditional_scores(q.transpose(1, 2), k.transpose(1, 2),
                        opts["cond_or_uncond"], image_start=text_len, image_tokens=images, mask=mask))
                    return base
                return img, attention_options(opts, apply)
            middle_image_block(clone, net, modify)
        elif self.krea:
            from . import krea_guidance
            def modify(x, opts, start, end):
                capture = opts.get(key)
                if capture is None or 1 not in opts["cond_or_uncond"]:
                    return x, opts
                def apply(base, q, k, v, mask):
                    capture.maps.extend(unconditional_scores(q.transpose(1, 2), k.transpose(1, 2),
                        opts["cond_or_uncond"], image_start=start, image_tokens=end-start, mask=mask))
                    return base
                return x, krea_guidance.attention_options(opts, apply)
            krea_guidance.wrap_blocks(clone, [(len(net.blocks)-1)//2], modify)
        else:
            path = f"diffusion_model.blocks.{(len(net.blocks)-1)//2}.self_attn.compute_qkv"
            native = clone.get_model_object(path)

        def observe(*args, transformer_options=None, **kwargs):
            opts = transformer_options or {}
            q,k,v = native(*args,transformer_options=opts,**kwargs)
            capture = opts.get(key)
            if capture is not None:
                capture.maps.extend(unconditional_scores(q,k,opts["cond_or_uncond"]))
            return q,k,v

        def predict(executor,x,sigma,model_options,seed=None):
            options = {**model_options,"transformer_options":{**model_options.get("transformer_options",{}),key:_Capture()}}
            return executor(x,sigma,options,seed)

        def correct(args):
            options = args["model_options"]
            remaining = dict(options["transformer_options"])
            capture = remaining.pop(key)
            if not capture.maps:
                raise ValueError("SAG did not observe the unconditional image attention")
            scores = torch.stack(capture.maps).mean(0) if len(capture.maps)>1 else capture.maps[0]
            original = args["uncond_denoised"]
            patch = 1 if self.moss else net.patch if self.krea else net.patch_size if self.flux else net.patch_spatial
            blurred = spatial_degrade(original,scores,patch,blur_sigma)
            perturbed,reference = flow_perturbation(args["input"],original,blurred,args["sigma"])
            alternate, = calc_cond_batch(args["model"],[args["uncond"]],perturbed,args["sigma"],
                                         {**options,"transformer_options":remaining})
            return args["denoised"] + scale*(reference-alternate)

        if not (self.moss or self.krea or self.flux):
            clone.add_object_patch(path,observe)
        clone.add_wrapper_with_key(WrappersMP.PREDICT_NOISE,key,predict)
        clone.set_model_sampler_post_cfg_function(correct,disable_cfg1_optimization=True)
        return (clone,)


class AsteriaMossSelfAttentionGuidance(AsteriaAnimaSelfAttentionGuidance):
    moss = True


class AsteriaKreaSelfAttentionGuidance(AsteriaAnimaSelfAttentionGuidance):
    krea = True


class AsteriaFluxSelfAttentionGuidance(AsteriaAnimaSelfAttentionGuidance):
    flux = True


NODE_CLASS_MAPPINGS = {"AsteriaAnimaSelfAttentionGuidance": AsteriaAnimaSelfAttentionGuidance,
                       "AsteriaFluxSelfAttentionGuidance": AsteriaFluxSelfAttentionGuidance,
                       "AsteriaMossSelfAttentionGuidance": AsteriaMossSelfAttentionGuidance,
                       "AsteriaKreaSelfAttentionGuidance": AsteriaKreaSelfAttentionGuidance}
