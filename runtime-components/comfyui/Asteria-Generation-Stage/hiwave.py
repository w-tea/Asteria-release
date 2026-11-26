"""Asteria's patchwise DDIM inversion and wavelet detail guidance (arXiv:2506.20452)."""
import math

import torch


# Symlet 4 orthonormal scaling filter. Use a periodic extension within each tile.
_SYM4=(-.07576571478927333,-.02963552764599851,.49761866763201545,.8037387518059161,
       .29785779560527736,-.09921954357684722,-.012603967262037833,.0322231006040427)


def _low_axis(x,axis):
    v=x.movedim(axis,-1)
    size=v.shape[-1]
    if size%2:
        v=torch.cat((v,v[..., -1:]),dim=-1)
    coeff=sum(h*v.roll(-k,dims=-1)[...,::2] for k,h in enumerate(_SYM4))
    expanded=torch.zeros_like(v)
    expanded[...,::2]=coeff
    out=sum(h*expanded.roll(k,dims=-1) for k,h in enumerate(_SYM4))
    return out[...,:size].movedim(-1,axis)


def low_band(x):
    return _low_axis(_low_axis(x,-1),-2)


def guide(positive,negative,detail):
    # Inverse DWT is linear. Eq. 6–10 preserve the conditional LL band and
    # apply the same CFG to LH, HL and HH. Only LL needs an explicit projection.
    if detail==1:
        return positive
    delta=positive-negative
    return positive+(detail-1)*(delta-low_band(delta))


def starts(size,tile):
    if size<=tile:
        return [0]
    locations=list(range(0,size-tile+1,max(1,tile//2)))
    if locations[-1]!=size-tile:
        locations.append(size-tile)
    return locations


def map_patches(x,tile,fn):
    h,w=x.shape[-2:]
    th,tw=min(tile,h),min(tile,w)
    def weights(s):
        return torch.sin((torch.arange(s,device=x.device,dtype=torch.float32)+.5)*(math.pi/s))
    weight=weights(th)[:,None]*weights(tw)[None,:]
    out=torch.zeros_like(x,dtype=torch.float32)
    count=torch.zeros((h,w),device=x.device,dtype=torch.float32)
    for top in starts(h,th):
        for left in starts(w,tw):
            window=(slice(top,top+th),slice(left,left+tw))
            out[...,window[0],window[1]]+=fn(x[...,window[0],window[1]],window).float()*weight
            count[window]+=weight
    return (out/count).to(x.dtype)


def invert_and_redraw(clean,sigmas,predict,*,early,start=0,end=None,invert=True,trace=None,progress=None):
    """Run DDIM in sigma coordinates; zero noise needs a one-sided first slope."""
    n=len(sigmas)-1
    if n<=0:
        return clean
    z=clean
    trace={} if trace is None else trace
    cutoff=math.ceil(n*early)
    for i in (reversed(range(n)) if invert else ()):
        lo,hi=sigmas[i+1],sigmas[i]
        probe=hi if i==n-1 and float(lo)==0. else lo
        if float(probe)==0.:
            continue
        denoised=predict(z,probe,True)
        z=z+((z-denoised)/probe)*(hi-lo)
        if i<cutoff:
            trace[i]=z.to("cpu",copy=True)
        if progress:
            progress("invert",i,z,denoised)
    for i in range(start,n if end is None else min(end,n)):
        hi,lo=sigmas[i],sigmas[i+1]
        if i in trace:
            coefficient=(1.+math.cos(i/n*math.pi))*.5
            z=torch.lerp(trace.pop(i).to(z),z,coefficient)
        denoised=predict(z,hi,False)
        z=denoised if float(lo)==0. else z+((z-denoised)/hi)*(lo-hi)
        if progress:
            progress("redraw",i,z,denoised)
    return z


class HiWaveRedraw:
    RETURN_TYPES=("LATENT",)
    FUNCTION="redraw"
    CATEGORY="Asteria/sampling"

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required":{"model":("MODEL",),"positive":("CONDITIONING",),"negative":("CONDITIONING",),
            "latent_image":("LATENT",),"seed":("INT",{"min":0,"max":2**64-1}),"steps":("INT",{"min":1}),
            "cfg":("FLOAT",),"scheduler":(list(comfy.samplers.SCHEDULER_NAMES)+["flux2"],),
            "denoise":("FLOAT",{"min":0.,"max":1.}),"start":("INT",{"min":0}),
            "end":("INT",{"min":-1}),"continuation":("BOOLEAN",{"default":False})},"optional":{"sigmas":("SIGMAS",)}}

    def redraw(self,model,positive,negative,latent_image,seed,steps,cfg,scheduler,denoise,start,end,continuation,sigmas=None):
        import comfy.model_management as mm
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import latent_preview
        # Only pass inverse samples and the exact sigma sequence between the two
        # stages of this workflow. Never mutate a cached upstream latent.
        state=dict(latent_image)
        prior=state.pop("_asteria_hiwave_handoff",None)
        if denoise==0:
            return (state,)
        if continuation and prior is None:
            raise ValueError("HiWave refiner input has no inversion trajectory")
        if scheduler=="flux2" and sigmas is None:
            raise ValueError("HiWave requires the FLUX.2 resolution-dependent schedule")
        sigmas=(prior[0] if continuation else sigmas if sigmas is not None else comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"),scheduler,max(steps,int(steps/denoise))).cpu()[-(steps+1):])
        steps = len(sigmas) - 1
        if steps <= 0:
            return (state,)
        trace=dict(prior[1]) if continuation else {}
        limit=steps if end<0 else min(steps,end)
        latent=state["samples"]
        mask=state.get("noise_mask")
        preview=latent_preview.prepare_callback(model,steps) if limit>start else None

        def sample(model_k,unused_noise,sequence,extra_args,callback=None,disable=None):
            original=model_k.latent_image
            opts=extra_args.get("model_options",{})
            denoise_mask=extra_args.get("denoise_mask")
            progress=comfy.utils.ProgressBar((0 if continuation else steps)+limit-start)

            def predict(z,sigma,inverse):
                def correct(args):
                    c=args["cond_denoised"]
                    if inverse: return c
                    u=args["uncond_denoised"]
                    # Keep the change made by APG, SAG etc., not their ordinary CFG twice.
                    return args["denoised"]-(cfg-1)*low_band(c-u)
                options={**opts,"disable_cfg1_optimization":True,
                         "sampler_post_cfg_function":[*opts.get("sampler_post_cfg_function",()),correct]}
                def patch(tile,window):
                    mm.throw_exception_if_processing_interrupted()
                    s=sigma.expand(tile.shape[0])
                    if denoise_mask is None:
                        return model_k.inner_model(tile,s,model_options=options,seed=seed)
                    local=comfy.samplers.KSamplerX0Inpaint(model_k.inner_model,sequence)
                    local.latent_image=original[...,window[0],window[1]]
                    local.noise=model_k.noise[...,window[0],window[1]]
                    return local(tile,s,denoise_mask[...,window[0],window[1]],model_options=options,seed=seed)
                return map_patches(z,128,patch)

            def report(phase,i,z,pred):
                if phase=="redraw" and callback is not None:
                    callback({"x":z,"i":i,"denoised":pred,"sigma":sequence[i],"sigma_hat":sequence[i]})
                progress.update(1)
            return invert_and_redraw(original,sequence,predict,early=.6 if max(original.shape[-2:])>=512 else .3,
                                     start=start,end=limit,invert=not continuation,trace=trace,progress=report)

        noise=(torch.zeros_like(latent,device="cpu") if mask is None else comfy.sample.prepare_noise(
            latent,seed,state.get("batch_index")))
        state["samples"]=comfy.sample.sample_custom(model,noise,cfg,comfy.samplers.KSAMPLER(sample),
            sigmas,positive,negative,latent,noise_mask=mask,callback=preview,seed=seed)
        if end>=0 and not continuation:
            state["_asteria_hiwave_handoff"]=(sigmas,trace)
        return (state,)


class ResizeLatentGrid:
    """Unlike stock LatentUpscale, grid dimensions are not hardcoded to an 8x VAE."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "width": ("INT", {"min": 1}),
                             "height": ("INT", {"min": 1}), "method": (["nearest-exact", "nearest", "bilinear", "area", "bicubic", "bislerp"],)}}
    FUNCTION = "resize"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "Asteria/internal"

    def resize(self, samples, width, height, method):
        import comfy.utils
        return ({**samples, "samples": comfy.utils.common_upscale(samples["samples"], width, height, method, "disabled")},)


NODE_CLASS_MAPPINGS={"AsteriaHiWaveRedraw":HiWaveRedraw, "AsteriaResizeLatentGrid":ResizeLatentGrid}
