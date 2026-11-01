"""Scope hooks to one native model call. No replacement attention output."""
from contextlib import ExitStack
import math
import time

import torch

from .word_maps import AttentionTap, MAX_CPU_BYTES, positive_rows, probability_maps


class WordObserver:
    def __init__(self, contract, *, height, width):
        self.contract = contract
        self.columns = [c['position'] for c in contract['columns'] if c['labels']]
        scale = min(1., 96 / max(height, width))
        self.output_grid = max(1, round(height*scale)), max(1, round(width*scale))
        self.total = None
        self.count = 0
        self.seconds = 0.
        self.error = None
        self.points = set()
        self.seen = set()
        self.observations = []

    def schedule(self, sigmas):
        values = sigmas.detach().float().cpu().tolist()[:-1]
        self.points = {float(values[min(len(values)-1, int(fraction*len(values)))])
                       for fraction in (.1,.3,.5,.7,.9)} if values else set()

    def apply(self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        options = transformer_options or {}
        flags = options.get('cond_or_uncond', [])
        if self.error or self.seconds >= 30 or 0 not in flags or not self.columns:
            return executor(x,t,c_concat,c_crossattn,control,options,**kwargs)
        sigma = float(t.flatten()[0])
        if sigma not in self.points:
            return executor(x,t,c_concat,c_crossattn,control,options,**kwargs)
        diffusion = executor.class_obj.diffusion_model
        modules = [(name, module) for name, module in diffusion.named_modules()
                   if name.endswith('.attn2') or name == 'attn2'
                   or (name.startswith('blocks.') and name.endswith('.cross_attn'))]
        chosen = [modules[i] for i in sorted({min(len(modules)-1,int(len(modules)*f)) for f in (.25,.5,.75)})] if modules else []
        if not chosen:
            self.error = 'cross_attention_not_found'
            return executor(x,t,c_concat,c_crossattn,control,options,**kwargs)
        taps = []

        def make_hook(name):
            def hook(module, args, call_kwargs):
                if (sigma,name) in self.seen: return None
                current = dict(call_kwargs.get('transformer_options', {}))
                if 'optimized_attention_override' in current:
                    self.error = 'attention_override_not_supported'
                    return None

                def observe(q,k,v,heads,mask=None,**attention_kwargs):
                    if mask is not None: raise ValueError('attention_mask_not_supported')
                    if (sigma,name) in self.seen: return
                    self.seen.add((sigma,name))
                    start = time.perf_counter()
                    try:
                        if not attention_kwargs.get('skip_reshape',False):
                            q = q.view(q.shape[0],q.shape[1],heads,-1).transpose(1,2)
                            k = k.view(k.shape[0],k.shape[1],heads,-1).transpose(1,2)
                        rows = positive_rows(q.shape[0],flags)
                        activations = current.get('activations_shape')
                        if self.contract['profile']=='sdxl' and activations:
                            grid = tuple(activations[-2:])
                        else:
                            # The native Anima spatial patch is 2x2, T=1.
                            grid = (math.ceil(x.shape[-2]/2), math.ceil(x.shape[-1]/2))
                        if math.prod(grid) != q.shape[-2]: raise ValueError('attention_grid_mismatch')
                        # Slices avoid duplicating the model's full Q/K batch.
                        predicted = 4*len(rows)*len(self.columns)*(math.prod(grid)+3*math.prod(self.output_grid))
                        if predicted > MAX_CPU_BYTES: raise ValueError('attention_memory_limit')
                        values = probability_maps(q[rows[0]:rows[-1]+1],k[rows[0]:rows[-1]+1],self.columns,
                                                  grid=grid,output_grid=self.output_grid,
                                                  context_length=self.contract['context_length'])
                        if self.total is None: self.total = values
                        else:
                            if self.total.shape != values.shape: raise ValueError('attention_batch_changed')
                            self.total.add_(values)
                        self.count += 1
                        self.observations.append({'module':name,'sigma':sigma,'query_grid':list(grid),
                                                  'context_length':k.shape[-2], 'positive_rows':rows})
                    finally: self.seconds += time.perf_counter()-start
                tap = AttentionTap(observe);taps.append(tap)
                current['optimized_attention_override'] = tap
                return args, {**call_kwargs,'transformer_options':current}
            return hook

        with ExitStack() as stack:
            for name, module in chosen:
                handle = module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
                stack.callback(handle.remove)
            output = executor(x,t,c_concat,c_crossattn,control,options,**kwargs)
        self.error = self.error or next((tap.error for tap in taps if tap.error),None)
        if self.seconds >= 30: self.error = 'attention_time_limit'
        return output

    def result(self):
        if self.error or self.total is None or not self.count:
            return None, {'status':'not_saved','reason':self.error or 'no_observations','seconds':self.seconds}
        self.total.div_(self.count)
        return self.total, {'status':'recorded','contract':self.contract,'columns':self.columns,
                            'grid':list(self.output_grid),'observations':self.observations,'seconds':self.seconds,
                            'count':self.count,'bytes':self.total.numel()*self.total.element_size()}
