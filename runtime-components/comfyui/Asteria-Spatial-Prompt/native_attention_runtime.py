"""Finite early placement bias over one unchanged full-text attention operation."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class NativeWindow:
    start: float
    end: float
    full_text_sha256: str
    context_length: int
    columns: tuple[tuple[int, ...], ...]


def shared_columns(encoders):
    """An SDXL row is steerable only when L and G assign it to the same target."""
    return tuple(tuple(sorted(set.intersection(*(
        set(encoder.targets[index].context_indices) for encoder in encoders
    )))) for index in range(len(encoders[0].targets)))


class NativeAttentionRouter:
    def __init__(self, *, windows, regions, family, model_sampling, backend=None,
                 cache_limit=16 * 1024 * 1024):
        self.windows = tuple(windows)
        self.regions = tuple(dict(r) for r in regions)
        self.family, self.backend = family, backend
        if family not in ("sdxl", "anima") or not self.windows or not self.regions:
            raise ValueError("Invalid native placement contract")
        for region in self.regions:
            if region["shape"] not in ("ellipse", "rectangle", "mask"):
                raise ValueError("Unsupported placement geometry")
            for field in ("center_x", "center_y", "width", "height", "strength", "blending", "depth_prior"):
                if not math.isfinite(float(region[field])):
                    raise ValueError("Placement geometry must be finite")
            if region["width"] <= 0 or region["height"] <= 0 or not 0 <= region["strength"] <= 1:
                raise ValueError("Invalid placement size or strength")
        for window in self.windows:
            if len(window.columns) != len(self.regions) or window.context_length < 1:
                raise ValueError("Placement token/target mismatch")
            columns = [i for span in window.columns for i in span]
            if len(columns) != len(set(columns)) or any(i < 0 or i >= window.context_length for i in columns):
                raise ValueError("Invalid full-text placement columns")
        self.window_sigmas = tuple((float(model_sampling.percent_to_sigma(w.start)),
                                   float(model_sampling.percent_to_sigma(w.end))) for w in self.windows)
        self.cache_limit = int(cache_limit)
        self._cache = OrderedDict()
        self._cache_bytes = self.peak_cache_bytes = 0
        self._sigma_identity = None
        self._schedule_identity = None
        self._schedule = ()
        self._step = self._window = 0
        self._active = False
        self.calls = Counter()
        self.routed = Counter()
        self.queries, self.keys = set(), set()
        self.forward_states = set()

    def clear(self):
        self._cache.clear()
        self._cache_bytes = 0

    def _state(self, options):
        current, schedule = options["sigmas"], options["sample_sigmas"]
        # Comfy's production sampler creates inference tensors without counters.
        version = None if current.is_inference() else current._version
        # One small synchronization per forward, not per attention layer.
        if self._sigma_identity is not current or self._sigma_version != version:
            if self._schedule_identity is not schedule:
                self._schedule = tuple(float(v) for v in schedule.detach().flatten().cpu())
                self._schedule_identity = schedule
            if len(self._schedule) < 2 or current.numel() == 0:
                raise ValueError("Missing native sampler schedule")
            sigma = float(current.detach().flatten()[0].cpu())
            self._step = min(range(len(self._schedule)-1), key=lambda i: abs(self._schedule[i]-sigma))
            # Midpoint/evaluation samplers need not use an exact Euler sigma.
            self._active = self._step < math.ceil((len(self._schedule)-1)/2)
            matching = [i for i,(start,end) in enumerate(self.window_sigmas) if end <= sigma <= start]
            self._window = matching[-1] if matching else None
            self._active = self._active and self._window is not None
            self._sigma_identity, self._sigma_version = current, version
            self.forward_states.add((self._step, self._window))
            if not self._active:
                self.clear()
        return self._active, self._window

    @staticmethod
    def _labels(options, batch):
        labels = tuple(options["cond_or_uncond"])
        if not labels or any(i not in (0,1) for i in labels) or batch % len(labels):
            raise ValueError("Invalid placement conditional batch")
        return labels

    @staticmethod
    def _grid(queries, original_shape):
        height, width = (int(v) for v in original_shape[-2:])
        for _ in range(16):
            if height*width == queries:
                return 1,height,width
            height,width=(height+1)//2,(width+1)//2
        raise ValueError("Placement query grid does not match the current latent")

    def post_input(self, payload):
        image = payload["img"]
        if image.ndim != 5:
            raise ValueError("Invalid native Anima image grid")
        payload["transformer_options"]["asteria_spatial_grid"] = tuple(int(v) for v in image.shape[1:4])
        self._state(payload["transformer_options"])
        return payload

    def _bias(self, q, key_count, labels, grid, window_index):
        window = self.windows[window_index]
        if key_count % window.context_length:
            raise ValueError("Native full-text attention length does not match its binding")
        cache_key = (q.device, q.dtype, q.shape[0], key_count, labels, grid, window_index)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        temporal,height,width = grid
        if temporal*height*width != q.shape[1]:
            raise ValueError("Native placement grid changed")
        y,x=torch.meshgrid((torch.arange(height,device=q.device,dtype=torch.float32)+.5)/height,
                           (torch.arange(width,device=q.device,dtype=torch.float32)+.5)/width,indexing="ij")
        fields=[]
        for region in self.regions:
            spread=.75+float(region["blending"])/100.
            dx=(x-region["center_x"])/(region["width"]*.5*spread)
            dy=(y-region["center_y"])/(region["height"]*.5*spread)
            if region["shape"] == "mask":
                field=torch.nn.functional.interpolate(region["mask"].reshape(1,1,*region["mask"].shape[-2:]).to(q.device),
                    size=(height,width),mode="bilinear",align_corners=False)[0,0]
            else:
                distance=dx.square()+dy.square() if region["shape"] == "ellipse" else torch.maximum(dx.abs(),dy.abs()).square()
                field=torch.exp(-.5*distance)
            # Geometry is a finite preference, never a hard drawing boundary.
            value=3.*(2.*field-1.)*region["strength"]
            value=value+math.log(max(.05,region["depth_prior"]))*field*region["strength"]
            fields.append(value.flatten().repeat(temporal).to(q.dtype))
        active=[i for i,r in enumerate(self.regions) if r["strength"] > 0 and window.columns[i]]
        common=torch.stack([fields[i] for i in active]).amax(0) if self.family == "anima" and len(active)>1 else None
        columns=torch.zeros((temporal*height*width,window.context_length),device=q.device,dtype=q.dtype)
        for i in active:
            values=fields[i] if common is None else fields[i]-common
            columns[:,list(window.columns[i])]=values[:,None]
        columns=columns.repeat(1,key_count//window.context_length)
        mask=torch.zeros((q.shape[0],temporal*height*width,key_count),device=q.device,dtype=q.dtype)
        count=q.shape[0]//len(labels)
        for index,label in enumerate(labels):
            if label == 0:
                mask[index*count:(index+1)*count]=columns
        size=mask.numel()*mask.element_size()
        if size <= self.cache_limit:
            while self._cache and self._cache_bytes+size > self.cache_limit:
                _,old=self._cache.popitem(last=False)
                self._cache_bytes-=old.numel()*old.element_size()
            self._cache[cache_key]=mask
            self._cache_bytes+=size
            self.peak_cache_bytes=max(self.peak_cache_bytes,self._cache_bytes)
        return mask

    def _backend(self, q, mask):
        if self.backend is not None:
            return self.backend
        from comfy.ldm.modules.attention import optimized_attention_for_device
        return optimized_attention_for_device(q.device,mask=mask is not None)

    def sdxl(self,q,k,v,options):
        labels=self._labels(options,q.shape[0])
        active,index=self._state(options)
        grid=self._grid(q.shape[1],options["original_shape"])
        mask=self._bias(q,k.shape[1],labels,grid,index) if active and 0 in labels else None
        result=self._backend(q,mask)(q,k,v,options["n_heads"],mask=mask,
            attn_precision=options.get("attn_precision"),transformer_options=options)
        layer=(*options["block"],options["block_index"])
        self._record(layer,q.shape[1],k.shape[1],mask is not None)
        return result

    def anima(self,index,original,q,k,v,options):
        labels=self._labels(options,q.shape[0])
        active,window=self._state(options)
        self._record(("anima",index),q.shape[1],k.shape[1],active and 0 in labels)
        if not active or 0 not in labels:
            return original(q,k,v,transformer_options=options)
        grid=options["asteria_spatial_grid"]
        mask=self._bias(q,k.shape[1],labels,grid,window)
        return self._backend(q,mask)(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),q.shape[-2],
            mask=mask,skip_reshape=True,transformer_options=options)

    def _record(self,layer,queries,keys,routed):
        self.calls[layer]+=1
        if routed:
            self.routed[layer]+=1
            self.queries.add(queries)
            self.keys.add(keys)

    def snapshot(self):
        return dict(router_revision="asteria-native-full-text-v9", attention_calls=sum(self.calls.values()),
            routed_calls=sum(self.routed.values()), layers=[list(k) for k in sorted(self.routed)],
            query_counts=sorted(self.queries),key_counts=sorted(self.keys),
            mask_cache_bytes=self._cache_bytes,peak_mask_cache_bytes=self.peak_cache_bytes,
            cache_limit_bytes=self.cache_limit,full_attention_calls=sum(self.calls.values()),
            windows=[dict(start=w.start,end=w.end,context_length=w.context_length,
                          columns=[list(c) for c in w.columns]) for w in self.windows])


class AnimaCrossCall:
    def __init__(self,router,index,original):
        self.router,self.index,self.original=router,index,original

    def __call__(self,q,k,v,transformer_options):
        return self.router.anima(self.index,self.original,q,k,v,transformer_options)
