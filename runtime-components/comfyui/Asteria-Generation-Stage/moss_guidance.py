"""Adapters for MOSS's image → text → optional reference sequence."""


def is_moss(net):
    from comfy.ldm.moss_image.model import MOSSImage
    return isinstance(net, MOSSImage)


def wrap_layers(model, name, indices, modify):
    from comfy.patcher_extension import WrappersMP

    layout_key = "asteria_moss_layout_" + name

    def layout(executor, x, timestep, context, transformer_options, **kwargs):
        return executor(x, timestep, context, {**transformer_options, layout_key:
                        (x.shape[-2] * x.shape[-1], context.shape[1])}, **kwargs)

    model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, layout_key, layout)

    def wrap(original):
        def forward(x, mod, pe, transformer_options, *args, **kwargs):
            spans = transformer_options.get(layout_key)
            # preprocess_refs uses the same layer but contains only reference tokens.
            if spans and kwargs.get("kv_out") is None:
                x, transformer_options = modify(x, transformer_options, *spans)
            return original(x, mod, pe, transformer_options, *args, **kwargs)
        return forward

    for i in indices:
        path = f"diffusion_model.layers.{i}.forward"
        model.add_object_patch(path, wrap(model.get_model_object(path)))


def attention_options(options, operation):
    previous = options.get("optimized_attention_override")

    def apply(native, q, k, v, heads, **kwargs):
        base = previous(native, q, k, v, heads, **kwargs) if previous else native(q, k, v, heads, **kwargs)
        def split(t):
            return t.unflatten(-1, (heads, -1)).transpose(1, 2)
        return operation(base, split(q), split(k), split(v), kwargs.get("mask"))
    return {**options, "optimized_attention_override": apply}
