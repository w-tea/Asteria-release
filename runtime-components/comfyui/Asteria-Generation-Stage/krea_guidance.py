"""Krea's transformer publishes the text/image boundary on its joint blocks."""


def is_krea(net):
    from comfy.ldm.krea2.model import SingleStreamDiT
    return isinstance(net, SingleStreamDiT)


def wrap_blocks(model, indices, modify):
    def wrap(original):
        def forward(x, *args, transformer_options, **kwargs):
            start, end = transformer_options["img_slice"]
            end -= sum(transformer_options.get("reference_image_num_tokens", ()))
            x, opts = modify(x, transformer_options, start, end)
            return original(x, *args, transformer_options=opts, **kwargs)
        return forward
    for index in indices:
        path = f"diffusion_model.blocks.{index}.forward"
        model.add_object_patch(path, wrap(model.get_model_object(path)))


def attention_options(options, operation):
    # Krea supplies post-RoPE, GQA-expanded [B,H,N,D] tensors, before its output gate.
    prior = options.get("optimized_attention_override")
    def apply(native, q, k, v, heads, **kwargs):
        base = prior(native, q, k, v, heads, **kwargs) if prior else native(q, k, v, heads, **kwargs)
        return operation(base, q, k, v, kwargs.get("mask"))
    return {**options, "optimized_attention_override": apply}
