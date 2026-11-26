"""FLUX.1/2 pass text and image features separately into their joint blocks."""


def middle_image_block(model, net, modify):
    path = f"diffusion_model.double_blocks.{(len(net.double_blocks)-1)//2}.forward"
    original = model.get_model_object(path)

    def forward(img, txt, *args, transformer_options=None, **kwargs):
        opts = transformer_options or {}
        image_len = img.shape[1] - sum(opts.get("reference_image_num_tokens", ()))
        img, opts = modify(img, opts, txt.shape[1], image_len)
        return original(img, txt, *args, transformer_options=opts, **kwargs)

    model.add_object_patch(path, forward)
