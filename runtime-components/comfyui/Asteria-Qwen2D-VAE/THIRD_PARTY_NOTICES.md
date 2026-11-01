# Third-party notice

The Qwen2D VAE architecture in this component is adapted from the
Apache-2.0-licensed `AutoencoderKLQwenImage` implementation maintained by the
Qwen-Image, Wan, and Hugging Face teams.

- Upstream repository: https://github.com/huggingface/diffusers
- Reviewed revision: `e87b2a7ad836f843295dbc60428e29ec4d5aadb0`
- Upstream file: `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`
- Upstream license: Apache-2.0
- Model format: https://huggingface.co/Anzhc/Qwen2D-Anime-VAE

Asteria removes temporal convolution and caching from the upstream video VAE
architecture, keeps image convolutions two-dimensional, returns the posterior
mean expected by ComfyUI, and integrates the result with ComfyUI's managed VAE
device and tiling lifecycle. The source of `anzhc-qwen2d-comfyui` by Anzhc was
used to confirm the model-identification contract and expected ComfyUI behavior;
its unlicensed source is not copied into this component.
