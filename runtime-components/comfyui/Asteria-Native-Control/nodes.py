"""Asteria-owned adapters for existing Krea/MOSS control weights.

No external custom-node implementation is imported. ComfyUI owns model patches,
VAE encoding and conditioning. Spatial control is independent of ordinary LoRAs.
"""
import torch
import torch.nn.functional as F
import comfy.sd
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter
import folder_paths
import node_helpers

from .catalog import control_models, control_profile


def _load(name, profile):
    path = folder_paths.get_full_path_or_raise('loras', name)
    weights = comfy.utils.load_torch_file(path, safe_load=True)
    if control_profile({k: {'shape': list(v.shape)} for k, v in weights.items()}) != profile:
        raise ValueError('Selected control adapter has incompatible input channels')
    if profile == 'krea2':
        weights = {key.removeprefix('diffusion_model.'): value for key, value in weights.items()}
    return weights


def _map_latent(vae, image, width, height):
    image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, 'bilinear', 'disabled').movedim(1, -1)
    return vae.encode(image[:, :, :, :3])


def _krea_loras(weights):
    result = {}
    for key, value in weights.items():
        if key.startswith('first.'):
            continue
        if key.endswith('.A'):
            key = key[:-2] + '.lora_down.weight'
        elif key.endswith('.B'):
            key = key[:-2] + '.lora_up.weight'
        result['diffusion_model.' + key.removeprefix('diffusion_model.')] = value
    return result


class DepthProjection:
    def __init__(self, projected, start, end):
        self.projected, self.start, self.end = projected, start, end

    def __call__(self, args):
        sigma = args['transformer_options'].get('sigmas')
        if sigma is None:
            raise ValueError('Sampling timestep is required for depth control')
        if sigma[0] > self.start or sigma[0] < self.end:
            return args
        image = args['img']
        addition = comfy.utils.repeat_to_batch_size(self.projected.to(image), image.shape[0])
        if addition.shape != image.shape:
            raise ValueError('Depth map token size differs from the generated image')
        return {**args, 'img': image + addition}


class AsteriaKreaDepthControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'vae': ('VAE',),
            'control_name': (control_models('krea2'),), 'image': ('IMAGE',),
            'width': ('INT', {'default': 1024, 'min': 1}), 'height': ('INT', {'default': 1024, 'min': 1}),
            'strength': ('FLOAT', {'default': 1.0}),
            'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0}),
            'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = 'Asteria/control'

    def apply(self, model, vae, control_name, image, width, height, strength, start_percent, end_percent):
        if not strength or start_percent >= end_percent:
            return (model,)
        weights = _load(control_name, 'krea2')
        loaded = model.get_attachment('asteria_depth_adapter')
        if loaded and loaded != control_name:
            raise ValueError('Krea depth inputs must use the same adapter')
        if not loaded:
            result, _ = comfy.sd.load_lora_for_models(model, None, _krea_loras(weights), 1.0, 0.0)
            # The trained first projection replaces only the base projection.
            # The workflow appends ordinary LoRAs afterwards, so their patches survive.
            result.add_patches({
                'diffusion_model.first.weight': ('set', (weights['first.weight'][:, :64].contiguous(),)),
                'diffusion_model.first.bias': ('set', (weights['first.bias'],)),
            })
            result.set_attachments('asteria_depth_adapter', control_name)
        else:
            result = model.clone()
        latent = model.model.process_latent_in(_map_latent(vae, image, width, height))
        if latent.ndim == 5 and latent.shape[2] == 1:
            latent = latent[:, :, 0]
        if latent.ndim != 4 or latent.shape[1] != 16:
            raise ValueError('Krea depth control requires the Krea image VAE')
        # Same token order as ComfyUI Krea process_img: (channel, patch-y, patch-x).
        b, c, h, w = latent.shape
        latent = F.pad(latent, (0, w % 2, 0, h % 2))
        h, w = latent.shape[-2:]
        tokens = latent.reshape(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, 64)
        projection = F.linear(tokens.float(), weights['first.weight'][:, 64:].float().to(tokens.device)) * strength
        sampling = result.get_model_object('model_sampling')
        result.set_model_patch(DepthProjection(projection.cpu(), sampling.percent_to_sigma(start_percent),
                                               sampling.percent_to_sigma(end_percent)), 'post_input')
        return (result,)


class AsteriaMossControlLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'control_name': (control_models('moss_image'),)}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'Asteria/control'

    def load(self, model, control_name):
        weights = _load(control_name, 'moss_image')
        result, _ = comfy.sd.load_lora_for_models(model, None, weights, 1.0, 0.0)
        return (result,)


class AsteriaMossLoraLoader:
    """Keep ordinary MOSS LoRAs on the original image columns, not the maps."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',),
            'lora_name': (folder_paths.get_filename_list('loras'),),
            'strength_model': ('FLOAT', {'default': 1.0}),
            'strength_clip': ('FLOAT', {'default': 1.0})}}
    RETURN_TYPES = ('MODEL', 'CLIP')
    FUNCTION = 'load_lora'
    CATEGORY = 'Asteria/control'

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        path = folder_paths.get_full_path_or_raise('loras', lora_name)
        weights, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        key = 'diffusion_model.video_embedder.weight'
        previous_count = len(model.patches.get(key, ()))
        result, clip = comfy.sd.load_lora_for_models(
            model, clip, weights, strength_model, strength_clip, lora_metadata=metadata)
        patches = list(result.patches.get(key, ()))
        for index in range(previous_count, len(patches)):
            strength, patch, model_strength, offset, function = patches[index]
            # Comfy's offset applies all of its usual alpha/DoRA/strength math
            # to a view. Existing control patches and previous LoRAs stay intact.
            # A LoRA explicitly trained on all 384 inputs keeps that full shape.
            full_input = isinstance(patch, LoRAAdapter) and patch.weights[1].shape[1] == 384
            if isinstance(patch, tuple) and len(patch) == 2 and patch[0] in {'diff', 'set'}:
                full_input = patch[1][0].ndim == 2 and patch[1][0].shape[1] == 384
            if offset is None and not full_input:
                patches[index] = (strength, patch, model_strength, (1, 0, 128), function)
        if patches:
            result.patches[key] = patches
        return result, clip


class AsteriaMossControlMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'vae': ('VAE',), 'image': ('IMAGE',), 'kind': (['depth', 'color'],),
            'width': ('INT', {'default': 1024, 'min': 1}), 'height': ('INT', {'default': 1024, 'min': 1}),
            'strength': ('FLOAT', {'default': 1.0}),
            'start_percent': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0}),
            'end_percent': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0})},
            'optional': {'previous': ('ASTERIA_MOSS_MAPS',)}}
    RETURN_TYPES = ('ASTERIA_MOSS_MAPS',)
    FUNCTION = 'encode'
    CATEGORY = 'Asteria/control'

    def encode(self, vae, image, kind, width, height, strength, start_percent, end_percent, previous=None):
        entries = list(previous or ())
        if strength and start_percent < end_percent:
            latent = _map_latent(vae, image, width, height) * strength
            zero = torch.zeros_like(latent)
            combined = torch.cat((latent, zero) if kind == 'color' else (zero, latent), dim=1)
            entries.append((start_percent, end_percent, combined))
        return (entries,)


class AsteriaMossControlApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'positive': ('CONDITIONING',), 'negative': ('CONDITIONING',),
                             'maps': ('ASTERIA_MOSS_MAPS',)}}
    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING')
    FUNCTION = 'apply'
    CATEGORY = 'Asteria/control'

    def apply(self, positive, negative, maps):
        # Use the native MOSS conditioning key. Split only at requested time
        # boundaries, summing active maps instead of overwriting another unit.
        bounds = sorted({0.0, 1.0, *(x for start, end, _ in maps for x in (start, end))})
        for start, end in zip(bounds, bounds[1:]):
            active = [latent for lo, hi, latent in maps if lo <= start and end <= hi]
            if not active:
                continue
            batch = max(latent.shape[0] for latent in active)
            control = sum(comfy.utils.resize_to_batch_size(latent, batch) for latent in active)
            values = {'concat_latent_image': control}
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, values, start, end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, values, start, end)
        return positive, negative
