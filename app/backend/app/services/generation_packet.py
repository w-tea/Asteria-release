from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4


class ForgeLineageTxt2ImgPacketBuilder:
    """Build A1111/Forge-lineage txt2img packets for Asteria runtime adapters."""

    def build(self, request: dict) -> dict:
        settings = request["settings"]
        model_slots = request.get("model_slots", {})
        prompt = self._prompt_with_loras(request.get("expanded_prompt", ""), request.get("loras", []))
        mode = settings.get("mode", "txt2img")
        payload = {
            "prompt": prompt,
            "negative_prompt": request.get("negative_prompt", ""),
            "styles": [],
            "seed": settings.get("seed", -1),
            "subseed": settings.get("subseed", -1),
            "subseed_strength": settings.get("subseed_strength", 0),
            "seed_resize_from_w": settings.get("seed_resize_from_w", 0),
            "seed_resize_from_h": settings.get("seed_resize_from_h", 0),
            "sampler_name": settings.get("sampler", "Euler a"),
            "scheduler": settings.get("scheduler", "Automatic"),
            "batch_size": settings.get("batch_size", 1),
            "n_iter": settings.get("batch_count", 1),
            "steps": settings["steps"],
            "cfg_scale": settings["cfg_scale"],
            "distilled_cfg_scale": settings.get("distilled_cfg_scale", 3.5),
            "width": settings["width"],
            "height": settings["height"],
            "restore_faces": settings.get("restore_faces", False),
            "tiling": settings.get("tiling", False),
            "enable_hr": settings.get("hires_enabled", False),
            "denoising_strength": settings.get("hires_denoising_strength", 0.7),
            "hr_scale": settings.get("hires_scale", 2),
            "hr_upscaler": settings.get("hires_upscaler", "Latent"),
            "hr_second_pass_steps": settings.get("hires_steps", 0),
            "hr_resize_x": settings.get("hires_resize_width", 0),
            "hr_resize_y": settings.get("hires_resize_height", 0),
            "hr_checkpoint_name": settings.get("hires_checkpoint", "Use same checkpoint"),
            "hr_sampler_name": settings.get("hires_sampler", "Use same sampler"),
            "hr_scheduler": settings.get("hires_scheduler", "Use same scheduler"),
            "hr_prompt": settings.get("hires_prompt", ""),
            "hr_negative_prompt": settings.get("hires_negative_prompt", ""),
            "hr_cfg": settings.get("hires_cfg_scale", 0),
            "hr_distilled_cfg": settings.get("hires_distilled_cfg_scale", 0),
            "refiner_checkpoint": settings.get("refiner_checkpoint", ""),
            "refiner_switch_at": settings.get("refiner_switch_at", 0.8),
            "override_settings": self._override_settings(model_slots, settings),
            "alwayson_scripts": {},
        }
        if mode in {"img2img", "inpaint"}:
            payload["init_images"] = [settings.get("source_init_image_path") or settings.get("init_image_path", "")]
            payload["denoising_strength"] = settings.get("denoising_strength", settings.get("denoise", 0.55))
            payload["enable_hr"] = False
        if mode == "inpaint":
            payload["mask"] = settings.get("source_mask_image_path") or settings.get("mask_image_path", "")
            payload["mask_blur"] = settings.get("mask_blur", 4)
            payload["inpainting_fill"] = settings.get("masked_content", "original")
            payload["inpaint_full_res"] = settings.get("inpaint_area", "whole_picture") != "whole_picture"
        if not settings.get("refiner_enabled"):
            payload.pop("refiner_checkpoint", None)
            payload.pop("refiner_switch_at", None)
        packet = {
            "runtime_job_id": f"asteria-{uuid4().hex}",
            "runtime_adapter": "asteria_native_packet",
            "request_family": f"a1111_forge_{mode}",
            "profile": request.get("profile", "sdxl"),
            "model_slots": model_slots,
            "loras": request.get("loras", []),
            "embeddings": request.get("embeddings", []),
            "prompt_expansion_selections": request.get("prompt_expansion_selections", []),
            "payload": payload,
            "infotext": self._infotext(payload, model_slots),
        }
        prompt_syntax = settings.get("prompt_syntax")
        if isinstance(prompt_syntax, dict):
            packet["prompt_syntax"] = prompt_syntax
        if settings.get("image_edit_lineage"):
            packet["image_edit_lineage"] = settings["image_edit_lineage"]
        return packet

    def write(self, packet: dict, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = str(packet.get("request_family", "txt2img")).rsplit("_", 1)[-1]
        path = output_dir / f"{packet['runtime_job_id']}_{mode}_packet.json"
        path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _override_settings(self, model_slots: dict[str, str], settings: dict) -> dict:
        override_settings = {}
        checkpoint = model_slots.get("checkpoint")
        if checkpoint:
            override_settings["sd_model_checkpoint"] = checkpoint
        diffusion_model = model_slots.get("diffusion_model")
        if diffusion_model:
            override_settings["forge_diffusion_model"] = diffusion_model
        vae = model_slots.get("vae") or settings.get("vae")
        if vae and vae != "Automatic":
            override_settings["sd_vae"] = vae
        additional_modules = [
            value
            for key, value in model_slots.items()
            if key not in {"checkpoint", "diffusion_model", "vae"} and value
        ]
        if additional_modules:
            override_settings["forge_additional_modules"] = additional_modules
        return override_settings

    def _prompt_with_loras(self, prompt: str, loras: list[dict]) -> str:
        additions = []
        for lora in loras:
            name = lora["name"]
            if f"<lora:{name}:" not in prompt:
                clip_weight = lora.get("clip_weight")
                if clip_weight is None or clip_weight == lora.get("weight", 1.0):
                    additions.append(f"<lora:{name}:{lora.get('weight', 1.0):g}>")
                else:
                    additions.append(f"<lora:{name}:{lora.get('weight', 1.0):g}:{clip_weight:g}>")
        if not additions:
            return prompt
        return f"{prompt} {' '.join(additions)}".strip()

    def _infotext(self, payload: dict, model_slots: dict[str, str]) -> str:
        lines = [payload["prompt"]]
        if payload.get("negative_prompt"):
            lines.append(f"Negative prompt: {payload['negative_prompt']}")
        params = {
            "Steps": payload["steps"],
            "Sampler": payload["sampler_name"],
            "Schedule type": payload["scheduler"],
            "CFG scale": payload["cfg_scale"],
            "Distilled CFG Scale": payload["distilled_cfg_scale"],
            "Seed": payload["seed"],
            "Size": f"{payload['width']}x{payload['height']}",
            "Model": model_slots.get("checkpoint") or model_slots.get("diffusion_model", ""),
            "VAE": model_slots.get("vae", ""),
        }
        if payload.get("enable_hr"):
            params["Hires upscale"] = payload.get("hr_scale")
            params["Hires upscaler"] = payload.get("hr_upscaler")
            params["Hires steps"] = payload.get("hr_second_pass_steps")
            params["Denoising strength"] = payload.get("denoising_strength")
        lines.append(", ".join(f"{key}: {value}" for key, value in params.items() if value not in {"", None}))
        return "\n".join(lines)
