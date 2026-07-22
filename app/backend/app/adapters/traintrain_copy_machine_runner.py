from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _resolve_path(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return str(Path(value).expanduser().resolve())


def _parse_set_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        try:
            parsed_value: Any = ast.literal_eval(value)
        except Exception:
            parsed_value = value
        overrides[key.strip()] = parsed_value
    return overrides


def _assign_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)
    config["mode"] = args.mode_name
    config["model"] = _resolve_path(args.model) or ""
    config["vae"] = _resolve_path(args.vae) or config.get("vae", "None") or "None"
    config["save_lora_name"] = args.output_name or config.get("save_lora_name", "")
    config["original image"] = _resolve_path(args.orig_image) or ""
    config["target image"] = _resolve_path(args.targ_image) or ""
    config["original prompt"] = args.orig_prompt or config.get("original prompt", "")
    config["target prompt"] = args.targ_prompt or config.get("target prompt", "")
    config["negative prompt"] = args.negative_prompt or config.get("negative prompt", "")
    for key, value in _parse_set_overrides(args.set_override or []).items():
        _assign_dotted(config, key, value)
    return config


def _write_effective_config_snapshot(
    config_path: Path,
    config: dict[str, Any],
    output_path: Path,
) -> None:
    resolved_config_path = config_path.expanduser().resolve()
    payload = {
        "schema": "asteria.copy_machine_effective_config.v1",
        "source_config_path": str(resolved_config_path),
        "source_config_sha256": hashlib.sha256(resolved_config_path.read_bytes()).hexdigest(),
        "effective_config": config,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _import_traintrain(traintrain_root: Path):
    if str(traintrain_root) not in sys.path:
        sys.path.insert(0, str(traintrain_root))
    from traintrain.trainer import trainer as trainer_module
    from traintrain.trainer import train as train_module
    import traintrain.scripts.traintrain  # noqa: F401 - populates trainer.all_configs

    return trainer_module, train_module


def _install_sdxl_clip_skip(trainer_module: Any, clip_skip: int) -> None:
    if clip_skip <= 1:
        return
    original_loader = trainer_module.load_checkpoint_model_xl

    def load_checkpoint_model_xl(*args: Any, **kwargs: Any):
        text_model, unet, vae = original_loader(*args, **kwargs)
        text_model.clip_skip = -clip_skip
        return text_model, unet, vae

    trainer_module.load_checkpoint_model_xl = load_checkpoint_model_xl


def run_training(args: argparse.Namespace) -> None:
    config = _build_config(args)
    if args.effective_config_output:
        _write_effective_config_snapshot(
            Path(args.config),
            config,
            Path(args.effective_config_output).expanduser().resolve(),
        )
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    traintrain_root = Path(args.traintrain_root).expanduser().resolve()
    for path_attr in ("models_dir", "ckpt_dir", "vae_dir", "lora_dir", "text_encoder_dir"):
        value = getattr(args, path_attr)
        if value:
            setattr(args, path_attr, _resolve_path(value) or "")
    Path(args.lora_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(traintrain_root)
    trainer_module, train_module = _import_traintrain(traintrain_root)
    _install_sdxl_clip_skip(trainer_module, args.clip_skip)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
        json.dump(config, tmp_file, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp_file.name)
    try:
        inputs = trainer_module.import_json(str(tmp_path), cli=True)
        cfg_len = len(trainer_module.all_configs)
        prompt_start = 3 + cfg_len * 2
        prompts = [
            config.get("original prompt", ""),
            config.get("target prompt", ""),
            config.get("negative prompt", ""),
        ]
        for idx, prompt in enumerate(prompts):
            inputs[prompt_start + idx] = prompt
        inputs[prompt_start + 3] = config.get("original image", "")
        inputs[prompt_start + 4] = config.get("target image", "")
        paths = [
            _resolve_path(args.models_dir),
            _resolve_path(args.ckpt_dir),
            _resolve_path(args.vae_dir),
            _resolve_path(args.lora_dir),
            _resolve_path(args.text_encoder_dir),
        ]
        result = train_module.train_main(paths, inputs[0], inputs[1], inputs[2], args.text_encoder or "None", *inputs[3:])
        print(result)
    finally:
        tmp_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Asteria TrainTrain copy-machine runner.")
    parser.add_argument("command", choices=["addift", "difference"])
    parser.add_argument("--traintrain-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vae", default="")
    parser.add_argument("--text-encoder", default="None")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--orig-image", required=True)
    parser.add_argument("--targ-image", required=True)
    parser.add_argument("--orig-prompt", default="")
    parser.add_argument("--targ-prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--clip-skip", type=int, default=1)
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--ckpt-dir", default="")
    parser.add_argument("--vae-dir", default="")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--text-encoder-dir", default="")
    parser.add_argument("--set", dest="set_override", action="append", default=[])
    parser.add_argument("--effective-config-output", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.mode_name = "ADDifT" if args.command == "addift" else "Difference"
    run_training(args)


if __name__ == "__main__":
    main()
