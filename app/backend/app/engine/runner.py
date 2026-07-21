from __future__ import annotations

import json
import sys
from pathlib import Path

from app.engine.png import write_solid_png


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(json.dumps({"status": "failed", "error": "usage: runner.py <packet_path> <output_dir>"}))
        return 2
    packet_path = Path(args[0])
    output_dir = Path(args[1])
    try:
        result = run(packet_path, output_dir)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


def run(packet_path: Path, output_dir: Path) -> dict:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("request_family") != "a1111_forge_txt2img":
        raise ValueError(f"unsupported request_family: {packet.get('request_family')}")
    payload = packet.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("packet payload is required")
    width = int(payload.get("width", 512))
    height = int(payload.get("height", 512))
    runtime_job_id = packet["runtime_job_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{runtime_job_id}_preview.png"
    manifest_path = output_dir / f"{runtime_job_id}_engine_manifest.json"
    write_solid_png(image_path, min(width, 512), min(height, 512), _preview_color(payload))
    manifest = {
        "runtime_job_id": runtime_job_id,
        "status": "preview",
        "note": "Asteria native engine runner validated the packet and emitted a development preview artifact.",
        "request_family": packet["request_family"],
        "profile": packet.get("profile"),
        "prompt": payload.get("prompt", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "sampler_name": payload.get("sampler_name"),
        "scheduler": payload.get("scheduler"),
        "steps": payload.get("steps"),
        "cfg_scale": payload.get("cfg_scale"),
        "model_slots": packet.get("model_slots", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "preview",
        "output_paths": [str(image_path), str(manifest_path)],
        "info": "asteria_native_engine_preview",
    }


def _preview_color(payload: dict) -> tuple[int, int, int]:
    seed = int(payload.get("seed", -1))
    prompt = str(payload.get("prompt", ""))
    value = seed if seed >= 0 else sum(prompt.encode("utf-8"))
    return (
        48 + value % 160,
        48 + (value // 7) % 160,
        48 + (value // 17) % 160,
    )


if __name__ == "__main__":
    raise SystemExit(main())
