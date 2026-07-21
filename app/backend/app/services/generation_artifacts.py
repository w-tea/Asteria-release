from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.services.user_settings import safe_segment


_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class OutputFinalizationError(Exception):
    pass


def apply_output_image_format(output_paths: list[str], image_format: str) -> list[str]:
    target_format = image_format.lower().strip()
    suffix_by_format = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}
    pil_format_by_setting = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}
    target_suffix = suffix_by_format.get(target_format)
    if not target_suffix:
        return output_paths

    converted: list[str] = []
    for output_path in output_paths:
        path = Path(output_path)
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not _is_file(path):
            converted.append(output_path)
            continue
        if path.suffix.lower() == target_suffix:
            converted.append(output_path)
            continue
        try:
            destination = unique_output_path(path.with_suffix(target_suffix))
            with Image.open(path) as image:
                output_image = image.convert("RGB") if target_format == "jpg" else image
                output_image.save(destination, format=pil_format_by_setting[target_format])
        except _PATH_ACCESS_ERRORS as exc:
            raise OutputFinalizationError(f"Could not finalize generated output: {exc}") from exc
        try:
            path.unlink()
        except _PATH_ACCESS_ERRORS:
            pass
        converted.append(str(destination))
    return converted


def apply_filename_template(
    output_paths: list[str],
    settings: dict,
    profile: str,
    generation_id: int,
) -> list[str]:
    template = str(settings.get("filename_template") or "").strip()
    if not template:
        return output_paths
    renamed: list[str] = []
    image_index = 0
    for output_path in output_paths:
        path = Path(output_path)
        if path.suffix.lower() not in _IMAGE_SUFFIXES or not _is_file(path):
            renamed.append(output_path)
            continue
        image_index += 1
        seeds = settings.get("generated_seeds", [settings.get("seed", "")])
        seed = seeds[image_index - 1] if isinstance(seeds, list) and len(seeds) >= image_index else settings.get("seed", "")
        raw_name = template.format(
            profile=profile,
            seed=seed,
            id=generation_id,
            index=image_index,
            sampler=settings.get("sampler", ""),
            scheduler=settings.get("scheduler", ""),
        )
        stem = safe_segment(raw_name) or f"{profile}_{generation_id}_{image_index}"
        try:
            destination = unique_output_path(path.with_name(f"{stem}{path.suffix.lower()}"))
            path.replace(destination)
        except _PATH_ACCESS_ERRORS:
            renamed.append(output_path)
            continue
        renamed.append(str(destination))
    return renamed


def remap_primary_outputs(
    primary_outputs: list[dict],
    before_paths: list[str],
    after_paths: list[str],
) -> tuple[list[dict], bool]:
    path_mapping = dict(zip(before_paths, after_paths, strict=True)) if len(before_paths) == len(after_paths) else {}
    after_set = set(after_paths)
    remapped: list[dict] = []
    for output in primary_outputs:
        old_path = str(output.get("path") or "")
        new_path = path_mapping.get(old_path, old_path if old_path in after_set else "")
        if not new_path:
            return [], False
        remapped.append({**output, "path": new_path})
    return remapped, True


def unique_output_path(path: Path) -> Path:
    if _is_available(path):
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if _is_available(candidate):
            return candidate
    return path.with_name(f"{path.stem}_{uuid4().hex[:8]}{path.suffix}")


def _is_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except _PATH_ACCESS_ERRORS:
        return False


def _is_available(path: Path) -> bool:
    try:
        return not path.exists()
    except _PATH_ACCESS_ERRORS:
        return False
