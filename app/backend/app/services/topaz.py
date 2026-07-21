from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import uuid

from PIL import Image
from pydantic import BaseModel, Field


class TopazSettings(BaseModel):
    enabled: bool = False
    lighting: bool = False
    color: bool = False
    sharpen: bool = False
    denoise: bool = False
    override: bool = False
    show_settings: bool = False
    png_compression: int = Field(default=2, ge=0, le=10)
    upscale_model: str = "Graphics"
    upscale_scale: int = Field(default=4, ge=1, le=8)


def build_topaz_command(
    executable: str,
    input_path: str,
    output_dir: str,
    settings: TopazSettings,
) -> list[str]:
    command = [
        executable,
        input_path,
        "--output",
        output_dir,
        "--format",
        "png",
        "--overwrite",
        "--compression",
        str(settings.png_compression),
    ]

    if settings.show_settings:
        command.append("--showSettings")

    if settings.enabled:
        if settings.override:
            command.append("--override")
        command.append("--upscale=enabled=true")
        command.append(f"--lighting=enabled={'true' if settings.lighting else 'false'}")
        command.append(f"--color=enabled={'true' if settings.color else 'false'}")
        command.append(f"--sharpen=enabled={'true' if settings.sharpen else 'false'}")
        command.append(f"--noise=enabled={'true' if settings.denoise else 'false'}")

    return command


class TopazError(RuntimeError):
    pass


_PATH_ACCESS_ERRORS = (TypeError, OSError, ValueError, RuntimeError)


def _path_exists_safely(path: Path) -> bool:
    try:
        return path.exists()
    except _PATH_ACCESS_ERRORS:
        return False


def _path_is_file_safely(path: Path) -> bool:
    try:
        return path.is_file()
    except _PATH_ACCESS_ERRORS:
        return False


def _new_file_candidate_is_available(path: Path) -> bool:
    try:
        return not path.exists()
    except _PATH_ACCESS_ERRORS:
        return False


def process_topaz_image(
    *,
    executable: Path,
    input_path: Path,
    output_dir: Path,
    settings: TopazSettings,
    target_size: tuple[int, int] | None = None,
) -> dict[str, str | list[str] | dict[str, int] | None]:
    if not _path_is_file_safely(executable):
        raise TopazError(f"Topaz Photo AI executable not found: {executable}")
    if not _path_is_file_safely(input_path):
        raise TopazError(f"Input image not found: {input_path}")

    work_dir: Path | None = None
    try:
        if _path_exists_safely(output_dir) and not output_dir.is_dir():
            raise TopazError(f"Topaz output path is not a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "topaz_work" / uuid.uuid4().hex
        work_dir.mkdir(parents=True, exist_ok=True)
        work_input = work_dir / input_path.name
        shutil.copy2(input_path, work_input)
    except TopazError:
        raise
    except _PATH_ACCESS_ERRORS as exc:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise TopazError(f"Topaz file preparation failed: {exc}") from exc
    input_size = _image_size(input_path)

    command = build_topaz_command(str(executable), str(work_input), str(work_dir), settings)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise TopazError(f"Topaz execution failed: {exc.stderr or exc.stdout or exc.returncode}") from exc
    except _PATH_ACCESS_ERRORS as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise TopazError(f"Topaz execution failed: {exc}") from exc

    try:
        candidates = sorted(
            [path for path in work_dir.glob("*.png") if path.name != work_input.name],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except _PATH_ACCESS_ERRORS as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise TopazError(f"Topaz output discovery failed: {exc}") from exc
    produced = candidates[0] if candidates else work_input
    destination = output_dir / f"{input_path.stem}_topaz.png"
    while not _new_file_candidate_is_available(destination):
        destination = output_dir / f"{input_path.stem}_topaz_{uuid.uuid4().hex[:8]}.png"
    try:
        shutil.copy2(produced, destination)
    except _PATH_ACCESS_ERRORS as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise TopazError(f"Topaz file finalization failed: {exc}") from exc
    produced_size = _image_size(destination)
    if target_size is not None and input_size is not None and produced_size is not None:
        target_is_larger = target_size[0] > input_size[0] or target_size[1] > input_size[1]
        produced_not_larger = produced_size[0] <= input_size[0] and produced_size[1] <= input_size[1]
        if target_is_larger and produced_not_larger:
            try:
                destination.unlink(missing_ok=True)
            except _PATH_ACCESS_ERRORS:
                pass
            shutil.rmtree(work_dir, ignore_errors=True)
            raise TopazError(
                "Topaz output did not upscale the source image; refusing to create a Hires.fix result "
                "with Asteria-only resize."
            )
    logs: list[str] = []
    if target_size is not None and produced_size is not None and (
        produced_size[0] < target_size[0] or produced_size[1] < target_size[1]
    ):
        logs.append(
            "Topaz output is smaller than requested Hires target; "
            f"keeping Topaz-produced size {produced_size[0]}x{produced_size[1]} instead of upscaling to {target_size[0]}x{target_size[1]}."
        )
    elif target_size is not None:
        try:
            _resize_image(destination, target_size)
        except _PATH_ACCESS_ERRORS as exc:
            try:
                destination.unlink(missing_ok=True)
            except _PATH_ACCESS_ERRORS:
                pass
            shutil.rmtree(work_dir, ignore_errors=True)
            raise TopazError(f"Topaz output resize failed: {exc}") from exc
        produced_size = _image_size(destination)
    shutil.rmtree(work_dir, ignore_errors=True)

    if input_size is not None and produced_size is not None:
        logs.append(f"Topaz dimensions: input={input_size[0]}x{input_size[1]}, produced={produced_size[0]}x{produced_size[1]}, target={target_size}")
    if completed.stdout:
        logs.append(completed.stdout)
    if completed.stderr:
        logs.append(completed.stderr)
    return {
        "output_path": str(destination),
        "logs": logs,
        "source_size": _size_dict(input_size),
        "output_size": _size_dict(produced_size),
    }


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except _PATH_ACCESS_ERRORS:
        return None


def _size_dict(size: tuple[int, int] | None) -> dict[str, int] | None:
    if size is None:
        return None
    return {"width": size[0], "height": size[1]}


def _resize_image(path: Path, target_size: tuple[int, int]) -> None:
    width, height = target_size
    if width <= 0 or height <= 0:
        return
    with Image.open(path) as image:
        if image.size == target_size:
            return
        image.resize(target_size, Image.Resampling.LANCZOS).save(path)
