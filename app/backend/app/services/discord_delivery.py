from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image
from pydantic import BaseModel, ConfigDict


class DiscordDeliveryError(RuntimeError):
    pass


class DiscordDeliveryArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    source_path: Path
    resized: bool
    warning_codes: tuple[str, ...] = ()


_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


class DiscordDeliveryService:
    def __init__(
        self,
        staging_root: Path,
        *,
        max_upload_bytes: int = 8 * 1024 * 1024,
        max_preview_dimension: int = 2048,
    ) -> None:
        if max_upload_bytes < 1024 or max_preview_dimension < 64:
            raise ValueError("discord_delivery_bounds_invalid")
        self.staging_root = staging_root
        self.max_upload_bytes = max_upload_bytes
        self.max_preview_dimension = max_preview_dimension

    def prepare(
        self,
        source_path: Path,
        *,
        operation_id: str,
        output_index: int,
    ) -> DiscordDeliveryArtifact:
        safe_id = _SAFE_ID.sub("-", operation_id).strip("-_")[:12] or "output"
        if output_index < 0:
            raise ValueError("discord_delivery_output_index_invalid")
        self.staging_root.mkdir(parents=True, exist_ok=True)
        destination = self.staging_root / f"asteria-{safe_id}-{output_index}.png"
        try:
            with Image.open(source_path) as source:
                source.load()
                pixel_image = Image.frombytes(
                    source.mode,
                    source.size,
                    source.tobytes(),
                )
                if pixel_image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    pixel_image = pixel_image.convert("RGBA")
            self._save_png_atomic(pixel_image, destination, optimize=False)
            resized = False
            if destination.stat().st_size > self.max_upload_bytes:
                pixel_image = self._bounded_preview(pixel_image)
                self._save_png_atomic(pixel_image, destination, optimize=True)
                resized = True
                while destination.stat().st_size > self.max_upload_bytes and min(pixel_image.size) > 64:
                    ratio = max(
                        0.5,
                        min(
                            0.9,
                            math.sqrt(
                                self.max_upload_bytes
                                / max(1, destination.stat().st_size)
                            )
                            * 0.95,
                        ),
                    )
                    next_size = (
                        max(64, int(pixel_image.width * ratio)),
                        max(64, int(pixel_image.height * ratio)),
                    )
                    if next_size == pixel_image.size:
                        break
                    pixel_image = pixel_image.resize(next_size, Image.Resampling.LANCZOS)
                    self._save_png_atomic(pixel_image, destination, optimize=True)
                if destination.stat().st_size > self.max_upload_bytes:
                    quantized = pixel_image.convert("RGB").quantize(colors=128)
                    self._save_png_atomic(quantized, destination, optimize=True)
                if destination.stat().st_size > self.max_upload_bytes:
                    raise DiscordDeliveryError("discord_delivery_upload_limit")
        except DiscordDeliveryError:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise DiscordDeliveryError("discord_delivery_prepare_failed") from exc
        return DiscordDeliveryArtifact(
            path=destination,
            source_path=source_path,
            resized=resized,
            warning_codes=("discord_delivery_preview_resized",) if resized else (),
        )

    def _bounded_preview(self, image: Image.Image) -> Image.Image:
        if max(image.size) <= self.max_preview_dimension:
            return image.copy()
        ratio = self.max_preview_dimension / max(image.size)
        size = (
            max(1, round(image.width * ratio)),
            max(1, round(image.height * ratio)),
        )
        return image.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _save_png_atomic(image: Image.Image, destination: Path, *, optimize: bool) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                prefix=f".{destination.stem}.",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            image.save(temporary_path, format="PNG", optimize=optimize)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
