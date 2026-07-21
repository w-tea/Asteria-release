import hashlib
import json
from pathlib import Path
from uuid import uuid4

from app.schemas.image_edit import ImageEditArtifact, ImageEditLineage


def image_edit_manifest_path(image_path: Path) -> Path:
    return image_path.with_suffix(f"{image_path.suffix}.asteria.edit.json")


def image_edit_artifact(path: Path, data: bytes, width: int, height: int) -> ImageEditArtifact:
    return ImageEditArtifact(
        path=str(path.resolve()),
        filename=path.name,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
    )


def load_image_edit_lineage(image_path: Path) -> ImageEditLineage | None:
    manifest_path = image_edit_manifest_path(image_path)
    if not manifest_path.is_file():
        return None
    return ImageEditLineage.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))


def write_image_edit_lineage(image_path: Path, lineage: ImageEditLineage) -> Path:
    manifest_path = image_edit_manifest_path(image_path)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(lineage.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return manifest_path
