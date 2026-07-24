from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageChops, ImageStat

from app.services.xyz_grid import XYZGridAxis, XYZGridCell, build_xyz_contact_sheet


@dataclass(frozen=True)
class RemasterComparisonResult:
    contact_sheet_path: str
    report_path: str
    metrics: dict[str, float]
    warnings: list[str]


def comparison_disabled_result() -> RemasterComparisonResult:
    return RemasterComparisonResult("", "", {}, [])


def build_remaster_comparison(
    source_path: Path,
    final_path: Path,
    output_dir: Path,
    metadata: dict,
) -> RemasterComparisonResult:
    token = str(metadata.get("remaster_id") or uuid4().hex[:12])
    sheet_path = output_dir / f"remaster-comparison-{token}.png"
    report_path = output_dir / f"remaster-comparison-{token}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        source_info = _image_info(source_path)
        final_info = _image_info(final_path)
        metrics = _difference_metrics(source_path, final_path)
        sheet = build_xyz_contact_sheet(
            [
                XYZGridCell(labels={"version": "Original"}, image_path=source_path),
                XYZGridCell(labels={"version": "Remaster"}, image_path=final_path),
            ],
            [XYZGridAxis(key="version", label="Version", values=("Original", "Remaster"))],
            margin=8,
        )
        sheet.save(sheet_path, format="PNG")
        report = {
            "source": source_info,
            "remaster": final_info,
            "metrics": metrics,
            "snapshot": metadata,
            "stage_sizes": metadata.get("stage_sizes", {}),
            "job_ids": metadata.get("job_ids", []),
            "warnings": [],
            "manual_review": {},
            "contact_sheet_path": str(sheet_path),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return RemasterComparisonResult(str(sheet_path), str(report_path), metrics, [])
    except (OSError, ValueError, TypeError):
        _unlink_owned(sheet_path)
        _unlink_owned(report_path)
        return RemasterComparisonResult("", "", {}, ["comparison_artifact_failed"])


def refresh_remaster_comparison_final_info(report_path: Path, final_path: Path) -> bool:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            return False
        report["remaster"] = _image_info(final_path)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _image_info(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "size": {"width": width, "height": height},
    }


def _difference_metrics(source_path: Path, final_path: Path) -> dict[str, float]:
    with Image.open(final_path) as final_image:
        final = final_image.convert("RGB")
    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB").resize(final.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(source, final)
    stat = ImageStat.Stat(difference)
    normalized_delta = sum(stat.mean) / (len(stat.mean) * 255.0)
    normalized_rms = sum(stat.rms) / (len(stat.rms) * 255.0)
    return {
        "normalized_delta": round(normalized_delta, 8),
        "normalized_rms": round(normalized_rms, 8),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_owned(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
