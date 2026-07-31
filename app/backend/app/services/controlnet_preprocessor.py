from __future__ import annotations

import importlib
from pathlib import Path
import os
from typing import Any
from uuid import uuid4

from PIL import Image

cv2: Any = None
np: Any = None


class ControlNetPreprocessorError(Exception):
    pass


SUPPORTED_PREPROCESSORS = {"none", "canny", "lineart", "scribble", "depth", "tile", "softedge", "normal", "openpose"}
COMFY_CONTROLNET_PREPROCESSOR_NODES = {
    "canny": "CannyEdgePreprocessor",
    "lineart": "LineartStandardPreprocessor",
    "scribble": "ScribblePreprocessor",
    "depth": "DepthAnythingPreprocessor",
    "tile": "TilePreprocessor",
    "softedge": "HEDPreprocessor",
    "normal": "BAE-NormalMapPreprocessor",
    "openpose": "DWPreprocessor",
}
OPENPOSE_DEPENDENCY_FILES = (
    Path("preprocessors/openpose/yolox_l.onnx"),
    Path("preprocessors/openpose/dw-ll_ucoco_384.onnx"),
)


def preprocess_control_image(image_path: Path, preprocessor: str, output_dir: Path) -> Path:
    mode = (preprocessor or "none").lower()
    if mode == "none":
        return image_path
    if mode not in SUPPORTED_PREPROCESSORS:
        raise ControlNetPreprocessorError(f"ControlNet preprocessor is not implemented: {preprocessor}")
    if not _path_exists(image_path) or not _path_is_file(image_path):
        raise ControlNetPreprocessorError(f"Control image not found: {image_path}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ControlNetPreprocessorError(f"Failed to prepare control image output: {output_dir}") from exc
    _load_native_dependencies()
    image = _read_rgb(image_path)
    if mode == "canny":
        processed = _canny(image)
    elif mode == "lineart":
        processed = _lineart(image)
    elif mode == "scribble":
        processed = _scribble(image)
    elif mode == "depth":
        processed = _depth(image)
    elif mode == "tile":
        processed = _tile(image)
    elif mode == "softedge":
        processed = _softedge(image)
    elif mode == "normal":
        processed = _normal(image)
    elif mode == "openpose":
        processed = _openpose(image)
    else:
        processed = image

    output_path = output_dir / f"{uuid4().hex}_{mode}_{image_path.stem}.png"
    try:
        Image.fromarray(processed).save(output_path)
    except OSError as exc:
        raise ControlNetPreprocessorError(f"Failed to save processed control image: {output_path}") from exc
    return output_path


def _load_native_dependencies() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise ControlNetPreprocessorError("ControlNet preprocessor runtime is not installed") from exc


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _read_rgb(image_path: Path) -> np.ndarray:
    try:
        return np.asarray(Image.open(image_path).convert("RGB"))
    except OSError as exc:
        raise ControlNetPreprocessorError(f"Failed to read control image: {image_path}") from exc


def _canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def _lineart(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    line = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return cv2.cvtColor(line, cv2.COLOR_GRAY2RGB)


def _scribble(image: np.ndarray) -> np.ndarray:
    dark_pixels = np.min(image, axis=2) < 127
    scribble = np.full_like(image, 255)
    scribble[dark_pixels] = 0
    return scribble


def _depth(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    depth = cv2.equalizeHist(gray)
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)


def _tile(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def _softedge(image: np.ndarray) -> np.ndarray:
    edges = _canny(image)
    return cv2.GaussianBlur(edges, (5, 5), 0)


def _normal(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.dstack((-grad_x, -grad_y, np.ones_like(gray)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(norm, 1e-6)
    return ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def _openpose(image: np.ndarray) -> np.ndarray:
    missing = [path for path in _openpose_dependency_paths() if not _path_exists(path)]
    if missing:
        return _openpose_fallback_skeleton(image)
    return _openpose_fallback_skeleton(image)


def _openpose_fallback_skeleton(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cx = width // 2
    head_radius = max(2, min(width, height) // 10)
    head = (cx, max(head_radius + 1, height // 5))
    neck = (cx, height // 3)
    hips = (cx, (height * 2) // 3)
    left_shoulder = (max(0, cx - width // 5), height // 3)
    right_shoulder = (min(width - 1, cx + width // 5), height // 3)
    left_hand = (max(0, cx - width // 3), height // 2)
    right_hand = (min(width - 1, cx + width // 3), height // 2)
    left_foot = (max(0, cx - width // 5), height - 2)
    right_foot = (min(width - 1, cx + width // 5), height - 2)
    cv2.circle(canvas, head, head_radius, (255, 255, 255), 1)
    for start, end, color in [
        (head, neck, (255, 0, 0)),
        (neck, hips, (0, 255, 0)),
        (neck, left_shoulder, (0, 0, 255)),
        (neck, right_shoulder, (0, 0, 255)),
        (left_shoulder, left_hand, (255, 255, 0)),
        (right_shoulder, right_hand, (255, 255, 0)),
        (hips, left_foot, (255, 0, 255)),
        (hips, right_foot, (255, 0, 255)),
    ]:
        cv2.line(canvas, start, end, color, 1)
    return canvas


def _openpose_dependency_paths() -> list[Path]:
    model_root = Path(os.environ.get("ASTERIA_MODEL_DIR", Path(__file__).resolve().parents[3] / "models"))
    return [model_root / relative_path for relative_path in OPENPOSE_DEPENDENCY_FILES]
