from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.schemas.video_upscale import VideoUpscaleRequest, VideoUpscaleWorkBundle


class VideoUpscaleAdapterError(ValueError):
    def __init__(self, code: str, *, retryable: bool = False):
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class VideoUpscaleExecutionInput:
    request: VideoUpscaleRequest
    bundle: VideoUpscaleWorkBundle
    bundle_path: Path
    latent_path: Path
    output_root: Path
    target_width: int
    target_height: int
    resolved_seed: int


@dataclass(frozen=True, slots=True)
class VideoUpscaleExecutionResult:
    runtime_job_id: str
    workflow_id: str
    output_paths: tuple[Path, ...]
    target_width: int
    target_height: int


class VideoUpscaleAdapter(Protocol):
    id: str
    revision: str

    def execute(
        self,
        execution_input: VideoUpscaleExecutionInput,
        *,
        operation_id: str,
        progress_callback: Callable[[int, int], None],
        cancel_requested: Callable[[], bool],
        runtime_job_callback: Callable[[str], None],
    ) -> VideoUpscaleExecutionResult: ...
