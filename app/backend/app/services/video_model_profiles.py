from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.managed_artifacts import ManagedArtifactSpec


FeaturePack = Literal["base", "reference", "all"]


class VideoProductionRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_bounds(self) -> "VideoProductionRange":
        if self.minimum > self.maximum:
            raise ValueError("video production range minimum cannot exceed maximum")
        return self


class VideoRuntimeRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: Literal["comfyui-core"]
    minimum_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    managed_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")


class VideoLicenseProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["minimax-h3-community"]
    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    url: str

    @model_validator(mode="after")
    def validate_url(self) -> "VideoLicenseProfile":
        _require_https(self.url, "video license")
        return self


class VideoModelAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]+$")
    label: str = Field(min_length=1)
    pack: Literal["shared", "base", "reference"]
    relative_path: str
    source_url: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    provenance_url: str
    size_bytes: int = Field(gt=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source_and_path(self) -> "VideoModelAsset":
        _safe_relative_path(self.relative_path, "video model path")
        _require_https(self.source_url, "video model source")
        _require_https(self.provenance_url, "video model provenance")
        if f"/resolve/{self.revision}/" not in self.source_url:
            raise ValueError("video model source must use an immutable revision")
        if f"/blob/{self.revision}/" not in self.provenance_url:
            raise ValueError("video model provenance must use an immutable revision")
        if not self.relative_path.casefold().endswith(".safetensors"):
            raise ValueError("video model path must target a safetensors file")
        return self

    @property
    def filename(self) -> str:
        return PurePosixPath(self.relative_path.replace("\\", "/")).name

    def managed_spec(self, runtime_root: Path) -> ManagedArtifactSpec:
        return ManagedArtifactSpec(
            manifest_version=1,
            component_id=self.id,
            source_url=self.source_url,
            destination=runtime_root / Path(self.relative_path),
            expected_size=self.size_bytes,
            checksum_sha256=self.checksum_sha256,
            kind="file",
            require_https=True,
            max_size=self.size_bytes,
        )


class VideoProductionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_steps: int = Field(ge=1)
    steps: VideoProductionRange
    samplers: tuple[str, ...]
    default_sampler: str
    schedulers: tuple[str, ...]
    default_scheduler: str
    default_video_shift: float = Field(ge=0, allow_inf_nan=False)
    video_shift: VideoProductionRange
    default_audio_shift: float = Field(ge=0, allow_inf_nan=False)
    audio_shift: VideoProductionRange
    lora_supported: bool
    lora_weight: VideoProductionRange | None = None
    default_batch_count: int = Field(default=1, ge=1)
    batch_count: VideoProductionRange

    @model_validator(mode="after")
    def validate_defaults(self) -> "VideoProductionProfile":
        if not self.samplers or len(self.samplers) != len(set(self.samplers)):
            raise ValueError("video profile samplers must be non-empty and unique")
        if not self.schedulers or len(self.schedulers) != len(set(self.schedulers)):
            raise ValueError("video profile schedulers must be non-empty and unique")
        if self.default_sampler not in self.samplers:
            raise ValueError("video profile default sampler must be supported")
        if self.default_scheduler not in self.schedulers:
            raise ValueError("video profile default scheduler must be supported")
        if self.lora_supported and self.lora_weight is None:
            raise ValueError("video profile LoRA support requires a weight range")
        if not self.lora_supported and self.lora_weight is not None:
            raise ValueError("video profile without LoRA support cannot carry a weight range")
        return self


class VideoModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["asteria.video.model-profile.v1"]
    profile_id: Literal["minimax-h3-local"]
    runtime_profile_id: Literal["minimax-h3"]
    version: Literal[1]
    provider: Literal["local_h3"]
    runtime: VideoRuntimeRequirement
    license: VideoLicenseProfile
    production: VideoProductionProfile
    assets: tuple[VideoModelAsset, ...]
    workflows: dict[
        Literal[
            "text",
            "image",
            "first_last",
            "reference",
            "sequence_initial",
            "sequence_reference",
            "sequence_motion_context",
        ],
        str,
    ]

    @model_validator(mode="after")
    def validate_identity(self) -> "VideoModelProfile":
        ids = [asset.id.casefold() for asset in self.assets]
        paths = [asset.relative_path.replace("\\", "/").casefold() for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("video profile has duplicate asset ids")
        if len(paths) != len(set(paths)):
            raise ValueError("video profile has duplicate asset paths")
        if set(self.workflows) != {
            "text",
            "image",
            "first_last",
            "reference",
            "sequence_initial",
            "sequence_reference",
            "sequence_motion_context",
        }:
            raise ValueError("video profile workflow coverage is incomplete")
        if not any(asset.pack == "base" for asset in self.assets):
            raise ValueError("video profile base pack is missing")
        if not any(asset.pack == "reference" for asset in self.assets):
            raise ValueError("video profile reference pack is missing")
        if not any(asset.pack == "shared" for asset in self.assets):
            raise ValueError("video profile shared assets are missing")
        return self

    def managed_specs_for_pack(
        self,
        pack: FeaturePack,
        runtime_root: Path,
    ) -> tuple[ManagedArtifactSpec, ...]:
        if pack not in {"base", "reference", "all"}:
            raise ValueError("video setup feature pack is invalid")
        selected = {
            "base": {"shared", "base"},
            "reference": {"shared", "reference"},
            "all": {"shared", "base", "reference"},
        }[pack]
        return tuple(
            asset.managed_spec(runtime_root)
            for asset in self.assets
            if asset.pack in selected
        )


class VideoModelProfileRepository:
    def __init__(self, path: Path, *, workflow_root: Path) -> None:
        self.path = path
        self.workflow_root = workflow_root

    def load(self) -> VideoModelProfile:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            profile = VideoModelProfile.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"video model profile is invalid: {exc}") from exc
        for relative in profile.workflows.values():
            path = _safe_relative_path(relative, "video workflow path")
            target = self.workflow_root / path
            if not target.is_file():
                raise ValueError("video workflow path is missing")
        return profile


@lru_cache(maxsize=1)
def load_default_video_model_profile() -> VideoModelProfile:
    project_root = Path(__file__).resolve().parents[3]
    return VideoModelProfileRepository(
        project_root / "profiles" / "video" / "minimax-h3-local.v1.json",
        workflow_root=project_root / "workflows" / "video" / "minimax-h3",
    ).load()


def _require_https(value: str, label: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{label} must use HTTPS without credentials")


def _safe_relative_path(value: str, label: str) -> Path:
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or ".." in windows.parts
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
        or any(_unsafe_windows_part(part) for part in posix.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return Path(*posix.parts)


def _unsafe_windows_part(part: str) -> bool:
    if ":" in part or part.endswith((" ", ".")):
        return True
    stem = part.split(".", 1)[0].casefold()
    return stem in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
