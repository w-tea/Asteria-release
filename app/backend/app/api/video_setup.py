from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from app.core.config import AppConfig
from app.services.managed_artifacts import ManagedArtifactSpec
from app.services.release_manifest import ReleaseManifestRepository
from app.services.video_model_profiles import VideoModelProfileRepository
from app.services.video_setup import VideoSetupError, VideoSetupService


router = APIRouter(prefix="/api/video/setup", tags=["video-setup"])


class StartLocalH3SetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_pack: Literal["base", "reference", "all"] = "base"
    license_version: str
    agree_and_download: bool
    idempotency_key: str | None = None


def get_video_setup_service(config: AppConfig) -> VideoSetupService:
    profile = VideoModelProfileRepository(
        config.root_dir / "profiles" / "video" / "minimax-h3-local.v1.json",
        workflow_root=config.root_dir / "workflows" / "video" / "minimax-h3",
    ).load()
    manifest = ReleaseManifestRepository(config.release_manifest_path).load()
    runtime = next(
        (artifact for artifact in manifest.artifacts if artifact.id == profile.runtime.component_id),
        None,
    )
    if runtime is None or runtime.version != profile.runtime.managed_version:
        raise VideoSetupError("video_setup_runtime_manifest_mismatch")
    runtime_root = (config.root_dir / runtime.relative_path).absolute()
    runtime_spec = ManagedArtifactSpec(
        manifest_version=manifest.version,
        component_id=runtime.id,
        source_url=runtime.url,
        destination=runtime_root,
        expected_size=runtime.size_bytes,
        checksum_sha256=runtime.checksum_sha256,
        kind="zip" if runtime.kind == "archive" else "file",
        strip_single_root=runtime.kind == "archive",
        require_https=True,
        max_size=runtime.size_bytes,
    )
    return VideoSetupService(
        profile=profile,
        runtime_root=runtime_root,
        data_dir=config.data_dir,
        runtime_spec=runtime_spec,
    )


@router.get("/local-h3")
def get_local_h3_setup(
    request: Request,
    feature_pack: Literal["base", "reference", "all"] = "base",
) -> dict:
    try:
        return get_video_setup_service(request.app.state.config).plan(feature_pack).model_dump(mode="json")
    except VideoSetupError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc


@router.post("/local-h3")
def start_local_h3_setup(request: Request, body: StartLocalH3SetupRequest) -> dict:
    try:
        result = get_video_setup_service(request.app.state.config).start(
            body.feature_pack,
            license_version=body.license_version,
            agree_and_download=body.agree_and_download,
            coordinator=request.app.state.operation_coordinator,
            idempotency_key=body.idempotency_key,
        )
    except VideoSetupError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
    return result.model_dump(mode="json")
