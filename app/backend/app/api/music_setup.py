from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from app.core.config import AppConfig
from app.services.managed_artifacts import ManagedArtifactSpec
from app.services.music_local_setup import (
    MusicExecutionProfile,
    MusicLocalSetupService,
    MusicModelProfileRepository,
    MusicSetupError,
)
from app.services.operation_repository import OperationNotFoundError
from app.services.release_manifest import ReleaseManifestRepository


router = APIRouter(prefix="/api/music/setup", tags=["music-setup"])


class StartLocalMusic3SetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_profile: MusicExecutionProfile = "layer_streaming"
    license_version: str
    agree_and_download: bool
    idempotency_key: str | None = None


class SelectLocalMusic3ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_profile: MusicExecutionProfile


def get_music_setup_service(config: AppConfig) -> MusicLocalSetupService:
    profile = MusicModelProfileRepository(
        config.root_dir / "profiles" / "music" / "minimax-music3-local.v1.json"
    ).load()
    manifest = ReleaseManifestRepository(config.release_manifest_path).load()
    runtime = next(
        (artifact for artifact in manifest.artifacts if artifact.id == profile.runtime.component_id),
        None,
    )
    if runtime is None or runtime.version != profile.runtime.managed_version:
        raise MusicSetupError("music_setup_runtime_manifest_mismatch")
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
    return MusicLocalSetupService(
        profile=profile,
        runtime_root=runtime_root,
        data_dir=config.data_dir,
        runtime_spec=runtime_spec,
    )


def _start(request: Request, body: StartLocalMusic3SetupRequest, *, repair: bool = False) -> dict:
    try:
        result = get_music_setup_service(request.app.state.config).start(
            body.execution_profile,
            license_version=body.license_version,
            agree_and_download=body.agree_and_download,
            coordinator=request.app.state.operation_coordinator,
            idempotency_key=body.idempotency_key,
            repair=repair,
        )
    except MusicSetupError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
    return result.model_dump(mode="json")


@router.get("/local-music3")
def get_local_music3_setup(
    request: Request,
    execution_profile: MusicExecutionProfile = "layer_streaming",
) -> dict:
    try:
        return get_music_setup_service(request.app.state.config).plan(execution_profile).model_dump(mode="json")
    except MusicSetupError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc


@router.post("/local-music3")
def start_local_music3_setup(request: Request, body: StartLocalMusic3SetupRequest) -> dict:
    return _start(request, body)


@router.patch("/local-music3")
def select_local_music3_profile(
    request: Request,
    body: SelectLocalMusic3ProfileRequest,
) -> dict:
    try:
        plan = get_music_setup_service(request.app.state.config).select_execution_profile(
            body.execution_profile
        )
    except MusicSetupError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from exc
    return plan.model_dump(mode="json")


@router.post("/local-music3/resume")
def resume_local_music3_setup(request: Request, body: StartLocalMusic3SetupRequest) -> dict:
    return _start(request, body)


@router.post("/local-music3/repair")
def repair_local_music3_setup(request: Request, body: StartLocalMusic3SetupRequest) -> dict:
    return _start(request, body, repair=True)


@router.post("/local-music3/operations/{operation_id}/cancel")
def cancel_local_music3_setup(operation_id: str, request: Request) -> dict:
    coordinator = request.app.state.operation_coordinator
    service = get_music_setup_service(request.app.state.config)
    try:
        operation = coordinator.get(operation_id)
    except OperationNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"}) from exc
    component_id = operation.request_snapshot.get("component_id")
    if (
        operation.kind != "managed.download"
        or component_id not in service.component_ids()
        or operation.request_snapshot.get("setup_owner")
        != "music:minimax-music3-local"
    ):
        raise HTTPException(status_code=404, detail={"code": "operation_not_found"})
    cancelled = coordinator.request_cancel(operation_id)
    return cancelled.model_dump(mode="json", exclude={"request_snapshot"})
