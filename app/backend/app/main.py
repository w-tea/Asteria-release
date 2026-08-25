from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api.advanced_styles import router as advanced_styles_router
from app.api.assets import router as assets_router
from app.api.eagle import router as eagle_router
from app.api.external_models import router as external_models_router
from app.api.downloads import configure_download_operation_coordinator, router as downloads_router
from app.api.discord import (
    router as discord_router,
    start_discord_runtime,
    stop_discord_runtime,
)
from app.api.diagnostics import router as diagnostics_router
from app.api.filesystem import router as filesystem_router
from app.api.image_metadata import router as image_metadata_router
from app.api.generation import configure_generation_operation_coordinator, router as generation_router
from app.api.remaster import router as remaster_router
from app.api.lora_training import configure_lora_operation_coordinator, router as lora_training_router
from app.api.merge import configure_merge_operation_coordinator, router as merge_router
from app.api.maintenance import router as maintenance_router
from app.api.models import router as models_router, start_model_bootstrap
from app.api.operations import get_operation_coordinator, router as operations_router
from app.api.prompt_assist import router as prompt_assist_router
from app.api.prompt_presets import router as prompt_presets_router
from app.api.presets import router as presets_router
from app.api.runtime import router as runtime_router
from app.api.resources import router as resources_router
from app.api.settings import router as settings_router
from app.api.startup import router as startup_router
from app.api.topaz import configure_topaz_operation_coordinator, router as topaz_router
from app.api.workflow_mapping import router as workflow_mapping_router
from app.core.config import AppConfig
from app.services.reference_mutation import reference_mutation_coordinator


MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
EDIT_ASSET_CLEANUP_PATH = "/api/assets/lifecycle/cleanup"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    coordinator = get_operation_coordinator()
    configure_generation_operation_coordinator(coordinator)
    configure_merge_operation_coordinator(coordinator)
    configure_topaz_operation_coordinator(coordinator)
    configure_lora_operation_coordinator(coordinator)
    configure_download_operation_coordinator(coordinator)
    coordinator.recover_startup()
    coordinator.start()
    start_discord_runtime()
    if AppConfig().auto_bootstrap_models:
        start_model_bootstrap()
    try:
        yield
    finally:
        stop_discord_runtime()
        coordinator.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(title="Asteria API", lifespan=lifespan)

    @app.middleware("http")
    async def coordinate_reference_mutations(request: Request, call_next):
        if request.method not in MUTATING_METHODS or request.url.path == EDIT_ASSET_CLEANUP_PATH:
            return await call_next(request)
        lease = await run_in_threadpool(reference_mutation_coordinator.acquire_publisher)
        try:
            return await call_next(request)
        finally:
            lease.release()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(advanced_styles_router)
    app.include_router(generation_router)
    app.include_router(remaster_router)
    app.include_router(lora_training_router)
    app.include_router(merge_router)
    app.include_router(maintenance_router)
    app.include_router(assets_router)
    app.include_router(filesystem_router)
    app.include_router(image_metadata_router)
    app.include_router(models_router)
    app.include_router(operations_router)
    app.include_router(runtime_router)
    app.include_router(resources_router)
    app.include_router(settings_router)
    app.include_router(startup_router)
    app.include_router(prompt_assist_router)
    app.include_router(prompt_presets_router)
    app.include_router(presets_router)
    app.include_router(topaz_router)
    app.include_router(eagle_router)
    app.include_router(external_models_router)
    app.include_router(downloads_router)
    app.include_router(discord_router)
    app.include_router(diagnostics_router)
    app.include_router(workflow_mapping_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "app": "asteria",
            "target_mode": AppConfig().target_mode,
        }

    return app


app = create_app()
