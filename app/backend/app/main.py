from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.advanced_styles import router as advanced_styles_router
from app.api.assets import router as assets_router
from app.api.body_review import router as body_review_router
from app.api.eagle import router as eagle_router
from app.api.external_models import router as external_models_router
from app.api.extensions import router as extensions_router
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
from app.api.prompt_comparisons import router as prompt_comparisons_router
from app.api.prompt_workshop import router as prompt_workshop_router
from app.api.prompt_materials import router as prompt_materials_router
from app.api.image_conditions import router as image_conditions_router
from app.api.enhancement_comparison import router as enhancement_comparison_router
from app.api.series_consistency import router as series_consistency_router
from app.api.lora_dataset_review import router as lora_dataset_review_router
from app.api.lora_practical_tests import router as lora_practical_tests_router
from app.api.presets import router as presets_router
from app.api.runtime import router as runtime_router
from app.api.resources import router as resources_router
from app.api.resource_advisory import router as resource_advisory_router
from app.api.settings import router as settings_router
from app.api.startup import router as startup_router
from app.api.topaz import configure_topaz_operation_coordinator, router as topaz_router
from app.api.video import configure_video_operation_coordinator, router as video_router
from app.api.video_upscale import (
    configure_video_upscale_operation_coordinator,
    router as video_upscale_router,
)
from app.api.video_sequences import (
    configure_video_sequence_operation_coordinator,
    router as video_sequences_router,
)
from app.api.video_setup import router as video_setup_router
from app.api.music_setup import router as music_setup_router
from app.api.music import configure_music_operation_coordinator, router as music_router
from app.api.workflow_mapping import router as workflow_mapping_router
from app.core.config import AppConfig
from app.services.extension_installer import configure_extension_operation_coordinator
from app.services.extension_catalog import ExtensionCatalogService
from app.services.extension_loader import ExtensionLoader, clear_loaded_extension_modules
from app.services.extension_repository import ExtensionRepository
from app.services.extension_startup import ExtensionStartupService
from app.services.resource_advisor import ResourceCapabilityRegistry
from app.services.reference_mutation import reference_mutation_coordinator
from app.services.video_runtime_adapters import default_video_adapters
from app.services.music_runtime_adapters import default_music_adapters
from app.services.history import HistoryRepository
from app.services.image_check_recovery import recover_image_publications


MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
EDIT_ASSET_CLEANUP_PATH = "/api/assets/lifecycle/cleanup"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    coordinator = get_operation_coordinator()
    config = getattr(_app.state, "config", AppConfig())
    configure_generation_operation_coordinator(coordinator)
    configure_merge_operation_coordinator(coordinator)
    configure_topaz_operation_coordinator(coordinator)
    configure_lora_operation_coordinator(coordinator)
    configure_download_operation_coordinator(coordinator, config=config)
    video_service = configure_video_operation_coordinator(coordinator, config=config)
    configure_video_upscale_operation_coordinator(
        coordinator,
        video_service=video_service,
        config=config,
    )
    sequence_service = configure_video_sequence_operation_coordinator(
        coordinator,
        config=config,
    )
    configure_music_operation_coordinator(coordinator, config=config)
    configure_extension_operation_coordinator(
        coordinator,
        extension_root=config.extension_root,
    )
    coordinator.recover_startup()
    recover_image_publications(HistoryRepository(config.data_dir / 'asteria.db'))
    if video_service is not None:
        video_service.cleanup_working_artifacts()
    sequence_service.sequences.interrupt_active_sequences()
    coordinator.start()
    start_discord_runtime()
    if config.auto_bootstrap_models:
        start_model_bootstrap()
    try:
        yield
    finally:
        stop_discord_runtime()
        coordinator.shutdown(wait=False)


def create_app(*, config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    app = FastAPI(title="Asteria API", lifespan=lifespan)
    app.state.config = config
    app.state.resource_capability_registry = ResourceCapabilityRegistry(
        tuple(
            (lambda adapter=adapter: adapter.capabilities())
            for adapter in (*default_video_adapters(), *default_music_adapters())
        )
    )

    @app.exception_handler(RequestValidationError)
    async def sanitize_extension_validation_error(
        request: Request,
        error: RequestValidationError,
    ):
        if not request.url.path.startswith("/api/extensions"):
            return await request_validation_exception_handler(request, error)
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "extension_request_invalid"}},
        )

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
    app.include_router(body_review_router)
    app.include_router(filesystem_router)
    app.include_router(image_metadata_router)
    app.include_router(models_router)
    app.include_router(operations_router)
    app.include_router(runtime_router)
    app.include_router(resources_router)
    app.include_router(resource_advisory_router)
    app.include_router(settings_router)
    app.include_router(startup_router)
    app.include_router(prompt_assist_router)
    app.include_router(prompt_presets_router)
    app.include_router(prompt_comparisons_router)
    app.include_router(prompt_workshop_router)
    app.include_router(prompt_materials_router)
    app.include_router(image_conditions_router)
    app.include_router(enhancement_comparison_router)
    app.include_router(series_consistency_router)
    app.include_router(lora_dataset_review_router)
    app.include_router(lora_practical_tests_router)
    app.include_router(presets_router)
    app.include_router(topaz_router)
    app.include_router(eagle_router)
    app.include_router(external_models_router)
    app.include_router(extensions_router)
    app.include_router(downloads_router)
    app.include_router(discord_router)
    app.include_router(diagnostics_router)
    app.include_router(video_router)
    app.include_router(video_upscale_router)
    app.include_router(video_sequences_router)
    app.include_router(video_setup_router)
    app.include_router(music_setup_router)
    app.include_router(music_router)
    app.include_router(workflow_mapping_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "app": "asteria",
            "target_mode": AppConfig().target_mode,
        }

    operation_coordinator = get_operation_coordinator()
    app.state.operation_coordinator = operation_coordinator
    operation_coordinator.clear_executor_groups(prefix="extension:")
    clear_loaded_extension_modules()
    extension_repository = ExtensionRepository(extension_root=config.extension_root)
    app.state.extension_catalog = ExtensionCatalogService(
        config.extension_catalog_cache_dir,
        catalog_url=config.extension_catalog_url,
        asteria_version=extension_repository.asteria_version,
    )
    extension_loader = ExtensionLoader(
        extension_repository,
        operation_coordinator=operation_coordinator,
    )
    app.state.extension_startup = ExtensionStartupService(
        extension_loader,
        journal_path=config.extension_root / "startup-journal.json",
    ).initialize(app, safe_mode=config.disable_extensions)

    return app


app = create_app()
