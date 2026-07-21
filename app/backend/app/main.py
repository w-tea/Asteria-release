from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.advanced_styles import router as advanced_styles_router
from app.api.assets import router as assets_router
from app.api.eagle import router as eagle_router
from app.api.filesystem import router as filesystem_router
from app.api.generation import router as generation_router
from app.api.remaster import router as remaster_router
from app.api.lora_training import router as lora_training_router
from app.api.merge import router as merge_router
from app.api.models import router as models_router, start_model_bootstrap
from app.api.prompt_assist import router as prompt_assist_router
from app.api.prompt_presets import router as prompt_presets_router
from app.api.presets import router as presets_router
from app.api.runtime import router as runtime_router
from app.api.settings import router as settings_router
from app.api.startup import router as startup_router
from app.api.topaz import router as topaz_router
from app.api.workflow_mapping import router as workflow_mapping_router
from app.core.config import AppConfig


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    if AppConfig().auto_bootstrap_models:
        start_model_bootstrap()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Asteria API", lifespan=lifespan)
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
    app.include_router(assets_router)
    app.include_router(filesystem_router)
    app.include_router(models_router)
    app.include_router(runtime_router)
    app.include_router(settings_router)
    app.include_router(startup_router)
    app.include_router(prompt_assist_router)
    app.include_router(prompt_presets_router)
    app.include_router(presets_router)
    app.include_router(topaz_router)
    app.include_router(eagle_router)
    app.include_router(workflow_mapping_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "app": "asteria"}

    return app


app = create_app()
