from pathlib import Path
import os

from pydantic import BaseModel, Field


_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _default_root() -> Path:
    return Path(os.environ.get("ASTERIA_ROOT_DIR", _SOURCE_ROOT))


def _optional_path_environment(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


class AppConfig(BaseModel):
    root_dir: Path = Field(default_factory=_default_root)
    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_DATA_DIR",
                _default_root() / "data",
            )
        )
    )
    secret_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_SECRET_DIR",
                Path(os.environ.get("ASTERIA_DATA_DIR", _default_root() / "data"))
                / "secrets",
            )
        )
    )
    schema_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_SCHEMA_DIR",
                _default_root() / "schemas",
            )
        )
    )
    mapping_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MAPPING_DIR",
                _default_root() / "mappings",
            )
        )
    )
    model_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MODEL_DIR",
                _default_root() / "models",
            )
        )
    )
    model_dependencies_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MODEL_DEPENDENCIES_PATH",
                _default_root() / "model-dependencies.yaml",
            )
        )
    )
    runtime_dependencies_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_RUNTIME_DEPENDENCIES_PATH",
                _default_root() / "runtime-dependencies.yaml",
            )
        )
    )
    release_manifest_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_RELEASE_MANIFEST_PATH",
                _default_root() / "release-manifest.yaml",
            )
        )
    )
    model_profiles_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MODEL_PROFILES_PATH",
                _default_root() / "model-profiles.yaml",
            )
        )
    )
    model_selections_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MODEL_SELECTIONS_PATH",
                _default_root() / "data" / "model-selections.json",
            )
        )
    )
    merge_preset_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_MERGE_PRESET_DIR",
                Path(os.environ.get("ASTERIA_DATA_DIR", _default_root() / "data")) / "config" / "merge-presets",
            )
        )
    )
    wildcard_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_WILDCARD_DIR",
                _default_root() / "wildcards",
            )
        )
    )
    generation_runtime: str = Field(default_factory=lambda: os.environ.get("ASTERIA_GENERATION_RUNTIME", "asteria"))
    target_mode: str = Field(
        default_factory=lambda: (
            "packaged"
            if os.environ.get("ASTERIA_TARGET_MODE") == "packaged"
            else "development"
        )
    )
    auto_bootstrap_models: bool = Field(default_factory=lambda: os.environ.get("ASTERIA_AUTO_BOOTSTRAP_MODELS", "1") not in {"0", "false", "False"})
    disable_extensions: bool = Field(
        default_factory=lambda: os.environ.get("ASTERIA_DISABLE_EXTENSIONS", "0") == "1"
    )
    extension_catalog_url: str = Field(
        default_factory=lambda: os.environ.get(
            "ASTERIA_EXTENSION_CATALOG_URL",
            "https://raw.githubusercontent.com/"
            "w-tea/Asteria-extensions-catalog/main/catalog-v1.json",
        )
    )
    engine_command: str = Field(default_factory=lambda: os.environ.get("ASTERIA_ENGINE_COMMAND", ""))
    eagle_base_url: str = Field(default_factory=lambda: os.environ.get("ASTERIA_EAGLE_BASE_URL", "http://localhost:41595"))
    topaz_executable: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_TOPAZ_EXECUTABLE",
                r"C:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe",
            )
        )
    )
    runtime_base_url: str = Field(default_factory=lambda: os.environ.get("ASTERIA_RUNTIME_BASE_URL", os.environ.get("ASTERIA_COMFY_BASE_URL", "http://127.0.0.1:8188")))
    comfy_dir: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_COMFY_DIR",
                _default_root() / "runtime" / "ComfyUI",
            )
        )
    )
    comfy_repo_url: str = Field(default_factory=lambda: os.environ.get("ASTERIA_COMFY_REPO_URL", "https://github.com/comfyanonymous/ComfyUI.git"))
    comfy_repo_ref: str = Field(default_factory=lambda: os.environ.get("ASTERIA_COMFY_REPO_REF", ""))
    comfy_python_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_COMFY_PYTHON",
                _default_root() / ".venv" / "Scripts" / "python.exe",
            )
        )
    )
    comfy_args: list[str] = Field(
        default_factory=lambda: os.environ.get(
            "ASTERIA_COMFY_ARGS",
            "--listen 127.0.0.1 --port 8188 --preview-method auto",
        ).split()
    )
    comfy_pid_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get(
                "ASTERIA_COMFY_PID_PATH",
                _default_root() / "data" / "comfyui.pid",
            )
        )
    )
    comfy_output_dir: Path | None = Field(
        default_factory=lambda: _optional_path_environment("ASTERIA_COMFY_OUTPUT_DIR")
    )
    remaster_max_final_pixels: int = Field(
        default_factory=lambda: int(os.environ.get("ASTERIA_REMASTER_MAX_FINAL_PIXELS", "16777216")),
        ge=1,
    )
    remaster_max_anchor_pixels: int = Field(
        default_factory=lambda: int(os.environ.get("ASTERIA_REMASTER_MAX_ANCHOR_PIXELS", "33554432")),
        ge=1,
    )

    @property
    def extension_root(self) -> Path:
        return self.data_dir / "extensions"

    @property
    def extension_catalog_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "extensions"

    @property
    def comfy_base_url(self) -> str:
        return self.runtime_base_url
