from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import subprocess
import re
import shlex
import shutil
import sys
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Literal
from uuid import uuid4
from urllib.parse import urlparse

import httpx
from PIL import ExifTags, Image, ImageFilter, ImageOps, ImageStat
from pydantic import BaseModel, Field

from app.services.image_metadata_text import metadata_value_to_text

TrainingProfile = Literal["sdxl", "anima"]
TrainingType = Literal["copy_lora", "character_lora", "style_lora"]
LoRATrainingRunStatus = Literal["planned", "blocked", "running", "completed", "failed", "cancelled"]
LoRATrainingCaptionAssistLanguage = Literal["en", "ja"]
LoRATrainingCaptionAssistStyle = Literal["tags", "natural"]
LoRATrainingCaptionAssistProviderMode = Literal["prompt_only", "openai_compatible", "openai_responses", "anthropic", "gemini"]
LoRATrainingCaptionOutputMode = Literal["caption_txt", "caption_txt_and_metadata_json", "metadata_json"]
LoRATrainingCaptionSourceContentKind = Literal["tags", "caption", "json", "anima_tagger_ron"]
LoRATrainingCaptionSourceToolStatus = Literal["ready", "missing"]
LoRATrainingCaptionSourceType = Literal[
    "metadata",
    "wd14_tagger",
    "vision_caption",
    "booru_api",
    "provider_draft",
    "manual_reference",
]
LoRATrainingSampleRating = Literal["good", "review", "poor"]
LoRACopyMachineMethod = Literal["direct_difference", "classic_copy_machine", "anima_experimental"]
LoRACopyMachineADDifTTimestepWindow = Literal["action_500_1000", "style_200_400"]
LoRACopyMachineADDifTSNRGamma = Literal["preset_5_0", "low_1_0"]
LoRACopyMachineStatus = Literal["draft", "ready", "blocked", "running", "completed", "failed"]
LoRACopyMachineTargetTransform = Literal[
    "grayscale",
    "high_contrast",
    "line_extract",
    "bold_line",
    "flat_palette",
    "posterize",
    "sharpen",
    "soft_blur",
    "sepia",
    "invert",
]
LoRACopyMachineAdapterStatus = Literal["ready", "missing"]
LoRACopyMachineTrainingRunStatus = Literal["planned", "blocked", "running", "completed", "failed"]
SAMPLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class LoRATrainingStorageError(Exception):
    """Raised when LoRA training state cannot be persisted safely."""


class LoRACopyMachineImageEvidence(BaseModel):
    image_path: str
    width: int
    height: int
    sha256: str


class LoRACopyMachineSourceReference(BaseModel):
    variant: Literal["source_reference"] = "source_reference"
    image_path: str
    image_sha256: str
    image_size_bytes: int
    width: int
    height: int


class LoRACopyMachineModelEvidence(BaseModel):
    model_path: str
    sha256: str
    size_bytes: int = 0


class LoRACopyMachineFileEvidence(BaseModel):
    path: str
    sha256: str
    size_bytes: int = 0


class LoRACopyMachineSessionCreateData(BaseModel):
    name: str = "Untitled copy-machine LoRA"
    profile: TrainingProfile = "sdxl"
    method: LoRACopyMachineMethod = "direct_difference"
    intent: str = "style_delta"
    source_image_path: str
    target_image_path: str = ""
    target_transform: str = "none"
    trigger_word: str = ""
    training_caption: str = ""


class LoRACopyMachineSession(BaseModel):
    id: str
    name: str
    profile: TrainingProfile
    method: LoRACopyMachineMethod
    intent: str
    source: LoRACopyMachineImageEvidence
    target: LoRACopyMachineImageEvidence | None = None
    target_transform: str = "none"
    trigger_word: str = ""
    training_caption: str = ""
    status: LoRACopyMachineStatus = "draft"
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class LoRACopyMachineTargetPrepareData(BaseModel):
    source_image_path: str
    output_dir: str
    transform: LoRACopyMachineTargetTransform = "grayscale"
    output_name: str = "copy-machine-target"


class LoRACopyMachineTargetPreparation(BaseModel):
    transform: LoRACopyMachineTargetTransform
    source: LoRACopyMachineImageEvidence
    target: LoRACopyMachineImageEvidence


class LoRACopyMachineAdapterReadinessItem(BaseModel):
    id: str
    label: str
    profile: TrainingProfile
    methods: list[LoRACopyMachineMethod]
    status: LoRACopyMachineAdapterStatus
    executable: str = ""
    executable_path: str = ""
    warnings: list[str] = Field(default_factory=list)


class LoRACopyMachineAdapterReadiness(BaseModel):
    items: list[LoRACopyMachineAdapterReadinessItem]
    ready_count: int = 0
    missing_count: int = 0


class LoRACopyMachineTrainingRunPrepareData(BaseModel):
    session_id: str
    model_path: str
    output_dir: str
    output_name: str = ""
    traintrain_path: str = ""
    python_command: str = "python"
    qwen3_path: str = ""
    vae_path: str = ""
    training_working_dir: str = ""
    accelerate_command: str = "accelerate"
    repeats: int | None = Field(default=None, ge=1, le=1000)
    max_train_epochs: int | None = Field(default=None, ge=1, le=1000)
    addift_timestep_window: LoRACopyMachineADDifTTimestepWindow = "action_500_1000"
    addift_snr_gamma: LoRACopyMachineADDifTSNRGamma = "preset_5_0"


class LoRACopyMachineTrainingRunPlan(BaseModel):
    session_id: str
    adapter_id: str = "traintrain_standalone"
    method: LoRACopyMachineMethod
    status: LoRACopyMachineTrainingRunStatus
    output_dir: str = ""
    output_name: str = ""
    effective_config_path: str = ""
    command_args: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LoRACopyMachineTrainingRun(BaseModel):
    id: str
    session_id: str
    adapter_id: str = "traintrain_standalone"
    method: LoRACopyMachineMethod
    status: LoRACopyMachineTrainingRunStatus
    output_dir: str = ""
    output_name: str = ""
    effective_config_path: str = ""
    effective_config_evidence: LoRACopyMachineFileEvidence | None = None
    command_args: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    log_path: str = ""
    trained_model_paths: list[str] = Field(default_factory=list)
    trained_model_evidence: list[LoRACopyMachineModelEvidence] = Field(default_factory=list)
    registered_model_paths: list[str] = Field(default_factory=list)
    delegated_training_run_id: str = ""
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    progress_message: str = ""
    pid: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str
    updated_at: str


class LoRACopyMachineComparisonPrepareData(BaseModel):
    run_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 1
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    positive_strength: float = Field(default=1.0, ge=-2.0, le=2.0)


class LoRACopyMachineComparisonLoRA(BaseModel):
    path: str
    strength: float


class LoRACopyMachineComparisonRequest(BaseModel):
    label: str
    prompt: str
    negative_prompt: str = ""
    seed: int
    width: int
    height: int
    loras: list[LoRACopyMachineComparisonLoRA] = Field(default_factory=list)


class LoRACopyMachineComparisonPlan(BaseModel):
    run_id: str
    status: Literal["planned", "blocked"]
    generation_requests: list[LoRACopyMachineComparisonRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LoRATrainingPreset(BaseModel):
    id: str
    name: str
    profile: TrainingProfile
    training_type: TrainingType
    description: str
    script: str
    network_module: str
    network_dim: int
    network_alpha: int
    learning_rate: str
    unet_lr: str | None = None
    text_encoder_lr: str | None = None
    optimizer_type: str
    lr_scheduler: str
    mixed_precision: str
    save_precision: str
    resolution: tuple[int, int]
    batch_size: int
    max_train_epochs: int
    repeats: int
    caption_extension: str = ".txt"
    shuffle_caption: bool = True
    keep_tokens: int = 1
    enable_bucket: bool = True
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    sigmoid_scale: float | None = None
    notes: list[str] = Field(default_factory=list)


class DatasetSubset(BaseModel):
    image_dir: str = Field(min_length=1)
    num_repeats: int = Field(default=10, ge=1, le=1000)
    caption_extension: str = ".txt"
    class_tokens: str = ""
    keep_tokens: int | None = Field(default=None, ge=0, le=32)
    shuffle_caption: bool | None = None


class DatasetConfigPreview(BaseModel):
    preset: LoRATrainingPreset
    toml: str
    command_args: list[str]
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceInput(BaseModel):
    id: str | None = None
    source_type: LoRATrainingCaptionSourceType
    label: str = ""
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)


class LoRATrainingCaptionSourceRecord(BaseModel):
    id: str
    source_type: LoRATrainingCaptionSourceType
    label: str = ""
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    raw_text: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: str
    updated_at: str


class LoRATrainingCaptionExportSettings(BaseModel):
    suppressed_tags: list[str] = Field(default_factory=list)
    organizational_tags: list[str] = Field(default_factory=list)
    min_source_confidence: float = Field(default=0.0, ge=0, le=1)
    exclude_rating_tags: bool = True
    category_prefixing: bool = False
    output_mode: LoRATrainingCaptionOutputMode = "caption_txt"


class LoRATrainingCharacterCaptionPolicy(BaseModel):
    trigger_token: str = Field(default="", max_length=200)
    excluded_identity_tags: list[str] = Field(default_factory=list, max_length=200)
    variable_detail_guidance: str = Field(default="", max_length=2000)


class LoRATrainingDatasetItem(BaseModel):
    id: str
    image_path: str
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    caption_sources: list[LoRATrainingCaptionSourceRecord] = Field(default_factory=list)
    source_generation_id: int | None = None
    source_output_path: str = ""
    created_at: str
    updated_at: str


class LoRATrainingDataset(BaseModel):
    id: str
    name: str
    preset_id: str
    profile: TrainingProfile
    training_type: TrainingType
    caption_export_settings: LoRATrainingCaptionExportSettings = Field(default_factory=LoRATrainingCaptionExportSettings)
    character_caption_policy: LoRATrainingCharacterCaptionPolicy = Field(
        default_factory=LoRATrainingCharacterCaptionPolicy
    )
    items: list[LoRATrainingDatasetItem] = Field(default_factory=list)
    created_at: str
    updated_at: str


class LoRATrainingDatasetDirectoryImportResult(BaseModel):
    dataset: LoRATrainingDataset
    imported_count: int = 0
    skipped_count: int = 0
    duplicate_count: int = 0
    sidecar_caption_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceImportDirectoryResult(BaseModel):
    dataset: LoRATrainingDataset
    source_type: LoRATrainingCaptionSourceType
    imported_count: int = 0
    missing_count: int = 0
    skipped_count: int = 0
    scanned_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceRefreshCommandResult(BaseModel):
    command: str
    command_args: list[str] = Field(default_factory=list)
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""


class LoRATrainingCaptionSourceRefreshResult(BaseModel):
    dataset: LoRATrainingDataset
    source_type: LoRATrainingCaptionSourceType
    imported_count: int = 0
    missing_count: int = 0
    skipped_count: int = 0
    scanned_count: int = 0
    commands: list[LoRATrainingCaptionSourceRefreshCommandResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingAnimaTaggerRonExportResult(BaseModel):
    dataset: LoRATrainingDataset
    directory_path: str
    written_count: int = 0
    skipped_count: int = 0
    existing_count: int = 0
    written_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceToolReadinessItem(BaseModel):
    id: str
    label: str
    source_type: LoRATrainingCaptionSourceType
    content_kind: LoRATrainingCaptionSourceContentKind
    source_extension: str
    default_label: str
    confidence: float | None = None
    status: LoRATrainingCaptionSourceToolStatus
    built_in: bool = False
    executable: str = ""
    executable_path: str = ""
    default_commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceToolReadiness(BaseModel):
    items: list[LoRATrainingCaptionSourceToolReadinessItem] = Field(default_factory=list)
    ready_count: int = 0
    missing_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingDatasetImageInput(BaseModel):
    image_path: str = Field(min_length=1)
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    source_generation_id: int | None = None
    source_output_path: str = ""


class LoRATrainingDatasetItemAnalysis(BaseModel):
    item_id: str
    image_path: str
    width: int
    height: int
    aspect_ratio: float
    orientation: str
    average_color: str
    dominant_tone: str
    suggested_caption: str
    suggested_tags: list[str] = Field(default_factory=list)
    metadata_tags: list[str] = Field(default_factory=list)
    source_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionSourceSummary(BaseModel):
    manual_tags: list[str] = Field(default_factory=list)
    metadata_tags: list[str] = Field(default_factory=list)
    source_tags: list[str] = Field(default_factory=list)
    source_captions: list[str] = Field(default_factory=list)
    source_records: list[LoRATrainingCaptionSourceRecord] = Field(default_factory=list)
    suppressed_tags: list[str] = Field(default_factory=list)
    organizational_tags: list[str] = Field(default_factory=list)
    candidate_tags: list[str] = Field(default_factory=list)
    exportable_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionAssist(BaseModel):
    item_id: str
    provider_mode: LoRATrainingCaptionAssistProviderMode = "prompt_only"
    language: LoRATrainingCaptionAssistLanguage
    caption_style: LoRATrainingCaptionAssistStyle
    system_prompt: str
    user_prompt: str
    suggested_caption: str
    suggested_tags: list[str] = Field(default_factory=list)
    analysis: LoRATrainingDatasetItemAnalysis
    source_summary: LoRATrainingCaptionSourceSummary
    provider_model: str = ""
    provider_response: str = ""
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingDatasetCaptionAssistResult(BaseModel):
    dataset_id: str
    target_count: int = 0
    drafted_count: int = 0
    skipped_count: int = 0
    assists: list[LoRATrainingCaptionAssist] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingCaptionCoverageItem(BaseModel):
    item_id: str
    image_path: str
    caption: str
    final_categories: list[str] = Field(default_factory=list)
    source_categories: list[str] = Field(default_factory=list)
    missing_final_categories: list[str] = Field(default_factory=list)
    needs_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingDatasetCaptionCoverageResult(BaseModel):
    dataset_id: str
    training_type: TrainingType
    item_count: int = 0
    review_count: int = 0
    items: list[LoRATrainingCaptionCoverageItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LoRATrainingSampleImageScore(BaseModel):
    image_path: str
    width: int = 0
    height: int = 0
    score: int = Field(ge=0, le=100)
    rating: LoRATrainingSampleRating
    average_brightness: float | None = None
    contrast: float | None = None
    warnings: list[str] = Field(default_factory=list)


class MaterializedDatasetItem(BaseModel):
    item_id: str
    source_path: str
    image_path: str
    caption_path: str
    metadata_path: str = ""
    caption: str


class LoRATrainingCharacterCaptionIdentityHit(BaseModel):
    item_id: str
    tags: list[str] = Field(default_factory=list)


class LoRATrainingCharacterCaptionGate(BaseModel):
    applicable: bool = False
    passed: bool = True
    total_count: int = 0
    trigger_first_count: int = 0
    trigger_first_rate: float = 0.0
    missing_trigger_item_ids: list[str] = Field(default_factory=list)
    misplaced_trigger_item_ids: list[str] = Field(default_factory=list)
    duplicate_trigger_item_ids: list[str] = Field(default_factory=list)
    identity_hit_count: int = 0
    identity_hit_items: list[LoRATrainingCharacterCaptionIdentityHit] = Field(default_factory=list)
    effective_keep_tokens: int = 0
    keep_tokens_ready: bool = True
    reasons: list[str] = Field(default_factory=list)


class MaterializedDataset(BaseModel):
    dataset: LoRATrainingDataset
    root_dir: str
    image_dir: str
    dataset_config_path: str
    sample_prompts_path: str = ""
    item_count: int
    items: list[MaterializedDatasetItem] = Field(default_factory=list)
    skipped_items: list[str] = Field(default_factory=list)
    command_args: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    character_caption_gate: LoRATrainingCharacterCaptionGate = Field(
        default_factory=LoRATrainingCharacterCaptionGate
    )


class LoRATrainingRunPreflight(BaseModel):
    ready: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    character_caption_gate: LoRATrainingCharacterCaptionGate = Field(
        default_factory=LoRATrainingCharacterCaptionGate
    )


class LoRATrainingRun(BaseModel):
    id: str
    operation_id: str = ""
    dataset_id: str
    dataset_name: str
    preset_id: str
    profile: TrainingProfile
    training_type: TrainingType
    status: LoRATrainingRunStatus
    materialized_root: str
    image_dir: str
    dataset_config_path: str
    sample_prompts_path: str = ""
    output_dir: str
    output_name: str
    working_dir: str
    command_args: list[str] = Field(default_factory=list)
    preflight: LoRATrainingRunPreflight
    pid: int | None = None
    exit_code: int | None = None
    log_path: str = ""
    log_tail: str = ""
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: float | None = None
    progress_message: str = ""
    trained_model_paths: list[str] = Field(default_factory=list)
    sample_image_paths: list[str] = Field(default_factory=list)
    sample_image_scores: list[LoRATrainingSampleImageScore] = Field(default_factory=list)
    registered_model_paths: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str
    updated_at: str


class LoRATrainingRunModelRegistration(BaseModel):
    run: LoRATrainingRun
    installed_model_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class LoRATrainingCaptionAssistRequestData:
    language: LoRATrainingCaptionAssistLanguage = "en"
    caption_style: LoRATrainingCaptionAssistStyle = "tags"
    max_tags: int = 30
    extra_instruction: str = ""
    provider_mode: LoRATrainingCaptionAssistProviderMode = "prompt_only"
    provider_endpoint: str = ""
    provider_model: str = ""
    provider_api_key: str = ""
    provider_api_key_env_var: str = ""
    provider_timeout_seconds: float = 60.0
    include_image: bool = False


@dataclass(frozen=True)
class DatasetPreviewRequestData:
    preset_id: str
    image_dir: str
    output_dir: str
    output_name: str
    pretrained_model_name_or_path: str
    vae: str = ""
    qwen3: str = ""
    repeats: int | None = None
    max_train_epochs: int | None = None
    class_tokens: str = ""
    caption_extension: str | None = None
    optimizer_type: str = ""
    optimizer_args: str = ""
    optimizer_low_vram: bool = False


@dataclass(frozen=True)
class DatasetMaterializeRequestData:
    data_dir: Path
    project_root: Path
    output_dir: str
    output_name: str
    pretrained_model_name_or_path: str = ""
    vae: str = ""
    qwen3: str = ""
    repeats: int | None = None
    max_train_epochs: int | None = None
    class_tokens: str = ""
    caption_extension: str | None = None
    optimizer_type: str = ""
    optimizer_args: str = ""
    optimizer_low_vram: bool = False


@dataclass(frozen=True)
class TrainingRunPrepareRequestData:
    data_dir: Path
    project_root: Path
    output_dir: str
    output_name: str
    pretrained_model_name_or_path: str = ""
    vae: str = ""
    qwen3: str = ""
    repeats: int | None = None
    max_train_epochs: int | None = None
    class_tokens: str = ""
    caption_extension: str | None = None
    working_dir: str = ""
    accelerate_command: str = "accelerate"
    optimizer_type: str = ""
    optimizer_args: str = ""
    optimizer_low_vram: bool = False


_PRESETS: tuple[LoRATrainingPreset, ...] = (
    LoRATrainingPreset(
        id="sdxl-copy-lora-starter",
        name="SDXL copy LoRA starter",
        profile="sdxl",
        training_type="copy_lora",
        description="A conservative SDXL preset for small object/product copy-style LoRAs.",
        script="sdxl_train_network.py",
        network_module="networks.lora",
        network_dim=64,
        network_alpha=32,
        learning_rate="1e-4",
        unet_lr="1e-4",
        text_encoder_lr="1e-5",
        optimizer_type="AdamW8bit",
        lr_scheduler="cosine",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=10,
        repeats=20,
        keep_tokens=1,
        notes=[
            "Starter preset only; review dataset size and overfit risk before long runs.",
            "Use captions that separate the trigger token from visual attributes.",
            "SDXL text encoder LR is intentionally conservative; raise only after caption quality is stable.",
        ],
    ),
    LoRATrainingPreset(
        id="sdxl-character-lora-starter",
        name="SDXL character LoRA starter",
        profile="sdxl",
        training_type="character_lora",
        description="A balanced SDXL character LoRA preset for tagged image sets.",
        script="sdxl_train_network.py",
        network_module="networks.lora",
        network_dim=32,
        network_alpha=16,
        learning_rate="1e-4",
        unet_lr="1e-4",
        text_encoder_lr="1e-5",
        optimizer_type="AdamW8bit",
        lr_scheduler="cosine",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=6,
        repeats=10,
        keep_tokens=1,
        notes=[
            "Prefer varied poses, expressions, outfits, and backgrounds.",
            "Keep a stable trigger token at the beginning of each caption.",
            "For a 30-image character set this starts around 1800 steps at batch size 1 before manual tuning.",
        ],
    ),
    LoRATrainingPreset(
        id="sdxl-style-lora-starter",
        name="SDXL style LoRA starter",
        profile="sdxl",
        training_type="style_lora",
        description="A lower-rank SDXL style preset intended to avoid memorizing subjects.",
        script="sdxl_train_network.py",
        network_module="networks.lora",
        network_dim=16,
        network_alpha=8,
        learning_rate="8e-5",
        unet_lr="8e-5",
        text_encoder_lr="1e-5",
        optimizer_type="AdamW8bit",
        lr_scheduler="cosine",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=6,
        repeats=8,
        keep_tokens=0,
        notes=[
            "Use diverse subjects so the LoRA learns style instead of a fixed character.",
            "Avoid using a single repeated character token unless intentionally training character+style.",
            "For a 30-image style set this starts around 1440 steps at batch size 1 before sample review.",
        ],
    ),
    LoRATrainingPreset(
        id="anima-copy-lora-starter",
        name="Anima single-image overfit starter",
        profile="anima",
        training_type="copy_lora",
        description="A target-only Anima preset for single-image overfit LoRAs. It does not learn a source-to-target difference.",
        script="anima_train_network.py",
        network_module="networks.lora_anima",
        network_dim=32,
        network_alpha=32,
        learning_rate="1.0",
        optimizer_type="Prodigy",
        lr_scheduler="constant",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=10,
        repeats=240,
        sigmoid_scale=1.3,
        keep_tokens=1,
        notes=[
            "Requires Anima DIT, Qwen3 text encoder, and VAE paths before execution.",
            "Use a small validation set to catch memorization before increasing epochs.",
            "Uses sigmoid timestep sampling with sigmoid_scale 1.3, matching the bundled TrainFlow default.",
        ],
    ),
    LoRATrainingPreset(
        id="anima-character-lora-starter",
        name="Anima character LoRA starter",
        profile="anima",
        training_type="character_lora",
        description="A cautious Anima LoRA preset using sd-scripts anima_train_network.py.",
        script="anima_train_network.py",
        network_module="networks.lora_anima",
        network_dim=32,
        network_alpha=16,
        learning_rate="1e-4",
        optimizer_type="AdamW8bit",
        lr_scheduler="constant",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=8,
        repeats=15,
        sigmoid_scale=1.3,
        keep_tokens=1,
        notes=[
            "Requires Anima DIT, Qwen3 text encoder, and VAE paths before execution.",
            "Treat fp16 as risky for Anima; prefer bf16-capable hardware.",
            "Uses sigmoid timestep sampling with sigmoid_scale 1.3, matching the bundled TrainFlow default.",
        ],
    ),
    LoRATrainingPreset(
        id="anima-style-lora-starter",
        name="Anima style LoRA starter",
        profile="anima",
        training_type="style_lora",
        description="A lower-rank Anima style preset for diverse style datasets.",
        script="anima_train_network.py",
        network_module="networks.lora_anima",
        network_dim=16,
        network_alpha=8,
        learning_rate="8e-5",
        optimizer_type="AdamW8bit",
        lr_scheduler="constant",
        mixed_precision="bf16",
        save_precision="bf16",
        resolution=(1024, 1024),
        batch_size=1,
        max_train_epochs=6,
        repeats=10,
        sigmoid_scale=1.3,
        keep_tokens=0,
        notes=[
            "Requires Anima-specific model component paths before execution.",
            "Keep dataset subjects diverse to reduce identity memorization.",
            "Uses sigmoid timestep sampling with sigmoid_scale 1.3, matching the bundled TrainFlow default.",
        ],
    ),
)


_RUN_PROCESSES: dict[str, subprocess.Popen] = {}
_COPY_MACHINE_RUN_PROCESSES: dict[str, subprocess.Popen] = {}


def list_lora_training_presets(
    *,
    profile: TrainingProfile | None = None,
    training_type: TrainingType | None = None,
) -> list[LoRATrainingPreset]:
    presets = list(_PRESETS)
    if profile:
        presets = [preset for preset in presets if preset.profile == profile]
    if training_type:
        presets = [preset for preset in presets if preset.training_type == training_type]
    return presets


def get_lora_training_preset(preset_id: str) -> LoRATrainingPreset | None:
    return next((preset for preset in _PRESETS if preset.id == preset_id), None)


def build_lora_training_caption_source_tool_readiness(
    *,
    project_root: Path,
) -> LoRATrainingCaptionSourceToolReadiness:
    specs = [
        {
            "id": "asteria_anima_tagger_ron_bridge",
            "label": "Asteria RON bridge",
            "source_type": "metadata",
            "content_kind": "anima_tagger_ron",
            "source_extension": ".ron",
            "default_label": "anima-tagger RON",
            "confidence": None,
            "built_in": True,
            "status": "ready",
            "notes": [
                "Export Asteria caption evidence as same-stem RON files, edit them externally, then import the edited RON files.",
            ],
        },
        {
            "id": "anima_tagger",
            "label": "anima-tagger",
            "source_type": "metadata",
            "content_kind": "anima_tagger_ron",
            "source_extension": ".ron",
            "default_label": "anima-tagger RON",
            "confidence": None,
            "executable": "anima-tagger",
            "default_commands": ['anima-tagger tag "{source_dir}" --force'],
            "notes": [
                "Use for manual tags, WD14 auto tags, Qwen3-VL captions, and Danbooru tags.",
            ],
        },
        {
            "id": "wd14_tagger",
            "label": "WD14 tagger",
            "source_type": "wd14_tagger",
            "content_kind": "tags",
            "source_extension": ".wd14.txt",
            "default_label": "WD14 bulk import",
            "confidence": 0.35,
            "default_commands": ['python tools/wd14_tagger.py "{source_dir}" --out-ext {source_extension}'],
            "notes": [
                "Use as candidate tag evidence for visible variables. Rating tags should stay review-only unless explicitly needed.",
            ],
        },
        {
            "id": "qwen3_vl_captioner",
            "label": "Qwen3-VL captioner",
            "source_type": "vision_caption",
            "content_kind": "caption",
            "source_extension": ".qwen.txt",
            "default_label": "Qwen3-VL caption",
            "confidence": None,
            "default_commands": ['python tools/qwen_caption.py "{source_dir}" --out-ext {source_extension}'],
            "notes": [
                "Use as vision caption evidence. Keep final captions tag-first for SDXL/Anima unless prose is genuinely useful.",
            ],
        },
        {
            "id": "danbooru_api",
            "label": "Danbooru API",
            "source_type": "booru_api",
            "content_kind": "json",
            "source_extension": ".booru.json",
            "default_label": "Danbooru API",
            "confidence": None,
            "default_commands": ['python tools/danbooru_fetch.py "{source_dir}" --out-ext {source_extension}'],
            "notes": [
                "Use only as provenance-backed candidate evidence; imported API tags are not auto-applied to final captions.",
            ],
        },
    ]
    items: list[LoRATrainingCaptionSourceToolReadinessItem] = []
    for spec in specs:
        items.append(_caption_source_tool_readiness_item(spec))
    return LoRATrainingCaptionSourceToolReadiness(
        items=items,
        ready_count=sum(1 for item in items if item.status == "ready"),
        missing_count=sum(1 for item in items if item.status == "missing"),
        warnings=[
            "First-party WD14/Qwen3-VL/Danbooru execution adapters are not embedded yet; use sidecar import, refresh commands, or the Asteria RON bridge until owned adapters are implemented and validated."
        ],
    )


def build_lora_copy_machine_adapter_readiness(
    *,
    project_root: Path,
    traintrain_path: str = "",
) -> LoRACopyMachineAdapterReadiness:
    warnings: list[str] = []
    executable_path: Path | None = None
    if traintrain_path.strip():
        try:
            configured_path = _resolve_copy_machine_traintrain_path(traintrain_path, project_root)
        except ValueError as exc:
            warnings.append(str(exc))
        else:
            script_path = _copy_machine_traintrain_script_path(configured_path)
            if _path_is_file(script_path) and _copy_machine_traintrain_configs_ready(configured_path):
                executable_path = script_path
            else:
                warnings.append("copy_machine_adapter_not_ready")
    else:
        executable_path = _discover_copy_machine_traintrain_cli_path()
    if executable_path:
        status: LoRACopyMachineAdapterStatus = "ready"
    else:
        status = "missing"
    if status == "missing" and not warnings:
        warnings.append("copy_machine_adapter_not_configured")
    items = [
        LoRACopyMachineAdapterReadinessItem(
            id="traintrain_standalone",
            label="TrainTrain standalone",
            profile="sdxl",
            methods=["direct_difference", "classic_copy_machine"],
            status=status,
            executable="traintrain",
            executable_path=str(executable_path) if executable_path else "",
            warnings=warnings,
        ),
        LoRACopyMachineAdapterReadinessItem(
            id="anima_copy_machine",
            label="Anima copy-machine training",
            profile="anima",
            methods=["anima_experimental"],
            status="missing",
            executable="",
            executable_path="",
            warnings=["copy_machine_anima_training_not_ready"],
        ),
    ]
    return LoRACopyMachineAdapterReadiness(
        items=items,
        ready_count=sum(1 for item in items if item.status == "ready"),
        missing_count=sum(1 for item in items if item.status == "missing"),
    )


def _caption_source_tool_readiness_item(spec: dict) -> LoRATrainingCaptionSourceToolReadinessItem:
    executable = str(spec.get("executable") or "")
    executable_path = shutil.which(executable) if executable else ""
    built_in = bool(spec.get("built_in"))
    if built_in:
        status: LoRATrainingCaptionSourceToolStatus = "ready"
    elif executable_path:
        status = "ready"
    else:
        status = "missing"
    warnings: list[str] = []
    if status == "missing":
        warnings.append("caption_source_tool_not_configured")
    return LoRATrainingCaptionSourceToolReadinessItem(
        id=str(spec["id"]),
        label=str(spec["label"]),
        source_type=spec["source_type"],
        content_kind=spec["content_kind"],
        source_extension=str(spec["source_extension"]),
        default_label=str(spec["default_label"]),
        confidence=spec.get("confidence"),
        status=status,
        built_in=built_in,
        executable=executable,
        executable_path=executable_path or "",
        default_commands=list(spec.get("default_commands") or []),
        notes=list(spec.get("notes") or []),
        warnings=warnings,
    )


def build_lora_copy_machine_training_caption(trigger_word: str, training_caption: str) -> str:
    trigger = trigger_word.strip()
    tokens = [token.strip() for token in training_caption.split(",") if token.strip()]
    remaining = [token for token in tokens if token.casefold() != trigger.casefold()]
    return ", ".join([trigger, *remaining])


def _anima_copy_machine_plan_errors(
    session: LoRACopyMachineSession,
    data: LoRACopyMachineTrainingRunPrepareData,
    *,
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    if session.target is None:
        errors.append("lora_copy_machine_target_image_required")
    elif not _path_is_file(Path(session.target.image_path)):
        errors.append("lora_copy_machine_target_image_required")

    for path_text, missing_code in (
        (data.model_path, "copy_machine_model_not_found"),
        (data.qwen3_path, "copy_machine_qwen3_not_found"),
        (data.vae_path, "copy_machine_vae_not_found"),
    ):
        try:
            path = _resolve_source_path(path_text, project_root) if path_text.strip() else Path()
        except OSError:
            path = Path()
        if not path_text.strip() or not _path_is_file(path):
            errors.append(missing_code)
        elif _is_disallowed_tool_execution_path(path, project_root):
            errors.append("copy_machine_training_input_path_not_allowed")

    working_dir_text = data.training_working_dir.strip()
    if not working_dir_text:
        errors.append("copy_machine_training_working_dir_required")
    else:
        try:
            working_dir = _resolve_source_path(working_dir_text, project_root)
        except OSError:
            working_dir = _source_path_candidate(working_dir_text, project_root)
        if _is_disallowed_tool_execution_path(working_dir, project_root):
            errors.append("lora_training_working_dir_not_allowed")
        else:
            for relative_path, missing_code in (
                ("anima_train_network.py", "training_script_not_found"),
                ("networks/lora_anima.py", "anima_lora_network_script_not_found"),
                ("networks/convert_anima_lora_to_comfy.py", "anima_comfy_conversion_script_not_found"),
            ):
                if not _path_is_file(working_dir / relative_path):
                    errors.append(missing_code)

    command_parts = _command_prefix(data.accelerate_command, project_root)
    if not _command_prefix_exists(command_parts):
        errors.append("accelerate_command_not_found")
    else:
        python_module = _python_module_name(command_parts)
        if python_module and python_module.startswith("accelerate") and not _python_module_available(
            command_parts[0], python_module
        ):
            errors.append("accelerate_python_module_not_found")
        if not _python_module_available(command_parts[0], "prodigyopt"):
            errors.append("optimizer_python_module_not_found")
    return list(dict.fromkeys(errors))


def create_lora_copy_machine_session(
    data: LoRACopyMachineSessionCreateData,
    repository: "LoRACopyMachineSessionRepository",
    *,
    project_root: Path,
) -> LoRACopyMachineSession:
    source = _copy_machine_image_evidence(data.source_image_path, project_root)
    target = (
        _copy_machine_image_evidence(data.target_image_path, project_root)
        if data.target_image_path.strip()
        else None
    )
    if data.method in {"direct_difference", "classic_copy_machine"} and target is None:
        raise ValueError("lora_copy_machine_target_image_required")
    trigger_word = data.trigger_word.strip()
    training_caption = data.training_caption.strip()
    if target is not None and (source.width, source.height) != (target.width, target.height):
        raise ValueError("lora_copy_machine_image_size_mismatch")
    now = _utc_now()
    session = LoRACopyMachineSession(
        id=uuid4().hex,
        name=data.name.strip() or "Untitled copy-machine LoRA",
        profile=data.profile,
        method=data.method,
        intent=data.intent.strip() or "style_delta",
        source=source,
        target=target,
        target_transform=data.target_transform.strip() or "none",
        trigger_word=trigger_word,
        training_caption=training_caption,
        created_at=now,
        updated_at=now,
    )
    return repository.create(session)


def prepare_lora_copy_machine_target_image(
    data: LoRACopyMachineTargetPrepareData,
    *,
    project_root: Path,
) -> LoRACopyMachineTargetPreparation:
    source = _copy_machine_image_evidence(data.source_image_path, project_root)
    output_dir = _resolve_source_path(data.output_dir, project_root)
    output_name = _safe_slug(data.output_name)
    output_path = output_dir / f"{output_name}-{uuid4().hex[:8]}.png"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(source.image_path) as image:
            target_image = _apply_copy_machine_target_transform(image.convert("RGB"), data.transform)
            target_image.save(output_path)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    target = _copy_machine_image_evidence(str(output_path), project_root)
    if (source.width, source.height) != (target.width, target.height):
        raise ValueError("lora_copy_machine_image_size_mismatch")
    return LoRACopyMachineTargetPreparation(
        transform=data.transform,
        source=source,
        target=target,
    )


def attach_lora_copy_machine_session_target(
    session_id: str,
    target_image_path: str,
    target_transform: str,
    repository: "LoRACopyMachineSessionRepository",
    *,
    project_root: Path,
) -> LoRACopyMachineSession | None:
    session = repository.get(session_id)
    if session is None:
        return None
    target = _copy_machine_image_evidence(target_image_path, project_root)
    if (session.source.width, session.source.height) != (target.width, target.height):
        raise ValueError("lora_copy_machine_image_size_mismatch")
    session.target = target
    session.target_transform = target_transform.strip() or "none"
    session.updated_at = _utc_now()
    return repository.save(session)


def prepare_lora_copy_machine_training_run(
    data: LoRACopyMachineTrainingRunPrepareData,
    repository: "LoRACopyMachineSessionRepository",
    *,
    project_root: Path,
) -> LoRACopyMachineTrainingRunPlan | None:
    session = repository.get(data.session_id)
    if session is None:
        return None
    output_dir = _resolve_source_path(data.output_dir, project_root)
    output_name = _safe_slug(data.output_name or session.name or "copy-machine-lora")
    plan = LoRACopyMachineTrainingRunPlan(
        session_id=session.id,
        method=session.method,
        status="blocked",
        output_dir=str(output_dir),
        output_name=output_name,
    )
    if session.profile == "anima" or session.method == "anima_experimental":
        plan.adapter_id = "asteria_anima_lora"
        plan.errors.append("copy_machine_anima_difference_not_ready")
        return plan
    if session.profile != "sdxl":
        plan.errors.append("copy_machine_adapter_profile_not_supported")
        return plan
    if data.addift_snr_gamma == "low_1_0":
        if session.method != "direct_difference":
            plan.errors.append("copy_machine_addift_snr_gamma_method_mismatch")
            return plan
        if data.addift_timestep_window != "style_200_400":
            plan.errors.append("copy_machine_addift_snr_gamma_requires_style_window")
            return plan
    if data.addift_timestep_window != "action_500_1000" and session.method != "direct_difference":
        plan.errors.append("copy_machine_addift_timestep_window_method_mismatch")
        return plan
    if session.target is None:
        plan.errors.append("lora_copy_machine_target_image_required")
        return plan
    model_path = _resolve_source_path(data.model_path, project_root)
    if not _path_is_file(model_path):
        plan.errors.append("copy_machine_model_not_found")
        return plan
    try:
        traintrain_root = _resolve_copy_machine_traintrain_path(data.traintrain_path, project_root)
    except ValueError as exc:
        plan.errors.append(str(exc))
        return plan
    script_path = _copy_machine_traintrain_script_path(traintrain_root)
    if not _path_is_file(script_path):
        plan.errors.append("copy_machine_adapter_not_ready")
        return plan
    command_name = "addift" if session.method == "direct_difference" else "difference"
    config_path = _copy_machine_traintrain_config_path(traintrain_root, session.method)
    if config_path is None:
        plan.errors.append("copy_machine_adapter_config_missing")
        return plan
    plan.command_args = [
        _resolve_command_executable(data.python_command.strip() or "python", project_root),
        str(_copy_machine_traintrain_runner_path()),
        command_name,
        "--traintrain-root",
        str(_copy_machine_traintrain_root(traintrain_root)),
        "--config",
        str(config_path),
        "--model",
        str(model_path),
        "--clip-skip",
        "2",
        "--orig-image",
        session.source.image_path,
        "--targ-image",
        session.target.image_path,
        "--output-name",
        output_name,
        "--lora-dir",
        str(output_dir),
    ]
    plan.effective_config_path = str(output_dir / f"{output_name}.asteria-effective-config.json")
    plan.command_args.extend(["--effective-config-output", plan.effective_config_path])
    if data.addift_timestep_window == "style_200_400":
        plan.command_args.extend(
            [
                "--set",
                "train_min_timesteps=200",
                "--set",
                "train_max_timesteps=400",
            ]
        )
        plan.warnings.append("experiment_override:addift_timestep_window=style_200_400")
    if data.addift_snr_gamma == "low_1_0":
        plan.command_args.extend(["--set", "train_snr_gamma='1.0'"])
        plan.warnings.append("experiment_override:addift_snr_gamma=low_1_0")
    if session.target_transform and session.target_transform != "none":
        plan.warnings.append(f"target_transform:{session.target_transform}")
    plan.status = "planned"
    return plan


def create_lora_copy_machine_training_run(
    data: LoRACopyMachineTrainingRunPrepareData,
    session_repository: "LoRACopyMachineSessionRepository",
    run_repository: "LoRACopyMachineTrainingRunRepository",
    *,
    project_root: Path,
    training_run_repository: "LoRATrainingRunRepository | None" = None,
    data_dir: Path | None = None,
) -> LoRACopyMachineTrainingRun | None:
    plan = prepare_lora_copy_machine_training_run(data, session_repository, project_root=project_root)
    if plan is None:
        return None
    session = session_repository.get(data.session_id)
    delegated_run: LoRATrainingRun | None = None
    if (
        session is not None
        and (session.profile == "anima" or session.method == "anima_experimental")
        and plan.status == "planned"
    ):
        if training_run_repository is None:
            raise ValueError("copy_machine_training_run_repository_required")
        if session.target is None:
            raise ValueError("lora_copy_machine_target_image_required")
        now = _utc_now()
        dataset = LoRATrainingDataset(
            id=uuid4().hex,
            name=f"{session.name} target",
            preset_id="anima-copy-lora-starter",
            profile="anima",
            training_type="copy_lora",
            items=[
                LoRATrainingDatasetItem(
                    id=uuid4().hex,
                    image_path=session.target.image_path,
                    caption=build_lora_copy_machine_training_caption(
                        session.trigger_word,
                        session.training_caption,
                    ),
                    created_at=now,
                    updated_at=now,
                )
            ],
            created_at=now,
            updated_at=now,
        )
        delegated_run = prepare_lora_training_run(
            dataset,
            TrainingRunPrepareRequestData(
                data_dir=data_dir or project_root / "data",
                project_root=project_root,
                output_dir=plan.output_dir,
                output_name=plan.output_name,
                pretrained_model_name_or_path=data.model_path,
                vae=data.vae_path,
                qwen3=data.qwen3_path,
                repeats=data.repeats,
                max_train_epochs=data.max_train_epochs,
                working_dir=data.training_working_dir,
                accelerate_command=data.accelerate_command,
            ),
            training_run_repository,
        )
        plan.command_args = delegated_run.command_args
        plan.status = "planned" if delegated_run.preflight.ready else "blocked"
        plan.errors = list(delegated_run.preflight.errors)
        plan.warnings = [*plan.warnings, *delegated_run.preflight.warnings]
    now = _utc_now()
    run_id = uuid4().hex
    if plan.effective_config_path:
        immutable_config_path = str(
            Path(plan.output_dir)
            / f"{plan.output_name}.{run_id}.asteria-effective-config.json"
        )
        try:
            config_arg_index = plan.command_args.index("--effective-config-output") + 1
            plan.command_args[config_arg_index] = immutable_config_path
        except (ValueError, IndexError):
            plan.errors.append("copy_machine_effective_config_output_missing")
            plan.status = "blocked"
        plan.effective_config_path = immutable_config_path
    run = LoRACopyMachineTrainingRun(
        id=run_id,
        session_id=plan.session_id,
        adapter_id=plan.adapter_id,
        method=plan.method,
        status=plan.status,
        output_dir=plan.output_dir,
        output_name=plan.output_name,
        effective_config_path=plan.effective_config_path,
        command_args=plan.command_args,
        warnings=plan.warnings,
        errors=plan.errors,
        delegated_training_run_id=delegated_run.id if delegated_run is not None else "",
        created_at=now,
        updated_at=now,
    )
    return run_repository.create(run)


def start_lora_copy_machine_training_run(
    run_id: str,
    repository: "LoRACopyMachineTrainingRunRepository",
    *,
    data_dir: Path,
    project_root: Path,
    training_run_repository: "LoRATrainingRunRepository | None" = None,
) -> LoRACopyMachineTrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if run.status == "blocked":
        raise ValueError("copy_machine_training_run_blocked")
    if run.status not in {"planned", "failed"}:
        raise ValueError("copy_machine_training_run_not_startable")
    if run.delegated_training_run_id:
        if training_run_repository is None:
            raise ValueError("copy_machine_training_run_repository_required")
        delegated = training_run_repository.get(run.delegated_training_run_id)
        if delegated is None:
            _fail_copy_machine_delegated_run(run, "copy_machine_training_process_lost")
            return repository.save(run)
        try:
            delegated = start_lora_training_run(
                delegated.id,
                training_run_repository,
                data_dir=data_dir,
                project_root=project_root,
            )
        except ValueError as exc:
            delegated = training_run_repository.get(run.delegated_training_run_id)
            if delegated is not None:
                _sync_copy_machine_delegated_run(run, delegated)
            code = (
                "copy_machine_training_command_start_failed"
                if str(exc) == "lora_training_command_start_failed"
                else str(exc)
            )
            _append_unique_issue(run.errors, code)
            run.status = "failed"
            run.updated_at = _utc_now()
            return repository.save(run)
        if delegated is None:
            _fail_copy_machine_delegated_run(run, "copy_machine_training_process_lost")
        else:
            _sync_copy_machine_delegated_run(run, delegated)
        return repository.save(run)
    if not run.command_args:
        raise ValueError("copy_machine_training_run_command_empty")
    output_dir = _resolve_source_path(run.output_dir, project_root)
    log_dir = data_dir / "lora-training" / "copy-machine-logs"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    log_path = log_dir / f"{run.id}.log"
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Asteria copy-machine LoRA run {run.id}\n")
            log_file.write(f"Command: {' '.join(run.command_args)}\n\n")
            log_file.flush()
            try:
                process = subprocess.Popen(
                    run.command_args,
                    cwd=str(project_root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                run.status = "failed"
                run.log_path = str(log_path)
                run.errors = [*run.errors, "copy_machine_training_command_start_failed"]
                log_file.write(f"Asteria failed to start the copy-machine LoRA command: {exc}\n")
                run.updated_at = _utc_now()
                repository.save(run)
                return run
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    _COPY_MACHINE_RUN_PROCESSES[run.id] = process
    run.status = "running"
    run.log_path = str(log_path)
    run.pid = process.pid
    run.exit_code = None
    run.started_at = _utc_now()
    run.finished_at = None
    run.updated_at = _utc_now()
    return repository.save(run)


def refresh_lora_copy_machine_training_run(
    run_id: str,
    repository: "LoRACopyMachineTrainingRunRepository",
    *,
    training_run_repository: "LoRATrainingRunRepository | None" = None,
    project_root: Path | None = None,
) -> LoRACopyMachineTrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if _ensure_copy_machine_effective_config_immutable(run):
        repository.save(run)
    if run.delegated_training_run_id:
        if training_run_repository is None:
            raise ValueError("copy_machine_training_run_repository_required")
        delegated = training_run_repository.get(run.delegated_training_run_id)
        if delegated is None:
            _fail_copy_machine_delegated_run(run, "copy_machine_training_process_lost")
            return repository.save(run)
        delegated = refresh_lora_training_run(
            delegated.id,
            training_run_repository,
            project_root=project_root,
        )
        if delegated is None:
            _fail_copy_machine_delegated_run(run, "copy_machine_training_process_lost")
        else:
            _sync_copy_machine_delegated_run(run, delegated)
        return repository.save(run)
    process = _COPY_MACHINE_RUN_PROCESSES.get(run.id)
    if process is None:
        if run.effective_config_path and run.effective_config_evidence is None:
            run.effective_config_evidence = _copy_machine_file_evidence(Path(run.effective_config_path))
        if run.status == "completed" and _copy_machine_training_log_has_nonfinite_loss(run.log_path):
            _fail_copy_machine_nonfinite_run(run)
            repository.save(run)
        elif run.status == "running":
            run.status = "failed"
            run.errors = [*run.errors, "copy_machine_training_process_lost"]
            run.finished_at = _utc_now()
            run.updated_at = _utc_now()
            repository.save(run)
        return run
    exit_code = process.poll()
    if exit_code is None:
        return run
    _COPY_MACHINE_RUN_PROCESSES.pop(run.id, None)
    run.exit_code = exit_code
    run.trained_model_paths = _discover_copy_machine_trained_models(run)
    run.trained_model_evidence = _copy_machine_model_evidence(run.trained_model_paths)
    if run.effective_config_path:
        run.effective_config_evidence = _copy_machine_file_evidence(Path(run.effective_config_path))
    nonfinite_loss = _copy_machine_training_log_has_nonfinite_loss(run.log_path)
    if exit_code == 0 and run.trained_model_paths and not nonfinite_loss:
        run.status = "completed"
    else:
        run.status = "failed"
    if exit_code != 0:
        run.errors = [*run.errors, "copy_machine_training_command_failed"]
    elif not run.trained_model_paths and "copy_machine_training_no_model_output" not in run.errors:
        run.errors = [*run.errors, "copy_machine_training_no_model_output"]
    elif nonfinite_loss and "copy_machine_training_nonfinite_loss" not in run.errors:
        run.errors = [*run.errors, "copy_machine_training_nonfinite_loss"]
    run.finished_at = _utc_now()
    run.updated_at = _utc_now()
    return repository.save(run)


def register_lora_copy_machine_training_run_models(
    run_id: str,
    repository: "LoRACopyMachineTrainingRunRepository",
    *,
    model_root: Path,
) -> LoRACopyMachineTrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if run.status == "completed" and _copy_machine_training_log_has_nonfinite_loss(run.log_path):
        _fail_copy_machine_nonfinite_run(run)
        repository.save(run)
    if run.status != "completed":
        raise ValueError("copy_machine_training_run_not_completed")
    sources = [Path(path) for path in run.trained_model_paths if _path_is_file(Path(path))]
    if not sources:
        raise ValueError("copy_machine_no_trained_models_to_register")
    target_dir = model_root / "loras" / "trained"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    registered: list[str] = []
    for source in sources:
        target = target_dir / source.name
        if not _new_file_candidate_is_available(target):
            target = target_dir / f"{source.stem}-{run.id[:8]}{source.suffix}"
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
        registered.append(str(target))
    run.registered_model_paths = registered
    run.updated_at = _utc_now()
    return repository.save(run)


def prepare_lora_copy_machine_comparison(
    data: LoRACopyMachineComparisonPrepareData,
    repository: "LoRACopyMachineTrainingRunRepository",
) -> LoRACopyMachineComparisonPlan | None:
    run = repository.get(data.run_id)
    if run is None:
        return None
    plan = LoRACopyMachineComparisonPlan(run_id=run.id, status="blocked")
    if run.status != "completed":
        plan.errors.append("copy_machine_training_run_not_completed")
        return plan
    registered = next((path for path in run.registered_model_paths if _path_is_file(Path(path))), "")
    if not registered:
        plan.errors.append("copy_machine_registered_model_required")
        return plan
    plan.status = "planned"
    plan.generation_requests = [
        LoRACopyMachineComparisonRequest(
            label="base",
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            seed=data.seed,
            width=data.width,
            height=data.height,
        ),
        LoRACopyMachineComparisonRequest(
            label="copy_machine_positive",
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            seed=data.seed,
            width=data.width,
            height=data.height,
            loras=[
                LoRACopyMachineComparisonLoRA(
                    path=registered,
                    strength=data.positive_strength,
                )
            ],
        ),
    ]
    return plan


def build_dataset_config_preview(request: DatasetPreviewRequestData) -> DatasetConfigPreview:
    preset = get_lora_training_preset(request.preset_id)
    if preset is None:
        raise ValueError("lora_training_preset_not_found")
    repeats = request.repeats if request.repeats is not None else preset.repeats
    caption_extension = _safe_caption_extension(request.caption_extension or preset.caption_extension)
    subset = DatasetSubset(
        image_dir=request.image_dir,
        num_repeats=repeats,
        caption_extension=caption_extension,
        class_tokens=request.class_tokens,
        keep_tokens=preset.keep_tokens,
        shuffle_caption=False if preset.profile == "anima" else preset.shuffle_caption,
    )
    toml = _render_dataset_toml(preset, subset)
    warnings = _preview_warnings(request, preset)
    return DatasetConfigPreview(
        preset=preset,
        toml=toml,
        command_args=_build_command_args(request, preset),
        warnings=warnings,
    )


def build_lora_training_character_caption_gate(
    dataset: LoRATrainingDataset,
    items: list[MaterializedDatasetItem],
    *,
    effective_keep_tokens: int,
) -> LoRATrainingCharacterCaptionGate:
    if dataset.training_type != "character_lora":
        return LoRATrainingCharacterCaptionGate(
            applicable=False,
            passed=True,
            total_count=len(items),
            effective_keep_tokens=effective_keep_tokens,
            keep_tokens_ready=True,
        )

    policy = _normalized_character_caption_policy(dataset.character_caption_policy)
    trigger_key = policy.trigger_token.casefold()
    excluded_by_key = {tag.casefold(): tag for tag in policy.excluded_identity_tags}
    trigger_first_count = 0
    identity_hit_count = 0
    missing_trigger_item_ids: list[str] = []
    misplaced_trigger_item_ids: list[str] = []
    duplicate_trigger_item_ids: list[str] = []
    identity_hit_items: list[LoRATrainingCharacterCaptionIdentityHit] = []

    for item in items:
        caption = ""
        caption_path = Path(item.caption_path)
        if _path_is_file(caption_path):
            try:
                caption = caption_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                caption = ""
        tokens = [" ".join(token.strip().split()) for token in caption.split(",") if token.strip()]
        token_keys = [token.casefold() for token in tokens]
        trigger_indexes = [index for index, key in enumerate(token_keys) if trigger_key and key == trigger_key]
        if trigger_indexes and trigger_indexes[0] == 0:
            trigger_first_count += 1
        if not trigger_indexes:
            missing_trigger_item_ids.append(item.item_id)
        elif trigger_indexes[0] != 0:
            misplaced_trigger_item_ids.append(item.item_id)
        if len(trigger_indexes) > 1:
            duplicate_trigger_item_ids.append(item.item_id)
        hits: list[str] = []
        for key in token_keys:
            tag = excluded_by_key.get(key)
            if tag and tag not in hits:
                hits.append(tag)
        if hits:
            identity_hit_count += len(hits)
            identity_hit_items.append(
                LoRATrainingCharacterCaptionIdentityHit(item_id=item.item_id, tags=hits)
            )

    total_count = len(items)
    keep_tokens_ready = effective_keep_tokens >= 1
    reasons: list[str] = []
    if not trigger_key:
        reasons.append("character_caption_policy_trigger_token_required")
    if not keep_tokens_ready:
        reasons.append("character_caption_keep_tokens_required")
    if trigger_first_count != total_count:
        reasons.append("character_caption_trigger_first_required")
    if duplicate_trigger_item_ids:
        reasons.append("character_caption_duplicate_trigger")
    if identity_hit_count:
        reasons.append("character_caption_identity_tags_present")
    return LoRATrainingCharacterCaptionGate(
        applicable=True,
        passed=not reasons,
        total_count=total_count,
        trigger_first_count=trigger_first_count,
        trigger_first_rate=(trigger_first_count / total_count) if total_count else 0.0,
        missing_trigger_item_ids=missing_trigger_item_ids,
        misplaced_trigger_item_ids=misplaced_trigger_item_ids,
        duplicate_trigger_item_ids=duplicate_trigger_item_ids,
        identity_hit_count=identity_hit_count,
        identity_hit_items=identity_hit_items,
        effective_keep_tokens=effective_keep_tokens,
        keep_tokens_ready=keep_tokens_ready,
        reasons=reasons,
    )


def materialize_lora_training_dataset(
    dataset: LoRATrainingDataset,
    request: DatasetMaterializeRequestData,
) -> MaterializedDataset:
    preset = get_lora_training_preset(dataset.preset_id)
    if preset is None:
        raise ValueError("lora_training_preset_not_found")
    if not dataset.items:
        raise ValueError("lora_training_dataset_empty")

    materialized_root = (request.data_dir / "lora-training" / "materialized").resolve()
    root_dir = (materialized_root / f"{_safe_slug(dataset.name)}-{dataset.id[:8]}").resolve()
    if root_dir.parent != materialized_root:
        raise ValueError("lora_training_materialize_path_invalid")
    image_dir = root_dir / "images"

    try:
        if root_dir.exists():
            shutil.rmtree(root_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    caption_extension = _safe_caption_extension(request.caption_extension or preset.caption_extension)
    items: list[MaterializedDatasetItem] = []
    skipped_items: list[str] = []
    materialize_warnings: list[str] = []
    for index, item in enumerate(dataset.items, start=1):
        source_path = _resolve_source_path(item.image_path, request.project_root)
        if not _path_is_file(source_path):
            skipped_items.append(item.id)
            continue
        suffix = source_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            skipped_items.append(item.id)
            continue
        stem = f"{index:04d}_{item.id[:8]}_{_safe_slug(source_path.stem)}"
        image_path = image_dir / f"{stem}{suffix}"
        caption_path = image_dir / f"{stem}{caption_extension}"
        metadata_path = image_dir / f"{stem}.caption.json"
        fallback_tags = _exportable_caption_tags(item.tags, settings=dataset.caption_export_settings)
        fallback_tags = _category_prefixed_tags(fallback_tags, "manual", dataset.caption_export_settings)
        if not item.caption.strip() and len(fallback_tags) < len(_normalize_tags(item.tags)):
            materialize_warnings.append(f"caption_non_exportable_tags_filtered:{item.id}")
        caption = item.caption.strip() or ", ".join(fallback_tags)
        try:
            shutil.copy2(source_path, image_path)
            if dataset.caption_export_settings.output_mode in {"caption_txt", "caption_txt_and_metadata_json"}:
                caption_path.write_text(caption, encoding="utf-8")
            else:
                materialize_warnings.append(f"caption_txt_not_written_metadata_only:{item.id}")
            if dataset.caption_export_settings.output_mode in {"caption_txt_and_metadata_json", "metadata_json"}:
                metadata_path.write_text(
                    json.dumps(
                        {
                            "item_id": item.id,
                            "source_path": str(source_path),
                            "image_path": str(image_path),
                            "caption": caption,
                            "tags": fallback_tags,
                            "tag_categories": {
                                "manual": _exportable_caption_tags(
                                    item.tags,
                                    settings=dataset.caption_export_settings,
                                ),
                                "suppressed": dataset.caption_export_settings.suppressed_tags,
                                "organizational": dataset.caption_export_settings.organizational_tags,
                            },
                            "export_settings": dataset.caption_export_settings.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
        items.append(
            MaterializedDatasetItem(
                item_id=item.id,
                source_path=str(source_path),
                image_path=str(image_path),
                caption_path=str(caption_path),
                metadata_path=str(metadata_path) if _path_exists(metadata_path) else "",
                caption=caption,
            )
        )

    if not items:
        raise ValueError("lora_training_dataset_has_no_materializable_images")

    preview = build_dataset_config_preview(
        DatasetPreviewRequestData(
            preset_id=dataset.preset_id,
            image_dir=str(image_dir),
            output_dir=_project_path_for_command(request.output_dir, request.project_root, force=True),
            output_name=request.output_name,
            pretrained_model_name_or_path=_project_path_for_command(
                request.pretrained_model_name_or_path, request.project_root
            ),
            vae=_project_path_for_command(request.vae, request.project_root),
            qwen3=_project_path_for_command(request.qwen3, request.project_root),
            repeats=request.repeats,
            max_train_epochs=request.max_train_epochs,
            class_tokens=request.class_tokens,
            caption_extension=caption_extension,
            optimizer_type=request.optimizer_type,
            optimizer_args=request.optimizer_args,
            optimizer_low_vram=request.optimizer_low_vram,
        )
    )
    dataset_config_path = root_dir / "dataset.toml"
    sample_prompts_path = root_dir / "sample_prompts.txt"
    try:
        dataset_config_path.write_text(preview.toml, encoding="utf-8")
        sample_prompts_path.write_text(_render_sample_prompts(dataset, preset), encoding="utf-8")
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    command_args = [
        str(dataset_config_path) if value == "<dataset.toml>" else value
        for value in preview.command_args
    ]
    command_args = _with_sample_prompts(command_args, sample_prompts_path, preset)
    character_caption_gate = build_lora_training_character_caption_gate(
        dataset,
        items,
        effective_keep_tokens=preset.keep_tokens,
    )
    return MaterializedDataset(
        dataset=dataset,
        root_dir=str(root_dir),
        image_dir=str(image_dir),
        dataset_config_path=str(dataset_config_path),
        sample_prompts_path=str(sample_prompts_path),
        item_count=len(items),
        items=items,
        skipped_items=skipped_items,
        command_args=command_args,
        warnings=[*preview.warnings, *materialize_warnings],
        character_caption_gate=character_caption_gate,
    )


def prepare_lora_training_run(
    dataset: LoRATrainingDataset,
    request: TrainingRunPrepareRequestData,
    repository: "LoRATrainingRunRepository",
) -> LoRATrainingRun:
    materialized = materialize_lora_training_dataset(
        dataset,
        DatasetMaterializeRequestData(
            data_dir=request.data_dir,
            project_root=request.project_root,
            output_dir=request.output_dir,
            output_name=request.output_name,
            pretrained_model_name_or_path=request.pretrained_model_name_or_path,
            vae=request.vae,
            qwen3=request.qwen3,
            repeats=request.repeats,
            max_train_epochs=request.max_train_epochs,
            class_tokens=request.class_tokens,
            caption_extension=request.caption_extension,
            optimizer_type=request.optimizer_type,
            optimizer_args=request.optimizer_args,
            optimizer_low_vram=request.optimizer_low_vram,
        ),
    )
    preset = get_lora_training_preset(dataset.preset_id)
    if preset is None:
        raise ValueError("lora_training_preset_not_found")

    command_args = _with_accelerate_command(materialized.command_args, request.accelerate_command, request.project_root)
    preflight = _build_run_preflight(
        preset=preset,
        request=request,
        dataset_config_path=Path(materialized.dataset_config_path),
        character_caption_gate=materialized.character_caption_gate,
    )
    return repository.create(
        dataset=dataset,
        materialized=materialized,
        output_dir=request.output_dir,
        output_name=request.output_name,
        working_dir=request.working_dir,
        command_args=command_args,
        preflight=preflight,
    )


def start_lora_training_run(
    run_id: str,
    repository: "LoRATrainingRunRepository",
    *,
    data_dir: Path,
    project_root: Path,
) -> LoRATrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if run.status == "running":
        return refresh_lora_training_run(run_id, repository, project_root=project_root)
    if not run.preflight.ready:
        raise ValueError("lora_training_run_preflight_not_ready")
    if run.status not in {"planned", "failed", "cancelled"}:
        raise ValueError("lora_training_run_not_startable")
    if not run.command_args:
        raise ValueError("lora_training_run_command_empty")

    log_dir = data_dir / "lora-training" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    log_path = (log_dir / f"{run.id}.log").resolve()
    working_dir = _resolve_training_working_dir(run.working_dir, project_root)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Asteria LoRA training run {run.id}\n")
            log_file.write(f"Command: {' '.join(run.command_args)}\n\n")
            log_file.flush()
            try:
                process = subprocess.Popen(
                    run.command_args,
                    cwd=str(working_dir),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                now = _utc_now()
                run.status = "failed"
                run.exit_code = -1
                run.log_path = str(log_path)
                run.log_tail = _append_log_tail_note(str(log_path), f"Asteria failed to start the LoRA training command: {exc}")
                run.started_at = now
                run.finished_at = now
                run.updated_at = now
                repository.save(run)
                raise ValueError("lora_training_command_start_failed") from exc
    except LoRATrainingStorageError:
        raise
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
    _RUN_PROCESSES[run.id] = process
    now = _utc_now()
    run.status = "running"
    run.pid = process.pid
    run.exit_code = None
    run.log_path = str(log_path)
    run.log_tail = _read_text_tail(log_path)
    run.trained_model_paths = []
    run.sample_image_paths = []
    run.sample_image_scores = []
    run.registered_model_paths = []
    _apply_run_progress_from_log(run)
    run.started_at = now
    run.finished_at = None
    run.updated_at = now
    repository.save(run)
    return run


def refresh_lora_training_run(
    run_id: str,
    repository: "LoRATrainingRunRepository",
    *,
    project_root: Path | None = None,
) -> LoRATrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    process = _RUN_PROCESSES.get(run.id)
    if process is not None:
        exit_code = process.poll()
        if exit_code is not None:
            _RUN_PROCESSES.pop(run.id, None)
            run.exit_code = int(exit_code)
            run.status = "completed" if exit_code == 0 else "failed"
            run.finished_at = _utc_now()
    elif run.status == "running":
        run.status = "failed"
        run.exit_code = run.exit_code if run.exit_code is not None else -1
        run.finished_at = _utc_now()
        run.log_tail = _append_log_tail_note(run.log_path, "Asteria lost the subprocess handle before completion.")
    if run.log_path:
        run.log_tail = _read_text_tail(Path(run.log_path))
    _apply_run_progress_from_log(run)
    if run.status == "completed":
        _finalize_anima_training_output(run, project_root=project_root)
    if run.status == "completed":
        _apply_run_trained_model_paths(run, project_root=project_root)
        _apply_run_sample_image_paths(run, project_root=project_root)
        _apply_run_sample_image_scores(run, project_root=project_root)
    run.updated_at = _utc_now()
    repository.save(run)
    return run


def score_lora_training_run_samples(
    run_id: str,
    repository: "LoRATrainingRunRepository",
    *,
    project_root: Path | None = None,
) -> LoRATrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if run.status != "completed":
        raise ValueError("lora_training_run_not_completed")
    _apply_run_sample_image_paths(run, project_root=project_root)
    _apply_run_sample_image_scores(run, project_root=project_root)
    run.updated_at = _utc_now()
    repository.save(run)
    return run


def score_lora_training_sample_image(path: Path) -> LoRATrainingSampleImageScore:
    return _score_lora_training_sample_image(path)


def cancel_lora_training_run(
    run_id: str,
    repository: "LoRATrainingRunRepository",
) -> LoRATrainingRun | None:
    run = repository.get(run_id)
    if run is None:
        return None
    process = _RUN_PROCESSES.pop(run.id, None)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    run.status = "cancelled"
    run.exit_code = process.returncode if process is not None else run.exit_code
    run.finished_at = _utc_now()
    if run.log_path:
        run.log_tail = _append_log_tail_note(run.log_path, "Asteria cancelled the LoRA training run.")
    _apply_run_progress_from_log(run)
    run.updated_at = _utc_now()
    repository.save(run)
    return run


def register_lora_training_run_models(
    run_id: str,
    repository: "LoRATrainingRunRepository",
    *,
    model_root: Path,
    project_root: Path | None = None,
) -> LoRATrainingRunModelRegistration | None:
    run = repository.get(run_id)
    if run is None:
        return None
    if run.status != "completed":
        raise ValueError("lora_training_run_not_completed")
    root = model_root.resolve()
    target_dir = (root / "loras" / "trained").resolve()
    if not _path_is_relative_to(target_dir, root):
        raise ValueError("lora_training_register_target_invalid")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    source_paths = [path for path in run.trained_model_paths if path.strip()]
    if not source_paths:
        _apply_run_trained_model_paths(run, project_root=project_root)
        source_paths = [path for path in run.trained_model_paths if path.strip()]
    if not source_paths:
        raise ValueError("lora_training_no_trained_models_to_register")

    installed_paths: list[str] = []
    warnings: list[str] = []
    root_loras = (root / "loras").resolve()
    source_root = project_root or Path.cwd()
    for source_text in source_paths:
        source = _resolve_source_path(source_text, source_root)
        if not _path_is_file(source) or source.suffix.lower() != ".safetensors":
            warnings.append(f"skipped_missing_or_unsupported:{source_text}")
            continue
        if _path_is_relative_to(source, root_loras):
            installed = source.resolve()
        else:
            installed = _unique_registered_lora_path(target_dir, source.name, run.id)
            try:
                shutil.copy2(source, installed)
            except OSError as exc:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
        installed_paths.append(str(installed))

    installed_paths = sorted(set(installed_paths), key=lambda value: value.lower())
    if not installed_paths:
        raise ValueError("lora_training_no_trained_models_to_register")
    run.registered_model_paths = installed_paths
    run.updated_at = _utc_now()
    repository.save(run)
    return LoRATrainingRunModelRegistration(run=run, installed_model_paths=installed_paths, warnings=warnings)


def import_lora_training_dataset_directory(
    dataset_id: str,
    repository: "LoRATrainingDatasetRepository",
    *,
    directory_path: str,
    project_root: Path,
    recursive: bool = False,
    caption_extension: str = ".txt",
    default_tags: list[str] | None = None,
) -> LoRATrainingDatasetDirectoryImportResult | None:
    dataset = repository.get(dataset_id)
    if dataset is None:
        return None
    directory = _resolve_source_path(directory_path, project_root)
    if not _path_is_dir(directory):
        raise ValueError("lora_training_import_directory_not_found")
    caption_suffix = _safe_caption_extension(caption_extension)
    existing_paths = {str(_resolve_source_path(item.image_path, project_root)) for item in dataset.items}
    tags = _normalize_tags(default_tags or [])
    image_inputs: list[LoRATrainingDatasetImageInput] = []
    duplicate_count = 0
    skipped_count = 0
    sidecar_caption_count = 0
    warnings: list[str] = []
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    for path in sorted(iterator, key=lambda candidate: str(candidate).lower()):
        try:
            path_is_file = path.is_file()
        except OSError:
            skipped_count += 1
            warnings.append(f"skipped_inaccessible_image:{path}")
            continue
        if not path_is_file:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            skipped_count += 1
            warnings.append(f"skipped_inaccessible_image:{path}")
            continue
        if path.suffix.lower() == caption_suffix.lower():
            continue
        if path.suffix.lower() not in SAMPLE_IMAGE_EXTENSIONS:
            skipped_count += 1
            continue
        if not _is_valid_image_file(resolved):
            skipped_count += 1
            warnings.append(f"skipped_invalid_image:{resolved}")
            continue
        if str(resolved) in existing_paths:
            duplicate_count += 1
            continue
        caption = ""
        sidecar = path.with_suffix(caption_suffix)
        if _path_is_file(sidecar):
            try:
                caption = sidecar.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                try:
                    sidecar_label = str(sidecar.resolve())
                except OSError:
                    sidecar_label = str(sidecar)
                warnings.append(f"caption_sidecar_unreadable:{sidecar_label}")
            if caption:
                sidecar_caption_count += 1
        image_inputs.append(
            LoRATrainingDatasetImageInput(
                image_path=str(resolved),
                caption=caption,
                tags=tags,
            )
        )
        existing_paths.add(str(resolved))
    updated = repository.add_images(dataset_id, image_inputs) if image_inputs else dataset
    if updated is None:
        return None
    return LoRATrainingDatasetDirectoryImportResult(
        dataset=updated,
        imported_count=len(image_inputs),
        skipped_count=skipped_count,
        duplicate_count=duplicate_count,
        sidecar_caption_count=sidecar_caption_count,
        warnings=warnings,
    )


def import_lora_training_caption_sources_from_directory(
    dataset_id: str,
    repository: "LoRATrainingDatasetRepository",
    *,
    directory_path: str,
    project_root: Path,
    source_type: LoRATrainingCaptionSourceType,
    source_extension: str = ".txt",
    content_kind: LoRATrainingCaptionSourceContentKind = "tags",
    label: str = "",
    confidence: float | None = None,
    recursive: bool = False,
    replace_existing: bool = True,
) -> LoRATrainingCaptionSourceImportDirectoryResult | None:
    dataset = repository.get(dataset_id)
    if dataset is None:
        return None
    directory = _resolve_source_path(directory_path, project_root)
    if not _path_is_dir(directory):
        raise ValueError("lora_training_caption_source_import_directory_not_found")
    extension = _safe_caption_source_extension(source_extension)
    source_files = _caption_source_sidecar_index(directory, extension, recursive=recursive)
    warnings: list[str] = []
    imported_count = 0
    missing_count = 0
    skipped_count = 0
    updated = dataset
    for item in dataset.items:
        item_key = Path(item.image_path).stem.casefold()
        source_path = source_files.get(item_key)
        if source_path is None:
            missing_count += 1
            continue
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            skipped_count += 1
            warnings.append(f"caption_source_unreadable:{source_path}")
            continue
        parsed = _caption_source_input_from_imported_text(
            text,
            source_type=source_type,
            content_kind=content_kind,
            label=label or source_path.name,
            confidence=confidence,
        )
        if parsed is None:
            skipped_count += 1
            warnings.append(f"caption_source_empty:{source_path}")
            continue
        source_id = _caption_source_import_id(item.id, source_type, label, extension) if replace_existing else None
        updated = repository.upsert_caption_source(
            dataset_id,
            item.id,
            parsed.model_copy(update={"id": source_id}),
        )
        if updated is None:
            return None
        imported_count += 1
    return LoRATrainingCaptionSourceImportDirectoryResult(
        dataset=updated,
        source_type=source_type,
        imported_count=imported_count,
        missing_count=missing_count,
        skipped_count=skipped_count,
        scanned_count=len(source_files),
        warnings=warnings,
    )


def refresh_lora_training_caption_sources(
    dataset_id: str,
    repository: "LoRATrainingDatasetRepository",
    *,
    directory_path: str,
    project_root: Path,
    commands: list[str],
    source_type: LoRATrainingCaptionSourceType,
    source_extension: str = ".txt",
    content_kind: LoRATrainingCaptionSourceContentKind = "tags",
    label: str = "",
    confidence: float | None = None,
    recursive: bool = False,
    replace_existing: bool = True,
    working_dir: str = "",
    timeout_seconds: float = 600.0,
) -> LoRATrainingCaptionSourceRefreshResult | None:
    dataset = repository.get(dataset_id)
    if dataset is None:
        return None
    directory = _resolve_source_path(directory_path, project_root)
    if not _path_is_dir(directory):
        raise ValueError("lora_training_caption_source_import_directory_not_found")
    extension = _safe_caption_source_extension(source_extension)
    command_lines = [command.strip() for command in commands if command.strip()]
    if not command_lines:
        raise ValueError("lora_training_caption_source_refresh_command_required")
    cwd = _resolve_refresh_working_dir(working_dir, project_root)
    placeholders = {
        "source_dir": str(directory),
        "dataset_dir": str(directory),
        "project_root": str(project_root.resolve()),
        "source_extension": extension,
        "source_type": source_type,
        "content_kind": content_kind,
        "label": label,
        "confidence": "" if confidence is None else str(confidence),
    }
    command_results: list[LoRATrainingCaptionSourceRefreshCommandResult] = []
    for command in command_lines:
        resolved_command = _format_caption_source_refresh_command(command, placeholders)
        command_args = _external_command_parts(resolved_command, project_root)
        if not command_args:
            raise ValueError("lora_training_caption_source_refresh_command_required")
        try:
            completed = subprocess.run(
                command_args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=max(1.0, min(3600.0, timeout_seconds)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            command_results.append(
                LoRATrainingCaptionSourceRefreshCommandResult(
                    command=resolved_command,
                    command_args=command_args,
                    exit_code=-1,
                    stdout_tail=_tail_text(exc.stdout or ""),
                    stderr_tail=_tail_text(exc.stderr or "caption_source_refresh_command_timeout"),
                )
            )
            raise ValueError("lora_training_caption_source_refresh_command_timeout") from exc
        except OSError as exc:
            command_results.append(
                LoRATrainingCaptionSourceRefreshCommandResult(
                    command=resolved_command,
                    command_args=command_args,
                    exit_code=-1,
                    stderr_tail=str(exc),
                )
            )
            raise ValueError("lora_training_caption_source_refresh_command_failed") from exc
        command_results.append(
            LoRATrainingCaptionSourceRefreshCommandResult(
                command=resolved_command,
                command_args=command_args,
                exit_code=int(completed.returncode),
                stdout_tail=_tail_text(completed.stdout),
                stderr_tail=_tail_text(completed.stderr),
            )
        )
        if completed.returncode != 0:
            raise ValueError("lora_training_caption_source_refresh_command_failed")

    imported = import_lora_training_caption_sources_from_directory(
        dataset_id,
        repository,
        directory_path=str(directory),
        project_root=project_root,
        source_type=source_type,
        source_extension=extension,
        content_kind=content_kind,
        label=label,
        confidence=confidence,
        recursive=recursive,
        replace_existing=replace_existing,
    )
    if imported is None:
        return None
    return LoRATrainingCaptionSourceRefreshResult(
        dataset=imported.dataset,
        source_type=imported.source_type,
        imported_count=imported.imported_count,
        missing_count=imported.missing_count,
        skipped_count=imported.skipped_count,
        scanned_count=imported.scanned_count,
        commands=command_results,
        warnings=imported.warnings,
    )


def export_lora_training_anima_tagger_ron_sidecars(
    dataset_id: str,
    repository: "LoRATrainingDatasetRepository",
    *,
    directory_path: str,
    project_root: Path,
    overwrite: bool = True,
) -> LoRATrainingAnimaTaggerRonExportResult | None:
    dataset = repository.get(dataset_id)
    if dataset is None:
        return None
    directory = _resolve_source_path(directory_path, project_root)
    if not _path_is_dir(directory):
        raise ValueError("lora_training_caption_source_import_directory_not_found")
    written_paths: list[str] = []
    warnings: list[str] = []
    skipped_count = 0
    existing_count = 0
    timestamp = _utc_now().replace("+00:00", "Z")
    for item in dataset.items:
        stem = Path(item.image_path).stem.strip()
        if not stem:
            skipped_count += 1
            warnings.append(f"anima_tagger_ron_missing_stem:{item.id}")
            continue
        sidecar_path = (directory / f"{stem}.ron").resolve()
        if not overwrite:
            try:
                sidecar_exists = sidecar_path.exists()
            except OSError:
                skipped_count += 1
                warnings.append(f"anima_tagger_ron_exists_check_failed:{sidecar_path}")
                continue
            if sidecar_exists:
                existing_count += 1
                continue
        text = _anima_tagger_ron_text_for_dataset_item(dataset, item, timestamp=timestamp)
        try:
            sidecar_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            skipped_count += 1
            warnings.append(f"anima_tagger_ron_write_failed:{sidecar_path}:{exc}")
            continue
        written_paths.append(str(sidecar_path))
    return LoRATrainingAnimaTaggerRonExportResult(
        dataset=dataset,
        directory_path=str(directory),
        written_count=len(written_paths),
        skipped_count=skipped_count,
        existing_count=existing_count,
        written_paths=written_paths,
        warnings=warnings,
    )


def analyze_lora_training_dataset_item(
    dataset: LoRATrainingDataset,
    item_id: str,
    *,
    project_root: Path,
) -> LoRATrainingDatasetItemAnalysis | None:
    item = next((entry for entry in dataset.items if entry.id == item_id), None)
    if item is None:
        return None
    image_path = _resolve_source_path(item.image_path, project_root)
    if not _path_is_file(image_path):
        raise ValueError("lora_training_dataset_image_not_found")
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("lora_training_dataset_image_invalid")
            average_rgb = image.resize((1, 1)).getpixel((0, 0))
    except OSError as exc:
        raise ValueError("lora_training_dataset_image_invalid") from exc
    orientation = _image_orientation(width, height)
    dominant_tone = _dominant_tone(average_rgb)
    metadata_tags, warnings = _dataset_item_metadata_tags(item, image_path, project_root)
    source_tags = _caption_source_record_tags(item)
    suggested_tags = _exportable_caption_tags([
        *item.tags,
        *metadata_tags,
        *source_tags,
        f"{orientation} orientation",
        dominant_tone,
    ])
    caption_candidates = _caption_candidate_tags(item.caption, suggested_tags)
    suggested_caption = ", ".join(caption_candidates[:18])
    if not suggested_caption:
        suggested_caption = f"{orientation} image, {dominant_tone}, {width}x{height}"
    return LoRATrainingDatasetItemAnalysis(
        item_id=item.id,
        image_path=str(image_path),
        width=width,
        height=height,
        aspect_ratio=round(width / height, 4),
        orientation=orientation,
        average_color=f"#{average_rgb[0]:02x}{average_rgb[1]:02x}{average_rgb[2]:02x}",
        dominant_tone=dominant_tone,
        suggested_caption=suggested_caption,
        suggested_tags=suggested_tags,
        metadata_tags=metadata_tags,
        source_tags=source_tags,
        warnings=warnings,
    )


def build_lora_training_caption_assist(
    dataset: LoRATrainingDataset,
    item_id: str,
    request: LoRATrainingCaptionAssistRequestData,
    *,
    project_root: Path,
) -> LoRATrainingCaptionAssist | None:
    analysis = analyze_lora_training_dataset_item(dataset, item_id, project_root=project_root)
    if analysis is None:
        return None
    max_tags = min(80, max(1, request.max_tags))
    item = next(entry for entry in dataset.items if entry.id == item_id)
    source_summary = _caption_source_summary(dataset, item, analysis)
    suggested_tags = analysis.suggested_tags[:max_tags]
    suggested_caption = ", ".join(_caption_candidate_tags(analysis.suggested_caption, suggested_tags)[:max_tags])
    if not suggested_caption:
        suggested_caption = analysis.suggested_caption
    system_prompt = (
        "You help prepare LoRA training captions for Asteria datasets. "
        "Keep LoRA trigger tokens and visual attributes stable. "
        "Return concise captions without inventing identity, brand, artist, or source claims that are not visible or provided. "
        "Treat manual edits, existing sidecars, WD14-family tagger output, vision captioners, and Danbooru/API tags as evidence sources; "
        "merge useful visible facts, suppress false positives, and never treat an automatic source as final without curation. "
        f"{_caption_assist_training_type_instruction(dataset.training_type)}"
        f"{_caption_assist_character_policy_instruction(dataset)}"
        f"{_caption_assist_profile_instruction(dataset.profile)}"
    )
    style_instruction = _caption_assist_output_instruction(dataset.profile, request.caption_style)
    lines = [
        f"Output language: {request.language}",
        f"Caption style: {request.caption_style}",
        f"Training type: {dataset.training_type}",
        style_instruction,
        f"Image: {analysis.width}x{analysis.height} {analysis.orientation}",
        f"Average color: {analysis.average_color}",
        f"Dominant tone: {analysis.dominant_tone}",
        f"Existing/generated caption: {analysis.suggested_caption}",
        f"Candidate tags: {', '.join(suggested_tags) if suggested_tags else '(none)'}",
        _caption_assist_source_instruction(),
        _caption_assist_coverage_instruction(dataset.training_type),
    ]
    if analysis.metadata_tags:
        lines.append(f"Metadata tags: {', '.join(analysis.metadata_tags)}")
    source_lines = _caption_source_prompt_lines(item)
    if source_lines:
        lines.append("Saved caption sources:")
        lines.extend(source_lines)
    if request.extra_instruction.strip():
        lines.append(f"Additional instruction: {request.extra_instruction.strip()}")
    lines.extend(
        [
            "Respond with:",
            "1. caption",
            "2. tags",
            "3. short warnings only when the provided evidence is insufficient",
        ]
    )
    warnings = [*analysis.warnings, *source_summary.warnings]
    if len(analysis.suggested_tags) > max_tags:
        warnings.append(f"limited_candidate_tags_to_{max_tags}")
    provider_response = ""
    provider_model = ""
    provider_mode = request.provider_mode
    provider_api_key = ""
    if provider_mode != "prompt_only":
        provider_api_key = _provider_api_key(request.provider_api_key, request.provider_api_key_env_var)
    if provider_mode == "openai_compatible":
        provider_response = _call_openai_compatible_caption_assist(
            endpoint=request.provider_endpoint,
            model=request.provider_model,
            api_key=provider_api_key,
            timeout_seconds=request.provider_timeout_seconds,
            system_prompt=system_prompt,
            user_prompt="\n".join(lines),
            image_path=Path(analysis.image_path) if request.include_image else None,
        )
        provider_model = request.provider_model.strip()
    elif provider_mode == "openai_responses":
        provider_response = _call_openai_responses_caption_assist(
            endpoint=request.provider_endpoint,
            model=request.provider_model,
            api_key=provider_api_key,
            timeout_seconds=request.provider_timeout_seconds,
            system_prompt=system_prompt,
            user_prompt="\n".join(lines),
        )
        provider_model = request.provider_model.strip()
    elif provider_mode == "anthropic":
        provider_response = _call_anthropic_caption_assist(
            endpoint=request.provider_endpoint,
            model=request.provider_model,
            api_key=provider_api_key,
            timeout_seconds=request.provider_timeout_seconds,
            system_prompt=system_prompt,
            user_prompt="\n".join(lines),
        )
        provider_model = request.provider_model.strip()
    elif provider_mode == "gemini":
        provider_response = _call_gemini_caption_assist(
            endpoint=request.provider_endpoint,
            api_key=provider_api_key,
            timeout_seconds=request.provider_timeout_seconds,
            system_prompt=system_prompt,
            user_prompt="\n".join(lines),
        )
        provider_model = request.provider_model.strip()
    if provider_response:
        parsed_caption, parsed_tags, parsed_warnings = _parse_caption_assist_provider_response(
            provider_response,
            fallback_caption=suggested_caption,
            fallback_tags=suggested_tags,
        )
        suggested_caption = parsed_caption
        suggested_tags = parsed_tags[:max_tags]
        warnings.extend(parsed_warnings)
    suggested_caption, suggested_tags = _apply_character_caption_policy_to_assist(
        dataset,
        suggested_caption,
        suggested_tags,
    )
    warnings.extend(_caption_assist_caption_quality_warnings(dataset.training_type, suggested_caption))
    return LoRATrainingCaptionAssist(
        item_id=analysis.item_id,
        provider_mode=provider_mode,
        language=request.language,
        caption_style=request.caption_style,
        system_prompt=system_prompt,
        user_prompt="\n".join(lines),
        suggested_caption=suggested_caption,
        suggested_tags=suggested_tags,
        analysis=analysis,
        source_summary=source_summary,
        provider_model=provider_model,
        provider_response=provider_response,
        warnings=warnings,
    )


def _caption_assist_training_type_instruction(training_type: TrainingType) -> str:
    if training_type == "character_lora":
        return (
            "Character LoRA captions must place the configured trigger token first, omit declared stable identity traits "
            "that the trigger should learn, and retain variable clothing, pose, expression, background, props, lighting, "
            "and camera terms. "
        )
    if training_type == "style_lora":
        return (
            "Style LoRA captions should prioritize line quality, color palette, texture, rendering, lighting, composition, "
            "and medium, but must still caption visible variable content such as subject type/count, clothing, accessories, "
            "pose, framing, background, props, and scene elements. Do not omit outfits or backgrounds just because this is "
            "a style LoRA; otherwise those repeated details can be learned as part of the style. Avoid character names and "
            "do not let one repeated subject become the style token."
        )
    return (
        "Copy LoRA captions should preserve the target object's visible shape, material, color, markings, and recurring "
        "details while avoiding unsupported background story."
    )


def _caption_assist_profile_instruction(profile: TrainingProfile) -> str:
    if profile == "anima":
        return (
            " For Anima, prefer Danbooru/Gelbooru-style tag tokens for visible entities and attributes, add short natural "
            "language only for situation, action, or scene relationships when tags are awkward, and prefer Gelbooru wording "
            "when Danbooru and Gelbooru usage differs. Tags should remain the primary caption vocabulary."
        )
    return (
        " Prefer Danbooru/Gelbooru-style tags for anime visual attributes when they are natural for the target model. "
        "For SDXL-style training, ordinary captions should be comma-separated tags rather than prose."
    )


def _caption_assist_output_instruction(profile: TrainingProfile, caption_style: LoRATrainingCaptionAssistStyle) -> str:
    if profile == "anima":
        if caption_style == "natural":
            return (
                "Return a tag-first caption: comma-separated Danbooru/Gelbooru-style tags for visible entities and attributes, "
                "with at most one short natural-language clause only for situation, action, or scene relationships that are awkward as tags."
            )
        return (
            "Return a comma-separated tag caption. For Anima, keep tags as the primary caption language; only include a very short "
            "natural-language fragment if a situation or relationship cannot be expressed cleanly as tags."
        )
    if caption_style == "natural":
        return (
            "Return a mostly comma-separated tag caption even when natural mode is requested; add one concise phrase only when needed "
            "for scene context that tags cannot express."
        )
    return "Return a comma-separated tag caption suitable for kohya/sd-scripts training."


def _caption_assist_source_instruction() -> str:
    return (
        "Source handling: use manual/user tags first, then same-stem sidecar or metadata tags, WD14-family auto tags, "
        "vision captioner facts, and Danbooru/API tags when available. Keep only visible or provided facts, remove false "
        "positive tags, avoid rating tags by default, and preserve review-only/organizational tags separately from exported captions. "
        "For normal dataset preparation, draft all images in bulk from WD14-family tags plus Qwen3-VL/vision captions, then let the user "
        "edit individual or batched captions before materializing training files."
    )


def _caption_assist_coverage_instruction(training_type: TrainingType) -> str:
    if training_type == "style_lora":
        return (
            "Coverage requirement: include both style descriptors and non-style visible variables. For style LoRA, a useful "
            "caption should normally include enough tags to separate line/color/rendering from clothing, accessories, pose, "
            "framing, background, props, and lighting. WD14-style taggers are useful for discovering these variable tags; "
            "filter rating tags unless the user explicitly asks for them."
        )
    if training_type == "character_lora":
        return (
            "Coverage requirement: keep the character trigger and stable identity traits clear, and only omit outfit, pose, "
            "background, or lighting when the user's dataset goal explicitly says those should be learned with the identity."
        )
    return (
        "Coverage requirement: caption visible object/material/marking variables separately from recurring presentation style."
    )


def _caption_assist_caption_quality_warnings(training_type: TrainingType, caption: str) -> list[str]:
    if training_type != "style_lora":
        return []
    tags = _split_caption_assist_tags(caption)
    if len(tags) < 10:
        return ["style_caption_too_sparse_for_training"]
    return []


_CAPTION_COVERAGE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "subject": (
        "1girl", "1boy", "girl", "boy", "woman", "man", "person", "character", "face", "eyes", "hair",
        "portrait",
    ),
    "clothing": (
        "dress", "skirt", "shirt", "blouse", "jacket", "coat", "hoodie", "kimono", "uniform", "lace",
        "frill", "sleeve", "outfit", "clothes", "wearing", "boots", "shoes", "gloves", "hat",
    ),
    "accessory_prop": (
        "ribbon", "bow", "flower", "bouquet", "jewelry", "earrings", "necklace", "hair ornament", "bag",
        "staff", "umbrella", "weapon", "book", "prop", "accessory",
    ),
    "pose_action": (
        "standing", "sitting", "walking", "running", "looking", "holding", "raising", "reaching", "pose",
        "smile", "arms", "hands", "turning", "lying", "dancing", "leaning",
    ),
    "framing": (
        "close-up", "close up", "upper body", "full body", "cowboy shot", "bust", "waist up", "wide shot",
        "from side", "profile", "looking at viewer", "centered", "composition",
    ),
    "background_scene": (
        "background", "garden", "room", "sky", "forest", "street", "city", "window", "flowers", "beach",
        "indoor", "outdoor", "scenery", "night", "field",
    ),
    "lighting": (
        "light", "lighting", "shadow", "backlit", "rim light", "sunlight", "glow", "luminous",
    ),
    "style_terms": (
        "lineart", "pastel", "shading", "rendering", "gradient", "illustration", "anime", "glossy",
        "texture", "delicate", "clean", "soft", "brush", "watercolor", "paint",
    ),
}

_STYLE_LORA_REQUIRED_VARIABLE_CATEGORIES = (
    "subject",
    "clothing",
    "pose_action",
    "framing",
    "background_scene",
)


def build_lora_training_dataset_caption_coverage(
    dataset: LoRATrainingDataset,
) -> LoRATrainingDatasetCaptionCoverageResult:
    items = [_caption_coverage_for_item(dataset, item) for item in dataset.items]
    warnings: list[str] = []
    if dataset.training_type == "style_lora" and any(item.needs_review for item in items):
        warnings.append("style_lora_caption_variable_coverage_review_needed")
    return LoRATrainingDatasetCaptionCoverageResult(
        dataset_id=dataset.id,
        training_type=dataset.training_type,
        item_count=len(items),
        review_count=sum(1 for item in items if item.needs_review),
        items=items,
        warnings=warnings,
    )


def _caption_coverage_for_item(
    dataset: LoRATrainingDataset,
    item: LoRATrainingDatasetItem,
) -> LoRATrainingCaptionCoverageItem:
    final_text = _caption_coverage_text([item.caption, *item.tags])
    source_text = _caption_coverage_text([
        *item.tags,
        *[source.caption for source in item.caption_sources],
        *[tag for source in item.caption_sources for tag in source.tags],
    ])
    final_categories = _caption_coverage_categories(final_text)
    source_categories = _caption_coverage_categories(source_text)
    missing_final_categories: list[str] = []
    warnings: list[str] = []
    needs_review = False
    if dataset.training_type == "style_lora":
        missing_final_categories = [
            category for category in _STYLE_LORA_REQUIRED_VARIABLE_CATEGORIES if category not in final_categories
        ]
        if missing_final_categories:
            needs_review = True
            warnings.append("style_caption_missing_variable_coverage")
        if len(set(source_categories) - set(final_categories)) >= 2:
            warnings.append("source_evidence_has_more_variable_coverage")
    return LoRATrainingCaptionCoverageItem(
        item_id=item.id,
        image_path=item.image_path,
        caption=item.caption,
        final_categories=final_categories,
        source_categories=source_categories,
        missing_final_categories=missing_final_categories,
        needs_review=needs_review,
        warnings=warnings,
    )


def _caption_coverage_text(parts: list[str]) -> str:
    return " ".join(str(part).casefold().replace("_", " ") for part in parts if str(part).strip())


def _caption_coverage_categories(text: str) -> list[str]:
    categories: list[str] = []
    padded = f" {text} "
    for category, keywords in _CAPTION_COVERAGE_KEYWORDS.items():
        if any(keyword in padded for keyword in keywords):
            categories.append(category)
    return categories


def build_lora_training_dataset_caption_assists(
    dataset: LoRATrainingDataset,
    request: LoRATrainingCaptionAssistRequestData,
    *,
    project_root: Path,
    only_missing_captions: bool = True,
    limit: int = 50,
) -> LoRATrainingDatasetCaptionAssistResult:
    max_items = min(100, max(1, limit))
    candidate_items = [
        item for item in dataset.items
        if not only_missing_captions or not item.caption.strip()
    ]
    selected_items = candidate_items[:max_items]
    warnings: list[str] = []
    if len(candidate_items) > len(selected_items):
        warnings.append(f"limited_caption_drafts_to_{max_items}")

    assists: list[LoRATrainingCaptionAssist] = []
    errors: list[str] = []
    for item in selected_items:
        try:
            assist = build_lora_training_caption_assist(
                dataset,
                item.id,
                request,
                project_root=project_root,
            )
        except ValueError as exc:
            errors.append(f"{item.id}:{exc}")
            continue
        if assist is not None:
            assists.append(assist)
    skipped_count = len(dataset.items) - len(selected_items)
    return LoRATrainingDatasetCaptionAssistResult(
        dataset_id=dataset.id,
        target_count=len(selected_items),
        drafted_count=len(assists),
        skipped_count=skipped_count,
        assists=assists,
        errors=errors,
        warnings=warnings,
    )


def _provider_api_key(api_key: str, api_key_env_var: str) -> str:
    direct_value = api_key.strip()
    if direct_value:
        return direct_value
    env_name = api_key_env_var.strip()
    if not env_name:
        return ""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ValueError("lora_caption_assist_provider_api_key_env_var_invalid")
    return os.environ.get(env_name, "").strip()


def _call_openai_compatible_caption_assist(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
    image_path: Path | None = None,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    if not model_value:
        raise ValueError("lora_caption_assist_provider_model_required")
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    user_content: str | list[dict] = user_prompt
    if image_path is not None:
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": _image_data_url_for_caption_assist(image_path)}},
        ]
    payload = {
        "model": model_value,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.post(endpoint_value, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"lora_caption_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"lora_caption_assist_provider_response_invalid: {exc}") from exc
    content = _openai_compatible_response_text(body)
    if not content.strip():
        raise ValueError("lora_caption_assist_provider_response_empty")
    return content.strip()


def _image_data_url_for_caption_assist(image_path: Path) -> str:
    try:
        with Image.open(image_path) as image:
            converted = image.convert("RGB")
            converted.thumbnail((1024, 1024))
            buffer = BytesIO()
            converted.save(buffer, format="JPEG", quality=88, optimize=True)
    except (OSError, ValueError) as exc:
        raise ValueError("lora_caption_assist_image_invalid") from exc
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _call_openai_responses_caption_assist(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    api_key_value = api_key.strip()
    if not model_value:
        raise ValueError("lora_caption_assist_provider_model_required")
    if not api_key_value:
        raise ValueError("lora_caption_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_value}",
    }
    payload = {
        "model": model_value,
        "instructions": system_prompt,
        "input": user_prompt,
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.post(endpoint_value, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"lora_caption_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"lora_caption_assist_provider_response_invalid: {exc}") from exc
    content = _openai_responses_response_text(body)
    if not content.strip():
        raise ValueError("lora_caption_assist_provider_response_empty")
    return content.strip()


def _call_anthropic_caption_assist(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    model_value = model.strip()
    api_key_value = api_key.strip()
    if not model_value:
        raise ValueError("lora_caption_assist_provider_model_required")
    if not api_key_value:
        raise ValueError("lora_caption_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key_value,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model_value,
        "max_tokens": 800,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.post(endpoint_value, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"lora_caption_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"lora_caption_assist_provider_response_invalid: {exc}") from exc
    content = _anthropic_response_text(body)
    if not content.strip():
        raise ValueError("lora_caption_assist_provider_response_empty")
    return content.strip()


def _call_gemini_caption_assist(
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint_value = _validated_provider_endpoint(endpoint)
    api_key_value = api_key.strip()
    if not api_key_value:
        raise ValueError("lora_caption_assist_provider_api_key_required")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key_value,
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    try:
        with httpx.Client(timeout=_provider_timeout(timeout_seconds)) as client:
            response = client.post(endpoint_value, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise ValueError(f"lora_caption_assist_provider_request_failed: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"lora_caption_assist_provider_response_invalid: {exc}") from exc
    content = _gemini_response_text(body)
    if not content.strip():
        raise ValueError("lora_caption_assist_provider_response_empty")
    return content.strip()


def _validated_provider_endpoint(endpoint: str) -> str:
    endpoint_value = endpoint.strip()
    if not endpoint_value:
        raise ValueError("lora_caption_assist_provider_endpoint_required")
    parsed_url = urlparse(endpoint_value)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("lora_caption_assist_provider_endpoint_invalid")
    return endpoint_value


def _provider_timeout(timeout_seconds: float) -> float:
    return max(1.0, min(300.0, timeout_seconds))


def _openai_compatible_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices_missing")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("choice_not_object")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("message_content_missing")


def _openai_responses_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = body.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("output_missing")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                parts.append(content_item["text"])
    if parts:
        return "\n".join(parts)
    raise ValueError("output_text_missing")


def _anthropic_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    content = body.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("content_missing")
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    if parts:
        return "\n".join(parts)
    raise ValueError("content_text_missing")


def _gemini_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise ValueError("response_body_not_object")
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates_missing")
    first = candidates[0]
    if not isinstance(first, dict):
        raise ValueError("candidate_not_object")
    content = first.get("content")
    if not isinstance(content, dict):
        raise ValueError("candidate_content_missing")
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("candidate_parts_missing")
    text_parts = [item["text"] for item in parts if isinstance(item, dict) and isinstance(item.get("text"), str)]
    if text_parts:
        return "\n".join(text_parts)
    raise ValueError("candidate_text_missing")


def _parse_caption_assist_provider_response(
    response_text: str,
    *,
    fallback_caption: str,
    fallback_tags: list[str],
) -> tuple[str, list[str], list[str]]:
    stripped = response_text.strip()
    warnings: list[str] = []
    parsed = _provider_response_json_object(stripped)
    if isinstance(parsed, dict):
        caption = str(parsed.get("caption") or fallback_caption).strip() or fallback_caption
        tags_value = parsed.get("tags", fallback_tags)
        if isinstance(tags_value, str):
            tags = _normalize_tags(_split_caption_assist_tags(tags_value))
        elif isinstance(tags_value, list):
            tags = _normalize_tags([str(item) for item in tags_value])
        else:
            tags = fallback_tags
        warning_value = parsed.get("warnings", [])
        if isinstance(warning_value, str) and warning_value.strip():
            warnings.extend(_caption_assist_warning_lines(warning_value))
        elif isinstance(warning_value, list):
            for item in warning_value:
                warnings.extend(_caption_assist_warning_lines(str(item)))
        return caption, tags, warnings

    caption = fallback_caption
    tags = fallback_tags
    caption_match = re.search(r"(?im)^\s*(?:caption|キャプション|説明|1\.\s*caption|1\.\s*キャプション)\s*[:：]\s*(.+)$", stripped)
    tags_match = re.search(r"(?im)^\s*(?:tags?|タグ|2\.\s*tags?|2\.\s*タグ)\s*[:：]\s*(.+)$", stripped)
    warnings_match = re.search(r"(?im)^\s*(?:warnings?|警告|注意|3\.\s*warnings?|3\.\s*警告)\s*[:：]\s*(.+)$", stripped)
    caption_section = _caption_assist_section(stripped, "caption")
    tags_section = _caption_assist_section(stripped, "tags")
    warnings_section = _caption_assist_section(stripped, "warnings")
    numbered_values = _caption_assist_numbered_values(stripped)
    provider_caption_used = False
    if caption_match:
        caption = caption_match.group(1).strip()
        provider_caption_used = True
    elif caption_section:
        caption = next((line.strip() for line in caption_section.splitlines() if line.strip()), fallback_caption)
        provider_caption_used = caption != fallback_caption
    elif numbered_values.get(1):
        caption = numbered_values[1]
        provider_caption_used = True
    else:
        first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
        if first_line:
            caption = first_line
            provider_caption_used = True
            warnings.append("provider_response_used_first_line_as_caption")
    if tags_match:
        tags = _normalize_tags(_split_caption_assist_tags(tags_match.group(1)))
    elif tags_section:
        tags = _normalize_tags(_split_caption_assist_tags(tags_section))
    elif numbered_values.get(2):
        tags = _normalize_tags(_split_caption_assist_tags(numbered_values[2]))
    elif provider_caption_used:
        caption_tags = _normalize_tags(_split_caption_assist_tags(caption))
        if caption_tags:
            tags = caption_tags
    elif not tags:
        tags = _normalize_tags(_split_caption_assist_tags(caption))
    if warnings_match:
        warnings.extend(_caption_assist_warning_lines(warnings_match.group(1)))
    elif warnings_section:
        warnings.extend(_caption_assist_warning_lines(warnings_section))
    elif numbered_values.get(3):
        warnings.extend(_caption_assist_warning_lines(numbered_values[3]))
    return caption, tags, warnings


def _caption_assist_warning_lines(value: str) -> list[str]:
    warnings: list[str] = []
    for line in value.splitlines():
        warning = line.strip().strip(".")
        if not warning:
            continue
        if warning.lower() in {"none", "no warning", "no warnings", "no warnings provided"}:
            continue
        warnings.append(line.strip())
    return warnings


def _caption_assist_numbered_values(response_text: str) -> dict[int, str]:
    values: dict[int, str] = {}
    for line in response_text.splitlines():
        match = re.match(r"^\s*([123])\s*[.)]\s+(.+?)\s*$", line)
        if not match:
            continue
        index = int(match.group(1))
        value = match.group(2).strip()
        if _caption_assist_heading_kind(value):
            continue
        values[index] = value
    return values


def _caption_assist_section(response_text: str, section: str) -> str:
    lines = response_text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if _caption_assist_heading_kind(line) == section:
            start_index = index + 1
            break
    if start_index is None:
        return ""

    section_lines: list[str] = []
    for line in lines[start_index:]:
        if _caption_assist_heading_kind(line):
            break
        section_lines.append(line)
    return "\n".join(line.strip() for line in section_lines if line.strip()).strip()


def _caption_assist_heading_kind(line: str) -> str:
    normalized = line.strip()
    if not normalized:
        return ""
    normalized = re.sub(r"^\d+\s*[.)]\s*", "", normalized)
    normalized = normalized.strip().rstrip(":：").strip()
    if re.fullmatch(r"(?:caption|キャプション|説明)", normalized, flags=re.IGNORECASE):
        return "caption"
    if re.fullmatch(r"(?:tags?|タグ)", normalized, flags=re.IGNORECASE):
        return "tags"
    if re.fullmatch(r"(?:warnings?|警告|注意)", normalized, flags=re.IGNORECASE):
        return "warnings"
    if re.match(r"short\s+warnings?\b", normalized, flags=re.IGNORECASE):
        return "warnings"
    return ""


def _provider_response_json_object(response_text: str) -> dict | None:
    candidates = [response_text]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", response_text))
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(response_text[first_brace : last_brace + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _split_caption_assist_tags(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,、;；]", value) if item.strip()]


def _render_dataset_toml(preset: LoRATrainingPreset, subset: DatasetSubset) -> str:
    width, height = preset.resolution
    lines = [
        "[general]",
        f"shuffle_caption = {_toml_bool(subset.shuffle_caption if subset.shuffle_caption is not None else preset.shuffle_caption)}",
        f"caption_extension = {_toml_string(subset.caption_extension)}",
        f"keep_tokens = {subset.keep_tokens if subset.keep_tokens is not None else preset.keep_tokens}",
        "",
        "[[datasets]]",
        f"resolution = [{width}, {height}]",
        f"batch_size = {preset.batch_size}",
        f"enable_bucket = {_toml_bool(preset.enable_bucket)}",
        "",
        "  [[datasets.subsets]]",
        f"  image_dir = {_toml_string(_normalize_path_for_toml(subset.image_dir))}",
        f"  num_repeats = {subset.num_repeats}",
    ]
    if subset.class_tokens.strip():
        lines.append(f"  class_tokens = {_toml_string(subset.class_tokens.strip())}")
    return "\n".join(lines) + "\n"


def _build_command_args(request: DatasetPreviewRequestData, preset: LoRATrainingPreset) -> list[str]:
    args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        preset.script,
        "--pretrained_model_name_or_path",
        request.pretrained_model_name_or_path,
        "--dataset_config",
        "<dataset.toml>",
        "--output_dir",
        request.output_dir,
        "--output_name",
        _training_output_name_for_preset(request.output_name, preset),
        "--save_model_as",
        "safetensors",
        "--save_precision",
        preset.save_precision,
        "--network_module",
        preset.network_module,
        "--network_dim",
        str(preset.network_dim),
        "--network_alpha",
        str(preset.network_alpha),
        "--learning_rate",
        preset.learning_rate,
        "--optimizer_type",
        _optimizer_type_for_request(request, preset),
        "--lr_scheduler",
        preset.lr_scheduler,
        "--max_train_epochs",
        str(request.max_train_epochs if request.max_train_epochs is not None else preset.max_train_epochs),
        "--mixed_precision",
        preset.mixed_precision,
    ]
    if preset.unet_lr:
        args.extend(["--unet_lr", preset.unet_lr])
    if preset.text_encoder_lr and preset.profile == "sdxl":
        args.extend(["--text_encoder_lr", preset.text_encoder_lr, preset.text_encoder_lr])
    elif preset.text_encoder_lr:
        args.extend(["--text_encoder_lr", preset.text_encoder_lr])
    if preset.gradient_checkpointing:
        args.append("--gradient_checkpointing")
    if preset.cache_latents:
        args.append("--cache_latents")
    optimizer_args = _optimizer_args_for_request(request)
    if optimizer_args:
        args.append("--optimizer_args")
        args.extend(optimizer_args)
    blocks_to_swap = _optimizer_blocks_to_swap_for_request(request, preset)
    if blocks_to_swap > 0:
        args.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if request.vae:
        args.extend(["--vae", request.vae])
    if preset.profile == "anima":
        args.extend(["--timestep_sampling", "sigmoid", "--discrete_flow_shift", "1.0"])
        if preset.sigmoid_scale is not None:
            args.extend(["--sigmoid_scale", str(preset.sigmoid_scale)])
        args.append("--cache_text_encoder_outputs")
        args.append("--network_train_unet_only")
        args.extend(["--vae_chunk_size", "64", "--vae_disable_cache"])
        if request.qwen3:
            args.extend(["--qwen3", request.qwen3])
    return args


def _optimizer_type_for_request(request: DatasetPreviewRequestData, preset: LoRATrainingPreset) -> str:
    optimizer_type = request.optimizer_type.strip()
    return optimizer_type or preset.optimizer_type


def _optimizer_args_for_request(request: DatasetPreviewRequestData) -> list[str]:
    if not request.optimizer_args.strip():
        return []
    return shlex.split(request.optimizer_args)


def _optimizer_blocks_to_swap_for_request(request: DatasetPreviewRequestData, preset: LoRATrainingPreset) -> int:
    optimizer_type = _optimizer_type_for_request(request, preset)
    if not request.optimizer_low_vram or optimizer_type.casefold() == "adamw":
        return 0
    return 6


def _training_output_name_for_preset(output_name: str, preset: LoRATrainingPreset) -> str:
    if preset.profile != "anima":
        return output_name
    stem = Path(output_name).stem
    return f"{stem}_unet" if stem else output_name


def _with_sample_prompts(command_args: list[str], sample_prompts_path: Path, preset: LoRATrainingPreset) -> list[str]:
    args = list(command_args)
    if "--sample_prompts" not in args:
        args.extend(["--sample_prompts", str(sample_prompts_path)])
    if "--sample_every_n_epochs" not in args and "--sample_every_n_steps" not in args:
        args.extend(["--sample_every_n_epochs", "1"])
    if "--sample_sampler" not in args:
        args.extend(["--sample_sampler", _sample_sampler_for_preset(preset)])
    return args


def _render_sample_prompts(dataset: LoRATrainingDataset, preset: LoRATrainingPreset) -> str:
    captions = [item.caption.strip() for item in dataset.items if item.caption.strip()]
    trigger = _shared_caption_lead_token(captions)
    caption_terms = _ordered_caption_terms(captions)
    if trigger and dataset.training_type == "style_lora":
        caption_terms = [term for term in caption_terms if term.casefold() != trigger.casefold()]
    style_terms = _sample_prompt_terms(caption_terms, dataset.training_type)
    width, height = _sample_prompt_dimensions(preset.resolution)
    prompt = _sample_prompt_base(dataset.training_type, trigger=trigger if dataset.training_type != "style_lora" else "")
    if style_terms:
        prompt = ", ".join([prompt, *style_terms])
    negative = "low quality, worst quality, blurry, text, watermark, bad anatomy"
    return "\n".join(
        [
            "# Asteria generated sd-scripts sample prompt.",
            "# Style LoRA prompts intentionally omit shared lead trigger tokens.",
            f"{prompt} --n {negative} --w {width} --h {height} --d 26062901 --l 6.0 --s 24",
            "",
        ]
    )


def _sample_sampler_for_preset(preset: LoRATrainingPreset) -> str:
    return "euler_a" if preset.profile == "sdxl" else "euler"


def _sample_prompt_dimensions(resolution: tuple[int, int]) -> tuple[int, int]:
    width, height = resolution
    longest = max(width, height)
    if longest <= 768:
        return width, height
    scale = 768 / longest
    sampled_width = max(512, int(width * scale) // 64 * 64)
    sampled_height = max(512, int(height * scale) // 64 * 64)
    return sampled_width, sampled_height


def _sample_prompt_base(training_type: TrainingType, *, trigger: str) -> str:
    if training_type == "character_lora":
        return ", ".join([part for part in [trigger, "solo character portrait", "anime illustration"] if part])
    if training_type == "style_lora":
        return "anime illustration, cohesive visual style study"
    return ", ".join([part for part in [trigger, "single subject study", "anime illustration"] if part])


def _sample_prompt_terms(terms: list[str], training_type: TrainingType) -> list[str]:
    excluded = {
        "masterpiece",
        "best quality",
        "high quality",
        "1girl",
        "1boy",
        "solo",
        "anime",
        "illustration",
        "anime illustration",
        "digital illustration",
        "anime style",
        "digital painting",
    }
    if training_type == "style_lora":
        excluded.update({"girl", "boy", "woman", "man", "female", "male"})
    result: list[str] = []
    for term in terms:
        normalized = term.casefold()
        if normalized in excluded:
            continue
        if term not in result:
            result.append(term)
        if len(result) >= 8:
            break
    return result


def _ordered_caption_terms(captions: list[str]) -> list[str]:
    terms: list[str] = []
    for caption in captions:
        for term in _split_caption_assist_tags(caption):
            if term not in terms:
                terms.append(term)
    return terms


def _shared_caption_lead_token(captions: list[str]) -> str:
    if not captions:
        return ""
    lead_counts: dict[str, int] = {}
    lead_values: dict[str, str] = {}
    for caption in captions:
        parts = _split_caption_assist_tags(caption)
        if not parts:
            continue
        lead = parts[0].strip()
        if not lead:
            continue
        key = lead.casefold()
        lead_counts[key] = lead_counts.get(key, 0) + 1
        lead_values.setdefault(key, lead)
    if not lead_counts:
        return ""
    key, count = max(lead_counts.items(), key=lambda item: item[1])
    if count / len(captions) < 0.6:
        return ""
    if key in {"masterpiece", "best quality", "high quality", "1girl", "1boy", "solo"}:
        return ""
    return lead_values[key]


def _preview_warnings(request: DatasetPreviewRequestData, preset: LoRATrainingPreset) -> list[str]:
    warnings = ["This is a preview only; Asteria does not start sd-scripts training from this endpoint."]
    if not request.pretrained_model_name_or_path.strip():
        warnings.append("A base model path is required before the command can be executed.")
    if preset.profile == "anima" and not request.qwen3.strip():
        warnings.append("Anima training requires a Qwen3 text encoder path before execution.")
    return warnings


def _build_run_preflight(
    *,
    preset: LoRATrainingPreset,
    request: TrainingRunPrepareRequestData,
    dataset_config_path: Path,
    character_caption_gate: LoRATrainingCharacterCaptionGate,
) -> LoRATrainingRunPreflight:
    errors: list[str] = []
    warnings: list[str] = []
    if not _path_is_file(dataset_config_path):
        errors.append("dataset_config_missing")
    if not request.pretrained_model_name_or_path.strip():
        errors.append("base_model_required")
    if not request.output_dir.strip():
        errors.append("output_dir_required")
    if not request.output_name.strip():
        errors.append("output_name_required")
    command_parts = _command_prefix(request.accelerate_command, request.project_root)
    command_exists = _command_prefix_exists(command_parts)
    if not command_exists:
        errors.append("accelerate_command_not_found")
    python_module = _python_module_name(command_parts)
    if command_exists and python_module and python_module.startswith("accelerate") and not _python_module_available(
        command_parts[0], python_module
    ):
        errors.append("accelerate_python_module_not_found")
    optimizer_type = request.optimizer_type.strip() or preset.optimizer_type
    if (
        command_exists
        and optimizer_type.casefold() == "prodigy"
        and not _python_module_available(command_parts[0], "prodigyopt")
    ):
        errors.append("optimizer_python_module_not_found")

    working_dir = request.working_dir.strip()
    if working_dir:
        working_dir_candidate = _source_path_candidate(working_dir, request.project_root)
        try:
            working_dir_path = working_dir_candidate.resolve()
        except OSError:
            working_dir_path = working_dir_candidate
        if _is_disallowed_tool_execution_path(working_dir_path, request.project_root):
            errors.append("lora_training_working_dir_not_allowed")
        else:
            script_path = working_dir_path / preset.script
            if not _path_is_file(script_path):
                errors.append("training_script_not_found")
                warnings.append(f"Expected training script at {script_path}.")
            if preset.profile == "anima":
                convert_script_path = working_dir_path / "networks" / "convert_anima_lora_to_comfy.py"
                if not _path_is_file(convert_script_path):
                    errors.append("anima_comfy_conversion_script_not_found")
                    warnings.append(f"Expected Anima LoRA conversion script at {convert_script_path}.")
    else:
        warnings.append("sd-scripts working directory is not set; run the command from a valid sd-scripts checkout.")

    if preset.profile == "anima" and not request.qwen3.strip():
        errors.append("qwen3_required_for_anima")
    errors.extend(reason for reason in character_caption_gate.reasons if reason not in errors)
    return LoRATrainingRunPreflight(
        ready=not errors,
        errors=errors,
        warnings=warnings,
        character_caption_gate=character_caption_gate,
    )


def _caption_assist_character_policy_instruction(dataset: LoRATrainingDataset) -> str:
    if dataset.training_type != "character_lora":
        return ""
    policy = _normalized_character_caption_policy(dataset.character_caption_policy)
    instructions: list[str] = []
    if policy.trigger_token:
        instructions.append(f'Place the trigger token "{policy.trigger_token}" first in every caption.')
    else:
        instructions.append("A trigger token must be configured before character captions are training-ready.")
    if policy.excluded_identity_tags:
        instructions.append(
            "Remove these declared stable identity tags: "
            f"{', '.join(policy.excluded_identity_tags)}."
        )
    if policy.variable_detail_guidance:
        instructions.append(policy.variable_detail_guidance)
    return " ".join(instructions) + " "


def _apply_character_caption_policy_to_assist(
    dataset: LoRATrainingDataset,
    caption: str,
    tags: list[str],
) -> tuple[str, list[str]]:
    if dataset.training_type != "character_lora":
        return caption, tags
    policy = _normalized_character_caption_policy(dataset.character_caption_policy)
    trigger_key = policy.trigger_token.casefold()
    excluded_keys = {tag.casefold() for tag in policy.excluded_identity_tags}

    caption_tokens = [" ".join(token.strip().split()) for token in caption.split(",") if token.strip()]
    filtered_caption_tokens = [
        token
        for token in caption_tokens
        if token.casefold() not in excluded_keys and (not trigger_key or token.casefold() != trigger_key)
    ]
    if policy.trigger_token:
        filtered_caption_tokens.insert(0, policy.trigger_token)

    filtered_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = " ".join(raw_tag.strip().split())
        key = tag.casefold()
        if not tag or key in excluded_keys or (trigger_key and key == trigger_key) or key in seen:
            continue
        seen.add(key)
        filtered_tags.append(tag)
    if policy.trigger_token:
        filtered_tags.insert(0, policy.trigger_token)
    return ", ".join(filtered_caption_tokens), filtered_tags


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _normalize_path_for_toml(path: str) -> str:
    if "\\" in path:
        return str(PureWindowsPath(path))
    return path


def _resolve_source_path(path: str, project_root: Path) -> Path:
    return _source_path_candidate(path, project_root).resolve()


def _source_path_candidate(path: str, project_root: Path) -> Path:
    source = Path(path)
    if not source.is_absolute():
        source = project_root / source
    return source


def resolve_lora_training_dataset_item_image_path(
    dataset: LoRATrainingDataset,
    item_id: str,
    *,
    project_root: Path,
) -> Path:
    item = next((candidate for candidate in dataset.items if candidate.id == item_id), None)
    if item is None:
        raise ValueError("lora_training_dataset_item_not_found")
    try:
        path = _resolve_source_path(item.image_path, project_root)
        path_available = path.is_file()
    except OSError as exc:
        raise ValueError("lora_training_dataset_image_not_found") from exc
    if not path_available:
        raise ValueError("lora_training_dataset_image_not_found")
    return path


def _project_path_for_command(value: str, project_root: Path, *, force: bool = False) -> str:
    text = value.strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    path = Path(text)
    if path.is_absolute():
        return str(path)
    normalized = text.replace("\\", "/")
    is_project_path = normalized.startswith(("./", "../", "data/", "models/", "runtime/"))
    resolved = (project_root / path).resolve()
    if force or is_project_path or _path_exists(resolved):
        return str(resolved)
    return text


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:80] or "dataset"


def _safe_caption_extension(value: str) -> str:
    extension = value.strip() or ".txt"
    if not extension.startswith("."):
        extension = f".{extension}"
    if not re.fullmatch(r"\.[A-Za-z0-9_-]{1,16}", extension):
        raise ValueError("lora_training_caption_extension_invalid")
    return extension


def _safe_caption_source_extension(value: str) -> str:
    extension = value.strip() or ".txt"
    if not extension.startswith("."):
        extension = f".{extension}"
    if not re.fullmatch(r"(?:\.[A-Za-z0-9_-]{1,24}){1,4}", extension):
        raise ValueError("lora_training_caption_source_extension_invalid")
    return extension


def _command_prefix(command: str, project_root: Path | None = None) -> list[str]:
    value = command.strip()
    if not value:
        return ["accelerate"]
    try:
        parts = shlex.split(value, posix=False)
    except ValueError:
        parts = [value]
    cleaned = [part.strip().strip('"').strip("'") for part in parts if part.strip()]
    if not cleaned:
        return ["accelerate"]
    cleaned[0] = _resolve_command_executable(cleaned[0], project_root)
    cleaned = _normalize_accelerate_python_module_command(cleaned)
    return cleaned


def _external_command_parts(command: str, project_root: Path | None = None) -> list[str]:
    value = command.strip()
    if not value:
        return []
    try:
        parts = shlex.split(value, posix=False)
    except ValueError:
        parts = [value]
    cleaned = [part.strip().strip('"').strip("'") for part in parts if part.strip()]
    if not cleaned:
        return []
    cleaned[0] = _resolve_command_executable(cleaned[0], project_root)
    return cleaned


def _format_caption_source_refresh_command(command: str, placeholders: dict[str, str]) -> str:
    try:
        return command.format_map(_CaptionSourceRefreshPlaceholders(placeholders))
    except (KeyError, ValueError) as exc:
        raise ValueError("lora_training_caption_source_refresh_placeholder_invalid") from exc


class _CaptionSourceRefreshPlaceholders(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _resolve_command_executable(executable: str, project_root: Path | None = None) -> str:
    path = Path(executable)
    path_like = path.parent != Path(".") or "/" in executable or "\\" in executable
    if project_root is not None and not path.is_absolute() and path_like:
        project_path = (project_root / path).resolve()
        if _path_is_file(project_path):
            return str(project_path)
        return str(project_path)
    if shutil.which(executable):
        return executable
    if _path_is_file(path):
        return str(path.resolve())
    if project_root is not None and not path.is_absolute() and not path_like:
        project_path = (project_root / path).resolve()
        if _path_is_file(project_path):
            return str(project_path)
    return executable


def _command_prefix_exists(command_parts: list[str]) -> bool:
    if not command_parts:
        return False
    executable = command_parts[0]
    executable_path = Path(executable)
    path_like = executable_path.parent != Path(".") or "/" in executable or "\\" in executable
    if executable_path.is_absolute() or path_like:
        return _path_is_file(executable_path)
    if shutil.which(executable):
        return True
    return _path_is_file(executable_path)


def _command_uses_python_module(command_parts: list[str], module: str) -> bool:
    for index, part in enumerate(command_parts[:-1]):
        if part == "-m" and command_parts[index + 1] == module:
            return True
    return False


def _python_module_name(command_parts: list[str]) -> str:
    for index, part in enumerate(command_parts[:-1]):
        if part == "-m":
            return command_parts[index + 1]
    return ""


def _normalize_accelerate_python_module_command(command_parts: list[str]) -> list[str]:
    normalized = list(command_parts)
    for index, part in enumerate(normalized[:-1]):
        if part == "-m" and normalized[index + 1] == "accelerate":
            normalized[index + 1] = "accelerate.commands.launch"
            break
    return normalized


def _python_module_available(python_executable: str, module: str) -> bool:
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({module!r}) is not None else 1)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _with_accelerate_command(
    command_args: list[str],
    accelerate_command: str,
    project_root: Path | None = None,
) -> list[str]:
    args = list(command_args)
    command = _command_prefix(accelerate_command, project_root)
    if args:
        rest = args[1:]
        if _command_uses_python_module(command, "accelerate.commands.launch") and rest[:1] == ["launch"]:
            rest = rest[1:]
        args = command + rest
    return args


def _resolve_training_working_dir(working_dir: str, project_root: Path) -> Path:
    if working_dir.strip():
        candidate = _source_path_candidate(working_dir, project_root)
    else:
        candidate = project_root
    if _is_disallowed_tool_execution_path(candidate, project_root):
        raise ValueError("lora_training_working_dir_not_allowed")
    try:
        path = candidate.resolve()
    except OSError:
        path = candidate
    if _is_disallowed_tool_execution_path(path, project_root):
        raise ValueError("lora_training_working_dir_not_allowed")
    if not _path_is_dir(path):
        raise ValueError("lora_training_working_dir_not_found")
    return path


def _resolve_refresh_working_dir(working_dir: str, project_root: Path) -> Path:
    if working_dir.strip():
        candidate = _source_path_candidate(working_dir, project_root)
    else:
        candidate = project_root
    if _is_disallowed_tool_execution_path(candidate, project_root):
        raise ValueError("lora_training_caption_source_working_dir_not_allowed")
    try:
        path = candidate.resolve()
    except OSError:
        path = candidate
    if _is_disallowed_tool_execution_path(path, project_root):
        raise ValueError("lora_training_caption_source_working_dir_not_allowed")
    if not _path_is_dir(path):
        raise ValueError("lora_training_caption_source_refresh_working_dir_not_found")
    return path


def _tail_text(value: object, *, limit: int = 4000) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value or "")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _read_text_tail(path: Path, *, max_chars: int = 8000) -> str:
    if not _path_is_file(path):
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _apply_run_progress_from_log(run: LoRATrainingRun) -> None:
    progress = _parse_training_progress(run.log_tail)
    if progress:
        current, total, percent, message = progress
        run.progress_current = current
        run.progress_total = total
        run.progress_percent = percent
        run.progress_message = message
    if run.status == "completed":
        if run.progress_total is not None:
            run.progress_current = run.progress_total
        run.progress_percent = 100.0
        run.progress_message = run.progress_message or "completed"


def _finalize_anima_training_output(run: LoRATrainingRun, *, project_root: Path | None) -> None:
    if run.profile != "anima":
        return
    output_dir = run.output_dir.strip()
    output_name = run.output_name.strip()
    if not output_dir or not output_name:
        return
    root = project_root or Path.cwd()
    directory = _resolve_source_path(output_dir, root)
    stem = Path(output_name).stem
    converted_path = directory / f"{stem}_comfy.safetensors"
    temporary_path = directory / f"{stem}_unet.safetensors"
    legacy_path = directory / f"{stem}.safetensors"
    if _path_is_file(converted_path):
        if _path_is_file(temporary_path):
            try:
                temporary_path.unlink()
            except OSError as exc:
                run.log_tail = _append_log_tail_note(run.log_path, f"Anima LoRA temporary cleanup failed: {exc}")
        return
    source_path = temporary_path if _path_is_file(temporary_path) else legacy_path
    if not _path_is_file(source_path):
        run.status = "failed"
        run.exit_code = -2
        run.log_tail = _append_log_tail_note(
            run.log_path,
            f"Anima LoRA conversion failed: source file not found ({temporary_path.name} or {legacy_path.name}).",
        )
        return
    try:
        working_dir = _resolve_training_working_dir(run.working_dir, root)
    except ValueError as exc:
        run.status = "failed"
        run.exit_code = -2
        run.log_tail = _append_log_tail_note(run.log_path, f"Anima LoRA conversion failed: {exc}")
        return
    convert_script = working_dir / "networks" / "convert_anima_lora_to_comfy.py"
    if not _path_is_file(convert_script):
        run.status = "failed"
        run.exit_code = -2
        run.log_tail = _append_log_tail_note(
            run.log_path,
            f"Anima LoRA conversion failed: conversion script not found at {convert_script}.",
        )
        return
    conversion_python = (
        run.command_args[0]
        if _command_uses_python_module(run.command_args, "accelerate.commands.launch")
        else sys.executable
    )
    command = [conversion_python, str(convert_script), str(source_path), str(converted_path)]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(working_dir) if not existing_pythonpath else f"{working_dir}{os.pathsep}{existing_pythonpath}"
    result = subprocess.run(
        command,
        cwd=str(working_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = _tail_text(result.stdout)
    run.log_tail = _append_log_tail_note(
        run.log_path,
        "\n".join(
            part
            for part in [
                f"Asteria Anima LoRA Comfy conversion command: {' '.join(command)}",
                output,
                f"Asteria Anima LoRA Comfy conversion exit code: {result.returncode}",
            ]
            if part
        ),
    )
    if result.returncode != 0 or not _path_is_file(converted_path):
        run.status = "failed"
        run.exit_code = result.returncode if result.returncode != 0 else -2
        return
    if source_path == temporary_path:
        try:
            temporary_path.unlink()
        except OSError as exc:
            run.log_tail = _append_log_tail_note(run.log_path, f"Anima LoRA temporary cleanup failed: {exc}")


def _apply_run_trained_model_paths(run: LoRATrainingRun, *, project_root: Path | None) -> None:
    output_dir = run.output_dir.strip()
    output_name = run.output_name.strip()
    if not output_dir or not output_name:
        run.trained_model_paths = []
        return
    root = project_root or Path.cwd()
    directory = _resolve_source_path(output_dir, root)
    if not _path_is_dir(directory):
        run.trained_model_paths = []
        return
    stem = Path(output_name).stem
    exact_names = {f"{stem}.safetensors"}
    if output_name.lower().endswith(".safetensors"):
        exact_names.add(Path(output_name).name)
    if run.profile == "anima":
        exact_names.add(f"{stem}_comfy.safetensors")
    candidates: list[Path] = []
    try:
        for path in directory.glob("*.safetensors"):
            if not _path_is_file(path):
                continue
            name = path.name
            if run.profile == "anima" and name == f"{stem}_unet.safetensors" and _path_is_file(directory / f"{stem}_comfy.safetensors"):
                continue
            if name in exact_names or name.startswith(f"{stem}-") or name.startswith(f"{stem}_"):
                try:
                    candidates.append(path.resolve())
                except OSError:
                    continue
    except OSError:
        candidates = []
    run.trained_model_paths = [str(path) for path in sorted(set(candidates), key=lambda item: item.name.lower())]


def _apply_run_sample_image_paths(run: LoRATrainingRun, *, project_root: Path | None) -> None:
    output_dir = run.output_dir.strip()
    output_name = run.output_name.strip()
    root = project_root or Path.cwd()
    directories: list[tuple[Path, bool]] = []
    if output_dir:
        output_path = _resolve_source_path(output_dir, root)
        directories.append((output_path, False))
        directories.append((output_path / "sample", True))
        directories.append((output_path / "samples", True))
    if run.materialized_root:
        materialized_root = _resolve_source_path(run.materialized_root, root)
        directories.append((materialized_root / "sample", True))
        directories.append((materialized_root / "samples", True))
    stem = Path(output_name).stem if output_name else ""
    candidates: list[Path] = []
    seen_directories: set[Path] = set()
    for directory, recursive in directories:
        try:
            resolved_dir = directory.resolve()
        except OSError:
            continue
        if resolved_dir in seen_directories or not _path_is_dir(resolved_dir):
            continue
        seen_directories.add(resolved_dir)
        try:
            pattern_iter = resolved_dir.rglob("*") if recursive else resolved_dir.glob("*")
            for path in pattern_iter:
                if not _path_is_file(path) or path.suffix.lower() not in SAMPLE_IMAGE_EXTENSIONS:
                    continue
                if stem and not (path.stem == stem or path.name.startswith(f"{stem}-") or path.name.startswith(f"{stem}_")):
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if _is_valid_image_file(resolved):
                    candidates.append(resolved)
        except OSError:
            continue
    run.sample_image_paths = [str(path) for path in sorted(set(candidates), key=lambda item: str(item).lower())]


def _apply_run_sample_image_scores(run: LoRATrainingRun, *, project_root: Path | None) -> None:
    root = project_root or Path.cwd()
    scores: list[LoRATrainingSampleImageScore] = []
    for sample_text in run.sample_image_paths:
        if not sample_text.strip():
            continue
        scores.append(_score_lora_training_sample_image(_resolve_source_path(sample_text, root)))
    run.sample_image_scores = scores


def _score_lora_training_sample_image(path: Path) -> LoRATrainingSampleImageScore:
    resolved = path.resolve()
    warnings: list[str] = []
    try:
        with Image.open(resolved) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("invalid image dimensions")
            grayscale = image.convert("L")
            stats = ImageStat.Stat(grayscale)
            brightness = round(float(stats.mean[0]), 2)
            contrast = round(float(stats.stddev[0]), 2)
    except (OSError, ValueError):
        return LoRATrainingSampleImageScore(
            image_path=str(resolved),
            score=0,
            rating="poor",
            warnings=["sample_image_unreadable"],
        )

    score = 100
    min_side = min(width, height)
    aspect_ratio = width / height if height else 0
    if min_side < 256:
        score -= 25
        warnings.append("sample_low_resolution")
    elif min_side < 512:
        score -= 10
        warnings.append("sample_medium_resolution")
    if contrast < 8:
        score -= 25
        warnings.append("sample_low_contrast")
    elif contrast < 18:
        score -= 10
        warnings.append("sample_soft_contrast")
    if brightness < 8 or brightness > 247:
        score -= 15
        warnings.append("sample_extreme_brightness")
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        score -= 10
        warnings.append("sample_extreme_aspect_ratio")

    score = max(0, min(100, score))
    if score >= 80 and not warnings:
        rating: LoRATrainingSampleRating = "good"
    elif score >= 45:
        rating = "review"
    else:
        rating = "poor"
    return LoRATrainingSampleImageScore(
        image_path=str(resolved),
        width=width,
        height=height,
        score=score,
        rating=rating,
        average_brightness=brightness,
        contrast=contrast,
        warnings=warnings,
    )


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return _path_is_relative_to_without_resolve(path, parent)
    return True


def _path_is_relative_to_without_resolve(path: Path, parent: Path) -> bool:
    candidate = path if path.is_absolute() else Path.cwd() / path
    root = parent if parent.is_absolute() else Path.cwd() / parent
    candidate_text = os.path.normcase(os.path.normpath(os.fspath(candidate)))
    root_text = os.path.normcase(os.path.normpath(os.fspath(root)))
    root_prefix = root_text.rstrip("\\/")
    return candidate_text == root_prefix or candidate_text.startswith(f"{root_prefix}{os.sep}")


def _is_disallowed_tool_execution_path(path: Path, project_root: Path) -> bool:
    # Last-resort guard for user-configured executable/tool paths.
    # Asteria runs installed tools, not documentation examples bundled with the repo.
    docs_root = project_root / "docs"
    return _path_is_relative_to(path, docs_root)


def _unique_registered_lora_path(target_dir: Path, source_name: str, run_id: str) -> Path:
    source = Path(source_name)
    suffix = source.suffix if source.suffix.lower() == ".safetensors" else ".safetensors"
    stem = _safe_slug(source.stem) or "trained_lora"
    candidate = target_dir / f"{stem}{suffix}"
    if _new_file_candidate_is_available(candidate):
        return candidate
    run_suffix = run_id[:8] if run_id else uuid4().hex[:8]
    candidate = target_dir / f"{stem}-{run_suffix}{suffix}"
    if _new_file_candidate_is_available(candidate):
        return candidate
    index = 2
    while True:
        candidate = target_dir / f"{stem}-{run_suffix}-{index}{suffix}"
        if _new_file_candidate_is_available(candidate):
            return candidate
        index += 1


def _is_valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError):
        return False
    return width > 0 and height > 0


def _copy_machine_image_evidence(path_text: str, project_root: Path) -> LoRACopyMachineImageEvidence:
    path = _resolve_source_path(path_text, project_root)
    if not _path_is_file(path):
        raise ValueError("lora_copy_machine_image_not_found")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("lora_copy_machine_image_invalid") from exc
    if width <= 0 or height <= 0:
        raise ValueError("lora_copy_machine_image_invalid")
    return LoRACopyMachineImageEvidence(
        image_path=str(path),
        width=width,
        height=height,
        sha256=digest.hexdigest(),
    )


def build_lora_copy_machine_source_reference(
    session: LoRACopyMachineSession,
) -> LoRACopyMachineSourceReference:
    path = Path(session.source.image_path)
    if not _path_is_file(path):
        raise ValueError("copy_machine_source_reference_missing")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        size_bytes = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise ValueError("copy_machine_source_reference_modified") from exc
    image_sha256 = digest.hexdigest()
    if image_sha256 != session.source.sha256:
        raise ValueError("copy_machine_source_reference_modified")
    return LoRACopyMachineSourceReference(
        image_path=str(path),
        image_sha256=image_sha256,
        image_size_bytes=size_bytes,
        width=width,
        height=height,
    )


def _apply_copy_machine_target_transform(
    image: Image.Image,
    transform: LoRACopyMachineTargetTransform,
) -> Image.Image:
    if transform == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if transform == "high_contrast":
        return ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=2).convert("RGB")
    if transform == "line_extract":
        return ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES).convert("RGB")
    if transform == "bold_line":
        return ImageOps.grayscale(image).filter(ImageFilter.MinFilter(3)).convert("RGB")
    if transform == "flat_palette":
        return image.quantize(colors=16).convert("RGB")
    if transform == "posterize":
        return ImageOps.posterize(image.convert("RGB"), bits=3)
    if transform == "sharpen":
        return image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.6, percent=180, threshold=2))
    if transform == "soft_blur":
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.2))
    if transform == "sepia":
        gray = ImageOps.grayscale(image)
        return ImageOps.colorize(gray, black="#1f1710", white="#f2d3a0").convert("RGB")
    if transform == "invert":
        return ImageOps.invert(image.convert("RGB"))
    raise ValueError("lora_copy_machine_target_transform_unsupported")


def _resolve_copy_machine_traintrain_path(path_text: str, project_root: Path) -> Path:
    if path_text.strip():
        candidate = _source_path_candidate(path_text, project_root)
        if _is_disallowed_tool_execution_path(candidate, project_root):
            raise ValueError("copy_machine_tool_path_not_allowed")
        try:
            path = candidate.resolve()
        except OSError:
            path = candidate
        if _is_disallowed_tool_execution_path(path, project_root):
            raise ValueError("copy_machine_tool_path_not_allowed")
        return path
    discovered = _discover_copy_machine_traintrain_cli_path()
    if discovered is not None:
        return discovered
    return project_root / "runtime" / "tools" / "__configure_traintrain_standalone__"


def _discover_copy_machine_traintrain_cli_path() -> Path | None:
    for command in ("traintrain", "traintrain-standalone"):
        executable = shutil.which(command)
        if not executable:
            continue
        executable_path = Path(executable)
        if executable_path.name.lower() == "train_cli.py":
            return executable_path
        for parent in [executable_path.parent, *executable_path.parents]:
            script = parent / "cli" / "train_cli.py"
            if _path_is_file(script) and _copy_machine_traintrain_configs_ready(parent):
                return script
    return None


def _copy_machine_traintrain_script_path(traintrain_path: Path) -> Path:
    if traintrain_path.name.lower() == "train_cli.py":
        return traintrain_path
    return traintrain_path / "cli" / "train_cli.py"


def _copy_machine_traintrain_runner_path() -> Path:
    return Path(__file__).resolve().parents[1] / "adapters" / "traintrain_copy_machine_runner.py"


def _copy_machine_traintrain_root(traintrain_path: Path) -> Path:
    if traintrain_path.name.lower() == "train_cli.py":
        return traintrain_path.parent.parent
    return traintrain_path


def _copy_machine_traintrain_config_path(traintrain_path: Path, method: LoRACopyMachineMethod) -> Path | None:
    root = _copy_machine_traintrain_root(traintrain_path)
    if method == "direct_difference":
        candidates = [
            root / "cli" / "configs" / "addift_sdxl_action.json",
            root / "traintrain" / "presets" / "ADDifT_Action_XL.json",
        ]
    elif method == "classic_copy_machine":
        candidates = [
            root / "cli" / "configs" / "difference_copy_sdxl.json",
            root / "traintrain" / "presets" / "Difference_Use2ndPass.json",
            root / "traintrain" / "presets" / "Difference.json",
        ]
    else:
        return None
    return next((path for path in candidates if _path_is_file(path)), None)


def _copy_machine_traintrain_configs_ready(traintrain_path: Path) -> bool:
    return all(
        _copy_machine_traintrain_config_path(traintrain_path, method) is not None
        for method in ("direct_difference", "classic_copy_machine")
    )


def _discover_copy_machine_trained_models(run: LoRACopyMachineTrainingRun) -> list[str]:
    output_dir = Path(run.output_dir)
    if not _path_is_dir(output_dir):
        return []
    output_name = _safe_slug(run.output_name or "copy-machine-lora")
    direct_candidates = [path for path in output_dir.glob(f"{output_name}*.safetensors") if _path_is_file(path)]
    recursive_candidates = [
        path
        for path in output_dir.rglob("*.safetensors")
        if _path_is_file(path) and _copy_machine_model_path_matches_output_name(path, output_dir, output_name)
    ]
    candidates = [*direct_candidates, *recursive_candidates]
    return [str(path) for path in sorted(set(candidates), key=lambda item: str(item).lower())]


def _copy_machine_model_path_matches_output_name(path: Path, output_dir: Path, output_name: str) -> bool:
    normalized_output_name = output_name.casefold()
    if path.stem.casefold().startswith(normalized_output_name):
        return True
    try:
        relative_parts = path.relative_to(output_dir).parts[:-1]
    except ValueError:
        relative_parts = path.parts[:-1]
    return any(part.casefold().startswith(normalized_output_name) for part in relative_parts)


def _copy_machine_model_file_evidence(path: Path) -> LoRACopyMachineModelEvidence | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return LoRACopyMachineModelEvidence(
            model_path=str(path),
            sha256=digest.hexdigest(),
            size_bytes=path.stat().st_size,
        )
    except OSError:
        return None


def _copy_machine_model_evidence(paths: list[str]) -> list[LoRACopyMachineModelEvidence]:
    evidence: list[LoRACopyMachineModelEvidence] = []
    for path_text in paths:
        item = _copy_machine_model_file_evidence(Path(path_text))
        if item is not None:
            evidence.append(item)
    return evidence


def _copy_machine_file_evidence(path: Path) -> LoRACopyMachineFileEvidence | None:
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return LoRACopyMachineFileEvidence(
            path=str(path),
            sha256=digest.hexdigest(),
            size_bytes=path.stat().st_size,
        )
    except OSError:
        return None


def _ensure_copy_machine_effective_config_immutable(
    run: LoRACopyMachineTrainingRun,
) -> bool:
    if not run.effective_config_path or not run.output_dir or not run.output_name:
        return False
    source_path = Path(run.effective_config_path)
    target_path = (
        Path(run.output_dir)
        / f"{run.output_name}.{run.id}.asteria-effective-config.json"
    )
    try:
        if source_path.resolve() == target_path.resolve():
            current_evidence = _copy_machine_file_evidence(target_path)
            if current_evidence is None:
                return False
            if run.effective_config_evidence != current_evidence:
                run.effective_config_evidence = current_evidence
                return True
            return False
        if not source_path.is_file():
            return False
        if target_path.exists():
            source_evidence = _copy_machine_file_evidence(source_path)
            target_evidence = _copy_machine_file_evidence(target_path)
            if (
                source_evidence is None
                or target_evidence is None
                or source_evidence.sha256 != target_evidence.sha256
            ):
                return False
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        run.effective_config_path = str(target_path)
        run.effective_config_evidence = _copy_machine_file_evidence(target_path)
        try:
            config_arg_index = run.command_args.index("--effective-config-output") + 1
            run.command_args[config_arg_index] = str(target_path)
        except (ValueError, IndexError):
            pass
        return True
    except OSError:
        return False


def _copy_machine_training_log_has_nonfinite_loss(log_path: str) -> bool:
    if not log_path:
        return False
    path = Path(log_path)
    if not _path_is_file(path):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Loss EMA * 1000: nan" in text or "Loss Velosity: nan" in text


def _append_unique_issue(issues: list[str], issue: str) -> None:
    if issue and issue not in issues:
        issues.append(issue)


def _fail_copy_machine_nonfinite_run(run: LoRACopyMachineTrainingRun) -> None:
    run.status = "failed"
    run.registered_model_paths = []
    _append_unique_issue(run.errors, "copy_machine_training_nonfinite_loss")
    run.finished_at = run.finished_at or _utc_now()
    run.updated_at = _utc_now()


def _fail_copy_machine_delegated_run(run: LoRACopyMachineTrainingRun, issue: str) -> None:
    run.status = "failed"
    _append_unique_issue(run.errors, issue)
    run.finished_at = run.finished_at or _utc_now()
    run.updated_at = _utc_now()


def _sync_copy_machine_delegated_run(
    run: LoRACopyMachineTrainingRun,
    delegated: LoRATrainingRun,
) -> None:
    status_map: dict[LoRATrainingRunStatus, LoRACopyMachineTrainingRunStatus] = {
        "planned": "planned",
        "blocked": "blocked",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "failed",
    }
    run.status = status_map[delegated.status]
    run.command_args = list(delegated.command_args)
    run.log_path = delegated.log_path
    run.pid = delegated.pid
    run.exit_code = delegated.exit_code
    run.started_at = delegated.started_at
    run.finished_at = delegated.finished_at
    run.progress_current = delegated.progress_current
    run.progress_total = delegated.progress_total
    run.progress_percent = delegated.progress_percent
    run.progress_message = delegated.progress_message
    run.trained_model_paths = list(delegated.trained_model_paths)
    run.trained_model_evidence = _copy_machine_model_evidence(run.trained_model_paths)
    for warning in delegated.preflight.warnings:
        _append_unique_issue(run.warnings, warning)
    for error in delegated.preflight.errors:
        _append_unique_issue(run.errors, error)

    nonfinite_loss = _copy_machine_training_log_has_nonfinite_loss(delegated.log_path)
    if nonfinite_loss:
        run.status = "failed"
        _append_unique_issue(run.errors, "copy_machine_training_nonfinite_loss")
    elif delegated.status == "completed" and not delegated.trained_model_paths:
        run.status = "failed"
        _append_unique_issue(run.errors, "copy_machine_training_no_model_output")
    elif delegated.status == "failed":
        log_text = delegated.log_tail.casefold()
        if "lost the subprocess handle" in log_text:
            _append_unique_issue(run.errors, "copy_machine_training_process_lost")
        elif delegated.exit_code == -2:
            if "source file not found" in log_text:
                _append_unique_issue(run.errors, "copy_machine_training_no_model_output")
            else:
                _append_unique_issue(run.errors, "copy_machine_training_conversion_failed")
        elif delegated.exit_code not in {None, 0}:
            _append_unique_issue(run.errors, "copy_machine_training_command_failed")
    run.updated_at = _utc_now()


def _parse_training_progress(log_text: str) -> tuple[int | None, int | None, float | None, str] | None:
    last_progress: tuple[int | None, int | None, float | None, str] | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        percent = _parse_progress_percent(line)
        fraction = _parse_progress_fraction(line)
        if percent is None and fraction is None:
            continue
        current: int | None = None
        total: int | None = None
        if fraction is not None:
            current, total = fraction
            if percent is None and total > 0:
                percent = round((current / total) * 100, 2)
        last_progress = (current, total, percent, line[-240:])
    return last_progress


def _parse_progress_percent(line: str) -> float | None:
    matches = re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%", line)
    if not matches:
        return None
    value = float(matches[-1])
    return max(0.0, min(100.0, round(value, 2)))


def _parse_progress_fraction(line: str) -> tuple[int, int] | None:
    matches = re.findall(r"(?<![\d/])(\d{1,9})\s*/\s*(\d{1,9})(?![\d/])", line)
    for current_text, total_text in reversed(matches):
        current = int(current_text)
        total = int(total_text)
        if total > 0 and 0 <= current <= total:
            return current, total
    return None


def _append_log_tail_note(path: str, note: str) -> str:
    if not path:
        return note
    log_path = Path(path)
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"\n{note}\n")
    except OSError:
        return note
    return _read_text_tail(log_path)


class LoRACopyMachineSessionRepository:
    def __init__(self, path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    def list(self) -> list[LoRACopyMachineSession]:
        return sorted(self._read(), key=lambda session: session.updated_at, reverse=True)

    def get(self, session_id: str) -> LoRACopyMachineSession | None:
        return next((session for session in self._read() if session.id == session_id), None)

    def create(self, session: LoRACopyMachineSession) -> LoRACopyMachineSession:
        sessions = self._read(strict=True)
        sessions.insert(0, session)
        self._write(sessions)
        return session

    def save(self, session: LoRACopyMachineSession) -> LoRACopyMachineSession:
        sessions = self._read(strict=True)
        for index, current in enumerate(sessions):
            if current.id == session.id:
                sessions[index] = session
                break
        else:
            sessions.insert(0, session)
        self._write(sessions)
        return session

    def _read(self, *, strict: bool = False) -> list[LoRACopyMachineSession]:
        try:
            exists = self.path.exists()
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not exists:
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not isinstance(payload, list):
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable")
            return []
        sessions: list[LoRACopyMachineSession] = []
        for item in payload:
            if not isinstance(item, dict):
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
            try:
                sessions.append(LoRACopyMachineSession.model_validate(item))
            except ValueError:
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
        return sessions

    def _write(self, sessions: list[LoRACopyMachineSession]) -> None:
        payload = [session.model_dump(mode="json") for session in sessions]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc


class LoRACopyMachineTrainingRunRepository:
    def __init__(self, path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    def list(self) -> list[LoRACopyMachineTrainingRun]:
        return sorted(self._read(), key=lambda run: run.updated_at, reverse=True)

    def get(self, run_id: str) -> LoRACopyMachineTrainingRun | None:
        return next((run for run in self._read() if run.id == run_id), None)

    def create(self, run: LoRACopyMachineTrainingRun) -> LoRACopyMachineTrainingRun:
        runs = self._read(strict=True)
        runs.insert(0, run)
        self._write(runs)
        return run

    def save(self, run: LoRACopyMachineTrainingRun) -> LoRACopyMachineTrainingRun:
        runs = self._read(strict=True)
        for index, current in enumerate(runs):
            if current.id == run.id:
                runs[index] = run
                break
        else:
            runs.insert(0, run)
        self._write(runs)
        return run

    def _read(self, *, strict: bool = False) -> list[LoRACopyMachineTrainingRun]:
        try:
            exists = self.path.exists()
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not exists:
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not isinstance(payload, list):
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable")
            return []
        runs: list[LoRACopyMachineTrainingRun] = []
        for item in payload:
            if not isinstance(item, dict):
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
            try:
                runs.append(LoRACopyMachineTrainingRun.model_validate(item))
            except ValueError:
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
        return runs

    def _write(self, runs: list[LoRACopyMachineTrainingRun]) -> None:
        payload = [run.model_dump(mode="json") for run in runs]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc


class LoRATrainingDatasetRepository:
    def __init__(self, path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    def list(self) -> list[LoRATrainingDataset]:
        return sorted(self._read(), key=lambda dataset: dataset.updated_at, reverse=True)

    def get(self, dataset_id: str) -> LoRATrainingDataset | None:
        return next((dataset for dataset in self._read() if dataset.id == dataset_id), None)

    def create(self, *, name: str, preset_id: str) -> LoRATrainingDataset:
        preset = get_lora_training_preset(preset_id)
        if preset is None:
            raise ValueError("lora_training_preset_not_found")
        now = _utc_now()
        dataset = LoRATrainingDataset(
            id=uuid4().hex,
            name=name.strip() or "Untitled LoRA dataset",
            preset_id=preset.id,
            profile=preset.profile,
            training_type=preset.training_type,
            created_at=now,
            updated_at=now,
        )
        datasets = self._read(strict=True)
        datasets.insert(0, dataset)
        self._write(datasets)
        return dataset

    def add_images(
        self,
        dataset_id: str,
        images: list[LoRATrainingDatasetImageInput],
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        existing_paths = {item.image_path for item in dataset.items}
        now = _utc_now()
        for image in images:
            image_path = image.image_path.strip()
            if not image_path or image_path in existing_paths:
                continue
            dataset.items.append(
                LoRATrainingDatasetItem(
                    id=uuid4().hex,
                    image_path=image_path,
                    caption=image.caption.strip(),
                    tags=_normalize_tags(image.tags),
                    source_generation_id=image.source_generation_id,
                    source_output_path=image.source_output_path.strip(),
                    created_at=now,
                    updated_at=now,
                )
            )
            existing_paths.add(image_path)
        dataset.updated_at = now
        self._write(datasets)
        return dataset

    def update_caption_export_settings(
        self,
        dataset_id: str,
        settings: LoRATrainingCaptionExportSettings,
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        dataset.caption_export_settings = _normalized_caption_export_settings(settings)
        dataset.updated_at = _utc_now()
        self._write(datasets)
        return dataset

    def update_character_caption_policy(
        self,
        dataset_id: str,
        policy: LoRATrainingCharacterCaptionPolicy,
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        if dataset.training_type != "character_lora":
            raise ValueError("lora_training_character_caption_policy_not_applicable")
        dataset.character_caption_policy = _normalized_character_caption_policy(policy)
        dataset.updated_at = _utc_now()
        self._write(datasets)
        return dataset

    def update_caption(
        self,
        dataset_id: str,
        item_id: str,
        *,
        caption: str,
        tags: list[str],
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        item = next((entry for entry in dataset.items if entry.id == item_id), None)
        if item is None:
            return None
        now = _utc_now()
        item.caption = caption.strip()
        item.tags = _normalize_tags(tags)
        item.updated_at = now
        dataset.updated_at = now
        self._write(datasets)
        return dataset

    def upsert_caption_source(
        self,
        dataset_id: str,
        item_id: str,
        source: LoRATrainingCaptionSourceInput,
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        item = next((entry for entry in dataset.items if entry.id == item_id), None)
        if item is None:
            return None
        now = _utc_now()
        record = _caption_source_record_from_input(source, now=now)
        existing_index = next((index for index, entry in enumerate(item.caption_sources) if entry.id == record.id), None)
        if existing_index is None:
            item.caption_sources.append(record)
        else:
            created_at = item.caption_sources[existing_index].created_at
            item.caption_sources[existing_index] = record.model_copy(update={"created_at": created_at})
        item.updated_at = now
        dataset.updated_at = now
        self._write(datasets)
        return dataset

    def delete_caption_source(
        self,
        dataset_id: str,
        item_id: str,
        source_id: str,
    ) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        item = next((entry for entry in dataset.items if entry.id == item_id), None)
        if item is None:
            return None
        next_sources = [source for source in item.caption_sources if source.id != source_id]
        if len(next_sources) == len(item.caption_sources):
            return None
        now = _utc_now()
        item.caption_sources = next_sources
        item.updated_at = now
        dataset.updated_at = now
        self._write(datasets)
        return dataset

    def delete_item(self, dataset_id: str, item_id: str) -> LoRATrainingDataset | None:
        datasets = self._read(strict=True)
        dataset = next((item for item in datasets if item.id == dataset_id), None)
        if dataset is None:
            return None
        next_items = [item for item in dataset.items if item.id != item_id]
        if len(next_items) == len(dataset.items):
            return None
        dataset.items = next_items
        dataset.updated_at = _utc_now()
        self._write(datasets)
        return dataset

    def delete_dataset(self, dataset_id: str) -> bool:
        datasets = self._read(strict=True)
        next_datasets = [dataset for dataset in datasets if dataset.id != dataset_id]
        if len(next_datasets) == len(datasets):
            return False
        self._write(next_datasets)
        return True

    def _read(self, *, strict: bool = False) -> list[LoRATrainingDataset]:
        try:
            exists = self.path.exists()
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not exists:
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not isinstance(payload, list):
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable")
            return []
        datasets = []
        for item in payload:
            if not isinstance(item, dict):
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
            try:
                datasets.append(LoRATrainingDataset.model_validate(item))
            except ValueError:
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
        return datasets

    def _write(self, datasets: list[LoRATrainingDataset]) -> None:
        payload = [dataset.model_dump(mode="json") for dataset in datasets]
        try:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc


class LoRATrainingRunRepository:
    def __init__(self, path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc

    def list(self) -> list[LoRATrainingRun]:
        return sorted(self._read(), key=lambda run: run.updated_at, reverse=True)

    def get(self, run_id: str) -> LoRATrainingRun | None:
        return next((run for run in self._read() if run.id == run_id), None)

    def create(
        self,
        *,
        dataset: LoRATrainingDataset,
        materialized: MaterializedDataset,
        output_dir: str,
        output_name: str,
        working_dir: str,
        command_args: list[str],
        preflight: LoRATrainingRunPreflight,
    ) -> LoRATrainingRun:
        now = _utc_now()
        run = LoRATrainingRun(
            id=uuid4().hex,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            preset_id=dataset.preset_id,
            profile=dataset.profile,
            training_type=dataset.training_type,
            status="planned" if preflight.ready else "blocked",
            materialized_root=materialized.root_dir,
            image_dir=materialized.image_dir,
            dataset_config_path=materialized.dataset_config_path,
            sample_prompts_path=materialized.sample_prompts_path,
            output_dir=output_dir,
            output_name=output_name,
            working_dir=working_dir.strip(),
            command_args=command_args,
            preflight=preflight,
            created_at=now,
            updated_at=now,
        )
        runs = self._read(strict=True)
        runs.insert(0, run)
        self._write(runs)
        return run

    def save(self, run: LoRATrainingRun) -> None:
        runs = self._read(strict=True)
        replaced = False
        for index, existing in enumerate(runs):
            if existing.id == run.id:
                runs[index] = run
                replaced = True
                break
        if not replaced:
            runs.insert(0, run)
        self._write(runs)

    def delete(self, run_id: str) -> bool:
        runs = self._read(strict=True)
        next_runs = [run for run in runs if run.id != run_id]
        if len(next_runs) == len(runs):
            return False
        self._write(next_runs)
        return True

    def _read(self, *, strict: bool = False) -> list[LoRATrainingRun]:
        try:
            exists = self.path.exists()
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not exists:
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        except json.JSONDecodeError as exc:
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc
            return []
        if not isinstance(payload, list):
            if strict:
                raise LoRATrainingStorageError("lora_training_storage_unavailable")
            return []
        runs = []
        for item in payload:
            if not isinstance(item, dict):
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
            try:
                runs.append(LoRATrainingRun.model_validate(item))
            except ValueError:
                if strict:
                    raise LoRATrainingStorageError("lora_training_storage_unavailable")
                continue
        return runs

    def _write(self, runs: list[LoRATrainingRun]) -> None:
        payload = [run.model_dump(mode="json") for run in runs]
        try:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise LoRATrainingStorageError("lora_training_storage_unavailable") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


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


def _new_file_candidate_is_available(path: Path) -> bool:
    try:
        return not path.exists()
    except OSError:
        return False


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    for tag in tags:
        value = " ".join(str(tag).strip().split())
        if value and value not in normalized:
            normalized.append(value)
    return normalized[:64]


def _caption_source_sidecar_index(
    directory: Path,
    extension: str,
    *,
    recursive: bool,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    iterator = directory.rglob(f"*{extension}") if recursive else directory.glob(f"*{extension}")
    for path in sorted(iterator, key=lambda candidate: str(candidate).lower()):
        if not _path_is_file(path):
            continue
        name = path.name
        if not name.lower().endswith(extension.lower()):
            continue
        key = name[: -len(extension)].casefold()
        if key and key not in files:
            try:
                files[key] = path.resolve()
            except OSError:
                continue
    return files


def _caption_source_input_from_imported_text(
    text: str,
    *,
    source_type: LoRATrainingCaptionSourceType,
    content_kind: LoRATrainingCaptionSourceContentKind,
    label: str,
    confidence: float | None,
) -> LoRATrainingCaptionSourceInput | None:
    value = text.strip()
    if not value:
        return None
    if content_kind == "caption":
        return LoRATrainingCaptionSourceInput(
            source_type=source_type,
            label=label,
            caption=" ".join(value.split()),
            raw_text=value,
            confidence=confidence,
        )
    if content_kind == "json":
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            tags = _normalize_tags(_caption_source_json_tags(payload))
            caption = _caption_source_json_caption(payload)
            parsed_confidence = _caption_source_json_confidence(payload)
            return LoRATrainingCaptionSourceInput(
                source_type=source_type,
                label=label,
                tags=tags,
                caption=caption,
                raw_text=value[:4000],
                confidence=confidence if confidence is not None else parsed_confidence,
            )
        return None
    if content_kind == "anima_tagger_ron":
        parsed = _caption_source_input_from_anima_tagger_ron(
            value,
            source_type=source_type,
            label=label,
            confidence=confidence,
        )
        if parsed is not None:
            return parsed
        return None
    tags = _normalize_tags(_split_caption_assist_tags(value))
    if not tags:
        return None
    return LoRATrainingCaptionSourceInput(
        source_type=source_type,
        label=label,
        tags=tags,
        raw_text=value,
        confidence=confidence,
    )


def _caption_source_input_from_anima_tagger_ron(
    text: str,
    *,
    source_type: LoRATrainingCaptionSourceType,
    label: str,
    confidence: float | None,
) -> LoRATrainingCaptionSourceInput | None:
    manual_tags = _caption_source_ron_string_list(text, "manual_tags")
    auto_tags = _caption_source_ron_tag_entries(text, "auto_tags", include_score=True)
    booru_tags = _caption_source_ron_tag_entries(text, "booru_tags", include_score=False)
    tags = _normalize_tags([*manual_tags, *[entry[0] for entry in auto_tags], *[entry[0] for entry in booru_tags]])
    caption = _caption_source_ron_manual_caption(text) or _caption_source_ron_captions(text)
    if not tags and not caption:
        return None
    parsed_confidence = _caption_source_ron_average_score(auto_tags)
    return LoRATrainingCaptionSourceInput(
        source_type=source_type,
        label=label,
        tags=tags,
        caption=caption,
        raw_text=text[:4000],
        confidence=confidence if confidence is not None else parsed_confidence,
    )


def _anima_tagger_ron_text_for_dataset_item(
    dataset: LoRATrainingDataset,
    item: LoRATrainingDatasetItem,
    *,
    timestamp: str,
) -> str:
    settings = _normalized_caption_export_settings(dataset.caption_export_settings)
    manual_tags = _normalize_tags([
        *(item.tags or _split_caption_assist_tags(item.caption)),
        *settings.suppressed_tags,
        *settings.organizational_tags,
    ])
    auto_tags: list[tuple[str, float, str]] = []
    booru_tags: list[tuple[str, str]] = []
    captions: list[tuple[str, str]] = []
    for source in item.caption_sources:
        source_tags = _normalize_tags(source.tags)
        if source.source_type == "wd14_tagger":
            score = _clamped_caption_source_score(source.confidence)
            auto_tags.extend((tag, score, "general") for tag in source_tags)
        elif source.source_type == "booru_api":
            booru_tags.extend((tag, "general") for tag in source_tags)
        else:
            for tag in source_tags:
                if tag not in manual_tags:
                    manual_tags.append(tag)
        caption = " ".join(source.caption.strip().split())
        if caption:
            captions.append((_anima_tagger_caption_key(source, captions), caption))
    lines = ["("]
    lines.append(f"  manual_tags: {_ron_string_list(manual_tags)},")
    lines.append("  auto_tags: [")
    for tag, score, category in auto_tags:
        lines.append(
            f"    (tag: {_ron_string(tag)}, score: {score:.6g}, category: {_ron_string(category)}),"
        )
    lines.append("  ],")
    lines.append("  booru_tags: [")
    for tag, category in booru_tags:
        lines.append(f"    (tag: {_ron_string(tag)}, category: {_ron_string(category)}),")
    lines.append("  ],")
    if captions:
        lines.append("  captions: {")
        for key, caption in captions:
            lines.append(
                f"    {_ron_string(key)}: (caption: {_ron_string(caption)}, captioned_at: {_ron_string(timestamp)}),"
            )
        lines.append("  },")
    final_caption = " ".join(item.caption.strip().split())
    if final_caption:
        lines.append(f"  manual_caption: Some({_ron_string(final_caption)}),")
    if auto_tags:
        lines.append(
            f"  tagger: Some((model: {_ron_string('Asteria source evidence')}, tagged_at: {_ron_string(timestamp)})),"
        )
    if booru_tags:
        lines.append(
            f"  booru: Some((source: {_ron_string('Asteria source evidence')}, fetched_at: {_ron_string(timestamp)})),"
        )
    lines.append(")")
    return "\n".join(lines) + "\n"


def _anima_tagger_caption_key(
    source: LoRATrainingCaptionSourceRecord,
    existing: list[tuple[str, str]],
) -> str:
    base = _safe_slug(source.label or source.source_type).replace("-", "_")
    if not base:
        base = source.source_type
    keys = {key for key, _ in existing}
    if base not in keys:
        return base
    index = 2
    while f"{base}_{index}" in keys:
        index += 1
    return f"{base}_{index}"


def _clamped_caption_source_score(value: float | None) -> float:
    if value is None:
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _ron_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _ron_string_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_ron_string(value) for value in values) + "]"


def _caption_source_ron_string_list(text: str, field: str) -> list[str]:
    block = _caption_source_ron_field_block(text, field, "[", "]")
    if not block:
        return []
    return _caption_source_ron_strings(block)


def _caption_source_ron_tag_entries(text: str, field: str, *, include_score: bool) -> list[tuple[str, float | None]]:
    block = _caption_source_ron_field_block(text, field, "[", "]")
    if not block:
        return []
    entries: list[tuple[str, float | None]] = []
    for match in re.finditer(r"\((.*?)\)", block, re.DOTALL):
        entry = match.group(1)
        tag = _caption_source_ron_named_string(entry, "tag")
        if not tag:
            continue
        score = _caption_source_ron_named_number(entry, "score") if include_score else None
        entries.append((tag, score))
    return entries


def _caption_source_ron_manual_caption(text: str) -> str:
    value = _caption_source_ron_option_string(text, "manual_caption")
    return " ".join(value.split()) if value else ""


def _caption_source_ron_captions(text: str) -> str:
    block = _caption_source_ron_field_block(text, "captions", "{", "}")
    if not block:
        legacy_caption = _caption_source_ron_option_string(text, "caption")
        return " ".join(legacy_caption.split()) if legacy_caption else ""
    captions: list[str] = []
    for match in re.finditer(r'"(?:\\.|[^"\\])*"\s*:\s*\((.*?)\)', block, re.DOTALL):
        entry = match.group(1)
        if re.search(r"\bskip\s*:\s*true\b", entry):
            continue
        caption = _caption_source_ron_named_string(entry, "caption")
        if caption:
            captions.append(" ".join(caption.split()))
    return "\n".join(caption for caption in captions if caption)


def _caption_source_ron_average_score(entries: list[tuple[str, float | None]]) -> float | None:
    scores = [score for _, score in entries if score is not None]
    if not scores:
        return None
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _caption_source_ron_field_block(text: str, field: str, opener: str, closer: str) -> str:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*\{opener}", text)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
            continue
        if char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def _caption_source_ron_strings(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', text):
        raw = match.group(1)
        try:
            value = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            value = raw.replace('\\"', '"').replace("\\\\", "\\")
        value = " ".join(str(value).split())
        if value:
            values.append(value)
    return values


def _caption_source_ron_named_string(text: str, field: str) -> str:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*\"((?:\\.|[^\"\\])*)\"", text, re.DOTALL)
    if not match:
        return ""
    values = _caption_source_ron_strings(f'"{match.group(1)}"')
    return values[0] if values else ""


def _caption_source_ron_option_string(text: str, field: str) -> str:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*Some\(\"((?:\\.|[^\"\\])*)\"\)", text, re.DOTALL)
    if not match:
        return ""
    values = _caption_source_ron_strings(f'"{match.group(1)}"')
    return values[0] if values else ""


def _caption_source_ron_named_number(text: str, field: str) -> float | None:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _caption_source_json_tags(payload: dict) -> list[str]:
    candidates: list[str] = []
    for key in ("tags", "general_tags", "danbooru_tags", "booru_tags", "tag_string", "caption_tags"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)
        elif isinstance(value, str):
            candidates.extend(_split_caption_assist_tags(value))
    return candidates


def _caption_source_json_caption(payload: dict) -> str:
    for key in ("caption", "description", "qwen_caption", "vision_caption"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    return ""


def _caption_source_json_confidence(payload: dict) -> float | None:
    for key in ("confidence", "score", "probability"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    return None


def _caption_source_import_id(
    item_id: str,
    source_type: LoRATrainingCaptionSourceType,
    label: str,
    extension: str,
) -> str:
    key = _safe_slug(f"{source_type}-{label or 'source'}-{extension.replace('.', '-')}")
    return f"import-{key}-{item_id[:12]}"


def _caption_tag_stem(tag: str) -> str:
    value = " ".join(str(tag).strip().split())
    while value.startswith(("-", "_")):
        value = value[1:].strip()
    return value.casefold()


def _is_suppression_tag(tag: str) -> bool:
    return str(tag).strip().startswith("-") and bool(_caption_tag_stem(tag))


def _is_organizational_tag(tag: str) -> bool:
    return str(tag).strip().startswith("_") and bool(_caption_tag_stem(tag))


def _normalized_caption_export_settings(
    settings: LoRATrainingCaptionExportSettings,
) -> LoRATrainingCaptionExportSettings:
    return LoRATrainingCaptionExportSettings(
        suppressed_tags=[
            tag if _is_suppression_tag(tag) else f"-{tag}"
            for tag in _normalize_tags(settings.suppressed_tags)
            if _caption_tag_stem(tag)
        ],
        organizational_tags=[
            tag if _is_organizational_tag(tag) else f"_{tag}"
            for tag in _normalize_tags(settings.organizational_tags)
            if _caption_tag_stem(tag)
        ],
        min_source_confidence=max(0.0, min(1.0, settings.min_source_confidence)),
        exclude_rating_tags=settings.exclude_rating_tags,
        category_prefixing=settings.category_prefixing,
        output_mode=settings.output_mode,
    )


def _normalized_character_caption_policy(
    policy: LoRATrainingCharacterCaptionPolicy,
) -> LoRATrainingCharacterCaptionPolicy:
    excluded_identity_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in policy.excluded_identity_tags:
        tag = " ".join(raw_tag.strip().split())
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        excluded_identity_tags.append(tag)
    return LoRATrainingCharacterCaptionPolicy(
        trigger_token=" ".join(policy.trigger_token.strip().split()),
        excluded_identity_tags=excluded_identity_tags,
        variable_detail_guidance=" ".join(policy.variable_detail_guidance.strip().split()),
    )


def _settings_suppressed_stems(settings: LoRATrainingCaptionExportSettings | None) -> set[str]:
    if settings is None:
        return set()
    return {_caption_tag_stem(tag) for tag in settings.suppressed_tags if _caption_tag_stem(tag)}


def _settings_organizational_tags(settings: LoRATrainingCaptionExportSettings | None) -> list[str]:
    if settings is None:
        return []
    return _normalize_tags(settings.organizational_tags)


def _is_rating_tag(tag: str) -> bool:
    value = str(tag).strip().casefold()
    return value.startswith(("rating:", "rating_", "rating ")) or value in {
        "safe",
        "sensitive",
        "questionable",
        "explicit",
        "general",
    }


def _exportable_caption_tags(
    tags: list[str],
    *,
    settings: LoRATrainingCaptionExportSettings | None = None,
) -> list[str]:
    normalized = _normalize_tags(tags)
    suppressed = {_caption_tag_stem(tag) for tag in normalized if _is_suppression_tag(tag)}
    suppressed.update(_settings_suppressed_stems(settings))
    exportable: list[str] = []
    for tag in normalized:
        if _is_suppression_tag(tag) or _is_organizational_tag(tag):
            continue
        if _caption_tag_stem(tag) in suppressed:
            continue
        if settings and settings.exclude_rating_tags and _is_rating_tag(tag):
            continue
        exportable.append(tag)
    return exportable[:64]


def _category_prefixed_tags(
    tags: list[str],
    category: str,
    settings: LoRATrainingCaptionExportSettings | None = None,
) -> list[str]:
    normalized = _normalize_tags(tags)
    if not settings or not settings.category_prefixing:
        return normalized
    return [f"{category}:{tag}" for tag in normalized]


def _caption_source_record_from_input(
    source: LoRATrainingCaptionSourceInput,
    *,
    now: str,
) -> LoRATrainingCaptionSourceRecord:
    tags = _normalize_tags(source.tags)
    caption = source.caption.strip()
    raw_text = source.raw_text.strip()
    if not tags and not caption and not raw_text:
        raise ValueError("lora_training_caption_source_empty")
    return LoRATrainingCaptionSourceRecord(
        id=(source.id or uuid4().hex).strip() or uuid4().hex,
        source_type=source.source_type,
        label=source.label.strip()[:120],
        caption=caption,
        tags=tags,
        raw_text=raw_text[:4000],
        confidence=source.confidence,
        created_at=now,
        updated_at=now,
    )


def _caption_source_record_tags(
    item: LoRATrainingDatasetItem,
    settings: LoRATrainingCaptionExportSettings | None = None,
) -> list[str]:
    tags: list[str] = []
    for source in item.caption_sources:
        if (
            settings is not None
            and source.confidence is not None
            and source.confidence < settings.min_source_confidence
        ):
            continue
        tags.extend(source.tags)
    return _normalize_tags(tags)


def _caption_source_record_captions(item: LoRATrainingDatasetItem) -> list[str]:
    captions: list[str] = []
    for source in item.caption_sources:
        caption = " ".join(source.caption.strip().split())
        if caption and caption not in captions:
            captions.append(caption)
    return captions[:20]


def _caption_source_prompt_lines(item: LoRATrainingDatasetItem) -> list[str]:
    lines: list[str] = []
    for source in item.caption_sources:
        parts = [source.source_type]
        if source.label.strip():
            parts.append(source.label.strip())
        if source.confidence is not None:
            parts.append(f"confidence={source.confidence:g}")
        evidence: list[str] = []
        source_tags = _normalize_tags(source.tags)
        if source_tags:
            evidence.append(f"tags: {', '.join(source_tags)}")
        if source.caption.strip():
            evidence.append(f"caption: {' '.join(source.caption.strip().split())}")
        if source.raw_text.strip():
            evidence.append(f"raw: {' '.join(source.raw_text.strip().split())[:240]}")
        if evidence:
            lines.append(f"- {' / '.join(parts)} -> {'; '.join(evidence)}")
    return lines[:24]


def _caption_source_summary(
    dataset: LoRATrainingDataset,
    item: LoRATrainingDatasetItem,
    analysis: LoRATrainingDatasetItemAnalysis,
) -> LoRATrainingCaptionSourceSummary:
    settings = _normalized_caption_export_settings(dataset.caption_export_settings)
    manual_tags = _normalize_tags(item.tags)
    source_tags = _caption_source_record_tags(item, settings)
    source_captions = _caption_source_record_captions(item)
    candidate_tags = _normalize_tags([
        *manual_tags,
        *analysis.metadata_tags,
        *source_tags,
        *settings.suppressed_tags,
        *settings.organizational_tags,
    ])
    suppressed_tags = [tag for tag in candidate_tags if _is_suppression_tag(tag)]
    organizational_tags = [tag for tag in candidate_tags if _is_organizational_tag(tag)]
    if settings.category_prefixing:
        exportable_tags = _normalize_tags([
            *_category_prefixed_tags(_exportable_caption_tags(manual_tags, settings=settings), "manual", settings),
            *_category_prefixed_tags(_exportable_caption_tags(analysis.metadata_tags, settings=settings), "metadata", settings),
            *_category_prefixed_tags(_exportable_caption_tags(source_tags, settings=settings), "source", settings),
        ])
    else:
        exportable_tags = _exportable_caption_tags(candidate_tags, settings=settings)
    warnings: list[str] = []
    if suppressed_tags:
        warnings.append("suppressed_tags_excluded_from_caption")
    if organizational_tags:
        warnings.append("organizational_tags_excluded_from_caption")
    if len(exportable_tags) < len([tag for tag in candidate_tags if not _is_suppression_tag(tag)]):
        warnings.append("non_exportable_tags_filtered")
    return LoRATrainingCaptionSourceSummary(
        manual_tags=manual_tags,
        metadata_tags=analysis.metadata_tags,
        source_tags=source_tags,
        source_captions=source_captions,
        source_records=item.caption_sources,
        suppressed_tags=suppressed_tags,
        organizational_tags=organizational_tags,
        candidate_tags=candidate_tags,
        exportable_tags=exportable_tags,
        warnings=warnings,
    )


def _image_orientation(width: int, height: int) -> str:
    if abs(width - height) <= max(width, height) * 0.04:
        return "square"
    return "portrait" if height > width else "landscape"


def _dominant_tone(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    luma = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    if luma < 42:
        return "dark tone"
    if luma > 214:
        return "bright tone"
    if blue >= red + 28 and blue >= green + 18:
        return "cool tone"
    if red >= blue + 28 and red >= green - 8:
        return "warm tone"
    if green >= red + 24 and green >= blue + 12:
        return "green tone"
    return "neutral tone"


def _dataset_item_metadata_tags(
    item: LoRATrainingDatasetItem,
    image_path: Path,
    project_root: Path,
) -> tuple[list[str], list[str]]:
    tags: list[str] = []
    warnings: list[str] = []
    candidates = [image_path.with_suffix(".asteria.json")]
    if item.source_output_path.strip():
        candidates.append(_resolve_source_path(item.source_output_path, project_root).with_suffix(".asteria.json"))
    seen_paths: set[Path] = set()
    for sidecar_path in candidates:
        try:
            resolved = sidecar_path.resolve()
        except OSError:
            warnings.append(f"Could not read sidecar metadata: {sidecar_path}")
            continue
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not _path_is_file(resolved):
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.append(f"Could not read sidecar metadata: {resolved}")
            continue
        if isinstance(payload, dict):
            tags.extend(_tags_from_sidecar_payload(payload))
    tags.extend(_tags_from_embedded_image_metadata(image_path))
    return _normalize_tags(tags), warnings


def _tags_from_embedded_image_metadata(image_path: Path) -> list[str]:
    tags: list[str] = []
    try:
        with Image.open(image_path) as image:
            metadata_values = list(image.info.values())
            try:
                exif = image.getexif()
            except (OSError, SyntaxError, ValueError):
                exif = None
            if exif:
                metadata_values.extend(exif.values())
                metadata_values.extend(exif.get_ifd(ExifTags.IFD.Exif).values())
    except (OSError, ValueError):
        return []
    for value in metadata_values:
        text = metadata_value_to_text(value)
        if not text:
            continue
        prompt_text = _embedded_generation_prompt_text(text)
        if prompt_text:
            tags.extend(_prompt_tokens_for_lora_caption(prompt_text))
    return _normalize_tags(tags)


def _embedded_generation_prompt_text(text: str) -> str:
    if not any(marker in text for marker in ("Negative prompt:", "Steps:", "Sampler:", "CFG scale:", "Seed:")):
        return ""
    prompt = re.split(
        r"(?:Negative prompt:|Steps:|Sampler:|Schedule type:|CFG scale:|Seed:|Size:|Model hash:|Model:)",
        text,
        maxsplit=1,
    )[0]
    prompt = re.sub(r"<[^>]+>", " ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def _tags_from_sidecar_payload(payload: dict) -> list[str]:
    tags: list[str] = []
    for key in ("metadata_tags", "generated_tags", "tags"):
        value = payload.get(key)
        if isinstance(value, list):
            tags.extend(str(item) for item in value)
    for key in ("expanded_prompt", "prompt"):
        value = payload.get(key)
        if isinstance(value, str):
            tags.extend(_prompt_tokens_for_lora_caption(value))
    settings = payload.get("settings")
    if isinstance(settings, dict):
        for key in ("expanded_prompt", "prompt"):
            value = settings.get(key)
            if isinstance(value, str):
                tags.extend(_prompt_tokens_for_lora_caption(value))
    return _normalize_tags(tags)


def _prompt_tokens_for_lora_caption(prompt: str) -> list[str]:
    tags: list[str] = []
    for raw_token in re.split(r"[,;\n]", prompt):
        token = re.sub(r"[\(\)\[\]\{\}:]+", " ", raw_token)
        token = re.sub(r"\s+", " ", token).strip()
        if not token or len(token) > 80:
            continue
        if token.lower() in {"masterpiece", "best quality", "high quality", "low quality", "worst quality"}:
            continue
        tags.append(token)
    return _normalize_tags(tags)


def _caption_candidate_tags(existing_caption: str, tags: list[str]) -> list[str]:
    generic = {"asteria", "sdxl", "anima", "flux", "flux1", "flux2", "model", "image"}
    candidates = _prompt_tokens_for_lora_caption(existing_caption)
    for tag in tags:
        if tag.lower() in generic:
            continue
        candidates.append(tag)
    return _exportable_caption_tags(candidates)
