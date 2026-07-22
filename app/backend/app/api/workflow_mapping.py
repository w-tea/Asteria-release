from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import AppConfig
from app.schemas.generation import Txt2ImgRequest
from app.services.model_profiles import ModelProfile, ModelSlot
from app.services.sampling_catalog import sampler_labels, scheduler_labels
from app.services.upscaler_catalog import upscaler_labels

router = APIRouter(prefix="/api/workflow-mapping", tags=["workflow-mapping"])

CORE_TXT2IMG_FIELDS = {
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "cfg_scale",
    "sampler",
    "scheduler",
    "width",
    "height",
    "batch_size",
    "batch_count",
}

HIRES_TXT2IMG_FIELDS = {
    "hires_enabled",
    "hires_upscaler",
    "hires_steps",
    "hires_denoising_strength",
    "hires_scale",
    "hires_sampler",
    "hires_scheduler",
}

KNOWN_WORKFLOW_BUILDERS = {
    "ComfyTxt2ImgWorkflowBuilder._build_sdxl",
    "ComfyTxt2ImgWorkflowBuilder._build_flux",
    "ComfyTxt2ImgWorkflowBuilder._build_flux2",
    "ComfyTxt2ImgWorkflowBuilder._build_anima",
}

KNOWN_DYNAMIC_SECTIONS = {
    "lora_stack",
    "controlnet_units",
    "hires_fix",
    "detailer_units",
    "img2img",
    "inpaint",
    "regional_prompting",
    "remaster",
    "seed_resize",
}


class MappingValidationIssue(BaseModel):
    file: str
    message: str


class MappingValidationResponse(BaseModel):
    valid: bool
    issues: list[MappingValidationIssue]
    profiles: dict[str, dict[str, Any]]


@router.get("/validate", response_model=MappingValidationResponse)
def validate_mapping() -> MappingValidationResponse:
    config = AppConfig()
    issues: list[MappingValidationIssue] = []
    profiles: dict[str, dict[str, Any]] = {}
    model_profiles = _model_profiles(config.model_profiles_path, issues)
    profile_ids = {profile.id for profile in model_profiles}
    profiles_by_id = {profile.id: profile for profile in model_profiles}
    profile_setting_groups = {profile.id: set(profile.setting_groups) for profile in model_profiles}
    dependency_contracts = _dependency_contracts(config.model_dependencies_path, issues)
    dependency_ids = {dependency["id"] for dependency in dependency_contracts if dependency.get("id")}
    schema_profiles: set[str] = set()
    try:
        schema_paths = sorted(config.schema_dir.glob("*_txt2img.yaml"))
    except OSError:
        schema_paths = []
        issues.append(MappingValidationIssue(file=str(config.schema_dir), message="schema directory could not be scanned"))
    for schema_path in schema_paths:
        payload = _load_yaml_object(schema_path, issues, "schema")
        profile_value = payload.get("profile")
        if profile_value is None:
            profile = schema_path.name.removesuffix("_txt2img.yaml")
        elif isinstance(profile_value, str):
            profile = profile_value
        else:
            issues.append(MappingValidationIssue(file=str(schema_path), message="schema profile must be a string"))
            profile = schema_path.name.removesuffix("_txt2img.yaml")
        schema_profiles.add(profile)
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            issues.append(MappingValidationIssue(file=str(schema_path), message="fields must be an object"))
            fields = {}
        fields = _valid_schema_fields(schema_path, fields, issues)
        _validate_schema_backend_keys(
            schema_path,
            fields,
            profiles_by_id.get(profile),
            issues,
        )
        defaults = payload.get("defaults")
        if defaults is None:
            defaults = {
                field_id: field.get("default")
                for field_id, field in fields.items()
                if isinstance(field, dict) and "default" in field
            }
        if not isinstance(defaults, dict):
            issues.append(MappingValidationIssue(file=str(schema_path), message="defaults must be an object"))
            defaults = {}
        for field_id in defaults:
            if field_id not in fields:
                issues.append(MappingValidationIssue(file=str(schema_path), message=f"default has no field: {field_id}"))
        required_core_fields = set(CORE_TXT2IMG_FIELDS)
        if "hires_fix" in profile_setting_groups.get(profile, set()):
            required_core_fields.update(HIRES_TXT2IMG_FIELDS)
        for field_id in sorted(required_core_fields - set(fields)):
            issues.append(MappingValidationIssue(file=str(schema_path), message=f"schema missing core field: {field_id}"))
        mapped_fields = _mapped_fields(fields, profile)
        if profile not in profile_ids:
            issues.append(MappingValidationIssue(file=str(schema_path), message=f"profile missing from model profiles: {profile}"))
        mapping_path = config.mapping_dir / schema_path.name
        mapping_payload = {}
        mapping_dynamic_sections: list[str] = []
        node_contracts: dict[Any, Any] = {}
        mapping_builder_target: str | None = None
        mapping_exists = _validation_path_exists(mapping_path, issues, "workflow mapping")
        if mapping_exists is True:
            mapping_payload = _load_yaml_object(mapping_path, issues, "workflow mapping")
            mapping_profile = _optional_string_value(
                mapping_payload.get("profile"),
                mapping_path,
                "workflow mapping profile must be a string",
                issues,
            )
            if mapping_profile != profile:
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message=f"workflow mapping profile mismatch: expected {profile}",
                    )
                )
            expected_schema = _schema_reference(schema_path, config.root_dir)
            mapping_schema = _optional_string_value(
                mapping_payload.get("schema"),
                mapping_path,
                "workflow mapping schema must be a string",
                issues,
            )
            if mapping_schema != expected_schema:
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message=f"workflow mapping schema mismatch: expected {expected_schema}",
                    )
                )
            schema_mode = _optional_string_value(
                payload.get("mode"),
                schema_path,
                "schema mode must be a string",
                issues,
            ) or "txt2img"
            mapping_mode = _optional_string_value(
                mapping_payload.get("mode"),
                mapping_path,
                "workflow mapping mode must be a string",
                issues,
            ) or ""
            if mapping_mode and mapping_mode != schema_mode:
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message=f"workflow mapping mode mismatch: expected {schema_mode}",
                    )
                )
            builder = _optional_string_value(
                mapping_payload.get("builder"),
                mapping_path,
                "workflow mapping builder must be a string",
                issues,
            ) or ""
            mapping_builder_target = builder or None
            if builder and builder not in KNOWN_WORKFLOW_BUILDERS:
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message=f"workflow mapping unknown builder: {builder}",
                    )
                )
            nodes = mapping_payload.get("nodes", {})
            if not isinstance(nodes, dict):
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message="workflow mapping nodes must be an object",
                    )
                )
            else:
                _validate_string_map(mapping_path, nodes, "workflow mapping node mapping must be string-to-string", issues)
            class_types = mapping_payload.get("class_types", {})
            if not isinstance(class_types, dict):
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message="workflow mapping class_types must be an object",
                    )
                )
            else:
                _validate_string_map(
                    mapping_path,
                    class_types,
                    "workflow mapping class type mapping must be string-to-string",
                    issues,
                )
            node_contracts = mapping_payload.get("node_contracts", {})
            if not isinstance(node_contracts, dict):
                issues.append(
                    MappingValidationIssue(
                        file=str(mapping_path),
                        message="workflow mapping node_contracts must be an object",
                    )
                )
                node_contracts = {}
            else:
                _validate_node_contracts(mapping_path, node_contracts, issues)
            dynamic_sections = mapping_payload.get("dynamic_sections", [])
            if not isinstance(dynamic_sections, list):
                issues.append(MappingValidationIssue(file=str(mapping_path), message="workflow mapping dynamic_sections must be a list"))
                dynamic_sections = []
            for section in dynamic_sections:
                if not isinstance(section, str):
                    issues.append(MappingValidationIssue(file=str(mapping_path), message="workflow mapping dynamic section must be a string"))
                    continue
                section_id = section
                mapping_dynamic_sections.append(section_id)
                if section_id not in KNOWN_DYNAMIC_SECTIONS:
                    issues.append(
                        MappingValidationIssue(
                            file=str(mapping_path),
                            message=f"workflow mapping unknown dynamic section: {section_id}",
                        )
                    )
        elif mapping_exists is False:
            issues.append(MappingValidationIssue(file=str(mapping_path), message=f"workflow mapping missing for profile: {profile}"))
        profiles[profile] = {
            "schema": schema_path.as_posix(),
            "workflow_mapping": mapping_path.as_posix(),
            "field_count": len(fields),
            "default_count": len(defaults),
            "fields": mapped_fields,
            "model_profile": profile if profile in profile_ids else None,
            "builder": "mapping_driven" if mapping_payload else "missing_mapping",
            "builder_target": mapping_builder_target,
            "dynamic_sections": mapping_dynamic_sections,
            "node_contracts": sorted(node_contracts),
        }
    for profile in model_profiles:
        if profile.id not in schema_profiles:
            issues.append(
                MappingValidationIssue(
                    file=str(config.model_profiles_path),
                    message=f"profile missing txt2img schema: {profile.id}",
                )
            )
        for dependency_id in profile.dependency_ids:
            if dependency_id not in dependency_ids:
                issues.append(
                    MappingValidationIssue(
                        file=str(config.model_profiles_path),
                        message=f"profile {profile.id} references unknown dependency: {dependency_id}",
                    )
                )
    for dependency in dependency_contracts:
        dependency_id = str(dependency.get("id") or "<missing id>")
        dependency_profile = str(dependency.get("profile") or "")
        if dependency_profile and dependency_profile != "shared" and dependency_profile not in profile_ids:
            issues.append(
                MappingValidationIssue(
                    file=str(config.model_dependencies_path),
                    message=f"dependency {dependency_id} references unknown profile: {dependency_profile}",
                )
            )
        slot_id = dependency.get("slot_id")
        if dependency_profile in profiles_by_id and slot_id:
            profile_slots = {slot.id for slot in profiles_by_id[dependency_profile].model_slots}
            if slot_id not in profile_slots:
                issues.append(
                    MappingValidationIssue(
                        file=str(config.model_dependencies_path),
                        message=f"dependency {dependency_id} targets unknown slot for profile {dependency_profile}: {slot_id}",
                    )
                )
        if dependency_id in dependency_ids:
            for profile in model_profiles:
                if dependency_id in profile.dependency_ids and dependency_profile not in {profile.id, "shared"}:
                    issues.append(
                        MappingValidationIssue(
                            file=str(config.model_profiles_path),
                            message=f"profile {profile.id} dependency {dependency_id} belongs to profile {dependency_profile}",
                        )
                    )
    return MappingValidationResponse(valid=not issues, issues=issues, profiles=profiles)


def _mapped_fields(fields: dict[str, Any], profile: str = "") -> dict[str, Any]:
    mapped = {field_id: dict(field) for field_id, field in fields.items() if isinstance(field, dict)}
    for field_id in ("sampler",):
        if field_id in mapped:
            mapped[field_id]["options"] = sampler_labels()
            mapped[field_id]["options_source"] = "sampling_catalog.samplers"
    for field_id in ("scheduler",):
        if field_id in mapped:
            if profile == "flux2":
                mapped[field_id]["options"] = ["Flux2"]
                mapped[field_id]["options_source"] = "profile_schema"
            else:
                mapped[field_id]["options"] = scheduler_labels()
                mapped[field_id]["options_source"] = "sampling_catalog.schedulers"
    if "hires_sampler" in mapped:
        mapped["hires_sampler"]["options"] = ["Use same sampler", *sampler_labels()]
        mapped["hires_sampler"]["options_source"] = "sampling_catalog.samplers"
    if "hires_scheduler" in mapped:
        if profile == "flux2":
            mapped["hires_scheduler"]["options"] = ["Use same scheduler"]
            mapped["hires_scheduler"]["options_source"] = "profile_schema"
        else:
            mapped["hires_scheduler"]["options"] = ["Use same scheduler", *scheduler_labels()]
            mapped["hires_scheduler"]["options_source"] = "sampling_catalog.schedulers"
    if "hires_upscaler" in mapped:
        mapped["hires_upscaler"]["options"] = upscaler_labels()
        mapped["hires_upscaler"]["options_source"] = "upscaler_catalog"
    return mapped


def _optional_string_value(
    value: Any,
    path: Path,
    message: str,
    issues: list[MappingValidationIssue],
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    issues.append(MappingValidationIssue(file=str(path), message=message))
    return None


def _valid_schema_fields(
    schema_path: Path,
    fields: dict[Any, Any],
    issues: list[MappingValidationIssue],
) -> dict[str, dict[str, Any]]:
    valid: dict[str, dict[str, Any]] = {}
    for field_id, field in fields.items():
        if not isinstance(field_id, str):
            issues.append(MappingValidationIssue(file=str(schema_path), message="schema field id must be a string"))
            continue
        if not isinstance(field, dict):
            issues.append(MappingValidationIssue(file=str(schema_path), message=f"schema field must be an object: {field_id}"))
            continue
        valid[field_id] = field
    return valid


def _validate_schema_backend_keys(
    schema_path: Path,
    fields: dict[str, dict[str, Any]],
    model_profile: ModelProfile | None,
    issues: list[MappingValidationIssue],
) -> None:
    allowed_keys = set(Txt2ImgRequest.model_fields)
    if model_profile is not None:
        allowed_keys.update(slot.id for slot in model_profile.model_slots)
    seen: set[str] = set()
    for field_id, field in fields.items():
        backend_key = field.get("backend_key", field_id)
        if not isinstance(backend_key, str) or not backend_key:
            issues.append(
                MappingValidationIssue(
                    file=str(schema_path),
                    message=f"schema backend_key must be a non-empty string: {field_id}",
                )
            )
            continue
        if backend_key in seen:
            issues.append(
                MappingValidationIssue(
                    file=str(schema_path),
                    message=f"schema duplicate backend_key: {backend_key}",
                )
            )
        else:
            seen.add(backend_key)
        if backend_key not in allowed_keys:
            issues.append(
                MappingValidationIssue(
                    file=str(schema_path),
                    message=(
                        "schema backend_key is not a request field or profile model slot: "
                        f"{backend_key}"
                    ),
                )
            )


def _validate_string_map(
    path: Path,
    values: dict[Any, Any],
    message: str,
    issues: list[MappingValidationIssue],
) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            issues.append(MappingValidationIssue(file=str(path), message=message))


def _validate_node_contracts(
    path: Path,
    values: dict[Any, Any],
    issues: list[MappingValidationIssue],
) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            issues.append(
                MappingValidationIssue(
                    file=str(path),
                    message="workflow mapping node contract must be a named object",
                )
            )
            continue
        for field_name in ("required_inputs", "accepted_inputs"):
            field_value = value.get(field_name, [])
            if not isinstance(field_value, list) or any(not isinstance(item, str) for item in field_value):
                issues.append(
                    MappingValidationIssue(
                        file=str(path),
                        message=f"workflow mapping node contract {field_name} must be a string list: {key}",
                    )
                )
        required_input_values = value.get("required_input_values", {})
        if not isinstance(required_input_values, dict):
            issues.append(
                MappingValidationIssue(
                    file=str(path),
                    message=f"workflow mapping node contract required_input_values must be an object: {key}",
                )
            )
        else:
            for input_name, input_values in required_input_values.items():
                if (
                    not isinstance(input_name, str)
                    or not isinstance(input_values, list)
                    or any(not isinstance(item, str) for item in input_values)
                ):
                    issues.append(
                        MappingValidationIssue(
                            file=str(path),
                            message=(
                                "workflow mapping node contract required_input_values must map "
                                f"input names to string lists: {key}"
                            ),
                        )
                    )
        outputs = value.get("outputs_by_index", {})
        if not isinstance(outputs, dict):
            issues.append(
                MappingValidationIssue(
                    file=str(path),
                    message=f"workflow mapping node contract outputs_by_index must be an object: {key}",
                )
            )
            continue
        for index, output_type in outputs.items():
            try:
                parsed_index = int(index)
            except (TypeError, ValueError):
                parsed_index = -1
            if parsed_index < 0 or not isinstance(output_type, str):
                issues.append(
                    MappingValidationIssue(
                        file=str(path),
                        message=f"workflow mapping node contract output must map a non-negative index to a string: {key}",
                    )
                )


def _dependency_contracts(path: Path, issues: list[MappingValidationIssue]) -> list[dict[str, Any]]:
    if not _validation_path_exists(path, issues, "model dependencies"):
        return []
    raw = _load_yaml_object(path, issues, "model dependencies")
    dependencies = raw.get("dependencies", [])
    if not isinstance(dependencies, list):
        issues.append(MappingValidationIssue(file=str(path), message="model dependencies list must be an array"))
        dependencies = []
    contracts: list[dict[str, Any]] = []
    for item in dependencies:
        if not isinstance(item, dict):
            continue
        dependency_id = item.get("id")
        if not isinstance(dependency_id, str):
            issues.append(MappingValidationIssue(file=str(path), message="model dependency id must be a string"))
            continue
        dependency_profile = item.get("profile")
        if dependency_profile is not None and not isinstance(dependency_profile, str):
            issues.append(MappingValidationIssue(file=str(path), message="model dependency profile must be a string"))
            continue
        target = item.get("target") if isinstance(item.get("target"), dict) else {}
        slot_id = target.get("slot_id")
        if slot_id is not None and not isinstance(slot_id, str):
            issues.append(MappingValidationIssue(file=str(path), message="model dependency target slot_id must be a string"))
            slot_id = None
        contracts.append(
            {
                "id": dependency_id,
                "profile": dependency_profile,
                "slot_id": slot_id,
            }
        )
    return contracts


def _model_profiles(path: Path, issues: list[MappingValidationIssue]) -> list[ModelProfile]:
    if not _validation_path_exists(path, issues, "model profiles"):
        return []
    raw = _load_yaml_object(path, issues, "model profiles")
    profiles = raw.get("profiles", [])
    if not isinstance(profiles, list):
        issues.append(MappingValidationIssue(file=str(path), message="model profiles list must be an array"))
        profiles = []
    parsed: list[ModelProfile] = []
    for item in profiles:
        if not isinstance(item, dict):
            issues.append(MappingValidationIssue(file=str(path), message="model profile must be an object"))
            continue
        if not isinstance(item.get("id"), str) or not isinstance(item.get("label"), str):
            issues.append(MappingValidationIssue(file=str(path), message="model profile id and label must be strings"))
            continue
        profile_id = item["id"]
        model_slots = item.get("model_slots", [])
        if not isinstance(model_slots, list):
            issues.append(
                MappingValidationIssue(
                    file=str(path),
                    message=f"model profile {profile_id or '<missing id>'} model_slots must be an array",
                )
            )
            model_slots = []
        dependency_ids = _valid_profile_string_list(
            path,
            item.get("dependency_ids", []) or [],
            f"model profile {profile_id} dependency id must be a string",
            issues,
        )
        setting_groups = _valid_profile_string_list(
            path,
            item.get("setting_groups", []) or [],
            f"model profile {profile_id} setting group must be a string",
            issues,
        )
        parsed.append(
            ModelProfile(
                id=profile_id,
                label=item["label"],
                dependency_ids=dependency_ids,
                model_slots=[
                    ModelSlot(
                        id=slot["id"],
                        label=slot["label"],
                        type=slot["type"],
                        required=bool(slot.get("required", True)),
                    )
                    for slot in model_slots
                    if _valid_profile_slot(path, profile_id, slot, issues)
                ],
                setting_groups=setting_groups,
            )
        )
    return parsed


def _valid_profile_string_list(
    path: Path,
    values: Any,
    message: str,
    issues: list[MappingValidationIssue],
) -> list[str]:
    if not isinstance(values, list):
        issues.append(MappingValidationIssue(file=str(path), message=message))
        return []
    valid: list[str] = []
    for value in values:
        if isinstance(value, str):
            valid.append(value)
        else:
            issues.append(MappingValidationIssue(file=str(path), message=message))
    return valid


def _valid_profile_slot(path: Path, profile_id: str, slot: Any, issues: list[MappingValidationIssue]) -> bool:
    if not (
        isinstance(slot, dict)
        and isinstance(slot.get("id"), str)
        and isinstance(slot.get("label"), str)
        and isinstance(slot.get("type"), str)
    ):
        issues.append(
            MappingValidationIssue(
                file=str(path),
                message=f"model profile {profile_id} slot id/label/type must be strings",
            )
        )
        return False
    if "required" in slot and not isinstance(slot.get("required"), bool):
        issues.append(
            MappingValidationIssue(
                file=str(path),
                message=f"model profile {profile_id} slot required must be a boolean",
            )
        )
        return False
    return True


def _load_yaml_object(path: Path, issues: list[MappingValidationIssue], label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        issues.append(MappingValidationIssue(file=str(path), message=f"{label} YAML could not be parsed"))
        return {}
    if isinstance(payload, dict):
        return payload
    issues.append(MappingValidationIssue(file=str(path), message=f"{label} must be an object"))
    return {}


def _validation_path_exists(path: Path, issues: list[MappingValidationIssue], label: str) -> bool | None:
    try:
        return path.exists()
    except OSError:
        issues.append(MappingValidationIssue(file=str(path), message=f"{label} could not be checked"))
        return None


def _schema_reference(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
