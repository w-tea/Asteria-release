from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class WorkflowMappingNotFound(Exception):
    pass


@dataclass(frozen=True)
class WorkflowNodeContract:
    required_inputs: tuple[str, ...] = ()
    accepted_inputs: tuple[str, ...] = ()
    required_input_values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    outputs_by_index: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowMapping:
    id: str
    profile: str
    mode: str
    nodes: dict[str, str] = field(default_factory=dict)
    class_types: dict[str, str] = field(default_factory=dict)
    node_contracts: dict[str, WorkflowNodeContract] = field(default_factory=dict)
    dynamic_sections: list[str] = field(default_factory=list)

    def node(self, key: str, fallback: str) -> str:
        return str(self.nodes.get(key) or fallback)

    def class_type(self, key: str, fallback: str) -> str:
        return str(self.class_types.get(key) or fallback)

    def node_contract(self, key: str) -> WorkflowNodeContract | None:
        return self.node_contracts.get(key)


class WorkflowMappingRepository:
    def __init__(self, mapping_dir: Path) -> None:
        self.mapping_dir = mapping_dir

    def get_txt2img(self, profile: str) -> WorkflowMapping:
        mapping_profile = "flux" if profile == "flux1" else profile
        path = self.mapping_dir / f"{mapping_profile}_txt2img.yaml"
        try:
            mapping_exists = path.exists()
        except OSError:
            mapping_exists = False
        if not mapping_exists:
            raise WorkflowMappingNotFound(profile)
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            raise WorkflowMappingNotFound(profile) from None
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise WorkflowMappingNotFound(profile)
        mapping_id = raw["id"]
        mapping_profile_value = raw.get("profile")
        mapping_profile = mapping_profile_value if isinstance(mapping_profile_value, str) else profile
        mode = raw.get("mode")
        if not isinstance(mode, str):
            mode = "txt2img"
        nodes = raw.get("nodes") or {}
        if not isinstance(nodes, dict):
            nodes = {}
        class_types = raw.get("class_types") or {}
        if not isinstance(class_types, dict):
            class_types = {}
        node_contracts = raw.get("node_contracts") or {}
        if not isinstance(node_contracts, dict):
            node_contracts = {}
        dynamic_sections = raw.get("dynamic_sections") or []
        if not isinstance(dynamic_sections, list):
            dynamic_sections = []
        return WorkflowMapping(
            id=mapping_id,
            profile=profile if profile == "flux1" else mapping_profile,
            mode=mode,
            nodes=self._string_map(nodes),
            class_types=self._string_map(class_types),
            node_contracts=self._node_contracts(node_contracts),
            dynamic_sections=[item for item in dynamic_sections if isinstance(item, str)],
        )

    def _string_map(self, values: dict[Any, Any]) -> dict[str, str]:
        return {
            key: value
            for key, value in values.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _node_contracts(self, values: dict[Any, Any]) -> dict[str, WorkflowNodeContract]:
        contracts: dict[str, WorkflowNodeContract] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            required_inputs = value.get("required_inputs") or []
            accepted_inputs = value.get("accepted_inputs") or []
            required_input_values = value.get("required_input_values") or {}
            outputs_by_index = value.get("outputs_by_index") or {}
            if not isinstance(required_inputs, list) or not isinstance(accepted_inputs, list):
                continue
            if not isinstance(required_input_values, dict) or not isinstance(outputs_by_index, dict):
                continue
            parsed_input_values = {
                input_name: tuple(item for item in input_values if isinstance(item, str))
                for input_name, input_values in required_input_values.items()
                if isinstance(input_name, str) and isinstance(input_values, list)
            }
            parsed_outputs: dict[int, str] = {}
            for index, output_type in outputs_by_index.items():
                try:
                    parsed_index = int(index)
                except (TypeError, ValueError):
                    continue
                if parsed_index >= 0 and isinstance(output_type, str):
                    parsed_outputs[parsed_index] = output_type
            contracts[key] = WorkflowNodeContract(
                required_inputs=tuple(item for item in required_inputs if isinstance(item, str)),
                accepted_inputs=tuple(item for item in accepted_inputs if isinstance(item, str)),
                required_input_values=parsed_input_values,
                outputs_by_index=parsed_outputs,
            )
        return contracts


def workflow_node_contract_issues(
    object_info: dict[str, Any],
    logical_name: str,
    class_type: str,
    contract: WorkflowNodeContract,
) -> list[str]:
    node_info = object_info.get(class_type)
    if not class_type or not isinstance(node_info, dict):
        return [f"{logical_name}:class"]
    inputs = node_info.get("input")
    if not isinstance(inputs, dict):
        inputs = {}
    required = inputs.get("required")
    optional = inputs.get("optional")
    required_names = set(required) if isinstance(required, dict) else set()
    optional_names = set(optional) if isinstance(optional, dict) else set()
    available_names = required_names | optional_names
    issues = [
        f"{logical_name}:input:{name}"
        for name in contract.required_inputs
        if name not in required_names
    ]
    issues.extend(
        f"{logical_name}:input:{name}"
        for name in contract.accepted_inputs
        if name not in available_names
    )
    for input_name, expected_values in contract.required_input_values.items():
        available_values = _node_input_values(node_info, input_name)
        issues.extend(
            f"{logical_name}:value:{input_name}:{expected}"
            for expected in expected_values
            if expected not in available_values
        )
    outputs = node_info.get("output")
    output_values = outputs if isinstance(outputs, list) else []
    for index, expected in contract.outputs_by_index.items():
        if index >= len(output_values) or output_values[index] != expected:
            issues.append(f"{logical_name}:output:{index}")
    return issues


def _node_input_values(node_info: dict[str, Any], input_name: str) -> set[str]:
    inputs = node_info.get("input")
    if not isinstance(inputs, dict):
        return set()
    definition: Any = None
    for section_name in ("required", "optional"):
        section = inputs.get(section_name)
        if isinstance(section, dict) and input_name in section:
            definition = section[input_name]
            break
    if not isinstance(definition, list) or not definition:
        return set()
    values = definition[0]
    if values == "COMBO" and len(definition) > 1 and isinstance(definition[1], dict):
        values = definition[1].get("options")
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}
