"""Compact Assistant presentations without changing core request validation."""
from __future__ import annotations

from copy import deepcopy


_IMAGE_JOB_ACTIONS = {'generation.start', 'generation.jobs.get', 'generation.jobs.cancel', 'generation.jobs.retry',
                      'generation.branches.start'}
_IMAGE_DRAFT_INPUTS = {'Txt2ImgRequest': 'txt2img', 'Img2ImgRequest': 'img2img', 'InpaintRequest': 'inpaint'}


def compact_schema(schema):
    """Remove generated titles, not field names, defaults, or validation rules."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key == 'title':
            continue
        if key in {'properties', '$defs', 'definitions', 'patternProperties', 'dependentSchemas'}:
            result[key] = {name: compact_schema(child) for name, child in value.items()}
        elif key in {'allOf', 'anyOf', 'oneOf', 'prefixItems'}:
            result[key] = [compact_schema(child) for child in value]
        elif key in {'items', 'additionalProperties', 'not', 'if', 'then', 'else', 'contains'}:
            result[key] = compact_schema(value)
        else:
            result[key] = deepcopy(value)
    return result


def compact_loopback_tool_schema(schema):
    """Share repeated request fields in the local Chat Completions schema only."""
    result = compact_schema(schema)
    definitions = result.get('$defs', {})
    variants = [definitions.get(name) for name in ('Txt2ImgRequest', 'Img2ImgRequest', 'InpaintRequest')]
    common_name = 'GenerationInputCommon'
    # Additional/object-wide restrictions must not be split across allOf.
    if common_name in definitions or any(
        not isinstance(variant, dict) or variant.get('type') != 'object'
        or not isinstance(variant.get('properties'), dict) or variant.keys() - {'type', 'properties', 'required'}
        for variant in variants
    ):
        return result
    properties = [variant['properties'] for variant in variants]
    common = {key: value for key, value in properties[0].items()
              if all(key in candidate and candidate[key] == value for candidate in properties[1:])}
    if common:
        definitions[common_name] = {'type': 'object', 'properties': common}
        for variant, fields in zip(variants, properties):
            variant['properties'] = {key: value for key, value in fields.items() if key not in common}
            variant['allOf'] = [{'$ref': '#/$defs/' + common_name}]
    return result


def _local_ref_names(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _local_ref_names(value)
    elif isinstance(node, list):
        for value in node:
            yield from _local_ref_names(value)
    elif isinstance(node, str) and node.startswith('#/$defs/'):
        yield node.removeprefix('#/$defs/').split('/', 1)[0]


def _prune_unused_definitions(schema):
    definitions = schema['$defs']
    reachable = set()
    pending = list(_local_ref_names({key: value for key, value in schema.items() if key != '$defs'}))
    while pending:
        name = pending.pop()
        if name not in reachable and name in definitions:
            reachable.add(name)
            pending.extend(_local_ref_names(definitions[name]))
    schema['$defs'] = {name: value for name, value in definitions.items() if name in reachable}


def compact_loopback_tool_catalog(tools):
    """Share image schemas and avoid describing server receipts as editable inputs."""
    schemas = {tool.name: compact_loopback_tool_schema(tool.input_schema) for tool in tools}
    for schema in schemas.values():
        definitions = schema.get('$defs', {})
        advisory = definitions.get('ResourceAdvisory', {})
        authorization = definitions.get('ResourceAdvisoryAuthorization', {})
        if (advisory.get('type') == 'object'
                and authorization.get('properties', {}).get('advisory') == {'$ref': '#/$defs/ResourceAdvisory'}):
            # Caller returns an existing, fingerprinted server result. The
            # permission decision and full validation remain in the core.
            definitions['ResourceAdvisory'] = {'type': 'object', 'description': (
                'Copy the advisory returned by Asteria for this operation unchanged. '
                'Do not invent or modify it. Acknowledge only when the user has accepted the warning.')}
            _prune_unused_definitions(schema)

    canonical = schemas.get('generation.drafts.create', {}).get('$defs')
    if not canonical or not _IMAGE_DRAFT_INPUTS.keys() <= canonical.keys():
        return schemas
    for action in ('generation.drafts.update', 'generation.repeats.update'):
        schema = schemas.get(action)
        if not schema or schema.get('$defs') != canonical:
            continue
        definitions = schema['$defs']
        for name, mode in _IMAGE_DRAFT_INPUTS.items():
            definitions[name] = {
                'type': 'object',
                'description': f'Use the complete {mode} input of generation__drafts__create in this tool list. '
                               'Keep unchanged saved settings when replacing an existing input.',
            }
        # Only discard definitions unreachable after removing the duplicated
        # inputs. The wrapper's discriminator mappings keep their normal refs.
        _prune_unused_definitions(schema)
    return schemas


def text_tool_result(name, result):
    """Binary previews are not prose; keep the original core receipt untouched."""
    presented = deepcopy(result)
    data = presented.get('data')
    if not isinstance(data, dict):
        return presented
    # Explicit image-reading tools send these bytes as image blocks instead.
    data.pop('image', None)
    if name in {'generation.history.restore_draft', 'generation.drafts.derive', 'media.history.restore_draft', 'media.drafts.save'} and isinstance(data.get('document'), dict):
        # The next operation only needs the saved identity/revision. Leave the
        # complete input in the core receipt and redundant default fields out
        # of the provider context; the existing read tool still returns them.
        document = data.pop('document')
        data['mode'] = document.get('mode')
        data['read_settings_with'] = 'generation.drafts.get'
    if name in _IMAGE_JOB_ACTIONS:
        # Progress/current preview and saved outputs suffice to track a job.
        # Execution diagnostics remain available from the dedicated history.
        data.pop('preview_paths', None)
        generation = data.get('generation')
        if isinstance(generation, dict) and 'settings' in generation:
            generation.pop('settings')
            generation['read_settings_with'] = 'generation.history.get'
    if name == 'models.library.list':
        for item in data.get('items', []):
            metadata = item.get('metadata')
            if isinstance(metadata, dict) and 'modelspec.thumbnail' in metadata:
                metadata['modelspec.thumbnail'] = '[embedded thumbnail omitted from text]'
    return presented
