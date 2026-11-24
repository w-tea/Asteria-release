"""Portable conversation text and references, not provider or transport logs."""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit, urlunsplit


_URL = re.compile(r"""(?:https?://|(?<![\w/])/(?!/))[^\s<>()"'\x60]+""", re.I)
_IDENTIFIER = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9:._-]*\Z')
_AUTH_PARAMETERS = {'access_token', 'api_key', 'apikey', 'authorization', 'id_token', 'oauth_token',
                    'password', 'refresh_token', 'sig', 'signature', 'token', 'x-amz-security-token',
                    'x-amz-signature', 'x-amz-credential', 'x-goog-signature', 'x-goog-credential'}


def _export_text(value: str) -> str:
    def url_text(match: re.Match) -> str:
        raw = match.group()
        try:
            parts = urlsplit(raw)
        except ValueError:
            return '[URL omitted]'
        def authenticated(part: str) -> bool:
            return any(key.casefold() in _AUTH_PARAMETERS for key, _ in parse_qsl(part, keep_blank_values=True))
        query, fragment = authenticated(parts.query), authenticated(parts.fragment)
        if not query and not fragment and '@' not in parts.netloc:
            return raw
        return urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path,
                           '' if query else parts.query, '' if fragment else parts.fragment))
    return _URL.sub(url_text, value)


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _attachment(raw: dict, kind: str) -> dict | None:
    if kind == 'images_attached':
        asset_id = _identifier(raw.get('asset_id'))
        if asset_id is None:
            return None
        identity = {'asset_id': asset_id, 'kind': 'image'}
    else:
        generation, artifact = raw.get('generation_id'), _identifier(raw.get('artifact_id'))
        if type(generation) is not int or generation < 1 or not artifact or raw.get('kind') not in {'image', 'video', 'audio'}:
            return None
        identity = {'generation_id': generation, 'artifact_id': artifact, 'kind': raw['kind']}
    filename = raw.get('filename')
    if isinstance(filename, str):
        identity['filename'] = _export_text(filename.replace('\\', '/').rpartition('/')[2])
    mime_type = raw.get('mime_type')
    if isinstance(mime_type, str) and mime_type.startswith(('image/', 'video/', 'audio/')):
        identity['mime_type'] = mime_type
    return identity


def _event_projection(kind: str, payload: dict) -> dict:
    if kind == 'draft_saved':
        draft = payload.get('draft') or {}
        return {'draft_id': _identifier(draft.get('id')), 'revision': draft.get('revision')}
    if kind == 'error':
        return {'code': _identifier(payload.get('code'))}
    if kind in {'tool_call', 'tool_result'}:
        result = payload.get('result')
        result = result if isinstance(result, dict) else {}
        data = result.get('data')
        data = data if isinstance(data, dict) else {}
        operation = data.get('operation')
        if isinstance(operation, dict):
            data = operation
        generation_ids = [identity for identity in (data.get('generation_ids') or []) if type(identity) is int]
        if payload.get('name') in {'library.items.compare', 'library.items.update_many'}:
            generation_ids.extend(item['generation_id'] for item in (data.get('items') or [])
                                  if isinstance(item, dict) and type(item.get('generation_id')) is int)
        if payload.get('name') == 'library.items.delete':
            generation_ids.extend(identity for identity in (data.get('deleted_ids') or []) if type(identity) is int)
        return {key: value for key, value in {
            'name': _identifier(payload.get('name')), 'call_id': _identifier(payload.get('call_id')),
            'id': _identifier(data.get('id') or data.get('operation_id')),
            'generation_id': data.get('generation_id') if type(data.get('generation_id')) is int else None,
            'generation_ids': list(dict.fromkeys(generation_ids)),
            'artifact_id': _identifier(data.get('artifact_id')),
            'export_id': _identifier(data.get('export_id')) if data.get('status') == 'completed' else None,
            'artifact_refs': [ref for ref in (data.get('artifact_refs') or [])
                              if isinstance(ref, str) and re.fullmatch(r'history:[0-9]+', ref)],
            'status_code': result.get('status_code') if type(result.get('status_code')) is int else None,
        }.items() if value is not None}
    return {}


def project_conversation(source: dict) -> dict:
    conversation = {**source['conversation'], 'title': _export_text(source['conversation']['title'])}
    attachments: dict[tuple, dict] = {}
    turns = []
    for turn in source['turns']:
        events, pending = [], []

        def flush_text(final: dict | None = None) -> None:
            if final is not None or pending:
                text = final['payload'].get('text') if final is not None else ''
                if not text:
                    text = ''.join(str(row['payload'].get('text', '')) for row in pending)
                if text:
                    events.append({**(final if final is not None else pending[0]), 'payload': {'text': _export_text(text)}})
                pending.clear()

        for event in turn['events']:
            kind, payload = event['kind'], event['payload']
            if kind == 'text_delta':
                pending.append(event)
                continue
            if kind == 'answer':
                flush_text(event)
                continue
            flush_text()
            if kind in {'images_attached', 'media_attached'}:
                key = 'images' if kind == 'images_attached' else 'media'
                refs = []
                for raw in payload.get(key, []):
                    if not isinstance(raw, dict) or (ref := _attachment(raw, kind)) is None:
                        continue
                    refs.append(ref)
                    identity = (ref.get('asset_id'), ref.get('generation_id'), ref.get('artifact_id'))
                    entry = attachments.setdefault(identity, {**ref, 'turn_ids': []})
                    if turn['id'] not in entry['turn_ids']:
                        entry['turn_ids'].append(turn['id'])
                projected = {key: refs}
            else:
                projected = _event_projection(kind, payload)
            events.append({**event, 'payload': projected})
        flush_text()
        turns.append({**turn, 'prompt': _export_text(turn['prompt']), 'events': events})
    return {'schema_version': 'asteria.assistant.conversation.v1', 'conversation': conversation,
            'turns': turns, 'attachments': list(attachments.values())}


def conversation_markdown(document: dict) -> str:
    title = ' '.join(document['conversation']['title'].split()) or 'Asteria conversation'
    out = [f'# {title}']
    for turn in document['turns']:
        out.extend(['## User', turn['prompt'], f'## Assistant ({turn["model"]})'])
        for event in turn['events']:
            kind, payload = event['kind'], event['payload']
            if kind in {'text_delta', 'answer'}:
                out.append(payload['text'])
            elif kind == 'tool_result' and payload.get('name'):
                refs = [str(payload[key]) for key in ('id', 'generation_id', 'artifact_id', 'export_id') if key in payload]
                refs.extend(f'#{identity}' for identity in payload.get('generation_ids', []))
                refs.extend(payload.get('artifact_refs', []))
                out.append('- Asteria: `' + payload['name'] + '`' + (': ' + ', '.join(refs) if refs else ''))
            elif kind == 'draft_saved' and payload.get('draft_id'):
                out.append(f'- Saved draft: {payload["draft_id"]} (revision {payload["revision"]})')
        references = [ref for ref in document['attachments'] if turn['id'] in ref['turn_ids']]
        if references:
            out.append('### Attachments')
            out.extend(f'- {ref.get("filename", ref["kind"])} — '
                       + (ref['asset_id'] if 'asset_id' in ref else f'#{ref["generation_id"]} / {ref["artifact_id"]}')
                       for ref in references)
    return '\n\n'.join(out) + '\n'
