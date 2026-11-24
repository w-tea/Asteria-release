"""Read saved editor images through the existing core visibility rules."""
from __future__ import annotations

import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from PIL import Image

from app.core.config import AppConfig
from app.schemas.action_contracts import AssistantDraftRecord, DraftImageContent, DraftImageDiagnostics, InlineImage
from app.schemas.media import image_media_type
from app.services.user_settings import UserSettingsRepository, resolve_output_root


_ASSET_DIRECTORIES = ('init_images', 'mask_images', 'control_images', 'editor_drafts')


def inspect_draft_inputs(config: AppConfig, draft: AssistantDraftRecord) -> tuple[
        list[DraftImageDiagnostics], list[Literal['init', 'mask']]]:
    settings = UserSettingsRepository(config.data_dir / 'user-settings.json').load().output
    output_root = resolve_output_root(config.data_dir, settings).absolute()
    asset_roots = {(config.data_dir / name).absolute() for name in _ASSET_DIRECTORIES}
    roles: tuple[Literal['init', 'mask'], ...] = ('init', 'mask') if draft.document['mode'] == 'inpaint' else ('init',)
    verified, unverified = [], []
    for role in roles:
        field = 'init_image_path' if role == 'init' else 'mask_image_path'
        path = Path(draft.document['input'][field])
        # A read-only diagnostic should not probe an arbitrary path in a draft.
        # Paths outside these configured roots are unverified, not incompatible.
        if not path.is_absolute() or '..' in path.parts or (
            path.parent not in asset_roots and not path.is_relative_to(output_root)
        ):
            unverified.append(role)
            continue
        image = read_draft_image(config, draft, role)
        verified.append(DraftImageDiagnostics(role=role, width=image.width, height=image.height,
                                              mime_type=image.mime_type))
    return verified, unverified


def read_draft_image(config: AppConfig, draft: AssistantDraftRecord,
                     role: Literal['init', 'mask']) -> DraftImageContent:
    from app.api.assets import asset_file
    from app.api.generation import output_file_by_path
    from app.services.action_images import ActionImageRepository

    stored_path = draft.document['input'].get('init_image_path' if role == 'init' else 'mask_image_path')
    if not stored_path:
        raise HTTPException(404, detail={'code': 'assistant_draft_image_not_found',
                                        'draft_id': draft.id, 'revision': draft.revision, 'role': role})
    try:
        path = Path(stored_path).resolve()
        directories = {name: (config.data_dir / name).resolve()
                       for name in _ASSET_DIRECTORIES}
    except (OSError, ValueError):
        raise HTTPException(404, detail={'code': 'asset_not_found'}) from None
    registered = ActionImageRepository(config.data_dir).get_by_saved_path(path)
    if registered is not None:
        return DraftImageContent(draft_id=draft.id, revision=draft.revision, role=role,
            **registered.model_dump(exclude={'asset_id', 'filename', 'role'}))
    directory = next((name for name, root in directories.items() if root == path.parent), None)
    # A draft path alone grants no file access. Reuse the visibility of ordinary
    # asset downloads or registered/current output images, including older roots.
    response = asset_file(directory, path.name) if directory else output_file_by_path(str(path))
    path = Path(response.path)
    try:
        data = path.read_bytes()
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except (OSError, ValueError):
        raise HTTPException(404, detail={'code': 'asset_not_found'}) from None
    fingerprint = sha256(data).hexdigest()
    lineage = draft.document['input'].get('image_edit_lineage')
    if role == 'init' and lineage and fingerprint != lineage['current']['sha256']:
        raise HTTPException(409, detail={'code': 'assistant_image_changed',
                                        'draft_id': draft.id, 'revision': draft.revision, 'role': role})
    mime_type = image_media_type(str(path))
    return DraftImageContent(draft_id=draft.id, revision=draft.revision, role=role,
        path=str(path), mime_type=mime_type, sha256=fingerprint, width=width, height=height,
        image=InlineImage(mime_type=mime_type, data_base64=base64.b64encode(data).decode('ascii')))
