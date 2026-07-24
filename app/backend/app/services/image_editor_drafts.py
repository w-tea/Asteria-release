from __future__ import annotations

import json
from pathlib import Path
import re
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.services.atomic_json import write_json_atomic


class DraftConflictError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DraftCropRect(BaseModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class DraftPoseJoint(BaseModel):
    id: str
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ImageEditorDraftWrite(BaseModel):
    draft_id: str = ""
    revision: int = Field(default=0, ge=0)
    source_path: str = ""
    mask_path: str = ""
    crop_rect: DraftCropRect | None = None
    pose_joints: list[DraftPoseJoint] = Field(default_factory=list, max_length=128)
    active_tool: Literal["mask", "erase", "crop", "pan"] = "mask"
    active_inspector: str = "source"
    brush_size: int = Field(default=56, ge=1, le=512)
    brush_opacity: float = Field(default=0.78, ge=0, le=1)
    mask_visible: bool = True

    @field_validator("draft_id")
    @classmethod
    def _validate_draft_id(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{32}", value):
            raise ValueError("invalid draft id")
        return value


class ImageEditorDraft(ImageEditorDraftWrite):
    draft_id: str
    revision: int = Field(ge=1)
    updated_at: float


class ImageEditorDraftRepository:
    _lock = RLock()

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "editor_drafts"
        self.current_path = self.root / "current.json"

    def load(self) -> ImageEditorDraft | None:
        with self._lock:
            return self._load_unlocked()

    def replace(self, write: ImageEditorDraftWrite, *, updated_at: float) -> ImageEditorDraft:
        with self._lock:
            previous = self._load_unlocked()
            draft = ImageEditorDraft(
                **write.model_dump(exclude={"draft_id", "revision", "mask_path"}),
                draft_id=uuid4().hex,
                revision=1,
                mask_path="",
                updated_at=updated_at,
            )
            write_json_atomic(self.current_path, draft.model_dump())
            self._remove_mask(previous)
            return draft

    def update(self, write: ImageEditorDraftWrite, *, updated_at: float) -> ImageEditorDraft:
        with self._lock:
            current = self._require_current(write.draft_id, write.revision)
            draft = ImageEditorDraft(
                **write.model_dump(exclude={"revision", "mask_path"}),
                revision=current.revision + 1,
                mask_path=current.mask_path,
                updated_at=updated_at,
            )
            write_json_atomic(self.current_path, draft.model_dump())
            return draft

    def save_mask(self, draft_id: str, revision: int, data: bytes, *, updated_at: float) -> ImageEditorDraft:
        with self._lock:
            current = self._require_current(draft_id, revision)
            path = self.root / f"{current.draft_id}_{current.revision + 1}_mask.png"
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                temporary.write_bytes(data)
                temporary.replace(path)
                draft = current.model_copy(update={
                    "revision": current.revision + 1,
                    "mask_path": str(path.resolve()),
                    "updated_at": updated_at,
                })
                write_json_atomic(self.current_path, draft.model_dump())
                if current.mask_path and current.mask_path != draft.mask_path:
                    try:
                        Path(current.mask_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                return draft
            finally:
                temporary.unlink(missing_ok=True)

    def discard(self, draft_id: str, revision: int) -> None:
        with self._lock:
            current = self._require_current(draft_id, revision)
            self.current_path.unlink(missing_ok=True)
            self._remove_mask(current)

    def _load_unlocked(self) -> ImageEditorDraft | None:
        if not self.current_path.exists():
            return None
        try:
            return ImageEditorDraft.model_validate(json.loads(self.current_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise DraftConflictError("image_editor_draft_corrupt") from exc

    def _require_current(self, draft_id: str, revision: int) -> ImageEditorDraft:
        current = self._load_unlocked()
        if current is None:
            raise DraftConflictError("image_editor_draft_missing")
        if current.draft_id != draft_id:
            raise DraftConflictError("image_editor_draft_superseded")
        if current.revision != revision:
            raise DraftConflictError("image_editor_draft_stale")
        return current

    @staticmethod
    def _remove_mask(draft: ImageEditorDraft | None) -> None:
        if not draft or not draft.mask_path:
            return
        try:
            Path(draft.mask_path).unlink(missing_ok=True)
        except OSError:
            pass
