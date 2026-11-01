from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import ntpath
import os
from pathlib import Path
import re
import stat
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.services import api_errors
from app.services.atomic_json import write_json_atomic
from app.services.image_validation import InvalidImageContentError, validate_image_bytes


_OPAQUE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_MASK_STAGE_PATTERN = re.compile(r"([0-9a-f]{32})_staged_mask\.png")
_CLAIMED_MASK_PATTERN = re.compile(r"([0-9a-f]{32})_claimed_mask\.png")


@dataclass(frozen=True)
class _DraftRootIdentity:
    device: int
    inode: int


class _DirectoryPin:
    def close(self) -> None:
        pass


class _WindowsDirectoryPin(_DirectoryPin):
    def __init__(self, handle: int, close_handle) -> None:
        self._handle = handle
        self._close_handle = close_handle

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        if not self._close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _acquire_directory_pin(path: Path) -> _DirectoryPin:
    if os.name != "nt":
        return _DirectoryPin()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        0x00000001 | 0x00000080,  # FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return _WindowsDirectoryPin(handle, close_handle)


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


class DraftPosePerson(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    joints: list[DraftPoseJoint] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def _unique_joints(self):
        if len({joint.id for joint in self.joints}) != len(self.joints):
            raise ValueError("duplicate pose joint id")
        return self


class ImageEditorDraftWrite(BaseModel):
    draft_id: str = ""
    revision: int = Field(default=0, ge=0)
    source_path: str = ""
    mask_path: str = ""
    crop_rect: DraftCropRect | None = None
    pose_joints: list[DraftPoseJoint] = Field(default_factory=list, max_length=128)
    pose_people: list[DraftPosePerson] | None = None
    selected_pose_id: str = Field(default="", max_length=128)
    active_tool: Literal["mask", "erase", "crop", "pan"] = "mask"
    active_inspector: str = "source"
    brush_size: int = Field(default=56, ge=1, le=512)
    brush_opacity: float = Field(default=0.78, ge=0, le=1)
    mask_visible: bool = True

    @model_validator(mode="after")
    def _unique_people(self):
        if self.pose_people is not None:
            ids = {person.id for person in self.pose_people}
            if len(ids) != len(self.pose_people):
                raise ValueError("duplicate pose person id")
            if self.selected_pose_id and self.selected_pose_id not in ids:
                raise ValueError("selected pose person is missing")
        return self

    @field_validator("draft_id")
    @classmethod
    def _validate_draft_id(cls, value: str) -> str:
        if value and not _OPAQUE_ID_PATTERN.fullmatch(value):
            raise ValueError("invalid draft id")
        return value


class ImageEditorDraft(ImageEditorDraftWrite):
    draft_id: str
    revision: int = Field(ge=1)
    updated_at: float


class ImageEditorDraftCommitRequest(BaseModel):
    operation: Literal["create", "update", "replace"]
    mask_intent: Literal["preserve", "replace", "clear"]
    expected_draft_id: str = ""
    expected_revision: int = Field(default=0, ge=0)
    stage_token: str = ""
    draft: ImageEditorDraftWrite

    @field_validator("expected_draft_id")
    @classmethod
    def _validate_expected_draft_id(cls, value: str) -> str:
        if value and not _OPAQUE_ID_PATTERN.fullmatch(value):
            raise ValueError("invalid expected draft id")
        return value

    @field_validator("stage_token")
    @classmethod
    def _validate_stage_token(cls, value: str) -> str:
        if value and not _OPAQUE_ID_PATTERN.fullmatch(value):
            raise ValueError("invalid stage token")
        return value

    @model_validator(mode="after")
    def _validate_mask_stage(self):
        if self.mask_intent == "replace" and not self.stage_token:
            raise ValueError("stage token is required when replacing a mask")
        if self.mask_intent != "replace" and self.stage_token:
            raise ValueError("stage token is only accepted when replacing a mask")
        return self


class ImageEditorDraftRepository:
    _lock = RLock()

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "editor_drafts"
        self.current_path = self.root / "current.json"

    def load(self) -> ImageEditorDraft | None:
        with self._lock:
            try:
                root_identity, root_pin = self._acquire_root_pin(create=False)
                if root_identity is None:
                    return None
                assert root_pin is not None
                try:
                    return self._load_unlocked(root_identity)
                finally:
                    root_pin.close()
            except OSError as exc:
                raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_CORRUPT) from exc

    def stage_mask(self, data: bytes) -> str:
        with self._lock:
            root_identity, root_pin = self._acquire_root_pin(create=True)
            assert root_identity is not None and root_pin is not None
            try:
                return self._stage_mask_unlocked(data, root_identity)
            finally:
                root_pin.close()

    def _stage_mask_unlocked(
        self,
        data: bytes,
        root_identity: _DraftRootIdentity,
    ) -> str:
        token = uuid4().hex
        path = self.root / f"{token}_staged_mask.png"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            self._assert_root_identity(root_identity)
            temporary.write_bytes(data)
            self._assert_root_identity(root_identity)
            with temporary.open("r+b") as temporary_file:
                os.fsync(temporary_file.fileno())
            self._assert_root_identity(root_identity)
            temporary.replace(path)
            self._assert_root_identity(root_identity)
            return token
        except OSError:
            self._unlink_if_root_unchanged(path, root_identity)
            raise
        finally:
            self._unlink_if_root_unchanged(temporary, root_identity)

    def cancel_stage(self, token: str) -> None:
        if not _OPAQUE_ID_PATTERN.fullmatch(token):
            return
        with self._lock:
            root_identity, root_pin = self._acquire_root_pin(create=False)
            if root_identity is None:
                return
            assert root_pin is not None
            try:
                path = self._owned_stage_path(token, root_identity)
                if path is None:
                    return
                self._assert_root_identity(root_identity)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                self._assert_root_identity(root_identity)
            finally:
                root_pin.close()

    def commit(
        self,
        request: ImageEditorDraftCommitRequest,
        *,
        updated_at: float,
    ) -> ImageEditorDraft:
        with self._lock:
            root_identity, root_pin = self._acquire_root_pin(create=True)
            assert root_identity is not None and root_pin is not None
            try:
                return self._commit_unlocked(request, updated_at, root_identity)
            finally:
                root_pin.close()

    def _commit_unlocked(
        self,
        request: ImageEditorDraftCommitRequest,
        updated_at: float,
        root_identity: _DraftRootIdentity,
    ) -> ImageEditorDraft:
        previous = self._validate_commit_preconditions(request, root_identity)
        if request.operation == "update":
            assert previous is not None
            draft_id = previous.draft_id
            revision = previous.revision + 1
        else:
            draft_id = uuid4().hex
            revision = 1

        source_mask: Path | None = None
        claimed_stage: Path | None = None
        if request.mask_intent == "replace":
            claimed_stage = self._claim_stage(
                request.stage_token,
                root_identity,
            )
            source_mask = claimed_stage
        elif request.mask_intent == "preserve" and previous and previous.mask_path:
            source_mask = self._owned_mask_path(previous, root_identity)
            if source_mask is None:
                raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_SAVE_FAILED)

        target_mask: Path | None = None
        if source_mask is not None:
            target_mask = self.root / f"{draft_id}_{revision}_mask.png"

        draft = ImageEditorDraft(
            **request.draft.model_dump(exclude={"draft_id", "revision", "mask_path"}),
            draft_id=draft_id,
            revision=revision,
            mask_path=str(Path(os.path.abspath(target_mask))) if target_mask else "",
            updated_at=updated_at,
        )

        try:
            mask_bytes: bytes | None = None
            if source_mask is not None:
                mask_bytes = self._read_owned_mask_bytes(
                    source_mask,
                    root_identity,
                )
                if claimed_stage is not None:
                    validate_image_bytes(mask_bytes, ".png")
            if mask_bytes is not None and target_mask is not None:
                self._materialize_mask(
                    mask_bytes,
                    target_mask,
                    root_identity,
                )
        except Exception as exc:
            self._remove_exact_mask(draft, root_identity)
            if claimed_stage is not None:
                self._restore_claimed_stage(
                    claimed_stage,
                    request.stage_token,
                    root_identity,
                )
            if isinstance(exc, InvalidImageContentError):
                raise OSError("image editor draft staged mask is invalid") from exc
            raise

        authoritative = draft
        try:
            self._assert_root_identity(root_identity)
            write_json_atomic(self.current_path, draft.model_dump())
            self._assert_root_identity(root_identity)
        except Exception as exc:
            reloaded = self._reload_exact_commit(draft, root_identity)
            if reloaded is not None:
                authoritative = reloaded
            else:
                self._remove_exact_mask(draft, root_identity)
                rollback_error: Exception | None = None
                try:
                    self._restore_previous(previous, root_identity)
                except Exception as restore_exc:
                    rollback_error = restore_exc
                try:
                    if claimed_stage is not None:
                        self._restore_claimed_stage(
                            claimed_stage,
                            request.stage_token,
                            root_identity,
                        )
                except Exception as restore_exc:
                    rollback_error = rollback_error or restore_exc
                if rollback_error is not None:
                    raise OSError(
                        "image editor draft rollback failed"
                    ) from rollback_error
                if isinstance(exc, OSError):
                    raise
                raise OSError("image editor draft publication failed") from exc

        if claimed_stage is not None:
            self._remove_claimed_stage(
                claimed_stage,
                request.stage_token,
                root_identity,
            )
        if previous and previous.mask_path != authoritative.mask_path:
            self._remove_exact_mask(previous, root_identity)
        return authoritative

    def discard(self, draft_id: str, revision: int) -> None:
        with self._lock:
            root_identity, root_pin = self._acquire_root_pin(create=False)
            if root_identity is None:
                raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_MISSING)
            assert root_pin is not None
            try:
                current = self._require_current(
                    draft_id,
                    revision,
                    root_identity,
                )
                self._assert_root_identity(root_identity)
                self.current_path.unlink(missing_ok=True)
                self._assert_root_identity(root_identity)
                self._remove_exact_mask(current, root_identity)
            finally:
                root_pin.close()

    def _validate_commit_preconditions(
        self,
        request: ImageEditorDraftCommitRequest,
        root_identity: _DraftRootIdentity,
    ) -> ImageEditorDraft | None:
        current = self._load_unlocked(root_identity)
        if request.operation == "create":
            if request.expected_draft_id or request.expected_revision:
                raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_STALE)
            if current is not None:
                raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_SUPERSEDED)
            return None

        current = self._require_current(
            request.expected_draft_id,
            request.expected_revision,
            root_identity,
        )
        same_source = _windows_source_identity(current.source_path) == _windows_source_identity(
            request.draft.source_path
        )
        if request.operation == "update" and not same_source:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_SUPERSEDED)
        if request.operation == "replace" and same_source:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_STALE)
        return current

    def _load_unlocked(
        self,
        root_identity: _DraftRootIdentity,
    ) -> ImageEditorDraft | None:
        self._assert_root_identity(root_identity)
        try:
            details = os.lstat(self.current_path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(details.st_mode) or _is_reparse(self.current_path, details):
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_CORRUPT)
        try:
            self._assert_root_identity(root_identity)
            draft = ImageEditorDraft.model_validate(
                json.loads(self.current_path.read_text(encoding="utf-8"))
            )
            self._assert_root_identity(root_identity)
            return draft
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_CORRUPT) from exc

    def _reload_exact_commit(
        self,
        draft: ImageEditorDraft,
        root_identity: _DraftRootIdentity,
    ) -> ImageEditorDraft | None:
        try:
            current = self._load_unlocked(root_identity)
        except (DraftConflictError, OSError):
            return None
        if current is None:
            return None
        if current.model_dump(mode="json") != draft.model_dump(mode="json"):
            return None
        return current

    def _restore_previous(
        self,
        previous: ImageEditorDraft | None,
        root_identity: _DraftRootIdentity,
    ) -> None:
        if previous is None:
            self._assert_root_identity(root_identity)
            self.current_path.unlink(missing_ok=True)
            self._assert_root_identity(root_identity)
            return
        try:
            self._assert_root_identity(root_identity)
            write_json_atomic(self.current_path, previous.model_dump())
            self._assert_root_identity(root_identity)
        except Exception:
            reloaded = self._reload_exact_commit(previous, root_identity)
            if reloaded is None:
                raise

    def _require_current(
        self,
        draft_id: str,
        revision: int,
        root_identity: _DraftRootIdentity,
    ) -> ImageEditorDraft:
        current = self._load_unlocked(root_identity)
        if current is None:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_MISSING)
        if current.draft_id != draft_id:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_SUPERSEDED)
        if current.revision != revision:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_STALE)
        return current

    def _acquire_root_pin(
        self,
        *,
        create: bool,
    ) -> tuple[_DraftRootIdentity | None, _DirectoryPin | None]:
        before = self._capture_root_identity(create=create)
        if before is None:
            return None, None
        root_pin = _acquire_directory_pin(self.root)
        try:
            after = self._capture_root_identity(create=False)
            if after is None or after != before:
                raise OSError(
                    "image editor draft root identity changed during pin acquisition"
                )
            return after, root_pin
        except Exception:
            root_pin.close()
            raise

    def _capture_root_identity(
        self,
        *,
        create: bool,
    ) -> _DraftRootIdentity | None:
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        try:
            details = os.lstat(self.root)
        except FileNotFoundError:
            if create:
                raise
            return None
        if not stat.S_ISDIR(details.st_mode) or _is_reparse(self.root, details):
            raise OSError("image editor draft root is not an owned directory")
        return _DraftRootIdentity(
            device=int(details.st_dev),
            inode=int(details.st_ino),
        )

    def _assert_root_identity(self, expected: _DraftRootIdentity) -> None:
        current = self._capture_root_identity(create=False)
        if current is None or current != expected:
            raise OSError("image editor draft root identity changed")

    def _unlink_if_root_unchanged(
        self,
        path: Path,
        root_identity: _DraftRootIdentity,
    ) -> None:
        try:
            self._assert_root_identity(root_identity)
            path.unlink(missing_ok=True)
            self._assert_root_identity(root_identity)
        except OSError:
            pass

    def _owned_stage_path(
        self,
        token: str,
        root_identity: _DraftRootIdentity,
    ) -> Path | None:
        path = self.root / f"{token}_staged_mask.png"
        if not _MASK_STAGE_PATTERN.fullmatch(path.name):
            return None
        return path if self._is_regular_owned_child(path, root_identity) else None

    def _owned_claimed_stage_path(
        self,
        token: str,
        root_identity: _DraftRootIdentity,
    ) -> Path | None:
        path = self.root / f"{token}_claimed_mask.png"
        if not _CLAIMED_MASK_PATTERN.fullmatch(path.name):
            return None
        return path if self._is_regular_owned_child(path, root_identity) else None

    def _claim_stage(
        self,
        token: str,
        root_identity: _DraftRootIdentity,
    ) -> Path:
        stage = self._owned_stage_path(token, root_identity)
        if stage is None:
            raise DraftConflictError(api_errors.IMAGE_EDITOR_DRAFT_STAGE_UNAVAILABLE)
        claimed = self.root / f"{token}_claimed_mask.png"
        self._assert_root_identity(root_identity)
        try:
            stage.replace(claimed)
        except FileNotFoundError as exc:
            raise DraftConflictError(
                api_errors.IMAGE_EDITOR_DRAFT_STAGE_UNAVAILABLE
            ) from exc
        self._assert_root_identity(root_identity)
        owned_claimed = self._owned_claimed_stage_path(token, root_identity)
        if owned_claimed is None:
            try:
                self._assert_root_identity(root_identity)
                claimed.replace(stage)
            except OSError:
                pass
            raise OSError("image editor draft stage claim is not owned")
        return owned_claimed

    def _owned_mask_path(
        self,
        draft: ImageEditorDraft,
        root_identity: _DraftRootIdentity,
    ) -> Path | None:
        if not draft.mask_path:
            return None
        expected = self.root / f"{draft.draft_id}_{draft.revision}_mask.png"
        if _normalized_absolute(Path(draft.mask_path)) != _normalized_absolute(expected):
            return None
        return expected if self._is_regular_owned_child(expected, root_identity) else None

    def _is_regular_owned_child(
        self,
        path: Path,
        root_identity: _DraftRootIdentity,
    ) -> bool:
        if _normalized_absolute(path.parent) != _normalized_absolute(self.root):
            return False
        self._assert_root_identity(root_identity)
        try:
            details = os.lstat(path)
        except OSError:
            return False
        owned = stat.S_ISREG(details.st_mode) and not _is_reparse(path, details)
        self._assert_root_identity(root_identity)
        return owned

    def _read_owned_mask_bytes(
        self,
        source: Path,
        root_identity: _DraftRootIdentity,
    ) -> bytes:
        if not self._is_regular_owned_child(source, root_identity):
            raise OSError("image editor draft mask source is not owned")
        self._assert_root_identity(root_identity)
        data = source.read_bytes()
        self._assert_root_identity(root_identity)
        if not self._is_regular_owned_child(source, root_identity):
            raise OSError("image editor draft mask source identity changed")
        return data

    def _materialize_mask(
        self,
        data: bytes,
        target: Path,
        root_identity: _DraftRootIdentity,
    ) -> None:
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            self._assert_root_identity(root_identity)
            temporary.write_bytes(data)
            self._assert_root_identity(root_identity)
            with temporary.open("r+b") as temporary_file:
                os.fsync(temporary_file.fileno())
            self._assert_root_identity(root_identity)
            temporary.replace(target)
            self._assert_root_identity(root_identity)
        finally:
            self._unlink_if_root_unchanged(temporary, root_identity)

    def _remove_exact_mask(
        self,
        draft: ImageEditorDraft,
        root_identity: _DraftRootIdentity,
    ) -> None:
        path = self._owned_mask_path(draft, root_identity)
        if path is None:
            return
        self._assert_root_identity(root_identity)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self._assert_root_identity(root_identity)

    def _restore_claimed_stage(
        self,
        path: Path,
        token: str,
        root_identity: _DraftRootIdentity,
    ) -> None:
        expected = self.root / f"{token}_claimed_mask.png"
        if _normalized_absolute(path) != _normalized_absolute(expected):
            raise OSError("image editor draft claimed stage path changed")
        claimed = self._owned_claimed_stage_path(token, root_identity)
        if claimed is None:
            raise OSError("image editor draft claimed stage is unavailable")
        staged = self.root / f"{token}_staged_mask.png"
        if self._owned_stage_path(token, root_identity) is not None:
            raise OSError("image editor draft stage already exists")
        self._assert_root_identity(root_identity)
        claimed.replace(staged)
        self._assert_root_identity(root_identity)

    def _remove_claimed_stage(
        self,
        path: Path,
        token: str,
        root_identity: _DraftRootIdentity,
    ) -> None:
        expected = self.root / f"{token}_claimed_mask.png"
        if _normalized_absolute(path) != _normalized_absolute(expected):
            return
        claimed = self._owned_claimed_stage_path(token, root_identity)
        if claimed is None:
            return
        self._assert_root_identity(root_identity)
        try:
            claimed.unlink(missing_ok=True)
        except OSError:
            pass
        self._assert_root_identity(root_identity)


def _windows_source_identity(source_path: str) -> str:
    return ntpath.normcase(ntpath.normpath(source_path.replace("/", "\\")))


def _normalized_absolute(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_reparse(path: Path, details: os.stat_result) -> bool:
    if stat.S_ISLNK(details.st_mode):
        return True
    attributes = getattr(details, "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        return True
    junction_check = getattr(os.path, "isjunction", None)
    if junction_check is None:
        return False
    try:
        return bool(junction_check(path))
    except OSError:
        return True
