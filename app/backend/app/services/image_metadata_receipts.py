from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from app.services.atomic_json import write_json_atomic
from app.services.image_assets import (
    ImageAssetError,
    SavedImageAsset,
    safe_image_filename,
    save_validated_image_bytes,
    write_image_bytes_atomic,
)
from app.services.image_metadata_text import read_image_metadata_values
from app.services.image_validation import (
    InvalidImageContentError,
    SUPPORTED_IMAGE_FORMATS,
    validate_image_bytes,
    validate_image_file,
)


RECEIPT_LIFETIME = timedelta(hours=1)
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ImageMetadataReceiptError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ImageMetadataReceipt:
    token: str
    filename: str
    staged_path: Path
    sha256: str
    size_bytes: int
    image_format: str
    width: int
    height: int
    created_at: datetime
    expires_at: datetime
    generation_id: int | None = None
    output_path: str | None = None


class ImageMetadataReceiptStore:
    def __init__(self, data_dir: Path, *, now: Callable[[], datetime] | None = None) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "staging" / "image_metadata"
        self.consumed_root = self.root / "consumed"
        self._now = now or (lambda: datetime.now(UTC))

    def stage_path(
        self,
        source: Path,
        *,
        generation_id: int | None = None,
        output_path: str | None = None,
    ) -> ImageMetadataReceipt:
        try:
            validate_image_file(source)
            data = source.read_bytes()
        except InvalidImageContentError as exc:
            raise ImageMetadataReceiptError("image_metadata_image_invalid") from exc
        except OSError as exc:
            raise ImageMetadataReceiptError("image_metadata_image_unavailable") from exc
        return self.stage_bytes(
            data,
            filename=source.name,
            generation_id=generation_id,
            output_path=output_path,
        )

    def stage_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        generation_id: int | None = None,
        output_path: str | None = None,
    ) -> ImageMetadataReceipt:
        self.cleanup_expired()
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_FORMATS:
            raise ImageMetadataReceiptError("image_metadata_image_unsupported")
        try:
            validate_image_bytes(data, suffix)
        except InvalidImageContentError as exc:
            raise ImageMetadataReceiptError("image_metadata_image_invalid") from exc
        token = uuid4().hex
        safe_name = safe_image_filename(filename)
        stage_dir = self.root / token
        staged_path = stage_dir / safe_name
        try:
            write_image_bytes_atomic(staged_path, data)
            _info, _values, image_format, width, height = read_image_metadata_values(staged_path)
            created_at = self._aware_now()
            receipt = ImageMetadataReceipt(
                token=token,
                filename=safe_name,
                staged_path=staged_path,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                image_format=image_format,
                width=width,
                height=height,
                created_at=created_at,
                expires_at=created_at + RECEIPT_LIFETIME,
                generation_id=generation_id,
                output_path=output_path,
            )
            write_json_atomic(stage_dir / "receipt.json", self._receipt_payload(receipt))
            return receipt
        except (ImageAssetError, OSError, ValueError) as exc:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ImageMetadataReceiptError("image_metadata_receipt_save_failed") from exc

    def get(self, token: str) -> ImageMetadataReceipt:
        self._validate_token(token)
        if (self.consumed_root / f"{token}.json").is_file():
            raise ImageMetadataReceiptError("image_metadata_receipt_consumed")
        receipt_path = self.root / token / "receipt.json"
        try:
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt = self._receipt_from_payload(token, payload)
        except FileNotFoundError as exc:
            raise ImageMetadataReceiptError("image_metadata_receipt_not_found") from exc
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ImageMetadataReceiptError("image_metadata_receipt_invalid") from exc
        if receipt.expires_at <= self._aware_now():
            shutil.rmtree(self.root / token, ignore_errors=True)
            raise ImageMetadataReceiptError("image_metadata_receipt_expired")
        return receipt

    def consume_as_init_image(self, token: str) -> SavedImageAsset:
        receipt = self.get(token)
        try:
            data = receipt.staged_path.read_bytes()
        except OSError as exc:
            raise ImageMetadataReceiptError("image_metadata_receipt_mismatch") from exc
        if len(data) != receipt.size_bytes or hashlib.sha256(data).hexdigest() != receipt.sha256:
            raise ImageMetadataReceiptError("image_metadata_receipt_mismatch")
        try:
            asset = save_validated_image_bytes(
                data,
                filename=receipt.filename,
                output_dir=self.data_dir / "init_images",
            )
        except ImageAssetError as exc:
            raise ImageMetadataReceiptError("image_metadata_input_save_failed") from exc
        self.consumed_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.consumed_root / f"{token}.json",
            {"consumed_at": self._aware_now().isoformat(), "expires_at": receipt.expires_at.isoformat()},
        )
        shutil.rmtree(self.root / token, ignore_errors=True)
        return asset

    def cleanup_expired(self) -> None:
        now = self._aware_now()
        try:
            children = list(self.root.iterdir()) if self.root.is_dir() else []
        except OSError:
            return
        for child in children:
            if child == self.consumed_root or not child.is_dir() or not TOKEN_PATTERN.fullmatch(child.name):
                continue
            receipt_path = child / "receipt.json"
            try:
                payload = json.loads(receipt_path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(str(payload["expires_at"]))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if self._as_aware(expires_at) <= now:
                shutil.rmtree(child, ignore_errors=True)
        try:
            tombstones = list(self.consumed_root.glob("*.json")) if self.consumed_root.is_dir() else []
        except OSError:
            return
        for tombstone in tombstones:
            try:
                payload = json.loads(tombstone.read_text(encoding="utf-8"))
                expires_at = self._as_aware(datetime.fromisoformat(str(payload["expires_at"])))
                if expires_at <= now:
                    tombstone.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def _receipt_from_payload(self, token: str, payload: object) -> ImageMetadataReceipt:
        if not isinstance(payload, dict) or payload.get("token") != token:
            raise ValueError("receipt token mismatch")
        filename = safe_image_filename(str(payload["filename"]))
        staged_path = self.root / token / filename
        return ImageMetadataReceipt(
            token=token,
            filename=filename,
            staged_path=staged_path,
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            image_format=str(payload["image_format"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            created_at=self._as_aware(datetime.fromisoformat(str(payload["created_at"]))),
            expires_at=self._as_aware(datetime.fromisoformat(str(payload["expires_at"]))),
            generation_id=int(payload["generation_id"]) if payload.get("generation_id") is not None else None,
            output_path=str(payload["output_path"]) if payload.get("output_path") is not None else None,
        )

    @staticmethod
    def _receipt_payload(receipt: ImageMetadataReceipt) -> dict[str, object]:
        return {
            "token": receipt.token,
            "filename": receipt.filename,
            "sha256": receipt.sha256,
            "size_bytes": receipt.size_bytes,
            "image_format": receipt.image_format,
            "width": receipt.width,
            "height": receipt.height,
            "created_at": receipt.created_at.isoformat(),
            "expires_at": receipt.expires_at.isoformat(),
            "generation_id": receipt.generation_id,
            "output_path": receipt.output_path,
        }

    @staticmethod
    def _validate_token(token: str) -> None:
        if not TOKEN_PATTERN.fullmatch(token):
            raise ImageMetadataReceiptError("image_metadata_receipt_not_found")

    def _aware_now(self) -> datetime:
        return self._as_aware(self._now())

    @staticmethod
    def _as_aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
