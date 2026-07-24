from __future__ import annotations

import base64
import hashlib
import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _redact_data_urls(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if "data:image/" in lowered and ";base64," in lowered:
            return "[redacted data URL]"
        return value
    if isinstance(value, list):
        return [_redact_data_urls(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_data_urls(item) for key, item in value.items()}
    return value


class EagleImportOptions(BaseModel):
    name: str | None = None
    website: str = ""
    annotation: str = ""
    tags: list[str] = Field(default_factory=list)
    rating: int = Field(default=0, ge=0, le=5)
    folder_id: str = ""


class EagleError(RuntimeError):
    def __init__(self, message: str, *, folder_applied: bool = False) -> None:
        super().__init__(message)
        self.folder_applied = folder_applied


def eagle_source_artifact(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    image_size = None
    try:
        with Image.open(path) as image:
            image_size = {"width": int(image.width), "height": int(image.height)}
    except (OSError, ValueError, RuntimeError, UnidentifiedImageError):
        image_size = None
    try:
        resolved_path = str(path.resolve())
    except (OSError, ValueError, RuntimeError):
        resolved_path = str(path)
    return {
        "path": resolved_path,
        "filename": path.name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "image_size": image_size,
    }


def register_eagle_image(
    path: Path,
    *,
    base_url: str,
    options: EagleImportOptions,
    skip_duplicates: bool = True,
    adapter: "EagleAdapter | None" = None,
) -> dict[str, Any]:
    source_artifact = eagle_source_artifact(path)
    active_adapter = adapter or EagleAdapter(base_url)
    transport = active_adapter.import_transport
    duplicate_check_error = ""
    try:
        duplicate = None
        if skip_duplicates:
            try:
                duplicate = active_adapter.find_duplicate_by_name(
                    name=options.name or path.stem,
                    ext=path.suffix,
                )
            except EagleError as exc:
                duplicate_check_error = str(exc)
        if duplicate:
            duplicate_id = str(duplicate.get("id") or "")
            update_result = None
            if duplicate_id:
                update_result = active_adapter.update_item(
                    duplicate_id,
                    rating=options.rating,
                    tags=options.tags,
                    annotation=options.annotation,
                )
            response: dict[str, Any] = {"status": "duplicate", "data": duplicate}
            if update_result is not None:
                response["update"] = update_result
            event = {
                "adapter": "eagle",
                "status": "duplicate_skipped",
                "filename": path.name,
                "source_artifact": source_artifact,
                "duplicate": duplicate,
                "tags": options.tags,
                "rating": options.rating,
                "folder_id": options.folder_id,
                "transport": transport,
                "folder_applied": False,
                "update": update_result,
            }
            return {
                "status": "duplicate_skipped",
                "filename": path.name,
                "eagle_response": response,
                "source_artifact": source_artifact,
                "duplicate": duplicate,
                "folder_id": options.folder_id,
                "transport": transport,
                "folder_applied": False,
                "event": event,
            }
        response = active_adapter.add_image(path, options)
        if duplicate_check_error:
            response = {**response, "duplicate_check_error": duplicate_check_error}
    except EagleError as exc:
        event = {
            "adapter": "eagle",
            "status": "failed",
            "filename": path.name,
            "source_artifact": source_artifact,
            "folder_id": options.folder_id,
            "transport": transport,
            "folder_applied": exc.folder_applied,
            "error": str(exc),
        }
        if duplicate_check_error:
            event["duplicate_check_error"] = duplicate_check_error
        return {
            "status": "failed",
            "filename": path.name,
            "eagle_response": {},
            "source_artifact": source_artifact,
            "duplicate": None,
            "folder_id": options.folder_id,
            "transport": transport,
            "folder_applied": exc.folder_applied,
            "event": event,
        }
    event = {
        "adapter": "eagle",
        "status": "success",
        "filename": path.name,
        "source_artifact": source_artifact,
        "tags": options.tags,
        "rating": options.rating,
        "folder_id": options.folder_id,
        "transport": transport,
        "folder_applied": True,
        "response": response,
    }
    if duplicate_check_error:
        event["duplicate_check_error"] = duplicate_check_error
    return {
        "status": "success",
        "filename": path.name,
        "eagle_response": response,
        "source_artifact": source_artifact,
        "duplicate": None,
        "folder_id": options.folder_id,
        "transport": transport,
        "folder_applied": True,
        "event": event,
    }


class EagleAdapter:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def import_transport(self) -> str:
        host = (urlsplit(self.base_url).hostname or "").lower()
        if host == "localhost":
            return "path"
        try:
            return "path" if ipaddress.ip_address(host).is_loopback else "base64"
        except ValueError:
            return "base64"

    def status(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/application/info")
        except httpx.HTTPError as exc:
            raise EagleError(str(_redact_data_urls(str(exc)))) from None
        if response.status_code >= 400:
            raise EagleError(f"Eagle API returned HTTP {response.status_code}")
        return self._json_object(response)

    def add_from_path(self, path: Path, options: EagleImportOptions) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": str(path),
            "name": options.name or path.stem,
            "website": options.website,
            "annotation": options.annotation,
            "tags": options.tags,
        }
        if options.folder_id:
            payload["folderId"] = options.folder_id
        result = self._post_import("/api/item/addFromPath", payload)
        item_id = self._item_id(result)
        if item_id and options.rating > 0:
            try:
                result["update"] = self.update_item(item_id, rating=options.rating)
            except EagleError as exc:
                exc.folder_applied = True
                raise
        return result

    def add_from_base64(self, path: Path, options: EagleImportOptions) -> dict[str, Any]:
        mime_type = _IMAGE_MIME_TYPES.get(path.suffix.lower())
        if not mime_type:
            raise EagleError("Unsupported Eagle image type")
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except (OSError, ValueError) as exc:
            raise EagleError("Could not read the Eagle import image") from exc
        payload: dict[str, Any] = {
            "url": f"data:{mime_type};base64,{encoded}",
            "name": options.name or path.stem,
            "website": options.website,
            "annotation": options.annotation,
            "tags": options.tags,
            "star": options.rating,
        }
        if options.folder_id:
            payload["folderId"] = options.folder_id
        return self._post_import("/api/item/addFromURL", payload)

    def add_image(self, path: Path, options: EagleImportOptions) -> dict[str, Any]:
        if self.import_transport == "path":
            return self.add_from_path(path, options)
        return self.add_from_base64(path, options)

    def update_item(
        self,
        item_id: str,
        *,
        rating: int | None = None,
        tags: list[str] | None = None,
        annotation: str | None = None,
        website: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": item_id}
        if tags is not None:
            payload["tags"] = tags
        if annotation is not None:
            payload["annotation"] = annotation
        if website is not None:
            payload["url"] = website
        if rating is not None:
            payload["star"] = rating
        if len(payload) == 1:
            return {"status": "skipped", "data": {"id": item_id}}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/api/item/update", json=payload)
        except httpx.HTTPError as exc:
            raise EagleError(str(_redact_data_urls(str(exc)))) from None
        if response.status_code >= 400:
            raise EagleError(f"Eagle API returned HTTP {response.status_code}")
        result = self._json_object(response)
        if result.get("status") not in {"success", "ok"}:
            raise EagleError(str(result.get("message") or result))
        return result

    def list_items(self, *, keyword: str = "", ext: str = "", limit: int = 50) -> dict[str, Any]:
        query = urlencode(
            {
                "keyword": keyword,
                "ext": ext,
                "limit": str(limit),
                "orderBy": "-CREATEDATE",
            }
        )
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/item/list?{query}")
        except httpx.HTTPError as exc:
            raise EagleError(str(_redact_data_urls(str(exc)))) from None
        if response.status_code >= 400:
            raise EagleError(f"Eagle API returned HTTP {response.status_code}")
        result = self._json_object(response)
        if result.get("status") not in {"success", "ok"}:
            raise EagleError(str(result.get("message") or result))
        return result

    def list_folders(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/folder/list")
        except httpx.HTTPError as exc:
            raise EagleError(str(_redact_data_urls(str(exc)))) from None
        if response.status_code >= 400:
            raise EagleError(f"Eagle API returned HTTP {response.status_code}")
        result = self._json_object(response)
        if result.get("status") not in {"success", "ok"}:
            raise EagleError(str(result.get("message") or result))
        return result

    def find_duplicate_by_name(self, *, name: str, ext: str) -> dict[str, Any] | None:
        result = self.list_items(keyword=name, ext=ext.lstrip(".").lower())
        items = result.get("data")
        if isinstance(items, dict):
            items = items.get("items") or items.get("list") or []
        if not isinstance(items, list):
            return None
        wanted_name = name.lower()
        wanted_ext = ext.lstrip(".").lower()
        for item in items:
            if not isinstance(item, dict):
                continue
            item_name = str(item.get("name") or "").lower()
            item_ext = str(item.get("ext") or "").lstrip(".").lower()
            if item_name == wanted_name and (not wanted_ext or item_ext == wanted_ext):
                return item
        return None

    def _item_id(self, result: dict[str, Any]) -> str:
        data = result.get("data")
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return str(data.get("id") or "")
        return str(result.get("id") or "")

    def _post_import(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}{endpoint}", json=payload)
        except httpx.HTTPError as exc:
            raise EagleError(str(_redact_data_urls(str(exc)))) from None
        if response.status_code >= 400:
            raise EagleError(f"Eagle API returned HTTP {response.status_code}")
        result = self._json_object(response)
        if result.get("status") not in {"success", "ok"}:
            raise EagleError(str(result.get("message") or result))
        return result

    def _json_object(self, response: httpx.Response) -> dict[str, Any]:
        try:
            result = response.json()
        except ValueError as exc:
            raise EagleError("Eagle API returned invalid JSON") from exc
        result = _redact_data_urls(result)
        if not isinstance(result, dict):
            raise EagleError("Eagle API returned an unexpected response")
        return result
