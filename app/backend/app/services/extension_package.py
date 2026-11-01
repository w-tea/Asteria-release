from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import zipfile


_CHUNK_SIZE = 1024 * 1024
_WINDOWS_RESERVED = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE)


class ExtensionPackageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PackageInspection:
    package_sha256: str
    package_size: int
    manifest_name: str
    members: tuple[str, ...]
    root_prefix: str | None
    uncompressed_size: int


def inspect_extension_zip(
    path: Path,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    max_members: int = 4096,
    max_member_bytes: int = 128 * 1024 * 1024,
    max_uncompressed_bytes: int = 1024 * 1024 * 1024,
) -> PackageInspection:
    try:
        package_size = path.stat().st_size
        if package_size <= 0 or package_size > max_bytes:
            raise ExtensionPackageError("package size exceeds limit")
        package_hash = _hash_file(path)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > max_members:
                raise ExtensionPackageError("package member count exceeds limit")
            names: list[str] = []
            seen: set[str] = set()
            manifests: list[str] = []
            total_uncompressed = 0
            for info in entries:
                normalized = info.filename.replace("\\", "/").rstrip("/")
                parts = _validate_member_path(normalized)
                _validate_member_type(info)
                if info.file_size < 0 or info.file_size > max_member_bytes:
                    raise ExtensionPackageError("package member exceeds limit")
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise ExtensionPackageError("package uncompressed size exceeds limit")
                key = "/".join(part.casefold() for part in parts)
                if key in seen:
                    raise ExtensionPackageError("package path collision")
                seen.add(key)
                name = "/".join(parts)
                names.append(name)
                if parts[-1].casefold() == "asteria-extension.json":
                    manifests.append(name)
            if len(manifests) != 1:
                raise ExtensionPackageError("package must contain exactly one manifest")
            manifest_parts = tuple(manifests[0].split("/"))
            root_prefix = None if len(manifest_parts) == 1 else manifest_parts[0]
            if root_prefix is not None and any(name.split("/", 1)[0] != root_prefix for name in names):
                raise ExtensionPackageError("package wrapper root is inconsistent")
            if len(manifest_parts) > 2:
                raise ExtensionPackageError("package manifest must be at root or one wrapper root")
        return PackageInspection(
            package_sha256=package_hash,
            package_size=package_size,
            manifest_name="asteria-extension.json",
            members=tuple(sorted(names, key=str.casefold)),
            root_prefix=root_prefix,
            uncompressed_size=total_uncompressed,
        )
    except ExtensionPackageError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ExtensionPackageError("package archive is invalid") from exc


def extract_extension_zip(
    path: Path,
    destination: Path,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    max_members: int = 4096,
    max_member_bytes: int = 128 * 1024 * 1024,
    max_uncompressed_bytes: int = 1024 * 1024 * 1024,
) -> PackageInspection:
    inspection = inspect_extension_zip(
        path,
        max_bytes=max_bytes,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    if destination.exists() or destination.is_symlink():
        raise ExtensionPackageError("package staging destination already exists")
    try:
        destination.mkdir(parents=True, exist_ok=False)
        destination_root = destination.resolve(strict=True)
        total_written = 0
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                normalized = info.filename.replace("\\", "/").rstrip("/")
                parts = list(_validate_member_path(normalized))
                if inspection.root_prefix is not None:
                    if not parts or parts.pop(0) != inspection.root_prefix:
                        raise ExtensionPackageError("package wrapper root changed during extraction")
                if not parts:
                    continue
                target = destination.joinpath(*parts)
                target_parent = target if info.is_dir() else target.parent
                target_parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(destination_root):
                    raise ExtensionPackageError("package path escapes staging")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                written = 0
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(_CHUNK_SIZE):
                        written += len(chunk)
                        total_written += len(chunk)
                        if written > max_member_bytes or total_written > max_uncompressed_bytes:
                            raise ExtensionPackageError("package extraction exceeds declared limits")
                        output.write(chunk)
                if written != info.file_size:
                    raise ExtensionPackageError("package member size changed during extraction")
        _assert_regular_tree(destination)
        return inspection
    except ExtensionPackageError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise ExtensionPackageError("package extraction failed") from exc


def _validate_member_path(value: str) -> tuple[str, ...]:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = posix.parts
    if (
        not value
        or value.startswith(("/", "//"))
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ExtensionPackageError("package path is unsafe")
    for part in parts:
        stem = part.split(".", 1)[0]
        if (
            ":" in part
            or part.endswith((" ", "."))
            or _WINDOWS_RESERVED.fullmatch(stem) is not None
        ):
            raise ExtensionPackageError("package path is unsafe")
    return tuple(parts)


def _validate_member_type(info: zipfile.ZipInfo) -> None:
    if info.create_system != 3:
        return
    file_type = (info.external_attr >> 16) & 0o170000
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ExtensionPackageError("package contains a symlink or special file")


def _assert_regular_tree(root: Path) -> None:
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *filenames):
            candidate = current_path / name
            details = candidate.lstat()
            is_reparse = bool(getattr(details, "st_file_attributes", 0) & 0x400)
            is_junction = bool(getattr(os.path, "isjunction", lambda _path: False)(candidate))
            if stat.S_ISLNK(details.st_mode) or is_reparse or is_junction:
                raise ExtensionPackageError("package extraction created a filesystem reparse point")
            if name in filenames and not stat.S_ISREG(details.st_mode):
                raise ExtensionPackageError("package extraction created a special file")


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
