from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from urllib.parse import urlparse

import yaml


_OWNERSHIP_CLASSES = {
    "mandatory-runtime",
    "mandatory-feature",
    "optional-starter",
    "user-owned",
}
_REPAIR_POLICIES = {"replace-managed-component", "reinstall-package-set"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXACT_REQUIREMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+==[^<>=!*~\s]+$")
_REVISION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERIFIED_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MANDATORY_RUNTIME_IDS = {
    "asteria-prompt-syntax",
    "asteria-merge-runtime",
    "asteria-h3-latent-assembly",
    "asteria-anima-lora-remap",
    "asteria-qwen2d-vae",
    "asteria-spatial-prompt",
    "comfyui-anima-3-8b",
    "comfyui-impact-pack",
    "comfyui-impact-subpack",
    "comfyui-controlnet-aux",
    "comfyui-inspire-pack",
    "comfyui-advanced-controlnet",
    "comfyui-h3-motion-context",
    "comfyui-mmh3-ultimate-upscale",
}
_MANDATORY_MODEL_IDS = {
    "realesrgan-x4plus",
    "esrgan-4x",
    "realesrgan-x4plus-anime6b",
    "sdxl-controlnet-canny-xinsir",
    "sdxl-controlnet-depth-xinsir",
    "sdxl-controlnet-tile-xinsir",
    "sdxl-controlnet-openpose-xinsir",
    "sdxl-controlnet-union-xinsir",
    "anima-lllite-regional-controlnet",
    "controlnet-openpose-yolox",
    "controlnet-openpose-dwpose",
}
_STARTER_ASSETS = {
    "sdxl-checkpoint": ("sdxl", 6938042316, "7beb0f8fab3d00c15d370142cb5ab9663cc4bffaf119dca74c519fafe53e25f4"),
    "sdxl-vae": ("sdxl", 334641164, "63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e"),
    "anima-base-v1": ("anima", 4182218328, "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"),
    "anima-text-encoder": ("anima", 1192135096, "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"),
    "anima-vae": ("anima", 253806246, "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
    "anima-3.8b-diffusion": ("anima-3.8b", 7504189974, "1432c925752447df86da7b277e3797f077d358bc24e3950685b13cc0e465c7d5"),
    "anima-3.8b-qwen35": ("anima-3.8b", 4779016600, "ea289be7c916726d09953c7db9971c82b280e694b5d7c47f8ad9ffad6acb54ba"),
    "anima-3.8b-progressive-adapter": ("anima-3.8b", 88131712, "f9851ac4668ce069f7be7cf99755335c98879b463f3d486aaa731083978f0d71"),
}


@dataclass(frozen=True)
class ReleaseArtifact:
    id: str
    label: str
    kind: str
    ownership: str
    version: str
    provider: str
    url: str
    revision: str
    relative_path: Path
    size_bytes: int
    checksum_sha256: str
    license_id: str
    license_url: str
    repair_policy: str
    verified_at: str
    provenance_url: str


@dataclass(frozen=True)
class ReleasePythonPackageSet:
    id: str
    ownership: str
    version: str
    index_url: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseManifest:
    version: int
    product_id: str
    product_version: str
    os: str
    arch: str
    gpu: str
    minimum_nvidia_driver: str
    artifacts: tuple[ReleaseArtifact, ...]
    package_sets: tuple[ReleasePythonPackageSet, ...]


class ReleaseManifestRepository:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def load(self) -> ReleaseManifest:
        try:
            payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("release manifest could not be loaded") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("release manifest version is unsupported")
        product = self._mapping(payload.get("product"), "product")
        platform = self._mapping(payload.get("platform"), "platform")
        artifacts_raw = payload.get("artifacts")
        package_sets_raw = payload.get("python_package_sets")
        if not isinstance(artifacts_raw, list) or not artifacts_raw:
            raise ValueError("release artifacts are missing")
        if not isinstance(package_sets_raw, list) or not package_sets_raw:
            raise ValueError("release package sets are missing")

        artifacts = tuple(self._parse_artifact(item) for item in artifacts_raw)
        artifact_ids = [artifact.id for artifact in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("release manifest has duplicate artifact id")
        targets = [self._normalized_path(artifact.relative_path) for artifact in artifacts]
        if len(set(targets)) != len(targets):
            raise ValueError("release manifest has duplicate artifact target")

        package_sets = tuple(self._parse_package_set(item) for item in package_sets_raw)
        package_set_ids = [package_set.id for package_set in package_sets]
        if len(set(package_set_ids)) != len(package_set_ids):
            raise ValueError("release manifest has duplicate package set id")

        return ReleaseManifest(
            version=1,
            product_id=self._required_string(product.get("id"), "product id"),
            product_version=self._required_string(product.get("version"), "product version"),
            os=self._required_string(platform.get("os"), "platform os"),
            arch=self._required_string(platform.get("arch"), "platform arch"),
            gpu=self._required_string(platform.get("gpu"), "platform gpu"),
            minimum_nvidia_driver=self._required_string(
                platform.get("minimum_nvidia_driver"),
                "minimum NVIDIA driver",
            ),
            artifacts=artifacts,
            package_sets=package_sets,
        )

    def _parse_artifact(self, item: object) -> ReleaseArtifact:
        entry = self._mapping(item, "artifact")
        source = self._mapping(entry.get("source"), "artifact source")
        target = self._mapping(entry.get("target"), "artifact target")
        license_info = self._mapping(entry.get("license"), "artifact license")
        ownership = self._required_string(entry.get("ownership"), "artifact ownership")
        if ownership not in _OWNERSHIP_CLASSES:
            raise ValueError("release artifact ownership is invalid")
        provider = self._required_string(source.get("provider"), "artifact provider")
        url = self._required_string(source.get("url"), "artifact URL")
        revision = self._required_string(source.get("revision"), "artifact revision")
        if provider != "https" or not self._is_https(url):
            raise ValueError("release artifact source must use HTTPS")
        lowered_url = url.casefold()
        if (
            "/resolve/main/" in lowered_url
            or "/latest/" in lowered_url
            or lowered_url.endswith("/latest")
            or lowered_url.endswith(".git")
        ):
            raise ValueError("release artifact URL must be immutable")
        size_bytes = entry.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ValueError("release artifact size must be a positive integer")
        checksum = entry.get("checksum_sha256")
        if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
            raise ValueError("release artifact checksum must be lowercase SHA-256")
        repair_policy = self._required_string(entry.get("repair_policy"), "artifact repair policy")
        if repair_policy not in _REPAIR_POLICIES:
            raise ValueError("release artifact repair policy is invalid")
        return ReleaseArtifact(
            id=self._required_string(entry.get("id"), "artifact id"),
            label=self._required_string(entry.get("label"), "artifact label"),
            kind=self._required_string(entry.get("kind"), "artifact kind"),
            ownership=ownership,
            version=self._required_string(entry.get("version"), "artifact version"),
            provider=provider,
            url=url,
            revision=revision,
            relative_path=self._relative_path(target.get("relative_path"), "artifact target"),
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            license_id=self._required_string(license_info.get("id"), "artifact license id"),
            license_url=self._https_url(license_info.get("url"), "artifact license URL"),
            repair_policy=repair_policy,
            verified_at=self._verified_date(source.get("verified_at"), "artifact verification date"),
            provenance_url=self._https_url(source.get("provenance_url"), "artifact provenance URL"),
        )

    def _parse_package_set(self, item: object) -> ReleasePythonPackageSet:
        entry = self._mapping(item, "package set")
        ownership = self._required_string(entry.get("ownership"), "package set ownership")
        requirements = entry.get("requirements")
        if ownership not in _OWNERSHIP_CLASSES:
            raise ValueError("release package set ownership is invalid")
        if (
            not isinstance(requirements, list)
            or not requirements
            or not all(isinstance(requirement, str) for requirement in requirements)
            or any(_EXACT_REQUIREMENT_RE.fullmatch(requirement) is None for requirement in requirements)
        ):
            raise ValueError("release package set requirements must use exact versions")
        return ReleasePythonPackageSet(
            id=self._required_string(entry.get("id"), "package set id"),
            ownership=ownership,
            version=self._required_string(entry.get("version"), "package set version"),
            index_url=self._https_url(entry.get("index_url"), "package set index URL"),
            requirements=tuple(requirements),
        )

    @staticmethod
    def _mapping(value: object, label: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError(f"release manifest {label} must be an object")
        return value

    @staticmethod
    def _required_string(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"release manifest {label} is missing")
        return value.strip()

    @classmethod
    def _https_url(cls, value: object, label: str) -> str:
        url = cls._required_string(value, label)
        if not cls._is_https(url):
            raise ValueError(f"release manifest {label} must use HTTPS")
        return url

    @staticmethod
    def _is_https(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme == "https" and bool(parsed.netloc)

    @classmethod
    def _relative_path(cls, value: object, label: str) -> Path:
        raw = cls._required_string(value, label)
        path = Path(raw)
        windows_path = PureWindowsPath(raw)
        posix_path = PurePosixPath(raw)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or posix_path.is_absolute()
            or windows_path.drive
            or ".." in (*windows_path.parts, *posix_path.parts)
        ):
            raise ValueError("release artifact target must stay within the application root")
        return path

    @staticmethod
    def _normalized_path(path: Path) -> str:
        return path.as_posix().replace("\\", "/").casefold()

    @classmethod
    def _verified_date(cls, value: object, label: str) -> str:
        verified_at = cls._required_string(value, label)
        if _VERIFIED_DATE_RE.fullmatch(verified_at) is None:
            raise ValueError(f"release manifest {label} must use YYYY-MM-DD")
        return verified_at


def validate_release_catalogs(
    runtime_dependencies: Iterable[object],
    model_dependencies: Iterable[object],
) -> None:
    """Fail closed when the feature catalogs cannot reproduce the V1 release."""

    runtimes = list(runtime_dependencies)
    models = list(model_dependencies)
    runtime_ids = {str(getattr(item, "id", "")) for item in runtimes}
    mandatory_model_ids = {
        str(getattr(item, "id", ""))
        for item in models
        if getattr(item, "release_class", None) == "mandatory-feature"
    }
    starter_ids = {
        str(getattr(item, "id", ""))
        for item in models
        if getattr(item, "release_class", None) == "optional-starter"
    }
    if runtime_ids != _MANDATORY_RUNTIME_IDS:
        raise ValueError("release catalog runtime component set is incomplete")
    if mandatory_model_ids != _MANDATORY_MODEL_IDS:
        raise ValueError("release catalog functional model set is incomplete")
    if starter_ids != set(_STARTER_ASSETS):
        raise ValueError("release catalog starter asset set is incomplete")

    runtime_targets: set[str] = set()
    for item in runtimes:
        _validate_runtime_catalog_item(item)
        target = _catalog_target(item)
        if target in runtime_targets:
            raise ValueError("release catalog has duplicate runtime target")
        runtime_targets.add(target)

    model_targets: set[str] = set()
    for item in models:
        _validate_model_catalog_item(item)
        target = _catalog_target(item)
        if target in model_targets:
            raise ValueError("release catalog has duplicate model target")
        model_targets.add(target)


def _validate_runtime_catalog_item(item: object) -> None:
    if getattr(item, "release_class", None) != "mandatory-feature":
        raise ValueError("release catalog runtime ownership is invalid")
    provider = getattr(item, "provider", None)
    if provider == "bundled":
        bundled_contracts = {
            "asteria-prompt-syntax": (
                "runtime-components/comfyui/Asteria-Prompt-Syntax",
                "asteria.prompt.step-schedule.v1",
            ),
            "asteria-merge-runtime": (
                "runtime-components/comfyui/Asteria-Merge-Runtime",
                "asteria.merge.runtime.v1",
            ),
            "asteria-h3-latent-assembly": (
                "runtime-components/comfyui/Asteria-H3-Latent-Assembly",
                "asteria.video.sequence.latent-assembly.v1",
            ),
            "asteria-anima-lora-remap": (
                "runtime-components/comfyui/Asteria-Anima-LoRA-Remap",
                "asteria.anima_lora_remap.v1",
            ),
            "asteria-qwen2d-vae": (
                "runtime-components/comfyui/Asteria-Qwen2D-VAE",
                "asteria.qwen2d_vae.v1",
            ),
            "asteria-spatial-prompt": (
                "runtime-components/comfyui/Asteria-Spatial-Prompt",
                "asteria.spatial-prompt.node.v4",
            ),
        }
        expected = bundled_contracts.get(getattr(item, "id", None))
        if expected is None:
            raise ValueError("release catalog bundled runtime ownership is invalid")
        if getattr(item, "url", None) != expected[0]:
            raise ValueError("release catalog bundled runtime source is invalid")
        if getattr(item, "ref", None) != expected[1]:
            raise ValueError("release catalog bundled runtime contract is invalid")
        if getattr(item, "auto_install", None) is not True:
            raise ValueError("release catalog mandatory runtime must auto install")
        return
    if provider != "archive":
        raise ValueError("release catalog runtime provider must use an archive")
    _validate_managed_source(item, require_commit_revision=True)
    if getattr(item, "auto_install", None) is not True:
        raise ValueError("release catalog mandatory runtime must auto install")


def _validate_model_catalog_item(item: object) -> None:
    release_class = getattr(item, "release_class", None)
    model_type = getattr(item, "type", None)
    if release_class == "mandatory-feature":
        if model_type in {"checkpoint", "text_encoder", "vae"}:
            raise ValueError("release catalog generation weights cannot be mandatory")
        _validate_managed_source(item, require_commit_revision=False)
        if getattr(item, "auto_install", None) is not True:
            raise ValueError("release catalog mandatory model must auto install")
        return
    if release_class == "optional-starter":
        item_id = str(getattr(item, "id", ""))
        expected_pack, expected_size, expected_hash = _STARTER_ASSETS.get(item_id, (None, None, None))
        if getattr(item, "starter_pack", None) != expected_pack:
            raise ValueError("release catalog starter group is invalid")
        if getattr(item, "selected_by_default", None) is not False or getattr(item, "auto_install", None) is not False:
            raise ValueError("release catalog starter assets must be opt in")
        _validate_managed_source(item, require_commit_revision=False)
        if getattr(item, "size_bytes", None) != expected_size or getattr(item, "checksum_sha256", None) != expected_hash:
            raise ValueError("release catalog starter metadata changed")
        if expected_pack == "anima" and getattr(item, "requires_license_confirmation", None) is not True:
            raise ValueError("release catalog Anima license confirmation is required")
        return
    if release_class == "user-owned":
        if getattr(item, "auto_install", None) is not False:
            raise ValueError("release catalog user-owned assets cannot auto install")
        return
    raise ValueError("release catalog model ownership is invalid")


def _validate_managed_source(item: object, *, require_commit_revision: bool) -> None:
    url = getattr(item, "url", None)
    revision = getattr(item, "ref", None) if require_commit_revision else getattr(item, "source_revision", None)
    if not isinstance(url, str) or not ReleaseManifestRepository._is_https(url):
        raise ValueError("release catalog managed source must use HTTPS")
    lowered_url = url.casefold()
    if "/resolve/main/" in lowered_url or "/latest/" in lowered_url or lowered_url.endswith(".git"):
        raise ValueError("release catalog managed source must be immutable")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("release catalog managed source revision is missing")
    if require_commit_revision and _REVISION_SHA_RE.fullmatch(revision) is None:
        raise ValueError("release catalog runtime revision must be a commit SHA")
    size_bytes = getattr(item, "size_bytes", None)
    checksum = getattr(item, "checksum_sha256", None)
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
        raise ValueError("release catalog managed size is invalid")
    if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError("release catalog managed checksum is invalid")
    if not isinstance(getattr(item, "license_id", None), str) or not getattr(item, "license_id").strip():
        raise ValueError("release catalog managed license is missing")
    license_url = getattr(item, "license_url", None)
    if not isinstance(license_url, str) or not ReleaseManifestRepository._is_https(license_url):
        raise ValueError("release catalog managed license URL is missing")
    if getattr(item, "repair_policy", None) != "replace-managed-component":
        raise ValueError("release catalog managed repair policy is invalid")
    verified_at = getattr(item, "source_verified_at", None)
    if not isinstance(verified_at, str) or _VERIFIED_DATE_RE.fullmatch(verified_at) is None:
        raise ValueError("release catalog verification date is missing")
    provenance_url = getattr(item, "source_provenance_url", None)
    if not isinstance(provenance_url, str) or not ReleaseManifestRepository._is_https(provenance_url):
        raise ValueError("release catalog provenance URL is missing")


def _catalog_target(item: object) -> str:
    target = getattr(item, "relative_path", None)
    if not isinstance(target, Path):
        raise ValueError("release catalog target is missing")
    return target.as_posix().replace("\\", "/").casefold()
