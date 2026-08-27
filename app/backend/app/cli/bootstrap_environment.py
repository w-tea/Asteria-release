from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.package_environment import PackageEnvironmentError, PackageEnvironmentManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install the minimal Asteria backend environment")
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("release-verifier", "backend"),
        default="backend",
    )
    return parser


def _emit(code: str, **values: object) -> None:
    print(json.dumps({"code": code, **values}, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> int:
    args = _parser().parse_args()
    if not args.app_root.is_absolute():
        _emit("bootstrap_backend_environment_invalid")
        return 64
    app_root = args.app_root.resolve()
    profiles = {
        "release-verifier": (
            "asteria-release-verifier",
            app_root / "app" / "backend" / "requirements-release-verifier.lock",
        ),
        "backend": (
            "asteria-backend-environment",
            app_root / "app" / "backend" / "requirements-release.lock",
        ),
    }
    environment_id, lock_path = profiles[args.profile]
    try:
        result = PackageEnvironmentManager(app_root).install(
            environment_id,
            app_root / "runtime" / "python" / "python.exe",
            lock_path,
        )
    except (OSError, ValueError, PackageEnvironmentError):
        _emit("bootstrap_backend_environment_failed")
        return 69
    _emit(
        "bootstrap_backend_environment_ready",
        component_id=environment_id,
        status=result.status,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
