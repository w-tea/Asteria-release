from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.package_environment import PackageEnvironmentError, PackageEnvironmentManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install the minimal Asteria backend environment")
    parser.add_argument("--app-root", type=Path, required=True)
    return parser


def _emit(code: str, **values: object) -> None:
    print(json.dumps({"code": code, **values}, ensure_ascii=False, separators=(",", ":")), flush=True)


def main() -> int:
    args = _parser().parse_args()
    if not args.app_root.is_absolute():
        _emit("bootstrap_backend_environment_invalid")
        return 64
    app_root = args.app_root.resolve()
    try:
        result = PackageEnvironmentManager(app_root).install(
            "asteria-backend-environment",
            app_root / "runtime" / "python" / "python.exe",
            app_root / "app" / "backend" / "requirements-release.lock",
        )
    except (OSError, ValueError, PackageEnvironmentError):
        _emit("bootstrap_backend_environment_failed")
        return 69
    _emit("bootstrap_backend_environment_ready", status=result.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
