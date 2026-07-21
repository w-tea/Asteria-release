from __future__ import annotations

import argparse
import json
from typing import Any

from app.api.startup import StartupReadinessResponse, startup_readiness


EXIT_READY = 0
EXIT_APP_BLOCKED = 10
EXIT_RUNTIME_BLOCKED = 11
EXIT_GENERATION_BLOCKED = 12
EXIT_EXECUTION_FAILED = 70


def _exit_code(response: StartupReadinessResponse) -> int:
    summary = response.summary
    if not summary.app_ready:
        return EXIT_APP_BLOCKED
    if not summary.runtime_ready:
        return EXIT_RUNTIME_BLOCKED
    if not summary.generation_ready:
        return EXIT_GENERATION_BLOCKED
    return EXIT_READY


def _payload(response: StartupReadinessResponse) -> dict[str, Any]:
    exit_code = _exit_code(response)
    return {
        "status": "ready" if exit_code == EXIT_READY else "blocked",
        "code": "startup_ready" if exit_code == EXIT_READY else "startup_readiness_blocked",
        "exit_code": exit_code,
        "report": response.model_dump(mode="json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Asteria startup readiness without changing setup.")
    parser.add_argument("--target-mode", choices=("development", "packaged"), default="development")
    parser.add_argument("--profile-id", default="sdxl")
    parser.add_argument("--feature", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        response = startup_readiness(
            target_mode=args.target_mode,
            profile_id=args.profile_id,
            feature=args.feature,
        )
        payload = _payload(response)
    except Exception:
        payload = {
            "status": "error",
            "code": "startup_readiness_execution_failed",
            "exit_code": EXIT_EXECUTION_FAILED,
            "report": None,
        }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
