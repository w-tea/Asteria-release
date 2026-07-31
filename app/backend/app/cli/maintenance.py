from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from app.services.release_signing import (
    load_trusted_public_keys,
    verify_artifact_hashes,
    verify_signed_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asteria-maintenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-release")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--signature", type=Path, required=True)
    verify.add_argument("--trusted-keys", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path)
    verify.add_argument("--current-data-application-version", default="")
    verify.add_argument("--coordinated-data-restore", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-release":
        payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
        envelope = json.loads(args.signature.read_text(encoding="utf-8"))
        verified = verify_signed_manifest(
            payload,
            envelope,
            trusted_public_keys=load_trusted_public_keys(args.trusted_keys),
            current_data_application_version=args.current_data_application_version or None,
            coordinated_data_restore=args.coordinated_data_restore,
        )
        if args.artifact_root is not None:
            verify_artifact_hashes(payload, args.artifact_root)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "key_id": verified.key_id,
                    "product_version": verified.product_version,
                    "canonical_sha256": verified.canonical_sha256,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
