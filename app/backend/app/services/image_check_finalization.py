"""Check completed candidates once, then publish only explicitly retained files."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import logging

from PIL import Image

from app.services.atomic_json import write_json_atomic
from app.services.generation_artifacts import apply_filename_template, apply_output_image_format, remap_primary_outputs, unique_output_path
from app.services.image_check import decide_image_check, remap_checked_image_paths
from app.services.image_check_operation import ImageCheckOperation
from app.services.image_check_recovery import (
    PENDING_DIRECTORY, PUBLICATION_SCHEMA, file_identity, move_without_replacement,
    owns_file, recover_image_publications,
)


_active_image_session: ContextVar[FinalImageSession | None] = ContextVar('final_image_session', default=None)


def replace_final_metadata(source: Path, destination: Path) -> None:
    session = _active_image_session.get()
    if session is None:
        source.replace(destination)
    else:
        session.replace_metadata(source, destination)


def is_private_image_candidate(path: str | Path) -> bool:
    return PENDING_DIRECTORY in {part.casefold() for part in Path(path).parts}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream) -> str:
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def finalize_completed_batch(
    operation: ImageCheckOperation,
    result: dict,
    settings: dict,
    *,
    batch_index: int = 0,
    checker: Callable | None = None,
    cancel_requested: Callable | None = None,
    progress: Callable | None = None,
) -> dict:
    """No sampling or correction occurs after this boundary."""
    from app.services.image_check_runtime import assess_image

    if result.get("image_check_finalized"):
        return result
    before = list(result["output_paths"])
    for value in before:
        path = Path(value)
        if path.is_symlink() or not path.resolve().is_relative_to(operation.output_root.resolve()):
            raise ValueError("image_check_candidate_outside_operation_root")
    image_format = operation.output_settings.image_format if operation.output_settings else "png"
    converted = apply_output_image_format(before, image_format)
    primary, mapped = remap_primary_outputs(result.get("primary_outputs", []), before, converted)
    if not mapped or not primary:
        raise ValueError("image_check_final_output_manifest_invalid")
    conversion = dict(zip(before, converted, strict=True))
    primary_paths = {item["path"] for item in primary}
    records, retained, accepted = [], set(), []
    paused = False
    cancelled = False
    assessor = checker or assess_image
    for image_index, item in enumerate(primary):
        if callable(cancel_requested) and cancel_requested():
            cancelled = True
            break
        image = Path(item["path"])
        if callable(progress):
            progress("image_check")
        original_hash = file_sha256(image)
        checked = assessor(operation, image, cancel_requested)
        if callable(cancel_requested) and cancel_requested():
            cancelled = True
            break
        if file_sha256(image) != original_hash:
            raise ValueError("image_check_candidate_changed")
        decision = decide_image_check(operation.settings, checked.assessment)
        with Image.open(image) as opened:
            width, height = opened.size
        record = {
            "candidate_id": image.stem, "target_stage": "final_output",
            "batch_index": batch_index, "image_index": image_index,
            "output_index": item.get("output_index", image_index),
            "seed": item.get("seed", settings.get("seed")),
            "runtime_job_id": item.get("runtime_job_id", result.get("runtime_job_id", "")),
            "status": decision.status, "source_sha256": original_hash,
            "source_width": width, "source_height": height, "source_format": image.suffix.lower(),
            "assessment": checked.assessment.model_dump(mode="json") if checked.assessment else None,
            "receipt": asdict(checked.receipt),
        }
        records.append(record)
        if decision.status == "pass" or decision.save_rejected_image:
            retained.add(str(image))
            record["saved_path"] = str(image)
        if decision.status == "pass":
            accepted.append({**item, "batch_index": batch_index, "image_index": image_index})
        if decision.pause:
            paused = True
            break

    accepted_indices = {item["image_index"] for item in accepted}
    auxiliary = []
    # Runtime-owned image identities avoid assuming that a filtered file order
    # still matches the original batch or its first-pass images.
    for item in result.get("image_outputs", []):
        path = conversion.get(item["path"], item["path"])
        if path not in primary_paths and item.get("image_index") in accepted_indices:
            auxiliary.append(path)
            retained.add(path)
    stages = [{**item, "batch_index": batch_index} for item in result.get("enhancement_stages", [])
              if item.get("image_index") in accepted_indices]
    return {**result, "status": "cancelled" if cancelled else result["status"],
            "output_paths": [path for path in converted if path in retained], "primary_outputs": accepted,
            "enhancement_stages": stages, "approved_auxiliary_paths": [*auxiliary, *(s["path"] for s in stages)],
            "image_check_results": records, "image_check_pause": paused or cancelled,
            "image_check_unjudged": len(primary) - len(records), "image_check_finalized": True}


class FinalImageSession:
    """Own the private candidate directory and a small publication journal."""

    def __init__(self, operation: ImageCheckOperation, final_dir: Path, *, history=None):
        self.operation = operation
        self.root = operation.output_root.resolve()
        self.final_dir = final_dir.resolve()
        self.directory: Path | None = None
        self.moves: list[dict] = []
        self.reserved_paths: set[Path] = set()
        self.committed = False
        self.history = history
        self._context_token = None

    def __enter__(self):
        if not self.final_dir.is_relative_to(self.root):
            raise ValueError("image_check_output_root_mismatch")
        parent = self.root / PENDING_DIRECTORY
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError("image_check_stage_root_mismatch")
        self.directory = Path(tempfile.mkdtemp(prefix="generation-", dir=parent))
        self._journal("generating")
        if self.history is not None:
            self.history.begin_image_publication(self.directory.name, self.root, self.directory)
        self._context_token = _active_image_session.set(self)
        return self

    def _journal(self, phase: str) -> None:
        write_json_atomic(self.directory / "publication.json", {
            "schema_version": PUBLICATION_SCHEMA, "phase": phase, "publication_id": self.directory.name,
            "output_root": str(self.root), "moves": self.moves,
        })

    def publish(self, paths: list[str], settings: dict, profile: str, generation_id: int) -> list[str]:
        check = settings.get("image_check", {})
        records = {item["saved_path"]: item for item in check.get("results", []) if item.get("saved_path")}
        auxiliary = set(check.get("auxiliary_paths", []))
        self.final_dir.mkdir(parents=True, exist_ok=True)
        destinations = []
        for value in paths:
            source = Path(value)
            if value not in records and value not in auxiliary:
                raise ValueError("image_check_unjudged_output")
            if source.is_symlink() or not source.resolve().is_relative_to(self.directory):
                raise ValueError("image_check_candidate_outside_session")
            record = records.get(value, {})
            if record.get("source_sha256") and file_sha256(source) != record["source_sha256"]:
                raise ValueError("image_check_candidate_changed")
            identity = {"seed": record.get("seed", settings.get("seed", "")),
                        "index": int(record.get("output_index", len(destinations))) + 1}
            # Resolve the final name while the file is still private.
            renamed = apply_filename_template([str(source)], {**settings, "generated_seeds": [identity["seed"]]},
                                              profile, generation_id, identities=[identity])
            source = Path(renamed[0])
            destination = unique_output_path(self.final_dir / source.name, include_sidecar=True,
                                             reserved_paths=self.reserved_paths)
            self.reserved_paths.update((destination, destination.with_suffix('.asteria.json')))
            self._publish_file(source, destination)
            # Reserve the sidecar as an owned file, not only a name checked
            # before another publisher might create it.
            sidecar_source = self.directory / f'sidecar-{len(self.moves)}.json'
            sidecar_source.write_text('{}', encoding='utf-8')
            self._publish_file(sidecar_source, destination.with_suffix('.asteria.json'))
            destinations.append(str(destination))
        remap_checked_image_paths(settings, paths, destinations)
        return destinations

    def _publish_file(self, source: Path, destination: Path) -> None:
        self.moves.append({'source': str(source), 'destination': str(destination),
                           'identities': [file_identity(source)], 'moved': False})
        self._journal('publishing')
        move_without_replacement(source, destination)
        self.moves[-1]['moved'] = True
        self._journal('publishing')

    def replace_metadata(self, source: Path, destination: Path) -> None:
        move = next((item for item in self.moves if item['destination'] == str(destination)), None)
        if move is None or not owns_file(destination, move['identities']):
            raise ValueError('image_publication_metadata_not_owned')
        move['identities'].append(file_identity(source))
        self._journal('committed' if self.committed else 'publishing')
        source.replace(destination)

    def seal_history(self, generation_id: int) -> None:
        if self.history is not None:
            self.history.complete_image_publication(self.directory.name, generation_id)

    def commit(self):
        if self.history is not None and not self.history.image_publication_committed(self.directory.name):
            raise RuntimeError('image_publication_history_not_committed')
        self.committed = True
        try:
            self._journal("committed")
        except OSError:
            logging.getLogger(__name__).warning("Image publication committed; journal update failed")

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.history is not None and not self.committed:
                self.committed = self.history.image_publication_committed(self.directory.name)
            if not self.committed:
                for move in reversed(self.moves):
                    destination = Path(move['destination'])
                    if destination.is_relative_to(self.final_dir) and owns_file(destination, move['identities']):
                        destination.unlink()
                if self.history is not None:
                    self.history.prepare_image_publication_cleanup(self.directory.name)
            if self.directory and self.directory.parent == self.root / PENDING_DIRECTORY:
                shutil.rmtree(self.directory)
            if self.history is not None:
                self.history.forget_image_publication(self.directory.name)
        except OSError:
            if not self.committed:
                raise
            logging.getLogger(__name__).warning('Committed image publication left private temporary files')
        finally:
            if self._context_token is not None:
                _active_image_session.reset(self._context_token)
