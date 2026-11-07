"""User-invoked candidate review. This service never judges anatomy or ownership."""
from __future__ import annotations

from contextlib import suppress
from hashlib import file_digest
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, RLock
import time

from PIL import Image, ImageOps

from app.services.atomic_json import write_json_atomic

CONTRACT = 'asteria.body-review.v1'
DETECTOR_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
DETECTOR_SHA = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
SAM_SHA = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
MAX_PIXELS = 32_000_000
MAX_SECONDS = 180
MAX_RAM = 6 * 1024**3
_lock = RLock()
_active: dict[str, Event] = {}
_cancelled: dict[str, float] = {}


def _pending_cancellation(operation_id: str) -> bool:
    with _lock:
        return time.monotonic() - _cancelled.get(operation_id, -1000) < 300


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def review_candidates(boxes, scores, labels, width, height) -> list[dict]:
    """Keep duplicates visible; labels are detector suggestions, never person IDs."""
    results = []
    for query, (box, values) in enumerate(zip(boxes, scores, strict=True)):
        if not all(math.isfinite(v) for v in box):
            continue
        rect = [min(1., max(0., v / size)) for v, size in zip(box, [width, height, width, height])]
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            continue
        for label, score in zip(labels, values, strict=True):
            if math.isfinite(score) and score >= .25:
                results.append(dict(id=f'q{query}-{label}', label=label, score=float(score), box=rect))
    return sorted(results, key=lambda row: (-row['score'], row['id']))


class BodyReviewService:
    def __init__(self, data_dir: Path, model_dir: Path):
        self.data_dir, self.model_dir = Path(data_dir), Path(model_dir)
        self.index_path = self.data_dir / 'body-review-index.json'

    def model_paths(self) -> tuple[Path, Path]:
        return (self.model_dir / 'body_review/grounding-dino-tiny',
                self.model_dir / 'sams/sam_vit_b_01ec64.pth')

    def capabilities(self) -> dict:
        detector, sam = self.model_paths()
        worker_packages = all(importlib.util.find_spec(name) for name in ('torch', 'psutil'))
        packages = worker_packages and importlib.util.find_spec('transformers')
        files = ('model.safetensors', 'config.json', 'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json')
        return dict(detector=bool(packages and all((detector / name).is_file() for name in files)),
                    sam=bool(worker_packages and importlib.util.find_spec('segment_anything') and sam.is_file()),
                    manual_editing=True, max_candidates=96, max_selection=16,
                    detector_model='Grounding DINO tiny', mask_model='SAM ViT-B',
                    model_license='Apache-2.0', device='cpu')

    def _index(self) -> dict:
        if not self.index_path.exists():
            return {}
        if self.index_path.stat().st_size > 8 * 1024**2:
            raise ValueError('body_review_index_limit')
        return json.loads(self.index_path.read_text('utf-8'))

    def create(self, source: Path, root: Path, operation_id: str) -> dict:
        if not re.fullmatch('[0-9a-f]{32}', operation_id):
            raise ValueError('body_review_id_invalid')
        source = source.resolve(strict=True)
        if source.stat().st_size > 64 * 1024**2:
            raise ValueError('body_review_image_limit')
        before = digest(source)
        with Image.open(source) as opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError('body_review_image_limit')
            image = ImageOps.exif_transpose(opened).convert('RGB')
        if digest(source) != before:
            raise ValueError('body_review_source_changed')
        root = root.resolve()
        parent = root / '_body_review'
        parent.mkdir(parents=True, exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError('body_review_path_invalid')
        with _lock:
            index = self._index()
            if operation_id in index:
                raise ValueError('body_review_exists')
            directory = parent / operation_id
            directory.mkdir(exist_ok=False)
            record = dict(contract=CONTRACT, id=operation_id, directory=str(directory),
                          output_root=str(root), source_path=str(source), source_sha256=before,
                          width=image.width, height=image.height, status='prepared', candidates=[], masks=[])
            image.save(directory / 'source.png')
            record['snapshot_sha256'] = digest(directory / 'source.png')
            write_json_atomic(directory / 'review.json', record)
            index[operation_id] = str(directory)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.index_path, index)
        return record

    def get(self, review_id: str) -> dict:
        if not re.fullmatch('[0-9a-f]{32}', review_id):
            raise ValueError('body_review_id_invalid')
        with _lock:
            directory = Path(self._index().get(review_id, ''))
        if not directory.is_absolute() or not (directory / 'review.json').is_file():
            raise ValueError('body_review_unavailable')
        if directory.is_symlink() or directory.resolve() != directory:
            raise ValueError('body_review_path_invalid')
        record = json.loads((directory / 'review.json').read_text('utf-8'))
        if (record.get('contract') != CONTRACT or record.get('id') != review_id
                or directory.parent != Path(record['output_root']) / '_body_review'):
            raise ValueError('body_review_path_invalid')
        return record

    def checked(self, review_id: str) -> dict:
        record = self.get(review_id)
        if digest(Path(record['source_path'])) != record['source_sha256']:
            raise ValueError('body_review_source_changed')
        if digest(Path(record['directory']) / 'source.png') != record.get('snapshot_sha256'):
            raise ValueError('body_review_source_changed')
        return record

    def latest(self, source: Path) -> dict | None:
        source = source.resolve()
        for key in reversed(list(self._index())):
            row = self.get(key)
            if Path(row['source_path']) == source and row['status'] == 'completed':
                try:
                    return self.checked(key)
                except ValueError as exc:
                    if str(exc) != 'body_review_source_changed':
                        raise
                    # An old/changed snapshot must not block an explicit new review.
                    return None
        return None

    def cancel(self, operation_id: str):
        if not re.fullmatch('[0-9a-f]{32}', operation_id):
            return
        with _lock:
            for key, at in list(_cancelled.items()):
                if time.monotonic() - at > 300:
                    del _cancelled[key]
            if len(_cancelled) >= 128:
                _cancelled.pop(next(iter(_cancelled)))
            _cancelled[operation_id] = time.monotonic()
            event = _active.get(operation_id)
            if event:
                event.set()

    def run(self, record: dict, stage: str, operation_id: str, selections: list[dict] | None = None) -> dict:
        import psutil
        from uuid import uuid4
        if not re.fullmatch('[0-9a-f]{32}', operation_id):
            raise ValueError('body_review_id_invalid')
        capability = self.capabilities()
        if not capability['detector' if stage == 'detect' else 'sam']:
            raise ValueError('body_review_model_unavailable')
        with _lock:
            if _pending_cancellation(operation_id):
                raise ValueError('body_review_cancelled')
            if _active:
                raise ValueError('body_review_busy')
            event = Event()
            _active[operation_id] = event
        directory = Path(record['directory'])
        token = uuid4().hex
        spec = directory / f'{token}-input.json'
        output = directory / f'{token}-result.json'
        detector, sam = self.model_paths()
        try:
            write_json_atomic(spec, dict(stage=stage, source=str(directory / 'source.png'),
                detector=str(detector), sam=str(sam), parts=record.get('parts', []),
                candidates=selections, output=str(output), mask_prefix=token))
        except Exception:
            with _lock:
                _active.pop(operation_id, None)
            raise
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        started, peak, reason, process = time.monotonic(), 0, '', None
        try:
            with (directory / f'{token}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-m', 'app.services.body_review_worker', str(spec)],
                    cwd=Path(__file__).resolve().parents[2], env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                owner = psutil.Process(process.pid)
                while process.poll() is None:
                    with suppress(psutil.Error):
                        peak = max(peak, sum(p.memory_info().rss for p in [owner, *owner.children(recursive=True)]))
                    reason = ('body_review_cancelled' if event.is_set() else
                              'body_review_time_limit' if time.monotonic() - started > MAX_SECONDS else
                              'body_review_ram_limit' if peak > MAX_RAM else '')
                    if reason:
                        process.kill()
                        break
                    event.wait(.15)
                code = process.wait()
            self.checked(record['id'])
            if reason or event.is_set() or code != 0 or not output.is_file():
                raise ValueError(reason or 'body_review_inference_failed')
            result = json.loads(output.read_text('utf-8'))
            with _lock:
                if event.is_set():
                    raise ValueError('body_review_cancelled')
                record = self.checked(record['id'])
                if stage == 'detect':
                    record.update(candidates=result['candidates'], candidate_count=result['candidate_count'],
                                  truncated=result['candidate_count'] > 96, parts=result['parts'], status='completed')
                else:
                    record.update(masks=result['masks'], selection=selections, mask_prefix=token)
                record[stage + '_receipt'] = dict(seconds=time.monotonic()-started, peak_ram_bytes=peak,
                                                  device='cpu', models=result['models'])
                write_json_atomic(directory / 'review.json', record)
            return record
        finally:
            if process is not None and process.poll() is None:
                process.kill(); process.wait()
            with _lock:
                _active.pop(operation_id, None)
