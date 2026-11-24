"""Image-grounded dataset review; never edits a source image or training caption."""
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from PIL import Image, ImageOps

from app.adapters.image_check_image import _read_source, prepare_image_check_image
from app.schemas.lora_dataset_review import DatasetAnnotation, ReviewSave
from app.services.lora_training import LoRATrainingDataset, build_lora_training_item_caption, resolve_lora_training_dataset_item_image_path


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def prepare_item(dataset: LoRATrainingDataset, item_id: str, project_root: Path):
    item = next((x for x in dataset.items if x.id == item_id), None)
    if item is None:
        raise ValueError('review_item_unavailable')
    path = resolve_lora_training_dataset_item_image_path(dataset, item_id, project_root=project_root)
    # Registration is dataset-owned. Reuse the same bounded decoder as generation checks.
    image = prepare_image_check_image(path, output_root=path.parent)
    _, _, raw = _read_source(path, path.parent)
    if sha256(raw).hexdigest() != image.receipt.source_sha256:
        raise ValueError('review_source_changed')
    with Image.open(BytesIO(raw)) as opened, ImageOps.exif_transpose(opened) as oriented, oriented.convert('RGBA') as rgba:
        pixel_hash = sha256(f'{rgba.width}x{rgba.height}:RGBA:'.encode() + rgba.tobytes()).hexdigest()
        with oriented.convert('L') as gray, gray.resize((9, 8), Image.Resampling.LANCZOS) as small:
            pixels = small.tobytes()
            difference = sum(int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
    caption = build_lora_training_item_caption(dataset, item)
    text_hash = _digest({'caption': caption, 'tags': item.tags, 'export': dataset.caption_export_settings.model_dump(),
                         'character_policy': dataset.character_caption_policy.model_dump()})
    source_hash = image.receipt.source_sha256
    snapshot = {'id': item.id, 'filename': path.name, 'caption': caption, 'source_sha256': source_hash,
                'text_sha256': text_hash, 'fingerprint': _digest([source_hash, text_hash]),
                'pixel_sha256': pixel_hash, 'difference_hash': f'{difference:016x}'}
    return snapshot, image


def review_summary(items: list[dict]) -> dict:
    groups = {key: defaultdict(list) for key in ['person_count', 'hair_color', 'hair_style', 'clothing', 'pose', 'background', 'hair_clothing', 'clothing_background']}
    confirmed = [x for x in items if x['state'] == 'confirmed']
    def add(key, values, item_id):
        normalized = tuple(' '.join(str(value).strip().casefold().split()) for value in values)
        if all(normalized):
            groups[key][normalized].append(item_id)
    for item in confirmed:
        annotation = item['review']['annotation']
        if annotation['person_count'] is not None:
            add('person_count', [annotation['person_count']], item['id'])
        add('background', [annotation['background']], item['id'])
        for person in annotation['persons']:
            for key in ['hair_color', 'hair_style', 'clothing', 'pose']:
                add(key, [person[key]], item['id'])
            add('hair_clothing', [person['hair_color'], person['clothing']], item['id'])
            add('clothing_background', [person['clothing'], annotation['background']], item['id'])
    return {'confirmed_images': len(confirmed), 'total_images': len(items), **{
        key: [{'values': list(values), 'count': len(ids), 'item_ids': list(dict.fromkeys(ids))}
              for values, ids in sorted(entries.items(), key=lambda x: (-len(x[1]), x[0]))]
        for key, entries in groups.items()}}


def inspect_dataset(dataset: LoRATrainingDataset, repo, project_root: Path) -> dict:
    if len(dataset.items) > 512:
        raise ValueError('review_dataset_limit')
    items = []
    duplicates = defaultdict(list)
    for item in dataset.items:
        review = repo.latest(dataset.id, item.id)
        try:
            snapshot, _ = prepare_item(dataset, item.id, project_root)
            state = ('stale' if review['snapshot']['fingerprint'] != snapshot['fingerprint'] else
                     'confirmed' if review['confirmed'] else 'draft') if review else 'unchecked'
            items.append({**snapshot, 'state': state, 'review': review})
            duplicates[snapshot['pixel_sha256']].append(item.id)
        except (ValueError, OSError):
            items.append({'id': item.id, 'filename': Path(item.image_path).name, 'state': 'unavailable', 'review': review,
                          'caption': build_lora_training_item_caption(dataset, item)})
    similar = []
    similar_count = 0
    for i, first in enumerate(items):
        if first['state'] == 'unavailable':
            continue
        for second in items[i + 1:]:
            if second['state'] == 'unavailable' or first['pixel_sha256'] == second['pixel_sha256']:
                continue
            distance = (int(first['difference_hash'], 16) ^ int(second['difference_hash'], 16)).bit_count()
            if distance <= 6:
                similar_count += 1
                if len(similar) < 100:
                    similar.append({'item_ids': [first['id'], second['id']], 'distance': distance})
    return {'dataset_id': dataset.id, 'items': items, 'duplicates': [ids for ids in duplicates.values() if len(ids) > 1],
            'similar': similar, 'similar_count': similar_count, 'summary': review_summary(items)}


class DatasetReviewRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS lora_dataset_reviews (id TEXT PRIMARY KEY, dataset_id TEXT, item_id TEXT, document TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS lora_dataset_reviews_item ON lora_dataset_reviews(dataset_id,item_id)')

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def history(self, dataset_id, item_id):
        with closing(self._connect()) as db:
            return [json.loads(row[0]) for row in db.execute('SELECT document FROM lora_dataset_reviews WHERE dataset_id=? AND item_id=? ORDER BY rowid DESC LIMIT 100', (dataset_id, item_id))]

    def latest(self, dataset_id, item_id):
        rows = self.history(dataset_id, item_id)
        return rows[0] if rows else None

    def get(self, review_id):
        with closing(self._connect()) as db:
            row = db.execute('SELECT document FROM lora_dataset_reviews WHERE id=?', (review_id,)).fetchone()
        if row is None:
            raise KeyError('review_not_found')
        return json.loads(row[0])

    def _append(self, dataset_id, item_id, snapshot, previous_id, **fields):
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT id,document FROM lora_dataset_reviews WHERE dataset_id=? AND item_id=? ORDER BY rowid DESC LIMIT 1', (dataset_id, item_id)).fetchone()
            if (row[0] if row else None) != previous_id:
                raise ValueError('review_changed')
            previous = json.loads(row[1]) if row else None
            if previous and previous['status'] == 'running':
                raise ValueError('review_running')
            doc = {'id': uuid4().hex, 'dataset_id': dataset_id, 'item_id': item_id, 'snapshot': snapshot,
                   'previous_id': previous_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                   'annotation': None, 'ai_annotation': None, 'receipt': None, 'error': None, **fields}
            if doc['confirmed'] and previous and previous['snapshot']['fingerprint'] == snapshot['fingerprint']:
                doc['ai_annotation'] = previous['ai_annotation']
                doc['receipt'] = previous['receipt']
            db.execute('INSERT INTO lora_dataset_reviews VALUES (?,?,?,?)', (doc['id'], dataset_id, item_id, json.dumps(doc, ensure_ascii=False)))
        return doc

    def start(self, dataset_id, item_id, snapshot, provider_id, previous_id, *, language='ja', provider=None):
        return self._append(dataset_id, item_id, snapshot, previous_id, status='pending', confirmed=False,
                            provider_id=provider_id, provider=provider or {}, language=language)

    def confirm(self, dataset_id, item_id, snapshot, body: ReviewSave):
        if snapshot['fingerprint'] != body.fingerprint:
            raise ValueError('review_source_changed')
        return self._append(dataset_id, item_id, snapshot, body.previous_id, status='completed', confirmed=True,
                            annotation=body.annotation.model_dump())

    def _update(self, review_id, change):
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT document FROM lora_dataset_reviews WHERE id=?', (review_id,)).fetchone()
            if row is None:
                raise KeyError('review_not_found')
            doc = json.loads(row[0]); change(doc)
            db.execute('UPDATE lora_dataset_reviews SET document=? WHERE id=?', (json.dumps(doc, ensure_ascii=False), review_id))
        return doc

    def claim(self, review_id):
        def change(doc):
            if doc['status'] != 'pending':
                raise ValueError('review_already_attempted')
            doc['status'] = 'running'
        return self._update(review_id, change)

    def finish(self, review_id, annotation: DatasetAnnotation | None, receipt=None, *, error=None):
        def change(doc):
            if doc['status'] not in {'pending', 'running'}:
                raise ValueError('review_already_finished')
            doc.update(status='completed' if annotation else 'failed', error=error, receipt=receipt,
                       annotation=annotation.model_dump() if annotation else None,
                       ai_annotation=annotation.model_dump() if annotation else None)
        return self._update(review_id, change)

    def cancel(self, review_id):
        def change(doc):
            if doc['status'] in {'pending', 'running'}:
                doc.update(status='cancelled', error='cancelled')
        return self._update(review_id, change)
