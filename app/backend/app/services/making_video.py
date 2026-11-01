"""Record existing preview callbacks without changing image-model execution."""
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
import logging
import sqlite3
from threading import RLock
import time
from uuid import uuid4

import av
from PIL import Image, ImageDraw, ImageOps

from app.schemas.media import MediaArtifact
from app.services.atomic_json import write_json_atomic
from app.services.media_artifacts import file_sha256, finalize_media_artifact

MAX_PREVIEW_FRAMES = 300
MAX_DISK_BYTES = 64 * 1024 * 1024
MAX_INPUT_PIXELS = 16 * 1024 * 1024
MAX_FINAL_IMAGES = 100
ENCODE_SECONDS = 30.0
FPS = 8
SIDE = 640


def small_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if source.width * source.height > MAX_INPUT_PIXELS:
            raise ValueError('making_input_too_large')
        # JPEG draft avoids decoding unnecessarily large previews at full size.
        source.draft('RGB', (SIDE, SIDE))
        image = ImageOps.exif_transpose(source)
        image.thumbnail((SIDE, SIDE), Image.Resampling.LANCZOS)
        return image.convert('RGB')


def encode_frames(directory: Path, frames: list[dict], *, cancelled: Callable[[], bool]) -> dict:
    started = time.monotonic()
    first = small_image(Path(frames[0]['path']))
    width, height = max(160, first.width + first.width % 2), first.height + first.height % 2 + 28
    first.close()
    paths = [directory / 'making.mp4', directory / 'making.gif']
    encoded = 0
    with av.open(str(paths[0]), 'w', options={'movflags': '+faststart'}) as mp4, av.open(str(paths[1]), 'w') as gif:
        streams = [mp4.add_stream('libx264', rate=FPS), gif.add_stream('gif', rate=FPS)]
        for stream, fmt in zip(streams, ['yuv420p', 'rgb8']):
            stream.width, stream.height, stream.pix_fmt = width, height, fmt
            stream.thread_count = 1
        streams[0].options = {'preset': 'veryfast', 'crf': '23'}
        for entry in frames:
            if cancelled() or time.monotonic() - started > ENCODE_SECONDS:
                raise TimeoutError('making_encoding_stopped')
            with small_image(Path(entry['path'])) as image:
                canvas = Image.new('RGB', (width, height), '#101827')
                image.thumbnail((width, height - 28), Image.Resampling.LANCZOS)
                canvas.paste(image, ((width - image.width) // 2, (height - 28 - image.height) // 2))
            label = (f"Final {entry['image'] + 1}" if entry['kind'] == 'final' else
                     f"Batch {entry['batch'] + 1} / Pass {entry['pass']}" +
                     (f" / Step {entry['step']}/{entry['steps']}" if entry['steps'] else ' / Preview'))
            ImageDraw.Draw(canvas).text((6, height - 22), f"{entry['elapsed_seconds']:.1f}s | {label}", fill='white')
            for _ in range(FPS * 2 if entry['kind'] == 'final' else 1):
                for container, stream in zip([mp4, gif], streams):
                    frame = av.VideoFrame.from_image(canvas)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                encoded += 1
            canvas.close()
            if sum(path.stat().st_size for path in paths if path.exists()) > MAX_DISK_BYTES:
                raise ValueError('making_output_too_large')
        for container, stream in zip([mp4, gif], streams):
            for packet in stream.encode():
                container.mux(packet)
    elapsed = time.monotonic() - started
    if elapsed > ENCODE_SECONDS or sum(path.stat().st_size for path in paths) > MAX_DISK_BYTES:
        raise ValueError('making_encoding_limit')
    return {'encoded_frames': encoded, 'encoding_seconds': round(elapsed, 4), 'width': width, 'height': height}


class MakingRecorder:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.started = clock()
        self.directory: Path | None = None
        self.output_root: Path | None = None
        self.frames: list[dict] = []
        self.skipped = 0
        self.bytes = 0
        self.reason = ''
        self.last_time = float('-inf')
        self.last_batch: int | None = None
        self.step, self.steps, self.pass_index = 0, 0, 1
        self.pending_pass_reset = False
        self.lock = RLock()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.discard()

    def begin(self, root: Path, *, expected_images: int):
        self.output_root = Path(root).resolve()
        if expected_images > MAX_FINAL_IMAGES:
            self.reason = 'too_many_images'
            return
        try:
            self.directory = self.output_root / '_making' / f'.capture-{uuid4().hex}'
            self.directory.mkdir(parents=True, exist_ok=False)
        except OSError:
            self.reason = 'recording_failed'

    def wrap(self, callback):
        def progress(*args, **kwargs):
            self.capture(*args, **kwargs)
            if callback is not None:
                return callback(*args, **kwargs)
        return progress

    def capture(self, current: int, total: int, preview_path: str = '', step: int | None = None,
                steps: int | None = None, phase: str = ''):
        with self.lock:
            if self.closed or self.reason or self.directory is None:
                return
            # Completion callbacks carry final files; they are not intermediate previews.
            if phase != 'sampling':
                return
            if self.last_batch != current:
                self.last_batch = current
                self.pass_index, self.step, self.steps = 1, 0, 0
                self.pending_pass_reset = False
            if step is not None:
                if step < self.step:
                    self.pending_pass_reset = True
                self.step, self.steps = max(0, int(step)), max(0, int(steps or 0))
            if not preview_path:
                return
            now = self.clock()
            if len(self.frames) >= MAX_PREVIEW_FRAMES or now - self.last_time < .1:
                self.skipped += 1
                return
            target = self.directory / f'frame-{len(self.frames):04d}.jpg'
            try:
                source = Path(preview_path).resolve(strict=True)
                if not source.is_relative_to(self.output_root):
                    raise ValueError('making_preview_outside_output')
                with small_image(source) as image:
                    image.save(target, format='JPEG', quality=90)
                size = target.stat().st_size
                if self.bytes + size > MAX_DISK_BYTES:
                    target.unlink()
                    self.skipped += 1
                    return
                self.bytes += size
                self.last_time = now
                if self.pending_pass_reset:
                    if self.frames and self.frames[-1]['batch'] == current:
                        self.pass_index += 1
                    self.pending_pass_reset = False
                self.frames.append({'path': str(target), 'kind': 'preview', 'batch': current, 'pass': self.pass_index,
                                    'step': self.step, 'steps': self.steps, 'elapsed_seconds': round(now - self.started, 4)})
            except (OSError, ValueError, Image.DecompressionBombError):
                self.reason = 'recording_failed'

    def discard(self):
        with self.lock:
            self.closed = True
            directory = self.directory
            if directory is None or self.output_root is None:
                return
            # Only the private, flat folder created by this recorder is owned here.
            if directory.parent != self.output_root / '_making' or not directory.name.startswith('.capture-'):
                return
            with suppress(OSError):
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
                directory.rmdir()

    def finish(self, history, source_id: int, source: dict, *, cancelled: Callable[[], bool] = lambda: False) -> dict:
        with self.lock:
            self.closed = True
            settings = source['settings']
            check = settings.get('image_check') or {}
            reason = self.reason
            if source['status'] != 'completed' or cancelled():
                reason = 'generation_incomplete'
            elif check.get('status') in {'reject', 'unknown'} or any(
                item.get('status', item.get('verdict')) in {'reject', 'unknown'} for item in check.get('results', [])
            ):
                reason = 'image_check_not_passed'
            elif not self.frames:
                reason = reason or 'no_previews'
            if reason:
                self.discard()
                return {'status': 'not_saved', 'reason': reason}
            destination = self.output_root / '_making' / f'{source_id}-{uuid4().hex}'
            committed = False
            try:
                previews = len(self.frames)
                final_paths = [Path(p) for p in source['output_paths'] if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
                if not final_paths or len(final_paths) > MAX_FINAL_IMAGES:
                    raise ValueError('making_final_images_missing')
                for index, path in enumerate(final_paths):
                    if not path.resolve().is_relative_to(self.output_root):
                        raise ValueError('making_final_outside_output')
                    self.frames.append({'path': str(path), 'kind': 'final', 'image': index,
                                        'elapsed_seconds': round(self.clock() - self.started, 4),
                                        'source_output_path': str(path), 'pre_metadata_sha256': file_sha256(path)})
                encoding = encode_frames(self.directory, self.frames, cancelled=cancelled)
                receipt = {'version': 1, 'source_generation_id': source_id, 'preview_frames': previews,
                           'final_images': len(final_paths), 'skipped_frames': self.skipped, 'limited': self.skipped > 0,
                           'fps': FPS, 'frames': [{k: v for k, v in frame.items() if k != 'path'} for frame in self.frames],
                           'output_root': str(self.output_root), **encoding}
                media = finalize_media_artifact(generation_id=source_id, source_path=self.directory / 'making.mp4', poster_root=self.directory)
                artifacts = []
                for item in media.artifacts:
                    changes = {'path': str(destination / Path(item.path).name)}
                    if item.kind == 'poster':
                        # A noisy first frame is not a useful Library thumbnail.
                        poster_path = Path(item.path)
                        with small_image(final_paths[-1]) as poster:
                            poster.save(poster_path, format='WEBP', quality=85)
                            changes.update(width=poster.width, height=poster.height)
                        changes.update(size_bytes=poster_path.stat().st_size, sha256=file_sha256(poster_path))
                    artifacts.append(item.model_copy(update=changes))
                gif_path = self.directory / 'making.gif'
                gif_id = f'media:making-gif:{source_id}:{destination.name}'
                artifacts.append(MediaArtifact(artifact_id=gif_id, kind='image', role='derived', path=str(destination / gif_path.name),
                    mime_type='image/gif', width=encoding['width'], height=encoding['height'], size_bytes=gif_path.stat().st_size,
                    sha256=file_sha256(gif_path), parent_artifact_ids=(artifacts[0].artifact_id,)))
                write_json_atomic(self.directory / 'making.asteria.json', {**receipt, 'artifacts': [item.model_dump(mode='json') for item in artifacts]})
                for frame in self.frames[:previews]:
                    Path(frame['path']).unlink()
                self.directory.replace(destination)
                output_paths = [str(destination / Path(path).name) for path in media.output_paths] + [str(destination / gif_path.name), str(destination / 'making.asteria.json')]
                generation_id = history.add(status='completed', profile=source['profile'], prompt=source['prompt'],
                    expanded_prompt=source['expanded_prompt'], negative_prompt=source['negative_prompt'],
                    settings={'output_root': str(self.output_root), 'making_source_id': source_id, 'making_recording': receipt},
                    model_slots=source['model_slots'], loras=source['loras'], embeddings=source['embeddings'],
                    output_paths=output_paths, media_kind='video', artifacts=artifacts, generated_tags=['making'])
                committed = True
                return {'status': 'completed', 'generation_id': generation_id, 'video_artifact_id': artifacts[0].artifact_id,
                        'gif_artifact_id': gif_id, **receipt}
            except (OSError, ValueError, RuntimeError, sqlite3.Error, av.FFmpegError):
                logging.getLogger(__name__).warning('Making video could not be finalized', exc_info=True)
                return {'status': 'failed', 'reason': 'encoding_failed'}
            finally:
                self.discard()
                if not committed and destination.is_dir():
                    # Roll back only this recorder's newly moved flat directory.
                    with suppress(OSError):
                        for path in destination.iterdir():
                            path.unlink()
                        destination.rmdir()
