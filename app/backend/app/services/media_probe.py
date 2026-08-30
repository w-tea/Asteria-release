from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import av
from PIL import Image

from app.schemas.media import MediaKind


MAX_DECODE_PACKETS = 512
MAX_DECODE_PIXELS = 64 * 1024 * 1024
POSTER_EDGE = 768
_OPEN_OPTIONS = {
    "probesize": str(5 * 1024 * 1024),
    "analyzeduration": "5000000",
}
_MEDIA_TYPES: dict[str, tuple[MediaKind, str]] = {
    ".mp4": ("video", "video/mp4"),
    ".webm": ("video", "video/webm"),
    ".wav": ("audio", "audio/wav"),
    ".flac": ("audio", "audio/flac"),
    ".mp3": ("audio", "audio/mpeg"),
    ".opus": ("audio", "audio/ogg"),
}


class MediaProbeError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MediaProbeResult:
    kind: MediaKind
    mime_type: str
    container_format: str
    size_bytes: int
    width: int | None
    height: int | None
    duration_ms: int
    fps_numerator: int | None
    fps_denominator: int | None
    codec: str
    has_audio: bool
    sample_rate: int | None
    channels: int | None
    bit_depth: int | None
    rotation_degrees: int
    poster_path: Path | None
    video_time_base_numerator: int | None = None
    video_time_base_denominator: int | None = None
    audio_time_base_numerator: int | None = None
    audio_time_base_denominator: int | None = None


def probe_media(
    source_path: Path,
    *,
    poster_path: Path | None = None,
    max_decode_packets: int = MAX_DECODE_PACKETS,
) -> MediaProbeResult:
    source = Path(source_path)
    kind_and_mime = _MEDIA_TYPES.get(source.suffix.lower())
    if kind_and_mime is None:
        raise MediaProbeError("media_probe_unsupported")
    kind, mime_type = kind_and_mime
    if poster_path is not None and kind != "video":
        raise MediaProbeError("media_poster_not_supported")
    if kind == "video" and max_decode_packets < 1:
        raise MediaProbeError("media_probe_resource_limit")
    try:
        resolved = source.resolve(strict=True)
        details = resolved.stat()
    except (OSError, RuntimeError) as error:
        raise MediaProbeError("media_probe_missing") from error
    if not resolved.is_file() or details.st_size <= 0:
        raise MediaProbeError("media_probe_missing")

    try:
        with av.open(str(resolved), mode="r", options=_OPEN_OPTIONS) as container:
            video_stream = next(iter(container.streams.video), None)
            audio_stream = next(iter(container.streams.audio), None)
            video_time_base, audio_time_base = _validate_stream_time_bases(
                video_stream,
                audio_stream,
            )
            if kind == "video":
                if video_stream is None:
                    raise MediaProbeError("media_video_stream_missing")
                width = int(video_stream.codec_context.width or 0)
                height = int(video_stream.codec_context.height or 0)
                if width <= 0 or height <= 0 or width * height > MAX_DECODE_PIXELS:
                    raise MediaProbeError("media_probe_resource_limit")
                frame = _first_video_frame(
                    container,
                    video_stream,
                    max_decode_packets=max_decode_packets,
                )
                rotation = _frame_rotation(frame, video_stream)
                image = frame.to_image()
                if rotation:
                    image = image.rotate(rotation, expand=True)
                display_width, display_height = image.size
                written_poster = (
                    _write_poster_atomic(image, Path(poster_path))
                    if poster_path is not None
                    else None
                )
                rate = _video_rate(video_stream)
                codec = str(video_stream.codec_context.name or "")
            else:
                if audio_stream is None:
                    raise MediaProbeError("media_audio_stream_missing")
                display_width = None
                display_height = None
                rotation = 0
                written_poster = None
                rate = None
                codec = str(audio_stream.codec_context.name or "")

            sample_rate, channels, bit_depth = _audio_metadata(audio_stream)
            duration_ms = _duration_ms(container, video_stream or audio_stream)
            return MediaProbeResult(
                kind=kind,
                mime_type=mime_type,
                container_format=str(container.format.name or ""),
                size_bytes=int(details.st_size),
                width=display_width,
                height=display_height,
                duration_ms=duration_ms,
                fps_numerator=rate.numerator if rate is not None else None,
                fps_denominator=rate.denominator if rate is not None else None,
                codec=codec,
                has_audio=audio_stream is not None,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                rotation_degrees=rotation,
                poster_path=written_poster,
                video_time_base_numerator=(
                    video_time_base.numerator if video_time_base is not None else None
                ),
                video_time_base_denominator=(
                    video_time_base.denominator if video_time_base is not None else None
                ),
                audio_time_base_numerator=(
                    audio_time_base.numerator if audio_time_base is not None else None
                ),
                audio_time_base_denominator=(
                    audio_time_base.denominator if audio_time_base is not None else None
                ),
            )
    except MediaProbeError:
        raise
    except (av.FFmpegError, EOFError, OSError, RuntimeError, ValueError) as error:
        raise MediaProbeError("media_probe_invalid") from error


def _first_video_frame(container, video_stream, *, max_decode_packets: int):
    packet_count = 0
    try:
        for packet in container.demux(video_stream):
            packet_count += 1
            if packet_count > max_decode_packets:
                raise MediaProbeError("media_probe_resource_limit")
            try:
                frames = packet.decode()
            except av.FFmpegError:
                continue
            for frame in frames:
                if frame.width * frame.height > MAX_DECODE_PIXELS:
                    raise MediaProbeError("media_probe_resource_limit")
                return frame
    except MediaProbeError:
        raise
    except av.FFmpegError as error:
        raise MediaProbeError("media_probe_invalid") from error
    raise MediaProbeError("media_video_no_decodable_frame")


def _frame_rotation(frame, video_stream) -> int:
    raw = getattr(frame, "rotation", None)
    if raw in {None, 0}:
        raw = video_stream.metadata.get("rotate", 0)
    try:
        normalized = int(round(float(raw))) % 360
    except (TypeError, ValueError, OverflowError):
        return 0
    if normalized > 180:
        normalized -= 360
    return normalized if normalized in {-180, -90, 0, 90, 180} else 0


def _video_rate(video_stream) -> Fraction | None:
    for value in (
        getattr(video_stream, "average_rate", None),
        getattr(video_stream, "guessed_rate", None),
        getattr(video_stream, "base_rate", None),
    ):
        if value is None:
            continue
        try:
            rate = Fraction(value)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def _validate_stream_time_bases(video_stream, audio_stream) -> tuple[Fraction | None, Fraction | None]:
    values: list[Fraction | None] = []
    for stream in (video_stream, audio_stream):
        if stream is None:
            values.append(None)
            continue
        try:
            time_base = Fraction(stream.time_base)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
            raise MediaProbeError("media_stream_time_base_invalid") from error
        if time_base <= 0:
            raise MediaProbeError("media_stream_time_base_invalid")
        values.append(time_base)
    return values[0], values[1]


def _duration_ms(container, primary_stream) -> int:
    duration = getattr(container, "duration", None)
    if duration is not None:
        try:
            return max(0, int(round(int(duration) / 1000)))
        except (TypeError, ValueError, OverflowError):
            pass
    stream_duration = getattr(primary_stream, "duration", None)
    time_base = getattr(primary_stream, "time_base", None)
    if stream_duration is None or time_base is None:
        return 0
    try:
        return max(0, int(round(float(stream_duration * time_base) * 1000)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _audio_metadata(audio_stream) -> tuple[int | None, int | None, int | None]:
    if audio_stream is None:
        return None, None, None
    context = audio_stream.codec_context
    sample_rate = _positive_int(getattr(context, "sample_rate", None))
    layout = getattr(context, "layout", None)
    channels = _positive_int(getattr(layout, "nb_channels", None))
    if channels is None:
        channels = _positive_int(getattr(context, "channels", None))
    audio_format = getattr(context, "format", None)
    bit_depth = _positive_int(getattr(audio_format, "bits", None))
    if bit_depth is None:
        bit_depth = _positive_int(getattr(context, "bits_per_raw_sample", None))
    return sample_rate, channels, bit_depth


def _positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _write_poster_atomic(image: Image.Image, target: Path) -> Path:
    if target.suffix.lower() != ".webp":
        raise MediaProbeError("media_poster_invalid_path")
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        output = image.convert("RGB")
        output.thumbnail((POSTER_EDGE, POSTER_EDGE), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        output.save(buffer, format="WEBP", quality=84, method=6)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, target)
        return target
    except MediaProbeError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise MediaProbeError("media_poster_write_failed") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
