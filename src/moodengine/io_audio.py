"""Audio I/O: decode, segment, and discover audio files.

Pure, torch-free utilities built on librosa/soundfile so the lightweight
pipeline stages can run without the deep-learning stack. Decoding always
yields mono float32 at a requested sample rate; segmentation is deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np

from moodengine._typing import SegmentSelection
from moodengine.config import Config
from moodengine.exceptions import AudioDecodeError

PathLike = Union[str, Path]


def _capped_indices(count: int, cap: int, selection: SegmentSelection) -> list[int]:
    """Indices of the windows kept when a ``count``-window track exceeds ``cap`` windows.

    ``"head"`` keeps the first ``cap`` (the pre-1.0 behavior); ``"uniform"`` keeps up to ``cap``
    windows spread across the whole track via :func:`numpy.linspace`, so a long track's embedding
    reflects its entirety rather than only its (systematically unrepresentative) intro. Returns
    ``list(range(count))`` unchanged when ``cap <= 0`` or ``count <= cap``. Deterministic. Shared by
    :func:`segment_waveform` and :func:`moodengine.mood_arc.segment_bounds` so the pooled windows
    and the displayed time bounds always select the SAME windows.
    """
    if cap <= 0 or count <= cap:
        return list(range(count))
    if selection == "uniform":
        idx = np.linspace(0, count - 1, cap).round().astype(int)
        return sorted({int(i) for i in idx})
    return list(range(cap))


def load_audio(path: PathLike, target_sr: int, config: Config | None = None) -> np.ndarray:
    """Decode any supported file to MONO float32 at ``target_sr``.

    Uses ``librosa.load(path, sr=target_sr, mono=True)`` and returns a 1-D
    ``np.float32`` array. A missing path raises ``FileNotFoundError`` and a
    file that exists but cannot be decoded raises
    :class:`~moodengine.exceptions.AudioDecodeError` — two different caller
    reactions (re-scan the library vs flag the file as damaged), so they are
    two different types. ``config`` is unused and kept only so existing
    3-argument callers keep working; it is deprecated and will be removed in a
    future minor release.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    try:
        waveform, _ = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as exc:  # noqa: BLE001 - re-raise with context
        raise AudioDecodeError(f"Failed to decode audio file: {path}") from exc
    return np.ascontiguousarray(waveform, dtype=np.float32)


def segment_waveform(waveform: np.ndarray, sr: int, config: Config) -> list[np.ndarray]:
    """Split a 1-D waveform into fixed-length windows.

    Windows are ``config.segment_seconds`` long with ``config.overlap_seconds``
    overlap. A trailing partial segment is kept only if its duration is
    ``>= config.min_segment_seconds``, otherwise dropped -- except that a whole
    track shorter than ``min_segment_seconds`` still returns its single short
    segment, so a non-empty waveform never yields an empty list. When
    ``config.max_segments_per_track > 0`` and there are more windows than the cap,
    ``config.segment_selection`` chooses which survive: ``"uniform"`` (default)
    spreads them across the whole track, ``"head"`` keeps the first N. Each
    segment is a contiguous float32 array.
    """
    wav = np.ascontiguousarray(np.asarray(waveform).reshape(-1), dtype=np.float32)
    n = wav.shape[0]
    if n == 0:
        return []

    seg_len = max(1, int(round(config.segment_seconds * sr)))
    overlap_len = max(0, int(round(config.overlap_seconds * sr)))
    overlap_len = min(overlap_len, seg_len - 1)  # guarantee forward progress
    step = seg_len - overlap_len
    min_len = int(round(config.min_segment_seconds * sr))

    # Whole track shorter than the minimum: never lose it.
    if n < min_len:
        return [wav.copy()]

    segments: list[np.ndarray] = []
    start = 0
    while start < n:
        end = start + seg_len
        chunk = wav[start:end]
        if chunk.shape[0] == seg_len:
            segments.append(np.ascontiguousarray(chunk, dtype=np.float32))
        else:
            # Trailing partial segment: keep only if long enough.
            if chunk.shape[0] >= min_len:
                segments.append(np.ascontiguousarray(chunk, dtype=np.float32))
            break
        start += step

    # Safety net: a non-empty waveform must yield at least one segment.
    if not segments:
        segments.append(wav.copy())

    cap = config.max_segments_per_track
    keep = _capped_indices(len(segments), cap, config.segment_selection)
    if len(keep) != len(segments):
        segments = [segments[i] for i in keep]

    return segments


def discover_audio_files(
    directory: PathLike, config: Config, *, recursive: bool = True
) -> list[Path]:
    """List audio files under ``directory``, sorted by path.

    A file is included when its lowercased suffix is in
    ``config.audio_extensions``. When ``recursive`` is ``True`` (the default)
    the entire tree under ``directory`` is walked; when ``False`` only files
    directly in ``directory`` are considered (its sub-directories are ignored).
    Returns a sorted ``list[pathlib.Path]``.
    """
    root = Path(directory)
    if not root.exists():
        return []
    exts = {e.lower() for e in config.audio_extensions}
    entries = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in entries if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)
