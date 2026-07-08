"""Tests for :mod:`moodengine.io_audio` — decode, segment, and discover audio files."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from assertpy import assert_that

from moodengine.config import default_config
from moodengine.exceptions import AudioDecodeError
from moodengine.io_audio import (
    _capped_indices,
    discover_audio_files,
    load_audio,
    segment_waveform,
)


def _write_sine(path: Path, sr: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    """Write a mono sine WAV to ``path`` and return the float32 samples."""
    t = np.linspace(0.0, seconds, int(round(sr * seconds)), endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wave, sr)
    return wave


def test_load_audio_mono_float32_target_sr(tmp_path) -> None:
    """Decoding resamples to ``target_sr`` and yields a 1-D float32 mono array."""
    cfg = default_config()
    src_sr = 22_050
    wav_path = tmp_path / "tone.wav"
    _write_sine(wav_path, src_sr, seconds=1.0)

    target_sr = 16_000
    out = load_audio(wav_path, target_sr=target_sr, config=cfg)

    assert_that(out.dtype).is_equal_to(np.float32)
    assert_that(out.ndim).is_equal_to(1)
    # Length scales with the resample ratio (allow a few samples of slack).
    expected = int(round(1.0 * target_sr))
    assert_that(out.shape[0]).is_close_to(expected, tolerance=64)


def test_load_audio_preserves_native_sr_length(tmp_path) -> None:
    """When ``target_sr`` matches the file SR the sample count is preserved."""
    cfg = default_config()
    sr = 16_000
    wav_path = tmp_path / "native.wav"
    wave = _write_sine(wav_path, sr, seconds=2.0)

    out = load_audio(wav_path, target_sr=sr, config=cfg)
    assert_that(out.shape[0]).is_equal_to(wave.shape[0])


def test_load_audio_corrupt_file_raises_decode_error(tmp_path) -> None:
    """A file that EXISTS but cannot be decoded raises AudioDecodeError naming the
    path — and stays catchable as the RuntimeError it used to be."""
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not really audio")

    with pytest.raises(AudioDecodeError, match="broken.wav") as excinfo:
        load_audio(bad, target_sr=16_000)

    assert_that(excinfo.value).is_instance_of(RuntimeError)  # pre-hierarchy catchers keep working


def test_load_audio_missing_file_raises_file_not_found(tmp_path) -> None:
    """A missing path is NOT a decode failure: the caller should re-scan the
    library, not flag a damaged file — stdlib FileNotFoundError says exactly that."""
    with pytest.raises(FileNotFoundError, match="nope.wav"):
        load_audio(tmp_path / "nope.wav", target_sr=16_000)


def test_load_audio_config_parameter_is_optional(tmp_path) -> None:
    """The (deprecated, unused) ``config`` parameter can be omitted; legacy
    3-argument call sites keep working identically."""
    sr = 16_000
    wav_path = tmp_path / "compat.wav"
    _write_sine(wav_path, sr, seconds=0.5)

    without_config = load_audio(wav_path, target_sr=sr)
    with_config = load_audio(wav_path, target_sr=sr, config=default_config())

    np.testing.assert_array_equal(without_config, with_config)


def test_segment_waveform_exact_count_no_overlap() -> None:
    """A clean multiple of the segment length yields exactly that many windows."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.0,
        min_segment_seconds=0.5,
        max_segments_per_track=0,
    )
    wav = np.ones(5 * sr, dtype=np.float32)  # 5 full seconds -> 5 windows
    segs = segment_waveform(wav, sr, cfg)
    assert_that(segs).is_length(5)
    for s in segs:
        assert_that(s.shape[0]).is_equal_to(sr)
        assert_that(s.dtype).is_equal_to(np.float32)


def test_segment_waveform_drops_short_trailing() -> None:
    """A short trailing partial (< min) is dropped, a long-enough one is kept."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.0,
        min_segment_seconds=0.5,
        max_segments_per_track=0,
    )
    # 2.2s: two full windows + a 0.2s tail (< 0.5s) -> dropped.
    short_tail = np.ones(int(2.2 * sr), dtype=np.float32)
    assert_that(segment_waveform(short_tail, sr, cfg)).is_length(2)

    # 2.6s: two full windows + a 0.6s tail (>= 0.5s) -> kept.
    long_tail = np.ones(int(2.6 * sr), dtype=np.float32)
    segs = segment_waveform(long_tail, sr, cfg)
    assert_that(segs).is_length(3)
    assert_that(segs[-1].shape[0]).is_equal_to(int(0.6 * sr))


def test_segment_waveform_keeps_whole_short_track() -> None:
    """A track shorter than ``min_segment_seconds`` is returned as one segment."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.0,
        min_segment_seconds=0.5,
        max_segments_per_track=0,
    )
    tiny = np.ones(int(0.3 * sr), dtype=np.float32)  # 0.3s < 0.5s min
    segs = segment_waveform(tiny, sr, cfg)
    assert_that(segs).is_length(1)
    assert_that(segs[0].shape[0]).is_equal_to(tiny.shape[0])


def test_segment_waveform_empty_returns_empty() -> None:
    """An empty waveform yields an empty list, not a crash."""
    cfg = default_config()
    assert_that(segment_waveform(np.zeros(0, dtype=np.float32), 16_000, cfg)).is_equal_to([])


def test_segment_waveform_with_overlap_increases_count() -> None:
    """Overlap shortens the stride, producing more (still full-length) windows."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.5,  # step = 0.5s
        min_segment_seconds=0.5,
        max_segments_per_track=0,
    )
    wav = np.ones(3 * sr, dtype=np.float32)
    segs = segment_waveform(wav, sr, cfg)
    # Full windows start at 0, 500, 1000, 1500, 2000 (start 2500 only has 500 < seg_len);
    # 2500..3500 tail is 500 samples >= min -> kept.
    assert_that(segs).is_length(6)
    for s in segs[:5]:
        assert_that(s.shape[0]).is_equal_to(sr)


def _ramp_windows(sr: int, n_windows: int) -> np.ndarray:
    """A waveform whose k-th one-second window is the constant ``k`` (kept windows are identifiable)."""
    return np.concatenate([np.full(sr, float(k), dtype=np.float32) for k in range(n_windows)])


def test_capped_indices_no_cap_returns_all() -> None:
    """No cap needed (count <= cap, or cap <= 0) -> every index is kept, unchanged."""
    assert_that(_capped_indices(5, 10, "uniform")).is_equal_to([0, 1, 2, 3, 4])
    assert_that(_capped_indices(5, 0, "head")).is_equal_to([0, 1, 2, 3, 4])


def test_capped_indices_head_keeps_first_uniform_spreads() -> None:
    """head -> the first ``cap``; uniform -> ``cap`` windows spread across the whole track."""
    assert_that(_capped_indices(10, 3, "head")).is_equal_to([0, 1, 2])
    assert_that(_capped_indices(10, 3, "uniform")).is_equal_to([0, 4, 9])  # linspace(0,9,3) rounded


def test_segment_waveform_cap_head_keeps_first_n() -> None:
    """``segment_selection='head'`` caps to the first N windows (the pre-1.0 behavior)."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.0,
        min_segment_seconds=0.5,
        max_segments_per_track=3,
        segment_selection="head",
    )
    segs = segment_waveform(_ramp_windows(sr, 10), sr, cfg)

    assert_that(segs).is_length(3)
    assert_that([float(s[0]) for s in segs]).is_equal_to([0.0, 1.0, 2.0])  # first three windows


def test_segment_waveform_cap_uniform_spreads_and_reaches_the_end() -> None:
    """``segment_selection='uniform'`` (the default) spreads the kept windows across the track, so the
    outro is represented — unlike head, which never sees past the first N windows."""
    sr = 1_000
    cfg = dataclasses.replace(
        default_config(),
        segment_seconds=1.0,
        overlap_seconds=0.0,
        min_segment_seconds=0.5,
        max_segments_per_track=3,
        segment_selection="uniform",
    )
    segs = segment_waveform(_ramp_windows(sr, 10), sr, cfg)

    assert_that(segs).is_length(3)
    assert_that([float(s[0]) for s in segs]).is_equal_to([0.0, 4.0, 9.0])  # start, middle, end


def test_discover_audio_files_recursive_sorted_filtered(tmp_path) -> None:
    """Discovery is recursive, extension-filtered (case-insensitive) and sorted."""
    cfg = default_config()
    (tmp_path / "sub").mkdir()
    paths = [
        tmp_path / "b.wav",
        tmp_path / "a.MP3",  # upper-case suffix still matches
        tmp_path / "sub" / "c.flac",
        tmp_path / "notes.txt",  # excluded
        tmp_path / "sub" / "cover.png",  # excluded
    ]
    for p in paths:
        p.write_bytes(b"x")

    found = discover_audio_files(tmp_path, cfg)
    assert_that(all(isinstance(p, Path) for p in found)).is_true()
    names = [p.name for p in found]
    assert_that(names).is_equal_to(sorted(names))
    assert_that(set(names)).is_equal_to({"a.MP3", "b.wav", "c.flac"})


def test_discover_audio_files_missing_dir_returns_empty(tmp_path) -> None:
    """A non-existent directory yields an empty list."""
    cfg = default_config()
    assert_that(discover_audio_files(tmp_path / "nope", cfg)).is_equal_to([])
