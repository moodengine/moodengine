"""Unit tests for moodengine.signals — tempo/key extraction + Camelot wheel. Torch-free, deterministic.

Tempo is checked on a synthetic click track (known BPM, ±2% after octave-folding); key on a synthesized
C-major triad (expects Camelot 8B or a wheel neighbour); the Camelot table is pinned exhaustively over
all 24 keys; determinism is asserted by running the extractor twice. Failure paths (signals too short
to analyse) must carry ``None`` — never a sentinel like ``0.0`` BPM or an empty Camelot string."""

from __future__ import annotations

import re
from collections import Counter
from string import ascii_uppercase

import librosa
import numpy as np
import pytest
from assertpy import assert_that

from moodengine.signals import (
    camelot_neighbors,
    estimate_key,
    estimate_tempo,
    extract_signals,
    segment_structure,
    to_camelot,
)

_SR = 22_050

# Exhaustive expected Camelot table (pitch-class 0..11), from the circle of fifths.
_MAJOR_EXPECTED = {
    0: "8B",
    1: "3B",
    2: "10B",
    3: "5B",
    4: "12B",
    5: "7B",
    6: "2B",
    7: "9B",
    8: "4B",
    9: "11B",
    10: "6B",
    11: "1B",
}
_MINOR_EXPECTED = {
    0: "5A",
    1: "12A",
    2: "7A",
    3: "2A",
    4: "9A",
    5: "4A",
    6: "11A",
    7: "6A",
    8: "1A",
    9: "8A",
    10: "3A",
    11: "10A",
}


def _tone(freq: float, sr: int, dur: float) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# --------------------------------------------------------------------------- #
# Camelot wheel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pc", range(12))
def test_to_camelot_major_table(pc):
    assert_that(to_camelot(pc, "major")).is_equal_to(_MAJOR_EXPECTED[pc])


@pytest.mark.parametrize("pc", range(12))
def test_to_camelot_minor_table(pc):
    assert_that(to_camelot(pc, "minor")).is_equal_to(_MINOR_EXPECTED[pc])


def test_relative_major_minor_share_number():
    # C major (8B) and its relative A minor (8A) share the number; letter distinguishes mode.
    assert_that(to_camelot(0, "major")).is_equal_to("8B")
    assert_that(to_camelot(9, "minor")).is_equal_to("8A")


def test_camelot_neighbors_are_the_three_classic_moves():
    assert_that(camelot_neighbors("8A")).is_equal_to(["7A", "9A", "8B"])
    assert_that(camelot_neighbors("1A")).is_equal_to(["12A", "2A", "1B"])  # wrap down 1 -> 12
    assert_that(camelot_neighbors("12B")).is_equal_to(["11B", "1B", "12A"])  # wrap up 12 -> 1


@pytest.mark.parametrize("num", range(1, 13))
@pytest.mark.parametrize("letter", ["A", "B"])
def test_camelot_neighbors_wheel_invariants(num, letter):
    nbrs = camelot_neighbors(f"{num}{letter}")
    assert_that(nbrs).is_length(3)
    # relative switch keeps the number, flips the letter
    assert_that(nbrs[2]).is_equal_to(f"{num}{'B' if letter == 'A' else 'A'}")
    # the two wheel steps keep the letter and are adjacent mod 12
    assert_that(all(n.endswith(letter) for n in nbrs[:2])).is_true()


# --------------------------------------------------------------------------- #
# Tempo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_estimate_tempo_recovers_click_bpm_within_2pct(bpm):
    dur = 12.0
    times = np.arange(0.0, dur, 60.0 / bpm)
    y = librosa.clicks(times=times, sr=_SR, click_duration=0.05, length=int(dur * _SR))

    est = estimate_tempo(y, _SR)

    assert_that(est.bpm).is_not_none()  # success never carries None
    assert_that(bool(np.isfinite(est.bpm))).is_true()
    assert_that(est.bpm).is_greater_than_or_equal_to(70.0)  # folded into the musical range
    assert_that(est.bpm).is_less_than(180.0)
    assert_that(abs(est.bpm - bpm) / bpm).is_less_than_or_equal_to(
        0.02
    )  # within 2% after octave-folding
    assert_that(est.confidence).is_between(0.0, 1.0)
    assert_that(len(est.beat_times)).is_greater_than(0)
    assert_that(est.stability).is_true()  # a metronome is maximally regular


def test_estimate_tempo_too_short_signal_yields_none_bpm():
    est = estimate_tempo(np.zeros(16, dtype=np.float32), _SR)

    assert_that(est.bpm).is_none()  # absence of measurement, not a 0.0 sentinel
    assert_that(est.confidence).is_equal_to(0.0)
    assert_that(est.beat_times).is_equal_to([])
    assert_that(est.stability).is_false()


# --------------------------------------------------------------------------- #
# Key
# --------------------------------------------------------------------------- #
def test_estimate_key_c_major_triad_is_8b_or_neighbour():
    dur = 6.0
    y = sum(_tone(f, _SR, dur) for f in (261.63, 329.63, 392.00)).astype(np.float32)  # C4 E4 G4

    est = estimate_key(y, _SR)

    allowed = {"8B", *camelot_neighbors("8B")}  # 8B or a harmonically-adjacent code
    assert_that(est.camelot).described_as(f"got {est.camelot} ({est.tonic} {est.mode})").is_in(
        *allowed
    )
    assert_that(est.mode).is_in("major", "minor")
    assert_that(est.confidence).is_between(0.0, 1.0)
    assert_that(est.alternatives).is_length(3)


def test_estimate_key_tonal_clip_camelot_matches_wheel_pattern():
    dur = 6.0
    y = sum(_tone(f, _SR, dur) for f in (261.63, 329.63, 392.00)).astype(np.float32)  # C4 E4 G4

    est = estimate_key(y, _SR)

    assert_that(est.camelot).is_not_none()
    assert_that(est.tonic).is_not_none()
    assert_that(est.mode).is_not_none()
    assert_that(
        re.fullmatch(r"([1-9]|1[0-2])[AB]", est.camelot)
    ).is_not_none()  # wheel code: number 1-12 + mode letter


def test_estimate_key_too_short_signal_yields_none_fields():
    est = estimate_key(np.zeros(16, dtype=np.float32), _SR)

    assert_that(est.camelot).is_none()
    assert_that(est.tonic).is_none()
    assert_that(est.mode).is_none()
    assert_that(est.confidence).is_equal_to(0.0)
    assert_that(est.alternatives).is_equal_to([])


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_extract_signals_is_deterministic():
    dur = 8.0
    times = np.arange(0.0, dur, 60.0 / 128)
    y = librosa.clicks(times=times, sr=_SR, length=int(dur * _SR))
    a = extract_signals(y, _SR)
    b = extract_signals(y, _SR)
    assert_that(a.tempo).is_equal_to(
        b.tempo
    )  # frozen dataclasses compare by value -> byte-identical
    assert_that(a.key).is_equal_to(b.key)


# --------------------------------------------------------------------------- #
# Structural segmentation
# --------------------------------------------------------------------------- #
_FORBIDDEN_ROLE_LABELS = {
    "chorus",
    "verse",
    "refrain",
    "intro",
    "drop",
    "couplet",
    "bridge",
    "outro",
}


def _abab_signal(sr: int = _SR, block: float = 4.0, pattern=(0, 1, 0, 1, 0, 1)) -> np.ndarray:
    """A synthetic A-B-A-B track: two alternating timbres (220/440 Hz + a harmonic) over a steady
    120-BPM click grid, so beat tracking finds a grid AND chroma/MFCC differ between blocks. The true
    section boundaries fall at every ``block`` seconds."""
    t = np.arange(int(block * sr)) / sr

    def timbre(which: int) -> np.ndarray:
        f0 = 220.0 if which == 0 else 440.0
        return (0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)).astype(
            np.float32
        )

    tones = np.concatenate([timbre(p) for p in pattern]).astype(np.float32)
    click_times = np.arange(0.0, len(tones) / sr, 0.5)  # 120 BPM
    clicks = librosa.clicks(times=click_times, sr=sr, length=len(tones)).astype(np.float32)
    return (tones + 0.4 * clicks).astype(np.float32)


def test_segment_structure_abab_boundaries_groups_and_hook():
    y = _abab_signal()
    st = segment_structure(y, _SR, kernel_size=16)  # small kernel suits a short synthetic
    detected = [s.start for s in st.segments[1:]]  # internal boundaries (first section starts at 0)
    # every true transition (4,8,12,16,20 s) has a detected boundary within ~1 beat (0.5 s grid).
    for tb in (4.0, 8.0, 12.0, 16.0, 20.0):
        assert_that(any(abs(tb - d) <= 0.6 for d in detected)).described_as(
            f"no boundary near {tb}s in {detected}"
        ).is_true()
    assert_that(len({s.group for s in st.segments})).is_greater_than_or_equal_to(
        2
    )  # >= 2 distinct groups
    # hook_group is a GENUINELY repeated group (count > 1) — a measured fact, not a role.
    assert_that(st.hook_group).is_not_none()
    assert_that(st.hook_start).is_not_none()
    counts = Counter(s.group for s in st.segments)
    assert_that(counts[st.hook_group]).is_greater_than(1)


def test_segment_structure_labels_are_honest_letters_never_roles():
    st = segment_structure(_abab_signal(), _SR, kernel_size=16)
    for seg in st.segments:
        assert_that(seg.label).is_length(1)  # 'A'..'Z' only
        assert_that(ascii_uppercase).contains(seg.label)
        assert_that(_FORBIDDEN_ROLE_LABELS).does_not_contain(
            seg.label.lower()
        )  # never a musical role


def test_segment_structure_is_deterministic():
    y = _abab_signal()
    a = segment_structure(y, _SR, kernel_size=16)
    b = segment_structure(y, _SR, kernel_size=16)
    assert_that(a).is_equal_to(b)  # frozen dataclasses -> value equality (byte-identical structure)


def test_segment_structure_empty_signal_is_empty():
    st = segment_structure(np.zeros(0, dtype=np.float32), _SR)
    assert_that(st.segments).is_equal_to([])
    assert_that(st.n_boundaries).is_equal_to(0)
    assert_that(st.hook_group).is_none()
    assert_that(st.hook_start).is_none()


def test_segment_structure_too_short_is_single_honest_section():
    st = segment_structure(np.zeros(int(0.2 * _SR), dtype=np.float32), _SR)  # no reliable beat grid
    assert_that([s.label for s in st.segments]).is_equal_to(
        ["A"]
    )  # a single honest section, no fabricated boundary
    assert_that(st.n_boundaries).is_equal_to(0)
    assert_that(st.hook_group).is_none()


def test_signals_module_is_torch_free():
    # Importing moodengine.signals AND running the full segmentation (which lazy-imports sklearn inside
    # _section_groups) must not pull torch. Drive a real A-B-A-B buffer so the clustering path actually
    # executes — an empty array would short-circuit at the duration<=0 guard before sklearn is imported.
    import subprocess
    import sys

    code = (
        "import sys, numpy as np, librosa, moodengine.signals as s; "
        "sr=22050; t=np.arange(int(4.0*sr))/sr; "
        "tone=lambda f: (0.5*np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*2*f*t)).astype('float32'); "
        "y=np.concatenate([tone(220.0),tone(440.0),tone(220.0),tone(440.0)]).astype('float32'); "
        "y=y+0.4*librosa.clicks(times=np.arange(0.0,len(y)/sr,0.5),sr=sr,length=len(y)).astype('float32'); "
        "st=s.segment_structure(y, sr, kernel_size=16); "
        "assert 'sklearn.cluster' in sys.modules, 'clustering path not exercised'; "
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')]; "
        "sys.exit('torch loaded: '+repr(bad)) if bad else None"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as((r.stdout + r.stderr).strip()).is_equal_to(0)


def test_estimate_tempo_silent_waveform_yields_none_bpm():
    # Long enough to analyse (>= one hop), but silence carries no beat to track: this
    # exercises the fold-path failure, not the too-short early return.
    y = np.zeros(22050, dtype=np.float32)

    est = estimate_tempo(y, 22050)

    assert_that(est.bpm).is_none()
    assert_that(est.confidence).is_equal_to(0.0)
