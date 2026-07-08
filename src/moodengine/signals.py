"""Musical signals: tempo/beats + key/tonality — the only non-mood dimensions the project allows.

Pure numpy + librosa, torch-free and deterministic (no unseeded RNG). Given an already-decoded mono
waveform, :func:`extract_signals` returns a :class:`SignalSet`:

  * **Tempo** via ``librosa.beat.beat_track`` (onset-envelope + dynamic-programming beat tracking):
    a global BPM (octave-folded into a musical range), a real confidence (onset-autocorrelation peak
    at the beat lag), a stability flag (regularity of the inter-beat interval) and the beat grid.
  * **Key** via **Krumhansl-Schmuckler**: the mean chromagram correlated against the 24 rotated
    major/minor key profiles → tonic + mode → **Camelot** code (``8B`` = C major, ``8A`` = A minor).

The **Camelot wheel** helpers (:func:`to_camelot`, :func:`camelot_neighbors`) are the harmonic-engine
primitive consumed by the harmonic/tempo-aware "radio" ranking in :mod:`moodengine.search`.

Every number is a real measurement. When an estimate cannot be made at all, it says so with
``None`` (``TempoEstimate.bpm``; ``KeyEstimate.camelot``/``tonic``/``mode`` on inputs too short
to analyse) — never a magic placeholder such as ``0.0`` or ``""``. A key estimate on analysable
but tonally empty audio (e.g. silence) still names the best-correlated key: there its
``confidence`` of ``0.0`` is what marks the readout meaningless. Two calls on the same waveform
return byte-identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

_HOP_LENGTH: int = (
    512  # librosa default for onset_strength / beat_track — kept explicit for the lag math
)
_CHROMA_SR: int = 22_050  # key estimation resamples here (chroma_cqt's natural rate)
_TEMPO_LO: float = 70.0
_TEMPO_HI: float = 180.0
_STABILITY_CV: float = 0.10  # inter-beat-interval coefficient-of-variation threshold for "stable"

_PITCH_CLASSES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles (major / natural minor), indexed from the tonic.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot number for each MAJOR tonic pitch-class (circle of fifths: B major = 1B, then +P5 each step).
# A minor key shares its relative major's NUMBER (relative major = tonic + 3 semitones), with letter 'A'.
_MAJOR_CAMELOT_NUM: dict[int, int] = {
    0: 8,
    1: 3,
    2: 10,
    3: 5,
    4: 12,
    5: 7,
    6: 2,
    7: 9,
    8: 4,
    9: 11,
    10: 6,
    11: 1,
}


@dataclass(frozen=True)
class TempoEstimate:
    """Real tempo readout. ``bpm`` is octave-folded into ``[70, 180)``, or ``None`` when tempo
    could not be measured (signal too short, no valid raw tempo) — a successful estimate never
    carries ``None``, and a failed one has ``confidence == 0.0``. ``confidence`` and ``stability``
    are genuine measurements; ``beat_times`` is the beat grid in seconds."""

    bpm: float | None
    confidence: float
    stability: bool
    beat_times: list[float]


@dataclass(frozen=True)
class KeyEstimate:
    """Real key readout: Camelot code (``8A``/``8B`` …), tonic pitch-class + mode, a correlation
    margin as confidence, and the next-best Camelot alternatives (descending). ``camelot``,
    ``tonic`` and ``mode`` are ``None`` when the key could not be measured (signal too short) —
    a successful estimate never carries ``None``, and a failed one has ``confidence == 0.0`` and
    empty ``alternatives``."""

    camelot: str | None
    tonic: str | None
    mode: str | None
    confidence: float
    alternatives: list[str]


@dataclass(frozen=True)
class SignalSet:
    tempo: TempoEstimate
    key: KeyEstimate


def _fold_octave(bpm: float, lo: float = _TEMPO_LO, hi: float = _TEMPO_HI) -> float | None:
    """Fold a raw BPM into ``[lo, hi)`` by doubling/halving — collapses octave errors (60→120).

    ``None`` for a non-finite or non-positive input: an invalid raw tempo is an absence of
    measurement, and folding it into range would fabricate one.
    """
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    while bpm < lo:
        bpm *= 2.0
    while bpm >= hi:
        bpm /= 2.0
    return float(bpm)


def _tempo_confidence(onset_env: np.ndarray, sr: int, raw_bpm: float) -> float:
    """Onset-autocorrelation peak at the (raw, un-folded) beat lag, in [0, 1]. Real sharpness."""
    if raw_bpm <= 0 or onset_env.size < 4:
        return 0.0
    ac = librosa.autocorrelate(onset_env, max_size=onset_env.size)
    ac0 = float(ac[0])
    if ac0 <= 0:
        return 0.0
    ac = ac / ac0  # lag-0 == 1
    lag = int(round((60.0 / raw_bpm) * sr / _HOP_LENGTH))  # one beat period, in onset frames
    if lag <= 0 or lag >= ac.size:
        return 0.0
    return float(np.clip(ac[lag], 0.0, 1.0))


def _tempo_stability(beat_times: list[float], cv_threshold: float = _STABILITY_CV) -> bool:
    """True when the inter-beat interval is regular (coeff. of variation below ``cv_threshold``)."""
    if len(beat_times) < 3:
        return False
    ibi = np.diff(np.asarray(beat_times, dtype=float))
    mean = float(ibi.mean())
    if ibi.size == 0 or mean <= 0:
        return False
    return bool(float(ibi.std()) / mean < cv_threshold)


def _bpm_from_beats(beat_times: list[float]) -> float:
    """BPM from the mean inter-beat interval of the tracked grid — averages out the frame
    quantization, so it is tighter than ``beat_track``'s prior-biased tempo estimate. 0 if < 2 beats."""
    if len(beat_times) < 2:
        return 0.0
    mean_ibi = float(np.mean(np.diff(np.asarray(beat_times, dtype=float))))
    return 60.0 / mean_ibi if mean_ibi > 0 else 0.0


def estimate_tempo(y: np.ndarray, sr: int) -> TempoEstimate:
    """BPM + beat grid via ``librosa.beat.beat_track`` (onset envelope → DP beat tracking).

    ``bpm`` is ``None`` when tempo could not be measured — waveform shorter than one analysis hop,
    or no valid raw tempo to fold (e.g. silence) — with ``confidence == 0.0``; a successful
    estimate never carries ``None``. The confidence is measured at the ORIGINAL detected tempo
    (before octave-folding), so a half/double-time detection is still scored against its own beat
    period. Deterministic.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(-1))
    if y.size < _HOP_LENGTH:  # too short to track — an absence, not a zero measurement
        return TempoEstimate(bpm=None, confidence=0.0, stability=False, beat_times=[])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH, units="frames"
    )
    beat_times = [float(t) for t in librosa.frames_to_time(beats, sr=sr, hop_length=_HOP_LENGTH)]
    # Prefer the beat-grid mean IBI (accurate to the metronome) over beat_track's prior-biased
    # tempo scalar; fall back to that scalar when there are too few beats to measure an interval.
    raw_bpm = _bpm_from_beats(beat_times) or float(np.atleast_1d(tempo)[0])
    return TempoEstimate(
        bpm=_fold_octave(raw_bpm),
        confidence=_tempo_confidence(onset_env, sr, raw_bpm),
        stability=_tempo_stability(beat_times),
        beat_times=beat_times,
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either vector has no variance (e.g. silence)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def estimate_key(y: np.ndarray, sr: int) -> KeyEstimate:
    """Key via Krumhansl-Schmuckler: mean chroma correlated against the 24 rotated key profiles.

    Confidence is the correlation margin of the winner over the runner-up; ``alternatives`` are the
    next three Camelot codes (descending). When the waveform is shorter than one analysis hop the
    key cannot be measured: ``camelot``/``tonic``/``mode`` are ``None``, ``confidence`` is ``0.0``
    and ``alternatives`` is empty; a successful estimate never carries ``None``. Deterministic;
    resamples to 22.05 kHz for the chromagram.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(-1))
    if y.size < _HOP_LENGTH:  # too short to measure — an absence, not an empty-string readout
        return KeyEstimate(camelot=None, tonic=None, mode=None, confidence=0.0, alternatives=[])
    y_chroma = y if sr == _CHROMA_SR else librosa.resample(y, orig_sr=sr, target_sr=_CHROMA_SR)
    chroma = librosa.feature.chroma_cqt(y=y_chroma, sr=_CHROMA_SR, hop_length=_HOP_LENGTH)
    profile = chroma.mean(axis=1)  # (12,) mean pitch-class energy

    cands: list[tuple[float, int, str]] = []  # (corr, tonic_pc, mode)
    for pc in range(12):
        cands.append((_pearson(profile, np.roll(_KS_MAJOR, pc)), pc, "major"))
        cands.append((_pearson(profile, np.roll(_KS_MINOR, pc)), pc, "minor"))
    # Descending correlation; deterministic tie-break by (pitch-class, mode).
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))

    best, second = cands[0], cands[1]
    tonic_pc, mode = best[1], best[2]
    return KeyEstimate(
        camelot=to_camelot(tonic_pc, mode),
        tonic=_PITCH_CLASSES[tonic_pc],
        mode=mode,
        confidence=float(np.clip(best[0] - second[0], 0.0, 1.0)),
        alternatives=[to_camelot(c[1], c[2]) for c in cands[1:4]],
    )


def to_camelot(tonic_pc: int, mode: str) -> str:
    """``(pitch-class, mode)`` → Camelot code. ``8B`` = C major, ``8A`` = A minor (its relative)."""
    pc = int(tonic_pc) % 12
    if mode == "major":
        return f"{_MAJOR_CAMELOT_NUM[pc]}B"
    return f"{_MAJOR_CAMELOT_NUM[(pc + 3) % 12]}A"  # minor shares its relative major's number


def camelot_neighbors(code: str) -> list[str]:
    """Harmonically-compatible Camelot codes: ±1 around the wheel + the relative-mode switch.

    The three classic mix-in-key moves from ``code``: one step counter-clockwise, one step
    clockwise (both keep the letter/mode), and the relative major↔minor flip (same number, other
    letter). E.g. ``8A`` → ``['7A', '9A', '8B']``. The primitive behind harmonic "radio" ranking.
    """
    num = int(code[:-1])
    letter = code[-1]
    down = f"{12 if num == 1 else num - 1}{letter}"
    up = f"{1 if num == 12 else num + 1}{letter}"
    relative = f"{num}{'B' if letter == 'A' else 'A'}"
    return [down, up, relative]


def extract_signals(y: np.ndarray, sr: int) -> SignalSet:
    """One-pass tempo + key extraction from an already-decoded waveform (no decoding here)."""
    return SignalSet(tempo=estimate_tempo(y, sr), key=estimate_key(y, sr))


# --------------------------------------------------------------------------------------------------
# Structural segmentation — a home-grown, librosa-native MSAF recipe.
#
# MSAF (Nieto & Bello) is the reference toolkit but is NOT bundled (stale, cross-platform wheel risk);
# its default structural-features/spectral-clustering pipeline is re-implemented here from primitives
# already present (librosa + sklearn, both torch-free core deps):
#   1. beat-synchronous chroma+MFCC features (aggregated on the ``estimate_tempo`` beat grid),
#   2. a cosine self-similarity matrix (SSM) of those beat frames,
#   3. a Foote checkerboard-novelty curve along the SSM diagonal,
#   4. peak-picked boundaries, and
#   5. AgglomerativeClustering of the per-section means into HONEST letter labels (A/B/C…).
#
# Deterministic (no unseeded RNG; the estimators used take none) and torch-free. Section labels are
# cluster ids, NEVER musical roles ('chorus'/'verse'): the most-repeated group (by total duration) is
# surfaced as a measured "hook", not an inferred role — the project's transparency invariant.
#
# Refs: Foote 2000 (checkerboard novelty); Serrà et al. 2014 (structural features); McFee & Ellis 2014
#       (spectral clustering); Nieto & Bello, MSAF (reference recipe, not bundled).
# --------------------------------------------------------------------------------------------------

_STRUCT_SR: int = 22_050  # documentation: callers should decode at this rate for chroma analysis
_STRUCT_KERNEL: int = 64  # Foote checkerboard kernel size (beat-sync frames); capped to the SSM
_STRUCT_MAX_LABELS: int = 6  # upper bound on distinct section clusters
_STRUCT_N_MFCC: int = 13
_STRUCT_MIN_BEATS: int = 8  # below this, a track is a single 'A' section (no reliable structure)
_STRUCT_PEAK_DELTA: float = (
    0.06  # novelty threshold above the local mean for a boundary (peak_pick)
)


@dataclass(frozen=True)
class Segment:
    """One structural section. ``label`` is an HONEST cluster id ('A','B','C'…), never a musical role;
    sections that repeat share a ``group`` (and therefore a label)."""

    start: float  # seconds
    end: float  # seconds
    label: str  # 'A'..'Z' — cluster id, NEVER 'chorus'/'verse'
    group: int  # section-cluster id (repeated sections share it)


@dataclass(frozen=True)
class Structure:
    """A track's structural segmentation. ``hook_group`` is the group with the greatest TOTAL duration
    among groups that occur more than once (the most-repeated passage — a measured fact, not a role);
    ``None`` when nothing repeats."""

    segments: list[Segment]
    n_boundaries: int  # number of internal boundaries detected (novelty peaks)
    hook_group: int | None
    hook_start: float | None  # start (s) of the first segment of ``hook_group``; None if no hook


def _checkerboard_kernel(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel (Foote 2000): +1 on the two on-diagonal quadrants, −1 on
    the off-diagonal ones, tapered by a 2-D Gaussian so distant frames weigh less. Even-sized."""
    L = size if size % 2 == 0 else size + 1
    half = L // 2
    # Symmetric half-integer coords ([-3.5..3.5] for L=8): equal +/- counts and no zero on the axes, so
    # the checkerboard sums to ~0 (a uniform SSM region yields ~0 novelty — a true Foote kernel). Using
    # np.arange(-half, half) would leave one extra negative sample + a zeroed centre cross (biased).
    coords = np.arange(L, dtype=np.float64) - (L - 1) / 2.0
    xx, yy = np.meshgrid(coords, coords)
    taper = np.exp(-((xx / half) ** 2 + (yy / half) ** 2) * 2.0)
    return taper * np.sign(xx) * np.sign(yy)


def _foote_novelty(ssm: np.ndarray, size: int) -> np.ndarray:
    """Correlate the checkerboard kernel along the SSM diagonal → a novelty curve, high where the block
    structure changes (Foote 2000). Edge-padded; clipped to ≥ 0 and peak-normalized to [0, 1]."""
    n = ssm.shape[0]
    L = min(size, n)
    if L % 2:
        L -= 1
    if L < 4:
        return np.zeros(n, dtype=np.float64)
    half = L // 2
    kernel = _checkerboard_kernel(L)
    padded = np.pad(ssm, half, mode="edge")
    nov = np.empty(n, dtype=np.float64)
    for i in range(n):
        nov[i] = float(np.sum(padded[i : i + L, i : i + L] * kernel))
    nov = np.maximum(nov, 0.0)
    peak = float(nov.max())
    return nov / peak if peak > 0.0 else nov


def _section_groups(section_feats: np.ndarray, max_labels: int) -> np.ndarray:
    """Cluster per-section mean features into ≤ ``max_labels`` groups (Agglomerative, cosine). Picks the
    cluster count by a silhouette sweep (deterministic); a single section → group 0."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = section_feats.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n == 2:
        # silhouette is undefined for 2 samples/2 clusters; decide by their cosine similarity.
        a, b = section_feats
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return np.array([0, 0] if sim > 0.9 else [0, 1], dtype=int)

    best_labels = np.zeros(n, dtype=int)
    best_score = -2.0
    for k in range(2, min(max_labels, n - 1) + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(section_feats)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(section_feats, labels, metric="cosine"))
        if score > best_score:
            best_score, best_labels = score, labels
    return _canonicalize_groups(best_labels)


def _canonicalize_groups(labels: np.ndarray) -> np.ndarray:
    """Relabel groups 0,1,2… in order of FIRST appearance, so the label alphabet is deterministic and
    independent of the clusterer's internal numbering (section 0 is always group 0 → 'A')."""
    remap: dict[int, int] = {}
    out = np.empty(len(labels), dtype=int)
    for i, g in enumerate(labels):
        gi = int(g)
        if gi not in remap:
            remap[gi] = len(remap)
        out[i] = remap[gi]
    return out


def segment_structure(
    y: np.ndarray,
    sr: int,
    *,
    kernel_size: int = _STRUCT_KERNEL,
    max_labels: int = _STRUCT_MAX_LABELS,
) -> Structure:
    """Segment a track into structural sections (intro/passage/repeated-passage…) with honest letter
    labels. See the module section header for the recipe (beat-sync features → cosine SSM → Foote
    novelty → peak-pick → Agglomerative section clustering). Deterministic and torch-free.

    Returns an empty :class:`Structure` for a null signal, and a single 'A' section when the track is
    too short to yield a reliable beat grid (never a fabricated boundary).
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).reshape(-1))
    duration = float(y.size) / float(sr) if sr else 0.0
    if duration <= 0.0:
        return Structure(segments=[], n_boundaries=0, hook_group=None, hook_start=None)

    beat_times = estimate_tempo(y, sr).beat_times
    if len(beat_times) < _STRUCT_MIN_BEATS:
        return Structure(
            segments=[Segment(0.0, duration, "A", 0)],
            n_boundaries=0,
            hook_group=None,
            hook_start=None,
        )

    # Beat-synchronous chroma+MFCC, L2-normalized columns (one column per inter-beat segment).
    beat_frames = np.unique(
        librosa.time_to_frames(np.asarray(beat_times), sr=sr, hop_length=_HOP_LENGTH)
    )
    beat_frames = beat_frames[beat_frames > 0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=_HOP_LENGTH, n_mfcc=_STRUCT_N_MFCC)
    feat = np.vstack([chroma, mfcc])
    sync = librosa.util.sync(
        feat, beat_frames.tolist(), aggregate=np.mean
    )  # (d, n_cols), n_cols = len(bf)+1
    sync = sync / np.maximum(np.linalg.norm(sync, axis=0, keepdims=True), 1e-8)
    n_cols = sync.shape[1]

    # Column j spans [starts_ext[j], starts_ext[j+1]) in seconds. Derive the edges from the SAME
    # ``beat_frames`` fed to ``librosa.util.sync`` (column j starts at ``beat_frames[j-1]``) — NOT a
    # positional slice of the unfiltered ``beat_times``, which shifts every boundary by a beat whenever
    # the ``> 0`` filter or the ``np.unique`` collapse shortened ``beat_frames``. A rare trailing
    # boundary-collapse (a beat at the last frame) is caught by the length guard → uniform-beat fallback.
    col_edges = librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP_LENGTH)
    starts_ext = np.concatenate(([0.0], col_edges, [duration]))
    if starts_ext.shape[0] != n_cols + 1:
        starts_ext = np.linspace(0.0, duration, n_cols + 1)

    # Cosine SSM → Foote checkerboard novelty → peak-picked internal boundaries (column indices).
    ssm = sync.T @ sync
    novelty = _foote_novelty(ssm, kernel_size)
    half = max(2, min(kernel_size, n_cols) // 4)
    wait = max(2, n_cols // (2 * max_labels))
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=half,
        post_max=half,
        pre_avg=half,
        post_avg=half,
        delta=_STRUCT_PEAK_DELTA,
        wait=wait,
    )
    boundaries = sorted(int(p) for p in np.atleast_1d(peaks) if 0 < int(p) < n_cols)

    # Sections = runs of columns between consecutive boundaries (0 and n_cols bracket the track).
    cuts = [0, *boundaries, n_cols]
    ranges = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    section_feats = np.vstack([sync[:, a:b].mean(axis=1) for a, b in ranges])
    groups = _section_groups(section_feats, max_labels)

    segments = [
        Segment(
            start=float(starts_ext[a]),
            end=float(starts_ext[b]),
            label=chr(ord("A") + int(g)),
            group=int(g),
        )
        for (a, b), g in zip(ranges, groups)
    ]

    # hook_group = the group with the greatest TOTAL duration among groups that occur >1× (a measured
    # repetition, not a role). Deterministic tie-break by lowest group id.
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for seg in segments:
        totals[seg.group] = totals.get(seg.group, 0.0) + (seg.end - seg.start)
        counts[seg.group] = counts.get(seg.group, 0) + 1
    repeated = [g for g, c in counts.items() if c > 1]
    hook_group = min(repeated, key=lambda g: (-totals[g], g)) if repeated else None
    hook_start = (
        next((seg.start for seg in segments if seg.group == hook_group), None)
        if hook_group is not None
        else None
    )

    return Structure(
        segments=segments,
        n_boundaries=len(boundaries),
        hook_group=hook_group,
        hook_start=hook_start,
    )
