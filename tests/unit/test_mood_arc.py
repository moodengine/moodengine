"""Unit tests for moodengine.mood_arc — per-segment mood scoring (the intra-track arc). Torch-free,
deterministic.

The load-bearing guarantees:
  * **triptych equivalence** (crit #1): with ``mean_cosine = sims.mean(axis=0)`` and ``n_seg ≥ 5`` the
    arc is BYTE-identical to ``recenter_similarities(sims) → softmax`` (the canonical labeling path);
  * **pool coherence** (crit #2): ``l2_normalize(mean(seg_embs)) == pool_clap(seg_embs)`` — the arc and
    the global triptych start from the same raw material;
  * **bounds alignment**: ``segment_bounds(len(y))`` yields exactly one span per ``segment_waveform``
    window (pinned across full / partial-kept / partial-dropped / short / capped / overlapped lengths),
    so the arc never reports a fabricated boundary.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from assertpy import assert_that

from moodengine import default_config
from moodengine.io_audio import segment_waveform
from moodengine.labeling import DEFAULT_TEMPERATURE, l2_normalize, recenter_similarities, softmax
from moodengine.mood_arc import score_segment_arc, segment_bounds, segment_embeddings
from moodengine.pooling import pool_clap

_SR = 48_000


def _label_matrix(n_moods: int, d: int, seed: int = 7) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    lm = l2_normalize(rng.standard_normal((n_moods, d)).astype(np.float32), axis=1)
    return lm, [f"m{i}" for i in range(n_moods)]


def _fine_config(**over):
    """A config with short windows so a few-second synthetic waveform yields several segments."""
    base = dict(
        segment_seconds=1.0, overlap_seconds=0.0, min_segment_seconds=0.5, max_segments_per_track=0
    )
    base.update(over)
    return replace(default_config(), **base)


# --------------------------------------------------------------------------- #
# Triptych equivalence (crit #1)
# --------------------------------------------------------------------------- #
def test_arc_equals_recenter_softmax_with_segment_mean():
    rng = np.random.default_rng(0)
    n_seg, d, n_moods = 7, 16, 5  # n_seg ≥ 5 so recenter_similarities engages
    seg = rng.standard_normal((n_seg, d)).astype(np.float32)
    lm, names = _label_matrix(n_moods, d)

    X = l2_normalize(seg, axis=1)
    sims = X @ lm.T
    probs_ref = softmax(
        recenter_similarities(sims, enable=True), temperature=DEFAULT_TEMPERATURE, axis=1
    )

    # mean_cosine = the segment mean → the library-calibration path collapses onto the self path.
    arc_explicit = score_segment_arc(seg, lm, names, mean_cosine=sims.mean(axis=0))
    # mean_cosine = None → recenter_similarities (self-contained) path.
    arc_auto = score_segment_arc(seg, lm, names)

    assert_that(np.array_equal(arc_explicit.probs, probs_ref)).is_true()  # BYTE-identical
    assert_that(
        np.array_equal(arc_auto.probs, probs_ref)
    ).is_true()  # both recover the canonical triptych


def test_arc_recenter_off_is_plain_softmax():
    rng = np.random.default_rng(2)
    seg = rng.standard_normal((6, 16)).astype(np.float32)
    lm, names = _label_matrix(5, 16)
    sims = l2_normalize(seg, axis=1) @ lm.T
    arc = score_segment_arc(seg, lm, names, recenter=False)
    assert_that(
        np.array_equal(arc.probs, softmax(sims, temperature=DEFAULT_TEMPERATURE, axis=1))
    ).is_true()


def test_library_mean_cosine_differs_from_self_mean():
    # Callers deploy the LIBRARY mean (X_clap-wide), not the track's own segment mean — they must
    # actually differ, else "comparable to the track top_score" would be vacuous.
    rng = np.random.default_rng(3)
    seg = rng.standard_normal((6, 16)).astype(np.float32)
    lm, names = _label_matrix(5, 16)
    sims = l2_normalize(seg, axis=1) @ lm.T
    library_mean = rng.standard_normal(5).astype(np.float32) * 0.1  # a different calibration vector
    arc_lib = score_segment_arc(seg, lm, names, mean_cosine=library_mean)
    arc_self = score_segment_arc(seg, lm, names, mean_cosine=sims.mean(axis=0))
    assert_that(np.array_equal(arc_lib.probs, arc_self.probs)).is_false()


# --------------------------------------------------------------------------- #
# Shape / normalization sanity
# (Pool coherence, crit #2, is proven in test_segment_embeddings_pool_reproduces_track_vector below —
#  over REAL segment_embeddings output, not a re-derivation of pool_clap's own formula.)
# --------------------------------------------------------------------------- #
def test_probs_are_valid_distributions_and_top3_consistent():
    rng = np.random.default_rng(4)
    seg = rng.standard_normal((5, 16)).astype(np.float32)
    lm, names = _label_matrix(6, 16)
    arc = score_segment_arc(seg, lm, names, mean_cosine=None, top_k=3)
    assert_that(arc.probs.shape).is_equal_to((5, 6))
    assert_that(
        bool(np.allclose(arc.probs.sum(axis=1), 1.0, atol=1e-5))
    ).is_true()  # each segment sums to ~1
    for s in range(5):
        assert_that(arc.top3[s]).is_length(3)
        assert_that(arc.top_moods[s]).is_equal_to(arc.top3[s][0][0])  # top mood == first of top3
        assert_that(arc.top_scores[s]).is_equal_to(arc.top3[s][0][1])
        # top3 is sorted descending by prob
        assert_that([p for _, p in arc.top3[s]]).is_equal_to(
            sorted((p for _, p in arc.top3[s]), reverse=True)
        )


def test_single_segment_and_empty_are_honest():
    lm, names = _label_matrix(5, 16)
    one = score_segment_arc(
        np.random.default_rng(5).standard_normal((1, 16)).astype(np.float32), lm, names
    )
    assert_that(one.probs.shape).is_equal_to((1, 5))
    assert_that(bool(np.allclose(one.probs.sum(axis=1), 1.0))).is_true()
    empty = score_segment_arc(np.zeros((0, 16), dtype=np.float32), lm, names)
    assert_that(empty.probs.shape).is_equal_to((0, 5))
    assert_that(empty.top_moods).is_equal_to([])
    assert_that(empty.top3).is_equal_to([])


def test_deterministic():
    rng = np.random.default_rng(6)
    seg = rng.standard_normal((7, 16)).astype(np.float32)
    lm, names = _label_matrix(5, 16)
    mc = (l2_normalize(seg, axis=1) @ lm.T).mean(axis=0)
    a = score_segment_arc(seg, lm, names, mean_cosine=mc)
    b = score_segment_arc(seg, lm, names, mean_cosine=mc)
    # numpy field → compare fields explicitly (a bare `==` on the dataclass is ambiguous).
    assert_that(np.array_equal(a.probs, b.probs)).is_true()
    assert_that(a.top_moods).is_equal_to(b.top_moods)
    assert_that(a.top3).is_equal_to(b.top3)


def test_inputs_not_mutated():
    rng = np.random.default_rng(9)
    seg = rng.standard_normal((5, 16)).astype(np.float32)
    lm, names = _label_matrix(5, 16)
    seg0, lm0 = seg.copy(), lm.copy()
    score_segment_arc(seg, lm, names, mean_cosine=None)
    assert_that(np.array_equal(seg, seg0)).is_true()
    assert_that(np.array_equal(lm, lm0)).is_true()


# --------------------------------------------------------------------------- #
# segment_embeddings + segment_bounds geometry (must mirror segment_waveform)
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """A deterministic, torch-free stand-in for CLAPEmbedder: one clip vector per segment."""

    name = "clap"
    sample_rate = _SR

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float32)
        return np.array([w.mean(), w.std(), float(w.size), float(sr)], dtype=np.float32)


def test_segment_embeddings_pool_reproduces_track_vector():
    """Crit #2 (POOL coherence): the arc's per-segment embeddings ARE the fixed windows pool_clap
    aggregates — pooling segment_embeddings' output reproduces the track vector (and yields exactly one
    embedding per segment_waveform window). Exercises the real arc surface (segment_embeddings), so it
    would fail if the arc ever segmented differently (e.g. structural sections) or mis-pooled."""
    cfg = _fine_config()
    y = np.random.default_rng(0).standard_normal(_SR * 3).astype(np.float32)  # 3 s → 3 windows
    emb = _FakeEmbedder()
    segs = segment_embeddings(y, emb, cfg)
    assert_that(segs).is_length(len(segment_waveform(y, emb.sample_rate, cfg)))
    pooled = pool_clap(segs, cfg)
    manual = l2_normalize(np.stack(segs).mean(axis=0), axis=-1)
    assert_that(bool(np.allclose(pooled, manual, atol=1e-6))).is_true()


def test_segment_bounds_align_1to1_with_segment_waveform():
    sr = _SR
    for cfg in (_fine_config(), _fine_config(overlap_seconds=0.5)):
        for n in (sr * 3, int(sr * 2.7), int(sr * 2.4), int(sr * 0.3), sr * 20):
            y = np.zeros(n, dtype=np.float32)
            segs = segment_waveform(y, sr, cfg)
            bounds = segment_bounds(n, sr, cfg)
            assert_that(len(bounds)).described_as(
                str((cfg.overlap_seconds, n, len(bounds), len(segs)))
            ).is_equal_to(len(segs))
            dur = n / sr
            assert_that(bounds[0][0]).is_equal_to(0.0)  # first section starts at 0
            assert_that(
                all(0.0 <= s <= e <= dur + 1e-6 for s, e in bounds)
            ).is_true()  # within the real duration
            assert_that([s for s, _ in bounds]).is_equal_to(
                sorted(s for s, _ in bounds)
            )  # monotone


def test_segment_bounds_respects_cap():
    cfg = _fine_config(max_segments_per_track=4)
    sr, n = _SR, _SR * 10
    y = np.zeros(n, dtype=np.float32)
    assert_that(len(segment_bounds(n, sr, cfg))).is_equal_to(4)
    assert_that(len(segment_waveform(y, sr, cfg))).is_equal_to(4)


def test_segment_bounds_uniform_cap_selects_same_windows_as_the_pool():
    """Under a biting cap + uniform selection, the displayed bounds are the SAME windows the pool
    keeps (both route through io_audio._capped_indices), so a mood arc lines up with the track
    vector — and they cover the whole track, not just the first N seconds."""
    cfg = _fine_config(max_segments_per_track=3, segment_selection="uniform")
    sr, n = _SR, _SR * 10  # 10 one-second windows, keep 3 spread across: seconds 0, 4, 9

    bounds = segment_bounds(n, sr, cfg)

    assert_that(len(bounds)).is_equal_to(3)
    assert_that([round(s, 3) for s, _ in bounds]).is_equal_to([0.0, 4.0, 9.0])


def test_segment_bounds_empty_waveform():
    assert_that(segment_bounds(0, _SR, _fine_config())).is_equal_to([])


# --------------------------------------------------------------------------- #
# Torch-free invariant
# --------------------------------------------------------------------------- #
def test_mood_arc_module_is_torch_free():
    # Importing moodengine.mood_arc AND running the full scorer must not pull torch.
    import subprocess
    import sys

    # NOTE: do NOT call default_config() here — it invokes get_device(), which imports torch for
    # device detection (a known trap). Use a plain namespace so the guard tests OUR module, not that.
    code = (
        "import sys, types, numpy as np; import moodengine.mood_arc as ma; "
        "from moodengine.labeling import l2_normalize; "
        "rng=np.random.default_rng(0); "
        "seg=rng.standard_normal((6,16)).astype('float32'); "
        "lm=l2_normalize(rng.standard_normal((5,16)).astype('float32'),axis=1); "
        "arc=ma.score_segment_arc(seg, lm, [f'm{i}' for i in range(5)], mean_cosine=None); "
        "assert arc.probs.shape==(6,5); "
        "cfg=types.SimpleNamespace(segment_seconds=1.0, overlap_seconds=0.0, "
        "min_segment_seconds=0.5, max_segments_per_track=0, segment_selection='uniform'); "
        "assert ma.segment_bounds(48000*3, 48000, cfg); "
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')]; "
        "sys.exit('torch loaded: '+repr(bad)) if bad else None"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as((r.stdout + r.stderr).strip()).is_equal_to(0)
