"""Unit tests for moodengine.novelty — global OOD scoring (Mahalanobis + deep-kNN distance).

Pure numpy, torch-free. Pins: a point deliberately far from a dense blob is ranked FIRST (most novel)
by BOTH scores, on the same synthetic library, measured + logged; scores are deterministic, bounded,
and never raise on degenerate input.
"""

from __future__ import annotations

import logging

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from assertpy import assert_that
from hypothesis import given

import moodengine.novelty as novelty
from moodengine.novelty import knn_distance_scores, mahalanobis_scores

logger = logging.getLogger(__name__)


def _blob_plus_outlier(seed: int = 0, n: int = 40, d: int = 16):
    """A dense unit-sphere blob (rows 0..n-1) + one point pushed to a far, sparse direction (row n).
    The outlier is the LAST row so 'ranked first' is a non-trivial claim."""
    rng = np.random.default_rng(seed)
    blob = rng.standard_normal((n, d)).astype(np.float32) * 0.15 + np.array(
        [1.0] + [0.0] * (d - 1), np.float32
    )
    blob /= np.linalg.norm(blob, axis=1, keepdims=True)
    outlier = np.zeros(d, np.float32)
    outlier[-1] = 1.0  # orthogonal to the blob's mean direction → far in cosine + Mahalanobis
    X = np.vstack([blob, outlier[None, :]]).astype(np.float32)
    return X, n  # n == index of the outlier row


def test_mahalanobis_ranks_the_outlier_first():
    X, out = _blob_plus_outlier()
    s = mahalanobis_scores(X)
    assert_that(s.shape).is_equal_to((X.shape[0],))
    assert_that(bool(np.all(s >= 0.0))).is_true()
    logger.info("mahalanobis: outlier=%.3f max_inlier=%.3f", s[out], s[:out].max())
    assert_that(int(np.argmax(s))).is_equal_to(out)  # the far point is the most novel
    assert_that(float(s[out])).is_greater_than(float(s[:out].max()))


def test_knn_distance_ranks_the_outlier_first_and_is_bounded():
    X, out = _blob_plus_outlier()
    s = knn_distance_scores(X, k=5)
    assert_that(s.shape).is_equal_to((X.shape[0],))
    assert_that(bool(np.all((s >= 0.0) & (s <= 2.0)))).is_true()  # cosine-distance range
    logger.info("knn_distance: outlier=%.3f max_inlier=%.3f", s[out], s[:out].max())
    assert_that(int(np.argmax(s))).is_equal_to(out)
    assert_that(float(s[out])).is_greater_than(float(s[:out].max()))


def test_scores_are_deterministic():
    X, _ = _blob_plus_outlier(seed=3)
    assert_that(
        bool(np.allclose(mahalanobis_scores(X), mahalanobis_scores(X), atol=1e-6))
    ).is_true()
    assert_that(
        bool(np.allclose(knn_distance_scores(X, k=7), knn_distance_scores(X, k=7), atol=1e-6))
    ).is_true()


def test_knn_excludes_self_and_clamps_k():
    # Two identical rows + one distinct: with self-exclusion, an identical row's nearest is its twin
    # (cosine 1 → distance 0), not itself.
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    s = knn_distance_scores(X, k=1)
    assert_that(float(s[0])).is_close_to(
        0.0, tolerance=1e-6
    )  # nearest OTHER row is its identical twin
    # k is clamped to the available neighbour count (n-1) — no crash when k exceeds it.
    s_big = knn_distance_scores(X, k=999)
    assert_that(s_big.shape).is_equal_to((3,))


def test_degenerate_inputs_never_raise():
    assert_that(mahalanobis_scores(np.zeros((0, 4), np.float32)).shape).is_equal_to((0,))
    assert_that(knn_distance_scores(np.zeros((0, 4), np.float32)).shape).is_equal_to((0,))
    one = np.ones((1, 4), np.float32)
    assert_that(
        bool(np.all(mahalanobis_scores(one) == 0.0))
    ).is_true()  # <2 reference rows → no distribution → zeros
    assert_that(
        bool(np.all(knn_distance_scores(one) == 0.0))
    ).is_true()  # no available neighbour → zeros


def test_ref_argument_uses_external_reference():
    # With an explicit ref (no self-exclusion), a point equal to a ref row scores ~0 kNN distance.
    ref = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    X = np.array([[1.0, 0.0]], dtype=np.float32)
    assert_that(float(knn_distance_scores(X, k=1, ref=ref)[0])).is_close_to(0.0, tolerance=1e-6)


@pytest.mark.parametrize("use_ref", [False, True], ids=["self", "external-ref"])
def test_knn_blockwise_equals_single_block(monkeypatch, use_ref):
    """The row-slab computation is a memory optimization only: forcing a tiny block
    size that never divides n evenly must give the same scores as the one-block
    path. Tolerance is float32-ULP-level, not exact — BLAS may accumulate a slab
    matmul in a different order than the full one."""
    import moodengine.novelty as novelty

    rng = np.random.default_rng(11)
    X = rng.standard_normal((53, 16)).astype(np.float32)
    ref = rng.standard_normal((37, 16)).astype(np.float32) if use_ref else None

    full = knn_distance_scores(X, k=5, ref=ref)  # n < default block → single slab
    monkeypatch.setattr(novelty, "_KNN_BLOCK_ROWS", 7)
    chunked = knn_distance_scores(X, k=5, ref=ref)

    np.testing.assert_allclose(chunked, full, rtol=0.0, atol=2e-6)


def test_knn_distance_is_nonnegative_with_exact_duplicates():
    """Regression: an exact duplicate makes two 512-d float32 rows' cosine round ABOVE 1.0, so ``1−cos``
    would be a physically-impossible NEGATIVE distance. It must clamp to >= 0 (the duplicate reads ~0).
    Loops over seeds so the float32 overflow is guaranteed to occur — and asserts it did (non-vacuous)."""
    saw_overflow = False
    for seed in range(30):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((25, 512)).astype(
            np.float32
        )  # 512-d like CLAP → float32 rounding shows
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.vstack([X, X[3][None, :]]).astype(np.float32)  # exact duplicate of row 3
        s = knn_distance_scores(X, k=1)
        assert_that(bool(np.all(s >= 0.0))).is_true()  # clamp holds: never a negative distance
        assert_that(float(s[3])).is_close_to(0.0, tolerance=1e-6)  # duplicate reads distance 0
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        if float(Xn[3] @ Xn[-1]) > 1.0:  # raw float32 self-cosine overflowed
            saw_overflow = True
    assert_that(saw_overflow).described_as(
        "vacuous test: no float32 cosine overflow occurred across the seeds"
    ).is_true()


def test_novelty_scores_reject_non_finite_input():
    """Degenerate SIZES yield zeros (documented), but non-finite DATA raises: a NaN
    row would poison the covariance / every neighbour distance silently."""
    X = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    X[2, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        knn_distance_scores(X, k=3)
    with pytest.raises(ValueError, match="non-finite"):
        mahalanobis_scores(X)


def test_knn_self_path_reuses_one_normalization(mocker) -> None:
    """`ref is None` means `R is X`, and normalization is idempotent — normalizing twice bought
    nothing and allocated a second (n, d) copy."""
    spy = mocker.spy(novelty, "l2_normalize")
    X = np.random.default_rng(4).standard_normal((20, 6)).astype(np.float32)

    knn_distance_scores(X, k=3)

    assert_that(spy.call_count).is_equal_to(1)


def test_knn_distance_scores_does_not_mutate_its_input() -> None:
    """The top-k select runs in place on the slab, which is freshly allocated per block — the
    caller's matrix must be untouched, as the docstring promises."""
    X = np.random.default_rng(5).standard_normal((30, 8)).astype(np.float32)
    before = X.copy()

    knn_distance_scores(X, k=4)

    np.testing.assert_array_equal(X, before)


# --------------------------------------------------------------------------- #
# chunk-max prefilter (the wide-row top-k select)
# --------------------------------------------------------------------------- #
#: A chunk this wide makes `n_chunks <= kk` for every row width used below, which is the
#: guard's first fallback condition — so patching it in forces the full-width partition and
#: gives the tests a reference path that is the code as it shipped before the prefilter.
_FORCE_FULL_WIDTH = 1 << 20


@pytest.mark.parametrize("n", [997, 1500, 2048, 5000], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("k", [3, 10, 25], ids=lambda k: f"k{k}")
def test_knn_prefilter_equals_the_full_width_select(monkeypatch, n, k):
    """Prefiltered scores must match the full-width partition at float32-ULP tolerance.

    Widths are chosen to cover the shapes the chunking can get wrong: 997 and 1500 are ragged
    (not multiples of the 32-wide chunk, so the tail is appended rather than chunked), 2048 is
    an exact multiple, and 1500/2048/5000 span more than one row slab."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, 8)).astype(np.float32)

    prefiltered = knn_distance_scores(X, k=k)
    monkeypatch.setattr(novelty, "_KNN_PREFILTER_CHUNK", _FORCE_FULL_WIDTH)
    full_width = knn_distance_scores(X, k=k)

    np.testing.assert_allclose(prefiltered, full_width, rtol=0.0, atol=2e-6)


def test_knn_prefilter_selects_the_same_values_under_heavy_ties():
    """Ties are what the disjoint-chunk argument has to survive, so pin them exactly.

    Sign vectors give enormous numbers of EXACTLY equal cosines, so which physical element a
    selection returns is ambiguous — but the multiset of VALUES is not, and that is what the
    mean consumes. Bitwise equality, not a tolerance: a wrong chunk count would change the
    values themselves, not merely their order."""
    rng = np.random.default_rng(17)
    X = (rng.integers(0, 2, (600, 24)).astype(np.float32) * 2 - 1) / np.sqrt(24)
    sims = (X @ X.T).astype(np.float32)
    reference = sims.copy()
    reference.partition(reference.shape[1] - 10, axis=1)

    selected = novelty._top_k_per_row(sims.copy(), 10)

    assert_that(
        np.array_equal(np.sort(selected, axis=1), np.sort(reference[:, -10:], axis=1))
    ).is_true()


def test_knn_prefilter_guard_falls_back_when_chunking_would_not_narrow_the_row():
    """The guard has to actually switch paths, so assert the switch, not just the answer.

    The two paths differ observably: the fallback partitions the slab IN PLACE (that is the
    allocation it saves), while the prefilter gathers candidates and leaves the slab intact.
    A narrow row must take the first, a wide row the second, and both must agree."""
    rng = np.random.default_rng(23)
    narrow = rng.standard_normal((4, 120)).astype(np.float32)  # 3 chunks, k=10 -> no narrowing
    wide = rng.standard_normal((4, 4000)).astype(np.float32)  # 125 chunks -> prefilter

    narrow_in, wide_in = narrow.copy(), wide.copy()
    narrow_out = novelty._top_k_per_row(narrow_in, 10)
    wide_out = novelty._top_k_per_row(wide_in, 10)

    assert_that(np.array_equal(narrow_in, narrow)).is_false()  # partitioned in place
    assert_that(np.array_equal(wide_in, wide)).is_true()  # left intact
    for got, src in ((narrow_out, narrow), (wide_out, wide)):
        want = src.copy()
        want.partition(want.shape[1] - 10, axis=1)
        assert_that(np.array_equal(np.sort(got, axis=1), np.sort(want[:, -10:], axis=1))).is_true()


# --------------------------------------------------------------------------- #
# Properties — mahalanobis_scores alone carried 48 surviving mutants, the most
# of any function outside calibration.
# --------------------------------------------------------------------------- #

_EMB = npst.arrays(
    dtype=np.float32,
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10),
    elements=st.floats(-50.0, 50.0, width=32, allow_nan=False, allow_infinity=False),
)


@given(X=_EMB)
def test_mahalanobis_scores_are_non_negative_and_finite(X: np.ndarray) -> None:
    """A Mahalanobis distance is a norm under a positive-definite metric: never below zero.

    The shrinkage term exists to keep the covariance invertible, so a non-finite score means it
    failed on a degenerate input rather than that the input was unusual.
    """
    scores = mahalanobis_scores(X)

    assert_that(scores.shape).is_equal_to((X.shape[0],))
    assert_that(bool(np.all(scores >= 0.0))).is_true()
    assert_that(bool(np.isfinite(scores).all())).is_true()


@given(X=_EMB)
def test_mahalanobis_scores_are_invariant_to_row_order(X: np.ndarray) -> None:
    """The score of a row depends on the distribution, not on where the row sits in the matrix.

    A caller shuffling its library must get the same per-track novelty back, permuted.
    """
    order = np.arange(X.shape[0])[::-1]

    scores = mahalanobis_scores(X)
    shuffled = mahalanobis_scores(X[order])

    np.testing.assert_allclose(shuffled, scores[order], rtol=1e-4, atol=1e-5)


@given(X=_EMB, k=st.integers(1, 6))
def test_knn_distance_scores_are_non_negative_and_finite(X: np.ndarray, k: int) -> None:
    """A distance to a neighbour cannot be negative, whatever k is relative to n."""
    scores = knn_distance_scores(X, k=k)

    assert_that(scores.shape).is_equal_to((X.shape[0],))
    assert_that(bool(np.all(scores >= 0.0))).is_true()
    assert_that(bool(np.isfinite(scores).all())).is_true()
