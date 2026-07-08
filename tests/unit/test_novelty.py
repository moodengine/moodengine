"""Unit tests for moodengine.novelty — global OOD scoring (Mahalanobis + deep-kNN distance).

Pure numpy, torch-free. Pins: a point deliberately far from a dense blob is ranked FIRST (most novel)
by BOTH scores, on the same synthetic library, measured + logged; scores are deterministic, bounded,
and never raise on degenerate input.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from assertpy import assert_that

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
