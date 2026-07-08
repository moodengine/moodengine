"""Unit tests for moodengine.adapt — the training-free Tip-Adapter key-value cache. Torch-free, AAA."""

from __future__ import annotations

import numpy as np
from assertpy import assert_that

from moodengine.adapt import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    acquisition_scores,
    diverse_subset,
    prototype_vector,
    tip_adapter_affinities,
)


def _l2(v):
    v = np.asarray(v, dtype=np.float32)
    return (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)


def test_defaults_are_sane():
    assert_that(DEFAULT_BETA).is_greater_than(0.0)
    assert_that(DEFAULT_ALPHA).is_greater_than_or_equal_to(0.0)


def test_empty_cache_yields_zeros():
    X = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    A = tip_adapter_affinities(X, np.zeros((0, 8), np.float32), np.zeros((0, 3), np.float32))
    assert_that(A.shape).is_equal_to((4, 3))
    assert_that(bool(np.all(A == 0.0))).is_true()


def test_empty_query_yields_zeros():
    A = tip_adapter_affinities(
        np.zeros((0, 8), np.float32), np.ones((2, 8), np.float32), np.eye(2, dtype=np.float32)
    )
    assert_that(A.shape).is_equal_to((0, 2))


def test_affinity_peaks_on_nearest_keys_label():
    d = 8
    k0 = _l2([1.0] + [0.0] * (d - 1))
    k1 = _l2([0.0, 1.0] + [0.0] * (d - 2))
    K = np.vstack([k0, k1]).astype(np.float32)
    V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # key0 -> label 0, key1 -> label 1
    A = tip_adapter_affinities(k0[None, :].astype(np.float32), K, V, beta=DEFAULT_BETA)
    assert_that(float(A[0, 0])).is_greater_than(
        float(A[0, 1])
    )  # query == key0 -> label 0 dominates
    assert_that(bool(np.all(A >= 0.0))).is_true()


def test_monotonic_decreasing_in_beta_for_distant_query():
    d = 8
    k = _l2(np.random.default_rng(1).standard_normal(d))
    q = _l2(k + 0.3 * np.random.default_rng(2).standard_normal(d))  # near but not equal -> dist > 0
    K, V, X = (
        k[None, :].astype(np.float32),
        np.array([[1.0]], np.float32),
        q[None, :].astype(np.float32),
    )
    a_low = float(tip_adapter_affinities(X, K, V, beta=1.0)[0, 0])
    a_high = float(tip_adapter_affinities(X, K, V, beta=10.0)[0, 0])
    assert_that(a_high).is_less_than(a_low)  # exp(-beta*dist) shrinks with beta when dist > 0


def test_deterministic_and_no_mutation():
    X = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    K = np.random.default_rng(1).standard_normal((2, 8)).astype(np.float32)
    V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X0, K0 = X.copy(), K.copy()
    a1 = tip_adapter_affinities(X, K, V)
    a2 = tip_adapter_affinities(X, K, V)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(X, X0)  # inputs unmutated
    np.testing.assert_array_equal(K, K0)
    assert_that(a1.dtype).is_equal_to(np.dtype("float32"))


# --------------------------------------------------------------------------- #
# prototype_vector (few-shot mood prototype)
# --------------------------------------------------------------------------- #
def test_prototype_single_row_equals_that_row_normalized():
    v = np.random.default_rng(4).standard_normal(8).astype(np.float32)
    proto = prototype_vector(v[None, :])
    np.testing.assert_allclose(proto, _l2(v), atol=1e-6)  # 1 seed -> its own direction
    assert_that(float(np.linalg.norm(proto))).is_close_to(1.0, tolerance=1e-5)


def test_prototype_is_normalized_mean_of_members():
    embs = np.random.default_rng(5).standard_normal((6, 8)).astype(np.float32)
    proto = prototype_vector(embs)
    np.testing.assert_allclose(proto, _l2(embs.mean(axis=0)), atol=1e-6)
    assert_that(proto.shape).is_equal_to((8,))
    assert_that(proto.dtype).is_equal_to(np.dtype("float32"))


def test_prototype_empty_selection_yields_zeros_on_inferred_d():
    proto = prototype_vector(np.zeros((0, 8), np.float32))
    assert_that(proto.shape).is_equal_to((8,))
    assert_that(bool(np.all(proto == 0.0))).is_true()  # a null mood the caller drops


def test_prototype_deterministic_and_no_mutation():
    embs = np.random.default_rng(6).standard_normal((5, 8)).astype(np.float32)
    before = embs.copy()
    p1 = prototype_vector(embs)
    p2 = prototype_vector(embs)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(embs, before)  # input unmutated


# --------------------------------------------------------------------------- #
# acquisition_scores (uncertainty sampling)
# --------------------------------------------------------------------------- #
def _rows_l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_entropy_bounds_and_extremes():
    C = 5
    uniform = np.full((1, C), 1.0 / C, dtype=np.float32)
    onehot = np.zeros((1, C), dtype=np.float32)
    onehot[0, 0] = 1.0
    assert_that(float(acquisition_scores(uniform, "entropy")[0])).is_close_to(
        float(np.log(C)), 1e-4
    )
    assert_that(float(acquisition_scores(onehot, "entropy")[0])).is_close_to(0.0, 1e-6)
    rng = np.random.default_rng(0)
    P = rng.random((10, C)).astype(np.float32)
    P /= P.sum(axis=1, keepdims=True)
    e = acquisition_scores(P, "entropy")
    assert_that(bool(np.all(e >= -1e-6) and np.all(e <= np.log(C) + 1e-5))).is_true()


def test_margin_bounds_and_extremes():
    onehot = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    tie = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    assert_that(float(acquisition_scores(onehot, "margin")[0])).is_close_to(
        0.0, 1e-6
    )  # p1-p2=1 -> 0
    assert_that(float(acquisition_scores(tie, "margin")[0])).is_close_to(1.0, 1e-6)  # p1-p2=0 -> 1
    m = float(acquisition_scores(np.array([[0.6, 0.3, 0.1]], np.float32), "margin")[0])
    assert_that(0.0 <= m <= 1.0).is_true()


def test_acquisition_guards():
    assert_that(acquisition_scores(np.zeros((0, 5), np.float32)).shape).is_equal_to((0,))
    assert_that(
        bool(np.all(acquisition_scores(np.ones((3, 1), np.float32)) == 0.0))
    ).is_true()  # C=1


# --------------------------------------------------------------------------- #
# diverse_subset (BADGE-style greedy coverage)
# --------------------------------------------------------------------------- #
def test_diverse_subset_starts_at_argmax_caps_and_gamma0_is_topn():
    rng = np.random.default_rng(1)
    X = _rows_l2(rng.standard_normal((6, 8)))
    scores = np.array([0.1, 0.9, 0.2, 0.5, 0.3, 0.4], dtype=np.float32)
    sub = diverse_subset(X, scores, 3, gamma=1.0)
    assert_that(len(sub)).is_less_than_or_equal_to(3)
    assert_that(sub[0]).is_equal_to(int(np.argmax(scores)))  # starts at argmax score
    assert_that(diverse_subset(X, scores, 3, gamma=1.0)).is_equal_to(sub)  # deterministic
    assert_that(diverse_subset(X, scores, 3, gamma=0.0)).is_equal_to(
        [1, 3, 5]
    )  # gamma=0 -> pure top-n
    assert_that(diverse_subset(X, scores, 0)).is_equal_to([])
    assert_that(
        diverse_subset(np.zeros((0, 8), np.float32), np.zeros(0, np.float32), 3)
    ).is_equal_to([])


def test_diverse_subset_increases_intra_list_diversity():
    d = 8
    e = np.eye(d, dtype=np.float32)
    rng = np.random.default_rng(2)
    cluster = _rows_l2(
        np.tile(e[0], (4, 1)) + 0.01 * rng.standard_normal((4, d))
    )  # tight, high-score
    others = np.vstack([e[1], e[2], e[3]]).astype(np.float32)  # orthogonal, lower-score
    X = np.vstack([cluster, others]).astype(np.float32)
    scores = np.array([1.0, 0.99, 0.98, 0.97, 0.5, 0.5, 0.5], dtype=np.float32)

    def mean_pairwise_cos_dist(idxs):
        sub = X[idxs]
        S = sub @ sub.T
        iu = np.triu_indices(len(idxs), 1)
        return float((1.0 - S[iu]).mean())

    topn = sorted(range(len(X)), key=lambda i: -scores[i])[:4]  # the tight cluster -> ~0 diversity
    div = diverse_subset(X, scores, 4, gamma=1.0)
    assert_that(mean_pairwise_cos_dist(div)).is_greater_than(mean_pairwise_cos_dist(topn))
