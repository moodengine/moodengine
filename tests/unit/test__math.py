"""Tests for moodengine._math — the shared numeric primitives."""

import numpy as np
from assertpy import assert_that

from moodengine import labeling, pooling
from moodengine._math import l2_normalize


def test_l2_normalize_rows_have_unit_norm() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((6, 8)).astype(np.float32) * 10.0

    out = l2_normalize(X, axis=1)

    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_l2_normalize_zero_vector_stays_zero_and_finite() -> None:
    x = np.zeros(4, dtype=np.float32)

    out = l2_normalize(x)

    assert_that(bool(np.isfinite(out).all())).is_true()
    np.testing.assert_array_equal(out, np.zeros(4, dtype=np.float32))


def test_l2_normalize_casts_float64_input_to_float32() -> None:
    x = np.ones((2, 3), dtype=np.float64)

    out = l2_normalize(x, axis=1)

    assert_that(str(out.dtype)).is_equal_to("float32")


def test_l2_normalize_is_idempotent() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 6)).astype(np.float32)

    once = l2_normalize(X, axis=1)
    twice = l2_normalize(once, axis=1)

    np.testing.assert_allclose(twice, once, atol=2e-6)


def test_l2_normalize_is_the_single_shared_implementation() -> None:
    # The historical public locations must re-export the SAME function object,
    # not a lookalike copy — that is the dedup contract.
    assert_that(pooling.l2_normalize).is_same_as(l2_normalize)
    assert_that(labeling.l2_normalize).is_same_as(l2_normalize)
