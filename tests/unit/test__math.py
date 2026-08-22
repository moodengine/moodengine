"""Tests for moodengine._math — the shared numeric primitives."""

import numpy as np
import pytest
from assertpy import assert_that

from moodengine import labeling, pooling
from moodengine._math import is_constant_series, l2_normalize


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


# --------------------------------------------------------------------------- #
# is_constant_series — the guard every correlation needs before dividing by a std
# --------------------------------------------------------------------------- #


def test_is_constant_series_accepts_a_series_constant_only_to_float_noise() -> None:
    """One ULP of difference is not variance, it is representation error.

    ``std == 0.0`` only catches an EXACT tie, and a series differing in its last bit sails past it
    into ``np.corrcoef``, which then reports a confident-looking correlation built from that bit.
    """
    exact = np.array([0.5, 0.5, 0.5])
    one_ulp = np.array([0.5, 0.5, 0.5 + 1e-16])

    assert_that(is_constant_series(exact)).is_true()
    assert_that(is_constant_series(one_ulp)).is_true()
    assert_that(float(one_ulp.std())).is_greater_than(0.0)  # the old `== 0.0` test would pass it


@pytest.mark.parametrize("spread", [1e-3, 1e-6, 1e-9, 1e-11])
def test_is_constant_series_keeps_a_genuinely_small_spread(spread: float) -> None:
    """A real signal eight orders above float noise must not be called constant."""
    values = np.array([0.5, 0.5 + spread / 2, 0.5 + spread])

    assert_that(is_constant_series(values)).is_false()


def test_is_constant_series_is_relative_to_the_scale_it_sits_on() -> None:
    """The same absolute spread is signal on small values and noise on large ones.

    An absolute threshold would have to be wrong for one of the two; the affect axes live in
    [0, 1] but this helper is shared, so it compares against the magnitude present.
    """
    small = np.array([1e-6, 2e-6, 3e-6])  # spread == the values themselves
    # +1e-3 and not +1e-6: one ULP at 1e12 is ~1.2e-4, so a smaller increment rounds away and the
    # array would be EXACTLY constant — which every version of this guard already catches.
    large = np.array([1e12, 1e12, 1e12 + 1e-3])  # a real increment, 15 orders below the values

    assert_that(is_constant_series(small)).is_false()
    assert_that(is_constant_series(large)).is_true()


def test_is_constant_series_treats_too_short_a_series_as_constant() -> None:
    """Nothing can vary in fewer than two points, so there is nothing to correlate."""
    assert_that(is_constant_series(np.array([]))).is_true()
    assert_that(is_constant_series(np.array([0.5]))).is_true()


def test_is_constant_series_handles_an_all_zero_series() -> None:
    """Scale 0 must not make the threshold meaningless — all-zero is constant, not undefined."""
    assert_that(is_constant_series(np.zeros(5))).is_true()
