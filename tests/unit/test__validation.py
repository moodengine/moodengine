"""Tests for :mod:`moodengine._validation` — the shared finite-matrix boundary check."""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine._validation import ensure_finite_2d


def test_valid_matrix_passes_through_as_float32() -> None:
    X = np.arange(6, dtype=np.float64).reshape(2, 3)

    out = ensure_finite_2d(X)

    assert_that(out.dtype).is_equal_to(np.dtype("float32"))
    assert_that(out.shape).is_equal_to((2, 3))
    np.testing.assert_allclose(out, X, atol=0)


def test_empty_matrix_is_accepted() -> None:
    """Degenerate SIZES stay each entry point's own contract — (0, d) is not an error here."""
    out = ensure_finite_2d(np.empty((0, 8), dtype=np.float32))
    assert_that(out.shape).is_equal_to((0, 8))


def test_non_2d_input_raises_with_the_array_name() -> None:
    with pytest.raises(ValueError, match=r"emb must be 2-D .* got shape \(4,\)"):
        ensure_finite_2d(np.zeros(4, dtype=np.float32), name="emb")


def test_nan_raises_naming_count_and_offending_rows() -> None:
    """The message must let the caller trace the bad rows back to tracks."""
    X = np.zeros((6, 3), dtype=np.float32)
    X[1, 2] = np.nan
    X[4, 0] = np.nan

    with pytest.raises(ValueError, match="X contains 2 non-finite") as excinfo:
        ensure_finite_2d(X)

    assert_that(str(excinfo.value)).contains("[1, 4]")


def test_infinity_is_rejected_like_nan() -> None:
    X = np.zeros((3, 2), dtype=np.float32)
    X[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        ensure_finite_2d(X)
