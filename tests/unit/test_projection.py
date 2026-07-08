"""Unit tests for the reusable 2-D map projection — moodengine.cluster.fit_projection /
transform_projection / procrustes_disparity / ProjectionMethodUnavailable.

Torch-free (numpy / umap-learn / sklearn / scipy). Pins the STABILITY FIX: a fitted reducer's .transform
reproduces the fit layout (round-trip disparity ~0), so keeping the fit coords for existing points and only
transforming new ones freezes the existing layout — whereas a naive refit moves the same points (measured &
logged, never fabricated). Also: fit_projection(umap) is byte-identical to the legacy reduce_umap(2),
densmap runs, a missing PaCMAP raises a clear ProjectionMethodUnavailable, and procrustes_disparity never
raises on degenerate input.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.cluster import (
    ProjectionMethodUnavailable,
    fit_projection,
    procrustes_disparity,
    reduce_umap,
    transform_projection,
)
from moodengine.config import Config


def _blobs(seed: int = 0, per: int = 20, d: int = 12) -> np.ndarray:
    """Three well-separated Gaussian blobs (n = 3*per ≥ 15 → the real UMAP path, not the tiny fallback)."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal(c, 0.4, (per, d)) for c in ([0.0] * d, [3.0] * d, [-3.0] * d)]
    ).astype(np.float32)


def _cfg(**kw) -> Config:
    return Config(umap_n_neighbors=10, seed=42, **kw)


def test_fit_projection_umap_is_byte_identical_to_reduce_umap():
    X = _blobs()
    coords, reducer = fit_projection(X, _cfg(projection_method="umap"))
    viz, _ = reduce_umap(X, 2, _cfg())
    assert_that(coords.shape).is_equal_to((X.shape[0], 2))
    assert_that(coords.dtype).is_equal_to(np.float32)
    np.testing.assert_array_equal(coords, viz)  # pinned-equivalent to the legacy viz path
    assert_that(hasattr(reducer, "transform")).is_true()


def test_transform_round_trip_reproduces_the_fit_layout():
    X = _blobs()
    coords, reducer = fit_projection(X, _cfg())
    rt = transform_projection(reducer, X)
    d = procrustes_disparity(coords, rt)
    assert_that(d).described_as(
        f"round-trip disparity {d} should be ~0 (same points, same layout)"
    ).is_not_none()
    assert_that(d).described_as(
        f"round-trip disparity {d} should be ~0 (same points, same layout)"
    ).is_less_than(1e-3)


def test_transform_freezes_existing_layout_vs_naive_refit(caplog):
    caplog.set_level(logging.INFO)
    X = _blobs()
    A, B = X[:40], X[40:]
    coords_A, reducer = fit_projection(A, _cfg())

    # THE FIX: to add B we transform ONLY B and KEEP A's fit coords. Keeping A is sound IFF the reducer's
    # .transform reproduces the fit layout for A's own points — so MEASURE that (comparing coords_A to
    # coords_A would be tautologically 0 and would pass even for a broken transform). A transform that
    # returned zeros/garbage makes this disparity large or None and fails here.
    coords_B = transform_projection(reducer, B)
    assert_that(coords_B.shape).is_equal_to((B.shape[0], 2))
    assert_that(bool(np.all(np.isfinite(coords_B)))).is_true()
    frozen = procrustes_disparity(
        coords_A, transform_projection(reducer, A)
    )  # transform must reproduce A

    # BASELINE: a naive refit of A+B (what run_clustering does today) moves the same A points.
    coords_AB, _ = fit_projection(np.vstack([A, B]), _cfg())
    naive = procrustes_disparity(coords_A, coords_AB[: len(A)])

    logging.getLogger("projection_bench").info(
        "map stability — transform(A) disparity=%.2e  vs  naive refit disparity=%.4f", frozen, naive
    )
    # transform reproduces A's layout (freeze is sound)
    assert_that(frozen).is_not_none()
    assert_that(frozen).is_less_than(1e-3)
    # the naive refit demonstrably reshuffles A
    assert_that(naive).is_not_none()
    assert_that(naive).is_greater_than(1e-2)
    assert_that(naive).is_greater_than(frozen)


def test_densmap_projection_fits_a_stable_layout():
    # densMAP produces a density-preserving FIT layout (deterministic for a fixed seed) but does NOT
    # support out-of-sample .transform in umap-learn — so it serves mode="refit" (a stable re-layout),
    # not incremental placement. Two fits with the same seed give the same layout.
    X = _blobs()
    coords, reducer = fit_projection(X, _cfg(projection_method="densmap"))
    assert_that(coords.shape).is_equal_to((X.shape[0], 2))
    coords2, _ = fit_projection(X, _cfg(projection_method="densmap"))
    np.testing.assert_array_equal(coords, coords2)


def test_pacmap_projection_when_installed():
    pytest.importorskip("pacmap")  # positive path: exercised by the CI extras job
    X = _blobs()
    coords, _reducer = fit_projection(X, _cfg(projection_method="pacmap"))
    assert_that(coords.shape).is_equal_to((X.shape[0], 2))
    assert_that(coords.dtype).is_equal_to(np.float32)


def test_pacmap_without_the_package_raises_projection_unavailable():
    import importlib.util

    if importlib.util.find_spec("pacmap") is not None:
        pytest.skip("pacmap is installed — the unavailable-path test does not apply")
    with pytest.raises(ProjectionMethodUnavailable) as exc:
        fit_projection(_blobs(), _cfg(projection_method="pacmap"))
    assert_that(exc.value.method).is_equal_to(
        "pacmap"
    )  # method-named, clear error (no silent fallback)


def test_fit_projection_is_deterministic_for_a_fixed_seed():
    X = _blobs()
    a, _ = fit_projection(X, _cfg())
    b, _ = fit_projection(X, _cfg())
    np.testing.assert_array_equal(a, b)


def test_tiny_input_uses_the_fallback_identity_reducer():
    X = _blobs(per=2, d=6)  # n=6 < max(umap_n_neighbors,4) → tiny path
    coords, reducer = fit_projection(X, _cfg())
    assert_that(coords.shape).is_equal_to((X.shape[0], 2))
    # the identity reducer keeps the signature stable and is itself reusable
    assert_that(transform_projection(reducer, X[:3]).shape).is_equal_to((3, 2))


def test_procrustes_disparity_is_none_on_degenerate_input_and_never_raises():
    # < 2 points
    assert_that(procrustes_disparity(np.zeros((1, 2)), np.zeros((1, 2)))).is_none()
    # shape mismatch
    assert_that(procrustes_disparity(np.zeros((3, 2)), np.zeros((2, 2)))).is_none()
    # constant → scipy raises → None
    assert_that(procrustes_disparity(np.zeros((3, 2)), np.zeros((3, 2)))).is_none()
    assert_that(
        procrustes_disparity(np.arange(6).reshape(3, 2), np.arange(6).reshape(3, 2))
    ).is_close_to(0.0, tolerance=1e-9)


def test_unknown_projection_method_raises_value_error():
    with pytest.raises(ValueError, match=r"projection_method must be one of"):
        fit_projection(_blobs(), _cfg(projection_method="tsne"))
