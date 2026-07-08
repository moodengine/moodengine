"""Unit tests for the metric adapter — moodengine.adapt.fit_supcon_projection / apply_projection.

fit_* needs torch (it trains a tiny real nn.Linear) and skips cleanly on a light install
(``importorskip``). Pins:
apply_projection is PURE NUMPY (torch never imported by it — subprocess-isolated), guards dim_in, and
L2-normalizes; a SupCon fit is deterministic at fixed seed (np.allclose) and RAISES the cosine
silhouette of the mood classes above the input space (same-mood closer, different-mood farther) — the
whole point of the adapter — measured + logged, for both 'supcon' and 'triplet'. Persistence
(projection_state/projection_from_state round-trip, save_projection/load_projection npz files, and
the validation of hand-built states) is covered torch-free on numpy-built projections.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from assertpy import assert_that
from sklearn.metrics import silhouette_score

from moodengine.adapt import (
    Projection,
    apply_projection,
    fit_supcon_projection,
    load_projection,
    projection_from_state,
    projection_state,
    save_projection,
)

logger = logging.getLogger(__name__)


def _blobs(seed: int = 0, n_classes: int = 3, per: int = 25, d: int = 16, spread: float = 0.55):
    """Separable-but-overlapping unit-sphere blobs: random class centroids + Gaussian jitter, each row
    re-normalized (frozen-CLAP-like). ``spread`` gives SupCon real room to tighten the classes."""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_classes, d)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    X, y = [], []
    for c in range(n_classes):
        pts = centroids[c] + spread * rng.standard_normal((per, d)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        X.append(pts)
        y += [c] * per
    return np.vstack(X).astype(np.float32), np.array(y)


def test_apply_projection_is_pure_numpy_and_l2_normalizes():
    W = np.eye(4, dtype=np.float32)
    proj = Projection(W=W, dim_in=4, dim_out=4, method="supcon", mood_names=["a", "b"])
    X = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    Z = apply_projection(proj, X)
    # Identity W → just L2-normalized rows: the [3,0,0,0] row becomes a unit vector; the zero row stays 0.
    assert_that(bool(np.allclose(Z[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6))).is_true()
    assert_that(bool(np.allclose(np.linalg.norm(Z[0]), 1.0, atol=1e-6))).is_true()
    assert_that(
        bool(np.allclose(Z[1], 0.0))
    ).is_true()  # zero vector is safe (eps guard), never NaN


def test_apply_projection_empty_and_dim_guard():
    proj = Projection(
        W=np.eye(4, dtype=np.float32), dim_in=4, dim_out=4, method="supcon", mood_names=[]
    )
    assert_that(apply_projection(proj, np.zeros((0, 4), dtype=np.float32)).shape).is_equal_to(
        (0, 4)
    )
    with pytest.raises(ValueError, match=r"projection dim_in"):
        apply_projection(proj, np.zeros((3, 5), dtype=np.float32))  # dim_in mismatch


def test_apply_projection_never_imports_torch():
    """Subprocess isolation (torch-free invariant): import the module + run apply_projection with a
    numpy-built Projection and assert torch was NEVER imported. Only fit_* may touch torch."""
    code = textwrap.dedent(
        """
        import sys, numpy as np
        from assertpy import assert_that
        from moodengine.adapt import Projection, apply_projection
        proj = Projection(W=np.eye(4, dtype=np.float32), dim_in=4, dim_out=4,
                          method="supcon", mood_names=[])
        out = apply_projection(proj, np.ones((5, 4), dtype=np.float32))
        assert_that(out.shape).is_equal_to((5, 4))
        assert_that(sys.modules).does_not_contain("torch")  # apply_projection must not import torch
        print("OK")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as(f"stdout={r.stdout!r} stderr={r.stderr!r}").is_equal_to(
        0
    )
    assert_that(r.stdout).contains("OK")


def test_fit_supcon_is_deterministic_at_fixed_seed():
    pytest.importorskip("torch")
    X, y = _blobs(seed=1)
    a = fit_supcon_projection(X, y, ["m0", "m1", "m2"], epochs=20, seed=7)
    b = fit_supcon_projection(X, y, ["m0", "m1", "m2"], epochs=20, seed=7)
    assert_that(a.W.shape).is_equal_to((16, 16))
    assert_that(a.dim_in).is_equal_to(16)
    assert_that(a.dim_out).is_equal_to(16)
    assert_that(a.method).is_equal_to("supcon")
    assert_that(
        bool(np.allclose(a.W, b.W, atol=1e-5))
    ).is_true()  # same seed → byte-comparable weights


@pytest.mark.parametrize("method", ["supcon", "triplet"])
def test_projection_raises_cosine_silhouette(method: str):
    """The defining property of the metric adapter: after projecting, the mood classes are MORE
    separable (higher cosine silhouette) than in the input CLAP space. Measured + logged, not asserted
    blind — a projection that did nothing (identity) would fail this."""
    pytest.importorskip("torch")
    X, y = _blobs(seed=2)
    sil_in = float(silhouette_score(X, y, metric="cosine"))
    proj = fit_supcon_projection(X, y, ["m0", "m1", "m2"], method=method, epochs=80, seed=0)
    Z = apply_projection(proj, X)
    sil_out = float(silhouette_score(Z, y, metric="cosine"))
    logger.info(
        "silhouette[%s]: input=%.4f projected=%.4f (Δ=%+.4f)",
        method,
        sil_in,
        sil_out,
        sil_out - sil_in,
    )
    assert_that(Z.shape).is_equal_to(X.shape)
    assert_that(sil_out).described_as(
        f"{method}: projected silhouette {sil_out:.4f} !> input {sil_in:.4f}"
    ).is_greater_than(sil_in)


def test_fit_rejects_degenerate_input():
    pytest.importorskip("torch")
    X, _ = _blobs(seed=3)
    with pytest.raises(ValueError, match=r"2 distinct classes"):
        fit_supcon_projection(X, np.zeros(X.shape[0], dtype=int), ["m0"], epochs=5)  # 1 class
    with pytest.raises(ValueError, match=r"at least 2 training examples"):
        fit_supcon_projection(X[:1], np.array([0]), ["m0"], epochs=5)  # n < 2


def _rect_projection(seed: int = 0, dim_in: int = 5, dim_out: int = 3) -> Projection:
    """Torch-free non-square projection: persistence must not assume dim_in == dim_out."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((dim_out, dim_in)).astype(np.float32)
    return Projection(W=W, dim_in=dim_in, dim_out=dim_out, method="supcon", mood_names=["a", "b"])


def test_projection_state_round_trip_is_byte_identical():
    proj = _rect_projection(seed=1)

    restored = projection_from_state(projection_state(proj))

    assert_that(restored.dim_in).is_equal_to(proj.dim_in)
    assert_that(restored.dim_out).is_equal_to(proj.dim_out)
    assert_that(restored.method).is_equal_to(proj.method)
    assert_that(restored.mood_names).is_equal_to(proj.mood_names)
    np.testing.assert_array_equal(restored.W, proj.W, strict=True)  # bytes AND dtype


def test_projection_state_round_trip_keeps_empty_mood_names():
    proj = Projection(
        W=np.eye(4, dtype=np.float32), dim_in=4, dim_out=4, method="triplet", mood_names=[]
    )

    restored = projection_from_state(projection_state(proj))

    assert_that(restored.mood_names).is_equal_to([])
    assert_that(restored.method).is_equal_to("triplet")


def test_projection_round_trip_preserves_apply_outputs():
    proj = _rect_projection(seed=2)
    X = np.random.default_rng(3).standard_normal((7, proj.dim_in)).astype(np.float32)
    before = apply_projection(proj, X)

    after = apply_projection(projection_from_state(projection_state(proj)), X)

    np.testing.assert_array_equal(after, before)  # same W bytes -> same projected bytes


def test_supcon_fitted_projection_survives_state_round_trip():
    pytest.importorskip("torch")
    X, y = _blobs(seed=4)
    proj = fit_supcon_projection(X, y, ["m0", "m1", "m2"], dim_out=8, epochs=10, seed=0)

    restored = projection_from_state(projection_state(proj))

    assert_that(restored.dim_out).is_equal_to(8)  # non-square survives
    assert_that(restored.dim_in).is_equal_to(16)
    np.testing.assert_array_equal(apply_projection(restored, X), apply_projection(proj, X))


@pytest.mark.parametrize("filename", ["proj.npz", "proj.adapter"])
def test_save_projection_writes_exact_path_and_loads_back(tmp_path, filename):
    # A suffix-less filename pins the open-handle contract: np.savez alone would append ".npz".
    proj = _rect_projection(seed=5)
    target = tmp_path / filename

    written = save_projection(proj, target)
    loaded = load_projection(written)

    assert_that(written).is_equal_to(target)
    assert_that(target.exists()).is_true()
    assert_that((tmp_path / f"{filename}.npz").exists()).is_false()
    np.testing.assert_array_equal(loaded.W, proj.W, strict=True)
    assert_that(loaded.mood_names).is_equal_to(proj.mood_names)


def test_load_projection_missing_path_raises_file_not_found(tmp_path):
    missing = tmp_path / "nowhere.npz"

    with pytest.raises(FileNotFoundError, match=re.escape(f"not found: {missing}")):
        load_projection(missing)


def test_projection_from_state_missing_key_raises():
    state = projection_state(_rect_projection(seed=6))
    del state["dim_in"]

    with pytest.raises(ValueError, match=r"missing keys \['dim_in'\]"):
        projection_from_state(state)


def test_projection_from_state_wrong_schema_raises():
    state = projection_state(_rect_projection(seed=7))
    state["schema"] = np.array("moodengine.probe/1")  # a probe state is NOT a projection state

    with pytest.raises(ValueError, match=r"expected.*moodengine\.projection/1"):
        projection_from_state(state)


def test_projection_from_state_shape_mismatch_raises():
    state = projection_state(_rect_projection(seed=8))
    state["W"] = state["W"][:, :-1]  # W no longer (dim_out, dim_in)

    with pytest.raises(ValueError, match=r"shape.*dim_out, dim_in"):
        projection_from_state(state)
