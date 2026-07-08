"""Unit tests for moodengine.journey — SLERP geodesic + opt-in OT morph. Torch-free, deterministic.

SLERP is pinned by its defining property: the angle from ``a`` grows at constant angular velocity, so
``cos(waypoint_t, a)`` is non-increasing and ``cos(waypoint_t, b)`` non-decreasing along the path, the
endpoints are exact, and every waypoint is a unit vector. The OT morph is tested only when POT is
installed (``importorskip``) — the module import itself must never require it."""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.journey import path_between


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _two_dirs(seed: int, d: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(d)), _unit(rng.standard_normal(d))


# --------------------------------------------------------------------------- #
# path_between (SLERP)
# --------------------------------------------------------------------------- #
def test_path_between_endpoints_are_exact() -> None:
    a, b = _two_dirs(0)
    wp = path_between(a, b, n=8)
    assert_that(wp.shape).is_equal_to((8, a.shape[0]))
    np.testing.assert_allclose(wp[0], a, atol=1e-5)
    np.testing.assert_allclose(wp[-1], b, atol=1e-5)


def test_path_between_all_waypoints_are_unit() -> None:
    a, b = _two_dirs(1)
    wp = path_between(a, b, n=10)
    np.testing.assert_allclose(np.linalg.norm(wp, axis=1), 1.0, atol=1e-5)


def test_path_between_is_a_monotone_geodesic() -> None:
    a, b = _two_dirs(2)
    wp = path_between(a, b, n=12)
    cos_a = wp @ a
    cos_b = wp @ b
    assert_that(bool(np.all(np.diff(cos_a) <= 1e-5))).is_true()  # angle from a grows
    assert_that(bool(np.all(np.diff(cos_b) >= -1e-5))).is_true()  # angle from b shrinks
    # SLERP's DEFINING property — constant angular velocity: arccos(wp·a) increases in EQUAL steps.
    # This uniquely separates SLERP from a normalized lerp, which traces the SAME great-circle arc but
    # bunches waypoints toward the endpoints (uneven mood-morph spacing) — so an nlerp regression that
    # still satisfies every monotonicity/endpoint/unit check is caught right here.
    theta = np.arccos(np.clip(cos_a, -1.0, 1.0))
    assert_that(bool(np.allclose(np.diff(theta), theta[-1] / (len(wp) - 1), atol=1e-4))).is_true()
    # symmetric arc: the odd-n midpoint is cos-equidistant from both endpoints
    m = path_between(a, b, n=9)[4]
    assert_that(float(m @ a)).is_close_to(float(m @ b), tolerance=1e-4)


def test_path_between_antipode_is_finite() -> None:
    # The a≈−b degenerate of the colinear fallback: the lerp passes through the origin, so the odd-n
    # midpoint is the zero vector — but the eps floor in l2_normalize keeps it FINITE (no NaN), and every
    # other waypoint stays unit. Pins the documented antipode carve-out.
    a = _unit(np.random.default_rng(7).standard_normal(12))
    for n in (5, 6):
        wp = path_between(a, -a, n=n)
        assert_that(bool(np.all(np.isfinite(wp)))).is_true()
        norms = np.linalg.norm(wp, axis=1)
        np.testing.assert_allclose(
            norms[norms > 1e-6], 1.0, atol=1e-5
        )  # all non-zero rows are unit


def test_path_between_colinear_falls_back_to_lerp() -> None:
    a = _unit(np.random.default_rng(3).standard_normal(16))
    wp = path_between(a, a.copy(), n=6)  # identical directions → Ω ≈ 0
    assert_that(bool(np.all(np.isfinite(wp)))).is_true()
    np.testing.assert_allclose(np.linalg.norm(wp, axis=1), 1.0, atol=1e-5)
    for row in wp:
        np.testing.assert_allclose(row, a, atol=1e-4)  # every waypoint collapses to a


def test_path_between_normalizes_non_unit_inputs() -> None:
    a, b = _two_dirs(4)
    wp = path_between(3.0 * a, -0.5 * b + 0.0, n=5)  # scaled inputs
    np.testing.assert_allclose(np.linalg.norm(wp, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(wp[0], a, atol=1e-5)  # direction of 3a is a


def test_path_between_is_deterministic() -> None:
    a, b = _two_dirs(5)
    assert_that(bool(np.array_equal(path_between(a, b, n=7), path_between(a, b, n=7)))).is_true()


def test_path_between_n_guards() -> None:
    a, b = _two_dirs(6)
    assert_that(path_between(a, b, n=0).shape).is_equal_to((0, a.shape[0]))
    assert_that(path_between(a, b, n=-3).shape).is_equal_to((0, a.shape[0]))
    one = path_between(a, b, n=1)
    assert_that(one.shape).is_equal_to((1, a.shape[0]))
    np.testing.assert_allclose(one[0], a, atol=1e-5)


def test_journey_module_is_torch_free() -> None:
    import subprocess
    import sys

    code = (
        "import sys, numpy as np, moodengine.journey as j; "
        "a=np.random.default_rng(0).standard_normal(16).astype('float32'); "
        "b=np.random.default_rng(1).standard_normal(16).astype('float32'); "
        "wp=j.path_between(a, b, n=6); "
        "assert wp.shape==(6,16); "
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')]; "
        "sys.exit('torch loaded: '+repr(bad)) if bad else None"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).is_equal_to(0)


# --------------------------------------------------------------------------- #
# ot_morph (opt-in — only when POT is installed)
# --------------------------------------------------------------------------- #
def test_ot_morph_shape_dedup_and_direction() -> None:
    pytest.importorskip("ot")  # POT is an opt-in extra; skip cleanly when absent
    from moodengine.journey import ot_morph

    rng = np.random.default_rng(9)
    X = (rng.standard_normal((60, 24))).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    a, b = _unit(rng.standard_normal(24)), _unit(rng.standard_normal(24))
    files = [f"t{i}" for i in range(60)]

    idxs = ot_morph(a, b, X, files, n=8)
    assert_that(len(idxs)).is_equal_to(len(set(idxs)))  # distinct row indices
    assert_that(len(set(idxs))).is_less_than_or_equal_to(8)  # capped at n
    assert_that(all(0 <= i < 60 for i in idxs)).is_true()
    # A→B progression: every pick is at least as B-ward as the previous one (monotonic by construction).
    sims_b = X @ b
    seq = [float(sims_b[i]) for i in idxs]
    assert_that(all(seq[k] <= seq[k + 1] + 1e-6 for k in range(len(seq) - 1))).is_true()
    assert_that(ot_morph(a, b, X, files, n=8)).is_equal_to(idxs)  # deterministic


def test_ot_morph_raises_importerror_without_pot(monkeypatch) -> None:
    # Force the lazy `import ot` to fail and assert ot_morph surfaces ImportError (callers can map it).
    import builtins

    real_import = builtins.__import__

    def _no_ot(name, *args, **kwargs):
        if name == "ot" or name.startswith("ot."):
            raise ImportError("POT not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ot)
    from moodengine.journey import ot_morph

    a = _unit(np.arange(8, dtype=np.float32))
    X = np.eye(8, dtype=np.float32)
    with pytest.raises(ImportError, match=r"ot_morph requires POT"):
        ot_morph(a, a, X, [f"t{i}" for i in range(8)], n=4)
