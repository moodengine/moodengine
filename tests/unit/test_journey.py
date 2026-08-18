"""Unit tests for moodengine.journey — SLERP geodesic + opt-in OT morph. Torch-free, deterministic.

SLERP is pinned by its defining property: the angle from ``a`` grows at constant angular velocity, so
``cos(waypoint_t, a)`` is non-increasing and ``cos(waypoint_t, b)`` non-decreasing along the path, the
endpoints are exact, and every waypoint is a unit vector. The OT morph is tested only when POT is
installed (``importorskip``) — the module import itself must never require it."""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine import journey
from moodengine.journey import journey_tracks, path_between, smooth_order


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
    # SLERP's DEFINING property — constant angular velocity: the cosine from a follows cos(t·Ω)
    # for EQUALLY spaced t over [0, 1]. Assert on the cosines directly, NOT on arccos(cos_a):
    # arccos is ill-conditioned near cos = 1 (the endpoints), where the float32 storage of the
    # waypoints amplifies rounding to ~1e-3 and flips across BLAS/CPU builds (a lowest-deps CI
    # flake). The cosine form is well-conditioned and still uniquely separates SLERP from a
    # normalized lerp, which traces the SAME great-circle arc but bunches waypoints toward the
    # endpoints (uneven mood-morph spacing) — an nlerp regression is caught right here.
    omega = float(np.arccos(np.clip(float(a @ b), -1.0, 1.0)))
    expected_cos_a = np.cos(np.linspace(0.0, 1.0, len(wp)) * omega)
    np.testing.assert_allclose(cos_a, expected_cos_a, atol=1e-4)
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


def _path_cost(X, order):
    Y = X[list(order)]
    return float(sum(1.0 - float(Y[i] @ Y[i + 1]) for i in range(len(order) - 1)))


def test_journey_tracks_returns_indices_like_the_other_mode() -> None:
    """The asymmetry this closes: `ot_morph` returned row indices while `path_between` returned
    waypoint VECTORS and stopped, so the SLERP mode produced no playlist at all — the module
    docstring described waypoint→track selection but nothing implemented it."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    picks = journey_tracks(X, X[0], X[-1], n=8)

    assert_that(picks).is_length(8)
    assert_that(len(set(picks))).is_equal_to(8)  # a journey must not stall on one track
    assert_that(all(isinstance(i, int) and 0 <= i < 40 for i in picks)).is_true()


def test_journey_tracks_starts_at_the_a_pole() -> None:
    """Every pick sits ON the geodesic, so the first waypoint is the A direction itself and the
    nearest track to it opens the journey."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    picks = journey_tracks(X, X[3], X[20], n=5)

    assert_that(picks[0]).is_equal_to(3)


def test_journey_tracks_pool_smaller_than_requested_is_truncated() -> None:
    """Fewer tracks than waypoints returns what exists rather than repeating picks."""
    X = np.eye(3, dtype=np.float32)

    assert_that(journey_tracks(X, X[0], X[2], n=10)).is_length(3)
    assert_that(journey_tracks(np.empty((0, 4), dtype=np.float32), X[0], X[2], n=5)).is_empty()


def test_journey_tracks_rejects_an_unknown_mode() -> None:
    X = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match=r"unknown journey mode 'nope'"):
        journey_tracks(X, X[0], X[1], n=2, mode="nope")


def test_smooth_order_is_exactly_optimal_below_the_dp_cutoff() -> None:
    """Held-Karp is exact, so below the cutoff the result must equal a brute-force minimum — the
    only assertion that distinguishes a correct DP from a plausible heuristic."""
    import itertools

    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    order = smooth_order(X)
    brute = min(itertools.permutations(range(8)), key=lambda t: _path_cost(X, t))

    assert_that(_path_cost(X, order)).is_close_to(_path_cost(X, brute), tolerance=1e-9)
    assert_that(sorted(order)).is_equal_to(list(range(8)))  # a permutation, nothing dropped


def test_smooth_order_beats_the_input_order() -> None:
    """The objective, stated: consecutive tracks end up more similar than they started."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 12)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    order = smooth_order(X)

    assert_that(_path_cost(X, order)).is_less_than(_path_cost(X, range(20)))


def test_smooth_order_two_opt_never_degrades_its_seed() -> None:
    """Above the exact cutoff the answer is a heuristic, so the guarantee that remains is
    monotonicity: 2-opt only accepts a reversal that lowers the cost, so it can never return
    something worse than the nearest-neighbour tour it started from."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((25, 10)).astype(np.float32)  # > the 12-track exact cutoff
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    order = smooth_order(X)
    greedy = journey._nn_tour((1.0 - X @ X.T).astype(np.float64), order[0])

    assert_that(_path_cost(X, order)).is_less_than_or_equal_to(_path_cost(X, greedy) + 1e-9)


def test_smooth_order_honours_a_pinned_start() -> None:
    """A caller with an opening track in mind must get it, in both the exact and heuristic paths."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((16, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    assert_that(smooth_order(X[:9], start=4)[0]).is_equal_to(4)  # exact path
    assert_that(smooth_order(X, start=7)[0]).is_equal_to(7)  # 2-opt path


def test_smooth_order_degenerate_inputs_are_returned_as_is() -> None:
    """Nothing to reorder below three tracks — an identity, never a raise."""
    assert_that(smooth_order(np.empty((0, 4), dtype=np.float32))).is_empty()
    assert_that(smooth_order(np.eye(2, dtype=np.float32))).is_equal_to([0, 1])
