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


def test_smooth_order_pinned_start_is_optimal_among_paths_opening_there() -> None:
    """Pinning must constrain the search, not weaken it: the answer is still the exact minimum
    over every path that opens at `start`. Guards the single-pass DP — seeding one origin instead
    of all of them has to keep Held-Karp exact."""
    import itertools

    rng = np.random.default_rng(6)
    X = rng.standard_normal((8, 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rest = [i for i in range(8) if i != 5]

    order = smooth_order(X, start=5)
    brute = min(([5, *p] for p in itertools.permutations(rest)), key=lambda t: _path_cost(X, t))

    assert_that(_path_cost(X, order)).is_close_to(_path_cost(X, brute), tolerance=1e-9)


class _CountingCost(np.ndarray):
    """A cost matrix that tallies its own reads, so DP work is observable without a clock.

    Counting `cost` lookups is what distinguishes ONE subset DP seeded with every origin from
    one DP re-run per origin. A wall-clock assertion would measure the same thing far less
    reliably, and counting calls to `_held_karp_path` measures only the caller — the per-origin
    loop lived INSIDE it.
    """

    reads = 0

    def __getitem__(self, key):
        type(self).reads += 1
        return super().__getitem__(key)


def _dp_reads(cost: np.ndarray, starts: list[int]) -> int:
    counting = cost.copy().view(_CountingCost)
    _CountingCost.reads = 0
    journey._held_karp_path(counting, starts)
    return _CountingCost.reads


def test_held_karp_seeds_every_origin_into_one_dp_pass() -> None:
    """Free start must cost ONE pass, not one per origin.

    Re-running the whole DP per origin multiplied the documented `O(2^n * n^2)` by n on the
    DEFAULT free-start path. Measured on this input: the shipped single pass reads the cost
    matrix 2.65x more for a free start than for a pinned one, while the per-origin shape read
    it 8.00x more — exactly n. The threshold sits between the two, so reverting the seeding
    fails this test.
    """
    n = 8
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cost = (1.0 - X @ X.T).astype(np.float64)
    np.fill_diagonal(cost, 0.0)

    free = _dp_reads(cost, list(range(n)))
    pinned = _dp_reads(cost, [0])

    assert_that(free).is_less_than(4 * pinned)  # one pass ~2.65x; per-origin would be ~n = 8x


def _clustered(n: int, d: int, seed: int) -> np.ndarray:
    """Clustered unit rows — 2-opt only finds reversals when the geometry has structure."""
    rng = np.random.default_rng(seed)
    k = max(2, n // 12)
    X = rng.standard_normal((k, d))[rng.integers(0, k, n)] + 0.15 * rng.standard_normal((n, d))
    X = X.astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


#: The exact permutations the heuristic path returns above the DP cutoff. 2-opt here is
#: FIRST-improvement — it applies a reversal the moment it finds one and keeps scanning the
#: mutated tour — and a best-improvement variant converges to a different permutation of equal
#: or better cost. Nothing else in this file distinguishes the two: the whole suite passes
#: against a best-improvement sweep. These pin the variant, so a rewrite has to reproduce the
#: scan order rather than merely the objective.
_GOLDEN_ORDER_60 = [
    56,
    20,
    50,
    39,
    26,
    57,
    38,
    15,
    13,
    53,
    46,
    35,
    14,
    49,
    0,
    42,
    4,
    25,
    36,
    16,
    17,
    41,
    32,
    48,
    54,
    7,
    34,
    31,
    47,
    43,
    23,
    44,
    29,
    11,
    37,
    27,
    40,
    8,
    52,
    33,
    21,
    58,
    30,
    10,
    12,
    6,
    1,
    55,
    45,
    18,
    28,
    3,
    22,
    9,
    59,
    19,
    51,
    24,
    2,
    5,
]
_GOLDEN_ORDER_200 = [
    107,
    49,
    100,
    79,
    118,
    93,
    105,
    131,
    154,
    8,
    78,
    73,
    169,
    10,
    187,
    139,
    168,
    110,
    42,
    29,
    91,
    13,
    167,
    47,
    156,
    33,
    180,
    41,
    112,
    127,
    2,
    115,
    18,
    145,
    22,
    17,
    4,
    26,
    191,
    96,
    175,
    182,
    130,
    36,
    102,
    83,
    21,
    166,
    159,
    193,
    126,
    151,
    92,
    89,
    178,
    9,
    185,
    65,
    177,
    153,
    50,
    46,
    37,
    81,
    25,
    90,
    31,
    103,
    140,
    148,
    108,
    52,
    144,
    181,
    164,
    150,
    75,
    63,
    80,
    88,
    114,
    53,
    43,
    109,
    143,
    120,
    68,
    72,
    95,
    7,
    194,
    56,
    138,
    129,
    70,
    121,
    85,
    71,
    137,
    133,
    195,
    45,
    69,
    101,
    173,
    132,
    51,
    5,
    147,
    27,
    28,
    94,
    161,
    170,
    106,
    66,
    62,
    165,
    123,
    158,
    1,
    57,
    59,
    157,
    61,
    35,
    40,
    84,
    55,
    14,
    97,
    58,
    38,
    179,
    44,
    23,
    122,
    141,
    67,
    146,
    199,
    76,
    3,
    48,
    64,
    174,
    99,
    136,
    172,
    163,
    111,
    124,
    192,
    184,
    77,
    155,
    30,
    162,
    189,
    54,
    104,
    34,
    0,
    171,
    15,
    82,
    128,
    197,
    6,
    32,
    160,
    119,
    134,
    11,
    125,
    188,
    198,
    87,
    98,
    117,
    12,
    19,
    39,
    116,
    60,
    135,
    190,
    149,
    142,
    20,
    152,
    16,
    113,
    196,
    24,
    176,
    86,
    74,
    186,
    183,
]


def test_smooth_order_returns_the_pinned_first_improvement_permutation() -> None:
    """The heuristic path is a specific 2-opt variant, and this is the permutation it returns."""
    assert_that(smooth_order(_clustered(60, 16, 11))).is_equal_to(_GOLDEN_ORDER_60)
    assert_that(smooth_order(_clustered(200, 16, 11))).is_equal_to(_GOLDEN_ORDER_200)


def test_cheapest_nn_start_breaks_an_exact_tie_toward_the_smaller_index() -> None:
    """Two starts with identical tour cost: the smaller index wins, and it is not index 0."""
    cost = np.zeros((4, 4))
    cost[0, 1] = cost[1, 0] = cost[0, 2] = cost[2, 0] = cost[0, 3] = cost[3, 0] = 0.1
    cost[1, 2] = cost[2, 1] = 0.3
    cost[1, 3] = cost[3, 1] = cost[2, 3] = cost[3, 2] = 0.2
    totals = [journey._tour_cost(cost, journey._nn_tour(cost, s)) for s in range(4)]

    winner = journey._cheapest_nn_start(cost)

    assert_that(totals[1]).is_equal_to(totals[2])  # the tie is real, not assumed
    assert_that(min(totals[0], totals[3])).is_greater_than(totals[1])  # and they are the cheapest
    assert_that(winner).is_equal_to(1)


def test_smooth_order_rejects_a_start_outside_the_row_range() -> None:
    """Refused, not reinterpreted. Out of range used to mean "choose freely" below the exact
    cutoff and "pin row 0" above it, so one typo gave two different playlists by input size."""
    X = np.eye(6, dtype=np.float32)

    with pytest.raises(ValueError, match=r"start must be a row index in \[0, 6\); got 99"):
        smooth_order(X, start=99)
    with pytest.raises(ValueError, match=r"start must be a row index in \[0, 6\); got -1"):
        smooth_order(X, start=-1)


def test_smooth_order_two_tracks_still_honour_a_pinned_start() -> None:
    """`n <= 2` has nothing to optimize, but which of the two OPENS is still the caller's call —
    the early return used to discard `start` before reading it."""
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert_that(smooth_order(X, start=1)).is_equal_to([1, 0])
    assert_that(smooth_order(X, start=0)).is_equal_to([0, 1])
    assert_that(smooth_order(X)).is_equal_to([0, 1])


def test_smooth_order_degenerate_inputs_are_returned_as_is() -> None:
    """Nothing to reorder below three tracks — an identity, never a raise."""
    assert_that(smooth_order(np.empty((0, 4), dtype=np.float32))).is_empty()
    # An empty input returns before `start` is validated: sizing a degenerate call never raises.
    assert_that(smooth_order(np.empty((0, 4), dtype=np.float32), start=0)).is_empty()
    assert_that(smooth_order(np.eye(2, dtype=np.float32))).is_equal_to([0, 1])
