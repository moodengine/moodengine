"""Ambient-journey path construction over the CLAP mood space.

Pure numpy (+ optional POT for the opt-in optimal-transport mode). Given two mood directions from the
label matrix, build an ordered sequence that MORPHS from mood A to mood B:

  * :func:`path_between` — the SLERP geodesic (Shoemake 1985): ``n`` unit waypoints on the shortest
    spherical arc between the two mood directions. Selecting the nearest-unused track at each waypoint
    keeps every pick ON the path, so the displayed valence/energy ramp is a real *consequence*, not a
    fabricated target.
  * :func:`ot_morph` (opt-in) — orders a neighbourhood of tracks A→B via an entropic optimal-transport
    plan (Sinkhorn, Cuturi 2013). Requires POT; the import is LAZY so the SLERP mode works without it.

Deterministic; torch-free (the deep-learning stack is never imported here).
"""

from __future__ import annotations

import numpy as np

from moodengine.exceptions import MissingDependencyError
from moodengine.pooling import l2_normalize


def path_between(v_a: np.ndarray, v_b: np.ndarray, n: int = 8, *, eps: float = 1e-8) -> np.ndarray:
    """``(n, d)`` unit waypoints on the SLERP geodesic between mood directions ``v_a`` and ``v_b``.

    Spherical linear interpolation (Shoemake 1985): with ``Ω = arccos(clip(â·b̂, −1, 1))``,

        ``slerp(t) = sin((1−t)·Ω)/sin Ω · â + sin(t·Ω)/sin Ω · b̂``,   ``t`` in ``n`` steps over ``[0, 1]``.

    The endpoints are exact (``[0] ≈ â``, ``[-1] ≈ b̂``). When the two directions are (nearly) colinear
    (``sin Ω ≈ 0``) it falls back to a normalized linear interpolation. Inputs are L2-normalized first;
    every returned row is a unit vector. Pure numpy, deterministic. ``n <= 0`` → an empty ``(0, d)`` array;
    ``n == 1`` → just ``[â]``.
    """
    a = l2_normalize(np.asarray(v_a, dtype=np.float32).reshape(-1), axis=-1)
    b = l2_normalize(np.asarray(v_b, dtype=np.float32).reshape(-1), axis=-1)
    d = a.shape[0]
    n = int(n)
    if n <= 0:
        return np.zeros((0, d), dtype=np.float32)
    if n == 1:
        return a[None, :].astype(np.float32)

    ts = np.linspace(0.0, 1.0, n, dtype=np.float64)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = float(np.arccos(dot))
    sin_o = float(np.sin(omega))
    if sin_o < eps:  # (nearly) colinear → normalized linear interpolation (SLERP is undefined here)
        pts = (1.0 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]
        return l2_normalize(pts.astype(np.float32), axis=1)

    wa = np.sin((1.0 - ts) * omega) / sin_o
    wb = np.sin(ts * omega) / sin_o
    pts = wa[:, None] * a[None, :] + wb[:, None] * b[None, :]
    return l2_normalize(pts.astype(np.float32), axis=1)


def _softmax(z: np.ndarray, tau: float) -> np.ndarray:
    """Temperature softmax → a strictly-positive probability simplex; shift-invariant, so it survives
    all-negative cosines without collapsing to uniform (unlike a clip-to-0 normalization)."""
    z = np.asarray(z, dtype=np.float64) / max(float(tau), 1e-12)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, descending, deterministic (stable tie-break)."""
    k = min(int(k), scores.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def ot_morph(
    v_a: np.ndarray,
    v_b: np.ndarray,
    X: np.ndarray,
    filenames: list[str],
    n: int = 8,
    *,
    reg: float = 0.05,
) -> list[int]:
    """Up to ``n`` DISTINCT row indices of ``X``, ordered A→B by an entropic optimal-transport plan

    .. note::
       ``filenames`` is accepted and never read — the return is row POSITIONS, not names. It is
       kept for signature stability and should be removed; pass ``[]``.

    (opt-in). Requires POT (``import ot`` is LAZY → ``ImportError`` when it isn't installed, which the
    caller can catch to degrade gracefully). Pure numpy + POT, deterministic.

    We transport ``n`` *ordered journey slots* (a uniform source) to the candidate pool (a fidelity-shaped
    target) IN THE 2-D ``(sim_a, sim_b)`` plane: slot ``k`` has an ideal point that fades from A-affinity
    to B-affinity, and the Sinkhorn plan assigns each slot the pool track that best fills it under the
    squared-Euclidean cost in that plane. Keeping the cost 2-D is what makes the transport *load-bearing*
    rather than a reg-smoothed 1-D ``sim_b`` sort: the perpendicular "on-arc fidelity" axis (``sim_a +
    sim_b``) re-ranks off-arc decoys a 1-D key would accept, and the ``nu`` target marginal spreads the
    picks by density. The transport only chooses WHICH distinct tracks fill the journey; the returned
    picks are then ordered by B-affinity, so the A→B progression is monotonic by construction. (Design
    from a 3-way design panel + judge synthesis; robust to N<n, all-negative cosines, a≈b, empty pool.)
    """
    try:
        import ot  # noqa: PLC0415 — lazy on purpose: keeps SLERP mode POT-free
    except ImportError as exc:
        raise MissingDependencyError("ot_morph", "POT", "ot") from exc

    Xn = l2_normalize(np.asarray(X, dtype=np.float64), axis=1)
    a = l2_normalize(np.asarray(v_a, dtype=np.float64).reshape(-1), axis=-1)
    b = l2_normalize(np.asarray(v_b, dtype=np.float64).reshape(-1), axis=-1)
    N = Xn.shape[0]
    n = max(1, int(n))
    if N == 0:
        return []

    sa = Xn @ a  # (N,) cosine to A (may be negative)
    sb = Xn @ b  # (N,) cosine to B
    # Pool = A-anchors ∪ B-anchors ∪ arc-middle (so disjoint A/B clusters still yield middle tracks).
    m = int(min(N, max(4 * n, 32)))
    pool = np.unique(np.concatenate([_topk(sa, m), _topk(sb, m), _topk(sa + sb, m)]))
    P = int(pool.shape[0])
    if P == 0:
        return []
    pa, pb = sa[pool], sb[pool]
    n_eff = int(min(n, P))
    if n_eff == 1:
        return [int(pool[int(np.argmax(pa + pb))])]
    if (
        float(np.linalg.norm(b - a)) < 1e-6
    ):  # a ≈ b: the A→B axis is undefined → skip OT, most on-arc
        return [int(pool[i]) for i in _topk(pa + pb, n_eff)]

    # n ordered ideal slots fading A→B, in the (sim_a, sim_b) plane; robust 5/95-pct endpoints.
    Q = np.stack([pa, pb], axis=1)  # (P, 2) each pool track's plane coordinate
    slot = (np.arange(n_eff) + 0.5) / n_eff
    a_hi, a_lo = np.quantile(pa, 0.95), np.quantile(pa, 0.05)
    b_hi, b_lo = np.quantile(pb, 0.95), np.quantile(pb, 0.05)
    S = np.stack(
        [
            (1.0 - slot) * a_hi + slot * a_lo,  # A-affinity fades
            (1.0 - slot) * b_lo + slot * b_hi,
        ],
        axis=1,
    )  # B-affinity grows → (n_eff, 2)
    cost = ((S[:, None, :] - Q[None, :, :]) ** 2).sum(-1)  # (n_eff, P) squared-Euclidean, ≥ 0
    cmax = float(cost.max())
    if cmax > 0.0:
        cost = cost / cmax  # scale to [0, 1] for Sinkhorn stability
    mu = np.full(n_eff, 1.0 / n_eff)  # uniform ORDERED slots (source)
    nu = _softmax(pa + pb, tau=0.1)  # fidelity-shaped pool target
    plan = np.asarray(
        ot.sinkhorn(mu, nu, cost, reg, numItermax=1000), dtype=np.float64
    )  # (n_eff, P)
    if not np.all(np.isfinite(plan)):  # Sinkhorn blow-up → 2-D nearest-ideal fallback
        plan = -cost

    used = np.zeros(P, dtype=bool)
    out: list[int] = []
    for k in range(n_eff):  # one DISTINCT track per ordered slot
        row = np.where(used, -np.inf, plan[k])
        j = int(np.argmax(row))  # first-max on ties → deterministic
        used[j] = True
        out.append(int(pool[j]))

    # Greedy per-slot assignment picks the right SET of tracks but does not by itself
    # guarantee they emerge ordered A→B, so make the promised progression explicit:
    # order the picks by B-affinity (original row index as a deterministic tie-break).
    out.sort(key=lambda t: (float(sb[t]), t))
    return out


def journey_tracks(
    X: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    n: int = 8,
    *,
    mode: str = "slerp",
    reg: float = 0.05,
) -> list[int]:
    """Up to ``n`` DISTINCT row indices of ``X``, ordered A→B — the same return shape for both modes.

    The two morph strategies disagreed on what they hand back: :func:`ot_morph` already returns row
    indices, while :func:`path_between` returns waypoint VECTORS and stops, leaving every caller to
    reinvent waypoint→track selection. The module docstring described that step ("selecting the
    nearest-unused track at each waypoint") but nothing implemented it, so the SLERP mode produced
    no playlist at all. This is that step, and it makes the two modes interchangeable.

    ``mode='slerp'`` walks the geodesic from :func:`path_between` and takes, at each waypoint, the
    nearest track not already used — so every pick sits ON the arc and the valence/energy ramp is a
    consequence of the path rather than a target imposed on it. ``mode='ot'`` delegates to
    :func:`ot_morph` (needs POT; raises :class:`~moodengine.exceptions.MissingDependencyError`
    when absent).

    ``X`` ``(m, d)`` are track embeddings, re-L2-normalized defensively so the ranking is cosine on
    any input. Returns fewer than ``n`` indices when the pool is smaller. Guards an empty pool and
    ``n <= 0`` with ``[]``. Deterministic; ties resolve to the lowest row index. Pure numpy in
    ``slerp`` mode.
    """
    Xn = l2_normalize(np.asarray(X, dtype=np.float32), axis=1)
    m = Xn.shape[0] if Xn.ndim == 2 else 0
    n = int(n)
    if m == 0 or n <= 0:
        return []
    if mode == "ot":
        # `filenames` is declared by `ot_morph` and never read (grep it) — the indices it returns
        # are row positions, not names. Passing [] rather than fabricating a list documents that
        # at the call site; the parameter itself should go, which is a separate change.
        return ot_morph(v_a, v_b, Xn, [], n, reg=reg)
    if mode != "slerp":
        raise ValueError(f"unknown journey mode {mode!r} (expected 'slerp' | 'ot')")

    waypoints = path_between(v_a, v_b, n)
    picks: list[int] = []
    used = np.zeros(m, dtype=bool)
    for point in waypoints:
        if used.all():
            break
        sims = Xn @ np.asarray(point, dtype=np.float32)
        sims[used] = -np.inf  # nearest UNUSED: a journey must not stall on one track
        choice = int(np.argmax(sims))
        picks.append(choice)
        used[choice] = True
    return picks


#: Above this many tracks the exact ordering is abandoned for 2-opt. Held-Karp is O(2^n · n²), so
#: 12 costs about 590k state transitions — measured at ~80 ms free-start on Apple Silicon, since
#: the DP is a plain Python loop — while 16 would cost ~17M and land in the tens of seconds. The
#: wall is steep and close, which is why the cutoff is a constant rather than a knob.
_EXACT_ORDER_MAX: int = 12


def smooth_order(X: np.ndarray, start: int | None = None) -> list[int]:
    """Order tracks so CONSECUTIVE ones are as similar as possible — the missing sequencing objective.

    Nothing in the library optimized transitions. :func:`journey_tracks` walks a path between two
    moods, which fixes the order by construction; this answers the other question: given a SET of
    tracks, in what order should they play so each hand-off is smooth? It minimizes the total
    cosine distance along the sequence — the open-path travelling-salesman objective, no return to
    the start.

    Exact up to :data:`_EXACT_ORDER_MAX` tracks via Held-Karp dynamic programming over subsets
    (``O(2^n · n²)``, ~80 ms at 12 — the DP is a Python loop, not a numpy kernel); above it, a
    nearest-neighbour tour refined by 2-opt until no swap improves, which is not optimal but is the
    standard practical answer and stays milliseconds at playlist sizes.

    ``X`` ``(n, d)`` are track embeddings, re-L2-normalized defensively. ``start`` pins the opening
    track (a row index) when the caller has one in mind; ``None`` lets the optimizer choose, at no
    asymptotic cost — every origin is seeded into the same DP pass. A ``start`` outside
    ``[0, n)`` raises ``ValueError`` rather than being reinterpreted.

    Returns a permutation of ``range(n)``. ``n <= 2`` has nothing to reorder, but a pinned
    ``start`` still decides which of two tracks opens. An empty input returns ``[]`` before
    ``start`` is validated, so sizing a degenerate call never raises. Deterministic. Pure numpy.
    """
    Xn = l2_normalize(np.asarray(X, dtype=np.float32), axis=1)
    n = Xn.shape[0] if Xn.ndim == 2 else 0
    if n == 0:
        return []
    if start is not None and not 0 <= int(start) < n:
        # Refused rather than reinterpreted. An out-of-range index used to mean "choose freely"
        # below the exact cutoff and "pin row 0" above it, so one typo produced two different
        # playlists depending on how many tracks were passed.
        raise ValueError(f"start must be a row index in [0, {n}); got {start}")
    if n <= 2:
        # Nothing to optimize, but `start` still decides which of the two opens.
        return [int(start), 1 - int(start)] if n == 2 and start is not None else list(range(n))

    cost = (1.0 - Xn @ Xn.T).astype(np.float64)  # cosine distance
    np.fill_diagonal(cost, 0.0)

    if n <= _EXACT_ORDER_MAX:
        return _held_karp_path(cost, [int(start)] if start is not None else list(range(n)))
    return _two_opt_path(cost, int(start) if start is not None else _cheapest_nn_start(cost))


def _held_karp_path(cost: np.ndarray, starts: list[int]) -> list[int]:
    """Exact minimum-cost open Hamiltonian path by subset DP, over the allowed start nodes.

    ONE pass however many origins are allowed: each is seeded as its own singleton state, so the
    DP explores all of them at once and the stated ``O(2^n · n²)`` holds.

    At 12 tracks this pass costs ~80 ms free-start and ~48 ms pinned. The shape it replaced —
    a full DP re-run per origin — cost 0.43 s free-start and 0.036 s pinned: it multiplied the
    complexity by ``n`` on the DEFAULT path, in exchange for skipping every mask not containing
    the single pinned origin. So pinning is ~1.33x slower here and the default is ~5.5x faster,
    which is the trade this deliberately takes.
    """
    n = cost.shape[0]
    full = 1 << n
    # dp[mask][j] = cheapest path visiting exactly `mask`, opening at any allowed start, ending at
    # `j`. A state is reachable only from a seeded singleton, so `parent == -1` still terminates
    # the reconstruction at whichever origin won.
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int32)
    for origin in starts:
        dp[1 << origin, origin] = 0.0

    for mask in range(full):
        for j in range(n):
            here = dp[mask, j]
            if not np.isfinite(here) or not (mask >> j) & 1:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                nmask = mask | (1 << nxt)
                cand = here + cost[j, nxt]
                if cand < dp[nmask, nxt]:
                    dp[nmask, nxt] = cand
                    parent[nmask, nxt] = j

    end = int(np.argmin(dp[full - 1]))
    if not np.isfinite(dp[full - 1, end]):  # pragma: no cover — unreachable for a finite cost
        return list(range(n))
    tour, mask, node = [], full - 1, end
    while node != -1:
        tour.append(node)
        prev = int(parent[mask, node])
        mask ^= 1 << node
        node = prev
    return [int(i) for i in tour[::-1]]


def _cheapest_nn_start(cost: np.ndarray) -> int:
    """The start whose greedy nearest-neighbour tour is cheapest — a better 2-opt seed than row 0."""
    return int(min(range(cost.shape[0]), key=lambda s: _tour_cost(cost, _nn_tour(cost, s))))


def _nn_tour(cost: np.ndarray, start: int) -> list[int]:
    n = cost.shape[0]
    unused = set(range(n)) - {start}
    tour = [start]
    while unused:
        nxt = min(unused, key=lambda j: (cost[tour[-1], j], j))
        tour.append(nxt)
        unused.discard(nxt)
    return tour


def _tour_cost(cost: np.ndarray, tour: list[int]) -> float:
    return float(sum(cost[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def _two_opt_path(cost: np.ndarray, start: int) -> list[int]:
    """Nearest-neighbour tour refined by 2-opt segment reversals until no swap improves."""
    tour = _nn_tour(cost, start)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reversing tour[i:j+1] only changes the two edges at its boundaries.
                before = cost[tour[i - 1], tour[i]] + (
                    cost[tour[j], tour[j + 1]] if j + 1 < n else 0.0
                )
                after = cost[tour[i - 1], tour[j]] + (
                    cost[tour[i], tour[j + 1]] if j + 1 < n else 0.0
                )
                if after < before - 1e-12:
                    tour[i : j + 1] = tour[i : j + 1][::-1]
                    improved = True
    return [int(i) for i in tour]
