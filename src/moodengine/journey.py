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
