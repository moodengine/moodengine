"""Explainability primitives — exact Shapley on a few players, an interpretable signal surrogate,
its (exact / TreeSHAP) attribution, and Wachter counterfactuals. Pure numpy + sklearn, torch-free,
deterministic.

Doctrine: **raw SHAP over the 512 CLAP dims is useless** and is NEVER done. Instead
the game here has only a handful of *players* — either the additive components of the blend
(zero-shot / few-shot examples / probe / calibration) or ≤ 8 interpretable musical signals (BPM,
Camelot key, tempo stability, energy, valence, …). With so few players the Shapley values are
computed **exactly** over the 2ⁿ coalitions (Shapley 1953; ``n ≤ 8`` ⇒ ≤ 256 value-function calls),
so nothing is approximated or fabricated.

  * :func:`shapley_exact` — exact Shapley values of an arbitrary coalition value-function
    (memoized). The *value-function itself* is supplied by the caller, so this
    module owns the algorithm, not the blend.
  * :class:`SignalSurrogate` / :func:`fit_signal_surrogate` — a shallow, interpretable classifier
    (decision tree / logistic) mapping musical **signals → mood**, with its **measured** fidelity
    (cross-validated accuracy vs the true read). It is explicitly a *correlational* view, not the
    CLAP mechanism — the caller flags ``is_surrogate`` and shows the fidelity.
  * :func:`surrogate_shap` — exact interventional Shapley of the signals under the surrogate (default,
    dependency-free), or ``shap.TreeExplainer`` (opt-in, imported lazily) which must concord with it.
  * :func:`counterfactual` — Wachter et al. (2017): the minimal signal perturbation (MAD-weighted,
    bounded) that flips the surrogate's predicted mood, or ``found=False`` (the caller shows nothing).

Everything is deterministic and torch-free; the surrogate uses ``sklearn`` (cross-platform wheels).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Callable, Protocol

import numpy as np

from moodengine.exceptions import MissingDependencyError

_EPS: float = 1e-8
_MAX_PLAYERS: int = 8  # 2ⁿ coalition evals — the exact-Shapley guard-rail


class SupportsPredictProba(Protocol):
    """Structural surface of the fitted classifier the surrogate wraps: any object
    with a sklearn-style ``predict_proba(X) -> (n, n_classes)``."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def shapley_exact(value: Callable[[frozenset], float], n_players: int) -> np.ndarray:
    """Exact Shapley values ``φ`` of a coalition value-function (Shapley 1953).

    ``value`` maps a coalition (a ``frozenset`` of player indices ``0..n_players-1``) to a real
    payoff; ``value(frozenset())`` is the empty coalition. Returns ``φ`` ``(n_players,)`` float64 such
    that — by construction — ``Σφ == value(full) − value(∅)`` (efficiency), interchangeable players
    get equal ``φ`` (symmetry), and a player that never changes the payoff gets ``φ == 0`` (dummy).

    Enumerates all ``2ⁿ`` coalitions once (``value`` is memoized over the ``frozenset``), so it is
    exact but bounded to ``n_players ≤ 8`` (``ValueError`` above — the caller must keep the game
    small: blend components or a whitelist of interpretable signals). ``n_players == 0`` ⇒ empty
    ``φ``. Deterministic; ``value`` must be a pure function of its coalition."""
    n = int(n_players)
    if n < 0:
        raise ValueError("n_players must be >= 0")
    if n > _MAX_PLAYERS:
        raise ValueError(f"n_players={n} exceeds the exact-Shapley cap {_MAX_PLAYERS} (2^n evals)")
    phi = np.zeros((n,), dtype=np.float64)
    if n == 0:
        return phi

    _cache: dict[frozenset, float] = {}

    def v(coalition: frozenset) -> float:
        got = _cache.get(coalition)
        if got is None:
            got = float(value(coalition))
            _cache[coalition] = got
        return got

    fact = [factorial(k) for k in range(n + 1)]
    n_fact = fact[n]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        m = len(others)  # n - 1
        for mask in range(1 << m):
            coalition = frozenset(others[t] for t in range(m) if (mask >> t) & 1)
            s = len(coalition)
            weight = fact[s] * fact[n - s - 1] / n_fact
            phi[i] += weight * (v(coalition | {i}) - v(coalition))
    return phi


@dataclass(frozen=True)
class SignalSurrogate:
    """A shallow, interpretable classifier ``signals → mood`` — the *correlational surrogate*.

    Holds the fitted sklearn ``model`` plus the ``feature_names`` (musical signals, mood-first) its
    columns align to, the ``mood_names`` it can predict (⊆ the caller's mood vocabulary — the classes
    actually present in the training reads), the library-median ``baseline`` (reference point for
    interventional Shapley + counterfactual distance), and the **measured** ``fidelity`` (accuracy vs
    the true read). Pure data — :func:`surrogate_shap` / :func:`counterfactual` read it."""

    kind: str  # 'tree' | 'linear'
    model: SupportsPredictProba  # sklearn DecisionTreeClassifier | LogisticRegression (picklable)
    feature_names: list[str]  # e.g. ['bpm', 'tempo_stability', 'energy', 'valence', 'key']
    mood_names: list[str]  # classes the surrogate can predict (⊆ config.moods)
    baseline: np.ndarray  # (n_features,) library medians
    fidelity: float  # cross-validated accuracy vs the true read, in [0, 1]


def _cv_accuracy(model, S: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    """Cross-validated accuracy of ``model`` predicting ``y`` from ``S`` — the honest fidelity number.

    Uses stratified k-fold (``k = min(5, smallest-class-count, n)``); falls back to resubstitution
    accuracy when there are too few per-class samples to fold honestly (documented, still measured)."""
    from sklearn.base import clone

    counts = np.bincount(y)
    min_class = int(counts[counts > 0].min())
    k = min(5, min_class, len(y))
    if k < 2:
        m = clone(model)
        m.fit(S, y)
        return float((m.predict(S) == y).mean())
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    return float(cross_val_score(model, S, y, cv=cv, scoring="accuracy").mean())


def fit_signal_surrogate(
    S: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    mood_names: list[str],
    *,
    kind: str = "tree",
    max_depth: int = 4,
    seed: int = 0,
) -> SignalSurrogate:
    """Fit a shallow, interpretable ``signals → mood`` surrogate and **measure** its fidelity.

    ``S`` ``(n, n_features)`` are per-track interpretable musical signals; ``y`` ``(n,)`` are the
    indices (into ``mood_names``) of the **true engine read** (``top_mood``) — so the surrogate learns
    to *mimic* the read from signals, and its cross-validated accuracy (``fidelity``) says how faithful
    that view is. ``kind='tree'`` fits a depth-limited ``DecisionTreeClassifier`` (the interpretable
    default); ``'linear'`` a multinomial ``LogisticRegression``. The stored ``mood_names`` are the
    classes actually present in ``y`` (⊆ the passed vocabulary — mood-first by construction, since the
    caller only ever passes musical ``feature_names`` and mood classes). ``baseline`` is the
    per-feature library median. Deterministic (fixed ``seed``); torch-free (sklearn). Raises
    ``ValueError`` on mis-shaped inputs, ``< 2`` rows, ``< 2`` distinct moods, or an unknown ``kind``.
    Inputs are never mutated."""
    S = np.asarray(S, dtype=np.float64)
    y = np.asarray(y).astype(int).ravel()
    if S.ndim != 2:
        raise ValueError("S must be a 2-D array (n, n_features)")
    n, nf = S.shape
    if y.shape[0] != n:
        raise ValueError("S and y must have the same number of rows")
    if len(feature_names) != nf:
        raise ValueError("feature_names must align with S columns")
    if n < 2:
        raise ValueError("need at least 2 rows to fit a surrogate")
    present = sorted({int(c) for c in y})
    if len(present) < 2:
        raise ValueError("need at least 2 distinct moods in y to fit a surrogate")
    if any(c < 0 or c >= len(mood_names) for c in present):
        raise ValueError("y contains a mood index outside mood_names")

    if kind == "tree":
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=int(max_depth), random_state=int(seed))
    elif kind == "linear":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown kind {kind!r} (expected 'tree' | 'linear')")

    fidelity = _cv_accuracy(model, S, y, seed=int(seed))
    model.fit(S, y)
    baseline = np.median(S, axis=0).astype(np.float64)
    # model.classes_ is sorted ascending == `present`; map back to names, keeping column alignment.
    learned_names = [mood_names[int(c)] for c in model.classes_]
    return SignalSurrogate(
        kind=kind,
        model=model,
        feature_names=list(feature_names),
        mood_names=learned_names,
        baseline=baseline,
        fidelity=float(fidelity),
    )


def _class_column(surr: SignalSurrogate, mood_idx: int) -> int:
    """Column of ``predict_proba`` for the ``mood_idx``-th surrogate mood (they align by position,
    since ``mood_names`` was built from ``model.classes_`` in order)."""
    if not (0 <= mood_idx < len(surr.mood_names)):
        raise ValueError("mood_idx out of range for surrogate.mood_names")
    return mood_idx


def surrogate_shap(
    surr: SignalSurrogate,
    x: np.ndarray,
    mood_idx: int,
    *,
    backend: str = "exact",
) -> np.ndarray:
    """Per-signal attribution ``φ`` for one track under the surrogate. ``Σφ == P(mood|x) − P(mood|baseline)``.

    ``x`` ``(n_features,)`` is the track's signal vector; ``mood_idx`` indexes ``surr.mood_names``.
    Default ``backend='exact'``: **interventional** Shapley via :func:`shapley_exact`, where
    ``value(S) = P_surrogate(mood | x on the signals in S, baseline elsewhere)`` — dependency-free and
    exact (``n_features ≤ 8``). A signal already at its baseline value contributes ``φ == 0`` exactly
    (its coalition marginals all vanish), so the caller may fill an unextracted signal with the
    baseline and it will not be (mis-)attributed. ``backend='treeshap'`` uses ``shap.TreeExplainer``
    with interventional perturbation against the same single ``baseline`` reference (imported lazily;
    :class:`~moodengine.exceptions.MissingDependencyError` if ``shap`` is absent) — which is
    mathematically the same game, so the two concord.
    Returns ``φ`` ``(n_features,)`` float64. Deterministic; inputs are never mutated."""
    x = np.asarray(x, dtype=np.float64).ravel()
    nf = len(surr.feature_names)
    if x.shape[0] != nf:
        raise ValueError(f"x has {x.shape[0]} features != surrogate {nf}")
    col = _class_column(surr, mood_idx)
    if backend == "treeshap":
        return _treeshap(surr, x, col)
    if backend != "exact":
        raise ValueError(f"unknown backend {backend!r} (expected 'exact' | 'treeshap')")

    baseline = np.asarray(surr.baseline, dtype=np.float64).ravel()

    def value(coalition: frozenset) -> float:
        vec = baseline.copy()
        for f in coalition:
            vec[f] = x[f]
        return float(surr.model.predict_proba(vec.reshape(1, -1))[0][col])

    return shapley_exact(value, nf)


def _treeshap(surr: SignalSurrogate, x: np.ndarray, col: int) -> np.ndarray:
    """TreeSHAP (Lundberg et al. 2020) with a single interventional baseline reference — imported
    lazily so ``shap`` stays an optional extra. Concords with the exact interventional Shapley."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
        raise MissingDependencyError(
            "backend='treeshap'",
            "shap",
            "explain",
            hint="or use backend='exact', the dependency-free default",
        ) from exc

    baseline = np.asarray(surr.baseline, dtype=np.float64).reshape(1, -1)
    explainer = shap.TreeExplainer(surr.model, data=baseline, feature_perturbation="interventional")
    raw = explainer.shap_values(x.reshape(1, -1), check_additivity=False)
    return _select_class_shap(raw, col, len(surr.feature_names), len(surr.mood_names))


def _select_class_shap(raw, col: int, nf: int, n_classes: int) -> np.ndarray:
    """Extract the ``(n_features,)`` attribution for class column ``col`` from shap's per-version
    output shape. Disambiguates the 3-D layouts by the FEATURE axis (``== nf``) rather than the class
    axis, so it stays correct even when ``nf == n_classes``; handles the binary single-output case
    (where class 0 is the negation of the positive-class attribution)."""
    if isinstance(raw, list):  # older shap: per-class list of (1, nf)
        return np.asarray(raw[col], dtype=np.float64).ravel()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == nf:  # modern: (1, nf, n_classes)
        return arr[0, :, col].ravel()
    if arr.ndim == 3 and arr.shape[0] == n_classes and arr.shape[2] == nf:  # (n_classes, 1, nf)
        return arr[col, 0, :].ravel()
    flat = arr.reshape(-1)
    if flat.size == nf:  # binary single-output: +ve class contrib
        return flat if col == 1 else -flat  # class 0 == negation for a 2-class model
    return flat[:nf]


@dataclass(frozen=True)
class Counterfactual:
    """The result of a counterfactual search: whether one was found and the applied signal deltas
    (``0`` on unchanged signals). ``found=False`` ⇒ ``deltas`` all-zero (the caller shows nothing)."""

    found: bool
    deltas: np.ndarray  # (n_features,) perturbation applied to x


def _weighted_dist(a: np.ndarray, b: np.ndarray, mad: np.ndarray) -> float:
    """Wachter's MAD-weighted L1 distance — robust, per-signal-scaled, so a 5-BPM move and a
    0.1-valence move are comparable."""
    scale = np.where(mad > _EPS, mad, 1.0)
    return float(np.sum(np.abs(a - b) / scale))


def counterfactual(
    surr: SignalSurrogate,
    x: np.ndarray,
    target_idx: int,
    *,
    mad: np.ndarray,
    bounds: np.ndarray,
    max_iter: int = 200,
    seed: int = 0,
) -> Counterfactual:
    """Wachter et al. (2017): the minimal signal perturbation that flips the surrogate to ``target_idx``.

    Solves ``argmin_Δ dist(x, x+Δ)  s.t.  argmax P_surrogate(x+Δ) == target_idx``, ``x+Δ ∈ bounds``,
    with ``dist`` the MAD-weighted L1 (:func:`_weighted_dist`). Deterministic **greedy coordinate**
    search over a per-signal grid spanning ``bounds`` (no torch, no RNG — ``seed`` is accepted for API
    symmetry): each step sets one signal to the grid value that most raises ``P(target)`` (ties broken
    by smaller distance), stopping as soon as the prediction flips; a final prune reverts any change
    not needed for the flip (minimality). Returns ``found=True`` with the deltas (verified to flip the
    surrogate), or ``found=False`` with all-zero deltas when no counterfactual is reached within
    ``bounds``/``max_iter`` (the caller then shows nothing — never a fabricated flip). ``target_idx``
    indexes ``surr.mood_names``; inputs are never mutated."""
    x = np.asarray(x, dtype=np.float64).ravel()
    mad = np.asarray(mad, dtype=np.float64).ravel()
    bounds = np.asarray(bounds, dtype=np.float64)
    nf = len(surr.feature_names)
    zeros = np.zeros((nf,), dtype=np.float64)
    if x.shape[0] != nf or mad.shape[0] != nf or bounds.shape != (nf, 2):
        raise ValueError("x / mad / bounds must align with the surrogate features")
    if not (0 <= target_idx < len(surr.mood_names)):
        return Counterfactual(found=False, deltas=zeros)

    def pred(vec: np.ndarray) -> int:
        return int(np.argmax(surr.model.predict_proba(vec.reshape(1, -1))[0]))

    def prob_target(vec: np.ndarray) -> float:
        return float(surr.model.predict_proba(vec.reshape(1, -1))[0][target_idx])

    if pred(x) == target_idx:
        return Counterfactual(found=True, deltas=zeros)  # already there — no change needed

    lo, hi = bounds[:, 0], bounds[:, 1]
    grids = [np.unique(np.concatenate([np.linspace(lo[f], hi[f], 9), [x[f]]])) for f in range(nf)]

    cur = x.copy()
    for _ in range(int(max_iter)):
        base_p = prob_target(cur)
        best: tuple[tuple[float, float], np.ndarray] | None = None  # (key, candidate) — paired
        for f in range(nf):
            for val in grids[f]:
                if val == cur[f]:
                    continue
                cand = cur.copy()
                cand[f] = val
                key = (-round(prob_target(cand), 12), _weighted_dist(cand, x, mad))
                if best is None or key < best[0]:
                    best = (key, cand)
        if best is None or (-best[0][0]) <= base_p:  # no strictly-improving single-signal move
            break
        cur = best[1]
        if pred(cur) == target_idx:
            deltas = _prune_deltas(surr, x, cur - x, target_idx)
            return Counterfactual(found=True, deltas=deltas)
    return Counterfactual(found=False, deltas=zeros)


def _prune_deltas(
    surr: SignalSurrogate, x: np.ndarray, deltas: np.ndarray, target_idx: int
) -> np.ndarray:
    """Minimality pass: revert each changed signal to its original value if the flip still holds —
    smaller, more actionable counterfactuals."""
    d = deltas.copy()
    for f in range(len(d)):
        if d[f] == 0.0:
            continue
        trial = d.copy()
        trial[f] = 0.0
        pred = int(np.argmax(surr.model.predict_proba((x + trial).reshape(1, -1))[0]))
        if pred == target_idx:
            d = trial
    return d
