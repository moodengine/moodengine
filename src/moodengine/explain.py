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
    vs the true read — reported as the pair ``(fidelity, fidelity_folds)``, since ``0`` folds means
    resubstitution and a depth-limited tree scores near the ceiling that way. It is explicitly a
    *correlational* view, not the CLAP mechanism — the caller flags ``is_surrogate`` and shows
    both halves of the pair.
  * :func:`surrogate_shap` — exact interventional Shapley of the signals under the surrogate (default,
    dependency-free), or ``shap.TreeExplainer`` (opt-in, imported lazily) which must concord with it.
  * :func:`counterfactual` — Wachter et al. (2017): the minimal signal perturbation (MAD-weighted,
    bounded) that flips the surrogate's predicted mood, or ``found=False`` (the caller shows nothing).

Everything is deterministic and torch-free; the surrogate uses ``sklearn`` (cross-platform wheels).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import factorial
from typing import Callable, Protocol

import numpy as np

from moodengine.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)

_EPS: float = 1e-8
_MAX_PLAYERS: int = 8  # 2ⁿ coalition evals — the exact-Shapley guard-rail


class SupportsPredictProba(Protocol):
    """Structural surface of the fitted classifier the surrogate wraps: any object
    with a sklearn-style ``predict_proba(X) -> (n, n_classes)``.

    **Rows must be scored independently.** ``predict_proba(X)[i]`` has to depend only on ``X[i]``,
    never on the other rows of the batch — which is what "sklearn-style" means in practice, and what
    every fitted sklearn estimator does. :func:`surrogate_shap` relies on it: it scores all ``2ⁿ``
    coalitions in ONE call, so an estimator that standardizes by the batch's own column means (a
    transform that conforms to this Protocol but is not row-independent) would see a different batch
    and return different attributions than a per-row loop.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def _ensure_within_player_cap(n: int) -> None:
    """Reject a game too large to enumerate exactly. Shared by BOTH exact-Shapley entry points.

    :func:`surrogate_shap` used to inherit this check by routing through :func:`shapley_exact`;
    batching the coalitions means it no longer does, and the cap is what keeps ``2ⁿ`` from being an
    allocation rather than a loop bound. One home for the rule so the two paths cannot drift.
    """
    if n < 0:
        raise ValueError("n_players must be >= 0")
    if n > _MAX_PLAYERS:
        raise ValueError(f"n_players={n} exceeds the exact-Shapley cap {_MAX_PLAYERS} (2^n evals)")


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
    _ensure_within_player_cap(n)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    # Every one of the 2ⁿ coalitions is needed (the old memo saw each exactly once), so evaluate
    # them into a table indexed by bitmask and hand that to the shared weighting below — the same
    # code path :func:`surrogate_shap` reaches after ONE batched model call, so the two cannot drift.
    payoff = np.fromiter(
        (float(value(frozenset(i for i in range(n) if (mask >> i) & 1))) for mask in range(1 << n)),
        dtype=np.float64,
        count=1 << n,
    )
    return _shapley_from_payoffs(payoff, n)


def _shapley_from_payoffs(payoff: np.ndarray, n: int) -> np.ndarray:
    """Shapley values from a payoff table indexed by coalition BITMASK (``payoff[mask]``).

    ``mask`` has bit ``i`` set when player ``i`` is in the coalition, so ``payoff`` is exactly what a
    single batched evaluation of all ``2ⁿ`` coalitions produces, in order. The per-player sum runs
    over the ``2ⁿ⁻¹`` masks without ``i`` as one dot product rather than a Python loop."""
    phi = np.zeros((n,), dtype=np.float64)
    fact = [factorial(k) for k in range(n + 1)]
    n_fact = fact[n]
    masks = np.arange(1 << n, dtype=np.intp)
    sizes = np.fromiter((int(m).bit_count() for m in masks), dtype=np.intp, count=1 << n)
    for i in range(n):
        bit = 1 << i
        without = masks[(masks & bit) == 0]
        s = sizes[without]
        weight = np.fromiter(
            (fact[int(k)] * fact[n - int(k) - 1] / n_fact for k in s),
            dtype=np.float64,
            count=without.shape[0],
        )
        phi[i] = float(np.dot(weight, payoff[without | bit] - payoff[without]))
    return phi


@dataclass(frozen=True)
class SignalSurrogate:
    """A shallow, interpretable classifier ``signals → mood`` — the *correlational surrogate*.

    Holds the fitted sklearn ``model`` plus the ``feature_names`` (musical signals, mood-first) its
    columns align to, the ``mood_names`` it can predict (⊆ the caller's mood vocabulary — the classes
    actually present in the training reads), the library-median ``baseline`` (reference point for
    interventional Shapley + counterfactual distance), and the **measured** ``fidelity`` (accuracy vs
    the true read).

    ``fidelity`` NEVER travels alone: ``fidelity_folds`` says how it was measured, and ``0`` means
    resubstitution — the surrogate was scored on the rows it was fit on. A depth-limited tree
    scores near the ceiling that way, so displaying such a number as "cross-validated accuracy"
    presents a poor surrogate as a faithful one. Follow the ``(value, support)`` convention the
    evaluation module uses: read the pair, or neither.

    Pure data — :func:`surrogate_shap` / :func:`counterfactual` read it."""

    kind: str  # 'tree' | 'linear'
    model: SupportsPredictProba  # sklearn DecisionTreeClassifier | LogisticRegression (picklable)
    feature_names: list[str]  # e.g. ['bpm', 'tempo_stability', 'energy', 'valence', 'key']
    mood_names: list[str]  # classes the surrogate can predict (⊆ config.moods)
    baseline: np.ndarray  # (n_features,) library medians
    fidelity: float  # accuracy vs the true read, in [0, 1] — read WITH fidelity_folds
    # Defaulted so a hand-built surrogate stays constructible: 0 already means "not
    # validated", which is exactly what an unstated fold count is.
    fidelity_folds: int = 0  # stratified folds behind `fidelity`; 0 == resubstitution/unknown


def _cv_accuracy(model, S: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[float, int]:
    """Cross-validated accuracy of ``model`` predicting ``y`` from ``S`` — the honest fidelity number.

    Returns ``(accuracy, n_folds)``, where ``n_folds == 0`` marks a RESUBSTITUTION score: the model
    was scored on the rows it was fit on, which for a depth-limited tree over a real vocabulary sits
    near the ceiling and is not evidence of anything. Callers must surface that distinction — the
    number and its fold count travel together.

    The fold count is derived from the classes that can actually be folded (≥ 2 members), not from
    the global smallest class. Deriving it globally meant a single mood appearing exactly ONCE in
    the whole library forced ``k = 1`` and silently downgraded the entire estimate to
    resubstitution: on a 180-row, 3-mood weak-signal set, appending one row of a fourth mood moved
    the reported fidelity from 0.361 to 0.608 without the surrogate getting any better. Rows of an
    unfoldable class are dropped from the estimate instead — they cannot be held out and predicted
    honestly — so the score describes the part of the vocabulary it can describe."""
    from sklearn.base import clone

    counts = np.bincount(y)
    foldable = np.flatnonzero(counts >= 2)
    mask = np.isin(y, foldable)
    k = min(5, int(counts[foldable].min())) if foldable.size >= 2 else 1

    if k < 2 or int(np.unique(y[mask]).size) < 2:
        m = clone(model)
        m.fit(S, y)
        return float((m.predict(S) == y).mean()), 0

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    dropped = int((~mask).sum())
    if dropped:
        logger.info(
            "fidelity: %d row(s) across %d mood(s) occur fewer than twice and cannot be held out; "
            "excluding them from the estimate (the surrogate is still fit on every row).",
            dropped,
            int(np.setdiff1d(np.unique(y), foldable).size),
        )
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    score = cross_val_score(model, S[mask], y[mask], cv=cv, scoring="accuracy").mean()
    return float(score), int(k)


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
    to *mimic* the read from signals, and its held-out accuracy (``fidelity``, measured over
    ``fidelity_folds`` stratified folds — ``0`` meaning it fell back to resubstitution and is NOT
    validated) says how faithful that view is. Rows whose mood occurs fewer than twice are excluded
    from that ESTIMATE only — they cannot be held out and predicted — so ``fidelity`` describes the
    foldable part of the vocabulary while the surrogate itself is fit on every row. ``kind='tree'`` fits a depth-limited ``DecisionTreeClassifier`` (the interpretable
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

        # ccp_alpha pinned to sklearn's default rather than left implicit: this surrogate exists
        # to be READ, and `max_depth` is the single knob that trades fidelity for readability.
        # Cost-complexity pruning would move the tree without appearing in the caller's arguments.
        model = DecisionTreeClassifier(
            max_depth=int(max_depth), random_state=int(seed), ccp_alpha=0.0
        )
    elif kind == "linear":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown kind {kind!r} (expected 'tree' | 'linear')")

    fidelity, fidelity_folds = _cv_accuracy(model, S, y, seed=int(seed))
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
        fidelity_folds=int(fidelity_folds),
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
    All ``2ⁿ`` coalitions are scored in a SINGLE ``predict_proba`` call, so the wrapped estimator
    must score rows independently — see :class:`SupportsPredictProba`. Every fitted sklearn
    classifier does; an estimator that standardizes against the batch it is given does not, and
    would return different attributions here than under a per-coalition loop.

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

    # Enforced HERE, not inherited: batching the coalitions means this no longer routes through
    # `shapley_exact`, and without the cap `1 << nf` below is an allocation, not a loop bound —
    # a 25-feature surrogate would ask for a (33M, 25) design matrix.
    _ensure_within_player_cap(nf)
    if nf == 0:
        # A surrogate with no features has nothing to attribute. Returning here deliberately:
        # the batched form below would otherwise hand `predict_proba` a (1, 0) design matrix,
        # which sklearn rejects — and the per-coalition form never called the model at all.
        return np.zeros((0,), dtype=np.float64)

    baseline = np.asarray(surr.baseline, dtype=np.float64).ravel()
    # One call, not 2ⁿ: row `mask` of the design matrix takes x on the signals in the coalition and
    # the baseline elsewhere, which is the interventional value function evaluated everywhere at
    # once. The cap checked above holds the matrix to at most (256, n_features).
    masks = np.arange(1 << nf, dtype=np.intp)
    in_coalition = ((masks[:, None] >> np.arange(nf, dtype=np.intp)) & 1).astype(bool)
    design = np.where(in_coalition, x, baseline)
    payoff = np.asarray(surr.model.predict_proba(design), dtype=np.float64)[:, col]
    return _shapley_from_payoffs(payoff, nf)


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


def _weighted_dists(rows: np.ndarray, b: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """MAD-weighted L1 distance of every row of ``rows`` (m, d) to ``b`` (d,). Returns (m,)."""
    scale = np.where(mad > _EPS, mad, 1.0)
    return np.sum(np.abs(rows - b) / scale, axis=1)


def _weighted_dist(a: np.ndarray, b: np.ndarray, mad: np.ndarray) -> float:
    """Wachter's MAD-weighted L1 distance — robust, per-signal-scaled, so a 5-BPM move and a
    0.1-valence move are comparable. Delegates to :func:`_weighted_dists` so the scalar and
    batched forms cannot drift apart."""
    return float(_weighted_dists(np.asarray(a).reshape(1, -1), b, mad)[0])


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
    # The whole grid, flattened once, in the order the nested loops visited it: feature ascending,
    # then grid value ascending (np.unique sorts). That order is load-bearing — the search keeps the
    # FIRST candidate of an equal key — so it is fixed here rather than rebuilt per step.
    feature_of = np.concatenate(
        [np.full(g.shape[0], f, dtype=np.intp) for f, g in enumerate(grids)]
    )
    value_of = np.concatenate(grids)

    cur = x.copy()
    for _ in range(int(max_iter)):
        base_p = prob_target(cur)
        movable = (
            value_of != cur[feature_of]
        )  # a candidate equal to the current value is not a move
        if not movable.any():
            break
        rows, cols = np.flatnonzero(movable), feature_of[movable]
        # One predict_proba over the entire candidate grid instead of one call per candidate. Rows
        # are scored independently — see SupportsPredictProba, whose row-independence requirement
        # this shares with surrogate_shap. Envelope: a (9·nf, nf) float64 block, which is the same
        # bound the per-candidate loop already implied in TIME.
        candidates = np.repeat(cur[None, :], rows.shape[0], axis=0)
        candidates[np.arange(rows.shape[0]), cols] = value_of[movable]
        probs = np.asarray(surr.model.predict_proba(candidates), dtype=np.float64)[:, target_idx]
        # Python's round(), NOT np.round(): the rounding is here to make near-ties TIE, and
        # np.round's rint(v·1e12)/1e12 disagrees with CPython's decimal rounding on roughly 1 double
        # in 20 000 — enough to break a tie and pick a different signal. The grid is ~9·nf wide, so
        # the comprehension costs nothing next to the model call above.
        keys = np.fromiter(
            (-round(float(p), 12) for p in probs), dtype=np.float64, count=probs.size
        )
        dists = _weighted_dists(candidates, x, mad)
        # lexsort's LAST key is primary, and it is stable — so this reproduces the tuple comparison
        # `(-p, dist)` with the earliest candidate winning an exact tie, exactly as `key < best[0]`.
        winner = int(np.lexsort((dists, keys))[0])
        if -keys[winner] <= base_p:  # no strictly-improving single-signal move
            break
        cur = candidates[winner]
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
