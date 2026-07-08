"""Confidence calibration for the mood softmax (Guo et al. 2017, "On Calibration of Modern Neural
Networks") — torch-free.

The triptych's ``softmax(rec, temperature)`` yields UN-calibrated probabilities: the default
temperature is a spread aesthetic, not a statistical optimum, so a 0.95 score does not mean "right
95% of the time". This module fits calibration on a gold set:

  * **temperature scaling** — the single parameter ``T`` minimizing the negative log-likelihood of
    ``softmax(logits / T)``. It is *monotone*, so it never reorders the argmax: top-1 accuracy is
    invariant, only the confidence becomes honest.
  * **Platt** (1D logistic) and **isotonic** (non-parametric) — comparison baselines.
  * **reliability_diagram** + **negative_log_likelihood** — the measurement surface.
  * **entropy** / **margin** + **aps_threshold** / **prediction_set** — intrinsic uncertainty and
    coverage-guaranteed conformal prediction sets on top of the same softmax (APS/RAPS).

The scalar **ECE lives in** :func:`moodengine.evaluation.expected_calibration_error` and is NOT
redefined here (a caller composes the two). Everything is pure numpy; ``scipy`` / ``sklearn`` are
imported lazily (mirroring :func:`moodengine.evaluation._spearman`) with a numpy golden-section
fallback for the temperature fit, so the module import stays light and torch-free.
"""

from __future__ import annotations

import numpy as np


def _softmax_T(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Row-wise softmax of ``logits / temperature`` (numerically stable). Internal helper so the fit
    and the callers share one definition."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of the true-class probabilities: ``−mean(log(clip(probs[i, labels[i]], 1e-12, 1)))``.

    ``probs`` is ``(n, n_classes)`` (rows sum to 1); ``labels`` ``(n,)`` integer class indices.
    Returns ``0.0`` on empty input. Never raises. The clip guards ``log(0)`` for a class the model
    assigned zero mass.
    """
    P = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(int).ravel()
    n = min(P.shape[0], y.shape[0]) if P.ndim == 2 else 0
    if n == 0:
        return 0.0
    idx = np.clip(y[:n], 0, P.shape[1] - 1)
    p = np.clip(P[np.arange(n), idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, bounds: tuple[float, float] = (1e-3, 100.0)
) -> float:
    """Temperature scaling (Guo+'17): the single ``T > 0`` minimizing ``NLL(softmax(logits / T))``.

    ``logits`` ``(n, n_moods)`` are the pre-softmax **rec** (recentered) vectors; ``labels`` ``(n,)``
    the gold class index in the same mood order. Minimizes over ``T ∈ bounds`` with
    ``scipy.optimize.minimize_scalar`` (bounded method); a deterministic numpy golden-section search
    is the fallback when ``scipy`` is absent (lazy import, like ``_spearman``). Returns ``1.0`` for
    ``n < 1`` (nothing to fit). Deterministic (no RNG). Because ``T`` is a positive scalar divisor,
    the ranking of every row is preserved — ``argmax(logits) == argmax(logits / T)`` — so this only
    rescales confidence, never the prediction.
    """
    L = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels).astype(int).ravel()
    if L.ndim != 2 or L.shape[0] < 1 or y.shape[0] < 1:
        return 1.0
    n = min(L.shape[0], y.shape[0])
    L, y = L[:n], y[:n]
    lo, hi = float(bounds[0]), float(bounds[1])

    def _nll(temp: float) -> float:
        return negative_log_likelihood(_softmax_T(L, temp), y)

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(_nll, bounds=(lo, hi), method="bounded")
        T = float(res.x)
    except Exception:
        # Golden-section search on [lo, hi] — deterministic, no RNG, no scipy.
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - gr * (b - a), a + gr * (b - a)
        for _ in range(200):
            if _nll(c) < _nll(d):
                b = d
            else:
                a = c
            c, d = b - gr * (b - a), a + gr * (b - a)
            if abs(b - a) < 1e-6:
                break
        T = 0.5 * (a + b)
    return float(min(max(T, lo), hi))


def reliability_diagram(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> list[dict]:
    """One dict per NON-EMPTY equal-width confidence bin (Guo+'17 reliability diagram).

    ``confidences`` ∈ [0, 1] are the top-1 confidences; ``correct`` is 0/1 (was the top-1 right).
    Returns ``[{"bin_lo","bin_hi","count","mean_confidence","accuracy"}, …]`` for the occupied bins
    only (a perfectly calibrated model has ``accuracy ≈ mean_confidence`` in every bin). Pure numpy;
    the data source for a UI diagram (no rendering here). Robust to ``n == 0`` → ``[]``.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    n = min(conf.shape[0], corr.shape[0])
    if n == 0:
        return []
    conf, corr = conf[:n], corr[:n]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, int(n_bins) - 1)
    out: list[dict] = []
    for b in range(int(n_bins)):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": count,
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(corr[mask].mean()),
            }
        )
    return out


def platt_scale(confidences: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Platt scaling baseline: a 1D logistic regression mapping confidence → P(correct).

    Returns ``(a, b)`` of ``sigmoid(a·x + b)`` (``sklearn.linear_model.LogisticRegression``, lazy
    import). Returns ``(0.0, 0.0)`` on degenerate input (fewer than 2 samples or a single class) —
    a flat, uninformative map rather than a raise.
    """
    x = np.asarray(confidences, dtype=np.float64).ravel()
    yb = np.asarray(correct).astype(int).ravel()
    n = min(x.shape[0], yb.shape[0])
    if n < 2 or len(set(yb[:n].tolist())) < 2:
        return 0.0, 0.0
    from sklearn.linear_model import LogisticRegression

    # Near-unregularized (large C) to approximate Platt's MLE across sklearn versions — textbook
    # Platt scaling is unpenalized, and the default C=1.0 would shrink the slope toward 0.
    clf = LogisticRegression(C=1e10)
    clf.fit(x[:n].reshape(-1, 1), yb[:n])
    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def isotonic_calibrate(confidences: np.ndarray, correct: np.ndarray):
    """Isotonic (non-parametric, monotone) calibration mapping confidence → empirical P(correct).

    Returns a fitted ``sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`` (lazy import) —
    ``.predict(conf)`` gives the calibrated probability. More flexible than Platt (no sigmoid shape
    assumption); needs more data. Fits on whatever is given (a single point yields a constant map).
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.asarray(confidences, dtype=np.float64).ravel()
    yb = np.asarray(correct, dtype=np.float64).ravel()
    n = min(x.shape[0], yb.shape[0])
    ir = IsotonicRegression(out_of_bounds="clip")
    # Always fit SOMETHING so the returned regressor is usable: on empty input, fit a trivial
    # identity so .predict() is a safe pass-through (never a landmine that raises later on use).
    if n >= 1:
        ir.fit(x[:n], yb[:n])
    else:
        ir.fit([0.0, 1.0], [0.0, 1.0])
    return ir


# --- conformal prediction sets + intrinsic uncertainty -----------------------
# The temperature-scaled softmax gives a full distribution per track; a single top-1 label hides how
# uncertain it is. These add an *honest* uncertainty layer on top of the SAME ``probs`` the triptych
# already computes — two intrinsic scalars (entropy, margin) and a coverage-guaranteed prediction set
# (APS/RAPS: Romano/Sesia/Candès 2020; Angelopoulos & Bates 2021). The conformal threshold q̂ is
# calibrated ONCE on a gold set by the caller (who owns the labeled overrides and storage); the algo
# here knows only ``(cal_probs, cal_true_idx)``. Pure numpy, deterministic, torch-free.


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in **nats** (base *e*) of a probability distribution along ``axis``.

    ``probs`` rows sum to 1 (softmax output). Computes ``−Σ p·ln p`` with the ``p·ln p → 0`` limit at
    ``p = 0`` handled by ``np.where`` (never ``nan``). ``(n, m) → (n,)``. Bounded ``[0, ln m]``: ``0``
    on a one-hot (fully certain), maximal ``ln m`` on the uniform distribution (maximally uncertain).
    A high entropy is the honest "the engine is genuinely torn" signal. Pure numpy.
    """
    p = np.asarray(probs, dtype=np.float64)
    terms = np.where(p > 0.0, p * np.log(np.where(p > 0.0, p, 1.0)), 0.0)
    return -np.sum(terms, axis=axis)


def margin(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Top-1 minus top-2 probability along ``axis`` — the confidence gap.

    ``(n, m) → (n,)``, bounded ``[0, 1]`` for a distribution: ``0`` when the two leading moods tie
    (maximally ambiguous), near ``1`` when one mood dominates. Complements :func:`entropy` (margin is
    local to the top pair; entropy uses the whole distribution). ``m == 1`` → the lone probability.
    Pure numpy; never mutates the input.
    """
    p = np.asarray(probs, dtype=np.float64)
    m = p.shape[axis]
    s = np.sort(p, axis=axis)
    top1 = np.take(s, m - 1, axis=axis)
    if m < 2:
        return top1
    top2 = np.take(s, m - 2, axis=axis)
    return top1 - top2


def _aps_scores(probs: np.ndarray, true_idx: np.ndarray, k_reg: int, lam_reg: float) -> np.ndarray:
    """APS non-conformity score per row: the cumulative probability mass of the moods ranked at least
    as high as the true mood (sorted ``probs``↓, up to & including the true class), plus the optional
    RAPS regularizer ``lam_reg·max(0, rank_true − k_reg)`` (``rank_true`` 1-indexed). ``lam_reg=0`` ⇒
    pure APS. ``(n, m) + (n,) → (n,)``."""
    P = np.asarray(probs, dtype=np.float64)
    y = np.asarray(true_idx).astype(int).ravel()
    order = np.argsort(-P, axis=1, kind="stable")  # moods sorted by prob descending
    cum = np.cumsum(np.take_along_axis(P, order, axis=1), axis=1)
    ranks = np.argmax(order == y[:, None], axis=1)  # 0-indexed position of the true mood
    scores = cum[np.arange(P.shape[0]), ranks]
    if lam_reg > 0.0:
        scores = scores + lam_reg * np.maximum(0.0, (ranks + 1) - int(k_reg))
    return scores


def aps_threshold(
    cal_probs: np.ndarray,
    cal_true_idx: np.ndarray,
    coverage_target: float,
    *,
    k_reg: int = 0,
    lam_reg: float = 0.0,
    rng_jitter: bool = False,
) -> float:
    """Calibrate the conformal threshold ``q̂`` for a coverage target ``1−ε`` (APS/RAPS, split-conformal).

    ``cal_probs`` ``(n_cal, m)`` are the softmax rows of the calibration (gold) tracks and
    ``cal_true_idx`` ``(n_cal,)`` their true mood indices (same mood order). Each gets an APS
    non-conformity score (see :func:`_aps_scores`); ``q̂`` is the ``⌈(n_cal+1)·coverage_target⌉``-th
    smallest score (the finite-sample conformal quantile). When that rank exceeds ``n_cal`` (target too
    high for the sample) ``q̂ = 1.0`` — every mood is included, the honest "can't guarantee this
    coverage at this n" behavior, never a fabricated tighter set. ``rng_jitter=False`` is the
    deterministic, non-randomized variant (reproducible; slightly conservative coverage). Returns
    ``q̂ ∈ [0, 1]`` for pure APS (``lam_reg=0``). Deterministic, pure numpy.
    """
    P = np.asarray(cal_probs, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] < 1:
        return 1.0  # nothing to calibrate → most conservative; the caller guards min_cal separately
    y = np.asarray(cal_true_idx).astype(int).ravel()
    n = min(P.shape[0], y.shape[0])
    scores = np.sort(_aps_scores(P[:n], y[:n], k_reg, lam_reg))
    k = int(np.ceil((n + 1) * float(coverage_target)))
    if k < 1:
        return (
            0.0  # coverage_target ≤ 0 → tightest possible set (prediction_set keeps the top-1 mood)
        )
    if k > n:
        # Target coverage unattainable at this n → include EVERY mood (no fabricated tightness). Pure
        # APS cumulates to ≤ 1, so 1.0 includes all; RAPS adds a rank penalty that pushes the cumulative
        # above 1, so only +inf guarantees the full set (see prediction_set's cum + lam_reg·… ).
        return 1.0 if lam_reg <= 0.0 else float("inf")
    q = float(scores[k - 1])  # k-th smallest calibration score (1-indexed order statistic)
    return float(min(max(q, 0.0), 1.0)) if lam_reg <= 0.0 else float(max(q, 0.0))


def prediction_set(
    probs_row: np.ndarray, q_hat: float, *, k_reg: int = 0, lam_reg: float = 0.0
) -> np.ndarray:
    """The APS/RAPS prediction set for one track: the smallest ``probs``↓ prefix whose cumulative
    (APS) score reaches ``q̂``.

    ``probs_row`` ``(m,)`` is the softmax row; ``q_hat`` the calibrated threshold. Returns the mood
    **indices** of the set, ordered by ``probs`` descending, **never empty** (at least the top-1 mood,
    even when ``q̂`` is tiny). With ``lam_reg=0`` this is pure APS: include moods top-down until the
    cumulative probability crosses ``q̂`` (the crossing mood is included — the non-randomized, slightly
    conservative rule that gives ``≥ 1−ε`` coverage). Deterministic (stable sort, no RNG). Pure numpy.
    """
    p = np.asarray(probs_row, dtype=np.float64).ravel()
    m = p.shape[0]
    order = np.argsort(-p, kind="stable")  # moods by prob descending
    cum = np.cumsum(p[order])
    if lam_reg > 0.0:
        cum = cum + lam_reg * np.maximum(0.0, (np.arange(m) + 1) - int(k_reg))
    # First prefix index whose cumulative score ≥ q̂; always keep at least the top-1 mood.
    cross = int(np.searchsorted(cum, float(q_hat), side="left"))
    cross = min(max(cross, 0), m - 1)
    return order[: cross + 1]
