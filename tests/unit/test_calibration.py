"""Tests for :mod:`moodengine.calibration` — Guo+'17 temperature scaling + baselines (torch-free).

The central invariants: temperature scaling improves NLL and ECE on an over-confident set while
leaving the argmax (top-1 accuracy) untouched; reliability is monotone-ish on a well-calibrated
synthetic set. ECE is imported from :mod:`moodengine.evaluation` — never redefined here.
"""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.calibration import (
    _PROB_FLOOR,
    _aps_scores,
    _softmax_T,
    _temperature_objective,
    aps_threshold,
    entropy,
    fit_temperature,
    isotonic_calibrate,
    margin,
    negative_log_likelihood,
    platt_scale,
    prediction_set,
    reliability_diagram,
)
from moodengine.evaluation import expected_calibration_error


def _softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _overconfident_set(seed=0, n=200, n_classes=4):
    """Logits whose argmax is right ~70% of the time but whose peaked scale makes softmax(·) far
    too confident — the canonical case temperature scaling fixes."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    logits = rng.standard_normal((n, n_classes))
    # Make the true class usually (but not always) the argmax, then blow up the scale (over-confident).
    for i, y in enumerate(labels):
        if rng.random() < 0.7:
            logits[i, y] = logits[i].max() + 1.0
    logits *= 6.0
    return logits, labels


# --- fit_temperature ---------------------------------------------------------
def test_temperature_scaling_improves_nll_and_ece() -> None:
    """On an over-confident set, the fitted T lowers both NLL and ECE (Guo+'17)."""
    logits, labels = _overconfident_set()
    probs_before = _softmax(logits, 1.0)
    conf_b, pred_b = probs_before.max(1), probs_before.argmax(1)
    correct = pred_b == labels
    ece_before, _ = expected_calibration_error(conf_b, correct)
    nll_before = negative_log_likelihood(probs_before, labels)

    T = fit_temperature(logits, labels)
    assert_that(T).is_greater_than(1.0)  # over-confident -> soften

    probs_after = _softmax(logits, T)
    ece_after, _ = expected_calibration_error(probs_after.max(1), probs_after.argmax(1) == labels)
    nll_after = negative_log_likelihood(probs_after, labels)

    assert_that(nll_after).is_less_than_or_equal_to(nll_before + 1e-9)
    assert_that(ece_after).is_less_than(ece_before)


def test_temperature_is_argmax_invariant() -> None:
    """T > 0 is monotone -> it never reorders the argmax (top-1 accuracy is invariant)."""
    logits, labels = _overconfident_set(seed=1)
    T = fit_temperature(logits, labels)
    before = logits.argmax(1)
    after = (logits / T).argmax(1)
    assert_that(np.array_equal(before, after)).is_true()


def test_fit_temperature_degenerate() -> None:
    """n < 1 -> 1.0 (nothing to fit); never raises."""
    assert_that(fit_temperature(np.empty((0, 3)), np.empty((0,)))).is_equal_to(1.0)
    assert_that(fit_temperature(np.array([[1.0, 2.0, 3.0]]), np.array([2]))).is_greater_than(0.0)


# --- negative_log_likelihood -------------------------------------------------
def test_nll_hand_value_and_empty() -> None:
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 1])
    expected = -np.mean([np.log(0.7), np.log(0.8)])
    assert_that(negative_log_likelihood(probs, labels)).is_close_to(float(expected), tolerance=1e-9)
    assert_that(negative_log_likelihood(np.empty((0, 2)), np.empty((0,)))).is_equal_to(0.0)


# --- reliability_diagram -----------------------------------------------------
def test_reliability_perfectly_calibrated_is_monotone_and_on_diagonal() -> None:
    """A synthetic set where accuracy tracks confidence -> per-bin accuracy ≈ mean_confidence, and
    accuracy rises with the bin (monotone)."""
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=5000)
    correct = (rng.uniform(0, 1, size=5000) < conf).astype(int)  # P(correct | conf) = conf exactly
    bins = reliability_diagram(conf, correct, n_bins=10)
    assert_that(len(bins)).is_greater_than_or_equal_to(5)
    for b in bins:
        assert_that(b["accuracy"]).is_close_to(
            b["mean_confidence"], tolerance=0.06
        )  # near the diagonal
    accs = [b["accuracy"] for b in bins]
    assert_that(accs).is_equal_to(sorted(accs))  # monotone non-decreasing across bins
    # and this set's ECE is ~0 (imported from evaluation, not redefined here)
    ece, n = expected_calibration_error(conf, correct)
    assert_that(ece).is_less_than(0.05)
    assert_that(n).is_equal_to(5000)


def test_reliability_empty() -> None:
    assert_that(reliability_diagram(np.array([]), np.array([]))).is_equal_to([])


# --- Platt / isotonic baselines ----------------------------------------------
def test_platt_scale_toy_and_degenerate() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=400)
    correct = (rng.uniform(0, 1, size=400) < conf).astype(int)
    a, b = platt_scale(conf, correct)
    assert_that(a).is_greater_than(0.0)  # higher confidence -> higher P(correct)
    assert_that(platt_scale(np.array([0.5, 0.6]), np.array([1, 1]))).is_equal_to(
        (0.0, 0.0)
    )  # single class -> flat


def test_isotonic_is_monotone() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=500)
    correct = (rng.uniform(0, 1, size=500) < conf).astype(int)
    ir = isotonic_calibrate(conf, correct)
    xs = np.linspace(0, 1, 11)
    ys = ir.predict(xs)
    assert_that(bool(np.all(np.diff(ys) >= -1e-9))).is_true()  # non-decreasing


def test_isotonic_empty_returns_usable_regressor() -> None:
    """Empty input must return a FITTED regressor (a safe identity pass-through), never one that
    raises on a later .predict() — the same never-raises-on-degenerate contract as the other fns."""
    ir = isotonic_calibrate([], [])
    out = ir.predict([0.0, 0.5, 1.0])  # must not raise
    assert_that(out).is_length(3)


# --- entropy / margin (intrinsic uncertainty) --------------------------------
def test_entropy_bounds_and_shape() -> None:
    m = 4
    uniform = np.full((1, m), 1.0 / m)
    one_hot = np.eye(m)[[0]]
    assert_that(float(entropy(uniform)[0])).is_close_to(
        float(np.log(m)), tolerance=1e-9
    )  # maximal on the uniform distribution
    assert_that(float(entropy(one_hot)[0])).is_close_to(
        0.0, tolerance=1e-12
    )  # zero on a one-hot (fully certain)
    probs = _softmax(np.random.default_rng(0).standard_normal((7, m)), 1.0)
    ent = entropy(probs)
    assert_that(ent.shape).is_equal_to((7,))  # (n, m) -> (n,)
    assert_that(bool(np.all(ent >= -1e-12))).is_true()
    assert_that(bool(np.all(ent <= np.log(m) + 1e-9))).is_true()  # bounded [0, ln m]


def test_margin_bounds() -> None:
    m = 5
    assert_that(float(margin(np.eye(m)[[0]])[0])).is_close_to(
        1.0, tolerance=1e-9
    )  # one-hot -> gap 1
    assert_that(float(margin(np.full((1, m), 1.0 / m))[0])).is_close_to(
        0.0, tolerance=1e-12
    )  # uniform -> gap 0
    probs = _softmax(np.random.default_rng(1).standard_normal((9, m)), 1.0)
    mg = margin(probs)
    assert_that(mg.shape).is_equal_to((9,))
    assert_that(bool(np.all(mg >= -1e-12))).is_true()
    assert_that(bool(np.all(mg <= 1.0 + 1e-12))).is_true()  # bounded [0, 1]


# --- APS conformal: coverage, monotonicity, non-empty, determinism -----------
def _aps_synthetic(seed=0, n=500, m=6):
    """Softmax rows + true labels drawn from those very probabilities (well-specified), so a
    coverage-1−ε set really should contain the truth ≈ 1−ε of the time."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, m)) * 2.0
    probs = _softmax(logits, 1.0)
    labels = np.array([rng.choice(m, p=probs[i]) for i in range(n)])
    return probs, labels


def test_aps_empirical_coverage_meets_target() -> None:
    """Split-conformal on a calibration half; measured coverage on a fresh test half is ≥ 1−ε − 1/(n+1)."""
    probs, labels = _aps_synthetic(seed=0, n=1000, m=6)
    cal_p, cal_y = probs[:500], labels[:500]
    test_p, test_y = probs[500:], labels[500:]
    eps = 0.1
    q = aps_threshold(cal_p, cal_y, coverage_target=1 - eps)
    covered = [test_y[i] in prediction_set(test_p[i], q) for i in range(len(test_y))]
    coverage = float(np.mean(covered))
    tol = 1.0 / (len(cal_y) + 1)
    assert_that(coverage).is_greater_than_or_equal_to(
        (1 - eps) - tol - 0.02
    )  # small slack for the finite test half


def test_aps_monotone_coverage_raises_qhat_and_set_size() -> None:
    probs, labels = _aps_synthetic(seed=1, n=600, m=6)
    targets = [0.5, 0.7, 0.9, 0.99]
    qs, sizes = [], []
    for c in targets:
        q = aps_threshold(probs, labels, coverage_target=c)
        qs.append(q)
        sizes.append(float(np.mean([len(prediction_set(p, q)) for p in probs])))
    assert_that(qs).is_equal_to(sorted(qs))  # coverage↑ ⇒ q̂↑
    assert_that(sizes).is_equal_to(sorted(sizes))  # q̂↑ ⇒ mean set size non-decreasing


def test_prediction_set_never_empty_and_ordered_by_prob() -> None:
    p = np.array([0.9, 0.05, 0.03, 0.02])
    s = prediction_set(p, q_hat=0.0)  # even a zero threshold keeps the top-1
    assert_that(len(s)).is_greater_than_or_equal_to(1)
    assert_that(int(s[0])).is_equal_to(0)  # ordered by prob descending -> top-1 first
    full = prediction_set(p, q_hat=1.0)
    assert_that(list(full)).is_equal_to(
        [0, 1, 2, 3]
    )  # threshold 1 includes everything, in prob↓ order


def test_prediction_set_singleton_when_confident_and_low_coverage() -> None:
    probs, labels = _aps_synthetic(seed=2, n=400, m=6)
    q = aps_threshold(probs, labels, coverage_target=0.3)  # low target -> tight sets
    confident = np.array([0.97, 0.01, 0.008, 0.006, 0.004, 0.002])
    assert_that(list(prediction_set(confident, q))).is_equal_to(
        [0]
    )  # a dominating top-1 -> just that mood


def test_aps_deterministic_no_jitter() -> None:
    probs, labels = _aps_synthetic(seed=3, n=300, m=5)
    q1 = aps_threshold(probs, labels, coverage_target=0.9, rng_jitter=False)
    q2 = aps_threshold(probs, labels, coverage_target=0.9, rng_jitter=False)
    assert_that(q1).is_equal_to(q2)  # reproducible


def test_aps_pure_when_reg_disabled() -> None:
    """k_reg=lam_reg=0 (default) ⇒ pure APS: the set is determined by the cumulative prob alone,
    and q̂ ∈ [0, 1]."""
    probs, labels = _aps_synthetic(seed=4, n=300, m=5)
    q = aps_threshold(probs, labels, coverage_target=0.9, k_reg=0, lam_reg=0.0)
    assert_that(q).is_between(0.0, 1.0)
    p = probs[0]
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    expected_len = int(np.searchsorted(cum, q, side="left")) + 1
    assert_that(len(prediction_set(p, q))).is_equal_to(min(max(expected_len, 1), len(p)))


def test_aps_threshold_rejects_a_true_index_outside_the_vocabulary() -> None:
    """``-1`` is this repo's own "unknown mood" sentinel, and ``np.argmax`` on an all-False row
    returns 0 — so an unlabelled calibration row was scored as if its true mood had ranked FIRST,
    the cheapest possible score. Measured with a quarter of the labels set to -1, q̂ fell 0.674 to
    0.590 and coverage 0.778 to 0.722: the sets got TIGHTER the more corrupt the input was, while
    the guarantee was still reported as intact. Clipping would fix the number and keep the wrong
    guarantee, so this raises."""
    rng = np.random.default_rng(0)
    probs = rng.random((20, 5))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 5, 20)
    labels[3] = -1  # an unlabelled row that slipped into the calibration set

    with pytest.raises(ValueError, match=r"cal_true_idx has 1 entry.*outside \[0, 5\)"):
        aps_threshold(probs, labels, coverage_target=0.9)


def test_aps_threshold_rng_jitter_warns_that_it_never_did_anything() -> None:
    """The parameter named the randomized APS variant but was never read, so passing it produced
    the deterministic threshold while implying otherwise. It must say so rather than keep pretending."""
    rng = np.random.default_rng(0)
    probs = rng.random((20, 5))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 5, 20)

    with pytest.warns(DeprecationWarning, match="never been implemented"):
        jittered = aps_threshold(probs, labels, coverage_target=0.9, rng_jitter=True)

    assert_that(jittered).is_equal_to(aps_threshold(probs, labels, coverage_target=0.9))


def test_aps_threshold_degenerate_returns_conservative() -> None:
    assert_that(aps_threshold(np.empty((0, 4)), np.empty((0,)), 0.9)).is_equal_to(
        1.0
    )  # nothing to calibrate
    # target unattainable at tiny n -> include-all (never a fabricated tight q̂)
    assert_that(aps_threshold(np.array([[0.7, 0.3]]), np.array([0]), 0.99)).is_equal_to(1.0)


def test_aps_coverage_zero_collapses_to_top1() -> None:
    """coverage_target=0 (the degenerate low end) must give the TIGHTEST set — just the top-1 mood —
    not wrap the k=0 index to scores[-1] (the max). Boundary of the monotonicity guarantee."""
    probs, labels = _aps_synthetic(seed=5, n=300, m=6)
    q0 = aps_threshold(probs, labels, coverage_target=0.0)
    assert_that(q0).is_equal_to(0.0)
    confident = np.array([0.5, 0.2, 0.15, 0.1, 0.03, 0.02])
    assert_that(list(prediction_set(confident, q0))).is_equal_to([0])  # tightest = top-1 only
    # and it is ≤ every positive target's q̂ (monotone through 0, no inversion)
    assert_that(q0).is_less_than_or_equal_to(aps_threshold(probs, labels, coverage_target=0.1))


def test_raps_include_all_when_target_unattainable() -> None:
    """With RAPS (lam_reg>0) the k>n include-all branch must ACTUALLY include every mood: a fixed 1.0
    would be crossed early by the rank penalty and truncate the set. +inf guarantees the full set."""
    m = 8
    probs, labels = _aps_synthetic(seed=6, n=6, m=m)
    q = aps_threshold(
        probs, labels, coverage_target=0.99, k_reg=0, lam_reg=0.5
    )  # k=ceil(7*0.99)=7 > 6
    assert_that(q).is_equal_to(float("inf"))
    row = probs[0]
    assert_that(len(prediction_set(row, q, k_reg=0, lam_reg=0.5))).is_equal_to(
        m
    )  # every mood, not a truncated prefix


def _aps_scores_by_sorting(probs, true_idx, k_reg=0, lam_reg=0.0):
    """Reference APS score via an explicit stable argsort — the definition, written the slow way."""
    P = np.asarray(probs, dtype=np.float64)
    y = np.asarray(true_idx).astype(int).ravel()
    order = np.argsort(-P, axis=1, kind="stable")
    cum = np.cumsum(np.take_along_axis(P, order, axis=1), axis=1)
    ranks = np.argmax(order == y[:, None], axis=1)
    scores = cum[np.arange(P.shape[0]), ranks]
    if lam_reg > 0.0:
        scores = scores + lam_reg * np.maximum(0.0, (ranks + 1) - int(k_reg))
    return scores


@pytest.mark.parametrize("kind", ["all_uniform", "quarter_quantized"])
def test_aps_score_reproduces_the_stable_sort_tie_rule(kind) -> None:
    """The score is a masked row sum, and the mask has to reproduce the sort's tie-break exactly.

    Every other fixture in this file uses continuous softmax rows, where exact ties never occur —
    so nothing here would catch a tie-rule regression. `argsort(-P, kind='stable')` breaks a tie
    toward the SMALLER column index, which the mask encodes as ``(P == p_true) & (cols <= y)``.
    These two fixtures are nothing but ties."""
    n, m = 200, 12
    rng = np.random.default_rng(19)
    if kind == "all_uniform":
        probs = np.full((n, m), 1.0 / m)  # every mood tied with every other
    else:
        counts = rng.integers(0, 5, (n, m)).astype(np.float64)
        counts[:, 0] = np.maximum(counts[:, 0], 1.0)  # never an all-zero row
        probs = counts / counts.sum(axis=1, keepdims=True)  # quarters -> many exact ties
    labels = rng.integers(0, m, n)

    got = _aps_scores(probs, labels, 0, 0.0)

    np.testing.assert_allclose(got, _aps_scores_by_sorting(probs, labels), rtol=0.0, atol=1e-12)


def test_raps_rank_penalty_reaches_the_scores() -> None:
    """The RAPS penalty must be exercised where it changes a SCORE, not only where q̂ is +inf.

    `test_raps_include_all_when_target_unattainable` takes the k>n branch and returns +inf before
    the scores are used, so the ``lam_reg·max(0, rank − k_reg)`` term itself was unpinned. Here the
    sample is large enough that q̂ is a real quantile, and the penalty has to lift the scores of
    rows whose true mood ranks beyond k_reg."""
    probs, labels = _aps_synthetic(seed=11, n=400, m=10)

    plain = _aps_scores(probs, labels, 0, 0.0)
    penalised = _aps_scores(probs, labels, 2, 0.5)

    ranks = np.count_nonzero(probs > probs[np.arange(len(labels)), labels][:, None], axis=1) + 1
    expected = plain + 0.5 * np.maximum(0.0, ranks - 2)
    np.testing.assert_allclose(penalised, expected, rtol=0.0, atol=1e-12)
    assert_that(int(np.count_nonzero(penalised > plain))).is_greater_than(0)  # non-vacuous
    assert_that(float(aps_threshold(probs, labels, 0.9, k_reg=2, lam_reg=0.5))).is_greater_than(
        float(aps_threshold(probs, labels, 0.9))
    )


def test_aps_threshold_rejects_non_finite_probabilities() -> None:
    """A NaN row cannot produce a conformal score, and silently dropping it understates q̂.

    The sorted form gave two different answers to the same broken input — a NaN ranked below the
    true mood vanished from the cumulative sum, a NaN ON the true mood produced NaN — and the
    silent half is the dangerous one: a score missing mass yields a q̂ that is too small, i.e. a
    wrong guarantee rather than a wrong number. Same reasoning as the out-of-range label guard."""
    probs, labels = _aps_synthetic(seed=12, n=20, m=6)
    probs[3, 2] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        aps_threshold(probs, labels, coverage_target=0.9)


@pytest.mark.parametrize("temperature", [1e-3, 0.01, 0.5, 1.0, 3.0, 100.0])
@pytest.mark.parametrize("scale", [0.3, 30.0], ids=["flat-logits", "peaked-logits"])
def test_temperature_objective_equals_the_readable_definition(temperature, scale) -> None:
    """The fast objective must equal ``negative_log_likelihood(_softmax_T(L, T), y)`` at every T.

    `fit_temperature` no longer builds a softmax per evaluation; it uses the logsumexp identity. The
    risk that buys is silent: `negative_log_likelihood` floors the probability at ``_PROB_FLOOR``
    before the log, which in logsumexp form is a CEILING on the per-row loss, and forgetting to
    transcribe it changes nothing until the gaps get large relative to T.

    The peaked-logits × small-T cells are where it bites, and it is not subtle there — with the
    ceiling the objective is ~26.5, without it ~56 800. Pinning the objective rather than the
    returned T is deliberate: T comes out of a minimizer, and in a flat basin two numerically
    equivalent objectives converge to slightly different points, so asserting on T would be both
    weaker and flakier."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((300, 18)) * scale
    labels = rng.integers(0, 18, 300)

    objective = _temperature_objective(logits, labels)

    reference = negative_log_likelihood(_softmax_T(logits, temperature), labels)
    assert_that(objective(temperature)).is_close_to(reference, tolerance=1e-9)


def test_temperature_objective_saturates_at_the_probability_floor() -> None:
    """Non-vacuity guard for the test above: prove the ceiling is actually reached somewhere.

    Without this, the equality test could pass forever on data where the clip never engages and the
    transcription is untested. Here it is engaged and dominant."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((300, 18)) * 30.0
    labels = rng.integers(0, 18, 300)
    gaps = logits - logits.max(axis=1, keepdims=True)
    unclipped = np.log(np.exp(gaps / 1e-3).sum(axis=1)) - gaps[np.arange(300), labels] / 1e-3

    assert_that(bool((unclipped > -np.log(_PROB_FLOOR)).any())).is_true()  # the clip engages
    assert_that(float(unclipped.mean())).is_greater_than(1000.0)  # and dominates if dropped
    assert_that(_temperature_objective(logits, labels)(1e-3)).is_less_than(
        -np.log(_PROB_FLOOR) + 1e-9
    )


@pytest.mark.parametrize("n_bins", [0, 1, 2, 10, 50])
def test_reliability_diagram_handles_every_bin_count(n_bins) -> None:
    """Nothing exercised ``n_bins`` other than 10, and the grouped form breaks at 0.

    The per-bin loop this replaced simply never ran for ``n_bins <= 0`` and returned ``[]``. A
    bincount instead receives the ``-1`` that ``np.clip(digitize, 0, n_bins - 1)`` produces and
    rejects it, so the empty case needs an explicit guard — which would have shipped unnoticed."""
    rng = np.random.default_rng(4)
    confidences = rng.random(500)
    correct = (rng.random(500) < confidences).astype(float)

    bins = reliability_diagram(confidences, correct, n_bins=n_bins)

    assert_that(bins).is_instance_of(list)
    assert_that(len(bins)).is_less_than_or_equal_to(max(n_bins, 0))
    assert_that(sum(b["count"] for b in bins)).is_equal_to(0 if n_bins < 1 else 500)
    assert_that([b["bin_lo"] for b in bins]).is_equal_to(sorted(b["bin_lo"] for b in bins))
    for b in bins:
        assert_that(b["count"]).is_greater_than(0)  # occupied bins only
        assert_that(b["mean_confidence"]).is_between(b["bin_lo"] - 1e-12, b["bin_hi"] + 1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_negative_log_likelihood_is_dtype_independent(dtype) -> None:
    """float32 probabilities must give the same number as float64 ones.

    `labeling.softmax` returns float32, so that is the realistic input — and no test passed one. The
    function gathers the true-class probabilities BEFORE promoting to float64 rather than promoting
    the whole matrix, which is what keeps the allocation proportional to n instead of n x n_classes;
    the promotion is exact, so the value must not move."""
    rng = np.random.default_rng(6)
    z = rng.standard_normal((400, 12))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True)).astype(dtype)
    labels = rng.integers(0, 12, 400)

    value = negative_log_likelihood(probs, labels)

    expected = float(
        -np.mean(
            np.log(
                np.clip(
                    np.asarray(probs, dtype=np.float64)[np.arange(400), labels], _PROB_FLOOR, 1.0
                )
            )
        )
    )
    assert_that(value).is_equal_to(expected)  # exact: promotion after the gather changes nothing


def test_negative_log_likelihood_allocation_does_not_grow_with_the_class_count() -> None:
    """The reason this function gathers before promoting — and the only way to pin it.

    Reading n true-class probabilities out of an (n, n_classes) float32 matrix does not require
    promoting the matrix. Doing so anyway costs 8 bytes per entry for values that are never read:
    measured at n=20 000, promoting first peaks at 1.9 MB for 8 classes and 10.9 MB for 64, while
    gathering first is flat at 0.64 MB for both.

    Asserted as a RATIO across two class counts rather than an absolute byte figure, so it does not
    depend on the allocator: the correct implementation is flat (1.0x) and the regression scales
    with n_classes (5.7x over this pair), which leaves the 2x threshold a wide margin."""
    import tracemalloc

    def peak_bytes(n_classes: int) -> int:
        rng = np.random.default_rng(0)
        z = rng.standard_normal((20_000, n_classes))
        e = np.exp(z - z.max(axis=1, keepdims=True))
        probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        labels = rng.integers(0, n_classes, 20_000)
        negative_log_likelihood(probs, labels)  # warm any lazy allocation
        tracemalloc.start()
        negative_log_likelihood(probs, labels)
        measured = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return measured

    narrow, wide = peak_bytes(8), peak_bytes(64)

    assert_that(wide).is_less_than(2 * narrow)


def test_calibration_module_is_torch_free() -> None:
    """Import + call the calibration fns in a subprocess -> torch must not load (CI guard)."""
    import subprocess
    import sys

    code = (
        "import numpy as np, sys\n"
        "import moodengine.calibration as c\n"
        "L = np.random.default_rng(0).standard_normal((20, 3)); y = np.zeros(20, int)\n"
        "c.fit_temperature(L, y); c.reliability_diagram(np.linspace(0,1,20), np.ones(20)); "
        "c.negative_log_likelihood(np.full((3,3),1/3), np.zeros(3,int)); "
        "c.platt_scale(np.linspace(0,1,20), (np.arange(20)%2)); "
        "c.isotonic_calibrate(np.linspace(0,1,20), (np.arange(20)%2))\n"
        "P=np.full((20,3),1/3); c.entropy(P); c.margin(P); "
        "q=c.aps_threshold(P, np.zeros(20,int), 0.9); c.prediction_set(P[0], q)\n"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).is_equal_to(0)
