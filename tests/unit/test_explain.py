"""Unit tests for the explainability primitives (moodengine.explain). AAA.

Pure numpy + sklearn, torch-free. Covers: the Shapley axioms (efficiency / symmetry / dummy) + the
``n_players ≤ 8`` guard; surrogate fidelity on separable signals + determinism; the exact
interventional Shapley efficiency; exact ⟷ TreeSHAP concordance (skipped when ``shap`` is absent);
the Wachter counterfactual actually flipping / honestly returning ``found=False``."""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.exceptions import MissingDependencyError
from moodengine.explain import (
    Counterfactual,
    SignalSurrogate,
    counterfactual,
    fit_signal_surrogate,
    shapley_exact,
    surrogate_shap,
)


# --- shapley_exact axioms ----------------------------------------------------
def test_shapley_efficiency_additive_game():
    # A purely additive game v(S) = Σ_{i∈S} w_i -> φ_i == w_i, and Σφ == v(full) − v(∅).
    w = np.array([1.0, -2.0, 3.5, 0.0])
    value = lambda S: float(sum(w[i] for i in S))  # noqa: E731
    phi = shapley_exact(value, len(w))
    assert_that(np.allclose(phi, w, atol=1e-9)).is_true()
    full = frozenset(range(len(w)))
    assert_that(float(phi.sum())).is_close_to(value(full) - value(frozenset()), 1e-9)


def test_shapley_efficiency_nonlinear_game():
    # Efficiency (Σφ == v(N) − v(∅)) must hold for ANY value-function, incl. interactions.
    rng = np.random.default_rng(0)
    payoff = {}
    n = 5
    for mask in range(1 << n):
        payoff[frozenset(i for i in range(n) if (mask >> i) & 1)] = float(rng.standard_normal())
    value = lambda S: payoff[frozenset(S)]  # noqa: E731
    phi = shapley_exact(value, n)
    assert_that(float(phi.sum())).is_close_to(value(frozenset(range(n))) - value(frozenset()), 1e-9)


def test_shapley_symmetry_and_dummy():
    # Players 0 and 1 are interchangeable (game depends only on how many of {0,1} are in S);
    # player 2 is a dummy (never changes the payoff).
    def value(S):
        return float(len(set(S) & {0, 1}))  # symmetric in 0,1; ignores 2

    phi = shapley_exact(value, 3)
    assert_that(phi[0]).is_close_to(phi[1], 1e-9)  # symmetry
    assert_that(float(phi[2])).is_close_to(0.0, 1e-9)  # dummy


def test_shapley_guards_player_cap_and_empty():
    with pytest.raises(ValueError, match=r"exceeds the exact-Shapley cap"):
        shapley_exact(lambda S: 0.0, 9)  # > 8 -> 2^n guard
    assert_that(shapley_exact(lambda S: 0.0, 0).shape).is_equal_to((0,))


# --- surrogate fidelity + determinism ---------------------------------------
def _separable_signals(n_per: int = 30, seed: int = 0):
    """Three moods separable by a single signal (feature 0), so a shallow tree is ~perfectly faithful.
    Features: [discriminative, noise, noise]."""
    rng = np.random.default_rng(seed)
    S_rows, y = [], []
    for j, center in enumerate((0.0, 5.0, 10.0)):
        col0 = center + 0.3 * rng.standard_normal(n_per)
        noise = rng.standard_normal((n_per, 2))
        S_rows.append(np.column_stack([col0, noise]))
        y.extend([j] * n_per)
    return np.vstack(S_rows), np.array(y), ["disc", "n1", "n2"], ["calm", "energetic", "dark"]


def test_surrogate_fits_and_measures_fidelity():
    S, y, feats, moods = _separable_signals(seed=1)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    assert_that(surr).is_instance_of(SignalSurrogate)
    assert_that(surr.kind).is_equal_to("tree")
    assert_that(surr.feature_names).is_equal_to(feats)
    assert_that(set(surr.mood_names).issubset(set(moods))).is_true()  # mood-first, ⊆ vocabulary
    assert_that(surr.baseline.shape).is_equal_to((3,))
    assert_that(surr.fidelity).is_greater_than(0.9)  # measured CV accuracy, never hard-coded


def test_surrogate_is_deterministic():
    S, y, feats, moods = _separable_signals(seed=2)
    a = fit_signal_surrogate(S, y, feats, moods, kind="tree", seed=7)
    b = fit_signal_surrogate(S, y, feats, moods, kind="tree", seed=7)
    assert_that(np.array_equal(a.baseline, b.baseline)).is_true()
    assert_that(float(a.fidelity)).is_equal_to(float(b.fidelity))


def test_surrogate_rejects_degenerate():
    S, y, feats, moods = _separable_signals(seed=3)
    with pytest.raises(ValueError, match=r"at least 2 rows"):
        fit_signal_surrogate(S[:1], y[:1], feats, moods)  # < 2 rows
    with pytest.raises(ValueError, match=r"at least 2 distinct moods"):
        fit_signal_surrogate(S, np.zeros_like(y), feats, moods)  # < 2 distinct moods
    with pytest.raises(ValueError, match=r"feature_names must align"):
        fit_signal_surrogate(S, y, feats[:-1], moods)  # feature_names misaligned
    with pytest.raises(ValueError, match=r"unknown kind"):
        fit_signal_surrogate(S, y, feats, moods, kind="bogus")


# --- surrogate_shap efficiency + concordance --------------------------------
def test_surrogate_shap_exact_efficiency():
    S, y, feats, moods = _separable_signals(seed=4)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = S[y == 1][0]  # a track read as mood index 1
    mood_idx = surr.mood_names.index("energetic")
    phi = surrogate_shap(surr, x, mood_idx, backend="exact")
    p_x = float(surr.model.predict_proba(x.reshape(1, -1))[0][mood_idx])
    p_base = float(surr.model.predict_proba(surr.baseline.reshape(1, -1))[0][mood_idx])
    assert_that(float(phi.sum())).is_close_to(p_x - p_base, 1e-6)  # interventional efficiency


def test_surrogate_shap_baseline_feature_gets_zero():
    # A signal already AT its baseline must get φ == 0 (so an unextracted signal, filled with the
    # baseline by the caller, is never mis-attributed).
    S, y, feats, moods = _separable_signals(seed=5)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = surr.baseline.copy()
    x[0] = S[y == 2][0][0]  # move only the discriminative signal off baseline
    phi = surrogate_shap(surr, x, 0, backend="exact")
    assert_that(np.allclose(phi[1:], 0.0, atol=1e-9)).is_true()  # untouched signals -> 0


def test_surrogate_shap_treeshap_concords_with_exact():
    pytest.importorskip("shap")
    S, y, feats, moods = _separable_signals(seed=6)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = S[y == 2][0]
    exact = surrogate_shap(surr, x, 2, backend="exact")
    tree = surrogate_shap(surr, x, 2, backend="treeshap")
    assert_that(np.allclose(exact, tree, rtol=1e-3, atol=1e-6)).is_true()


def test_surrogate_shap_treeshap_without_shap_raises(monkeypatch):
    import builtins

    S, y, feats, moods = _separable_signals(seed=7)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    real_import = builtins.__import__

    def _no_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("shap absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_shap)
    with pytest.raises(MissingDependencyError, match=r"moodengine\[explain\]"):
        surrogate_shap(surr, S[0], 0, backend="treeshap")


# --- counterfactual (Wachter) -----------------------------------------------
def test_counterfactual_flips_prediction():
    S, y, feats, moods = _separable_signals(seed=8)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = S[y == 0][0]  # currently read as mood 0 (low 'disc')
    target = 2  # mood 2 lives at high 'disc'
    mad = np.maximum(np.median(np.abs(S - np.median(S, axis=0)), axis=0), 1e-6)
    bounds = np.column_stack([S.min(axis=0), S.max(axis=0)])
    cf = counterfactual(surr, x, target, mad=mad, bounds=bounds)
    assert_that(cf).is_instance_of(Counterfactual)
    assert_that(cf.found).is_true()
    flipped = int(np.argmax(surr.model.predict_proba((x + cf.deltas).reshape(1, -1))[0]))
    assert_that(flipped).is_equal_to(target)  # the CF actually flips the surrogate


def test_counterfactual_is_deterministic():
    S, y, feats, moods = _separable_signals(seed=9)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = S[y == 0][0]
    mad = np.maximum(np.median(np.abs(S - np.median(S, axis=0)), axis=0), 1e-6)
    bounds = np.column_stack([S.min(axis=0), S.max(axis=0)])
    a = counterfactual(surr, x, 2, mad=mad, bounds=bounds)
    b = counterfactual(surr, x, 2, mad=mad, bounds=bounds)
    assert_that(np.array_equal(a.deltas, b.deltas)).is_true()


def test_counterfactual_not_found_when_target_unreachable():
    # Constrain bounds to the current point -> no perturbation is possible -> found=False, no fake.
    S, y, feats, moods = _separable_signals(seed=10)
    surr = fit_signal_surrogate(S, y, feats, moods, kind="tree")
    x = S[y == 0][0]
    mad = np.ones(3)
    bounds = np.column_stack([x, x])  # zero-width bounds
    cf = counterfactual(surr, x, 2, mad=mad, bounds=bounds)
    assert_that(cf.found).is_false()
    assert_that(bool(np.all(cf.deltas == 0.0))).is_true()


def test_explain_module_is_torch_free():
    # Importing the module + exercising the default (non-treeshap) paths must never pull in torch.
    # Checked in an ISOLATED interpreter (session-global sys.modules is polluted by torch-using
    # tests like test_adapt_probe's mlp case, so an in-process assert would be flaky).
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from assertpy import assert_that
        from moodengine.explain import fit_signal_surrogate, surrogate_shap, counterfactual
        rng = np.random.default_rng(0)
        S = np.vstack([np.column_stack([c + 0.3*rng.standard_normal(20), rng.standard_normal((20,2))])
                       for c in (0.0, 5.0, 10.0)])
        y = np.repeat([0, 1, 2], 20)
        surr = fit_signal_surrogate(S, y, ["a","b","c"], ["calm","energetic","dark"])
        surrogate_shap(surr, S[0], 0, backend="exact")
        counterfactual(surr, S[0], 2,
                       mad=np.ones(3), bounds=np.column_stack([S.min(0), S.max(0)]))
        assert_that("torch" in sys.modules).described_as(
            "explain default paths must not import torch"
        ).is_false()
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert_that(proc.returncode).described_as(proc.stderr).is_equal_to(0)
