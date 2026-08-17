"""Unit tests for the linear probe / MLP head (moodengine.adapt). AAA.

The 'linear' path is torch-free (sklearn OvR); 'mlp' imports torch lazily and is guarded by
``importorskip``. Covers: OvR macro-F1 > 0.9 on separable synthetic data, determinism at a fixed
seed, the dim guard, the ValueError contract, and persistence (probe_state/probe_from_state
round-trip, save_probe/load_probe npz files, and the validation of hand-built states)."""

from __future__ import annotations

import re

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.adapt import (
    ProbeHead,
    fit_linear_probe,
    load_probe,
    predict_probe,
    probe_from_state,
    probe_state,
    save_probe,
)


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def _separable_dataset(n_per: int = 40, d: int = 16, n_moods: int = 4, seed: int = 0):
    """Well-separated single-label multi-class data as L2-normalized embeddings + OvR multi-hot Y."""
    rng = np.random.default_rng(seed)
    centers = _l2(rng.standard_normal((n_moods, d)))
    X_rows, y_idx = [], []
    for j in range(n_moods):
        pts = centers[j] + 0.05 * rng.standard_normal((n_per, d))
        X_rows.append(pts)
        y_idx.extend([j] * n_per)
    X = _l2(np.vstack(X_rows))
    y_idx = np.array(y_idx)
    Y = np.zeros((X.shape[0], n_moods), dtype=np.float32)
    Y[np.arange(X.shape[0]), y_idx] = 1.0
    return X, Y, [f"m{j}" for j in range(n_moods)]


def _macro_f1(Y_true: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    Y_pred = (logits > 0.0).astype(int)
    return float(f1_score(Y_true.astype(int), Y_pred, average="macro", zero_division=0))


def test_linear_probe_fits_separable_data():
    X, Y, moods = _separable_dataset(seed=1)
    head = fit_linear_probe(X, Y, moods, method="linear")
    assert_that(head).is_instance_of(ProbeHead)
    assert_that(head.method).is_equal_to("linear")
    assert_that(head.W.shape).is_equal_to((len(moods), X.shape[1]))
    assert_that(head.b.shape).is_equal_to((len(moods),))
    assert_that(head.dim).is_equal_to(X.shape[1])
    assert_that(head.hidden).is_none()
    logits = predict_probe(head, X)
    assert_that(logits.shape).is_equal_to(Y.shape)
    assert_that(_macro_f1(Y, logits)).is_greater_than(0.9)  # generalizes the zero-shot prior


def _imbalanced_dataset(counts, d=512, noise=0.6, seed=0):
    """Single-label data with a deliberately RARE last class, at CLAP's real width.

    The dimensionality matters: the rare-mood collapse this guards against does not appear on
    low-dimensional, cleanly-separated toy data — it needs the wide, overlapping geometry of frozen
    512-d embeddings, which is the regime the probe actually runs in.
    """
    rng = np.random.default_rng(seed)
    centers = _l2(rng.standard_normal((len(counts), d)))
    X_rows, y_idx = [], []
    for j, count in enumerate(counts):
        X_rows.append(centers[j] + noise * rng.standard_normal((count, d)))
        y_idx.extend([j] * count)
    X = _l2(np.vstack(X_rows))
    y_idx = np.array(y_idx)
    Y = np.zeros((X.shape[0], len(counts)), dtype=np.float32)
    Y[np.arange(X.shape[0]), y_idx] = 1.0
    return X, Y, [f"m{j}" for j in range(len(counts))], y_idx


def test_linear_probe_rare_mood_can_still_win_the_argmax():
    """Teaching a rare mood must not SUPPRESS it. Fitting each One-vs-Rest column unweighted gave
    the 6-of-48 mood an intercept around -2.0 against a weight norm of ~2, i.e. a maximum
    achievable logit below zero — it was never the argmax, not even on its own tracks (0/6), so
    the personalization did the opposite of what the user asked. Class balancing removes the
    majority's grip on that intercept."""
    X, Y, moods, _ = _imbalanced_dataset([14, 14, 14, 6])
    rare = len(moods) - 1
    rare_rows = slice(14 * 3, None)

    logits = predict_probe(fit_linear_probe(X, Y, moods, method="linear"), X)

    wins = int((logits.argmax(axis=1)[rare_rows] == rare).sum())
    assert_that(wins).is_equal_to(6)  # every one of its own tracks, not zero


def test_linear_probe_keeps_a_usable_logit_scale_on_separable_data():
    """A tie-break trap, not a hypothetical: every ``C`` scores a perfect average precision on
    well-separated moods, and sklearn resolves a tie by keeping the FIRST grid candidate. With an
    ascending grid that is the most-regularized fit — weights shrank to ‖W‖ ≈ 0.005 and every
    logit landed inside ±0.005. The ranking stayed perfect, so an argmax-only assertion passes
    while the probe is in fact inert: these logits are ADDED to the zero-shot prior, and a column
    that small changes nothing. Assert the scale, not just the order."""
    X, Y, moods = _separable_dataset(seed=1)

    head = fit_linear_probe(X, Y, moods, method="linear")

    assert_that(float(np.linalg.norm(head.W, axis=1).mean())).is_greater_than(1.0)
    assert_that(_macro_f1(Y, predict_probe(head, X))).is_greater_than(0.9)


def test_linear_probe_reports_held_out_score_per_mood():
    """The head used to report nothing about its own generalization, so a memorizing fit was
    indistinguishable from a good one. ``cv_score`` is that missing number."""
    X, Y, moods = _separable_dataset(seed=1)

    head = fit_linear_probe(X, Y, moods, method="linear")

    assert_that(head.cv_score.shape).is_equal_to((len(moods),))
    assert_that(head.cv_score.dtype).is_equal_to(np.float32)
    assert_that(bool(np.all(np.isfinite(head.cv_score)))).is_true()
    assert_that(float(head.cv_score.min())).is_greater_than(0.9)  # separable data: near-perfect AP


def test_linear_probe_pinned_C_skips_the_sweep_and_reports_no_score():
    """An explicit ``C`` is an opt-out of validation, so there is no held-out number to report —
    ``nan`` says that, rather than a fabricated score."""
    X, Y, moods = _separable_dataset(seed=1)

    head = fit_linear_probe(X, Y, moods, method="linear", C=1.0)

    assert_that(bool(np.all(np.isnan(head.cv_score)))).is_true()


def test_linear_probe_single_positive_column_is_fit_but_unscored():
    """One positive example cannot be stratified across folds. The column is still fit (its
    ranking is usable) but must carry ``nan`` rather than an unvalidated score."""
    X, Y, moods = _separable_dataset(n_per=10, seed=4)
    Y[:, 0] = 0.0
    Y[0, 0] = 1.0  # exactly one positive for mood 0

    head = fit_linear_probe(X, Y, moods, method="linear")

    assert_that(bool(np.isnan(head.cv_score[0]))).is_true()
    assert_that(bool(np.any(head.W[0] != 0.0))).is_true()  # fit happened anyway


def test_probe_state_without_cv_score_loads_as_unmeasured():
    """A head persisted before ``cv_score`` existed is not corrupt — it genuinely carries no
    validation score, and all-nan is the faithful reading. Dropping the key must not raise."""
    X, Y, moods = _separable_dataset(seed=1)
    state = dict(probe_state(fit_linear_probe(X, Y, moods, method="linear")))
    del state["cv_score"]

    restored = probe_from_state(state)

    assert_that(restored.cv_score.shape).is_equal_to((len(moods),))
    assert_that(bool(np.all(np.isnan(restored.cv_score)))).is_true()


def test_probe_state_mis_sized_cv_score_raises():
    """Present but wrong-length IS corrupt, unlike absent — it must fail loudly at load."""
    X, Y, moods = _separable_dataset(seed=1)
    state = dict(probe_state(fit_linear_probe(X, Y, moods, method="linear")))
    state["cv_score"] = state["cv_score"][:-1]

    with pytest.raises(ValueError, match=r"cv_score has shape.*one held-out score per mood"):
        probe_from_state(state)


def test_linear_probe_is_deterministic():
    X, Y, moods = _separable_dataset(seed=2)
    a = fit_linear_probe(X, Y, moods, method="linear", seed=7)
    b = fit_linear_probe(X, Y, moods, method="linear", seed=7)
    assert_that(np.array_equal(a.W, b.W)).is_true()  # sklearn lbfgs is deterministic on fixed data
    assert_that(np.array_equal(a.b, b.b)).is_true()


def test_predict_probe_dim_guard():
    X, Y, moods = _separable_dataset(seed=3)
    head = fit_linear_probe(X, Y, moods, method="linear")
    with pytest.raises(ValueError):
        predict_probe(head, X[:, :-1])  # wrong input width


def test_fit_rejects_degenerate_shapes():
    X, Y, moods = _separable_dataset(seed=4)
    with pytest.raises(ValueError):
        fit_linear_probe(X[:1], Y[:1], moods)  # n < 2
    with pytest.raises(ValueError):
        fit_linear_probe(X, Y[:, :1], moods[:1])  # n_moods < 2
    with pytest.raises(ValueError):
        fit_linear_probe(X, Y, moods[:-1])  # mood_names misaligned with Y columns
    with pytest.raises(ValueError):
        fit_linear_probe(X, Y, moods, method="bogus")


def test_degenerate_column_gets_saturated_bias():
    # A mood present in NO row -> constant negative logit; present in EVERY row -> constant positive.
    X, Y, moods = _separable_dataset(seed=5)
    Y2 = Y.copy()
    Y2[:, 0] = 0.0  # mood 0 never positive
    Y2[:, 1] = 1.0  # mood 1 always positive
    head = fit_linear_probe(X, Y2, moods, method="linear")
    logits = predict_probe(head, X)
    assert_that(bool(np.all(logits[:, 0] < 0.0))).is_true()
    assert_that(bool(np.all(logits[:, 1] > 0.0))).is_true()
    assert_that(float(np.abs(head.W[0]).max())).is_equal_to(0.0)  # no fit for a degenerate column


def test_mlp_probe_fits_and_predicts_torch_free_inference():
    pytest.importorskip("torch")
    X, Y, moods = _separable_dataset(seed=6)
    head = fit_linear_probe(X, Y, moods, method="mlp", seed=0)
    assert_that(head.method).is_equal_to("mlp")
    assert_that(head.hidden).is_not_none()
    assert_that(len(head.hidden)).is_equal_to(4)
    assert_that(head.dim).is_equal_to(X.shape[1])
    logits = predict_probe(head, X)  # pure numpy forward pass (no torch)
    assert_that(logits.shape).is_equal_to(Y.shape)
    assert_that(_macro_f1(Y, logits)).is_greater_than(0.9)


def test_mlp_probe_reports_held_out_score_when_a_holdout_exists():
    """The MLP head gained the same honesty contract as the linear one: a slice is held out to
    early-stop on, and ``cv_score`` reports average precision there. Untested, three claims in its
    docstring had already drifted from the code."""
    pytest.importorskip("torch")
    X, Y, moods = _separable_dataset(n_per=10, seed=8)  # 40 rows -> an 8-row holdout

    head = fit_linear_probe(X, Y, moods, method="mlp", seed=0)

    assert_that(head.cv_score.shape).is_equal_to((len(moods),))
    assert_that(head.cv_score.dtype).is_equal_to(np.float32)
    assert_that(bool(np.any(np.isfinite(head.cv_score)))).is_true()


def test_mlp_probe_without_room_for_a_holdout_reports_no_score():
    """A holdout needs one validation row AND two training rows, so below 3 rows there is nothing
    to hold out — ``nan`` says so rather than reporting a training-set number as validation."""
    pytest.importorskip("torch")
    X = _l2(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    Y = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    head = fit_linear_probe(X, Y, ["a", "b"], method="mlp", seed=0)

    assert_that(bool(np.all(np.isnan(head.cv_score)))).is_true()


def test_mlp_probe_early_stop_beats_the_full_budget_on_memorizable_data():
    """Early stopping must actually bite: restoring the best-validation weights has to generalize
    at least as well as running every epoch, which on a few dozen rows is pure memorization."""
    pytest.importorskip("torch")
    X, Y, moods = _separable_dataset(n_per=12, seed=9)
    train, test = slice(None, 36), slice(36, None)

    head = fit_linear_probe(X[train], Y[train], moods, method="mlp", seed=0)

    assert_that(_macro_f1(Y[test], predict_probe(head, X[test]))).is_greater_than(0.9)


def test_probe_state_round_trip_linear_is_byte_identical():
    X, Y, moods = _separable_dataset(seed=7)
    head = fit_linear_probe(X, Y, moods, method="linear")

    restored = probe_from_state(probe_state(head))

    assert_that(restored.mood_names).is_equal_to(head.mood_names)
    assert_that(restored.method).is_equal_to("linear")
    assert_that(restored.dim).is_equal_to(head.dim)
    assert_that(restored.hidden).is_none()
    np.testing.assert_array_equal(restored.W, head.W, strict=True)  # bytes AND dtype
    np.testing.assert_array_equal(restored.b, head.b, strict=True)


def test_probe_state_round_trip_mlp_preserves_hidden_weights():
    pytest.importorskip("torch")
    X, Y, moods = _separable_dataset(seed=8)
    head = fit_linear_probe(X, Y, moods, method="mlp", seed=0)

    restored = probe_from_state(probe_state(head))

    assert_that(restored.method).is_equal_to("mlp")
    assert_that(restored.hidden).is_not_none()
    assert_that(len(restored.hidden)).is_equal_to(4)
    for original, roundtripped in zip(head.hidden, restored.hidden):
        np.testing.assert_array_equal(roundtripped, original, strict=True)


def test_probe_round_trip_preserves_predictions():
    X, Y, moods = _separable_dataset(seed=9)
    head = fit_linear_probe(X, Y, moods, method="linear")
    before = predict_probe(head, X)

    after = predict_probe(probe_from_state(probe_state(head)), X)

    np.testing.assert_array_equal(after, before)  # same weights bytes -> same logits bytes


@pytest.mark.parametrize("filename", ["head.npz", "head.probe"])
def test_save_probe_writes_exact_path_and_loads_back(tmp_path, filename):
    # A suffix-less filename pins the open-handle contract: np.savez alone would append ".npz".
    X, Y, moods = _separable_dataset(seed=10)
    head = fit_linear_probe(X, Y, moods, method="linear")
    target = tmp_path / filename

    written = save_probe(head, target)
    loaded = load_probe(written)

    assert_that(written).is_equal_to(target)
    assert_that(target.exists()).is_true()
    assert_that((tmp_path / f"{filename}.npz").exists()).is_false()
    np.testing.assert_array_equal(loaded.W, head.W, strict=True)
    np.testing.assert_array_equal(predict_probe(loaded, X), predict_probe(head, X))


def test_load_probe_missing_path_raises_file_not_found(tmp_path):
    missing = tmp_path / "nowhere.npz"

    with pytest.raises(FileNotFoundError, match=re.escape(f"not found: {missing}")):
        load_probe(missing)


def test_probe_from_state_missing_key_raises():
    X, Y, moods = _separable_dataset(seed=11)
    state = probe_state(fit_linear_probe(X, Y, moods, method="linear"))
    del state["W"]

    with pytest.raises(ValueError, match=r"missing keys \['W'\]"):
        probe_from_state(state)


def test_probe_from_state_wrong_schema_raises():
    X, Y, moods = _separable_dataset(seed=12)
    state = probe_state(fit_linear_probe(X, Y, moods, method="linear"))
    state["schema"] = np.array("moodengine.probe/999")

    with pytest.raises(ValueError, match=r"moodengine\.probe/999.*expected.*moodengine\.probe/1"):
        probe_from_state(state)


def test_probe_from_state_mood_row_mismatch_raises():
    X, Y, moods = _separable_dataset(seed=13)
    state = probe_state(fit_linear_probe(X, Y, moods, method="linear"))
    state["W"] = state["W"][:-1]  # one weight row fewer than mood_names

    with pytest.raises(ValueError, match=r"rows but mood_names"):
        probe_from_state(state)


def test_probe_from_state_unknown_method_raises():
    X, Y, moods = _separable_dataset(seed=14)
    state = probe_state(fit_linear_probe(X, Y, moods, method="linear"))
    state["method"] = np.array("bogus")

    with pytest.raises(ValueError, match=r"'bogus'.*expected 'linear' \| 'mlp'"):
        probe_from_state(state)


def test_probe_from_state_bias_shape_mismatch_raises():
    X, Y, moods = _separable_dataset(seed=15)
    state = probe_state(fit_linear_probe(X, Y, moods, method="linear"))
    state["b"] = state["b"][:-1]  # one bias fewer than mood_names

    with pytest.raises(ValueError, match=r"b has shape.*one bias per mood"):
        probe_from_state(state)


def test_probe_from_state_mlp_hidden_shape_mismatch_raises():
    # Hand-build a consistent mlp state, then corrupt W2's width: the load must fail loudly
    # instead of deferring to an opaque broadcast error inside predict_probe.
    d, h, n_moods = 6, 4, 3
    head = ProbeHead(
        mood_names=[f"m{i}" for i in range(n_moods)],
        W=np.zeros((n_moods, d), dtype=np.float32),
        b=np.zeros(n_moods, dtype=np.float32),
        method="mlp",
        hidden=(
            np.zeros((d, h), dtype=np.float32),
            np.zeros(h, dtype=np.float32),
            np.zeros((h, n_moods), dtype=np.float32),
            np.zeros(n_moods, dtype=np.float32),
        ),
        dim=d,
        cv_score=np.full(n_moods, np.nan, dtype=np.float32),
    )
    state = probe_state(head)
    state["hidden2"] = state["hidden2"][:, :-1]  # W2 loses one mood column

    with pytest.raises(ValueError, match=r"hidden2 \(W2\) has shape"):
        probe_from_state(state)
