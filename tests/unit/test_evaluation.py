"""Tests for :mod:`moodengine.evaluation` — falsifiable metrics (torch-free).

A tiny fake CLAP embedder maps query strings to deterministic vectors so the
text-query retrieval metric is reproducible without a real model.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from assertpy import assert_that

from moodengine.evaluation import (
    average_precision,
    axis_ranking_auc,
    concordance_correlation_coefficient,
    evaluate_against_gold,
    evaluate_text_queries,
    expected_calibration_error,
    load_gold,
    macro_f1,
    ndcg_at_k,
    nmi,
    procrustes_disparity,
    recall_at_k,
    retrieval_precision_at_k,
)


def _hash_unit_vec(text: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _FakeCLAP:
    def __init__(self, dim: int = 3, overrides: dict[str, np.ndarray] | None = None) -> None:
        self.dim = dim
        self.overrides = {k: np.asarray(v, dtype=np.float32) for k, v in (overrides or {}).items()}

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        rows = [
            self.overrides[p] if p in self.overrides else _hash_unit_vec(p, self.dim)
            for p in prompts
        ]
        return np.vstack(rows).astype(np.float32)


# --------------------------------------------------------------------------- #
# axis_ranking_auc
# --------------------------------------------------------------------------- #


def test_axis_ranking_auc_perfect_alignment() -> None:
    """When scores rank tracks identically to the axis, AUC is ~1.0."""
    axis = np.linspace(0.0, 1.0, 20)
    scores = axis.copy()  # perfectly aligned
    assert_that(axis_ranking_auc(scores, axis)).is_close_to(1.0, tolerance=1e-6)


def test_axis_ranking_auc_perfect_inversion() -> None:
    """A perfectly anti-aligned score gives AUC ~0.0."""
    axis = np.linspace(0.0, 1.0, 20)
    scores = -axis
    assert_that(axis_ranking_auc(scores, axis)).is_close_to(0.0, tolerance=1e-6)


def test_axis_ranking_auc_degenerate_returns_half() -> None:
    """Fewer than 2 samples or a single-class split -> chance (0.5)."""
    assert_that(axis_ranking_auc(np.array([1.0]), np.array([0.5]))).is_equal_to(0.5)
    # All axis values equal -> median split collapses to one class.
    assert_that(axis_ranking_auc(np.array([0.1, 0.9, 0.3]), np.array([0.5, 0.5, 0.5]))).is_equal_to(
        0.5
    )


# --------------------------------------------------------------------------- #
# retrieval_precision_at_k
# --------------------------------------------------------------------------- #


def test_retrieval_precision_at_k_counts_hits() -> None:
    """P@k is hits-in-top-k divided by k."""
    ranked = [3, 1, 7, 2, 9]
    relevant = {1, 2, 5}
    assert_that(retrieval_precision_at_k(ranked, relevant, k=3)).is_close_to(1 / 3, tolerance=1e-9)
    assert_that(retrieval_precision_at_k(ranked, relevant, k=5)).is_close_to(2 / 5, tolerance=1e-9)


def test_retrieval_precision_at_k_perfect_and_zero() -> None:
    """All-relevant top-k -> 1.0; nothing relevant -> 0.0."""
    assert_that(retrieval_precision_at_k([0, 1, 2], {0, 1, 2}, k=3)).is_close_to(
        1.0, tolerance=1e-9
    )
    assert_that(retrieval_precision_at_k([0, 1, 2], {9}, k=3)).is_close_to(0.0, tolerance=1e-12)


def test_retrieval_precision_at_k_nonpositive_k() -> None:
    """``k <= 0`` is guarded and returns 0.0."""
    assert_that(retrieval_precision_at_k([0, 1], {0}, k=0)).is_equal_to(0.0)
    assert_that(retrieval_precision_at_k([0, 1], {0}, k=-2)).is_equal_to(0.0)


# --------------------------------------------------------------------------- #
# average_precision
# --------------------------------------------------------------------------- #


def test_average_precision_all_relevant_first_is_one() -> None:
    """Every relevant item in the top ``len(relevant)`` positions -> AP == 1.0 (the maximum)."""
    assert_that(average_precision([0, 1, 2, 9], {0, 1, 2})).is_close_to(1.0, tolerance=1e-9)
    # Order *within* the relevant prefix does not matter — all precisions are still 1.
    assert_that(average_precision([2, 0, 1, 9], {0, 1, 2})).is_close_to(1.0, tolerance=1e-9)


def test_average_precision_hand_value_single_hit_at_rank_two() -> None:
    """One relevant item at rank 2 -> precision 1/2 there, averaged over 1 relevant -> 0.5."""
    assert_that(average_precision([9, 1, 8], {1})).is_close_to(0.5, tolerance=1e-9)


def test_average_precision_hand_value_two_hits() -> None:
    """Hits at ranks 1 and 3 -> (1/1 + 2/3) / 2."""
    expected = (1.0 + 2.0 / 3.0) / 2.0

    assert_that(average_precision([0, 9, 1, 8], {0, 1})).is_close_to(expected, tolerance=1e-9)


def test_average_precision_ranks_relevant_earlier_scores_higher() -> None:
    """Monotone in rank position: promoting the relevant item strictly raises AP."""
    early = average_precision([1, 9, 8], {1})
    late = average_precision([9, 8, 1], {1})

    assert_that(early).is_greater_than(late)
    assert_that(early).is_close_to(1.0, tolerance=1e-9)
    assert_that(late).is_close_to(1.0 / 3.0, tolerance=1e-9)


def test_average_precision_unranked_relevant_caps_below_one() -> None:
    """The denominator is the gold size, so a relevant item absent from the ranking costs AP."""
    # Item 0 is ranked first (precision 1.0) but item 1 never appears -> 1.0 / 2 relevant.
    val = average_precision([0], {0, 1})

    assert_that(val).is_close_to(0.5, tolerance=1e-9)
    assert_that(val).is_less_than(1.0)


def test_average_precision_guards() -> None:
    """No relevant items, or none of them ranked -> 0.0 (never raises)."""
    assert_that(average_precision([0, 1, 2], set())).is_equal_to(0.0)
    assert_that(average_precision([0, 1, 2], {9})).is_equal_to(0.0)
    assert_that(average_precision([], {0})).is_equal_to(0.0)


def test_average_precision_is_bounded_in_unit_interval() -> None:
    """AP stays within [0, 1] across random rankings and gold sets."""
    rng = np.random.default_rng(20260716)

    for _ in range(50):
        ranked = rng.permutation(12).tolist()
        relevant = {int(i) for i in rng.choice(12, size=4, replace=False)}
        val = average_precision(ranked, relevant)

        assert_that(val).is_between(0.0, 1.0)


# --------------------------------------------------------------------------- #
# evaluate_text_queries
# --------------------------------------------------------------------------- #


def test_evaluate_text_queries_perfect_retrieval() -> None:
    """A query embedded onto a track ranks it first -> P@1 == 1.0, AP == 1.0."""
    e = np.eye(4, dtype=np.float32)
    X = e.copy()
    embedder = _FakeCLAP(dim=4, overrides={"calm": e[2]})
    res = evaluate_text_queries({"calm": {2}}, X, embedder, k=1)

    assert_that(set(res.keys())).is_equal_to(
        {"per_query", "macro_precision_at_k", "macro_map", "k"}
    )
    pq = res["per_query"]["calm"]
    assert_that(pq["precision_at_k"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(pq["average_precision"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(pq["n_relevant"]).is_equal_to(1)
    assert_that(res["macro_precision_at_k"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(res["macro_map"]).is_close_to(1.0, tolerance=1e-9)


def test_evaluate_text_queries_empty_guards() -> None:
    """Empty X or empty queries -> zeroed macros, no embedder call needed."""
    embedder = _FakeCLAP(dim=4)
    out = evaluate_text_queries({}, np.empty((0, 0), dtype=np.float32), embedder, k=5)
    assert_that(out["macro_precision_at_k"]).is_equal_to(0.0)
    assert_that(out["macro_map"]).is_equal_to(0.0)
    assert_that(out["per_query"]).is_equal_to({})


# --------------------------------------------------------------------------- #
# load_gold
# --------------------------------------------------------------------------- #


def test_load_gold_missing_returns_empty(tmp_path) -> None:
    """A missing gold file yields ``{}`` (never raises)."""
    assert_that(load_gold(tmp_path / "does_not_exist.json")).is_equal_to({})


def test_load_gold_bad_json_returns_empty(tmp_path) -> None:
    """Unparseable JSON yields ``{}``."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert_that(load_gold(bad)).is_equal_to({})


def test_load_gold_reads_valid_mapping(tmp_path) -> None:
    """A well-formed gold JSON round-trips into a dict."""
    gold = {"a.wav": {"moods": ["happy"], "energy": 0.8, "valence": 0.9}}
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(gold), encoding="utf-8")
    assert_that(load_gold(p)).is_equal_to(gold)


# --------------------------------------------------------------------------- #
# evaluate_against_gold
# --------------------------------------------------------------------------- #


def test_evaluate_against_gold_no_overlap_returns_empty() -> None:
    """No filename overlap between df and gold -> {}."""
    df = pd.DataFrame({"filename": ["a.wav"], "top_mood": ["happy"]})
    assert_that(evaluate_against_gold(df, {"z.wav": {"moods": ["calm"]}})).is_equal_to({})
    # Also empty when the df has no filename column.
    assert_that(evaluate_against_gold(pd.DataFrame({"x": [1]}), {"a.wav": {}})).is_equal_to({})


def test_evaluate_against_gold_accuracy_and_correlation() -> None:
    """Top-mood accuracy and energy/valence correlations on the overlap."""
    df = pd.DataFrame(
        {
            "filename": ["a.wav", "b.wav", "c.wav"],
            "top_mood": ["happy", "dark", "calm"],
            "energy": [0.9, 0.2, 0.5],
            "valence": [0.8, 0.1, 0.5],
        }
    )
    gold = {
        "a.wav": {"moods": ["happy"], "energy": 0.85, "valence": 0.9},
        "b.wav": {"moods": ["dark", "tense"], "energy": 0.15, "valence": 0.05},
        "c.wav": {"moods": ["energetic"], "energy": 0.55, "valence": 0.5},
    }
    out = evaluate_against_gold(df, gold)
    assert_that(out["n_overlap"]).is_equal_to(3)
    # a + b correct, c wrong (calm not in [energetic]) -> 2/3.
    assert_that(out["top_mood_accuracy"]).is_close_to(2 / 3, tolerance=1e-9)
    # Predictions track gold monotonically -> strong positive correlation.
    assert_that(out["energy_pearson"]).is_greater_than(0.9)
    assert_that(out["valence_spearman"]).is_close_to(1.0, tolerance=1e-6)
    # CCC is reported for both axes and is high when predictions sit near the y=x line.
    assert_that(out["energy_ccc"]).is_greater_than(0.9)
    assert_that(out["valence_ccc"]).is_greater_than(0.9)


# --------------------------------------------------------------------------- #
# concordance_correlation_coefficient
# --------------------------------------------------------------------------- #


def test_ccc_identical_series_is_one() -> None:
    """Perfect agreement on the y=x line -> CCC == 1.0, support == n."""
    x = np.linspace(0.0, 1.0, 25)

    ccc, support = concordance_correlation_coefficient(x, x.copy())

    assert_that(ccc).is_close_to(1.0, tolerance=1e-9)
    assert_that(support).is_equal_to(25)


def test_ccc_penalizes_scale_mismatch_below_pearson() -> None:
    """A perfectly correlated but rescaled prediction (r=1) scores CCC < 1 — the property
    that distinguishes CCC from Pearson and makes it the right axis-regression metric."""
    gold = np.linspace(0.0, 1.0, 30)
    pred = 0.5 * gold + 0.25  # r == 1 but wrong scale/offset

    ccc, _ = concordance_correlation_coefficient(pred, gold)

    assert_that(ccc).is_greater_than(0.0)
    assert_that(ccc).is_less_than(0.95)


def test_ccc_constant_series_is_nan() -> None:
    """Both series constant -> denominator 0 -> (nan, n), never a divide error."""
    ccc, support = concordance_correlation_coefficient(np.full(8, 0.4), np.full(8, 0.4))

    assert_that(ccc).is_nan()
    assert_that(support).is_equal_to(8)


def test_ccc_too_few_points_nan() -> None:
    """Fewer than 2 aligned points -> (nan, n) (undefined)."""
    ccc, support = concordance_correlation_coefficient(np.array([0.5]), np.array([0.5]))

    assert_that(ccc).is_nan()
    assert_that(support).is_equal_to(1)


def test_ccc_drops_non_finite_pairs() -> None:
    """A NaN in either series drops that pair rather than poisoning the whole score."""
    clean = concordance_correlation_coefficient(
        np.array([0.1, 0.4, 0.9]), np.array([0.1, 0.4, 0.9])
    )[0]
    withnan = concordance_correlation_coefficient(
        np.array([0.1, 0.4, 0.9, np.nan]), np.array([0.1, 0.4, 0.9, 0.5])
    )[0]

    assert_that(withnan).is_close_to(clean, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# ndcg_at_k
# --------------------------------------------------------------------------- #


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    """All relevant items ranked first -> nDCG == 1.0 (DCG == IDCG)."""
    assert_that(ndcg_at_k([0, 1, 2, 3, 4], {0, 1, 2}, k=5)).is_close_to(1.0, tolerance=1e-9)
    # Enough relevant to fill the whole window is also perfect.
    assert_that(ndcg_at_k([2, 0, 1], {0, 1, 2}, k=3)).is_close_to(1.0, tolerance=1e-9)


def test_ndcg_at_k_worse_ordering_below_one_and_hand_value() -> None:
    """A single relevant item at rank 2 -> DCG = 1/log2(3), IDCG = 1 -> nDCG = 1/log2(3)."""
    val = ndcg_at_k([9, 1, 8], {1}, k=3)
    assert_that(val).is_close_to(float(1.0 / np.log2(3)), tolerance=1e-9)
    assert_that(val).is_less_than(1.0)


def test_ndcg_at_k_guards() -> None:
    """``k <= 0`` or no relevant items -> 0.0 (never raises)."""
    assert_that(ndcg_at_k([0, 1, 2], {0}, k=0)).is_equal_to(0.0)
    assert_that(ndcg_at_k([0, 1, 2], set(), k=3)).is_equal_to(0.0)


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #


def test_recall_at_k_fraction_and_monotone() -> None:
    """Recall = hits∩top-k / |relevant|, non-decreasing in k."""
    ranked = [3, 1, 7, 2, 9]
    relevant = {1, 2, 5}  # 5 is never retrieved
    r1 = recall_at_k(ranked, relevant, k=1)  # top-1 = {3}: 0 hits
    r2 = recall_at_k(ranked, relevant, k=2)  # +1 -> 1/3
    r4 = recall_at_k(ranked, relevant, k=4)  # +2 -> 2/3
    assert_that(r1).is_close_to(0.0, tolerance=1e-12)
    assert_that(r2).is_close_to(1 / 3, tolerance=1e-9)
    assert_that(r4).is_close_to(2 / 3, tolerance=1e-9)
    assert_that(r1).is_less_than_or_equal_to(r2)  # monotone in k
    assert_that(r2).is_less_than_or_equal_to(r4)


def test_recall_at_k_guards() -> None:
    """Empty gold or ``k <= 0`` -> 0.0."""
    assert_that(recall_at_k([0, 1], set(), k=3)).is_equal_to(0.0)
    assert_that(recall_at_k([0, 1], {0}, k=0)).is_equal_to(0.0)


# --------------------------------------------------------------------------- #
# macro_f1
# --------------------------------------------------------------------------- #


def test_macro_f1_hand_value_and_support() -> None:
    """Hand-computed macro-F1: class0 F1=2/3, class1 F1=4/5 -> macro=(2/3+4/5)/2."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    f1, support = macro_f1(y_true, y_pred)
    assert_that(f1).is_close_to(((2 / 3) + (4 / 5)) / 2, tolerance=1e-9)
    assert_that(support).is_equal_to(4)


def test_macro_f1_empty_and_mismatch() -> None:
    """Empty or length-mismatched input -> (0.0, 0)."""
    assert_that(macro_f1([], [])).is_equal_to((0.0, 0))
    assert_that(macro_f1([0, 1], [0])).is_equal_to((0.0, 0))


# --------------------------------------------------------------------------- #
# nmi
# --------------------------------------------------------------------------- #


def test_nmi_identical_partition_is_one() -> None:
    """Identical clusterings (up to relabeling) -> NMI == 1.0, support == n."""
    val, support = nmi([0, 0, 1, 1], [1, 1, 0, 0])
    assert_that(val).is_close_to(1.0, tolerance=1e-9)
    assert_that(support).is_equal_to(4)


def test_nmi_empty_and_mismatch() -> None:
    assert_that(nmi([], [])).is_equal_to((0.0, 0))
    assert_that(nmi([0, 1, 1], [0, 1])).is_equal_to((0.0, 0))


# --------------------------------------------------------------------------- #
# expected_calibration_error
# --------------------------------------------------------------------------- #


def test_ece_perfectly_calibrated_is_zero() -> None:
    """conf == accuracy in every occupied bin -> ECE ≈ 0."""
    conf = np.array([1.0] * 5 + [0.0] * 5)
    correct = np.array([1] * 5 + [0] * 5)  # conf 1.0 all right, conf 0.0 all wrong
    ece, support = expected_calibration_error(conf, correct, n_bins=10)
    assert_that(ece).is_close_to(0.0, tolerance=1e-9)
    assert_that(support).is_equal_to(10)


def test_ece_miscalibrated_is_large() -> None:
    """Confident (0.9) but always wrong -> ECE ≈ 0.9."""
    conf = np.full(10, 0.9)
    correct = np.zeros(10)
    ece, support = expected_calibration_error(conf, correct, n_bins=10)
    assert_that(ece).is_close_to(0.9, tolerance=1e-9)
    assert_that(support).is_equal_to(10)


def test_ece_empty() -> None:
    assert_that(expected_calibration_error(np.array([]), np.array([]))).is_equal_to((0.0, 0))


# --------------------------------------------------------------------------- #
# procrustes_disparity
# --------------------------------------------------------------------------- #


def test_procrustes_identical_clouds_zero() -> None:
    """Identical layouts -> disparity 0 (invariant to translation/scale/rotation)."""
    A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    disp, n = procrustes_disparity(A, A.copy())
    assert_that(disp).is_close_to(0.0, tolerance=1e-10)
    assert_that(n).is_equal_to(4)


def test_procrustes_different_shape_positive() -> None:
    """A square vs a rectangle differ in aspect ratio (not a similarity transform) -> disparity > 0."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rect = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
    disp, n = procrustes_disparity(square, rect)
    assert_that(disp).is_greater_than(1e-6)
    assert_that(n).is_equal_to(4)


def test_procrustes_too_few_points_nan() -> None:
    """Fewer than 2 shared rows -> (nan, n) (disparity undefined)."""
    disp, n = procrustes_disparity(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
    assert_that(disp).is_nan()
    assert_that(n).is_equal_to(1)


def test_procrustes_numpy_fallback_matches_scipy(monkeypatch) -> None:
    """The numpy fallback (no scipy) must return the SAME disparity M² as scipy — it applies the
    optimal uniform scale, so both compute 1 - s². Forcing the scipy import to fail exercises it."""
    import sys

    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 2))
    B = rng.standard_normal((6, 2))
    scipy_val, _ = procrustes_disparity(A, B)  # scipy path (scipy is installed)
    monkeypatch.setitem(
        sys.modules, "scipy.spatial", None
    )  # make `from scipy.spatial import ...` fail
    fallback_val, _ = procrustes_disparity(A, B)  # numpy fallback path
    assert_that(fallback_val).is_close_to(scipy_val, tolerance=1e-9)


def test_evaluation_module_is_torch_free() -> None:
    """Importing the metrics module (and calling the new fns) must not pull torch in — CI guard.

    Uses subprocess import-isolation, not a session-global ``sys.modules`` check (other tests may
    have imported torch already)."""
    import subprocess
    import sys

    code = (
        "import moodengine.evaluation as e; "
        "e.macro_f1([0,1],[0,1]); e.nmi([0,1],[0,1]); "
        "import numpy as np; e.ndcg_at_k([0,1],{0},2); e.recall_at_k([0,1],{0},2); "
        "e.expected_calibration_error(np.array([0.9]), np.array([1])); "
        "e.concordance_correlation_coefficient(np.array([0.1,0.9]), np.array([0.1,0.9])); "
        "e.procrustes_disparity(np.eye(3), np.eye(3)); "
        "import sys; assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as(r.stderr).is_equal_to(0)
