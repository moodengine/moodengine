"""Tests for :mod:`moodengine.labeling` — calibrated zero-shot moods, attribute axes,
cluster mood profiles + cluster naming (torch-free).

A tiny fake CLAP embedder stands in for the real model: ``embed_text`` maps
prompt strings to deterministic vectors so every labeling stage can be exercised
without torch and with fully reproducible numbers.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

from moodengine.labeling import (
    DEFAULT_MOOD_PROMPTS,
    DEFAULT_TEMPERATURE,
    ENERGY_PROMPTS,
    VALENCE_PROMPTS,
    MoodScores,
    attribute_scores,
    build_label_matrix,
    cluster_mood_profiles,
    compose_mood_vector,
    label_tracks,
    labeling_quality_metrics,
    l2_normalize,
    name_clusters,
    recenter_similarities,
    score_axis,
    score_moods,
    softmax,
    zero_shot_moods,
)


def _hash_vec(text: str, dim: int) -> np.ndarray:
    """Deterministic, prompt-dependent unit vector in ``dim`` dims."""
    seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


class _FakeCLAP:
    """Minimal CLAP stand-in: ``embed_text`` -> deterministic L2-normed rows.

    Each prompt string hashes to a fixed (un-normalized) vector; an optional
    ``overrides`` dict pins specific prompts to chosen vectors so the ensembling
    maths can be checked exactly. ``embed_text`` returns the rows L2-normalized
    (matching real CLAP, whose text embeddings are unit vectors).
    """

    def __init__(self, dim: int = 6, overrides: dict[str, np.ndarray] | None = None) -> None:
        self.dim = dim
        self.overrides = {k: np.asarray(v, dtype=np.float32) for k, v in (overrides or {}).items()}
        self.calls: list[list[str]] = []

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        self.calls.append(list(prompts))
        rows = [
            self.overrides[p] if p in self.overrides else _hash_vec(p, self.dim) for p in prompts
        ]
        mat = np.vstack(rows).astype(np.float32)
        return l2_normalize(mat, axis=1)


# --------------------------------------------------------------------------- #
# Prompt tables
# --------------------------------------------------------------------------- #
def test_default_mood_prompts_are_ensembled_lists() -> None:
    """The mood table maps each name to a non-empty list of prompt strings."""
    assert_that(DEFAULT_MOOD_PROMPTS).is_instance_of(dict)
    assert_that(len(DEFAULT_MOOD_PROMPTS)).is_greater_than_or_equal_to(8)
    assert_that(DEFAULT_MOOD_PROMPTS).contains_key("energetic")
    assert_that(DEFAULT_MOOD_PROMPTS).contains_key("melancholic")
    for name, prompts in DEFAULT_MOOD_PROMPTS.items():
        assert_that(name).is_instance_of(str)
        assert_that(name).is_not_empty()
        assert_that(prompts).is_instance_of(list)
        assert_that(len(prompts)).is_greater_than_or_equal_to(1)
        assert_that(all(isinstance(p, str) and len(p) > 3 for p in prompts)).is_true()


def test_axis_prompt_tables_have_two_poles() -> None:
    """Energy/valence axes each carry exactly two ensembled poles."""
    for table in (ENERGY_PROMPTS, VALENCE_PROMPTS):
        assert_that(table).is_length(2)
        for prompts in table.values():
            assert_that(prompts).is_instance_of(list)
            assert_that(len(prompts)).is_greater_than_or_equal_to(1)


# --------------------------------------------------------------------------- #
# softmax / l2_normalize
# --------------------------------------------------------------------------- #
def test_softmax_rows_sum_to_one() -> None:
    """Softmax over the last axis yields a valid probability distribution."""
    scores = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.0]], dtype=np.float32)
    probs = softmax(scores, temperature=DEFAULT_TEMPERATURE, axis=1)
    assert_that(probs.shape).is_equal_to(scores.shape)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert_that(bool(np.all(probs >= 0.0))).is_true()


def test_softmax_lower_temperature_sharpens() -> None:
    """A smaller temperature concentrates mass on the top score."""
    scores = np.array([0.30, 0.20, 0.10], dtype=np.float32)
    sharp = softmax(scores, temperature=0.02)
    soft = softmax(scores, temperature=1.0)
    # Both peak at index 0, but the cold softmax is more peaked there.
    assert_that(int(sharp.argmax())).is_equal_to(0)
    assert_that(int(soft.argmax())).is_equal_to(0)
    assert_that(float(sharp[0])).is_greater_than(float(soft[0]))


def test_l2_normalize_unit_rows_and_zero_safe() -> None:
    """Rows become unit-norm; an all-zero row does not divide by zero."""
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(x, axis=1)
    np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-6)
    assert_that(bool(np.all(np.isfinite(out)))).is_true()


# --------------------------------------------------------------------------- #
# build_label_matrix
# --------------------------------------------------------------------------- #
def test_build_label_matrix_shapes_and_l2_norm() -> None:
    """One L2-normalized row per label, with the right ``(n_labels, dim)`` shape."""
    embedder = _FakeCLAP(dim=6)
    prompts = {"a": ["p0", "p1"], "b": ["p2"], "c": ["p3", "p4", "p5"]}
    names, matrix = build_label_matrix(embedder, prompts)
    assert_that(names).is_equal_to(["a", "b", "c"])
    assert_that(matrix.shape).is_equal_to((3, 6))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    # Encoded once, flattened in dict order.
    assert_that(embedder.calls).is_equal_to([["p0", "p1", "p2", "p3", "p4", "p5"]])


def test_build_label_matrix_ensembles_mean_of_prompts() -> None:
    """A label's row is the L2-normed mean of its prompts' (unit) text vectors."""
    v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embedder = _FakeCLAP(dim=3, overrides={"q0": v0, "q1": v1})
    names, matrix = build_label_matrix(embedder, {"mix": ["q0", "q1"]})
    assert_that(names).is_equal_to(["mix"])
    # mean([1,0,0],[0,1,0]) = [.5,.5,0] -> normalized -> [1,1,0]/sqrt(2).
    expected = l2_normalize((v0 + v1) / 2.0, axis=0)
    np.testing.assert_allclose(matrix[0], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# zero_shot_moods (kept)
# --------------------------------------------------------------------------- #
def test_zero_shot_moods_ranks_aligned_vector_first() -> None:
    """The mood whose text vector aligns with the audio gets the top score."""
    text = np.eye(3, dtype=np.float32)  # orthonormal: m0, m1, m2
    moods = ["m0", "m1", "m2"]
    audio = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # aligned with m1
    ranked = zero_shot_moods(audio, text, moods, top_k=3)
    assert_that(ranked[0][0]).is_equal_to("m1")
    assert_that(float(ranked[0][1])).is_close_to(1.0, 1e-6)
    assert_that(float(ranked[1][1])).is_close_to(0.0, 1e-6)


def test_zero_shot_moods_respects_top_k() -> None:
    """``top_k`` truncates the ranking and sorts strictly descending."""
    rng = np.random.default_rng(0)
    text = rng.standard_normal((5, 4)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    audio = text[2]
    moods = [f"m{i}" for i in range(5)]
    ranked = zero_shot_moods(audio, text, moods, top_k=2)
    assert_that(ranked).is_length(2)
    assert_that(ranked[0][0]).is_equal_to("m2")
    scores = [s for _, s in ranked]
    assert_that(scores).is_equal_to(sorted(scores, reverse=True))


def test_zero_shot_moods_top_k_zero_empty() -> None:
    """``top_k=0`` returns no moods (guarded, no crash)."""
    text = np.eye(2, dtype=np.float32)
    assert_that(zero_shot_moods(np.array([1.0, 0.0]), text, ["a", "b"], top_k=0)).is_equal_to([])


# --------------------------------------------------------------------------- #
# score_moods — the pure sims→recenter→softmax triptych over a precomputed matrix
# --------------------------------------------------------------------------- #
def test_score_moods_stages_shapes_and_calibration() -> None:
    """All three stages come back ``(n, n_moods)`` float32; probs rows sum to 1."""
    rng = np.random.default_rng(20)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    names, matrix = build_label_matrix(_FakeCLAP(dim=8), {"a": ["p0"], "b": ["p1"], "c": ["p2"]})

    scores = score_moods(audio, names, matrix)

    assert_that(scores).is_instance_of(MoodScores)
    assert_that(scores.mood_names).is_equal_to(names)
    for stage in (scores.sims, scores.recentered, scores.probs):
        assert_that(stage.shape).is_equal_to((6, 3))
        assert_that(stage.dtype).is_equal_to(np.dtype("float32"))
    np.testing.assert_allclose(scores.probs.sum(axis=1), 1.0, atol=1e-5)
    # n >= 5 and recenter on by default → each mood column of `recentered` is centered.
    np.testing.assert_allclose(scores.recentered.mean(axis=0), 0.0, atol=1e-5)


def test_score_moods_recenter_off_keeps_raw_sims() -> None:
    """With recentering off the second stage is the raw cosine block, untouched."""
    rng = np.random.default_rng(21)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    names, matrix = build_label_matrix(_FakeCLAP(dim=8), {"a": ["p0"], "b": ["p1"]})

    scores = score_moods(audio, names, matrix, recenter=False)

    np.testing.assert_array_equal(scores.recentered, scores.sims)
    np.testing.assert_allclose(scores.sims, audio @ matrix.T, atol=1e-6)


def test_score_moods_single_track_promoted_to_one_row() -> None:
    """A 1-D audio embedding scores as a single-row batch."""
    names, matrix = build_label_matrix(_FakeCLAP(dim=4), {"a": ["p0"], "b": ["p1"]})

    scores = score_moods(np.ones(4, dtype=np.float32), names, matrix)

    assert_that(scores.probs.shape).is_equal_to((1, 2))


def test_score_moods_rejects_non_finite_embeddings() -> None:
    """A NaN row would silently poison the per-mood recentering means for every
    track — the boundary rejects it, naming the offending rows."""
    names, matrix = build_label_matrix(_FakeCLAP(dim=4), {"a": ["p0"], "b": ["p1"]})
    audio = np.zeros((6, 4), dtype=np.float32)
    audio[3, 1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        score_moods(audio, names, matrix)


# --------------------------------------------------------------------------- #
# precomputed label_matrix — score without a live embedder
# --------------------------------------------------------------------------- #
def test_label_tracks_precomputed_matrix_equals_embedder_path() -> None:
    """A precomputed ``(names, matrix)`` yields byte-identical labels — and the
    embedder is genuinely not needed (``None`` would crash on any consultation)."""
    rng = np.random.default_rng(22)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    lm = build_label_matrix(embedder, DEFAULT_MOOD_PROMPTS)

    via_embedder = label_tracks(audio, embedder)
    via_matrix = label_tracks(audio, label_matrix=lm)

    pd.testing.assert_frame_equal(via_matrix, via_embedder)


def test_label_tracks_with_matrix_never_consults_the_embedder() -> None:
    """When a matrix is supplied the embedder must not be called at all (the whole
    point: no second text-encoder forward)."""
    rng = np.random.default_rng(23)
    audio = l2_normalize(rng.standard_normal((5, 8)).astype(np.float32), axis=1)
    lm = build_label_matrix(_FakeCLAP(dim=8), DEFAULT_MOOD_PROMPTS)
    watched = _FakeCLAP(dim=8)

    label_tracks(audio, watched, label_matrix=lm)

    assert_that(watched.calls).is_equal_to([])


def test_cluster_mood_profiles_precomputed_matrix_equals_embedder_path() -> None:
    """Same equivalence for cluster profiles, embedder-free."""
    rng = np.random.default_rng(24)
    audio = l2_normalize(rng.standard_normal((9, 8)).astype(np.float32), axis=1)
    labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
    embedder = _FakeCLAP(dim=8)
    lm = build_label_matrix(embedder, DEFAULT_MOOD_PROMPTS)

    via_embedder = cluster_mood_profiles(audio, labels, embedder)
    via_matrix = cluster_mood_profiles(audio, labels, label_matrix=lm)

    assert_that(via_matrix).is_equal_to(via_embedder)


def test_labeling_without_embedder_or_matrix_raises() -> None:
    """Neither an embedder nor a matrix → a clear ValueError, not an AttributeError."""
    audio = np.ones((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="label_matrix"):
        label_tracks(audio)
    with pytest.raises(ValueError, match="label_matrix"):
        cluster_mood_profiles(audio, np.zeros(2, dtype=int))


# --------------------------------------------------------------------------- #
# label_tracks (new API: probs + scores column)
# --------------------------------------------------------------------------- #
def test_label_tracks_columns_and_calibrated_probs() -> None:
    """``label_tracks`` returns the documented columns; probs are calibrated."""
    # Three moods as an orthonormal basis; three audio vectors each aligned to one.
    v = np.eye(3, dtype=np.float32)
    embedder = _FakeCLAP(dim=3, overrides={"p0": v[0], "p1": v[1], "p2": v[2]})
    prompts = {"alpha": ["p0"], "beta": ["p1"], "gamma": ["p2"]}
    audio = np.eye(3, dtype=np.float32)
    df = label_tracks(audio, embedder, prompts=prompts, top_k=2)

    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(list(df.columns)).is_equal_to(
        ["top_mood", "top_score", "mood_topk", "mood_topk_scores"]
    )
    assert_that(df).is_length(3)
    assert_that(list(df.index)).is_equal_to([0, 1, 2])
    assert_that(df["top_mood"].tolist()).is_equal_to(["alpha", "beta", "gamma"])
    # top_score is a softmax probability in (0, 1].
    assert_that(all(0.0 < s <= 1.0 for s in df["top_score"])).is_true()
    # topk lists carry k entries, scores aligned and sorted descending.
    for topk, scores in zip(df["mood_topk"], df["mood_topk_scores"]):
        assert_that(topk).is_length(2)
        assert_that(scores).is_length(2)
        assert_that(scores).is_equal_to(sorted(scores, reverse=True))
        assert_that(all(0.0 <= sc <= 1.0 for sc in scores)).is_true()


def test_label_tracks_probs_sum_to_one_over_all_moods() -> None:
    """Per-track probabilities over the FULL mood vocabulary sum to ~1."""
    rng = np.random.default_rng(7)
    audio = l2_normalize(rng.standard_normal((4, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    n_moods = len(DEFAULT_MOOD_PROMPTS)
    # Ask for every mood so the topk scores cover the whole distribution.
    df = label_tracks(audio, embedder, top_k=n_moods)
    for scores in df["mood_topk_scores"]:
        assert_that(sum(scores)).is_close_to(1.0, 1e-4)


def test_label_tracks_single_track_1d_input() -> None:
    """A 1-D audio embedding is treated as a single track."""
    v = np.eye(2, dtype=np.float32)
    embedder = _FakeCLAP(dim=2, overrides={"pa": v[0], "pb": v[1]})
    prompts = {"a": ["pa"], "b": ["pb"]}
    df = label_tracks(np.array([1.0, 0.0], dtype=np.float32), embedder, prompts=prompts)
    assert_that(df).is_length(1)
    assert_that(df.loc[0, "top_mood"]).is_equal_to("a")


# --------------------------------------------------------------------------- #
# score_axis / attribute_scores
# --------------------------------------------------------------------------- #
def test_score_axis_in_range_and_monotone() -> None:
    """A track near the positive pole scores > 0.5; near the negative pole < 0.5."""
    neg = np.array([1.0, 0.0], dtype=np.float32)
    pos = np.array([0.0, 1.0], dtype=np.float32)
    embedder = _FakeCLAP(dim=2, overrides={"lo": neg, "hi": pos})
    axis_prompts = {"low": ["lo"], "high": ["hi"]}
    # audio[0] aligns with the positive pole, audio[1] with the negative pole.
    audio = np.array([pos, neg], dtype=np.float32)
    scores = score_axis(audio, embedder, axis_prompts)
    assert_that(scores.shape).is_equal_to((2,))
    assert_that(bool(np.all((scores >= 0.0) & (scores <= 1.0)))).is_true()
    assert_that(float(scores[0])).is_greater_than(0.5)
    assert_that(float(scores[1])).is_less_than(0.5)


def test_score_axis_rejects_non_two_pole() -> None:
    """An axis without exactly two poles is rejected."""
    embedder = _FakeCLAP(dim=4)
    with pytest.raises(ValueError, match="exactly 2 poles"):
        score_axis(np.zeros((1, 4), dtype=np.float32), embedder, {"only": ["p"]})


def test_attribute_scores_columns_and_range() -> None:
    """``attribute_scores`` yields energy & valence columns in [0, 1]."""
    rng = np.random.default_rng(3)
    audio = l2_normalize(rng.standard_normal((5, 8)).astype(np.float32), axis=1)
    df = attribute_scores(audio, _FakeCLAP(dim=8))
    assert_that(list(df.columns)).is_equal_to(["energy", "valence"])
    assert_that(df).is_length(5)
    for col in ("energy", "valence"):
        vals = df[col].to_numpy()
        assert_that(bool(np.all((vals >= 0.0) & (vals <= 1.0)))).is_true()


# --------------------------------------------------------------------------- #
# cluster_mood_profiles
# --------------------------------------------------------------------------- #
def test_cluster_mood_profiles_keys_and_sorted() -> None:
    """Profiles cover exactly the unique cluster ids; scores sort descending."""
    rng = np.random.default_rng(11)
    audio = l2_normalize(rng.standard_normal((9, 8)).astype(np.float32), axis=1)
    labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
    embedder = _FakeCLAP(dim=8)
    profiles = cluster_mood_profiles(audio, labels, embedder, top_k=3)

    assert_that(set(profiles.keys())).is_equal_to({-1, 0, 1})
    for cid, profs in profiles.items():
        assert_that(profs).is_length(3)
        scores = [s for _, s in profs]
        assert_that(scores).is_equal_to(sorted(scores, reverse=True))
        assert_that(all(isinstance(m, str) for m, _ in profs)).is_true()


# --------------------------------------------------------------------------- #
# name_clusters (kept)
# --------------------------------------------------------------------------- #
def test_name_clusters_majority_vote() -> None:
    """Each cluster is named by its most common per-track top mood."""
    labels = np.array([0, 0, 0, 1, 1])
    top_moods = ["happy", "happy", "calm", "dark", "dark"]
    named = name_clusters(labels, top_moods)
    assert_that(named).is_equal_to({0: "happy", 1: "dark"})


def test_name_clusters_includes_noise() -> None:
    """The noise cluster (-1) is named just like any other cluster."""
    labels = np.array([-1, -1, 0, 0])
    top_moods = ["calm", "calm", "epic", "epic"]
    named = name_clusters(labels, top_moods)
    assert_that(named[-1]).is_equal_to("calm")
    assert_that(named[0]).is_equal_to("epic")


def test_name_clusters_tie_breaks_to_first_seen() -> None:
    """On a tie the mood that appears first (insertion order) wins, deterministically."""
    labels = np.array([0, 0])
    top_moods = ["groovy", "tense"]  # 1-1 tie -> first-seen 'groovy'
    named = name_clusters(labels, top_moods)
    assert_that(named[0]).is_equal_to("groovy")


# --------------------------------------------------------------------------- #
# recenter_similarities
# --------------------------------------------------------------------------- #


def test_recenter_similarities_centers_columns() -> None:
    """With enough rows, each label column is shifted to zero mean."""
    rng = np.random.default_rng(0)
    sims = rng.standard_normal((8, 3)).astype(np.float32)
    out = recenter_similarities(sims, enable=True)
    assert_that(out.shape).is_equal_to(sims.shape)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    # It is a pure per-column shift: (sims - out) is constant down each column.
    shift = sims - out
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-5)
    np.testing.assert_allclose(shift[0], sims.mean(axis=0), atol=1e-5)


def test_recenter_similarities_identity_for_small_n() -> None:
    """Below ``min_n`` rows the input is returned unchanged (default n<5)."""
    sims = np.array([[0.2, 0.3], [0.4, 0.1], [0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    out = recenter_similarities(sims, enable=True)  # n == 4 < 5
    np.testing.assert_array_equal(out, sims)


def test_recenter_similarities_disabled_is_identity() -> None:
    """``enable=False`` never centers, even with many rows."""
    rng = np.random.default_rng(1)
    sims = rng.standard_normal((10, 4)).astype(np.float32)
    out = recenter_similarities(sims, enable=False)
    np.testing.assert_array_equal(out, sims)


def test_recenter_similarities_does_not_mutate_input() -> None:
    """The original similarity array is left untouched."""
    rng = np.random.default_rng(2)
    sims = rng.standard_normal((6, 3)).astype(np.float32)
    before = sims.copy()
    _ = recenter_similarities(sims, enable=True)
    np.testing.assert_array_equal(sims, before)


# --------------------------------------------------------------------------- #
# recenter flag plumbing through the scoring stages
# --------------------------------------------------------------------------- #


def test_label_tracks_recenter_flag_changes_output() -> None:
    """The ``recenter`` flag is plumbed into ``label_tracks`` (n>=5 -> effect)."""
    rng = np.random.default_rng(5)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    centered = label_tracks(audio, embedder, recenter=True)
    raw = label_tracks(audio, embedder, recenter=False)
    # Centering the per-mood similarities before the softmax shifts the calibrated
    # top scores, so the two runs disagree (n>=5 so recentering is active).
    assert_that(centered["top_score"].tolist()).is_not_equal_to(raw["top_score"].tolist())
    # Scores stay valid softmax probabilities in (0, 1].
    assert_that(all(0.0 < s <= 1.0 for s in centered["top_score"])).is_true()


def test_label_tracks_recenter_noop_below_min_n() -> None:
    """For n<5 the recenter flag is a no-op (centering would need >=5 rows)."""
    rng = np.random.default_rng(6)
    audio = l2_normalize(rng.standard_normal((4, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    on = label_tracks(audio, embedder, recenter=True)
    off = label_tracks(audio, embedder, recenter=False)
    assert_that(on["top_score"].tolist()).is_equal_to(off["top_score"].tolist())
    assert_that(on["top_mood"].tolist()).is_equal_to(off["top_mood"].tolist())


def test_score_axis_recenter_flag_plumbed() -> None:
    """``score_axis`` accepts and honours the recenter flag (2-col centering ok)."""
    rng = np.random.default_rng(7)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    on = score_axis(audio, embedder, ENERGY_PROMPTS, recenter=True)
    off = score_axis(audio, embedder, ENERGY_PROMPTS, recenter=False)
    assert_that(on.shape).is_equal_to((6,))
    assert_that(bool(np.all((on >= 0.0) & (on <= 1.0)))).is_true()
    assert_that(bool(np.allclose(on, off))).is_false()


def test_attribute_scores_recenter_flag_plumbed() -> None:
    """``attribute_scores`` forwards recenter to both axes without error."""
    rng = np.random.default_rng(8)
    audio = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32), axis=1)
    embedder = _FakeCLAP(dim=8)
    df = attribute_scores(audio, embedder, recenter=True)
    assert_that(list(df.columns)).is_equal_to(["energy", "valence"])
    assert_that(float(df.to_numpy().min())).is_greater_than_or_equal_to(0.0)
    assert_that(float(df.to_numpy().max())).is_less_than_or_equal_to(1.0)


def test_cluster_mood_profiles_recenter_flag_plumbed() -> None:
    """``cluster_mood_profiles`` accepts recenter and stays well-formed."""
    rng = np.random.default_rng(9)
    audio = l2_normalize(rng.standard_normal((9, 8)).astype(np.float32), axis=1)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    embedder = _FakeCLAP(dim=8)
    profiles = cluster_mood_profiles(audio, labels, embedder, top_k=3, recenter=True)
    assert_that(set(profiles.keys())).is_equal_to({0, 1, 2})
    for profs in profiles.values():
        scores = [s for _, s in profs]
        assert_that(scores).is_equal_to(sorted(scores, reverse=True))


# --------------------------------------------------------------------------- #
# labeling_quality_metrics
# --------------------------------------------------------------------------- #


def test_labeling_quality_metrics_keys_and_values() -> None:
    """Health metrics report diversity, dominance and confidence margins."""
    df = pd.DataFrame(
        {
            "top_mood": ["happy", "happy", "dark", "calm"],
            "top_score": [0.6, 0.5, 0.7, 0.4],
            "mood_topk_scores": [
                [0.6, 0.2, 0.1],
                [0.5, 0.4, 0.1],
                [0.7, 0.1, 0.1],
                [0.4, 0.3, 0.3],
            ],
        }
    )
    m = labeling_quality_metrics(df)
    assert_that(set(m.keys())).is_equal_to(
        {
            "n_distinct_top_moods",
            "top_mood_histogram",
            "max_mood_share",
            "mean_top1_minus_top2",
            "mean_top_score",
        }
    )
    assert_that(m["n_distinct_top_moods"]).is_equal_to(3)
    assert_that(m["top_mood_histogram"]).is_equal_to({"happy": 2, "dark": 1, "calm": 1})
    assert_that(m["max_mood_share"]).is_close_to(2 / 4, 1e-6)
    # margins: 0.4, 0.1, 0.6, 0.1 -> mean 0.3
    assert_that(m["mean_top1_minus_top2"]).is_close_to(0.3, 1e-6)
    assert_that(m["mean_top_score"]).is_close_to((0.6 + 0.5 + 0.7 + 0.4) / 4, 1e-6)


def test_labeling_quality_metrics_margin_zero_when_single_score() -> None:
    """A topk list with fewer than 2 scores contributes a 0.0 margin."""
    df = pd.DataFrame(
        {
            "top_mood": ["happy", "dark"],
            "top_score": [0.6, 0.7],
            "mood_topk_scores": [[0.6], [0.7]],
        }
    )
    m = labeling_quality_metrics(df)
    assert_that(m["mean_top1_minus_top2"]).is_equal_to(0.0)


def test_labeling_quality_metrics_empty_df_robust() -> None:
    """An empty frame yields zeroed metrics, not an error."""
    m = labeling_quality_metrics(pd.DataFrame())
    assert_that(m["n_distinct_top_moods"]).is_equal_to(0)
    assert_that(m["top_mood_histogram"]).is_equal_to({})
    assert_that(m["max_mood_share"]).is_equal_to(0.0)
    assert_that(m["mean_top1_minus_top2"]).is_equal_to(0.0)
    assert_that(m["mean_top_score"]).is_equal_to(0.0)


def test_labeling_quality_metrics_from_label_tracks_output() -> None:
    """The metrics consume a real ``label_tracks`` DataFrame unchanged."""
    rng = np.random.default_rng(10)
    audio = l2_normalize(rng.standard_normal((7, 8)).astype(np.float32), axis=1)
    df = label_tracks(audio, _FakeCLAP(dim=8))
    m = labeling_quality_metrics(df)
    assert_that(m["n_distinct_top_moods"]).is_greater_than_or_equal_to(1)
    assert_that(m["max_mood_share"]).is_greater_than(0.0)
    assert_that(m["max_mood_share"]).is_less_than_or_equal_to(1.0)
    assert_that(m["mean_top_score"]).is_between(0.0, 1.0)


# --------------------------------------------------------------------------- #
# compose_mood_vector (mood-vector arithmetic)
# --------------------------------------------------------------------------- #
def test_compose_single_positive_term_equals_label_row() -> None:
    """A single positive term returns exactly that (unit) mood direction."""
    M = np.eye(3, dtype=np.float32)  # three orthonormal mood directions
    names = ["a", "b", "c"]
    vec = compose_mood_vector(M, names, [("b", 1.0)])
    np.testing.assert_allclose(vec, M[1], atol=1e-6)
    assert_that(float(np.linalg.norm(vec))).is_close_to(1.0, 1e-5)
    assert_that(vec.dtype).is_equal_to(np.dtype("float32"))


def test_compose_negative_weight_inverts_direction() -> None:
    """A negative weight flips the mood direction ("but not …")."""
    M = np.eye(3, dtype=np.float32)
    vec = compose_mood_vector(M, ["a", "b", "c"], [("a", -1.0)])
    np.testing.assert_allclose(vec, -M[0], atol=1e-6)


def test_compose_signed_combination_points_between_and_away() -> None:
    """calm(+1) minus melancholic(-1): the result leans toward calm, away from melancholic."""
    M = np.eye(3, dtype=np.float32)  # a=calm, b=melancholic
    vec = compose_mood_vector(M, ["a", "b", "c"], [("a", 1.0), ("b", -1.0)])
    expected = l2_normalize(M[0] - M[1], axis=-1)
    np.testing.assert_allclose(vec, expected, atol=1e-6)
    # toward calm, away from melancholic
    assert_that(float(vec @ M[0])).is_greater_than(0.0)
    assert_that(float(vec @ M[1])).is_less_than(0.0)


def test_compose_unknown_names_ignored() -> None:
    """Names outside the vocabulary are silently dropped (UI only offers real moods)."""
    M = np.eye(3, dtype=np.float32)
    vec = compose_mood_vector(M, ["a", "b", "c"], [("a", 1.0), ("nope", 5.0)])
    np.testing.assert_allclose(vec, M[0], atol=1e-6)  # 'nope' contributes nothing


def test_compose_empty_and_cancelling_yield_zeros() -> None:
    """Empty terms, unknown-only, or perfectly cancelling combinations -> a null vector."""
    M = np.eye(3, dtype=np.float32)
    for terms in ([], [("nope", 1.0)], [("a", 1.0), ("a", -1.0)]):
        vec = compose_mood_vector(M, ["a", "b", "c"], terms)
        assert_that(vec.shape).is_equal_to((3,))
        assert_that(bool(np.all(vec == 0.0))).is_true()


def test_compose_deterministic_and_no_mutation() -> None:
    """Deterministic; the label matrix is never mutated."""
    M = l2_normalize(np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32), axis=1)
    names = [f"m{i}" for i in range(5)]
    before = M.copy()
    v1 = compose_mood_vector(M, names, [("m0", 1.0), ("m3", -0.5)])
    v2 = compose_mood_vector(M, names, [("m0", 1.0), ("m3", -0.5)])
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(M, before)
