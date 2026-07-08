"""Tests for :mod:`moodengine.pooling` — track-level pooling of frame/clip embeddings."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.config import default_config
from moodengine.pooling import (
    POOLERS,
    l2_normalize,
    pool_clap,
    pool_frames,
    pool_mert,
    weight_layers,
)


def test_l2_normalize_unit_norm() -> None:
    """A non-zero vector is scaled to unit L2 norm and stays float32."""
    x = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
    out = l2_normalize(x)
    assert_that(out.dtype).is_equal_to(np.float32)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)
    np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)


def test_l2_normalize_zero_vector_safe() -> None:
    """A zero vector does not divide-by-zero; it stays finite (near zero)."""
    out = l2_normalize(np.zeros(8, dtype=np.float32))
    assert_that(bool(np.all(np.isfinite(out)))).is_true()
    assert_that(float(np.linalg.norm(out))).is_less_than(1e-3)


def test_l2_normalize_along_axis() -> None:
    """Row-wise normalization (axis=-1) gives each row unit norm."""
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(x, axis=-1)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), [1.0, 1.0], atol=1e-6)


def test_weight_layers_uniform_is_layer_mean() -> None:
    """``uniform`` mode averages over the layer axis -> (n_frames, hidden)."""
    rng = np.random.default_rng(0)
    frame_emb = rng.standard_normal((3, 5, 4)).astype(np.float32)
    out = weight_layers(frame_emb, "uniform")
    assert_that(out.shape).is_equal_to((5, 4))
    np.testing.assert_allclose(out, frame_emb.mean(axis=0), rtol=1e-5)


def test_weight_layers_last_is_last_layer() -> None:
    """``last`` mode returns the final layer's frames."""
    rng = np.random.default_rng(1)
    frame_emb = rng.standard_normal((3, 5, 4)).astype(np.float32)
    out = weight_layers(frame_emb, "last")
    assert_that(out.shape).is_equal_to((5, 4))
    np.testing.assert_allclose(out, frame_emb[-1], rtol=1e-5)


def test_weight_layers_bad_mode_raises() -> None:
    """An unknown weighting mode raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"unknown layer weighting mode"):
        weight_layers(np.zeros((2, 3, 4), dtype=np.float32), "nope")


def test_weight_layers_subset_default_middle_third() -> None:
    """``subset`` with no explicit layers averages the middle-third band."""
    rng = np.random.default_rng(10)
    n_layers = 12
    frame_emb = rng.standard_normal((n_layers, 5, 4)).astype(np.float32)
    out = weight_layers(frame_emb, "subset")
    assert_that(out.shape).is_equal_to((5, 4))
    # Middle third: range(12//3, 2*12//3 + 1) == range(4, 9) -> layers 4..8.
    expected = frame_emb[list(range(4, 9))].mean(axis=0)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    # It is NOT the same as the uniform mean over all layers.
    assert_that(bool(np.allclose(out, frame_emb.mean(axis=0)))).is_false()


def test_weight_layers_subset_explicit_indices() -> None:
    """``subset`` with explicit ``layers`` averages exactly those layers."""
    rng = np.random.default_rng(11)
    frame_emb = rng.standard_normal((6, 4, 3)).astype(np.float32)
    out = weight_layers(frame_emb, "subset", layers=(1, 3))
    expected = frame_emb[[1, 3]].mean(axis=0)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_weight_layers_subset_clamps_out_of_range_indices() -> None:
    """Indices outside [0, n_layers) are dropped; invalid-only falls back to all."""
    rng = np.random.default_rng(12)
    frame_emb = rng.standard_normal((4, 4, 3)).astype(np.float32)
    out = weight_layers(frame_emb, "subset", layers=(1, 99, -5))
    np.testing.assert_allclose(out, frame_emb[1], rtol=1e-5)
    # All-invalid layers fall back to a uniform mean over every layer.
    fallback = weight_layers(frame_emb, "subset", layers=(50, 60))
    np.testing.assert_allclose(fallback, frame_emb.mean(axis=0), rtol=1e-5)


def test_weight_layers_weighted_matches_softmax_weights() -> None:
    """``weighted`` mode is the softmax(layer_weights)-weighted layer sum."""
    rng = np.random.default_rng(13)
    n_layers, n_frames, hidden = 3, 5, 4
    frame_emb = rng.standard_normal((n_layers, n_frames, hidden)).astype(np.float32)
    weights = (2.0, 0.0, -1.0)
    out = weight_layers(frame_emb, "weighted", layer_weights=weights)
    assert_that(out.shape).is_equal_to((n_frames, hidden))

    w = np.exp(np.asarray(weights, dtype=np.float64))
    w = w / w.sum()
    expected = np.tensordot(w.astype(np.float32), frame_emb, axes=([0], [0]))
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_weight_layers_weighted_rejects_missing_or_mismatched_weights() -> None:
    """``weighted`` with None / wrong-length ``layer_weights`` is a config bug: it
    must fail fast — a silent uniform fallback would return a numerically
    different result than the one asked for, with no signal."""
    rng = np.random.default_rng(14)
    frame_emb = rng.standard_normal((4, 5, 3)).astype(np.float32)

    with pytest.raises(ValueError, match="one logit per layer"):
        weight_layers(frame_emb, "weighted", layer_weights=None)
    with pytest.raises(ValueError, match="2 entries but the model produced 4 layers"):
        weight_layers(frame_emb, "weighted", layer_weights=(1.0, 2.0))


def test_pool_mert_threads_subset_layer_mode(monkeypatch) -> None:
    """``pool_mert`` forwards ``config.mert_layers`` / mode into ``weight_layers``."""
    import moodengine.pooling as pooling

    captured: dict = {}
    real = pooling.weight_layers

    def _spy(frame_emb, mode, layers=None, layer_weights=None):
        captured["mode"] = mode
        captured["layers"] = layers
        captured["layer_weights"] = layer_weights
        return real(frame_emb, mode, layers=layers, layer_weights=layer_weights)

    monkeypatch.setattr(pooling, "weight_layers", _spy)
    cfg = dataclasses.replace(
        default_config(),
        pooling_mode="mean",
        mert_layer_weighting="subset",
        mert_layers=(1, 2),
    )
    seg = np.random.default_rng(15).standard_normal((4, 3, 5)).astype(np.float32)
    pooling.pool_mert([seg], cfg)
    assert_that(captured["mode"]).is_equal_to("subset")
    assert_that(captured["layers"]).is_equal_to((1, 2))


def test_pool_frames_mean_shape_and_value() -> None:
    """``mean`` pooling collapses the frame axis -> (hidden,)."""
    frames = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    out = pool_frames(frames, "mean")
    assert_that(out.shape).is_equal_to((2,))
    np.testing.assert_allclose(out, [2.0, 4.0], atol=1e-6)


def test_pool_frames_mean_std_shape_and_value() -> None:
    """``mean_std`` concatenates mean and std -> (2*hidden,)."""
    frames = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    out = pool_frames(frames, "mean_std")
    assert_that(out.shape).is_equal_to((4,))
    expected = np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_pool_frames_bad_mode_raises() -> None:
    """An unknown pooling mode raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"unknown pooling mode"):
        pool_frames(np.zeros((2, 3), dtype=np.float32), "median")


def test_pool_mert_mean_std_shape_and_unit_norm() -> None:
    """MERT pooling returns a unit-norm float32 vector of size 2*hidden."""
    cfg = dataclasses.replace(
        default_config(), pooling_mode="mean_std", mert_layer_weighting="uniform"
    )
    rng = np.random.default_rng(2)
    n_layers, hidden = 4, 6
    segments = [
        rng.standard_normal((n_layers, 5, hidden)).astype(np.float32),
        rng.standard_normal((n_layers, 7, hidden)).astype(np.float32),
    ]
    out = pool_mert(segments, cfg)
    assert_that(out.dtype).is_equal_to(np.float32)
    assert_that(out.shape).is_equal_to((2 * hidden,))
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_pool_mert_mean_mode_shape() -> None:
    """With ``mean`` pooling the MERT track vector is just (hidden,)."""
    cfg = dataclasses.replace(default_config(), pooling_mode="mean", mert_layer_weighting="last")
    rng = np.random.default_rng(3)
    hidden = 8
    segments = [rng.standard_normal((3, 4, hidden)).astype(np.float32)]
    out = pool_mert(segments, cfg)
    assert_that(out.shape).is_equal_to((hidden,))
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_pool_mert_concatenates_frames_across_segments() -> None:
    """All segment frames are pooled jointly (concat along the frame axis)."""
    cfg = dataclasses.replace(default_config(), pooling_mode="mean", mert_layer_weighting="last")
    # Two single-layer segments, distinct constant frames -> joint mean = overall mean.
    seg_a = np.full((1, 2, 3), 1.0, dtype=np.float32)  # 2 frames of 1.0
    seg_b = np.full((1, 4, 3), 4.0, dtype=np.float32)  # 4 frames of 4.0
    out = pool_mert([seg_a, seg_b], cfg)
    # Pre-normalization mean across 6 frames = (2*1 + 4*4)/6 = 3.0 in every dim;
    # after L2-normalize a constant vector of length 3 has each entry 1/sqrt(3).
    np.testing.assert_allclose(out, np.full(3, 1.0 / np.sqrt(3.0)), atol=1e-5)


def test_pool_mert_empty_raises() -> None:
    """Pooling with no segments is an error, not a silent empty vector."""
    with pytest.raises(ValueError, match=r"pool_mert received no segments"):
        pool_mert([], default_config())


def test_pool_clap_mean_then_normalize() -> None:
    """CLAP pooling averages clip embeddings then L2-normalizes."""
    cfg = default_config()
    segments = [
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ]
    out = pool_clap(segments, cfg)
    assert_that(out.shape).is_equal_to((3,))
    # mean = [1,0,0] -> normalized = [1,0,0]
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-6)


def test_pool_clap_empty_raises() -> None:
    """CLAP pooling with no segments raises ``ValueError``."""
    with pytest.raises(ValueError, match=r"pool_clap received no segments"):
        pool_clap([], default_config())


def test_poolers_registry_maps_names() -> None:
    """The registry exposes both poolers by embedder name."""
    assert_that(POOLERS["mert"]).is_same_as(pool_mert)
    assert_that(POOLERS["clap"]).is_same_as(pool_clap)
