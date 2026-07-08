"""Integration test: track_embedding -> .npy cache -> reload.

Exercises the real on-disk embedding cache end-to-end (decode + segment + pool +
save + warm reload) with a torch-free fake embedder. AAA + fluent assertions;
uses pytest-mock to assert the warm call is a genuine cache hit (no re-embed).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import soundfile as sf
from assertpy import assert_that

from moodengine.config import default_config
from moodengine.embeddings.base import cache_key
from moodengine.pipeline import track_embedding


def test_track_embedding_caches_and_warm_reloads(tmp_path, make_fake_embedder, synth_clip, mocker):
    # Arrange
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=cache, segment_seconds=1.0)
    embedder = make_fake_embedder("clap", 48_000)
    spy = mocker.spy(embedder, "extract")

    # Act — cold (computes + writes), then warm (should hit cache)
    cold = track_embedding(embedder, wav, cfg)
    cold_calls = spy.call_count
    npys = list(cache.glob("*.npy"))
    warm = track_embedding(embedder, wav, cfg)

    # Assert
    assert_that(cold.ndim).is_equal_to(1)  # 1-D track vector
    assert_that(npys).is_length(1)  # exactly one cached embedding
    assert_that(cold_calls).is_greater_than(0)  # cold path embedded segments
    assert_that(spy.call_count).is_equal_to(cold_calls)  # warm path did NOT re-embed
    np.testing.assert_array_equal(warm, cold)  # identical vector returned


def test_legacy_equivalent_config_keeps_the_legacy_key(tmp_path, make_fake_embedder, synth_clip):
    """Backward-compat pin: a config equivalent to the pre-1.0 defaults (``head`` segment selection)
    still mints the exact pre-variant-tag key, so a CLAP cache computed under those settings is not
    orphaned. The 1.0 defaults (24 kHz MERT, ``uniform`` selection) deliberately mint a new key —
    see :func:`test_default_config_busts_the_legacy_key`."""
    # Arrange
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(
        default_config(),
        raw_dir=raw,
        cache_dir=cache,
        segment_seconds=1.0,
        segment_selection="head",  # the pre-1.0 behavior; CLAP is already 48 kHz by default
    )

    # Act
    track_embedding(make_fake_embedder("clap", 48_000), wav, cfg)

    # Assert
    expected = cache_key(wav, "clap", extra=f"{cfg.pooling_mode}_seg{int(cfg.segment_seconds)}")
    assert_that([p.name for p in cache.glob("*.npy")]).is_equal_to([f"{expected}.npy"])


def test_default_config_busts_the_legacy_key(tmp_path, make_fake_embedder, synth_clip):
    """The 1.0 default ``segment_selection="uniform"`` intentionally mints a NEW cache key, so an
    upgrade recomputes vectors instead of reusing head-truncated ones for tracks past the cap."""
    # Arrange
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=cache, segment_seconds=1.0)

    # Act
    track_embedding(make_fake_embedder("clap", 48_000), wav, cfg)

    # Assert
    legacy = (
        f"{cache_key(wav, 'clap', extra=f'{cfg.pooling_mode}_seg{int(cfg.segment_seconds)}')}.npy"
    )
    names = [p.name for p in cache.glob("*.npy")]
    assert_that(names).is_length(1)
    assert_that(names[0]).is_not_equal_to(legacy)
    assert_that(names[0]).contains("_cfg-")


def test_changing_model_variant_recomputes_instead_of_serving_stale_cache(
    tmp_path, make_fake_embedder, synth_clip, mocker
):
    # Arrange — warm the cache with the default CLAP variant
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=cache, segment_seconds=1.0)
    embedder = make_fake_embedder("clap", 48_000)
    track_embedding(embedder, wav, cfg)
    spy = mocker.spy(embedder, "extract")

    # Act — same file, different CLAP variants: each must get its own key
    for variant in (
        dataclasses.replace(cfg, clap_amodel="HTSAT-tiny"),
        dataclasses.replace(cfg, clap_checkpoint="/models/music_epoch_15.pt"),
        dataclasses.replace(cfg, clap_enable_fusion=True),
    ):
        track_embedding(embedder, wav, variant)

    # Assert — three re-embeds (no stale hit), four distinct cache entries
    assert_that(spy.call_count).is_greater_than_or_equal_to(3)
    assert_that(list(cache.glob("*.npy"))).is_length(4)


def test_mert_variant_does_not_invalidate_clap_cache(
    tmp_path, make_fake_embedder, synth_clip, mocker
):
    # Arrange — warm the CLAP cache
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=cache, segment_seconds=1.0)
    embedder = make_fake_embedder("clap", 48_000)
    track_embedding(embedder, wav, cfg)
    spy = mocker.spy(embedder, "extract")

    # Act — a MERT variant change is irrelevant to CLAP keys
    other = dataclasses.replace(cfg, mert_model_name="m-a-p/MERT-v1-330M")
    track_embedding(embedder, wav, other)

    # Assert — still a cache hit
    assert_that(spy.call_count).is_equal_to(0)


def test_force_recomputes_bypassing_cache(tmp_path, make_fake_embedder, synth_clip, mocker):
    # Arrange
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    wav = raw / "tone.wav"
    sf.write(wav, synth_clip("tone", seconds=2.0), 48_000)
    cfg = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=cache, segment_seconds=1.0)
    embedder = make_fake_embedder("clap", 48_000)
    track_embedding(embedder, wav, cfg)  # warm the cache
    spy = mocker.spy(embedder, "extract")

    # Act
    track_embedding(embedder, wav, cfg, force=True)

    # Assert
    assert_that(spy.call_count).is_greater_than(0)  # force re-embeds despite cache
