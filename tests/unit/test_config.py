"""Tests for :mod:`moodengine.config` — the frozen run configuration + helpers."""

from __future__ import annotations

import dataclasses
import pathlib

import pytest
from assertpy import assert_that

from moodengine.config import (
    AUDIO_EXTENSIONS,
    Config,
    default_cache_dir,
    default_config,
    get_device,
)


def test_default_config_returns_config() -> None:
    """``default_config`` yields a :class:`Config` with the documented defaults."""
    cfg = default_config()
    assert_that(cfg).is_instance_of(Config)
    # MERT-v1's feature extractor expects 24 kHz; the default must match the model's rate.
    assert_that(cfg.mert_sample_rate).is_equal_to(24_000)
    assert_that(cfg.clap_sample_rate).is_equal_to(48_000)
    assert_that(cfg.segment_seconds).is_equal_to(10.0)
    assert_that(cfg.pooling_mode).is_equal_to("mean_std")
    assert_that(cfg.mert_layer_weighting).is_equal_to("uniform")
    assert_that(cfg.segment_selection).is_equal_to("uniform")
    assert_that(cfg.seed).is_equal_to(42)
    assert_that(cfg.audio_extensions).is_equal_to(AUDIO_EXTENSIONS)


def test_config_is_frozen() -> None:
    """The dataclass is immutable: attribute assignment raises."""
    cfg = default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 7  # type: ignore[misc]


def test_config_replace_overrides_field() -> None:
    """``dataclasses.replace`` produces a new config with one field changed."""
    cfg = default_config()
    other = dataclasses.replace(cfg, seed=123, kmeans_n_clusters=3)
    assert_that(other.seed).is_equal_to(123)
    assert_that(other.kmeans_n_clusters).is_equal_to(3)
    # Original is untouched (frozen + replace returns a copy).
    assert_that(cfg.seed).is_equal_to(42)
    assert_that(other).is_not_same_as(cfg)


def test_get_device_returns_known_string() -> None:
    """``get_device`` returns one of the supported device names."""
    assert_that(get_device()).is_in("cpu", "cuda", "mps")
    # The default config picks up a device via the same logic.
    assert_that(default_config().device).is_in("cpu", "cuda", "mps")


def test_default_paths_never_point_inside_the_package() -> None:
    """Installed as a wheel, the library must never write next to its own code:
    the default cache is the per-user platform cache dir, and raw/output default
    to workspace-relative paths meant to be overridden."""
    import moodengine

    package_root = pathlib.Path(moodengine.__file__).resolve().parent
    cfg = default_config()

    assert_that(default_cache_dir().is_absolute()).is_true()
    assert_that(cfg.cache_dir.resolve().is_relative_to(package_root)).is_false()
    assert_that(cfg.cache_dir).is_equal_to(default_cache_dir())
    assert_that(cfg.raw_dir.is_absolute()).is_false()  # workspace-relative: explicit in real use
    assert_that(cfg.output_dir.is_absolute()).is_false()


def test_audio_extensions_lowercase_with_dot() -> None:
    """Every supported extension is lowercase and dot-prefixed."""
    assert_that(AUDIO_EXTENSIONS).contains(".mp3")
    assert_that(AUDIO_EXTENSIONS).contains(".wav")
    for ext in AUDIO_EXTENSIONS:
        assert_that(ext).starts_with(".")
        assert_that(ext).is_equal_to(ext.lower())


def test_ensure_dirs_creates_directories(tmp_path) -> None:
    """``ensure_dirs`` creates the raw/cache/output directories if missing."""
    cfg = dataclasses.replace(
        default_config(),
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )
    cfg.ensure_dirs()
    assert_that(cfg.raw_dir.is_dir()).is_true()
    assert_that(cfg.cache_dir.is_dir()).is_true()
    assert_that(cfg.output_dir.is_dir()).is_true()


# --------------------------------------------------------------------------- #
# construction-time validation — a typo must fail fast, not produce a silently
# empty result three stages later
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pooling_mode", "mean-std", "pooling_mode"),  # the classic dash-typo
        ("mert_layer_weighting", "unifrom", "mert_layer_weighting"),
        ("projection_method", "tsne", "projection_method"),
        ("segment_seconds", 0.0, "segment_seconds"),
        ("segment_seconds", -1.0, "segment_seconds"),
        ("min_segment_seconds", 0.0, "min_segment_seconds"),
        ("overlap_seconds", -0.5, "overlap_seconds"),
        ("max_segments_per_track", -1, "max_segments_per_track"),
        ("fusion_weights", (1.0, 1.0, 1.0), "fusion_weights"),
        ("fusion_weights", (0.0, 0.0), "fusion_weights"),
        ("fusion_weights", (-0.5, 1.0), "fusion_weights"),
        ("umap_n_neighbors", 1, "umap_n_neighbors"),
        ("kmeans_n_clusters", 0, "kmeans_n_clusters"),
        ("leiden_resolution", 0.0, "leiden_resolution"),
        ("bootstrap_n", -1, "bootstrap_n"),
    ],
)
def test_config_rejects_invalid_value(field, value, match) -> None:
    """Each invalid value raises at construction, naming the offending field."""
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(default_config(), **{field: value})


def test_config_rejects_overlap_at_or_above_segment_length() -> None:
    """``overlap_seconds >= segment_seconds`` would stall the window; it is an
    argument error at construction, no longer a silent clamp downstream."""
    with pytest.raises(ValueError, match="overlap_seconds"):
        dataclasses.replace(default_config(), segment_seconds=10.0, overlap_seconds=10.0)


def test_config_replace_revalidates_derived_configs() -> None:
    """``dataclasses.replace`` re-runs ``__init__`` → derived configs are covered."""
    valid = default_config()

    with pytest.raises(ValueError, match="pooling_mode"):
        dataclasses.replace(valid, pooling_mode="nope")


def test_config_error_message_reports_received_value_and_options() -> None:
    """The message carries the received value AND the valid vocabulary."""
    with pytest.raises(ValueError, match=r"'mean-std'") as excinfo:
        dataclasses.replace(default_config(), pooling_mode="mean-std")

    assert_that(str(excinfo.value)).contains("mean_std")  # the valid options are listed


def test_config_boundary_values_are_accepted() -> None:
    """Legal edge values construct fine (0 overlap, 0 = no segment cap, k = 1)."""
    cfg = dataclasses.replace(
        default_config(),
        overlap_seconds=0.0,
        max_segments_per_track=0,
        kmeans_n_clusters=1,
        bootstrap_n=0,
        umap_n_neighbors=2,
    )
    assert_that(cfg.max_segments_per_track).is_equal_to(0)
