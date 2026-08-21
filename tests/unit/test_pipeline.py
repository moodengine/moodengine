"""End-to-end tests for :mod:`moodengine.pipeline` on the light (torch-free) stack.

``moodengine.pipeline.get_embedder`` is monkeypatched to return a fake embedder so no
real MERT/CLAP model (and therefore no torch) is ever constructed. The fake
turns a decoded waveform into a small deterministic vector and maps text prompts
to deterministic L2-normed vectors, which is enough to drive clustering,
labeling, attribute scoring, the markdown report and the HTML artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

import moodengine.io_audio as _io
import moodengine.pipeline as pipeline
from moodengine.config import default_config, ensure_clap_fusion_supported
from moodengine.exceptions import MissingDependencyError
from moodengine.labeling import attribute_scores

DIM = 8  # audio + text embedding dimensionality used by the fake embedder.

# The real get_embedder, captured before the autouse fixture below monkeypatches it, so the
# missing-``models``-extra test can exercise the genuine import path. torch lives only in that
# extra, so a light install is exactly the "backbones absent" situation the guard must handle.
_real_get_embedder = pipeline.get_embedder
_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


def _hash_unit_vec(key: bytes, dim: int) -> np.ndarray:
    """Deterministic unit vector seeded by ``key``."""
    seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _FakeEmbedder:
    """Torch-free stand-in usable as either a 'clap' or 'mert' embedder.

    ``extract`` derives a deterministic vector from the waveform's content so that
    distinct clips land at distinct points (enabling real clustering). For the
    'clap' name it returns a ``(dim,)`` clip embedding (what ``pool_clap``
    expects); for 'mert' a ``(n_layers, n_frames, hidden)`` tensor (what
    ``pool_mert`` expects). ``embed_text`` maps prompts to deterministic L2-normed
    rows so labeling/attribute stages are reproducible.
    """

    def __init__(self, name: str, sample_rate: int, dim: int = DIM) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.dim = dim

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        # A content fingerprint of the segment -> deterministic per-clip vector.
        key = wav.tobytes()[:4096] + str(round(float(wav.sum()), 3)).encode()
        vec = _hash_unit_vec(key, self.dim)
        if self.name == "mert":
            # (n_layers, n_frames, hidden): one layer, two frames of the vector.
            return np.stack([vec, vec], axis=0)[None, :, :].astype(np.float32)
        return vec  # CLAP-style clip embedding (hidden,)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        rows = [_hash_unit_vec(("txt:" + p).encode(), self.dim) for p in prompts]
        return np.vstack(rows).astype(np.float32)


@pytest.fixture()
def tmp_config(tmp_path):
    """A Config pointing every directory at an isolated tmp tree, with tiny audio."""
    base = default_config()
    return dataclasses.replace(
        base,
        raw_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        segment_seconds=0.5,
        min_segment_seconds=0.25,
        max_segments_per_track=2,
        kmeans_n_clusters=3,
    )


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch, tmp_config):
    """Replace ``get_embedder`` so no torch-backed model is ever constructed."""
    sr_by_name = {"clap": tmp_config.clap_sample_rate, "mert": tmp_config.mert_sample_rate}

    def _fake_get_embedder(name: str, config):
        key = name.lower()
        if key not in sr_by_name:
            raise ValueError(f"unknown embedder name: {name!r}")
        return _FakeEmbedder(key, sr_by_name[key])

    monkeypatch.setattr(pipeline, "get_embedder", _fake_get_embedder)


@pytest.mark.skipif(
    _TORCH_INSTALLED,
    reason="exercises the models-extra-absent path; only meaningful on a torch-free install",
)
@pytest.mark.parametrize("name", ["mert", "clap"])
def test_get_embedder_without_models_extra_raises_missing_dependency(name: str) -> None:
    # Building a real embedder without the models extra must surface the actionable
    # MissingDependencyError naming the extra, not a bare ModuleNotFoundError.
    with pytest.raises(MissingDependencyError, match=r"moodengine\[models\]"):
        _real_get_embedder(name, default_config())


# --------------------------------------------------------------------------- #
# CLAP fusion refusal — a config that would cache irreproducible vectors
# --------------------------------------------------------------------------- #


def test_clap_fusion_limit_accepts_the_default_segment_length() -> None:
    """The 10 s default lands on exactly the limit at 48 kHz, so it must pass."""
    ensure_clap_fusion_supported(default_config())


@pytest.mark.parametrize("segment_seconds", [10.01, 15.0, 30.0])
def test_clap_fusion_limit_refuses_segments_past_the_truncation_point(segment_seconds) -> None:
    """Past the limit laion-clap picks which chunks to keep from the GLOBAL unseeded numpy RNG,
    so the vector is not reproducible and would be cached as if it were."""
    config = dataclasses.replace(default_config(), segment_seconds=segment_seconds)

    with pytest.raises(ValueError, match="unseeded global numpy RNG"):
        ensure_clap_fusion_supported(config)


def test_lazy_embedder_refuses_the_clap_fusion_config_at_construction() -> None:
    """Eagerly, before any weight is touched: deferring the load moved the refusal inside
    `extract_embeddings`' per-file `except`, which turned it into one warning per file and an
    empty result matrix. It must not depend on how much of the run is served from cache either."""
    config = dataclasses.replace(default_config(), segment_seconds=15.0)

    with pytest.raises(ValueError, match="segment_seconds=15.0"):
        pipeline.LazyEmbedder("clap", config)


def test_lazy_embedder_allows_long_segments_for_mert() -> None:
    """The limit is a property of laion-clap, not of the configuration — a MERT-only run at 15 s
    segments is perfectly valid and must not be blocked by CLAP's constraint."""
    config = dataclasses.replace(default_config(), segment_seconds=15.0)

    embedder = pipeline.LazyEmbedder("mert", config)

    assert_that(embedder.loaded).is_false()
    assert_that(embedder.sample_rate).is_equal_to(config.mert_sample_rate)


def test_extract_embeddings_propagates_the_fusion_refusal_instead_of_skipping(
    tmp_path, monkeypatch
) -> None:
    """A configuration error is not a property of one file. Skipping it logged one identical
    warning per track and returned an empty matrix that reads exactly like "no audio found"."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.wav").write_bytes(b"")
    monkeypatch.setattr(_io, "discover_audio_files", lambda *_a, **_k: [raw / "a.wav"])
    config = dataclasses.replace(
        default_config(), raw_dir=raw, cache_dir=tmp_path / "cache", segment_seconds=15.0
    )

    with pytest.raises(ValueError, match="unseeded global numpy RNG"):
        pipeline.extract_embeddings(config, "clap")


def test_extract_embeddings_reraises_a_missing_dependency_instead_of_skipping_every_file(
    tmp_path, monkeypatch
) -> None:
    """Same reasoning for a missing backbone: the run cannot embed ANY file, so it fails once
    rather than emitting the same warning per track."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.wav").write_bytes(b"")
    monkeypatch.setattr(_io, "discover_audio_files", lambda *_a, **_k: [raw / "a.wav"])
    monkeypatch.setattr(_io, "load_audio", lambda *_a, **_k: np.zeros(24_000, dtype=np.float32))

    def _absent(_name, _config):
        raise MissingDependencyError("MuQ-MuLan embedding", "torch + muq", "models,muq")

    monkeypatch.setattr(pipeline, "get_embedder", _absent)
    config = dataclasses.replace(default_config(), raw_dir=raw, cache_dir=tmp_path / "cache")

    with pytest.raises(MissingDependencyError, match=r"moodengine\[models,muq\]"):
        pipeline.extract_embeddings(config, "mulan")


def test_cache_extra_encodes_the_decode_rate_for_an_embedder_with_no_legacy_cache() -> None:
    """An embedder absent from `_LEGACY_SAMPLE_RATE` must still get its decode rate into the key.

    This is the case the `-1` sentinel exists for, and the only one that isolates it. Defaulting
    the "legacy" rate to the embedder's CURRENT rate made the comparison a tautology: the field
    could never differ from its own legacy, so a config sitting on the legacy defaults everywhere
    else produced a key that did not encode the rate at all — and two runs at different decode
    rates would then share one cache entry.

    Deliberately NOT parametrized on 'mulan': `_embedding_cache_extra` folds `mulan_revision`
    into that branch, which forces the hash to differ on its own and would mask the sentinel.
    """
    config = dataclasses.replace(default_config(), segment_selection="head")

    at_24k = pipeline._embedding_cache_extra(_FakeEmbedder("future_backbone", 24_000), config)
    at_16k = pipeline._embedding_cache_extra(_FakeEmbedder("future_backbone", 16_000), config)

    assert_that(at_24k).contains("_cfg-")
    assert_that(at_24k).is_not_equal_to(at_16k)


def test_cache_extra_still_encodes_the_mulan_rate_through_its_own_branch() -> None:
    """MuQ-MuLan reaches the same outcome by a second route — its revision field — so pin that
    too rather than assuming the sentinel is what does the work there."""
    config = dataclasses.replace(default_config(), segment_selection="head")

    at_24k = pipeline._embedding_cache_extra(_FakeEmbedder("mulan", 24_000), config)
    at_16k = pipeline._embedding_cache_extra(_FakeEmbedder("mulan", 16_000), config)

    assert_that(at_24k).is_not_equal_to(at_16k)


def test_cache_extra_encodes_the_mulan_hub_revision() -> None:
    """The weights behind a cached vector are part of what produced it, and an unpinned hub
    reference can move underneath the cache — exactly why MERT pins its revision in the key."""
    base = default_config()
    emb = _FakeEmbedder("mulan", 24_000)

    assert_that(pipeline._embedding_cache_extra(emb, base)).is_not_equal_to(
        pipeline._embedding_cache_extra(emb, dataclasses.replace(base, mulan_revision="abc1234"))
    )


_BASE_COLUMNS = ["filename", "path", "cluster", "x", "y", "is_medoid", "outlier_score"]
_LABEL_COLUMNS = _BASE_COLUMNS + [
    "top_mood",
    "top_score",
    "mood_top3",
    "mood_top3_scores",
    "energy",
    "valence",
    "cluster_mood",
    "cluster_profile",
]


# --------------------------------------------------------------------------- #
# embedding cache key (_embedding_cache_extra) — every vector-affecting field
# --------------------------------------------------------------------------- #


def _legacy_config():
    """A config equivalent to the pre-1.0 defaults (head selection + 16 kHz MERT)."""
    return dataclasses.replace(default_config(), segment_selection="head", mert_sample_rate=16_000)


def test_cache_extra_preserves_legacy_key_for_legacy_config() -> None:
    """A legacy-equivalent config yields the byte-identical pre-1.0 tag (no ``_cfg`` suffix), so
    existing on-disk caches stay valid."""
    legacy = _legacy_config()

    mert = pipeline._embedding_cache_extra(_FakeEmbedder("mert", 16_000), legacy)
    clap = pipeline._embedding_cache_extra(_FakeEmbedder("clap", 48_000), legacy)

    assert_that(mert).is_equal_to("mean_std_seg10")
    assert_that(clap).is_equal_to("mean_std_seg10")


def test_cache_extra_busts_mert_for_the_24khz_default() -> None:
    """The 24 kHz + uniform default mints a NEW MERT key, so upgrading recomputes the off-rate
    16 kHz vectors instead of silently reusing them."""
    default_key = pipeline._embedding_cache_extra(_FakeEmbedder("mert", 24_000), default_config())

    assert_that(default_key).starts_with("mean_std_seg10_cfg-")
    assert_that(default_key).is_not_equal_to(
        pipeline._embedding_cache_extra(_FakeEmbedder("mert", 16_000), _legacy_config())
    )


def test_extract_embeddings_does_not_load_the_model_on_a_fully_cached_run(
    tmp_config, make_audio_library, monkeypatch
) -> None:
    """The weights used to be loaded before the cache was consulted, so a run that reads every
    vector off disk still paid a full model construction — ~7.5 s and gigabytes resident for CLAP,
    four times over inside `compare_spaces`."""
    make_audio_library(tmp_config.raw_dir, n=3)
    pipeline.extract_embeddings(tmp_config, "clap")  # warm the cache

    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "get_embedder",
        lambda name, config: calls.append(name) or _FakeEmbedder(name, 48_000),
    )
    files, X = pipeline.extract_embeddings(tmp_config, "clap")

    assert_that(calls).is_empty()  # every vector served from disk, no weights touched
    assert_that(files).is_length(3)
    assert_that(X.shape[0]).is_equal_to(3)


def test_lazy_embedder_exposes_identity_without_loading() -> None:
    """`name` and `sample_rate` are the only two facts a cache key needs, and both come from the
    config — reading them must not materialize a model."""
    lazy = pipeline.LazyEmbedder("clap", default_config())

    assert_that(lazy.name).is_equal_to("clap")
    assert_that(lazy.sample_rate).is_equal_to(48_000)
    assert_that(lazy.loaded).is_false()


def test_lazy_embedder_rejects_an_unknown_name_before_loading() -> None:
    """A typo must fail here, not after a multi-second weight load."""
    with pytest.raises(ValueError, match=r"unknown embedder name: 'nope'"):
        pipeline.LazyEmbedder("nope", default_config())


def test_track_embedding_from_waveform_falls_back_when_batching_is_absent(tmp_config) -> None:
    """The embedder contract is STRUCTURAL: a caller's own object need only provide `extract`.
    Requiring `extract_batch` would break every such implementation for a 12 % gain, so it is
    probed, not required."""

    class _ExtractOnly:
        name, sample_rate = "clap", 48_000

        def extract(self, waveform, sr):
            return np.full(4, float(np.asarray(waveform).size), dtype=np.float32)

    waveform = np.zeros(48_000 * 25, dtype=np.float32)

    vector = pipeline.track_embedding_from_waveform(_ExtractOnly(), waveform, 48_000, tmp_config)

    assert_that(vector.ndim).is_equal_to(1)
    assert_that(bool(np.all(np.isfinite(vector)))).is_true()


def test_track_embedding_from_waveform_rejects_an_off_rate_waveform() -> None:
    """The public post-decode entry point: a caller reaching it holds a waveform it decoded itself,
    and off-rate audio is time/pitch-warped in a way no downstream stage can detect — every
    embedding is silently degraded rather than failing. ``track_embedding`` always passes the
    embedder's own rate, so only this door needed the guard."""
    embedder = _FakeEmbedder("clap", 48_000)
    waveform = np.zeros(48_000, dtype=np.float32)

    with pytest.raises(ValueError, match=r"decoded at 44100 Hz but the 'clap' embedder expects"):
        pipeline.track_embedding_from_waveform(embedder, waveform, 44_100, default_config())


def test_cache_extra_distinguishes_fractional_segment_seconds() -> None:
    """10.2 s and 10.7 s no longer collide under the old ``int()`` truncation."""
    base = _legacy_config()
    emb = _FakeEmbedder("clap", 48_000)

    a = pipeline._embedding_cache_extra(emb, dataclasses.replace(base, segment_seconds=10.2))
    b = pipeline._embedding_cache_extra(emb, dataclasses.replace(base, segment_seconds=10.7))

    assert_that(a).is_not_equal_to(b)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("overlap_seconds", 2.0),
        ("min_segment_seconds", 3.0),
        ("max_segments_per_track", 6),
        ("segment_selection", "head"),
    ],
)
def test_cache_extra_changes_when_a_vector_affecting_field_changes(field, value) -> None:
    """Every previously-omitted segmentation field now enters the key, so changing it can never
    serve another setting's cached vectors."""
    base = default_config()
    emb = _FakeEmbedder("clap", 48_000)
    changed = dataclasses.replace(base, **{field: value})

    assert_that(pipeline._embedding_cache_extra(emb, base)).is_not_equal_to(
        pipeline._embedding_cache_extra(emb, changed)
    )


def test_run_pipeline_full_with_labels(tmp_config, make_audio_library):
    """A labeled kmeans run yields the full schema + all on-disk artifacts."""
    make_audio_library(tmp_config.raw_dir, n=8)
    df = pipeline.run_pipeline(tmp_config, embedder_name="clap", method="kmeans", with_labels=True)

    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(df).is_length(8)
    assert_that(list(df.columns)).is_equal_to(_LABEL_COLUMNS)

    # Label columns are well-formed.
    assert_that(bool(df["top_mood"].map(lambda m: isinstance(m, str) and m).all())).is_true()
    assert_that(bool(df["top_score"].between(0.0, 1.0).all())).is_true()
    assert_that(bool(df["energy"].between(0.0, 1.0).all())).is_true()
    assert_that(bool(df["valence"].between(0.0, 1.0).all())).is_true()
    for top3, scores in zip(df["mood_top3"], df["mood_top3_scores"]):
        assert_that(top3).is_instance_of(list)
        assert_that(scores).is_instance_of(list)
        assert_that(len(top3)).is_equal_to(len(scores))
        assert_that(len(scores)).is_greater_than_or_equal_to(1)
    assert_that(bool(df["cluster_profile"].map(lambda s: isinstance(s, str)).all())).is_true()
    # cluster_mood is consistent within a cluster.
    for _, grp in df.groupby("cluster"):
        assert_that(grp["cluster_mood"].nunique()).is_equal_to(1)

    # Medoid / outlier columns are well-formed; one medoid per non-noise cluster.
    assert_that(df["is_medoid"].dtype).is_equal_to(bool)
    assert_that(bool(df["outlier_score"].between(0.0, 1.0).all())).is_true()
    n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()
    assert_that(int(df.loc[df["cluster"] != -1, "is_medoid"].sum())).is_equal_to(n_clusters)

    out = tmp_config.output_dir
    for artifact in (
        "assignments.parquet",
        "clusters.html",
        "mood_space.html",
        "cluster_report.md",
        "dashboard.html",
    ):
        assert_that((out / artifact).exists()).is_true()

    # One .m3u playlist per cluster was written.
    m3us = list(out.glob("cluster_*.m3u"))
    assert_that(m3us).is_length(df["cluster"].nunique())

    # The parquet round-trips (list columns survive via pyarrow).
    round_trip = pd.read_parquet(out / "assignments.parquet")
    assert_that(round_trip).is_length(8)
    assert_that(list(round_trip.columns)).contains(*_LABEL_COLUMNS)

    # clusters.html carries the rich per-point hover (moods % + energy/valence).
    clusters_html = (out / "clusters.html").read_text(encoding="utf-8")
    assert_that(clusters_html).contains("Plotly")
    # dashboard.html is self-contained: plotly.js is inlined (its bundle text is
    # present) rather than pulled from a CDN <script src> include. (The inlined
    # bundle itself embeds inert SVG/XML namespace + mapbox icon URL string
    # literals, so we check for an actual external <script src> tag, not any
    # "http" substring.)
    dash_html = (out / "dashboard.html").read_text(encoding="utf-8")
    assert_that(dash_html).contains("Plotly")
    assert_that(dash_html.replace(" ", "")).does_not_contain("<script src=")
    assert_that(dash_html.replace(" ", "")).does_not_contain("<linkhref=")
    report = (out / "cluster_report.md").read_text(encoding="utf-8")
    assert_that(report).starts_with("#")
    assert_that(report).contains("kmeans")


def test_run_pipeline_auto_k_picks_cluster_count(tmp_config, make_audio_library):
    """KMeans auto-k chooses k via silhouette; the run stays internally consistent."""
    make_audio_library(tmp_config.raw_dir, n=8)
    df = pipeline.run_pipeline(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=True, auto_k=True
    )
    n_clusters = df.loc[df["cluster"] != -1, "cluster"].nunique()
    assert_that(n_clusters).is_greater_than_or_equal_to(2)  # auto-k found a non-trivial structure
    # Every row got a coordinate and a (possibly noisy) cluster id.
    assert_that(bool(df[["x", "y"]].notna().all().all())).is_true()


def test_run_pipeline_without_labels_has_base_schema_only(tmp_config, make_audio_library):
    """``with_labels=False`` keeps the base columns (incl. medoid/outlier) only."""
    make_audio_library(tmp_config.raw_dir, n=6)
    df = pipeline.run_pipeline(tmp_config, embedder_name="clap", method="kmeans", with_labels=False)
    assert_that(list(df.columns)).is_equal_to(_BASE_COLUMNS)
    # Medoid/outlier are computed even without labels (only need the clustering X).
    assert_that(df["is_medoid"].dtype).is_equal_to(bool)
    assert_that(bool(df["outlier_score"].between(0.0, 1.0).all())).is_true()
    out = tmp_config.output_dir
    assert_that((out / "assignments.parquet").exists()).is_true()
    assert_that((out / "clusters.html").exists()).is_true()
    assert_that((out / "cluster_report.md").exists()).is_true()
    assert_that((out / "dashboard.html").exists()).is_true()
    assert_that(list(out.glob("cluster_*.m3u"))).is_not_empty()
    # No mood space when labels are off.
    assert_that((out / "mood_space.html").exists()).is_false()


def test_run_pipeline_fused_space(tmp_config, make_audio_library):
    """``embedder_name='fused'`` clusters in the block-L2 MERT+CLAP space."""
    make_audio_library(tmp_config.raw_dir, n=8)
    df = pipeline.run_pipeline(tmp_config, embedder_name="fused", method="kmeans", with_labels=True)
    assert_that(df).is_length(8)
    assert_that(list(df.columns)).is_equal_to(_LABEL_COLUMNS)
    # Labels still come from CLAP regardless of the fused clustering space.
    assert_that(bool(df["top_mood"].map(lambda m: isinstance(m, str) and m).all())).is_true()


def test_fused_embeddings_dimension(tmp_config, make_audio_library):
    """Fused matrix width == MERT width + CLAP width over the shared files."""
    make_audio_library(tmp_config.raw_dir, n=6)
    _fm, Xm = pipeline.extract_embeddings(tmp_config, "mert")
    _fc, Xc = pipeline.extract_embeddings(tmp_config, "clap")
    files, Xf = pipeline.fused_embeddings(tmp_config)
    assert_that(files).is_length(6)
    assert_that(Xf.shape).is_equal_to((6, Xm.shape[1] + Xc.shape[1]))


def test_run_pipeline_empty_raw_dir_returns_empty_full_schema(tmp_config):
    """No audio -> an empty DataFrame with the full labeled schema, no crash."""
    tmp_config.raw_dir.mkdir(parents=True, exist_ok=True)
    df = pipeline.run_pipeline(tmp_config, embedder_name="clap", method="kmeans", with_labels=True)
    assert_that(df).is_length(0)
    assert_that(list(df.columns)).is_equal_to(_LABEL_COLUMNS)
    out = tmp_config.output_dir
    for artifact in (
        "assignments.parquet",
        "clusters.html",
        "mood_space.html",
        "cluster_report.md",
        "dashboard.html",
    ):
        assert_that((out / artifact).exists()).is_true()


def test_compare_spaces_returns_metrics_for_all_spaces(tmp_config, make_audio_library):
    """``compare_spaces`` reports MERT, CLAP and fused with the extended metrics."""
    make_audio_library(tmp_config.raw_dir, n=8)
    out = pipeline.compare_spaces(tmp_config, method="kmeans", auto_k=True)
    assert_that(set(out.keys())).is_equal_to({"mert", "clap", "fused"})
    for space, metrics in out.items():
        assert_that(metrics).is_instance_of(dict)
        assert_that(metrics).contains_key(
            "n_clusters",
            "noise_ratio",
            "silhouette",
            "silhouette_original",
            "stability_ari",
        )
        assert_that(metrics["stability_ari"]).is_instance_of(float)


def test_write_cluster_report_standalone(tmp_config):
    """``write_cluster_report`` writes a readable markdown file and returns its path."""
    df = pd.DataFrame(
        {
            "filename": ["a.wav", "b.wav", "c.wav"],
            "cluster": [0, 0, 1],
            "cluster_mood": ["happy", "happy", "dark"],
            "energy": [0.6, 0.7, 0.2],
            "valence": [0.8, 0.9, 0.1],
        }
    )
    profiles = {0: [("happy", 0.42), ("calm", 0.21)], 1: [("dark", 0.33)]}
    metrics = {"n_clusters": 2, "noise_ratio": 0.0, "silhouette": 0.55}
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    path = pipeline.write_cluster_report(
        df, profiles, metrics, tmp_config, "kmeans", out_path=out_path
    )
    assert_that(path).is_equal_to(out_path)
    text = path.read_text(encoding="utf-8")
    assert_that(text).contains("# Mood cluster report")
    assert_that(text).contains("kmeans")
    assert_that(text).contains("Cluster 0")
    assert_that(text).contains("Cluster 1")
    assert_that(text).contains("happy")
    assert_that(text).contains("dark")
    assert_that(text).contains("a.wav")  # example filename listed


def test_write_cluster_report_surfaces_the_original_space_silhouette_and_verdict(tmp_config):
    """The report labels the space its headline silhouette was scored in and prints the
    original-space score beside it, so a persisted file cannot certify UMAP's own artifact."""
    df = pd.DataFrame({"filename": ["a.wav"], "cluster": [0]})
    metrics = {
        "n_clusters": 3,
        "noise_ratio": 0.0,
        "silhouette": 0.275,
        "silhouette_space": "reduced",
        "silhouette_original": 0.011,
        "structure": "weak",
    }
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    text = pipeline.write_cluster_report(
        df, {}, metrics, tmp_config, "hdbscan", out_path=out_path
    ).read_text(encoding="utf-8")

    assert_that(text).contains("**Silhouette:** 0.275 (reduced space, euclidean)")
    assert_that(text).contains("**Silhouette (original space, cosine):** 0.011")
    assert_that(text).contains("**Structure:** weak")


def test_write_cluster_report_names_the_metric_not_only_the_space(tmp_config):
    """Both silhouettes can be scored in the SAME space and still differ — `cluster_metrics` uses
    sklearn's euclidean default while `silhouette_original` is always cosine. Labelling both
    "(original space)" printed two different numbers under one label, which reads as a
    contradiction."""
    df = pd.DataFrame({"filename": ["a.wav"], "cluster": [0]})
    metrics = {
        "n_clusters": 3,
        "noise_ratio": 0.0,
        "silhouette": 0.903,
        "silhouette_space": "original",
        "silhouette_original": 0.991,
        "structure": "clustered",
    }
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    text = pipeline.write_cluster_report(
        df, {}, metrics, tmp_config, "kmeans", out_path=out_path
    ).read_text(encoding="utf-8")

    assert_that(text).contains("**Silhouette:** 0.903 (original space, euclidean)")
    assert_that(text).contains("**Silhouette (original space, cosine):** 0.991")


def test_write_cluster_report_does_not_blame_a_reduction_that_never_ran(tmp_config):
    """On the original-space path there IS no reduction, so the warning must not attribute the
    clusters to one."""
    df = pd.DataFrame({"filename": ["a.wav"], "cluster": [0]})
    metrics = {
        "n_clusters": 19,
        "noise_ratio": 0.1,
        "silhouette": 0.2,
        "silhouette_space": "original",
        "silhouette_original": 0.01,
        "structure": "none_detected",
    }
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    text = pipeline.write_cluster_report(
        df, {}, metrics, tmp_config, "hdbscan", out_path=out_path
    ).read_text(encoding="utf-8")

    assert_that(text).contains("No substantial structure")
    assert_that(text).does_not_contain("artifact of the dimensionality reduction")
    assert_that(text).contains("produced in that same space")


def test_write_cluster_report_warns_in_the_file_when_no_structure_is_detected(tmp_config):
    """A `none_detected` verdict opens the report with an explicit warning. The log line
    run_clustering emits is gone by the time someone reads the file; this one is not."""
    df = pd.DataFrame({"filename": ["a.wav"], "cluster": [0]})
    metrics = {
        "n_clusters": 19,
        "noise_ratio": 0.1,
        "silhouette": 0.275,
        "silhouette_space": "reduced",
        "silhouette_original": 0.011,
        "structure": "none_detected",
    }
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    text = pipeline.write_cluster_report(
        df, {}, metrics, tmp_config, "hdbscan", out_path=out_path
    ).read_text(encoding="utf-8")

    assert_that(text).contains("No substantial structure")
    assert_that(text).contains("artifact of the dimensionality reduction")


def test_write_cluster_report_omits_the_honesty_fields_when_absent(tmp_config):
    """Bare `cluster_metrics` output (no run_clustering stamps) still renders a valid report:
    the extra lines are read with `.get`, never required."""
    df = pd.DataFrame({"filename": ["a.wav"], "cluster": [0]})
    metrics = {"n_clusters": 1, "noise_ratio": 0.0, "silhouette": 0.5}
    out_path = tmp_config.output_dir / "report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)

    text = pipeline.write_cluster_report(
        df, {}, metrics, tmp_config, "kmeans", out_path=out_path
    ).read_text(encoding="utf-8")

    assert_that(text).contains("**Silhouette:** 0.500")
    assert_that(text).does_not_contain("original space")
    assert_that(text).does_not_contain("**Structure:**")


def test_write_cluster_report_tolerates_missing_optionals(tmp_config):
    """The report does not crash when profiles/optional columns are absent."""
    df = pd.DataFrame({"filename": ["x.wav", "y.wav"], "cluster": [0, -1]})
    out_path = tmp_config.output_dir / "min_report.md"
    tmp_config.output_dir.mkdir(parents=True, exist_ok=True)
    path = pipeline.write_cluster_report(df, {}, {}, tmp_config, "hdbscan", out_path=out_path)
    text = path.read_text(encoding="utf-8")
    assert_that(text).contains("# Mood cluster report")
    # Noise cluster rendered last and labelled.
    assert_that(text).contains("noise")


def test_run_pipeline_passes_recenter_from_config(tmp_config, monkeypatch, make_audio_library):
    """``config.recenter_labels`` is plumbed into every labeling call."""
    make_audio_library(tmp_config.raw_dir, n=6)

    seen: dict[str, list[bool]] = {"label": [], "attr": [], "profiles": []}
    real_label = pipeline._labeling.label_tracks
    real_attr = pipeline._labeling.attribute_scores
    real_profiles = pipeline._labeling.cluster_mood_profiles

    def _spy_label(*args, recenter=True, **kw):
        seen["label"].append(recenter)
        return real_label(*args, recenter=recenter, **kw)

    def _spy_attr(*args, recenter=True, **kw):
        seen["attr"].append(recenter)
        return real_attr(*args, recenter=recenter, **kw)

    def _spy_profiles(*args, recenter=True, **kw):
        seen["profiles"].append(recenter)
        return real_profiles(*args, recenter=recenter, **kw)

    monkeypatch.setattr(pipeline._labeling, "label_tracks", _spy_label)
    monkeypatch.setattr(pipeline._labeling, "attribute_scores", _spy_attr)
    monkeypatch.setattr(pipeline._labeling, "cluster_mood_profiles", _spy_profiles)

    cfg_off = dataclasses.replace(tmp_config, recenter_labels=False)
    pipeline.run_pipeline(cfg_off, embedder_name="clap", method="kmeans", with_labels=True)
    assert_that(seen["label"]).is_equal_to([False])
    assert_that(seen["attr"]).is_equal_to([False])
    assert_that(seen["profiles"]).is_equal_to([False])

    for v in seen.values():
        v.clear()
    cfg_on = dataclasses.replace(tmp_config, recenter_labels=True)
    pipeline.run_pipeline(cfg_on, embedder_name="clap", method="kmeans", with_labels=True)
    assert_that(seen["label"]).is_equal_to([True])
    assert_that(seen["attr"]).is_equal_to([True])
    assert_that(seen["profiles"]).is_equal_to([True])


def test_extract_embeddings_on_progress_ticks_every_file(tmp_config, make_audio_library):
    """``on_progress(done, total, path)`` fires once per discovered file, in order."""
    make_audio_library(tmp_config.raw_dir, n=5)
    ticks: list[tuple[int, int, str]] = []

    files, X = pipeline.extract_embeddings(
        tmp_config,
        "clap",
        on_progress=lambda done, total, path: ticks.append((done, total, path.name)),
    )

    assert_that(files).is_length(5)
    assert_that(X.shape[0]).is_equal_to(5)
    assert_that([t[0] for t in ticks]).is_equal_to([1, 2, 3, 4, 5])
    assert_that(all(t[1] == 5 for t in ticks)).is_true()
    assert_that([t[2] for t in ticks]).is_equal_to([f.name for f in files])


def test_extract_embeddings_on_error_reports_skipped_files(
    tmp_config, monkeypatch, make_audio_library
):
    """A failing file lands in ``on_error`` (programmatic, not log-only) AND still
    ticks ``on_progress``; the run continues with the healthy files."""
    make_audio_library(tmp_config.raw_dir, n=4)
    poison = sorted(tmp_config.raw_dir.glob("*.wav"))[1]
    real_track_embedding = pipeline.track_embedding

    def _failing(embedder, path, config, force=False):
        if Path(path).name == poison.name:
            raise RuntimeError("simulated decode failure")
        return real_track_embedding(embedder, path, config, force=force)

    monkeypatch.setattr(pipeline, "track_embedding", _failing)
    errors: list[tuple[str, str]] = []
    ticks: list[int] = []

    files, X = pipeline.extract_embeddings(
        tmp_config,
        "clap",
        on_progress=lambda done, total, path: ticks.append(done),
        on_error=lambda path, exc: errors.append((path.name, str(exc))),
    )

    assert_that(files).is_length(3)  # the poison file was skipped
    assert_that(X.shape[0]).is_equal_to(3)
    assert_that(errors).is_equal_to([(poison.name, "simulated decode failure")])
    assert_that(ticks).is_equal_to([1, 2, 3, 4])  # progress covered the failed file too


def test_extract_embeddings_callback_exception_cancels_the_run(tmp_config, make_audio_library):
    """The documented cancellation path: raising inside ``on_progress`` aborts
    cleanly at a file boundary and propagates to the caller."""
    make_audio_library(tmp_config.raw_dir, n=4)

    class _Cancelled(Exception):
        pass

    def _cancel_after_two(done, total, path):
        if done == 2:
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        pipeline.extract_embeddings(tmp_config, "clap", on_progress=_cancel_after_two)


def test_run_pipeline_failed_clap_row_gets_sentinels_not_fabricated_labels(
    tmp_config, monkeypatch, make_audio_library
):
    """A track whose CLAP embedding fails keeps its cluster row but must show the
    honest sentinels (blank mood, NaN scores, empty lists) — labels computed from
    a zero vector would look exactly as plausible as the real ones."""
    make_audio_library(tmp_config.raw_dir, n=6)
    poison = sorted(tmp_config.raw_dir.glob("*.wav"))[2]
    real_track_embedding = pipeline.track_embedding

    def _clap_fails_for_poison(embedder, path, config, force=False):
        if embedder.name == "clap" and Path(path).name == poison.name:
            raise RuntimeError("simulated CLAP failure")
        return real_track_embedding(embedder, path, config, force=force)

    monkeypatch.setattr(pipeline, "track_embedding", _clap_fails_for_poison)

    # Cluster on MERT (so the poison file stays a row); label via CLAP (where it fails).
    df = pipeline.run_pipeline(tmp_config, embedder_name="mert", method="kmeans", with_labels=True)

    assert_that(df).is_length(6)  # the row is kept — it clusters on the primary space
    bad = df[df["filename"] == poison.name].iloc[0]
    good = df[df["filename"] != poison.name]
    assert_that(bad["top_mood"]).is_equal_to("")
    assert_that(bool(np.isnan(bad["top_score"]))).is_true()
    assert_that(bad["mood_top3"]).is_equal_to([])
    assert_that(bad["mood_top3_scores"]).is_equal_to([])
    assert_that(bool(np.isnan(bad["energy"]))).is_true()
    assert_that(bool(np.isnan(bad["valence"]))).is_true()
    # The healthy rows are labeled normally (the sentinel is per-row, not global).
    assert_that(bool((good["top_mood"] != "").all())).is_true()
    assert_that(bool(good[["top_score", "energy", "valence"]].notna().all().all())).is_true()


def test_run_pipeline_builds_the_mood_label_matrix_once(
    tmp_config, monkeypatch, make_audio_library
):
    """The mood vocabulary costs a text-encoder forward over every prompt; per-track
    labels and cluster profiles must share ONE build instead of paying it twice."""
    make_audio_library(tmp_config.raw_dir, n=6)

    seen_prompt_tables: list[dict] = []
    real_build = pipeline._labeling.build_label_matrix

    def _spy_build(embedder, prompts):
        seen_prompt_tables.append(prompts)
        return real_build(embedder, prompts)

    monkeypatch.setattr(pipeline._labeling, "build_label_matrix", _spy_build)

    pipeline.run_pipeline(tmp_config, embedder_name="clap", method="kmeans", with_labels=True)

    mood_builds = [p for p in seen_prompt_tables if p is pipeline._labeling.DEFAULT_MOOD_PROMPTS]
    assert_that(mood_builds).is_length(1)  # shared by label_tracks + cluster_mood_profiles
    # The only other builds are the two tiny attribute axes (energy, valence).
    assert_that(seen_prompt_tables).is_length(3)


# ---------------------------------------------------------------------------
# run_pipeline_core / write_artifacts — the compute/persist split
# ---------------------------------------------------------------------------


def test_run_pipeline_core_computes_everything_but_writes_nothing(tmp_config, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=6)

    result = pipeline.run_pipeline_core(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=True
    )

    assert_that(result).is_instance_of(pipeline.PipelineResult)
    assert_that(list(result.assignments.columns)).is_equal_to(_LABEL_COLUMNS)
    assert_that(result.assignments).is_length(6)
    assert_that(result.labels_requested).is_true()
    assert_that(result.have_labels).is_true()
    assert_that(result.coords2d.shape).is_equal_to((6, 2))
    assert_that(result.metrics["n_clusters"]).is_greater_than_or_equal_to(1)
    assert_that(result.profiles).is_not_empty()  # cluster profiles travel with the result
    # The whole point of the split: no artifact dir, no artifact files.
    assert_that(tmp_config.output_dir.exists()).is_false()


def test_run_pipeline_core_returns_all_three_recentering_priors(tmp_config, make_audio_library):
    """Every offset the run scored against travels with the result. The mood prior alone left
    `energy` and `valence` — two of the three shipped label columns — with no handle at all."""
    make_audio_library(tmp_config.raw_dir, n=6)

    result = pipeline.run_pipeline_core(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=True
    )

    assert_that(result.mood_prior).is_not_none()
    assert_that(result.energy_prior.shape).is_equal_to((2,))
    assert_that(result.valence_prior.shape).is_equal_to((2,))
    assert_that(result.energy_prior.dtype).is_equal_to(np.float32)


def test_run_pipeline_core_withholds_priors_below_the_recentering_floor(
    tmp_config, make_audio_library, caplog
):
    """A prior derived from too few rows and fed straight back subtracts each row's own
    similarity from itself — at n=1 exactly zero, collapsing every axis to 0.5. Below
    `RECENTER_MIN_N` the priors stay None and the documented small-batch behaviour applies."""
    make_audio_library(tmp_config.raw_dir, n=2)

    with caplog.at_level(logging.INFO, logger="moodengine.pipeline"):
        result = pipeline.run_pipeline_core(
            tmp_config, embedder_name="clap", method="kmeans", with_labels=True
        )

    assert_that(result.mood_prior).is_none()
    assert_that(result.energy_prior).is_none()
    assert_that(result.valence_prior).is_none()
    assert_that(caplog.text).contains("not estimating recentering priors")
    # ...and the axes are not the self-cancelling 0.5 the prior path produced.
    assert_that(float(result.assignments["energy"].iloc[0])).is_not_close_to(0.5, tolerance=1e-9)


def test_run_pipeline_core_priors_reproduce_a_single_track_rescore(tmp_config, make_audio_library):
    """The property the priors exist for: re-scoring ONE track later, with the offsets this run
    returned, lands on the same energy and valence the run published for it. Without them that
    call falls below `min_n` and skips centering entirely."""
    make_audio_library(tmp_config.raw_dir, n=6)
    result = pipeline.run_pipeline_core(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=True
    )
    files, clap_X = pipeline.extract_embeddings(tmp_config, "clap")
    row = files.index(Path(result.assignments["path"].iloc[0]))

    rescored = attribute_scores(
        clap_X[row : row + 1],
        pipeline.get_embedder("clap", tmp_config),
        energy_prior=result.energy_prior,
        valence_prior=result.valence_prior,
    )

    assert_that(float(rescored["energy"].iloc[0])).is_close_to(
        float(result.assignments["energy"].iloc[0]), tolerance=1e-6
    )
    assert_that(float(rescored["valence"].iloc[0])).is_close_to(
        float(result.assignments["valence"].iloc[0]), tolerance=1e-6
    )


def test_write_artifacts_persists_the_full_set_and_returns_paths(tmp_config, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=6)
    result = pipeline.run_pipeline_core(tmp_config, embedder_name="clap", method="kmeans")

    written = pipeline.write_artifacts(result)

    for key in ("assignments", "clusters_html", "mood_space_html", "report", "dashboard"):
        assert_that(written).contains_key(key)
        assert_that(written[key].exists()).is_true()
    m3us = [p for p in written.values() if p.suffix == ".m3u"]
    assert_that(m3us).is_length(result.assignments["cluster"].nunique())


def test_write_artifacts_honors_an_explicit_out_dir(tmp_config, tmp_path, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=6)
    result = pipeline.run_pipeline_core(tmp_config, embedder_name="clap", method="kmeans")
    elsewhere = tmp_path / "elsewhere"

    written = pipeline.write_artifacts(result, out_dir=elsewhere)

    assert_that(all(p.parent == elsewhere for p in written.values())).is_true()
    assert_that((elsewhere / "assignments.parquet").exists()).is_true()
    assert_that(tmp_config.output_dir.exists()).is_false()  # the default location stays untouched


def test_write_artifacts_without_labels_requested_skips_mood_space(tmp_config, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=6)
    result = pipeline.run_pipeline_core(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=False
    )

    written = pipeline.write_artifacts(result)

    assert_that(result.labels_requested).is_false()
    assert_that(result.have_labels).is_false()
    assert_that(written).does_not_contain_key("mood_space_html")
    assert_that((tmp_config.output_dir / "mood_space.html").exists()).is_false()


def test_run_pipeline_equals_core_plus_write_artifacts(tmp_config, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=6)

    df = pipeline.run_pipeline(tmp_config, embedder_name="clap", method="kmeans")
    result = pipeline.run_pipeline_core(tmp_config, embedder_name="clap", method="kmeans")

    # Deterministic seed + shared embedding cache: the wrapper IS core + write.
    pd.testing.assert_frame_equal(df, result.assignments)


def test_run_pipeline_core_bakes_auto_k_into_the_result_config(tmp_config, make_audio_library):
    make_audio_library(tmp_config.raw_dir, n=8)

    result = pipeline.run_pipeline_core(
        tmp_config, embedder_name="clap", method="kmeans", with_labels=False, auto_k=True
    )

    # The effective config carries the silhouette-picked k, and it matches the labels.
    assert_that(result.config.kmeans_n_clusters).is_equal_to(result.metrics["n_clusters"])


def test_run_pipeline_core_empty_raw_dir_yields_empty_result_with_schema(tmp_config):
    tmp_config.raw_dir.mkdir(parents=True, exist_ok=True)

    result = pipeline.run_pipeline_core(tmp_config, embedder_name="clap", with_labels=True)

    assert_that(result.assignments).is_length(0)
    assert_that(list(result.assignments.columns)).is_equal_to(_LABEL_COLUMNS)
    assert_that(result.labels_requested).is_true()
    assert_that(result.have_labels).is_false()
    assert_that(result.coords2d.shape).is_equal_to((0, 2))
    assert_that(result.profiles).is_equal_to({})


def test_build_hover_text_exact_strings() -> None:
    """Nothing asserted anything about the hover strings; the integration test would catch a crash
    or a mis-zip, not a wrong string. Covers the substitution for absent optional columns, an empty
    mood list, NaN energy/valence, and a non-default index."""
    df = pd.DataFrame(
        {
            "filename": ["a.wav", "b.wav", "c.wav"],
            "mood_top3": [["calm", "dark"], [], ["hyped"]],
            "mood_top3_scores": [[0.62, 0.31], [], [0.9]],
            "energy": [0.5, float("nan"), 0.25],
            "valence": [0.75, 0.4, float("nan")],
        },
        index=[7, 8, 9],  # a non-default index must not reach the output
    )

    hover = pipeline._build_hover_text(df)

    assert_that(hover).is_equal_to(
        [
            "a.wav<br>moods: calm 62%, dark 31%<br>energy 0.50 · valence 0.75",
            "b.wav",  # empty mood list AND NaN energy -> filename only
            "c.wav<br>moods: hyped 90%",  # NaN valence drops the attribute line
        ]
    )


def test_build_hover_text_without_the_optional_columns() -> None:
    """A frame carrying only `filename` must yield bare filenames, not a KeyError: the column-wise
    read substitutes a column of None where `row.get()` returned None for a missing key."""
    df = pd.DataFrame({"filename": ["only.wav", "two.wav"]})

    assert_that(pipeline._build_hover_text(df)).is_equal_to(["only.wav", "two.wav"])
