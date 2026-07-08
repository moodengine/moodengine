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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

import moodengine.pipeline as pipeline
from moodengine.config import default_config

DIM = 8  # audio + text embedding dimensionality used by the fake embedder.


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
