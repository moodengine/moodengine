"""End-to-end orchestration: embed -> cluster -> (optional) zero-shot mood labels.

This module wires the lightweight stages (I/O, pooling, clustering, labeling,
viz) together and handles the on-disk embedding cache. The concrete embedders
(MERT/CLAP) pull in torch, so they are imported *lazily* inside
:func:`get_embedder` -- importing this module only needs the lightweight stack.
Determinism follows ``config.seed`` wherever the underlying stages allow it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from moodengine import cluster as _cluster
from moodengine import io_audio as _io
from moodengine import labeling as _labeling
from moodengine import pooling as _pooling
from moodengine import viz as _viz
from moodengine._typing import ClusterMethod, ClusterMetrics, SupportsEmbedText
from moodengine.config import Config
from moodengine.embeddings.base import cache_key, load_cached, save_cached
from moodengine.pooling import POOLERS

logger = logging.getLogger(__name__)


def get_embedder(name: str, config: Config):
    """Lazily construct a concrete embedder by name.

    ``name`` is ``'mert'`` or ``'clap'``. The concrete module is imported inside
    this function so torch is only required when an embedder is actually built.
    Raises :class:`ValueError` for an unknown name.
    """
    key = name.lower()
    if key == "mert":
        from moodengine.embeddings.mert import MERTEmbedder

        return MERTEmbedder(config)
    if key == "clap":
        from moodengine.embeddings.clap import CLAPEmbedder

        return CLAPEmbedder(config)
    raise ValueError(f"unknown embedder name: {name!r} (expected 'mert' or 'clap')")


# The exact model variants the pre-tag cache keys were computed with. Frozen
# forever: vectors cached without a variant tag are only valid for these
# models, so even a future change of the Config defaults must not let a new
# default claim the untagged keys.
_LEGACY_MERT_MODEL = "m-a-p/MERT-v1-95M"
_LEGACY_CLAP_VARIANT = ("HTSAT-base", None, False)  # (amodel, checkpoint, enable_fusion)

# The decode rate the pre-1.0 cache keys were computed at, per embedder. MERT vectors cached before
# the 24 kHz fix were computed from 16 kHz audio, so the current 24 kHz default must NOT reuse them;
# CLAP was always 48 kHz, so its default keys stay valid. See :func:`_embedding_cache_extra`.
_LEGACY_SAMPLE_RATE = {"mert": 16_000, "clap": 48_000}

# The segmentation-slice defaults the legacy (hand-picked) cache key implicitly assumed. A config
# whose vector-affecting fields all equal these keeps its byte-identical legacy key; any deviation
# adds a hash suffix so a changed knob never silently serves vectors computed under the old value.
_LEGACY_CACHE_DEFAULTS = {
    "overlap_seconds": 0.0,
    "min_segment_seconds": 1.0,
    "max_segments_per_track": 12,
    "segment_selection": "head",  # the pre-1.0 first-N behavior
    # Only the FRACTIONAL part of segment_seconds — the integer seconds are already in the readable
    # prefix (`seg{int(...)}`); hashing the fraction is what distinguishes 10.2 s from 10.7 s (which
    # the legacy int() truncation collapsed) without busting integer-second configs like 1.0 or 10.0.
    "segment_frac": 0.0,
}


def _model_variant_tag(embedder_name: str, config: Config) -> str:
    """Cache-key component identifying the model variant behind ``embedder_name``.

    Empty for the legacy variants above (existing on-disk caches keep their
    keys); any other variant yields a distinct, filesystem-safe tag. Without
    this, switching model variants would silently serve vectors computed by
    the previous model. Checkpoint paths are hashed, not embedded verbatim —
    they can contain separators and are irrelevant to key readability.
    """
    if embedder_name == "mert":
        if config.mert_model_name == _LEGACY_MERT_MODEL:
            return ""
        return config.mert_model_name.replace("/", "_")
    if embedder_name == "clap":
        current = (config.clap_amodel, config.clap_checkpoint, config.clap_enable_fusion)
        if current == _LEGACY_CLAP_VARIANT:
            return ""
        parts = [config.clap_amodel.replace("/", "_")]
        if config.clap_enable_fusion:
            parts.append("fusion")
        if config.clap_checkpoint is not None:
            parts.append(hashlib.sha1(str(config.clap_checkpoint).encode()).hexdigest()[:8])
        return "-".join(parts)
    return ""


def _embedding_cache_extra(embedder, config: Config) -> str:
    """Build the cache-key ``extra`` tag covering every config field that changes the pooled vector.

    Two parts. A human-readable legacy prefix (``pooling_mode`` + integer ``segment_seconds``
    [+ MERT layer weighting] [+ model variant]) keeps existing default caches debuggable. Then a
    ``_cfg-<hash>`` suffix hashing the remaining vector-affecting fields — the segmentation slice
    (overlap, min-segment, cap, selection policy, and the FRACTIONAL part of ``segment_seconds``,
    fixing the old ``int()`` truncation that collided 10.2 s with 10.7 s), the decode sample rate,
    and (MERT only) the hub revision and per-layer weights — but ONLY when any of them differs from
    the pre-1.0 defaults (:data:`_LEGACY_CACHE_DEFAULTS` / :data:`_LEGACY_SAMPLE_RATE`). So a
    legacy-equivalent config yields the byte-identical legacy key, while changing e.g. ``overlap``
    or the 24 kHz MERT rate mints a new key instead of silently reusing another setting's vectors.
    """
    extra = f"{config.pooling_mode}_seg{int(config.segment_seconds)}"
    layer_mode = getattr(config, "mert_layer_weighting", "uniform")
    if layer_mode != "uniform":
        extra = f"{extra}_lw-{layer_mode}"
        layers = getattr(config, "mert_layers", None)
        if layers:
            extra = f"{extra}-{'.'.join(str(int(i)) for i in layers)}"
    variant = _model_variant_tag(embedder.name, config)
    if variant:
        extra = f"{extra}_mv-{variant}"

    fields: dict[str, Any] = {
        "overlap_seconds": float(config.overlap_seconds),
        "min_segment_seconds": float(config.min_segment_seconds),
        "max_segments_per_track": int(config.max_segments_per_track),
        "segment_selection": str(config.segment_selection),
        "segment_frac": round(float(config.segment_seconds) - int(config.segment_seconds), 6),
        "sample_rate": int(embedder.sample_rate),
    }
    legacy = dict(_LEGACY_CACHE_DEFAULTS)
    legacy["sample_rate"] = _LEGACY_SAMPLE_RATE.get(embedder.name, int(embedder.sample_rate))
    if embedder.name == "mert":
        fields["mert_revision"] = config.mert_revision
        fields["mert_layer_weights"] = (
            list(config.mert_layer_weights) if config.mert_layer_weights else None
        )
        legacy["mert_revision"] = None
        legacy["mert_layer_weights"] = None

    if fields != legacy:
        digest = hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[
            :10
        ]
        extra = f"{extra}_cfg-{digest}"
    return extra


def track_embedding(embedder, path, config: Config, force: bool = False) -> np.ndarray:
    """Return the cached track-level embedding for one audio file.

    The cache key is ``cache_key(path, embedder.name, extra=_embedding_cache_extra(embedder,
    config))`` — the ``extra`` tag covers EVERY config field that changes the pooled vector (see
    :func:`_embedding_cache_extra`), so changing a segmentation or model knob mints a new key
    instead of silently serving vectors computed under the old value. On a cache hit the stored
    vector is returned (unless ``force``). Otherwise the file is decoded at ``embedder.sample_rate``,
    segmented, each segment embedded via ``embedder.extract``, pooled by ``POOLERS[embedder.name]``,
    persisted, and returned. Result is a 1-D float32 array.

    A legacy-equivalent config keeps its byte-identical pre-1.0 key; the 24 kHz MERT default and the
    ``"uniform"`` segment-selection default deliberately mint new keys, so upgrading recomputes the
    affected vectors once rather than reusing off-rate / head-truncated ones.
    """
    key = cache_key(path, embedder.name, extra=_embedding_cache_extra(embedder, config))

    if not force:
        cached = load_cached(config.cache_dir, key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)

    waveform = _io.load_audio(path, embedder.sample_rate, config)
    vector = track_embedding_from_waveform(embedder, waveform, embedder.sample_rate, config)

    save_cached(config.cache_dir, key, vector)
    return vector


def track_embedding_from_waveform(embedder, waveform, sr, config: Config) -> np.ndarray:
    """The post-decode half of :func:`track_embedding`: segment → embed → pool → 1-D float32.

    Split out so a caller that has ALREADY decoded the file can avoid a second decode — e.g. the
    caller's indexer decodes each track once and derives both the CLAP embedding (here) and the
    librosa BPM/key signals from the same waveform. ``sr`` must be ``embedder.sample_rate`` (the rate
    the waveform was decoded at). Byte-identical to the inline body it replaced, so existing ``.npy``
    caches stay valid. Does no I/O and no caching (the caller owns those).
    """
    segments = _io.segment_waveform(waveform, sr, config)
    embedded = [embedder.extract(seg, sr) for seg in segments]
    pooler = POOLERS[embedder.name]
    return np.asarray(pooler(embedded, config), dtype=np.float32)


def extract_embeddings(
    config: Config,
    embedder_name: str,
    force: bool = False,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Embed every discovered audio file under ``config.raw_dir``.

    Discovers files, computes a cached :func:`track_embedding` for each, and
    returns the aligned ``(files, X)`` where ``X`` is ``(n, d)`` float32. Files
    that fail to decode/embed are logged and skipped, so ``files`` and ``X`` stay
    row-aligned; ``on_error(path, exc)`` — when given — additionally receives
    each skipped file, so failures are observable programmatically, not only in
    the logs. ``on_progress(done, total, path)`` — when given — is called after
    EVERY file, succeeded or skipped. The file boundary is also the natural
    cancellation point: an exception raised inside either callback aborts the
    run cleanly, and the per-file cache makes the next run resume for free.
    Returns an empty list and an empty ``(0, 0)`` matrix when nothing is found
    or everything fails.
    """
    embedder = get_embedder(embedder_name, config)
    discovered = _io.discover_audio_files(config.raw_dir, config)
    total = len(discovered)

    files: list[Path] = []
    vectors: list[np.ndarray] = []
    for done, path in enumerate(discovered, start=1):
        try:
            vec = track_embedding(embedder, path, config, force=force)
        except Exception as exc:  # noqa: BLE001 - skip + continue on any failure
            logger.warning("Skipping %s: %s", path, exc)
            if on_error is not None:
                on_error(Path(path), exc)
        else:
            files.append(Path(path))
            vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        if on_progress is not None:
            on_progress(done, total, Path(path))

    if not vectors:
        return files, np.empty((0, 0), dtype=np.float32)

    X = np.vstack(vectors).astype(np.float32)
    return files, X


def fused_embeddings(
    config: Config,
    force: bool = False,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Extract a fused MERT+CLAP track-embedding space.

    Extracts both the MERT and CLAP track matrices over ``config.raw_dir`` (each
    via :func:`extract_embeddings`), aligns them on the files common to both
    (matching on path; both extract from the same ``raw_dir`` so order normally
    coincides, but the intersection is taken defensively), block-L2-normalizes
    each matrix, scales them by ``config.fusion_weights`` ``(w_m, w_c)`` and
    horizontally stacks them into ``(n, d_mert + d_clap)``. Returns the aligned
    ``(files, X_fused)``. ``on_progress`` / ``on_error`` follow the
    :func:`extract_embeddings` contract; the ``(done, total)`` counters restart
    for each embedding space (MERT pass first, then CLAP). An empty ``(0, 0)``
    matrix is returned when either space has no usable embeddings or the two
    share no files.
    """
    files_m, Xm = extract_embeddings(
        config, "mert", force=force, on_progress=on_progress, on_error=on_error
    )
    files_c, Xc = extract_embeddings(
        config, "clap", force=force, on_progress=on_progress, on_error=on_error
    )

    if Xm.shape[0] == 0 or Xm.shape[1] == 0 or Xc.shape[0] == 0 or Xc.shape[1] == 0:
        logger.warning("Fused space unavailable: an input space had no embeddings.")
        return [], np.empty((0, 0), dtype=np.float32)

    # Align on the intersection of paths, preserving the MERT (raw_dir) order.
    idx_c = {str(p): i for i, p in enumerate(files_c)}
    files: list[Path] = []
    rows_m: list[int] = []
    rows_c: list[int] = []
    for i, p in enumerate(files_m):
        j = idx_c.get(str(p))
        if j is not None:
            files.append(Path(p))
            rows_m.append(i)
            rows_c.append(j)

    if not files:
        logger.warning("Fused space unavailable: MERT and CLAP share no files.")
        return [], np.empty((0, 0), dtype=np.float32)

    Xm_a = _pooling.l2_normalize(Xm[rows_m], axis=1)
    Xc_a = _pooling.l2_normalize(Xc[rows_c], axis=1)
    w_m, w_c = config.fusion_weights
    fused = np.hstack([Xm_a * float(w_m), Xc_a * float(w_c)]).astype(np.float32)
    return files, fused


@dataclass(frozen=True)
class PipelineResult:
    """Everything one pipeline run computed, before any artifact is written.

    ``assignments`` is the per-track DataFrame (column contract in
    :func:`run_pipeline_core`); ``labels`` / ``coords2d`` are its cluster ids
    and 2-D map coordinates as arrays; ``metrics`` is the clustering summary
    and ``profiles`` maps each cluster id to its ranked ``(mood, score)``
    profile (empty when labels were not computed). ``config`` is the EFFECTIVE
    configuration — KMeans auto-k is baked into ``kmeans_n_clusters``.
    ``labels_requested`` records the ``with_labels`` argument, and
    ``have_labels`` tells whether the label columns carry real content
    (``False`` when labeling was requested but no CLAP embedding succeeded).
    """

    assignments: pd.DataFrame
    labels: NDArray[np.int_]
    coords2d: NDArray[np.float32]
    metrics: ClusterMetrics
    method: ClusterMethod
    profiles: dict[int, list[tuple[str, float]]]
    config: Config
    labels_requested: bool
    have_labels: bool


def run_pipeline_core(
    config: Config,
    embedder_name: str = "mert",
    method: ClusterMethod = "hdbscan",
    with_labels: bool = True,
    force: bool = False,
    auto_k: bool = True,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> PipelineResult:
    """Compute the full pipeline and return a :class:`PipelineResult` — no artifacts.

    The pure half of :func:`run_pipeline`: it embeds, clusters and (when
    ``with_labels``) labels, but writes nothing under ``config.output_dir`` and
    creates no directories (the embedding cache under ``config.cache_dir`` is
    the embedders' own concern and fills itself). Persisting the artifact set
    is :func:`write_artifacts`; an application that only wants the DataFrame
    simply never calls it.

    ``embedder_name`` selects the CLUSTERING space: ``'mert'`` (default),
    ``'clap'`` or ``'fused'`` (block-L2-normalized MERT+CLAP via
    :func:`fused_embeddings`). Labels ALWAYS come from CLAP regardless of the
    clustering space.

    The ``assignments`` columns are ``filename``, ``path``, ``cluster``, ``x``,
    ``y``, ``is_medoid``, ``outlier_score``; with labels the frame also carries
    ``top_mood``, ``top_score``, ``mood_top3``, ``mood_top3_scores``,
    ``energy``, ``valence``, ``cluster_mood`` and ``cluster_profile``.
    ``is_medoid`` flags each cluster's representative track and
    ``outlier_score`` is ``1 - cosine`` to the cluster centroid (both computed
    on the clustering space ``X``, even when ``with_labels`` is False).

    For ``method == 'kmeans'`` with ``auto_k`` and ``n >= 3``, the number of
    clusters is chosen by silhouette via :func:`cluster.select_kmeans_k` and
    baked into ``result.config``. For labeling, CLAP audio embeddings are
    needed: when ``embedder_name == 'clap'`` the already-computed matrix is
    reused; otherwise a separate set of CLAP track embeddings is extracted for
    the same files — a file whose CLAP embedding fails keeps its row (clustered
    on the primary space) but gets the honest sentinels in every label column
    (``top_mood=""``, NaN scores, empty lists) instead of labels fabricated
    from a zero vector. Per-axis label recentering follows
    ``config.recenter_labels``. ``on_progress`` / ``on_error`` follow the
    :func:`extract_embeddings` contract and cover every per-file embedding
    loop of the run (the counters restart per pass). ``n == 0`` yields an
    empty ``assignments`` frame with the full column schema, so downstream
    consumers see a stable shape.
    """
    if embedder_name.lower() == "fused":
        files, X = fused_embeddings(config, force=force, on_progress=on_progress, on_error=on_error)
    else:
        files, X = extract_embeddings(
            config, embedder_name, force=force, on_progress=on_progress, on_error=on_error
        )
    filenames = [p.name for p in files]
    paths = [str(p) for p in files]
    n = len(files)

    # Degenerate guard: nothing to cluster/label, but keep the full schema.
    if n == 0:
        df = pd.DataFrame(
            {
                "filename": pd.Series([], dtype=str),
                "path": pd.Series([], dtype=str),
                "cluster": pd.Series([], dtype=int),
                "x": pd.Series([], dtype=float),
                "y": pd.Series([], dtype=float),
                "is_medoid": pd.Series([], dtype=bool),
                "outlier_score": pd.Series([], dtype=float),
            }
        )
        if with_labels:
            df["top_mood"] = pd.Series([], dtype=str)
            df["top_score"] = pd.Series([], dtype=float)
            df["mood_top3"] = pd.Series([], dtype=object)
            df["mood_top3_scores"] = pd.Series([], dtype=object)
            df["energy"] = pd.Series([], dtype=float)
            df["valence"] = pd.Series([], dtype=float)
            df["cluster_mood"] = pd.Series([], dtype=str)
            df["cluster_profile"] = pd.Series([], dtype=str)
        return PipelineResult(
            assignments=df,
            labels=np.empty(0, dtype=int),
            coords2d=np.empty((0, 2), dtype=np.float32),
            metrics=_cluster.cluster_metrics(np.empty((0, 0)), np.empty(0)),
            method=method,
            profiles={},
            config=config,
            labels_requested=with_labels,
            have_labels=False,
        )

    # KMeans auto-k: pick the silhouette-best k and bake it into a copy of config.
    if method == "kmeans" and auto_k and n >= 3:
        best_k, _scores = _cluster.select_kmeans_k(X, config)
        config = replace(config, kmeans_n_clusters=best_k)

    result = _cluster.run_clustering(X, method, config)
    labels = np.asarray(result["labels"], dtype=int)
    coords2d = np.asarray(result["coords2d"], dtype=np.float32)
    metrics = result["metrics"]

    df = pd.DataFrame(
        {
            "filename": filenames,
            "path": paths,
            "cluster": labels.astype(int),
            "x": coords2d[:, 0].astype(float),
            "y": coords2d[:, 1].astype(float),
        }
    )

    # Medoid (representative) + outlier score on the clustering space. Independent
    # of labeling, so they are attached even when ``with_labels`` is False.
    medoids = _cluster.cluster_medoids(X, labels)
    medoid_idx = set(medoids.values())
    df["is_medoid"] = [i in medoid_idx for i in range(n)]
    df["outlier_score"] = _cluster.outlier_scores(X, labels).astype(float)

    profiles: dict[int, list[tuple[str, float]]] = {}
    have_labels = False

    if with_labels:
        clap_X, clap_embedder, clap_valid = _clap_embeddings_for(
            config, embedder_name, files, X, force=force, on_progress=on_progress, on_error=on_error
        )
        if clap_X.shape[1] == 0:
            # Every CLAP audio embedding failed (or yielded an empty vector):
            # labeling has nothing to score against, so emit blank label columns
            # rather than crashing on a (n, 0) @ (n_moods, d) shape mismatch.
            logger.warning("No CLAP audio embeddings available; skipping mood labels.")
            df["top_mood"] = ""
            df["top_score"] = float("nan")
            df["mood_top3"] = [[] for _ in range(n)]
            df["mood_top3_scores"] = [[] for _ in range(n)]
            df["energy"] = float("nan")
            df["valence"] = float("nan")
            df["cluster_mood"] = ""
            df["cluster_profile"] = ""
            profiles = {}
        else:
            recenter = config.recenter_labels
            # The mood label matrix costs a text-encoder forward over the whole
            # prompt vocabulary — build it once and share it between per-track
            # labels and cluster profiles instead of paying it twice.
            mood_lm = _labeling.build_label_matrix(clap_embedder, _labeling.DEFAULT_MOOD_PROMPTS)

            # Score ONLY the rows whose CLAP embedding succeeded: a zero row would
            # both receive fabricated labels and bias the per-mood recentering
            # means for every real track.
            valid_idx = np.flatnonzero(clap_valid)
            X_valid = clap_X if clap_valid.all() else clap_X[valid_idx]
            ld = _labeling.label_tracks(X_valid, recenter=recenter, label_matrix=mood_lm)
            attr = _labeling.attribute_scores(X_valid, clap_embedder, recenter=recenter)

            # Scatter back into full-length columns; failed rows get the same
            # sentinels as the all-failed path (blank mood, NaN scores, empty lists).
            top_mood = np.full(n, "", dtype=object)
            top_mood[valid_idx] = ld["top_mood"].astype(str).to_numpy()
            top_score = np.full(n, np.nan, dtype=float)
            top_score[valid_idx] = ld["top_score"].astype(float).to_numpy()
            top3: list[list] = [[] for _ in range(n)]
            top3_scores: list[list] = [[] for _ in range(n)]
            for pos, row in enumerate(valid_idx):
                top3[row] = ld["mood_topk"].iloc[pos]
                top3_scores[row] = ld["mood_topk_scores"].iloc[pos]
            energy = np.full(n, np.nan, dtype=float)
            energy[valid_idx] = attr["energy"].astype(float).to_numpy()
            valence = np.full(n, np.nan, dtype=float)
            valence[valid_idx] = attr["valence"].astype(float).to_numpy()

            df["top_mood"] = [str(v) for v in top_mood]
            df["top_score"] = top_score
            df["mood_top3"] = top3
            df["mood_top3_scores"] = top3_scores
            df["energy"] = energy
            df["valence"] = valence

            profiles = _labeling.cluster_mood_profiles(
                X_valid, labels[valid_idx], recenter=recenter, label_matrix=mood_lm
            )
            cluster_mood = {cid: (profs[0][0] if profs else "") for cid, profs in profiles.items()}
            cluster_profile = {
                cid: ", ".join(f"{m} {s:.2f}" for m, s in profs) for cid, profs in profiles.items()
            }
            df["cluster_mood"] = df["cluster"].map(cluster_mood).fillna("").astype(str)
            df["cluster_profile"] = df["cluster"].map(cluster_profile).fillna("").astype(str)

            have_labels = True

    return PipelineResult(
        assignments=df,
        labels=labels,
        coords2d=coords2d,
        metrics=metrics,
        method=method,
        profiles=profiles,
        config=config,
        labels_requested=with_labels,
        have_labels=have_labels,
    )


def write_artifacts(result: PipelineResult, out_dir: Path | None = None) -> dict[str, Path]:
    """Persist the artifact set for ``result`` and return ``{name: path}``.

    The imperative half of :func:`run_pipeline`. Writes ``assignments.parquet``,
    an interactive cluster scatter ``clusters.html``, a mood-space scatter
    ``mood_space.html`` (whenever labels were REQUESTED — empty when they were
    unavailable, so the artifact set stays stable), a readable
    ``cluster_report.md``, a self-contained ``dashboard.html`` and one ``.m3u``
    playlist per cluster (keyed by playlist file stem). ``out_dir`` defaults to
    ``result.config.output_dir``; it is created if missing.
    """
    config = result.config
    out = Path(out_dir) if out_dir is not None else Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = result.assignments
    filenames = df["filename"].astype(str).tolist()
    moods = df["top_mood"].astype(str).tolist() if result.have_labels else None
    hover = _build_hover_text(df) if result.have_labels else None

    written: dict[str, Path] = {}

    df.to_parquet(out / "assignments.parquet", index=False)
    written["assignments"] = out / "assignments.parquet"

    _viz.plot_clusters(
        result.coords2d,
        list(result.labels),
        filenames,
        moods=moods,
        title="Mood clusters",
        out_html=out / "clusters.html",
        hover_text=hover,
    )
    written["clusters_html"] = out / "clusters.html"

    if result.have_labels:
        _viz.plot_attributes(
            df["energy"].tolist(),
            df["valence"].tolist(),
            list(result.labels),
            filenames,
            moods=moods,
            title="Mood space (valence × energy)",
            out_html=out / "mood_space.html",
        )
        written["mood_space_html"] = out / "mood_space.html"
    elif result.labels_requested:
        # Labels requested but unavailable: still emit an empty scatter so the
        # artifact set is stable for downstream consumers.
        _viz.plot_attributes(
            [],
            [],
            [],
            [],
            title="Mood space (valence × energy)",
            out_html=out / "mood_space.html",
        )
        written["mood_space_html"] = out / "mood_space.html"

    written["report"] = write_cluster_report(
        df,
        result.profiles,
        result.metrics,
        config,
        result.method,
        out_path=out / "cluster_report.md",
    )

    # Restitution / UX: a self-contained dashboard + per-cluster .m3u playlists.
    _viz.build_dashboard(df, out / "dashboard.html", audio_dir=config.raw_dir)
    written["dashboard"] = out / "dashboard.html"
    for m3u in _viz.export_m3u(df, out):
        written[m3u.stem] = m3u

    return written


def run_pipeline(
    config: Config,
    embedder_name: str = "mert",
    method: ClusterMethod = "hdbscan",
    with_labels: bool = True,
    force: bool = False,
    auto_k: bool = True,
    *,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> pd.DataFrame:
    """Run the full pipeline, write the artifact set, and return the assignments.

    Convenience composition of :func:`run_pipeline_core` (compute — see it for
    the argument and DataFrame column contracts) and :func:`write_artifacts`
    (persist under ``config.output_dir``, after ``config.ensure_dirs()``).
    Callers that need the artifact paths, the cluster profiles or the effective
    config (e.g. the auto-picked ``k``) call the two halves themselves.
    """
    config.ensure_dirs()
    result = run_pipeline_core(
        config,
        embedder_name,
        method,
        with_labels,
        force,
        auto_k,
        on_progress=on_progress,
        on_error=on_error,
    )
    write_artifacts(result)
    return result.assignments


def _build_hover_text(df: pd.DataFrame) -> list[str]:
    """Per-point hover strings: filename + top-k moods (as %) + energy/valence."""
    hover: list[str] = []
    for _, row in df.iterrows():
        parts = [str(row["filename"])]
        moods = row.get("mood_top3") or []
        scores = row.get("mood_top3_scores") or []
        if len(moods):
            pieces = [f"{m} {float(s) * 100:.0f}%" for m, s in zip(moods, scores)]
            parts.append("moods: " + ", ".join(pieces))
        energy = row.get("energy")
        valence = row.get("valence")
        if energy is not None and valence is not None and pd.notna(energy) and pd.notna(valence):
            parts.append(f"energy {float(energy):.2f} · valence {float(valence):.2f}")
        hover.append("<br>".join(parts))
    return hover


def write_cluster_report(
    df: pd.DataFrame,
    profiles: dict[int, list[tuple[str, float]]],
    metrics: Mapping[str, Any],
    config: Config,
    method: str,
    out_path=None,
) -> Path:
    """Write a readable Markdown cluster report and return its path.

    Renders a title, the clustering method, overall metrics (n_clusters,
    noise_ratio, silhouette) and a per-cluster section (ascending by id, with the
    noise cluster ``-1`` rendered last and labelled "noise"): size, dominant
    ``cluster_mood``, the ranked ``profiles[cid]``, the mean energy & valence over
    the cluster's rows (when those columns exist) and up to six example filenames.
    Defaults to ``config.output_dir/'cluster_report.md'``. Pure string building
    plus one file write; tolerant of missing optional columns / profiles.
    """
    out = Path(out_path) if out_path is not None else Path(config.output_dir) / "cluster_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    metrics = metrics or {}
    profiles = profiles or {}
    has_df = df is not None and len(df) > 0 and "cluster" in df.columns
    has_energy = has_df and "energy" in df.columns
    has_valence = has_df and "valence" in df.columns

    lines: list[str] = ["# Mood cluster report", ""]
    lines.append(f"- **Method:** {method}")
    n_tracks = int(len(df)) if df is not None else 0
    lines.append(f"- **Tracks:** {n_tracks}")
    lines.append(f"- **Clusters:** {metrics.get('n_clusters', 0)}")
    noise_ratio = metrics.get("noise_ratio")
    if noise_ratio is not None:
        lines.append(f"- **Noise ratio:** {float(noise_ratio):.2%}")
    sil = metrics.get("silhouette")
    lines.append(f"- **Silhouette:** {f'{float(sil):.3f}' if sil is not None else 'n/a'}")
    lines.append("")

    if not has_df:
        lines.append("_No tracks to report._")
        lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    # Cluster ids ascending with the noise cluster (-1) pushed to the end.
    cluster_ids = sorted({int(c) for c in df["cluster"].tolist()})
    cluster_ids = [c for c in cluster_ids if c != -1] + ([-1] if -1 in cluster_ids else [])

    for cid in cluster_ids:
        rows = df[df["cluster"] == cid]
        heading = "noise" if cid == -1 else f"Cluster {cid}"
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(f"- **Size:** {len(rows)}")

        if "cluster_mood" in df.columns and len(rows):
            mood = str(rows["cluster_mood"].iloc[0])
            if mood:
                lines.append(f"- **Dominant mood:** {mood}")

        profs = profiles.get(cid) or profiles.get(int(cid)) or []
        if profs:
            ranked = ", ".join(f"{m} {float(s):.2f}" for m, s in profs)
            lines.append(f"- **Mood profile:** {ranked}")

        if has_energy and len(rows):
            mean_e = float(pd.to_numeric(rows["energy"], errors="coerce").mean())
            if pd.notna(mean_e):
                lines.append(f"- **Mean energy:** {mean_e:.2f}")
        if has_valence and len(rows):
            mean_v = float(pd.to_numeric(rows["valence"], errors="coerce").mean())
            if pd.notna(mean_v):
                lines.append(f"- **Mean valence:** {mean_v:.2f}")

        if "filename" in df.columns:
            examples = rows["filename"].astype(str).tolist()[:6]
            if examples:
                lines.append("- **Examples:**")
                for fn in examples:
                    lines.append(f"  - {fn}")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def compare_spaces(
    config: Config,
    method: ClusterMethod = "kmeans",
    auto_k: bool = True,
    force: bool = False,
) -> dict:
    """Cluster the same tracks in the MERT, CLAP and fused spaces; return metrics.

    Extracts each space's track embeddings for the same discovered files, clusters
    each with ``method`` (applying KMeans auto-k per space when requested), and
    returns ``{space: metrics}`` where ``space`` is one of ``"mert"``, ``"clap"``,
    ``"fused"``. Each value is the ``metrics`` dict from
    :func:`cluster.run_clustering` augmented with ``"silhouette_original"`` (cosine
    silhouette on the ORIGINAL pre-UMAP matrix) and ``"stability_ari"`` (mean
    adjusted Rand index from :func:`cluster.bootstrap_stability`). A space whose
    embeddings are empty is skipped and logged. Print-friendly; writes no files.
    """
    out: dict[str, dict] = {}
    for name in ("mert", "clap", "fused"):
        if name == "fused":
            _files, X = fused_embeddings(config, force=force)
        else:
            _files, X = extract_embeddings(config, name, force=force)
        if X.shape[0] == 0 or X.shape[1] == 0:
            logger.warning("Skipping %s space: no embeddings available.", name)
            continue
        cfg = config
        if method == "kmeans" and auto_k and X.shape[0] >= 3:
            best_k, _ = _cluster.select_kmeans_k(X, cfg)
            cfg = replace(cfg, kmeans_n_clusters=best_k)
        result = _cluster.run_clustering(X, method, cfg)
        metrics = dict(result["metrics"])
        metrics["silhouette_original"] = _cluster.silhouette_original(
            X, np.asarray(result["labels"], dtype=int), metric="cosine"
        )
        stability = _cluster.bootstrap_stability(X, method, cfg)
        metrics["stability_ari"] = stability["mean_ari"]
        out[name] = metrics
    return out


def _clap_embeddings_for(
    config: Config,
    embedder_name: str,
    files: list[Path],
    X: np.ndarray,
    force: bool,
    on_progress: Callable[[int, int, Path], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> tuple[np.ndarray, SupportsEmbedText, np.ndarray]:
    """Return ``(clap_X, clap_embedder, valid_mask)`` aligned to ``files``.

    A single CLAP embedder is constructed and returned so the caller can reuse it
    for text embedding too (avoiding a second model load). When the primary
    embedder was already CLAP, ``X`` is reused as the audio matrix and every row
    is valid (extraction already dropped failures). Otherwise the same files are
    embedded with CLAP; a file whose embedding fails keeps a zero row so the
    matrix stays aligned with ``files``, and its ``valid_mask`` entry is False so
    the caller excludes it from labeling instead of scoring the zero vector.
    ``on_progress`` / ``on_error`` follow the :func:`extract_embeddings` contract.
    """
    clap_embedder = get_embedder("clap", config)
    if embedder_name.lower() == "clap":
        return np.asarray(X, dtype=np.float32), clap_embedder, np.ones(len(files), dtype=bool)

    rows: list[np.ndarray | None] = []  # None marks a failed embed, filled once dim is known
    dim = 0
    total = len(files)
    for done, path in enumerate(files, start=1):
        vec: np.ndarray | None
        try:
            vec = np.asarray(
                track_embedding(clap_embedder, path, config, force=force), dtype=np.float32
            ).reshape(-1)
            dim = max(dim, vec.shape[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("CLAP embedding failed for %s: %s", path, exc)
            if on_error is not None:
                on_error(Path(path), exc)
            vec = None
        rows.append(vec)
        if on_progress is not None:
            on_progress(done, total, Path(path))

    valid = np.array([r is not None for r in rows], dtype=bool)
    if dim == 0:
        return np.empty((len(files), 0), dtype=np.float32), clap_embedder, valid
    filled = [r if r is not None else np.zeros(dim, dtype=np.float32) for r in rows]
    return np.vstack(filled).astype(np.float32), clap_embedder, valid
