"""Central configuration: paths, audio params, model + clustering hyper-parameters.

This module must import cleanly *without* torch installed (device detection
degrades gracefully to CPU), so that the lightweight pipeline stages can run
on machines that don't have the deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

from platformdirs import user_cache_path

from moodengine._typing import (
    ClusterSpace,
    HDBSCANSelection,
    LayerWeighting,
    PoolingMode,
    ProjectionMethod,
    SegmentSelection,
)

# Audio file extensions discovered when scanning an input directory.
AUDIO_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")

# Samples per segment above which laion-clap's fusion truncation starts drawing which chunks to
# keep from the GLOBAL, unseeded numpy RNG (its `training/data.py` `len > 480000` branch).
# moodengine seeds only its own Generator instances, so past this length every CLAP track vector
# depends on ambient process state, is persisted under a cache key that cannot encode the draw, and
# differs on a `force=True` recompute. The 10 s default lands on exactly 480000 samples at 48 kHz
# and so takes the deterministic branch — safe by arithmetic coincidence, not by design, which is
# why it is enforced rather than merely documented. Enforced by `ensure_clap_fusion_supported`
# below rather than in `Config.__post_init__`, which cannot know whether CLAP will be used at all.
CLAP_FUSION_SAMPLE_LIMIT: int = 480_000


def ensure_clap_fusion_supported(config: "Config") -> None:
    """Raise ``ValueError`` when this config would push CLAP past :data:`CLAP_FUSION_SAMPLE_LIMIT`.

    Pure arithmetic on two config fields — no torch, no weights — which is what lets the check run
    at both entry points that matter. ``CLAPEmbedder.__init__`` calls it for anyone constructing
    the embedder directly, and the pipeline's lazy proxy calls it at CONSTRUCTION, before any
    per-file loop that skips-and-continues could turn a refusal into a warning, and regardless of
    how much of the run is served from cache. A configuration error must not depend on cache state.

    No-op for every other embedder: a MERT-only run at 15 s segments is perfectly valid, which is
    exactly why this cannot live in ``Config.__post_init__``.
    """
    samples = config.segment_seconds * config.clap_sample_rate
    if samples > CLAP_FUSION_SAMPLE_LIMIT:
        raise ValueError(
            f"segment_seconds={config.segment_seconds} at clap_sample_rate="
            f"{config.clap_sample_rate} gives {samples:.0f} samples per segment, above the "
            f"{CLAP_FUSION_SAMPLE_LIMIT} at which laion-clap truncates by drawing chunks from "
            "the unseeded global numpy RNG. Embeddings past that point are not reproducible "
            "and would be cached as if they were, so this refuses rather than producing them. "
            f"Use segment_seconds <= {CLAP_FUSION_SAMPLE_LIMIT / config.clap_sample_rate:g} "
            "for CLAP (MERT has no such limit)."
        )


def default_cache_dir() -> Path:
    """Per-user, OS-appropriate cache root for computed embeddings.

    A cache must survive environment rebuilds and must never live inside the
    installed package (site-packages is recreated on every sync), so the
    default is the platform cache location (``%LOCALAPPDATA%`` on Windows,
    ``~/Library/Caches`` on macOS, ``$XDG_CACHE_HOME`` on Linux). Applications
    embedding this library should pass their own ``cache_dir`` instead.
    """
    # appauthor=False: without it Windows nests <author>/<app> and the author
    # defaults to the app name, yielding a doubled moodengine/moodengine path.
    return user_cache_path("moodengine", appauthor=False)


def get_device() -> str:
    """Return the best available torch device name.

    Order of preference: CUDA > Apple MPS > CPU. Returns ``"cpu"`` when torch
    is not installed, so importing this module never requires torch.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise ``ValueError`` naming the field, the received value and the valid options."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")


@dataclass(frozen=True)
class Config:
    """Immutable run configuration. Construct with :func:`default_config` and
    override fields via :meth:`dataclasses.replace`."""

    # --- paths ---
    # Inputs and outputs are workspace-scoped: the defaults are relative to the
    # current working directory and only suit quick experiments — real callers
    # pass explicit absolute paths. The cache is machine-scoped (see
    # :func:`default_cache_dir`) so recomputed embeddings survive venv rebuilds.
    raw_dir: Path = Path("data/raw")
    cache_dir: Path = field(default_factory=default_cache_dir)
    output_dir: Path = Path("outputs")

    # --- audio I/O ---
    # MERT-v1 is a 24 kHz model (its feature extractor declares 24 kHz); decoding at any other rate
    # feeds it time/pitch-warped audio and silently degrades every embedding, so this must match the
    # model's rate. The embedder validates the decode rate against its extractor and raises on a
    # mismatch rather than relabeling it.
    mert_sample_rate: int = 24_000
    clap_sample_rate: int = 48_000
    segment_seconds: float = 10.0
    overlap_seconds: float = 0.0
    min_segment_seconds: float = 1.0
    # Cap segments per track to bound compute on long files (0 = no cap).
    max_segments_per_track: int = 12
    # When the cap bites, which windows survive: "uniform" spreads them across the whole track
    # (a track's intro is systematically unrepresentative of its overall mood); "head" keeps the
    # first N (the pre-1.0 behavior). Below the cap the two are identical.
    segment_selection: SegmentSelection = "uniform"
    audio_extensions: tuple[str, ...] = AUDIO_EXTENSIONS

    # --- models ---
    mert_model_name: str = "m-a-p/MERT-v1-95M"
    # Hub revision (branch/tag/SHA) for mert_model_name. None pins the default
    # model to a reviewed snapshot (MERT runs remote code — see embeddings.mert);
    # custom models fall back to the hub's latest unless a revision is given here.
    mert_revision: str | None = None
    # MuQ-MuLan: the alternative audio-text backbone (embedder name "mulan"). 24 kHz like MERT,
    # NOT CLAP's 48 kHz. Weights are CC-BY-NC-4.0 — see embeddings/mulan.py.
    mulan_model_name: str = "OpenMuQ/MuQ-MuLan-large"
    # Hub revision (branch/tag/SHA) for mulan_model_name, mirroring mert_revision. None takes the
    # hub's latest, which means the weights behind a cached vector are not pinned; set it to a SHA
    # for a reproducible run. Whatever it resolves to also enters the embedding cache key, so a
    # revision change mints new keys instead of serving another snapshot's vectors.
    mulan_revision: str | None = None
    mulan_sample_rate: int = 24_000
    clap_enable_fusion: bool = False
    clap_amodel: str = "HTSAT-base"
    # None -> let laion-clap download its default (music) checkpoint.
    clap_checkpoint: str | None = None

    # --- pooling (frame/segment level -> track level) ---
    pooling_mode: PoolingMode = "mean_std"
    # "subset" = mean of a mid-layer band; "weighted" = softmax(mert_layer_weights) mix.
    mert_layer_weighting: LayerWeighting = "uniform"
    mert_layers: tuple[int, ...] | None = (
        None  # explicit indices for "subset"; None -> middle third
    )
    mert_layer_weights: tuple[float, ...] | None = None  # per-layer logits for "weighted"

    # --- embedding fusion (MERT + CLAP) ---
    fusion_weights: tuple[float, float] = (
        0.5,
        0.5,
    )  # (mert, clap) block weights when clustering "fused"

    # --- labeling calibration ---
    # Subtract the per-mood mean cosine before softmax to cancel CLAP's modality-gap / per-prompt
    # prior. The pipeline estimates that offset ONCE over the whole library and returns it as
    # PipelineResult.mood_prior; pass it back as `prior=` when scoring anything later, or the
    # labeling functions fall back to the batch's own means (n >= 5 only) and a track's label then
    # depends on which tracks were scored with it.
    recenter_labels: bool = True

    # --- clustering ---
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components_cluster: int = 10  # dims of the space we cluster in
    umap_n_components_viz: int = 2  # dims for the 2D scatter
    umap_metric: str = "cosine"
    projection_method: ProjectionMethod = "umap"  # 2-D map projection
    # Which space the clustering itself runs in. "reduced" is what the pipeline has always done;
    # "original" skips the reduction for the clustering step (the 2-D map is still produced).
    cluster_space: ClusterSpace = "reduced"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    # The knobs that address HDBSCAN's known failure on a homogeneous library — one dominant
    # cluster swallowing most of the catalogue. Both backends accept all four; leaving them
    # unexposed meant the only remedy was changing min_cluster_size and hoping.
    #   "eom"  — excess of mass, the default: prefers few large, stable clusters
    #   "leaf" — the finest stable clusters, which is what splits that dominant blob
    hdbscan_cluster_selection_method: HDBSCANSelection = "eom"
    hdbscan_cluster_selection_epsilon: float = 0.0  # merge clusters closer than this distance
    hdbscan_max_cluster_size: int | None = None  # cap a runaway cluster (standalone backend only)
    hdbscan_allow_single_cluster: bool = False  # let a genuinely uniform library say so
    kmeans_n_clusters: int = 8
    # Leiden community detection — kNN graph degree + RBConfiguration resolution. Optional
    # backend (leidenalg + python-igraph); these knobs are inert unless method == "leiden".
    leiden_k_neighbors: int = 15
    leiden_resolution: float = 1.0

    # --- evaluation ---
    bootstrap_n: int = 50  # resamples for cluster stability (ARI / AMI)

    # --- misc ---
    device: str = field(default_factory=get_device)
    seed: int = 42

    def __post_init__(self) -> None:
        """Fail fast on invalid values, at the single point every config passes through.

        Without this, a typo like ``pooling_mode="mean-std"`` only surfaces after
        models are loaded and audio is decoded — and the per-file error handling
        then turns it into "every file skipped", i.e. a silently empty result.
        ``dataclasses.replace()`` re-runs ``__init__``, so derived configs are
        re-validated too. Only values with a silent-failure mode are checked;
        anything else stays the caller's responsibility.
        """
        # Vocabularies come from the exported Literal aliases, so the runtime
        # check and the type checker can never drift apart.
        _check_choice("pooling_mode", self.pooling_mode, get_args(PoolingMode))
        _check_choice("mert_layer_weighting", self.mert_layer_weighting, get_args(LayerWeighting))
        _check_choice("projection_method", self.projection_method, get_args(ProjectionMethod))
        _check_choice("segment_selection", self.segment_selection, get_args(SegmentSelection))
        _check_choice("cluster_space", self.cluster_space, get_args(ClusterSpace))
        _check_choice(
            "hdbscan_cluster_selection_method",
            self.hdbscan_cluster_selection_method,
            get_args(HDBSCANSelection),
        )

        if self.segment_seconds <= 0:
            raise ValueError(f"segment_seconds must be > 0; got {self.segment_seconds}")
        if self.min_segment_seconds <= 0:
            raise ValueError(f"min_segment_seconds must be > 0; got {self.min_segment_seconds}")
        if not 0 <= self.overlap_seconds < self.segment_seconds:
            raise ValueError(
                f"overlap_seconds must be in [0, segment_seconds); got "
                f"overlap_seconds={self.overlap_seconds} with segment_seconds={self.segment_seconds}"
            )
        if self.max_segments_per_track < 0:
            raise ValueError(
                f"max_segments_per_track must be >= 0 (0 = no cap); "
                f"got {self.max_segments_per_track}"
            )
        # NOTE: the CLAP fusion-truncation limit on `segment_seconds * clap_sample_rate` is NOT
        # checked here. It is a property of laion-clap, not of the configuration, and a MERT-only
        # run at 15 s segments is perfectly valid — raising here would block it for a reason that
        # never applies to it. `CLAPEmbedder.__init__` enforces it instead, before any audio is
        # embedded. See `_CLAP_FUSION_SAMPLE_LIMIT`.

        if len(self.fusion_weights) != 2:
            raise ValueError(
                f"fusion_weights must be a (mert, clap) pair; got {self.fusion_weights!r}"
            )
        if any(w < 0 for w in self.fusion_weights) or not any(self.fusion_weights):
            raise ValueError(
                f"fusion_weights must be >= 0 and not all zero (a zero pair would silently "
                f"produce an all-zero fused space); got {self.fusion_weights!r}"
            )

        if self.umap_n_neighbors < 2:
            raise ValueError(f"umap_n_neighbors must be >= 2; got {self.umap_n_neighbors}")
        if self.kmeans_n_clusters < 1:
            raise ValueError(f"kmeans_n_clusters must be >= 1; got {self.kmeans_n_clusters}")
        if self.leiden_resolution <= 0:
            raise ValueError(f"leiden_resolution must be > 0; got {self.leiden_resolution}")
        if self.hdbscan_cluster_selection_epsilon < 0:
            raise ValueError(
                "hdbscan_cluster_selection_epsilon must be >= 0; got "
                f"{self.hdbscan_cluster_selection_epsilon}"
            )
        if self.hdbscan_max_cluster_size is not None and self.hdbscan_max_cluster_size < 2:
            raise ValueError(
                f"hdbscan_max_cluster_size must be >= 2 or None; got {self.hdbscan_max_cluster_size}"
            )
        if self.bootstrap_n < 0:
            raise ValueError(f"bootstrap_n must be >= 0; got {self.bootstrap_n}")

    def ensure_dirs(self) -> None:
        """Create the data/cache/output directories if missing (side effect)."""
        for d in (self.raw_dir, self.cache_dir, self.output_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


def default_config() -> Config:
    """Return a :class:`Config` with all defaults (device auto-detected)."""
    return Config()
