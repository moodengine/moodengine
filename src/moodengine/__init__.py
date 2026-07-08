"""moodengine — music mood embedding, clustering & zero-shot labeling.

A pure, stateless engine: audio -> embeddings (MERT / CLAP) -> clustering ->
zero-shot mood labels + energy/valence axes -> search & evaluation. The deep
learning backbones are an optional extra (``pip install "moodengine[models]"``);
everything else (clustering, labeling, search, eval, viz on precomputed
embeddings) runs on the lightweight core. Importing this package is torch-free —
the embedders import torch lazily inside :func:`get_embedder`.
"""

from __future__ import annotations

__version__ = "0.1.1"  # x-release-please-version

# --- typing vocabulary (Literal aliases, protocols, result shapes) ---
from moodengine._typing import (
    ClusteringResult,
    ClusterMethod,
    ClusterMetrics,
    CoverageEntropyResult,
    LayerWeighting,
    PoolingMode,
    ProjectionMethod,
    Reducer2D,
    SegmentSelection,
    StabilityMetrics,
    SubClusterResult,
    SupportsEmbedText,
)

# --- configuration ---
from moodengine.config import (
    AUDIO_EXTENSIONS,
    Config,
    default_cache_dir,
    default_config,
    get_device,
)

# --- exception hierarchy (catch MoodengineError for any library-specific failure) ---
from moodengine.exceptions import (
    AudioDecodeError,
    MissingDependencyError,
    ModelLoadError,
    MoodengineError,
)

# --- embedder interface + cache (torch-free; concrete embedders load lazily) ---
from moodengine.embeddings.base import (
    Embedder,
    cache_key,
    file_fingerprint,
    load_cached,
    provenance_cache_key,
    save_cached,
)

# --- audio I/O ---
from moodengine.io_audio import discover_audio_files, load_audio, segment_waveform

# --- pooling ---
from moodengine.pooling import POOLERS, pool_clap, pool_mert

# --- clustering + cluster diagnostics ---
from moodengine.cluster import (
    bootstrap_stability,
    cluster_hdbscan,
    cluster_kmeans,
    cluster_medoids,
    cluster_metrics,
    coverage_entropy,
    outlier_scores,
    per_cluster_silhouette,
    reduce_umap,
    run_clustering,
    select_kmeans_k,
    silhouette_original,
    sub_cluster,
)

# --- zero-shot labeling + attributes ---
from moodengine.labeling import (
    DEFAULT_MOOD_PROMPTS,
    DEFAULT_TEMPERATURE,
    ENERGY_PROMPTS,
    VALENCE_PROMPTS,
    MoodScores,
    l2_normalize,
    attribute_scores,
    build_label_matrix,
    cluster_mood_profiles,
    label_tracks,
    labeling_quality_metrics,
    name_clusters,
    recenter_similarities,
    score_axis,
    score_moods,
    softmax,
    zero_shot_moods,
)

# --- intra-track mood arc (per-segment scoring; torch-free) ---
from moodengine.mood_arc import (
    SegmentArc,
    score_segment_arc,
    segment_bounds,
    segment_embeddings,
)

# --- ambient journey (SLERP geodesic + opt-in optimal transport; torch-free) ---
from moodengine.journey import ot_morph, path_between

# --- search (text->audio, audio->audio) ---
from moodengine.search import (
    find_neighbours,
    find_neighbours_harmonic,
    find_neighbours_mmr,
    find_similar,
    late_interaction_scores,
    near_duplicate_pairs,
    playlist_from_text,
    search_by_text,
    similarity_matrix,
)

# --- novelty / out-of-distribution scoring (torch-free) ---
from moodengine.novelty import knn_distance_scores, mahalanobis_scores

# --- personalization: Tip-Adapter, probe head, metric adapter (inference is torch-free) ---
from moodengine.adapt import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    ProbeHead,
    Projection,
    acquisition_scores,
    apply_projection,
    diverse_subset,
    fit_linear_probe,
    fit_supcon_projection,
    load_probe,
    load_projection,
    predict_probe,
    probe_from_state,
    probe_state,
    projection_from_state,
    projection_state,
    prototype_vector,
    save_probe,
    save_projection,
    tip_adapter_affinities,
)

# --- musical signals: tempo/key (Camelot) + structure segmentation ---
from moodengine.signals import (
    KeyEstimate,
    Segment,
    SignalSet,
    Structure,
    TempoEstimate,
    camelot_neighbors,
    estimate_key,
    estimate_tempo,
    extract_signals,
    segment_structure,
    to_camelot,
)

# --- next-track sequence model (module import is torch-free; training/loading needs torch) ---
from moodengine.sequence import (
    SequenceConfig,
    SequenceModel,
    evaluate_sequence_model,
    load_sequence_model,
    save_sequence_model,
    train_sequence_model,
)

# --- attribution / explanation (exact Shapley by default; TreeSHAP optional) ---
from moodengine.explain import (
    Counterfactual,
    SignalSurrogate,
    SupportsPredictProba,
    counterfactual,
    fit_signal_surrogate,
    shapley_exact,
    surrogate_shap,
)

# --- implicit feedback aggregation ---
from moodengine.feedback import aggregate_implicit, implicit_weight

# --- evaluation ---
from moodengine.evaluation import (
    axis_ranking_auc,
    concordance_correlation_coefficient,
    evaluate_against_gold,
    evaluate_text_queries,
    load_gold,
    retrieval_precision_at_k,
)

# --- calibration (Guo+'17 temperature scaling + baselines) + conformal uncertainty ---
from moodengine.calibration import (
    aps_threshold,
    entropy,
    fit_temperature,
    isotonic_calibrate,
    margin,
    negative_log_likelihood,
    platt_scale,
    prediction_set,
    reliability_diagram,
)

# --- visualization + exports ---
from moodengine.viz import (
    build_dashboard,
    build_labeling_ui,
    export_m3u,
    export_playlists,
    plot_attributes,
    plot_clusters,
)

# --- end-to-end orchestration ---
from moodengine.pipeline import (
    PipelineResult,
    compare_spaces,
    extract_embeddings,
    fused_embeddings,
    get_embedder,
    run_pipeline,
    run_pipeline_core,
    track_embedding,
    write_artifacts,
    write_cluster_report,
)

__all__ = [
    "__version__",
    # typing vocabulary
    "ClusterMethod",
    "PoolingMode",
    "LayerWeighting",
    "SegmentSelection",
    "ProjectionMethod",
    "SupportsEmbedText",
    "Reducer2D",
    "ClusteringResult",
    "ClusterMetrics",
    "StabilityMetrics",
    "CoverageEntropyResult",
    "SubClusterResult",
    # config
    "Config",
    "default_config",
    "default_cache_dir",
    "get_device",
    "AUDIO_EXTENSIONS",
    # exceptions
    "MoodengineError",
    "AudioDecodeError",
    "MissingDependencyError",
    "ModelLoadError",
    # embedders + cache
    "Embedder",
    "get_embedder",
    "cache_key",
    "file_fingerprint",
    "load_cached",
    "save_cached",
    "provenance_cache_key",
    # io
    "load_audio",
    "segment_waveform",
    "discover_audio_files",
    # pooling
    "POOLERS",
    "pool_mert",
    "pool_clap",
    # clustering
    "run_clustering",
    "reduce_umap",
    "cluster_hdbscan",
    "cluster_kmeans",
    "cluster_metrics",
    "select_kmeans_k",
    "silhouette_original",
    "cluster_medoids",
    "outlier_scores",
    "bootstrap_stability",
    "coverage_entropy",
    "per_cluster_silhouette",
    "sub_cluster",
    # labeling
    "label_tracks",
    "attribute_scores",
    "cluster_mood_profiles",
    "name_clusters",
    "zero_shot_moods",
    "build_label_matrix",
    "recenter_similarities",
    "score_axis",
    "score_moods",
    "MoodScores",
    "softmax",
    "labeling_quality_metrics",
    "l2_normalize",
    "DEFAULT_MOOD_PROMPTS",
    "DEFAULT_TEMPERATURE",
    "ENERGY_PROMPTS",
    "VALENCE_PROMPTS",
    # mood arc (per-segment)
    "SegmentArc",
    "score_segment_arc",
    "segment_embeddings",
    "segment_bounds",
    # ambient journey
    "path_between",
    "ot_morph",
    # search
    "similarity_matrix",
    "find_similar",
    "find_neighbours",
    "find_neighbours_mmr",
    "find_neighbours_harmonic",
    "late_interaction_scores",
    "search_by_text",
    "playlist_from_text",
    "near_duplicate_pairs",
    # personalization (Tip-Adapter, probe head, metric adapter)
    "tip_adapter_affinities",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "prototype_vector",
    "acquisition_scores",
    "diverse_subset",
    "ProbeHead",
    "fit_linear_probe",
    "predict_probe",
    "probe_state",
    "probe_from_state",
    "save_probe",
    "load_probe",
    "Projection",
    "fit_supcon_projection",
    "apply_projection",
    "projection_state",
    "projection_from_state",
    "save_projection",
    "load_projection",
    # musical signals (tempo/key/structure)
    "SignalSet",
    "TempoEstimate",
    "KeyEstimate",
    "estimate_tempo",
    "estimate_key",
    "extract_signals",
    "to_camelot",
    "camelot_neighbors",
    "Segment",
    "Structure",
    "segment_structure",
    # sequence model
    "SequenceConfig",
    "SequenceModel",
    "train_sequence_model",
    "evaluate_sequence_model",
    "save_sequence_model",
    "load_sequence_model",
    # explanation / attribution
    "SignalSurrogate",
    "SupportsPredictProba",
    "fit_signal_surrogate",
    "surrogate_shap",
    "shapley_exact",
    "Counterfactual",
    "counterfactual",
    # implicit feedback
    "implicit_weight",
    "aggregate_implicit",
    # novelty / OOD
    "mahalanobis_scores",
    "knn_distance_scores",
    # evaluation
    "axis_ranking_auc",
    "retrieval_precision_at_k",
    "evaluate_text_queries",
    "load_gold",
    "evaluate_against_gold",
    "concordance_correlation_coefficient",
    # calibration + conformal uncertainty
    "fit_temperature",
    "negative_log_likelihood",
    "reliability_diagram",
    "platt_scale",
    "isotonic_calibrate",
    "entropy",
    "margin",
    "aps_threshold",
    "prediction_set",
    # viz
    "plot_clusters",
    "plot_attributes",
    "build_dashboard",
    "build_labeling_ui",
    "export_playlists",
    "export_m3u",
    # pipeline
    "run_pipeline",
    "run_pipeline_core",
    "write_artifacts",
    "PipelineResult",
    "extract_embeddings",
    "track_embedding",
    "fused_embeddings",
    "compare_spaces",
    "write_cluster_report",
]
