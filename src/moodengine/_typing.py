"""Public typing vocabulary: ``Literal`` aliases, protocols and result shapes.

Everything here is importable from the package root (``from moodengine import
ClusterMethod``) so downstream type checkers verify calls against the same
closed vocabularies and dict shapes the runtime enforces. Pure typing module:
importing it pulls nothing heavy and executes no computation.
"""

from __future__ import annotations

from typing import Literal, NotRequired, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray

ClusterMethod = Literal["hdbscan", "kmeans", "spherical", "leiden"]
"""Clustering backends accepted by :func:`moodengine.cluster.run_clustering`."""

PoolingMode = Literal["mean", "mean_std"]
"""Frame/segment pooling modes (``Config.pooling_mode``)."""

LayerWeighting = Literal["uniform", "last", "subset", "weighted"]
"""MERT layer-combination modes (``Config.mert_layer_weighting``)."""

SegmentSelection = Literal["head", "uniform"]
"""Which windows survive the per-track cap (``Config.segment_selection``): ``"head"`` keeps the
first N (legacy), ``"uniform"`` spreads N across the whole track so a long track's mood is not
represented by its intro alone."""

ClusterSpace = Literal["reduced", "original"]
"""Which space :func:`moodengine.cluster.run_clustering` clusters in (``Config.cluster_space``).

``"reduced"`` (the default, and what the pipeline has always done) clusters the UMAP layout;
``"original"`` clusters the embeddings themselves. The choice is exposed rather than decided
because measurement does not settle it: the reduced space recovers overlapping blobs better in
some regimes and worse in others. What it does settle is that COSINE is meaningless on a UMAP
layout — see the note on ``cluster_spherical_kmeans``."""

ProjectionMethod = Literal["umap", "densmap", "pacmap"]
"""2-D map projections (``Config.projection_method``)."""


@runtime_checkable
class SupportsEmbedText(Protocol):
    """Anything that maps a batch of text prompts to embedding rows.

    The structural contract behind every ``clap_embedder`` parameter in
    :mod:`moodengine.labeling` and :mod:`moodengine.evaluation`: a single
    batched call returning a ``(len(prompts), d)`` float array. Satisfied by
    :class:`moodengine.embeddings.clap.CLAPEmbedder` and by any test fake.
    """

    def embed_text(self, prompts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Reducer2D(Protocol):
    """A fitted 2-D reducer able to place NEW points into its existing layout.

    The contract behind :func:`moodengine.cluster.transform_projection`:
    ``transform`` maps ``(m, d)`` vectors to ``(m, 2)`` coordinates without
    refitting. Satisfied by fitted UMAP/PaCMAP models and by the identity
    reducer used on tiny inputs.
    """

    def transform(self, X: np.ndarray) -> np.ndarray: ...


#: How much structure the ORIGINAL-space silhouette actually supports (see
#: :func:`moodengine.cluster.structure_verdict`). ``"none_detected"`` means the reported clusters
#: are an artifact of the dimensionality reduction, not of the data.
StructureVerdict = Literal["clustered", "weak", "none_detected"]


class ClusterMetrics(TypedDict):
    """Shape of :func:`moodengine.cluster.cluster_metrics` results.

    ``run_clustering`` additionally stamps ``reduction`` (so the tiny-input UMAP skip is visible in
    the result, not only in the logs) plus the two-space honesty fields ``silhouette_space``,
    ``silhouette_original`` and ``structure``. Those three need to know about BOTH the reduced and
    the original space, which only ``run_clustering`` does — :func:`cluster_metrics` scores one
    given space and cannot set them.
    """

    n_clusters: int
    noise_ratio: float
    cluster_sizes: dict[int, int]
    #: Silhouette in whichever space the clustering ran in — the REDUCED (UMAP) space on the
    #: normal path. Read ``silhouette_original`` before treating it as evidence of structure.
    silhouette: float | None
    reduction: NotRequired[Literal["umap", "none_tiny_input"]]
    #: Which space ``silhouette`` was computed in, so the number is self-describing.
    silhouette_space: NotRequired[Literal["reduced", "original"]]
    #: Silhouette of the same labels in the ORIGINAL embedding space, always COSINE — unlike
    #: ``silhouette``, which cluster_metrics scores with sklearn's euclidean default. ``None``
    #: when undefined (< 2 clusters).
    silhouette_original: NotRequired[float | None]
    #: Verdict derived from ``silhouette_original`` — see :func:`moodengine.cluster.structure_verdict`.
    structure: NotRequired[StructureVerdict | None]


class HDBSCANDetail(TypedDict):
    """Shape of :func:`moodengine.cluster.cluster_hdbscan_detailed` results.

    The per-point diagnostics an HDBSCAN fit already computes and used to discard.
    """

    labels: NDArray[np.int_]
    #: Membership strength of each point in its OWN cluster, 0.0 for noise.
    probabilities: NDArray[np.float32]
    #: GLOSH density-based outlier score per point, or ``None`` under the scikit-learn backend,
    #: which does not expose it.
    glosh: NDArray[np.float32] | None
    #: Which implementation produced the partition — the two agree closely but not exactly.
    backend: Literal["hdbscan", "sklearn", "none_tiny_input"]


class ClusteringResult(TypedDict):
    """Shape of :func:`moodengine.cluster.run_clustering` results."""

    labels: NDArray[np.int_]
    coords2d: NDArray[np.float32]
    metrics: ClusterMetrics
    method: ClusterMethod


class StabilityMetrics(TypedDict):
    """Shape of :func:`moodengine.cluster.bootstrap_stability` results.

    ``mean_ari`` / ``mean_ami`` measure cluster-shape agreement over points that are
    non-noise in both bootstrap runs; ``mean_noise_agreement`` is reported separately so
    two runs that agree only on WHICH points are noise cannot inflate the shape scores.
    """

    mean_ari: float
    std_ari: float
    mean_ami: float
    mean_noise_agreement: float
    n_boot: int


class CoverageEntropyResult(TypedDict):
    """Shape of :func:`moodengine.cluster.coverage_entropy` results."""

    entropy: float
    normalized_entropy: float
    perplexity: float
    n_bins: int
    shares: dict[int, float]


class SubClusterResult(TypedDict):
    """Shape of :func:`moodengine.cluster.sub_cluster` results (indices local to the subset)."""

    sub_labels: NDArray[np.int_]
    sub_k: int
    silhouette: float | None
    medoids: dict[int, int]
    per_cluster_silhouette: dict[int, float | None]
