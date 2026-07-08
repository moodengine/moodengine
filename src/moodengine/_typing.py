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


class ClusterMetrics(TypedDict):
    """Shape of :func:`moodengine.cluster.cluster_metrics` results.

    ``run_clustering`` additionally stamps ``reduction`` so the tiny-input
    UMAP skip is visible in the result, not only in the logs.
    """

    n_clusters: int
    noise_ratio: float
    cluster_sizes: dict[int, int]
    silhouette: float | None
    reduction: NotRequired[Literal["umap", "none_tiny_input"]]


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
