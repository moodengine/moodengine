"""Tests for moodengine._typing — the exported typing vocabulary.

These are drift guards: every Literal alias must list exactly what the runtime
accepts, and every TypedDict must mirror the keys the functions actually
return. If a vocabulary or a result shape changes without updating the export,
one of these fails.
"""

from dataclasses import replace
from typing import get_args

import numpy as np
import pytest
from assertpy import assert_that

import moodengine
from moodengine import (
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
    default_config,
)
from moodengine.cluster import (
    _IdentityReducer,
    bootstrap_stability,
    cluster_metrics,
    coverage_entropy,
    run_clustering,
    sub_cluster,
)

# ---------------------------------------------------------------------------
# Literal aliases <-> runtime vocabularies
# ---------------------------------------------------------------------------


def test_cluster_method_alias_matches_run_clustering_vocabulary() -> None:
    X = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="method must be") as err:
        run_clustering(X, "bogus", default_config())  # type: ignore[arg-type]

    for member in get_args(ClusterMethod):
        assert_that(str(err.value)).contains(f"'{member}'")


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("pooling_mode", PoolingMode),
        ("mert_layer_weighting", LayerWeighting),
        ("projection_method", ProjectionMethod),
        ("segment_selection", SegmentSelection),
    ],
)
def test_config_literal_fields_match_their_runtime_checks(field: str, alias: object) -> None:
    members = get_args(alias)

    # Every alias member is accepted...
    for value in members:
        replace(default_config(), **{field: value})

    # ...and the rejection message lists the alias vocabulary exactly.
    with pytest.raises(ValueError, match=field) as err:
        replace(default_config(), **{field: "bogus"})
    for member in members:
        assert_that(str(err.value)).contains(f"'{member}'")


# ---------------------------------------------------------------------------
# TypedDicts <-> actual result keys
# ---------------------------------------------------------------------------


def test_clustering_result_keys_match_the_typeddict() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((12, 6)).astype(np.float32)

    result = run_clustering(X, "kmeans", replace(default_config(), kmeans_n_clusters=2))

    assert_that(set(result)).is_equal_to(set(ClusteringResult.__annotations__))
    assert_that(set(result["metrics"])).is_equal_to(set(ClusterMetrics.__annotations__))


def test_cluster_metrics_keys_match_the_typeddict_minus_the_run_only_key() -> None:
    X = np.eye(4, dtype=np.float32)

    metrics = cluster_metrics(X, np.array([0, 0, 1, 1]))

    # ``reduction`` is NotRequired: stamped by run_clustering, absent here.
    assert_that(set(metrics)).is_equal_to(set(ClusterMetrics.__annotations__) - {"reduction"})


def test_coverage_entropy_keys_match_the_typeddict() -> None:
    result = coverage_entropy(np.array([0, 0, 1, -1]))

    assert_that(set(result)).is_equal_to(set(CoverageEntropyResult.__annotations__))


def test_stability_metrics_keys_match_the_typeddict() -> None:
    result = bootstrap_stability(np.eye(2, dtype=np.float32), "kmeans", default_config())

    assert_that(set(result)).is_equal_to(set(StabilityMetrics.__annotations__))


def test_sub_cluster_keys_match_the_typeddict() -> None:
    result = sub_cluster(np.eye(2, dtype=np.float32), default_config())

    assert_that(set(result)).is_equal_to(set(SubClusterResult.__annotations__))


# ---------------------------------------------------------------------------
# Protocols (runtime_checkable) + package-root export
# ---------------------------------------------------------------------------


def test_fake_embedder_satisfies_supports_embed_text(fake_clap) -> None:
    assert_that(isinstance(fake_clap, SupportsEmbedText)).is_true()


def test_identity_reducer_satisfies_reducer2d() -> None:
    assert_that(isinstance(_IdentityReducer(default_config()), Reducer2D)).is_true()


def test_typing_vocabulary_is_exported_from_the_package_root() -> None:
    for name in (
        "ClusterMethod",
        "PoolingMode",
        "LayerWeighting",
        "ProjectionMethod",
        "SupportsEmbedText",
        "Reducer2D",
        "ClusteringResult",
        "ClusterMetrics",
        "StabilityMetrics",
        "CoverageEntropyResult",
        "SubClusterResult",
    ):
        assert_that(moodengine.__all__).contains(name)
        assert_that(hasattr(moodengine, name)).is_true()
