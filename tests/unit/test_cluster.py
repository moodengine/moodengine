"""Tests for :mod:`moodengine.cluster` — UMAP reduction + HDBSCAN/KMeans clustering."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
from assertpy import assert_that

import moodengine.cluster
from moodengine._math import l2_normalize
from moodengine.cluster import (
    bootstrap_stability,
    cluster_hdbscan,
    cluster_hdbscan_detailed,
    cluster_hierarchy,
    cluster_kmeans,
    cluster_leiden,
    cluster_medoids,
    cluster_metrics,
    coverage_entropy,
    density_validity_index,
    cluster_spherical_kmeans,
    graph_modularity,
    outlier_scores,
    per_cluster_silhouette,
    reduce_umap,
    run_clustering,
    select_kmeans_k,
    silhouette_original,
    structure_verdict,
    sub_cluster,
)
from moodengine.config import default_config


def _three_blobs(seed: int = 0, per: int = 12, dim: int = 8):
    """Three well-separated isotropic gaussian blobs -> (X, true_labels)."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[10.0] + [0.0] * (dim - 1), [-10.0] + [0.0] * (dim - 1), [0.0] * (dim - 1) + [10.0]],
        dtype=np.float32,
    )
    blocks, truth = [], []
    for k, c in enumerate(centers):
        blocks.append(c + 0.25 * rng.standard_normal((per, dim)).astype(np.float32))
        truth.extend([k] * per)
    return np.concatenate(blocks, axis=0).astype(np.float32), np.array(truth)


def _n_blobs(n_blobs: int, seed: int = 0, per: int = 12, dim: int = 8):
    """``n_blobs`` well-separated isotropic gaussian blobs -> (X, true_labels)."""
    rng = np.random.default_rng(seed)
    blocks, truth = [], []
    for k in range(n_blobs):
        center = np.zeros(dim, dtype=np.float32)
        center[k % dim] = 12.0 * (1 + k // dim)  # distinct, far-apart centers
        blocks.append(center + 0.20 * rng.standard_normal((per, dim)).astype(np.float32))
        truth.extend([k] * per)
    return np.concatenate(blocks, axis=0).astype(np.float32), np.array(truth)


def test_cluster_kmeans_recovers_three_blobs() -> None:
    """KMeans(3) on three separated blobs recovers three high-quality clusters."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    cfg = default_config()
    X, truth = _three_blobs(seed=1)
    labels = cluster_kmeans(X, n_clusters=3, config=cfg)
    assert_that(labels.shape).is_equal_to((X.shape[0],))
    assert_that(len(set(labels.tolist()))).is_equal_to(3)
    assert_that(adjusted_rand_score(truth, labels)).is_greater_than(0.9)
    assert_that(silhouette_score(X, labels)).is_greater_than(0.7)


def test_cluster_kmeans_deterministic_with_seed() -> None:
    """Same seed -> identical KMeans labels."""
    cfg = default_config()
    X, _ = _three_blobs(seed=2)
    a = cluster_kmeans(X, 3, cfg)
    b = cluster_kmeans(X, 3, cfg)
    np.testing.assert_array_equal(a, b)


def test_cluster_hdbscan_finds_multiple_clusters() -> None:
    """HDBSCAN on separated blobs finds at least two clusters."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X, _ = _three_blobs(seed=3, per=12)
    labels = cluster_hdbscan(X, cfg)
    assert_that(labels.shape).is_equal_to((X.shape[0],))
    n_clusters = len({lbl for lbl in labels.tolist() if lbl != -1})
    assert_that(n_clusters).is_greater_than_or_equal_to(2)


def test_cluster_hdbscan_tiny_input_all_noise() -> None:
    """A single sample cannot be clustered -> labelled noise without crashing."""
    cfg = default_config()
    labels = cluster_hdbscan(np.zeros((1, 4), dtype=np.float32), cfg)
    assert_that(labels.tolist()).is_equal_to([-1])


def test_cluster_metrics_keys_and_counts() -> None:
    """Metrics report cluster count (excl. noise), sizes, ratio and silhouette."""
    from sklearn.metrics import silhouette_score

    X, _ = _three_blobs(seed=4, per=6)
    # Hand-built labels: two real clusters + six noise points (the third blob).
    labels = np.array([0] * 6 + [1] * 6 + [-1] * 6)
    m = cluster_metrics(X, labels)
    assert_that(set(m.keys())).is_equal_to(
        {"n_clusters", "noise_ratio", "cluster_sizes", "silhouette"}
    )
    assert_that(m["n_clusters"]).is_equal_to(2)
    assert_that(m["cluster_sizes"]).is_equal_to({-1: 6, 0: 6, 1: 6})
    assert_that(m["noise_ratio"]).is_close_to(6 / 18, tolerance=1e-6)
    assert_that(m["silhouette"]).is_not_none()
    assert_that(m["silhouette"]).is_between(-1.0, 1.0)
    # Spec: silhouette is computed on NON-NOISE points only. The noise points
    # here form a third tight blob, so including them (e.g. treating -1 as a real
    # cluster) yields a *different* value -- pin the noise-excluded result so a
    # noise-inclusion bug is caught, not silently accepted.
    mask = labels != -1
    expected = silhouette_score(X[mask], labels[mask])
    assert_that(m["silhouette"]).is_close_to(expected, tolerance=1e-6)
    assert_that(m["silhouette"]).is_not_close_to(silhouette_score(X, labels), tolerance=1e-6)


def test_cluster_metrics_single_cluster_no_silhouette() -> None:
    """With a single cluster the silhouette is undefined -> None (no crash)."""
    X, _ = _three_blobs(seed=5, per=4)
    labels = np.zeros(X.shape[0], dtype=int)
    m = cluster_metrics(X, labels)
    assert_that(m["n_clusters"]).is_equal_to(1)
    assert_that(m["silhouette"]).is_none()
    assert_that(m["noise_ratio"]).is_equal_to(0.0)


def test_cluster_metrics_all_noise() -> None:
    """All-noise input is summarized gracefully (0 clusters, ratio 1.0)."""
    X = np.zeros((5, 3), dtype=np.float32)
    labels = np.full(5, -1, dtype=int)
    m = cluster_metrics(X, labels)
    assert_that(m["n_clusters"]).is_equal_to(0)
    assert_that(m["noise_ratio"]).is_equal_to(1.0)
    assert_that(m["silhouette"]).is_none()
    assert_that(m["cluster_sizes"]).is_equal_to({-1: 5})


def test_reduce_umap_shape() -> None:
    """UMAP reduces (n, d) to (n, n_components) and returns the fitted reducer."""
    cfg = default_config()
    X, _ = _three_blobs(seed=6, per=12)  # 36 samples > n_neighbors
    emb, reducer = reduce_umap(X, n_components=2, config=cfg)
    assert_that(emb.shape).is_equal_to((X.shape[0], 2))
    assert_that(emb.dtype).is_equal_to(np.float32)
    assert_that(hasattr(reducer, "transform")).is_true()


def _two_elongated_clusters(seed: int = 0, per: int = 80):
    """Two long, thin, parallel bands — dense but nowhere near globular. This is the shape HDBSCAN
    is chosen for and the shape a centroid-distance outlier score cannot describe."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, per)
    lower = np.stack([t * 10.0, np.zeros_like(t)], axis=1) + rng.normal(0, 0.15, (per, 2))
    upper = np.stack([t * 10.0, np.full_like(t, 4.0)], axis=1) + rng.normal(0, 0.15, (per, 2))
    return np.vstack([lower, upper]).astype(np.float32)


def test_cluster_hdbscan_detailed_agrees_with_the_labels_only_call() -> None:
    """Same seam, same parameters: the detailed call must not be a second, subtly different fit."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X, _ = _three_blobs(seed=20, per=12)

    detail = cluster_hdbscan_detailed(X, cfg)

    np.testing.assert_array_equal(detail["labels"], cluster_hdbscan(X, cfg))


def test_cluster_hdbscan_detailed_reports_membership_strength() -> None:
    """``probabilities`` is the per-point confidence the fit already computed and used to discard —
    a bare label cannot say whether a track sits at its cluster's core or on its fringe."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X = _two_elongated_clusters()

    detail = cluster_hdbscan_detailed(X, cfg)

    assert_that(detail["probabilities"].shape).is_equal_to((X.shape[0],))
    assert_that(float(detail["probabilities"].min())).is_greater_than_or_equal_to(0.0)
    assert_that(float(detail["probabilities"].max())).is_less_than_or_equal_to(1.0)
    # Noise, where present, carries no membership at all.
    noise = detail["labels"] == -1
    if bool(noise.any()):
        np.testing.assert_allclose(detail["probabilities"][noise], 0.0, atol=1e-6)


def test_glosh_and_centroid_outlier_scores_disagree_on_elongated_clusters() -> None:
    """The substantive reason to surface GLOSH. ``outlier_scores`` measures distance to a centroid,
    which presumes globular clusters — precisely the assumption HDBSCAN exists to avoid. On two
    long thin bands the two rankings share little: Spearman rho ~0.24, so they are not
    interchangeable and the density-based one is the honest signal for a density partition."""
    from scipy.stats import spearmanr

    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X = _two_elongated_clusters()

    detail = cluster_hdbscan_detailed(X, cfg)

    if detail["glosh"] is None:
        pytest.skip("GLOSH needs the standalone hdbscan backend")
    clustered = detail["labels"] != -1
    rho = float(
        spearmanr(
            detail["glosh"][clustered], outlier_scores(X, detail["labels"])[clustered]
        ).statistic
    )
    assert_that(abs(rho)).is_less_than(0.6)  # measuring different things, not a reimplementation


def test_cluster_hdbscan_detailed_tiny_input_is_all_noise() -> None:
    """One sample cannot form a cluster of size >= 2, so it is noise by definition — reported
    without touching a backend, and with no fabricated diagnostics."""
    detail = cluster_hdbscan_detailed(np.zeros((1, 4), dtype=np.float32), default_config())

    assert_that(detail["labels"].tolist()).is_equal_to([-1])
    assert_that(detail["probabilities"].tolist()).is_equal_to([0.0])
    assert_that(detail["glosh"]).is_none()
    assert_that(detail["backend"]).is_equal_to("none_tiny_input")


def test_density_validity_index_ranks_real_density_above_noise() -> None:
    """DBCV is the index silhouette cannot be for density clusters. Two genuinely dense bands must
    score well above a structureless cloud cut into the same number of pieces."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X = _two_elongated_clusters()
    labels = cluster_hdbscan(X, cfg)
    noise = np.random.default_rng(1).standard_normal(X.shape).astype(np.float32)

    real = density_validity_index(X, labels)
    fake = density_validity_index(noise, labels)

    if real is None or fake is None:
        pytest.skip("DBCV needs the standalone hdbscan package")
    assert_that(real).is_greater_than(fake)
    assert_that(real).is_between(-1.0, 1.0)


def test_density_validity_index_absent_when_it_cannot_be_measured() -> None:
    """Fewer than two clusters leaves nothing to score the separation between — an absence, never
    a fabricated 0.0."""
    X = np.eye(6, dtype=np.float32)

    assert_that(density_validity_index(X, np.zeros(6, dtype=int))).is_none()
    assert_that(density_validity_index(X, np.full(6, -1, dtype=int))).is_none()


def test_density_validity_index_sampling_is_reproducible() -> None:
    """The pairwise pass is O(m²) per cluster (~512 MB at m=8000), so it is sampleable — and the
    sample must be seeded, or the reported validity would drift between identical runs."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X = _two_elongated_clusters()
    labels = cluster_hdbscan(X, cfg)

    first = density_validity_index(X, labels, sample_size=60, random_state=7)
    second = density_validity_index(X, labels, sample_size=60, random_state=7)

    if first is None:
        pytest.skip("DBCV needs the standalone hdbscan package")
    assert_that(first).is_equal_to(second)


def test_run_clustering_tiny_input_does_not_crash() -> None:
    """A handful of samples (< n_neighbors) skips UMAP yet returns full results —
    and the switch is visible in the result, not just in the docstring."""
    cfg = default_config()
    X, _ = _three_blobs(seed=7, per=3)  # 9 samples < max(15, 4)
    out = run_clustering(X, method="kmeans", config=cfg)
    n = X.shape[0]
    assert_that(set(out.keys())).is_equal_to({"labels", "coords2d", "metrics", "method"})
    assert_that(out["labels"].shape).is_equal_to((n,))
    assert_that(out["coords2d"].shape).is_equal_to((n, 2))
    assert_that(out["method"]).is_equal_to("kmeans")
    assert_that(out["metrics"]).is_instance_of(dict)
    assert_that(out["metrics"]["reduction"]).is_equal_to("none_tiny_input")


def test_run_clustering_hdbscan_full_path() -> None:
    """With enough samples UMAP runs and HDBSCAN yields aligned shapes."""
    cfg = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)
    X, _ = _three_blobs(seed=8, per=12)  # 36 samples -> real UMAP path
    out = run_clustering(X, method="hdbscan", config=cfg)
    n = X.shape[0]
    assert_that(out["labels"].shape).is_equal_to((n,))
    assert_that(out["coords2d"].shape).is_equal_to((n, 2))
    assert_that(out["method"]).is_equal_to("hdbscan")
    assert_that(out["metrics"]["reduction"]).is_equal_to("umap")


def test_run_clustering_reports_structure_none_on_pure_noise() -> None:
    """The defect this guards: metrics were scored inside the UMAP layout, which is fit to separate
    exactly these points — so structureless input came back certified as many well-separated
    clusters. On 600 rows of isotropic noise the reduced silhouette reads ≈0.27 while the original
    space reads ≈0.01. The verdict must follow the ORIGINAL space."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 64)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # no structure by construction

    metrics = run_clustering(X, method="hdbscan", config=default_config())["metrics"]

    assert_that(metrics["structure"]).is_equal_to("none_detected")
    assert_that(metrics["silhouette_space"]).is_equal_to("reduced")
    assert_that(metrics["silhouette_original"]).is_less_than(0.05)
    # The reduced score is still reported — and is still much higher. That gap IS the finding.
    assert_that(metrics["silhouette"]).is_greater_than(metrics["silhouette_original"])


def test_run_clustering_reports_structure_clustered_on_real_blobs() -> None:
    """The gate must not cry wolf: genuinely separated blobs read as clustered.

    ``kmeans_n_clusters`` is pinned to the true blob count. Leaving it at the default 8 splits
    three blobs eight ways, and the resulting original-space silhouette lands at 0.251 against
    this band's 0.25 edge — a margin thinner than the float differences between platforms, which
    is a test of the fixture's arithmetic rather than of the gate. At the true k it is 0.995.
    """
    X, _ = _three_blobs(seed=8, per=40)
    cfg = dataclasses.replace(default_config(), kmeans_n_clusters=3)

    metrics = run_clustering(X, method="kmeans", config=cfg)["metrics"]

    assert_that(metrics["structure"]).is_equal_to("clustered")
    assert_that(metrics["silhouette_original"]).is_greater_than(0.9)


def test_run_clustering_tiny_input_marks_silhouette_space_original() -> None:
    """When UMAP is skipped the two spaces ARE the same one — but the honesty fields must still be
    populated rather than left absent, and the cosine score must be recomputed rather than aliased
    to ``silhouette`` (which cluster_metrics produces with sklearn's euclidean default, a different
    quantity from the one structure_verdict's bands were calibrated on)."""
    X, _ = _three_blobs(seed=7, per=3)  # 9 samples -> UMAP skipped

    metrics = run_clustering(X, method="kmeans", config=default_config())["metrics"]

    assert_that(metrics["silhouette_space"]).is_equal_to("original")
    # Not equal to metrics["silhouette"]: that one is euclidean (cluster_metrics' default) while
    # the verdict's bands are calibrated on cosine, so it is recomputed rather than reused.
    assert_that(metrics["silhouette_original"]).is_not_none()
    assert_that(metrics["structure"]).is_not_none()  # a verdict is still reached on this path


@pytest.mark.parametrize(
    ("silhouette", "expected"),
    [
        (0.0, "none_detected"),
        (0.05, "none_detected"),  # inclusive upper edge
        (0.051, "weak"),
        (0.25, "weak"),  # inclusive upper edge
        (0.26, "clustered"),
        (None, None),  # < 2 clusters: an absence, never a fabricated verdict
        (float("nan"), None),
    ],
)
def test_structure_verdict_maps_a_silhouette_to_its_band(silhouette, expected) -> None:
    assert_that(structure_verdict(silhouette)).is_equal_to(expected)


def test_silhouette_original_sampling_is_seeded_and_deterministic() -> None:
    """The O(n²·d) pass is capped by sampling on the run_clustering path; the estimate must stay
    reproducible, or the reported structure verdict would drift between identical runs."""
    X, truth = _three_blobs(seed=3, per=60)

    first = silhouette_original(X, truth, sample_size=50, random_state=42)
    second = silhouette_original(X, truth, sample_size=50, random_state=42)

    assert_that(first).is_not_none()
    assert_that(first).is_equal_to(second)


def test_cluster_space_original_skips_the_reduction_for_clustering_only() -> None:
    """`cluster_space` picks where the CLUSTERING runs; the 2-D map is always a UMAP fit, because
    it exists to be looked at. The default is unchanged, so no existing result moves."""
    X, _ = _three_blobs(seed=30, per=20)
    cfg = dataclasses.replace(default_config(), cluster_space="original", kmeans_n_clusters=3)

    out = run_clustering(X, method="kmeans", config=cfg)

    assert_that(out["metrics"]["silhouette_space"]).is_equal_to("original")
    assert_that(out["metrics"]["reduction"]).is_equal_to("umap")  # the map still gets one
    assert_that(out["coords2d"].shape).is_equal_to((X.shape[0], 2))


def test_spherical_kmeans_warns_on_a_degenerate_cosine_space(caplog) -> None:
    """A UMAP layout's coordinates sit far from an arbitrary origin, so every pair is nearly
    parallel — measured mean pairwise cosine ~0.99 against ~0.0 for centred data. Spherical
    k-means still returns a plausible partition there, decided by float residue rather than by
    angle, so it has to say so."""
    rng = np.random.default_rng(0)
    far_from_origin = (rng.standard_normal((60, 8)) + 100.0).astype(np.float32)

    with caplog.at_level("WARNING"):
        cluster_spherical_kmeans(far_from_origin, 3, default_config())

    assert_that(caplog.text).contains("nearly parallel")


def test_spherical_kmeans_quiet_on_a_real_cosine_space(caplog) -> None:
    """The counterpart: genuine unit-sphere data must not trip the warning, or it becomes noise."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    with caplog.at_level("WARNING"):
        cluster_spherical_kmeans(X, 3, default_config())

    assert_that(caplog.text).does_not_contain("nearly parallel")


@pytest.mark.parametrize("d", [8, 64], ids=lambda d: f"d{d}")
@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_spherical_kmeans_recovers_separated_blobs_exactly(d, seed) -> None:
    """The centroid update had NOTHING pinning it — a wrong one still returns a plausible partition.

    On angularly separated blobs the correct update recovers them exactly, so the returned
    partition is a real assertion about the arithmetic rather than about plausibility. Deliberately
    NOT run on degenerate geometry: where the mean pairwise cosine is near 1 the function itself
    warns that the assignment turns on float residue, and any partition there is arbitrary."""
    rng = np.random.default_rng(seed)
    k = 4
    centres = rng.standard_normal((k, d)) * 6.0
    truth = np.repeat(np.arange(k), 60)
    X = (centres[truth] + rng.standard_normal((k * 60, d)) * 0.1).astype(np.float32)

    labels = cluster_spherical_kmeans(X, k, default_config(), n_init=5)

    # A permutation of the blob ids is fine; splitting or merging a blob is not.
    recovered = {int(b): set(np.flatnonzero(labels == b).tolist()) for b in np.unique(labels)}
    expected = {int(b): set(np.flatnonzero(truth == b).tolist()) for b in range(k)}
    assert_that(sorted(map(sorted, recovered.values()))).is_equal_to(
        sorted(map(sorted, expected.values()))
    )


@pytest.mark.parametrize("force_fallback", [False, True], ids=["one-hot", "large-k-fallback"])
def test_spherical_centroid_update_seats_each_centroid_on_its_unit_member_mean(
    monkeypatch, force_fallback
) -> None:
    """The centroid update itself, asserted against its definition rather than through Lloyd.

    Reached through `cluster_spherical_kmeans` alone, a broken update still returns a plausible
    partition — un-normalized centroids, for instance, recover well-separated blobs perfectly well
    while biasing every assignment toward the longest mean. So the three properties are pinned
    directly: the value, the unit norm the cosine assignment depends on, and the emptied-cluster
    rule that keeps ``k`` distinct centres alive.

    The large-k branch exists so a big ``k`` never allocates a dense ``(n, k)`` one-hot — ``Config``
    bounds ``kmeans_n_clusters`` only at ``>= 1`` and ``k`` is clamped to ``n``, so ``k == n`` is
    reachable and the one-hot would be ``(n, n)``, 1.6 GB at 20 000 tracks. It is only taken on
    inputs too big for a unit test, so the cap is lowered to reach it."""
    rng = np.random.default_rng(4)
    Xn = rng.standard_normal((300, 16)).astype(np.float32)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    k = 12
    labels = rng.integers(0, k, 300)
    labels[labels == 5] = 4  # cluster 5 left empty on purpose
    seed_centroids = Xn[:k].copy()
    if force_fallback:
        monkeypatch.setattr(moodengine.cluster, "_ONEHOT_MAX_ELEMENTS", 1)

    centroids = seed_centroids.copy()
    moodengine.cluster._spherical_centroid_update(Xn, labels, centroids, k)

    filled = [j for j in range(k) if np.any(labels == j)]
    for j in filled:
        member_mean = Xn[labels == j].mean(axis=0)
        expected = member_mean / np.linalg.norm(member_mean)
        np.testing.assert_allclose(centroids[j], expected, rtol=0.0, atol=2e-6)
    norms = np.linalg.norm(centroids[filled], axis=1)
    np.testing.assert_allclose(norms, np.ones(len(filled)), rtol=0.0, atol=2e-6)
    assert_that(np.array_equal(centroids[5], seed_centroids[5])).is_true()  # empty kept intact


def test_spherical_degeneracy_probe_matches_the_explicit_gram() -> None:
    """The probe reports mean(P @ Pᵀ) without building the gram; the two must agree.

    ``mean(P @ Pᵀ) == ‖Σp‖² / m²`` because summing every pairwise dot is the dot of the sums. The
    threshold it feeds is 0.9, so the identity has to hold far tighter than that — it does, by
    orders of magnitude, on both an ordinary and a near-degenerate space."""
    for maker in (
        lambda r: r.standard_normal((400, 32)),
        lambda r: np.ones((400, 32)) + 0.001 * r.standard_normal((400, 32)),
    ):
        P = l2_normalize(maker(np.random.default_rng(0)).astype(np.float32), axis=1)
        explicit = float((P @ P.T).mean())
        sums = P.sum(axis=0, dtype=np.float64)

        identity = float(sums @ sums) / float(P.shape[0]) ** 2

        assert_that(abs(explicit - identity)).is_less_than(1e-6)


def test_spherical_kmeans_restarts_never_lose_to_a_single_draw() -> None:
    """`n_init` matches what `cluster_kmeans` already gets from sklearn. The kept partition must be
    the best by the objective the assignment maximizes — total cosine to one's own centroid — so
    more restarts can only help."""
    X, _ = _three_blobs(seed=31, per=25)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

    def objective(labels):
        centroids = np.vstack(
            [
                Xn[labels == j].mean(axis=0)
                / max(np.linalg.norm(Xn[labels == j].mean(axis=0)), 1e-8)
                for j in sorted(set(labels.tolist()))
            ]
        )
        remap = {c: i for i, c in enumerate(sorted(set(labels.tolist())))}
        return float(np.sum(Xn * centroids[[remap[int(v)] for v in labels]]))

    single = cluster_spherical_kmeans(X, 3, default_config(), n_init=1)
    many = cluster_spherical_kmeans(X, 3, default_config(), n_init=10)

    assert_that(objective(many)).is_greater_than_or_equal_to(objective(single) - 1e-5)


def test_bootstrap_refit_reduction_gap_is_zero_on_real_structure() -> None:
    """The gap between the two modes is the diagnostic, and it must NOT be a systematic offset:
    on genuinely separated blobs, re-fitting the projection per replicate changes nothing."""
    X, _ = _three_blobs(seed=32, per=40)
    cfg = dataclasses.replace(default_config(), bootstrap_n=5, kmeans_n_clusters=3)

    frozen = bootstrap_stability(X, "kmeans", cfg, n_boot=5)["mean_ari"]
    refit = bootstrap_stability(X, "kmeans", cfg, n_boot=5, refit_reduction=True)["mean_ari"]

    assert_that(abs(frozen - refit)).is_less_than(0.15)


def test_hdbscan_leaf_selection_splits_what_eom_keeps_whole() -> None:
    """The knob that addresses HDBSCAN's known failure on a homogeneous library — one dominant
    cluster swallowing the catalogue. `leaf` takes the finest stable clusters where the default
    `eom` prefers few large ones, and until now neither was reachable from Config."""
    X, _ = _n_blobs(6, seed=33, per=20)
    base = dataclasses.replace(default_config(), hdbscan_min_cluster_size=5)

    eom = cluster_hdbscan(X, base)
    leaf = cluster_hdbscan(X, dataclasses.replace(base, hdbscan_cluster_selection_method="leaf"))

    assert_that(len(set(leaf.tolist()) - {-1})).is_greater_than_or_equal_to(
        len(set(eom.tolist()) - {-1})
    )


def test_run_clustering_rejects_non_finite_input() -> None:
    """A NaN row must be named at the moodengine boundary — not surface as a deep
    'Input contains NaN' from UMAP/sklearn with no row context."""
    cfg = default_config()
    X, _ = _three_blobs(seed=10, per=3)
    X[4, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run_clustering(X, method="kmeans", config=cfg)


def test_run_clustering_bad_method_raises() -> None:
    """An unsupported method is rejected."""
    cfg = default_config()
    X, _ = _three_blobs(seed=9, per=3)
    with pytest.raises(
        ValueError, match=r"method must be 'hdbscan', 'kmeans', 'spherical' or 'leiden'"
    ):
        run_clustering(X, method="spectral", config=cfg)


def test_select_kmeans_k_recovers_true_blob_count() -> None:
    """Silhouette sweep picks the k matching the number of well-separated blobs."""
    cfg = default_config()
    for true_k in (3, 4):
        X, _ = _n_blobs(true_k, seed=20 + true_k, per=10)
        best_k, scores = select_kmeans_k(X, cfg, k_min=2, k_max=8)
        assert_that(best_k).is_equal_to(true_k)
        # Every swept k that produced >= 2 clusters has a silhouette in [-1, 1].
        assert_that(scores).contains(true_k)
        assert_that(all(-1.0 <= s <= 1.0 for s in scores.values())).is_true()
        assert_that(scores[best_k]).is_equal_to(max(scores.values()))


def test_select_kmeans_k_deterministic() -> None:
    """Same seed -> identical (best_k, scores)."""
    cfg = default_config()
    X, _ = _n_blobs(4, seed=99, per=10)
    a_k, a_scores = select_kmeans_k(X, cfg)
    b_k, b_scores = select_kmeans_k(X, cfg)
    assert_that(a_k).is_equal_to(b_k)
    assert_that(a_scores).is_equal_to(b_scores)


def test_select_kmeans_k_too_few_samples_returns_one_empty() -> None:
    """Fewer than 3 samples -> (1, {}) since silhouette is undefined."""
    cfg = default_config()
    best_k, scores = select_kmeans_k(np.zeros((2, 4), dtype=np.float32), cfg)
    assert_that(best_k).is_equal_to(1)
    assert_that(scores).is_equal_to({})


# --------------------------------------------------------------------------- #
# silhouette_original
# --------------------------------------------------------------------------- #


def test_silhouette_original_high_on_separated_blobs() -> None:
    """Cosine silhouette on the original space is high for clean blobs."""
    X, truth = _three_blobs(seed=30, per=10)
    sil = silhouette_original(X, truth, metric="cosine")
    assert_that(sil).is_not_none()
    assert_that(sil).is_greater_than(0.5)


def test_silhouette_original_excludes_noise() -> None:
    """Noise points (-1) are dropped before scoring; the score stays valid."""
    X, truth = _three_blobs(seed=31, per=8)
    labels = truth.copy()
    labels[:4] = -1  # mark a few points as noise
    sil = silhouette_original(X, labels, metric="cosine")
    assert_that(sil).is_not_none()
    assert_that(sil).is_between(-1.0, 1.0)


def test_silhouette_original_single_cluster_returns_none() -> None:
    """Fewer than 2 distinct non-noise clusters -> None (never raises)."""
    X, _ = _three_blobs(seed=32, per=6)
    labels = np.zeros(X.shape[0], dtype=int)
    assert_that(silhouette_original(X, labels)).is_none()
    # All-noise is likewise undefined.
    assert_that(silhouette_original(X, np.full(X.shape[0], -1, dtype=int))).is_none()


# --------------------------------------------------------------------------- #
# cluster_medoids
# --------------------------------------------------------------------------- #


def test_cluster_medoids_member_inside_each_cluster() -> None:
    """Each cluster's medoid is one of that cluster's own member indices."""
    X, truth = _three_blobs(seed=33, per=9)
    medoids = cluster_medoids(X, truth)
    assert_that(set(medoids.keys())).is_equal_to({0, 1, 2})
    for cid, idx in medoids.items():
        assert_that(truth[idx]).is_equal_to(cid)  # the medoid belongs to its cluster


def test_cluster_medoids_matches_naive_similarity_matrix() -> None:
    """The O(m·d) row-sum formula must pick the same medoids as the naive (m, m)
    cosine matrix it replaced — same argmax of mean cosine to clustermates."""
    for seed in (0, 7, 33, 52, 91):
        X, truth = _three_blobs(seed=seed, per=11)
        medoids = cluster_medoids(X, truth)

        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        for cid, med in medoids.items():
            idx = np.flatnonzero(truth == cid)
            members = Xn[idx]
            sims = members @ members.T
            np.fill_diagonal(sims, 0.0)
            expected = int(idx[int(np.argmax(sims.sum(axis=1) / (idx.size - 1)))])
            assert_that(med).described_as(f"seed={seed} cluster={cid}").is_equal_to(expected)


def test_cluster_medoids_excludes_noise() -> None:
    """The noise label (-1) gets no medoid."""
    X, truth = _three_blobs(seed=34, per=6)
    labels = truth.copy()
    labels[truth == 2] = -1
    medoids = cluster_medoids(X, labels)
    assert_that(set(medoids.keys())).is_equal_to({0, 1})
    assert_that(medoids).does_not_contain(-1)


def test_cluster_medoids_singleton_cluster() -> None:
    """A lone-member cluster reports that single index as its medoid."""
    X = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    medoids = cluster_medoids(X, labels)
    assert_that(medoids[1]).is_equal_to(2)


# --------------------------------------------------------------------------- #
# outlier_scores
# --------------------------------------------------------------------------- #


def test_outlier_scores_range_and_noise_is_one() -> None:
    """Scores are 1 - cosine in [0, ~2]; noise points score exactly 1.0."""
    X, truth = _three_blobs(seed=35, per=8)
    labels = truth.copy()
    labels[:5] = -1
    scores = outlier_scores(X, labels)
    assert_that(scores.shape).is_equal_to((X.shape[0],))
    np.testing.assert_allclose(scores[:5], 1.0, atol=1e-6)  # noise -> 1.0
    # Tight blobs -> in-cluster members sit close to their centroid (small score).
    non_noise = scores[labels != -1]
    assert_that(bool(np.all(non_noise >= -1e-6))).is_true()
    assert_that(float(non_noise.mean())).is_less_than(0.1)


def test_outlier_scores_all_noise() -> None:
    """An all-noise input scores every track 1.0 without raising."""
    X = np.zeros((5, 4), dtype=np.float32)
    scores = outlier_scores(X, np.full(5, -1, dtype=int))
    np.testing.assert_allclose(scores, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# bootstrap_stability
# --------------------------------------------------------------------------- #


def test_bootstrap_stability_keys_and_high_ari_on_blobs() -> None:
    """Stable, well-separated blobs yield high mean ARI and the documented keys."""
    cfg = dataclasses.replace(default_config(), kmeans_n_clusters=3, bootstrap_n=8, seed=0)
    X, _ = _three_blobs(seed=36, per=15)
    out = bootstrap_stability(X, method="kmeans", config=cfg)
    assert_that(set(out.keys())).is_equal_to(
        {"mean_ari", "std_ari", "mean_ami", "mean_noise_agreement", "n_boot", "space"}
    )
    assert_that(out["n_boot"]).is_equal_to(8)
    assert_that(out["mean_ari"]).is_greater_than(0.8)  # clustering is highly reproducible
    assert_that(out["std_ari"]).is_greater_than_or_equal_to(0.0)
    # kmeans never emits noise, so the noise/non-noise split trivially agrees everywhere.
    assert_that(out["mean_noise_agreement"]).is_close_to(1.0, tolerance=1e-9)


def test_bootstrap_stability_deterministic() -> None:
    """Same config.seed -> identical stability summary (seeded subsampling)."""
    cfg = dataclasses.replace(default_config(), kmeans_n_clusters=3, bootstrap_n=6, seed=7)
    X, _ = _three_blobs(seed=37, per=12)
    a = bootstrap_stability(X, "kmeans", cfg)
    b = bootstrap_stability(X, "kmeans", cfg)
    assert_that(a).is_equal_to(b)


def test_bootstrap_stability_degenerate_returns_zeros() -> None:
    """Too few samples (or too few bootstraps) yields zeroed metrics."""
    cfg = dataclasses.replace(default_config(), bootstrap_n=5)
    out = bootstrap_stability(np.zeros((3, 4), dtype=np.float32), "kmeans", cfg)
    assert_that(out).is_equal_to(
        {
            "mean_ari": 0.0,
            "std_ari": 0.0,
            "mean_ami": 0.0,
            "mean_noise_agreement": 0.0,
            "n_boot": 5,
            "space": "original",  # n=3 is below the UMAP floor, so the raw rows would be clustered
        }
    )


def test_bootstrap_stability_reports_the_reduced_space_on_the_default_path() -> None:
    """The default config clusters the UMAP layout, and the result says so."""
    cfg = dataclasses.replace(default_config(), kmeans_n_clusters=3, bootstrap_n=4, seed=0)
    X, _ = _three_blobs(seed=41, per=15)

    out = bootstrap_stability(X, "kmeans", cfg)

    assert_that(out["space"]).is_equal_to("reduced")


def test_bootstrap_stability_skips_the_reduction_under_cluster_space_original(mocker) -> None:
    """Under cluster_space='original' the replicates cluster the raw embeddings, exactly as
    run_clustering does — so no UMAP is fit at all and the result names the space it used.

    Reducing here while the pipeline clusters the originals would validate a partition that is
    never shipped, which is the one thing a stability number must not do."""
    cfg = dataclasses.replace(
        default_config(), cluster_space="original", kmeans_n_clusters=3, bootstrap_n=4, seed=0
    )
    X, _ = _three_blobs(seed=42, per=15)
    spy = mocker.spy(moodengine.cluster, "reduce_umap")

    out = bootstrap_stability(X, "kmeans", cfg)

    assert_that(spy.call_count).is_equal_to(0)
    assert_that(out["space"]).is_equal_to("original")
    assert_that(out["mean_ari"]).is_greater_than(0.8)  # separable blobs stay separable unreduced


def test_bootstrap_stability_refit_reduction_is_inert_and_logged_in_the_original_space(
    caplog,
) -> None:
    """`refit_reduction` has nothing to re-fit when clustering the originals. The documented
    reading is the GAP between the two modes, so returning the frozen number in silence would
    read as 'the structure is real' precisely when nothing was re-fit."""
    cfg = dataclasses.replace(
        default_config(), cluster_space="original", kmeans_n_clusters=3, bootstrap_n=4, seed=0
    )
    X, _ = _three_blobs(seed=43, per=15)

    with caplog.at_level(logging.INFO, logger="moodengine.cluster"):
        refit = bootstrap_stability(X, "kmeans", cfg, refit_reduction=True)
    frozen = bootstrap_stability(X, "kmeans", cfg)

    assert_that(refit).is_equal_to(frozen)
    assert_that(caplog.text).contains("refit_reduction=True is inert")


def test_bootstrap_stability_reduces_once_on_the_default_path(mocker) -> None:
    """The frozen mode fits UMAP exactly once and subsamples its rows, rather than per replicate —
    that is what isolates clustering variance from UMAP's own run-to-run noise."""
    cfg = dataclasses.replace(default_config(), kmeans_n_clusters=3, bootstrap_n=5, seed=0)
    X, _ = _three_blobs(seed=44, per=15)
    spy = mocker.spy(moodengine.cluster, "reduce_umap")

    bootstrap_stability(X, "kmeans", cfg)

    assert_that(spy.call_count).is_equal_to(1)


def test_bootstrap_stability_hdbscan_noise_agreement_is_a_fraction() -> None:
    """With HDBSCAN (the one method that emits -1), the noise-split agreement is a proper
    fraction and the shape ARI stays in range — noise is scored separately, not mixed in."""
    cfg = dataclasses.replace(default_config(), bootstrap_n=6, seed=1)
    X, _ = _three_blobs(seed=51, per=20)

    out = bootstrap_stability(X, method="hdbscan", config=cfg)

    assert_that(out["mean_noise_agreement"]).is_between(0.0, 1.0)
    assert_that(out["mean_ari"]).is_between(-1.0, 1.0)
    assert_that(out["n_boot"]).is_equal_to(6)


def test_bootstrap_stability_bad_method_raises() -> None:
    """An unsupported method is rejected."""
    cfg = default_config()
    X, _ = _three_blobs(seed=38, per=10)
    with pytest.raises(
        ValueError, match=r"method must be 'hdbscan', 'kmeans', 'spherical' or 'leiden'"
    ):
        bootstrap_stability(X, "spectral", cfg)


# --------------------------------------------------------------------------- #
# spherical k-means (cosine Lloyd)
# --------------------------------------------------------------------------- #
def _angular_clusters(seed: int = 0, per: int = 15, dim: int = 8, k: int = 3):
    """k clusters separated by DIRECTION (orthogonal basis dirs) with widely varied magnitudes.

    Euclidean k-means tends to split these by magnitude band; cosine (spherical) recovers the
    directional structure. Returns (X, truth)."""
    rng = np.random.default_rng(seed)
    blocks, truth = [], []
    for c in range(k):
        direction = np.zeros(dim, dtype=np.float32)
        direction[c % dim] = 1.0  # orthogonal per-cluster direction
        for _ in range(per):
            v = direction + 0.03 * rng.standard_normal(dim).astype(np.float32)
            v = v / np.linalg.norm(v)
            radius = float(rng.uniform(0.3, 6.0))  # varied norms -> magnitude dominates euclidean
            blocks.append((radius * v).astype(np.float32))
        truth.extend([c] * per)
    return np.stack(blocks).astype(np.float32), np.array(truth)


def test_spherical_deterministic_with_seed() -> None:
    """Same seed -> identical spherical labels."""
    cfg = default_config()
    X, _ = _angular_clusters(seed=3)
    np.testing.assert_array_equal(
        cluster_spherical_kmeans(X, 3, cfg), cluster_spherical_kmeans(X, 3, cfg)
    )


def test_spherical_never_returns_noise() -> None:
    """Spherical assignment is total — no -1 label."""
    cfg = default_config()
    X, _ = _angular_clusters(seed=4)
    labels = cluster_spherical_kmeans(X, 3, cfg)
    assert_that(labels.shape).is_equal_to((X.shape[0],))
    assert_that(set(labels.tolist())).does_not_contain(-1)


def test_spherical_clamps_k_to_n() -> None:
    """k is clamped to at most n; labels stay in range."""
    cfg = default_config()
    X = np.random.default_rng(0).standard_normal((3, 6)).astype(np.float32)
    labels = cluster_spherical_kmeans(X, n_clusters=10, config=cfg)
    assert_that(labels.shape).is_equal_to((3,))
    assert_that(set(labels.tolist()).issubset(set(range(3)))).is_true()


def test_spherical_beats_or_ties_kmeans_on_cosine_silhouette() -> None:
    """Criterion #4: on an angularly-separated set spherical's cosine silhouette >= kmeans' — logged."""
    cfg = default_config()
    X, _ = _angular_clusters(seed=5, k=3)
    s_sph = silhouette_original(X, cluster_spherical_kmeans(X, 3, cfg), metric="cosine")
    s_km = silhouette_original(X, cluster_kmeans(X, 3, cfg), metric="cosine")
    print(f"[spherical bench] cosine_silhouette spherical={s_sph:.3f} kmeans={s_km:.3f}")
    assert_that(s_sph).is_not_none()
    assert_that(s_km).is_not_none()
    assert_that(s_sph).is_greater_than_or_equal_to(s_km)


def test_run_clustering_spherical_shape() -> None:
    """run_clustering routes 'spherical' and returns the standard shape, no -1."""
    cfg = default_config()
    X, _ = _angular_clusters(seed=6)
    out = run_clustering(X, "spherical", cfg)
    assert_that(out["method"]).is_equal_to("spherical")
    assert_that(out["labels"].shape).is_equal_to((X.shape[0],))
    assert_that(out["coords2d"].shape[1]).is_equal_to(2)
    assert_that(set(out["labels"].tolist())).does_not_contain(-1)


def test_bootstrap_stability_spherical_runs() -> None:
    """bootstrap_stability accepts 'spherical' and returns the standard keys."""
    cfg = dataclasses.replace(default_config(), bootstrap_n=5, kmeans_n_clusters=3)
    X, _ = _angular_clusters(seed=7, per=20)
    out = bootstrap_stability(X, "spherical", cfg)
    assert_that(set(out)).is_equal_to(
        {"mean_ari", "std_ari", "mean_ami", "mean_noise_agreement", "n_boot", "space"}
    )
    assert_that(out["n_boot"]).is_equal_to(5)


# --- coverage entropy --------------------------------------------------------


def test_coverage_entropy_uniform_is_one_singleton_is_zero():
    uniform = coverage_entropy(np.array([0, 0, 1, 1, 2, 2]))
    assert_that(uniform["normalized_entropy"]).is_close_to(1.0, tolerance=1e-9)
    assert_that(uniform["n_bins"]).is_equal_to(3)
    assert_that(uniform["perplexity"]).is_close_to(
        3.0, tolerance=1e-6
    )  # 3 equally-occupied regions

    singleton = coverage_entropy(np.array([5, 5, 5, 5]))
    assert_that(singleton["entropy"]).is_close_to(0.0, tolerance=1e-9)
    assert_that(singleton["normalized_entropy"]).is_close_to(
        0.0, tolerance=1e-6
    )  # n_bins<=1 → defined as 0.0: a single occupied region is minimal diversity (entropy 0)
    assert_that(singleton["perplexity"]).is_close_to(1.0, tolerance=1e-6)


def test_coverage_entropy_perplexity_is_base_pow_entropy_and_shares_sum_to_one():
    labels = np.array([0, 0, 0, 1, 2, 2])
    out = coverage_entropy(labels, base=2.0)
    assert_that(out["perplexity"]).is_close_to(2.0 ** out["entropy"], tolerance=1e-6)
    assert_that(sum(out["shares"].values())).is_close_to(1.0, tolerance=1e-9)
    assert_that(out["shares"][0]).is_close_to(0.5, tolerance=1e-6)


def test_coverage_entropy_counts_noise_as_a_bin():
    # -1 (HDBSCAN noise) is an honest region, not dropped.
    out = coverage_entropy(np.array([-1, -1, 0, 0]))
    assert_that(out["n_bins"]).is_equal_to(2)
    assert_that(out["shares"]).contains(-1)
    assert_that(out["shares"][-1]).is_close_to(0.5, tolerance=1e-6)
    assert_that(out["normalized_entropy"]).is_close_to(
        1.0, tolerance=1e-9
    )  # 2 equal bins → uniform


def test_coverage_entropy_normalized_never_exceeds_one_for_uniform():
    """Regression: a perfectly uniform occupancy gives entropy == log_base(n_bins), but the division
    can round to 1.0000000000000002; normalized_entropy must stay clamped in [0, 1] (a downstream
    ``<= 1.0`` bound would otherwise 500 on the ideal maximum-diversity library)."""
    for n_bins in (5, 7, 11, 13):
        out = coverage_entropy(np.repeat(np.arange(n_bins), 4))
        assert_that(out["normalized_entropy"]).is_between(0.0, 1.0)
        assert_that(out["normalized_entropy"]).is_close_to(1.0, tolerance=1e-9)


def test_coverage_entropy_empty_never_raises():
    out = coverage_entropy(np.array([], dtype=int))
    assert_that(out).is_equal_to(
        {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "perplexity": 0.0,
            "n_bins": 0,
            "shares": {},
        }
    )


# --------------------------------------------------------------------------- #
# per-cluster silhouette + sub-clustering (emergent moods / sub-moods)
# --------------------------------------------------------------------------- #


def test_per_cluster_silhouette_high_on_separated_blobs() -> None:
    """Three well-separated blobs → a high (positive) silhouette per cluster; keys = the cluster ids."""
    X, truth = _three_blobs(seed=60, per=12)
    pcs = per_cluster_silhouette(X, truth, metric="cosine")
    assert_that(set(pcs.keys())).is_equal_to({0, 1, 2})
    assert_that(all(v is not None and v > 0.5 for v in pcs.values())).is_true()


def test_per_cluster_silhouette_excludes_noise() -> None:
    """The noise label (-1) gets no entry; remaining clusters still score."""
    X, truth = _three_blobs(seed=61, per=9)
    labels = truth.copy()
    labels[truth == 2] = -1
    pcs = per_cluster_silhouette(X, labels)
    assert_that(set(pcs.keys())).is_equal_to({0, 1})
    assert_that(pcs).does_not_contain(-1)


def test_per_cluster_silhouette_single_cluster_is_none() -> None:
    """Fewer than 2 non-noise clusters → that cluster is None (not calculable), never raises."""
    X, _ = _three_blobs(seed=62, per=6)
    pcs = per_cluster_silhouette(X, np.zeros(X.shape[0], dtype=int))
    assert_that(pcs).is_equal_to({0: None})
    # All-noise → no entries at all.
    assert_that(per_cluster_silhouette(X, np.full(X.shape[0], -1, dtype=int))).is_equal_to({})


def test_sub_cluster_exact_partition_and_recovers_k() -> None:
    """sub_cluster partitions the subset EXACTLY (every row one sub_id) and recovers a clean 3-way split."""
    cfg = default_config()
    X, truth = _n_blobs(3, seed=63, per=14)
    out = sub_cluster(X, cfg, k_min=2, k_max=6)
    assert_that(set(out.keys())).is_equal_to(
        {
            "sub_labels",
            "sub_k",
            "silhouette",
            "medoids",
            "per_cluster_silhouette",
        }
    )
    labels = out["sub_labels"]
    assert_that(labels.shape).is_equal_to((X.shape[0],))
    # exact partition: sizes of the sub-clusters sum to the parent size, no row unassigned / duplicated
    _, counts = np.unique(labels, return_counts=True)
    assert_that(int(counts.sum())).is_equal_to(X.shape[0])
    assert_that(out["sub_k"]).is_greater_than_or_equal_to(
        2
    )  # a clean mixture yields >= 2 sub-moods
    assert_that(set(out["medoids"].keys())).is_equal_to(
        set(labels.tolist())
    )  # one medoid per sub-cluster
    assert_that(set(out["per_cluster_silhouette"].keys())).is_equal_to(set(labels.tolist()))
    assert_that(out["silhouette"] is None or -1.0 <= out["silhouette"] <= 1.0).is_true()


def test_sub_cluster_tiny_input_single_submood() -> None:
    """m < 3 → a single sub-mood (sub_k=1, all labels 0), no crash, silhouette None."""
    cfg = default_config()
    out = sub_cluster(np.random.default_rng(0).standard_normal((2, 6)).astype(np.float32), cfg)
    assert_that(out["sub_k"]).is_equal_to(1)
    assert_that(out["sub_labels"].tolist()).is_equal_to([0, 0])
    assert_that(out["silhouette"]).is_none()
    assert_that(out["per_cluster_silhouette"]).is_equal_to({0: None})


def test_sub_cluster_deterministic_with_seed() -> None:
    """Same config.seed → identical sub-labels (seeded select_kmeans_k + kmeans)."""
    cfg = default_config()
    X, _ = _n_blobs(4, seed=64, per=12)
    a = sub_cluster(X, cfg)
    b = sub_cluster(X, cfg)
    np.testing.assert_array_equal(a["sub_labels"], b["sub_labels"])
    assert_that(a["sub_k"]).is_equal_to(b["sub_k"])


# --------------------------------------------------------------------------- #
# Leiden community detection — optional backend, guarded by importorskip
# --------------------------------------------------------------------------- #


def test_cluster_leiden_recovers_blobs() -> None:
    """Criterion #2: Leiden recovers three well-separated blobs (ARI >= 0.9) with no -1, contiguous."""
    pytest.importorskip("leidenalg")
    from sklearn.metrics import adjusted_rand_score

    cfg = default_config()
    X, truth = _three_blobs(
        seed=40, per=20
    )  # per > leiden_k_neighbors → within-blob neighbourhoods
    labels = cluster_leiden(X, cfg)
    assert_that(labels.shape).is_equal_to((X.shape[0],))
    assert_that(set(labels.tolist())).does_not_contain(-1)  # Leiden partitions the whole graph
    assert_that(set(labels.tolist())).is_equal_to(
        set(range(len(set(labels.tolist()))))
    )  # contiguous 0..K-1
    assert_that(adjusted_rand_score(truth, labels)).is_greater_than_or_equal_to(0.9)


def test_cluster_leiden_deterministic_with_seed() -> None:
    """Same seed → identical Leiden labels."""
    pytest.importorskip("leidenalg")
    cfg = default_config()
    X, _ = _three_blobs(seed=41, per=18)
    np.testing.assert_array_equal(cluster_leiden(X, cfg), cluster_leiden(X, cfg))


def test_cluster_leiden_resolution_increases_communities() -> None:
    """The resolution knob is LIVE: on a CONNECTED single cloud (whose kNN graph has no natural cut),
    a low resolution collapses to one community and a high resolution fragments into strictly more —
    a strict inequality that would FAIL if resolution_parameter were ignored (both would tie)."""
    pytest.importorskip("leidenalg")
    X = np.random.default_rng(42).standard_normal((80, 6)).astype(np.float32)  # one connected blob
    low = cluster_leiden(X, dataclasses.replace(default_config(), leiden_resolution=0.1))
    high = cluster_leiden(X, dataclasses.replace(default_config(), leiden_resolution=3.0))
    assert_that(len(set(low.tolist()))).is_equal_to(1)  # low res → the whole cloud is one community
    assert_that(len(set(high.tolist()))).is_greater_than(
        len(set(low.tolist()))
    )  # high res → strictly more (knob is wired)


def test_cluster_leiden_tiny_input_all_zero() -> None:
    """A lone point cannot form a graph → all-zero labels without touching the backend."""
    cfg = default_config()
    labels = cluster_leiden(np.zeros((1, 4), dtype=np.float32), cfg)
    assert_that(labels.tolist()).is_equal_to([0])


def test_cluster_leiden_two_row_input_no_crash() -> None:
    """n == 2 is the kNN boundary (k must drop to 1, not the default floor of 2): leiden must return
    clean labels like the other methods, never crash kneighbors_graph on an out-of-range k (regression)."""
    pytest.importorskip("leidenalg")
    cfg = default_config()
    X = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    labels = cluster_leiden(X, cfg)
    assert_that(labels.shape).is_equal_to((2,))
    assert_that(set(labels.tolist())).does_not_contain(-1)
    # run_clustering routes the same tiny input without raising (parity with kmeans/hdbscan/spherical).
    out = run_clustering(X, "leiden", cfg)
    assert_that(out["labels"].shape).is_equal_to((2,))
    assert_that(set(out["labels"].tolist())).does_not_contain(-1)


def test_cluster_leiden_no_dep_raises(monkeypatch) -> None:
    """Absent leidenalg → MissingDependencyError naming the extra (never a fabricated
    clustering). Runs even when the package IS installed (simulates absence via sys.modules)."""
    import sys

    from moodengine.exceptions import MissingDependencyError

    monkeypatch.setitem(sys.modules, "leidenalg", None)  # `import leidenalg` → ImportError
    cfg = default_config()
    X, _ = _three_blobs(seed=43, per=6)
    with pytest.raises(MissingDependencyError, match=r"moodengine\[cluster-graph\]") as exc:
        cluster_leiden(X, cfg)
    assert_that(exc.value.extra).is_equal_to("cluster-graph")


def test_run_clustering_leiden_full_path() -> None:
    """run_clustering routes 'leiden' and returns the standard shape, method tag, no -1."""
    pytest.importorskip("leidenalg")
    cfg = default_config()
    X, _ = _three_blobs(seed=44, per=20)  # >= max(umap_n_neighbors,4) → real UMAP+leiden path
    out = run_clustering(X, "leiden", cfg)
    assert_that(set(out.keys())).is_equal_to({"labels", "coords2d", "metrics", "method"})
    assert_that(out["method"]).is_equal_to("leiden")
    assert_that(out["labels"].shape).is_equal_to((X.shape[0],))
    assert_that(out["coords2d"].shape).is_equal_to((X.shape[0], 2))
    assert_that(set(out["labels"].tolist())).does_not_contain(-1)


def test_graph_modularity_range() -> None:
    """graph_modularity is a real float in [-0.5, 1.0] on a clean partition; None when degenerate."""
    pytest.importorskip("leidenalg")
    cfg = default_config()
    X, truth = _three_blobs(seed=45, per=18)
    mod = graph_modularity(X, truth, cfg)
    assert_that(mod).is_not_none()
    assert_that(mod).is_between(-0.5, 1.0)
    assert_that(mod).is_greater_than(0.3)  # three separated blobs form a strongly modular kNN graph
    # Fewer than 2 clusters → None (no fabricated number).
    assert_that(graph_modularity(X, np.zeros(X.shape[0], dtype=int), cfg)).is_none()


def test_bootstrap_stability_leiden_keys() -> None:
    """bootstrap_stability accepts 'leiden' and returns the standard keys with high ARI on blobs."""
    pytest.importorskip("leidenalg")
    cfg = dataclasses.replace(default_config(), bootstrap_n=6)
    X, _ = _three_blobs(seed=46, per=20)
    out = bootstrap_stability(X, "leiden", cfg)
    assert_that(set(out)).is_equal_to(
        {"mean_ari", "std_ari", "mean_ami", "mean_noise_agreement", "n_boot", "space"}
    )
    assert_that(out["n_boot"]).is_equal_to(6)
    assert_that(out["mean_ari"]).is_greater_than(
        0.8
    )  # well-separated blobs are stable under resampling


# --------------------------------------------------------------------------- #
# cluster hierarchy — scipy linkage over medoids, scipy-only (no leidenalg)
# --------------------------------------------------------------------------- #


def test_cluster_hierarchy_shapes() -> None:
    """The hierarchy dict has the documented, JSON-serializable shape over K medoids."""
    import json

    cfg = default_config()
    X, truth = _n_blobs(5, seed=50, per=12)
    labels = cluster_kmeans(X, 5, cfg)
    h = cluster_hierarchy(X, labels, cfg)
    assert_that(set(h.keys())).is_equal_to(
        {
            "linkage",
            "cluster_ids",
            "cophenetic",
            "linkage_method",
            "metric",
            "n_super_groups",
            "super_group_of",
        }
    )
    K = len(set(labels.tolist()))
    assert_that(h["cluster_ids"]).is_equal_to(sorted(set(int(c) for c in labels)))
    assert_that(h["linkage"]).is_length(K - 1)  # a full dendrogram over K leaves
    assert_that(all(len(row) == 4 for row in h["linkage"])).is_true()
    assert_that(h["linkage_method"]).is_equal_to("average")
    assert_that(h["metric"]).is_equal_to("cosine")
    assert_that(h["cophenetic"]).is_instance_of(float)  # K >= 3 → a real correlation
    assert_that(h["n_super_groups"]).is_between(2, K)
    json.dumps(h)  # must be JSON-serializable (native types)


def test_cluster_hierarchy_cophenetic_none_on_degenerate_distances() -> None:
    """A degenerate medoid distance matrix (identical medoids → zero-variance cosine distances) makes
    scipy's cophenet return NaN (0/0 Pearson), not raise. cophenetic must be None then — a NaN is not
    a real correlation and would break JSON rendering (allow_nan=False) once persisted downstream."""
    cfg = default_config()
    X = np.ones((9, 8), dtype=np.float32)  # all points identical → identical per-cluster medoids
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # K == 3 triggers the cophenetic path
    h = cluster_hierarchy(X, labels, cfg)
    assert_that(h["cophenetic"]).is_none()


def test_cluster_hierarchy_cut_covers_all_clusters() -> None:
    """Every cluster is placed in exactly one non-empty super-group; the count is honest."""
    cfg = default_config()
    X, _ = _n_blobs(6, seed=51, per=10)
    labels = cluster_kmeans(X, 6, cfg)
    h = cluster_hierarchy(X, labels, cfg)
    cluster_ids = set(int(c) for c in labels)
    # every cluster id appears exactly once as a super_group_of key
    assert_that(set(int(k) for k in h["super_group_of"].keys())).is_equal_to(cluster_ids)
    groups = set(h["super_group_of"].values())
    assert_that(groups).is_equal_to(
        set(range(h["n_super_groups"]))
    )  # contiguous 0..n_super-1, all non-empty
    assert_that(groups).is_length(h["n_super_groups"])


def test_cluster_hierarchy_explicit_super_groups_clamped() -> None:
    """An explicit n_super_groups is clamped to [2, K]; out-of-range never over/under-cuts."""
    cfg = default_config()
    X, _ = _n_blobs(4, seed=52, per=10)
    labels = cluster_kmeans(X, 4, cfg)
    assert_that(cluster_hierarchy(X, labels, cfg, n_super_groups=99)["n_super_groups"]).is_equal_to(
        4
    )  # clamp to K
    assert_that(cluster_hierarchy(X, labels, cfg, n_super_groups=1)["n_super_groups"]).is_equal_to(
        2
    )  # floor at 2


def test_cluster_hierarchy_single_cluster_is_empty() -> None:
    """K <= 1 → an honest empty structure (no linkage, no cophenetic) without raising."""
    cfg = default_config()
    X, _ = _three_blobs(seed=53, per=6)
    h = cluster_hierarchy(X, np.zeros(X.shape[0], dtype=int), cfg)
    assert_that(h["linkage"]).is_equal_to([])
    assert_that(h["cophenetic"]).is_none()
    assert_that(h["n_super_groups"]).is_equal_to(1)
    assert_that(h["super_group_of"]).is_equal_to({"0": 0})
    # All-noise → no clusters at all.
    h0 = cluster_hierarchy(X, np.full(X.shape[0], -1, dtype=int), cfg)
    assert_that(h0["cluster_ids"]).is_equal_to([])
    assert_that(h0["super_group_of"]).is_equal_to({})
    assert_that(h0["n_super_groups"]).is_equal_to(0)


def test_sub_cluster_fits_kmeans_once_per_candidate_k_and_no_more(mocker) -> None:
    """Pins the REUSE, which no assertion on the returned value can.

    The k-sweep fits KMeans at every candidate k; the old code then re-fit at the winning k, a sixth
    fit on top of five. Because that refit is deterministic it returns exactly the labels the sweep
    already had — so the result is identical either way and only the call COUNT distinguishes them.
    Asserting on the labels alone would leave the optimization free to be reverted silently."""
    spy = mocker.spy(moodengine.cluster, "cluster_kmeans")
    rng = np.random.default_rng(11)
    centres = rng.standard_normal((4, 16)) * 4.0
    X = (centres[rng.integers(0, 4, 240)] + rng.standard_normal((240, 16)) * 0.4).astype(np.float32)

    sub_cluster(X, default_config(), k_min=2, k_max=6)

    assert_that(spy.call_count).is_equal_to(5)  # k = 2..6, and nothing after


def test_sub_cluster_labels_equal_a_fresh_fit_at_the_same_k() -> None:
    """Reusing the sweep's labels is valid only because `cluster_kmeans` is deterministic in
    ``(X, k, config.seed)``. That is an assumption the reuse rests on, so it is asserted rather than
    trusted."""
    rng = np.random.default_rng(11)
    centres = rng.standard_normal((4, 16)) * 4.0
    X = (centres[rng.integers(0, 4, 240)] + rng.standard_normal((240, 16)) * 0.4).astype(np.float32)
    config = default_config()

    result = sub_cluster(X, config)

    assert_that(
        np.array_equal(result["sub_labels"], cluster_kmeans(X, result["sub_k"], config))
    ).is_true()


def test_sub_cluster_silhouette_matches_the_public_scorer() -> None:
    """Both silhouette figures now come from one pairwise pass; they must not drift from it.

    `sub_cluster` derives its overall score as the mean of the per-point samples the per-cluster
    breakdown already computes, instead of calling `silhouette_original` separately — sklearn
    defines `silhouette_score` as exactly that mean. Equality is asserted EXACTLY, not within a
    tolerance, because the two really are the same arithmetic on the same values."""
    rng = np.random.default_rng(12)
    centres = rng.standard_normal((3, 24)) * 3.0
    X = (centres[rng.integers(0, 3, 200)] + rng.standard_normal((200, 24)) * 0.5).astype(np.float32)

    result = sub_cluster(X, default_config())

    labels = result["sub_labels"]
    assert_that(result["silhouette"]).is_equal_to(silhouette_original(X, labels, metric="cosine"))
    assert_that(result["per_cluster_silhouette"]).is_equal_to(
        per_cluster_silhouette(X, labels, metric="cosine")
    )


def test_sub_cluster_survives_a_subset_with_no_separable_k() -> None:
    """When no k yields two distinct clusters the sweep scores nothing, so there is no fit to reuse.

    All-identical rows are the reachable case — a parent mood whose members are duplicates or
    re-encodes. The sweep then returns its fallback k with NO labels attached, and `sub_cluster`
    has to cluster for itself; taking the missing labels at face value would crash here."""
    X = np.ones((60, 12), dtype=np.float32)

    result = sub_cluster(X, default_config())

    assert_that(result["sub_labels"].shape).is_equal_to((60,))
    assert_that(sorted(set(result["sub_labels"].tolist()))).is_equal_to([0])  # one sub-mood
    assert_that(result["silhouette"]).is_none()  # undefined with a single cluster
    assert_that(result["sub_k"]).is_equal_to(1)


def test_bootstrap_stability_scores_ari_and_ami_over_the_same_replicate_pairs(mocker) -> None:
    """A replicate pair contributes to BOTH agreement metrics or to neither.

    Appending the ARI before computing the AMI let an AMI-only failure leave the two lists at
    different lengths, so ``mean_ari``/``std_ari`` and ``mean_ami`` were reported over different
    samples of the same pairs — a discrepancy nothing in the output would reveal.

    Both metrics are stubbed to return the SAME value for a given pair, so aligned lists give
    identical means and any desync separates them.
    """
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(c, 1.6, (40, 4)) for c in (0.0, 1.0, 2.0)]).astype(np.float32)
    config = dataclasses.replace(
        default_config(), seed=11, kmeans_n_clusters=3, umap_n_neighbors=5, cluster_space="original"
    )
    counter = iter(range(1, 500))
    last_ari = [0.0]
    pair = [0]

    def fake_ari(_a, _b):
        # Squares, not consecutive integers: an arithmetic sequence has the SAME mean as its
        # every-other-element subsequence, so a linear stub cannot see the desync at all.
        last_ari[0] = float(next(counter)) ** 2
        return last_ari[0]

    def fake_ami(_a, _b):
        pair[0] += 1
        if pair[0] % 2 == 0:  # this metric fails on every other pair
            raise RuntimeError("AMI failed on this pair")
        return last_ari[0]

    mocker.patch("sklearn.metrics.adjusted_rand_score", side_effect=fake_ari)
    mocker.patch("sklearn.metrics.adjusted_mutual_info_score", side_effect=fake_ami)

    metrics = bootstrap_stability(X, "kmeans", config, n_boot=6)

    assert_that(pair[0]).is_greater_than(2)  # the failure arm was actually exercised
    assert_that(metrics["mean_ari"]).is_close_to(metrics["mean_ami"], tolerance=1e-12)
