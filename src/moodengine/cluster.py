"""Dimensionality reduction + clustering for track-level embeddings.

Torch-free: only numpy / scikit-learn / umap-learn / hdbscan are used here so
this stage runs on the lightweight pipeline install. UMAP and HDBSCAN are
imported lazily inside the functions that need them, keeping module import cheap
and the degenerate (tiny-input) paths usable without those packages on the hot
path. Functions are deterministic given ``config.seed`` where the underlying
algorithm allows it.
"""

from __future__ import annotations

import logging
from typing import get_args

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from moodengine._math import l2_normalize
from moodengine._typing import (
    ClusteringResult,
    ClusterMethod,
    ClusterMetrics,
    CoverageEntropyResult,
    Reducer2D,
    StabilityMetrics,
    SubClusterResult,
)
from moodengine._validation import ensure_finite_2d
from moodengine.config import Config
from moodengine.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


def reduce_umap(X: np.ndarray, n_components: int, config: Config) -> tuple[np.ndarray, object]:
    """Fit UMAP reducing ``X`` (n, d) to (n, n_components).

    Uses ``config.umap_n_neighbors``, ``config.umap_min_dist``,
    ``config.umap_metric`` and ``config.seed``. Returns the embedding and the
    fitted reducer.
    """
    import umap

    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    # UMAP requires n_neighbors < n_samples; clamp defensively.
    n_neighbors = max(2, min(config.umap_n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.seed,
    )
    embedding = reducer.fit_transform(X)
    return np.asarray(embedding, dtype=np.float32), reducer


class ProjectionMethodUnavailable(MissingDependencyError, RuntimeError):
    """A 2-D projection method needs an optional package that isn't installed (e.g. PaCMAP → numba).
    Carries ``.method`` so the caller surfaces a clear, method-named error — never a silent fallback.
    Keeps ``RuntimeError`` as a secondary base: that was its public base before the
    :mod:`moodengine.exceptions` hierarchy existed, and existing catchers must keep working."""

    def __init__(self, method: str, package: str = "pacmap", extra: str = "pacmap") -> None:
        super().__init__(f"projection method {method!r}", package, extra)
        self.method = method


class _IdentityReducer:
    """Reducer for the degenerate (tiny-n) path: exposes ``.transform`` == the same PCA/first-dims
    fallback, so :func:`fit_projection` keeps a stable ``(coords, reducer)`` signature with no fitted state."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        return _coords2d_fallback(np.asarray(X_new, dtype=np.float32), self._config)


def fit_projection(X: np.ndarray, config: Config) -> tuple[np.ndarray, object]:
    """Fit a REUSABLE 2-D map projection of ``X`` (n, d) → ``(coords (n, 2) float32, reducer)``.

    ``reducer.transform(X_new) -> (m, 2)`` places new points in the SAME layout — the stability
    fix (recluster / incremental add no longer reshuffle the map, unlike the throwaway fit in
    :func:`run_clustering`). Dispatches on ``config.projection_method``:
      - ``"umap"``    — same n_neighbors clamp / min_dist / metric / seed as :func:`reduce_umap` at
        ``n_components=2`` (so ``fit_projection(umap)`` is pinned-equivalent to ``reduce_umap(2)``).
      - ``"densmap"`` — UMAP + ``densmap=True`` (density-preserving), via umap-learn (already a dep).
        Fits a stable layout but does NOT support out-of-sample ``.transform`` (umap-learn limitation) —
        so densMAP serves a stable re-layout (``mode="refit"``), not incremental new-point placement.
      - ``"pacmap"``  — lazy import; raises :class:`ProjectionMethodUnavailable` when the optional
        package is absent (no crash, no silent fallback).
    Tiny inputs (``n < max(config.umap_n_neighbors, 4)``, matching :func:`run_clustering`) skip UMAP and
    return the PCA/first-dims fallback with an identity reducer. Deterministic for a fixed ``config.seed``.
    """
    X = ensure_finite_2d(X, name="X")
    method = getattr(config, "projection_method", "umap")
    n_samples = X.shape[0]

    if n_samples < max(config.umap_n_neighbors, 4):
        return _coords2d_fallback(X, config), _IdentityReducer(config)

    n_neighbors = max(2, min(config.umap_n_neighbors, n_samples - 1))
    if method in ("umap", "densmap"):
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_state=config.seed,
            densmap=(method == "densmap"),
        )
    elif method == "pacmap":
        try:
            import pacmap
        except (
            ImportError
        ) as exc:  # optional dep (numba wheels) — clear error, never a crash/fallback
            raise ProjectionMethodUnavailable(method) from exc

        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=config.seed)
    else:
        raise ValueError(
            f"unknown projection_method {method!r}; expected 'umap', 'densmap' or 'pacmap'"
        )

    coords = reducer.fit_transform(X)
    return np.asarray(coords, dtype=np.float32), reducer


def transform_projection(reducer: Reducer2D, X_new: np.ndarray) -> np.ndarray:
    """Place new points ``X_new`` (m, d) into an EXISTING layout via ``reducer.transform`` — ``(m, 2)``
    float32, in ms (no refit). Deterministic for a fixed ``random_state``; PaCMAP's ``transform`` needs the
    original fitted reducer (documented). Empty input → ``(0, 2)``."""
    X_new = np.asarray(X_new, dtype=np.float32)
    if X_new.ndim != 2:
        raise ValueError(f"X_new must be 2-D (m, d); got shape {X_new.shape}")
    if X_new.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(reducer.transform(X_new), dtype=np.float32)


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> "float | None":
    """Procrustes disparity M² ∈ [0, 1] between two ROW-ALIGNED point sets (0 = identical layout up to
    translation / rotation / uniform scale) — the map-drift metric. Returns ``None`` on < 2 points
    or incompatible/degenerate shapes; NEVER raises (a drift number is diagnostic, not load-bearing)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[0] < 2:
        return None
    try:
        from scipy.spatial import procrustes

        _, _, disparity = procrustes(A, B)
        return float(disparity)
    except Exception:  # noqa: BLE001 — degenerate input (e.g. a constant column) must not raise
        return None


def cluster_hdbscan(X: np.ndarray, config: Config) -> np.ndarray:
    """Cluster ``X`` with HDBSCAN; ``-1`` marks noise.

    Uses ``config.hdbscan_min_cluster_size`` and ``config.hdbscan_min_samples``.
    Prefers the standalone ``hdbscan`` package and falls back to
    ``sklearn.cluster.HDBSCAN`` when it is unavailable. Returns integer labels
    of shape (n,).
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    # Both HDBSCAN backends raise on a single sample (their nearest-neighbour
    # queries need >1 point). A lone point can never form a cluster of size >= 2,
    # so it is noise by definition; return all-noise without touching the backend.
    if n_samples < 2:
        return np.full(n_samples, -1, dtype=int)
    # min_cluster_size must be >= 2 and cannot exceed the sample count.
    min_cluster_size = max(2, min(config.hdbscan_min_cluster_size, n_samples))
    try:
        import hdbscan as _hdbscan

        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=config.hdbscan_min_samples,
        )
    except ImportError:
        from sklearn.cluster import HDBSCAN

        # The two backends give close but NOT identical partitions — say which one ran.
        logger.info("hdbscan package unavailable; falling back to sklearn.cluster.HDBSCAN")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=config.hdbscan_min_samples,
        )
    labels = clusterer.fit_predict(X)
    return np.asarray(labels, dtype=int)


def cluster_kmeans(X: np.ndarray, n_clusters: int, config: Config) -> np.ndarray:
    """Cluster ``X`` with scikit-learn KMeans. Returns labels of shape (n,)."""
    X = np.asarray(X, dtype=np.float32)
    # KMeans needs at least as many samples as clusters.
    k = max(1, min(n_clusters, X.shape[0]))
    km = KMeans(n_clusters=k, random_state=config.seed, n_init=10)
    labels = km.fit_predict(X)
    return np.asarray(labels, dtype=int)


def _kpp_cosine_init(Xn: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """k-means++ seeding on cosine distance ``1 - cos`` over pre-normalized rows ``Xn``.

    Deterministic given ``rng``. Picks the first center uniformly, then each subsequent center with
    probability proportional to its cosine distance to the nearest chosen center (D-weighting on the
    bounded [0, 2] cosine distance spreads seeds well on the unit sphere). Already-chosen points have
    distance 0 → probability 0, so the ``k`` returned row indices are distinct."""
    n = Xn.shape[0]
    first = int(rng.integers(n))
    idx = [first]
    if k <= 1:
        return idx
    closest_sim = Xn @ Xn[first]  # (n,) cosine to the one chosen center
    for _ in range(1, k):
        dist = np.clip(1.0 - closest_sim, 0.0, None)
        total = float(dist.sum())
        if total <= 0.0:  # every point coincides with a chosen center — take any unused index
            remaining = [i for i in range(n) if i not in idx]
            nxt = remaining[0] if remaining else idx[-1]
        else:
            nxt = int(rng.choice(n, p=dist / total))
        idx.append(nxt)
        closest_sim = np.maximum(closest_sim, Xn @ Xn[nxt])
    return idx


def cluster_spherical_kmeans(
    X: np.ndarray, n_clusters: int, config: Config, max_iter: int = 100
) -> np.ndarray:
    """Spherical k-means: Lloyd's algorithm on COSINE similarity. Returns labels (n,), never ``-1``.

    Unlike euclidean :func:`cluster_kmeans`, this matches the geometry of L2-normalized CLAP
    embeddings: rows are normalized to the unit sphere, assignment is ``argmax`` of the cosine
    similarity to unit centroids, and each centroid is the L2-renormalized mean of its members. ``k``
    is clamped to ``max(1, min(n_clusters, n))``; init is deterministic (k-means++ on cosine, seeded
    by ``config.seed``), so identical inputs give identical labels. Assignment is total (every point
    gets a cluster — no noise label). Pure numpy."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=int)
    k = max(1, min(int(n_clusters), n))
    Xn = l2_normalize(X, axis=1)
    rng = np.random.default_rng(config.seed)
    C = Xn[_kpp_cosine_init(Xn, k, rng)].copy()  # (k, d) unit-norm centroids

    labels = np.zeros(n, dtype=int)
    for _ in range(int(max_iter)):
        new_labels = np.argmax(Xn @ C.T, axis=1).astype(int)  # cosine assignment
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = Xn[labels == j]
            if members.shape[0] == 0:
                continue  # keep the (distinct) init center for an emptied cluster
            mean = members.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                C[j] = mean / norm
    return labels


def _knn_igraph(X: np.ndarray, config: Config):
    """Build a symmetric cosine-similarity kNN graph over the rows of ``X`` as an ``igraph.Graph``.

    Shared by :func:`cluster_leiden` and :func:`graph_modularity` so both see the *same* graph (same
    ``config.leiden_k_neighbors`` degree, L2-normalized rows, cosine metric, ``1 - distance`` similarity
    weights). Rows are L2-normalized; each point's ``k = max(2, min(leiden_k_neighbors, n-1))`` nearest
    neighbours contribute an edge weighted by cosine similarity (floored to a tiny positive so a
    non-attracting neighbour never yields a non-positive weight). The directed kNN relation is
    symmetrized with an element-wise max (undirected graph). Lazy ``igraph`` import (optional backend).
    Returns ``(graph, edge_weights)`` where ``edge_weights`` is aligned to the graph's edge order."""
    import igraph as ig
    from sklearn.neighbors import kneighbors_graph

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    Xn = l2_normalize(X, axis=1)
    # kneighbors_graph needs n_neighbors < n (self excluded); floor at 1 so a 2-row input yields k=1
    # (its single neighbour) instead of an out-of-range k=2 crash — same tiny-input robustness the other
    # methods give. Callers guarantee n >= 2 (cluster_leiden returns early for n < 2), so n-1 >= 1.
    k = max(1, min(int(config.leiden_k_neighbors), n - 1))
    A = kneighbors_graph(Xn, n_neighbors=k, mode="distance", metric="cosine").tocsr()
    # Cosine distance (1 - cos) on the stored kNN entries → cosine-similarity weight, floored positive.
    A.data = np.clip(1.0 - A.data, 1e-6, None)
    A = A.maximum(
        A.T
    )  # symmetrize: undirected edge keeps the larger of the two directed similarities
    coo = A.tocoo()
    upper = coo.row < coo.col  # each undirected edge exactly once
    edges = list(zip(coo.row[upper].tolist(), coo.col[upper].tolist()))
    weights = coo.data[upper].astype(float).tolist()
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights
    return g, weights


def cluster_leiden(X: np.ndarray, config: Config) -> np.ndarray:
    """Leiden community detection on a cosine-kNN graph of ``X`` (Traag, Waltman & van Eck 2019).

    No imposed ``k``: granularity is driven by ``config.leiden_resolution`` on the
    ``RBConfigurationVertexPartition`` (higher → more, smaller communities). Deterministic for
    ``config.seed``. Labels are relabelled contiguous ``0..K-1`` with **no** ``-1`` (Leiden partitions
    the whole graph). ``n < 2`` → all-zero. ``leidenalg``/``python-igraph`` are OPTIONAL and imported
    lazily; their absence raises :class:`~moodengine.exceptions.MissingDependencyError` naming the
    ``cluster-graph`` extra (never a fabricated clustering), so the baseline methods stay usable
    without the package."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    try:
        import leidenalg
        import igraph  # noqa: F401 — the graph backend must be present too
    except ImportError as exc:
        raise MissingDependencyError(
            "leiden clustering", "leidenalg + python-igraph", "cluster-graph"
        ) from exc

    g, _ = _knn_igraph(X, config)
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(config.leiden_resolution),
        seed=int(config.seed),
    )
    membership = np.asarray(part.membership, dtype=int)
    # Relabel to contiguous 0..K-1 (defensive; leidenalg is already 0-based) — never a -1.
    _, contiguous = np.unique(membership, return_inverse=True)
    return contiguous.astype(int)


def graph_modularity(X: np.ndarray, labels: np.ndarray, config: Config) -> float | None:
    """Modularity of the partition ``labels`` on the SAME cosine-kNN graph construction as
    :func:`cluster_leiden` (via :func:`_knn_igraph`) — a real, measured community-quality number in
    ``[-0.5, 1.0]``. ``None`` when there are fewer than 2 clusters, fewer than 2 rows, or the optional
    ``igraph`` backend is absent. Never raises on degenerate input (a quality number is diagnostic)."""
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    n = X.shape[0]
    n_clusters = len({lbl for lbl in labels.tolist() if lbl != -1})
    if n < 2 or n_clusters < 2:
        return None
    try:
        g, weights = _knn_igraph(X, config)
        return float(g.modularity(labels.tolist(), weights=weights))
    except Exception:  # noqa: BLE001 — missing backend / degenerate graph must not raise
        return None


def cluster_hierarchy(
    X: np.ndarray, labels: np.ndarray, config: Config, n_super_groups: int | None = None
) -> dict:
    """Agglomerative hierarchy over the cluster MEDOIDS → super-groups for the map's semantic zoom.

    Builds the medoid matrix ``M`` (K, d) via :func:`cluster_medoids` (ordered by ascending cluster id,
    noise ``-1`` excluded), runs scipy ``linkage(M, method="average", metric="cosine")``, and cuts it
    into ``n_super_groups`` (default ``max(2, round(sqrt(K)))`` clamped to ``[2, K]``). Returns a
    JSON-serializable dict (native Python types) — the whole payload is persisted in ``runs.metrics_json``
    (no new table). ``cophenetic`` is the cophenetic correlation (defensively ``None`` for ``K < 3``).
    ``K <= 1`` → an honest empty structure (``linkage=[]``, ``cophenetic=None``) without raising."""
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    medoids = cluster_medoids(X, labels)  # {cluster_id: row_index}, excludes -1
    cluster_ids = sorted(medoids.keys())
    K = len(cluster_ids)

    if K <= 1:
        return {
            "linkage": [],
            "cluster_ids": [int(c) for c in cluster_ids],
            "cophenetic": None,
            "linkage_method": "average",
            "metric": "cosine",
            "n_super_groups": K,
            "super_group_of": {str(int(c)): 0 for c in cluster_ids},
        }

    M = np.stack([X[medoids[cid]] for cid in cluster_ids]).astype(
        np.float32
    )  # (K, d) medoid vectors
    Z = linkage(M, method="average", metric="cosine")

    g = max(2, round(float(np.sqrt(K)))) if n_super_groups is None else int(n_super_groups)
    g = max(2, min(g, K))  # clamp requested super-group count to [2, K]
    cut = fcluster(
        Z, g, criterion="maxclust"
    )  # 1-based group id per cluster (in cluster_ids order)
    remap = {old: i for i, old in enumerate(sorted(set(cut.tolist())))}  # → 0-based contiguous
    super_group_of = {str(int(cid)): int(remap[int(cut[i])]) for i, cid in enumerate(cluster_ids)}
    n_super = len(
        remap
    )  # actual distinct super-groups (honest count; each is non-empty by construction)

    cophenetic: float | None = None
    if K >= 3:
        try:
            corr, _ = cophenet(Z, pdist(M, metric="cosine"))
            cophenetic = float(corr)
        except Exception:  # noqa: BLE001 — a degenerate distance matrix must not raise
            cophenetic = None

    linkage_rows = [[int(a), int(b), float(dist), int(cnt)] for a, b, dist, cnt in Z]
    return {
        "linkage": linkage_rows,
        "cluster_ids": [int(c) for c in cluster_ids],
        "cophenetic": cophenetic,
        "linkage_method": "average",
        "metric": "cosine",
        "n_super_groups": int(n_super),
        "super_group_of": super_group_of,
    }


def select_kmeans_k(
    X: np.ndarray, config: Config, k_min: int = 2, k_max: int = 12
) -> tuple[int, dict[int, float]]:
    """Pick the KMeans ``k`` that maximizes silhouette on ``X``.

    Sweeps ``k`` in ``[k_min, min(k_max, n_samples - 1)]`` with
    ``KMeans(k, random_state=config.seed, n_init=10)`` and scores each by
    silhouette. Returns ``(best_k, {k: silhouette})``. Falls back to ``(1, {})``
    for fewer than 3 samples (silhouette is undefined). Deterministic.
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]
    if n_samples < 3:
        return 1, {}
    hi = min(int(k_max), n_samples - 1)
    lo = max(2, int(k_min))
    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        labels = cluster_kmeans(X, k, config)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            # Silhouette is O(n²·d): sample it above 2k points — the k RANKING it
            # drives is stable under sampling, and the full pass costs minutes at
            # 10k tracks for no better decision. Seeded, so selection stays
            # deterministic.
            scores[k] = float(
                silhouette_score(
                    X, labels, sample_size=min(n_samples, 2000), random_state=config.seed
                )
            )
        except Exception:
            continue
    if not scores:
        return min(lo, max(1, n_samples)), {}
    best_k = max(scores, key=lambda k: scores[k])
    return best_k, scores


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> ClusterMetrics:
    """Summarize a clustering. Never raises on degenerate inputs.

    Returns ``{'n_clusters', 'noise_ratio', 'cluster_sizes', 'silhouette'}``
    where ``n_clusters`` excludes the noise label (-1), ``cluster_sizes`` maps
    every label (incl. -1) to its count, and ``silhouette`` is computed on
    non-noise points only when there are >= 2 clusters and >= 2 non-noise
    samples, else ``None``.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique, counts)}

    non_noise_labels = unique[unique != -1]
    n_clusters = int(non_noise_labels.size)

    n = int(labels.size)
    noise_count = cluster_sizes.get(-1, 0)
    noise_ratio = float(noise_count / n) if n > 0 else 0.0

    silhouette: float | None = None
    non_noise_mask = labels != -1
    n_non_noise = int(non_noise_mask.sum())
    if n_clusters >= 2 and n_non_noise >= 2:
        try:
            silhouette = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
        except Exception:  # never raise on degenerate input
            silhouette = None

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "cluster_sizes": cluster_sizes,
        "silhouette": silhouette,
    }


def coverage_entropy(labels: np.ndarray, *, base: float = 2.0) -> CoverageEntropyResult:
    """Shannon entropy of the cluster-occupancy distribution — how evenly the library spreads across
    its vibe regions (the "70 % of your library in 3 regions" diversity read).

    Treats every label as a bin, INCLUDING the HDBSCAN noise label ``-1`` (an honest "unclassified"
    region, not dropped). Returns ``{'entropy', 'normalized_entropy', 'perplexity', 'n_bins',
    'shares'}`` where ``entropy = -Σ p·log_base(p)`` over the region shares ``p``,
    ``normalized_entropy = entropy / log_base(n_bins)`` in ``[0, 1]`` (``1.0`` for a perfectly uniform
    spread, ``0.0`` for a single region; defined as ``1.0`` when ``n_bins <= 1``), ``perplexity =
    base ** entropy`` (the effective number of equally-occupied regions), and ``shares`` maps each
    label to its fraction of the library. Never raises: empty ``labels`` → all zeros / empty shares.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.size)
    if n == 0:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "perplexity": 0.0,
            "n_bins": 0,
            "shares": {},
        }
    unique, counts = np.unique(labels, return_counts=True)
    shares = {int(lbl): float(cnt / n) for lbl, cnt in zip(unique, counts)}
    n_bins = int(unique.size)
    p = counts / n
    entropy = float(-(p * (np.log(p) / np.log(base))).sum())  # base-`base` Shannon entropy
    denom = np.log(n_bins) / np.log(base) if n_bins > 1 else 0.0
    # Clamp to [0, 1]: a perfectly uniform occupancy gives entropy == log_base(n_bins) exactly in theory,
    # but float rounding of the division can land at 1.0000000000000002 — mathematically the normalized
    # entropy is bounded by 1, so this keeps a downstream ``<= 1.0`` contract honest, not fabricated.
    normalized = float(np.clip(entropy / denom, 0.0, 1.0)) if denom > 0 else 1.0
    perplexity = float(base**entropy)
    return {
        "entropy": entropy,
        "normalized_entropy": normalized,
        "perplexity": perplexity,
        "n_bins": n_bins,
        "shares": shares,
    }


def _coords2d_fallback(X: np.ndarray, config: Config) -> np.ndarray:
    """2-D coordinates for tiny inputs: PCA when possible, else padded raw dims."""
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    if n_samples >= 2 and n_features >= 2:
        try:
            coords = PCA(n_components=2, random_state=config.seed).fit_transform(X)
            return np.asarray(coords, dtype=np.float32)
        except Exception:
            pass
    # Fall back to first two dimensions, zero-padded as needed.
    coords = np.zeros((n_samples, 2), dtype=np.float32)
    take = min(2, n_features)
    coords[:, :take] = X[:, :take]
    return coords


def _ensure_cluster_method(method: str) -> None:
    """Runtime guard behind :data:`ClusterMethod` — one message for every entry point."""
    if method not in get_args(ClusterMethod):
        raise ValueError(
            f"method must be 'hdbscan', 'kmeans', 'spherical' or 'leiden'; got {method!r}"
        )


def _cluster_with(method: ClusterMethod, X: np.ndarray, config: Config) -> np.ndarray:
    """The single place mapping a method name to its clustering backend."""
    _ensure_cluster_method(method)
    if method == "hdbscan":
        return cluster_hdbscan(X, config)
    if method == "spherical":
        return cluster_spherical_kmeans(X, config.kmeans_n_clusters, config)
    if method == "leiden":
        return cluster_leiden(X, config)
    return cluster_kmeans(X, config.kmeans_n_clusters, config)


def run_clustering(X: np.ndarray, method: ClusterMethod, config: Config) -> ClusteringResult:
    """Reduce + cluster + summarize, robust to tiny POC inputs.

    ``method`` is ``'hdbscan'``, ``'kmeans'``, ``'spherical'`` or ``'leiden'`` (the last needs
    the optional leidenalg+igraph backend). Embeddings are reduced to
    ``config.umap_n_components_cluster`` dims for clustering and a separate 2-D
    UMAP (``config.umap_n_components_viz``) provides ``coords2d``. When
    ``n_samples < max(config.umap_n_neighbors, 4)`` UMAP is skipped: clustering
    runs on ``X`` directly and ``coords2d`` falls back to PCA / first-2-dims —
    logged, and recorded as ``metrics['reduction'] == 'none_tiny_input'``
    (``'umap'`` on the normal path) so the switch is visible in the result, not
    just in the docstring. ``X`` must be finite (a NaN/Inf row raises
    ``ValueError`` naming the offending rows, instead of a deep UMAP/sklearn
    error later).

    Returns ``{'labels': (n,), 'coords2d': (n, 2), 'metrics': dict,
    'method': method}``.
    """
    X = ensure_finite_2d(X, name="X")
    n_samples = X.shape[0]

    _ensure_cluster_method(method)

    tiny = n_samples < max(config.umap_n_neighbors, 4)

    if tiny:
        # Skip UMAP entirely; cluster on the raw vectors.
        logger.info(
            "n_samples=%d < %d: skipping UMAP, clustering raw vectors (coords2d via PCA fallback)",
            n_samples,
            max(config.umap_n_neighbors, 4),
        )
        cluster_input = X
        coords2d = _coords2d_fallback(X, config)
    else:
        cluster_input, _ = reduce_umap(X, config.umap_n_components_cluster, config)
        viz_emb, _ = reduce_umap(X, config.umap_n_components_viz, config)
        coords2d = np.asarray(viz_emb, dtype=np.float32)
        # Guard: ensure exactly 2 viz dims regardless of config override.
        if coords2d.shape[1] != 2:
            coords2d = _coords2d_fallback(coords2d, config)

    labels = _cluster_with(method, cluster_input, config)

    metrics = cluster_metrics(cluster_input, labels)
    metrics["reduction"] = "none_tiny_input" if tiny else "umap"

    return {
        "labels": np.asarray(labels, dtype=int),
        "coords2d": np.asarray(coords2d, dtype=np.float32),
        "metrics": metrics,
        "method": method,
    }


def silhouette_original(X: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float | None:
    """Silhouette score on the ORIGINAL (pre-UMAP) embedding space.

    Computed over non-noise points only (label != -1) using ``metric`` (cosine
    by default). Requires >= 2 distinct non-noise clusters and >= 2 non-noise
    samples, else returns ``None``. Never raises on degenerate input.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    mask = labels != -1
    X_nn = X[mask]
    lab_nn = labels[mask]
    if lab_nn.size < 2 or len(set(lab_nn.tolist())) < 2:
        return None
    try:
        return float(silhouette_score(X_nn, lab_nn, metric=metric))
    except Exception:  # never raise on degenerate input
        return None


def cluster_medoids(X: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Medoid (representative) row index per cluster (excluding noise ``-1``).

    The medoid of a cluster is the member maximizing mean cosine similarity to
    its clustermates. Returns ``{cluster_id: row_index}`` over the original
    rows of ``X``. Never raises.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    Xn = l2_normalize(X, axis=1)
    medoids: dict[int, int] = {}
    for lbl in sorted(set(labels.tolist())):
        if lbl == -1:
            continue
        member_idx = np.flatnonzero(labels == lbl)
        if member_idx.size == 0:
            continue
        if member_idx.size == 1:
            medoids[int(lbl)] = int(member_idx[0])
            continue
        members = Xn[member_idx]
        # Mean cosine to clustermates (self excluded) WITHOUT the (m, m) similarity
        # matrix: row_sums == members @ members.sum(axis=0) minus each row's
        # self-cosine (‖row‖² — 1.0 for unit rows, 0.0 for the zero-vector guard).
        # O(m·d) memory instead of O(m²), which matters for the one dominant
        # cluster HDBSCAN often yields on a homogeneous library.
        total = members.sum(axis=0)
        self_sim = np.einsum("ij,ij->i", members, members)
        mean_sims = (members @ total - self_sim) / (member_idx.size - 1)
        medoids[int(lbl)] = int(member_idx[int(np.argmax(mean_sims))])
    return medoids


def per_cluster_silhouette(
    X: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> dict[int, float | None]:
    """Mean silhouette PER cluster (excluding noise ``-1``) — the separability of each emergent region.

    Computes ``silhouette_samples`` on the non-noise points and averages per cluster. A cluster is
    ``None`` (not calculable) when there are fewer than 2 non-noise clusters / samples, or that cluster
    has fewer than 2 members (a singleton's silhouette is not meaningful). Never raises on degenerate
    input (style of :func:`silhouette_original`). Deterministic; pure sklearn."""
    from sklearn.metrics import silhouette_samples

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    mask = labels != -1
    X_nn = X[mask]
    lab_nn = labels[mask]
    uniq = sorted(set(lab_nn.tolist()))
    if lab_nn.size < 2 or len(uniq) < 2:
        return {int(c): None for c in uniq}
    try:
        samples = silhouette_samples(X_nn, lab_nn, metric=metric)
    except Exception:  # noqa: BLE001 — degenerate geometry must not raise
        return {int(c): None for c in uniq}
    out: dict[int, float | None] = {}
    for c in uniq:
        members = lab_nn == c
        out[int(c)] = float(samples[members].mean()) if int(members.sum()) >= 2 else None
    return out


def sub_cluster(X: np.ndarray, config: Config, k_min: int = 2, k_max: int = 6) -> SubClusterResult:
    """Sub-cluster a SUBSET ``X`` (the members of one parent mood) into hierarchical sub-moods.

    Picks the sub-``k`` by silhouette (:func:`select_kmeans_k`), clusters (:func:`cluster_kmeans`), and
    returns ``{'sub_labels': (m,) int, 'sub_k': int, 'silhouette': float|None, 'medoids': {sub_id:
    row_idx}, 'per_cluster_silhouette': {sub_id: float|None}}`` — all indices/labels LOCAL to ``X``.
    Assignment is total (every member gets exactly one ``sub_id`` — an EXACT partition of the parent).
    Degenerate: ``m < 3`` → ``sub_k = 1`` (one sub-mood, all labels 0); ``m == 0`` → empty. Deterministic
    for ``config.seed``. numpy/sklearn only (torch-free)."""
    X = np.asarray(X, dtype=np.float32)
    m = X.shape[0]
    if m < 3:
        labels = np.zeros(m, dtype=int)
        return {
            "sub_labels": labels,
            "sub_k": 1 if m > 0 else 0,
            "silhouette": None,
            "medoids": cluster_medoids(X, labels),
            "per_cluster_silhouette": per_cluster_silhouette(X, labels, metric="cosine"),
        }
    best_k, _ = select_kmeans_k(X, config, k_min=k_min, k_max=k_max)
    labels = cluster_kmeans(X, max(1, int(best_k)), config)
    return {
        "sub_labels": labels,
        "sub_k": int(len(set(labels.tolist()))),
        "silhouette": silhouette_original(X, labels, metric="cosine"),
        "medoids": cluster_medoids(X, labels),
        "per_cluster_silhouette": per_cluster_silhouette(X, labels, metric="cosine"),
    }


def outlier_scores(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-track outlier score ``1 - cosine(track, its cluster centroid)``.

    The cluster centroid is the L2-normalized mean of its (clustermate) members.
    Noise points (label ``-1``) score ``1.0``. Returns shape ``(n,)``. Never
    raises.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    n = X.shape[0]
    scores = np.ones(n, dtype=np.float32)
    Xn = l2_normalize(X, axis=1)
    for lbl in set(labels.tolist()):
        if lbl == -1:
            continue
        member_idx = np.flatnonzero(labels == lbl)
        if member_idx.size == 0:
            continue
        centroid = Xn[member_idx].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        cos = Xn[member_idx] @ centroid
        scores[member_idx] = (1.0 - cos).astype(np.float32)
    return scores


def bootstrap_stability(
    X: np.ndarray,
    method: ClusterMethod,
    config: Config,
    n_boot: int | None = None,
    subsample: float = 0.8,
) -> StabilityMetrics:
    """Clustering stability via subsample-and-recluster bootstrap, in the deployed space.

    Reduces ``X`` ONCE to ``config.umap_n_components_cluster`` dims — the exact space
    :func:`run_clustering` clusters in — then repeatedly subsamples ``subsample`` of the
    reduced rows (seeded by ``config.seed + i``) and runs the SAME ``method`` clustering
    (``'kmeans'``, ``'hdbscan'``, ``'spherical'`` or ``'leiden'``) on each subsample.
    Two things make the number trustworthy where a raw-space bootstrap would mislead:

    * **Same space as production.** Clustering the reduced embedding (not the raw
      high-dim matrix) measures the stability of the partition the pipeline actually
      ships. Fitting UMAP once and subsampling its rows — rather than re-running UMAP per
      bootstrap — keeps the cost bounded and isolates clustering variance from UMAP's own
      run-to-run noise. Tiny inputs (``n < max(config.umap_n_neighbors, 4)``) skip UMAP
      and cluster the raw rows, exactly as :func:`run_clustering` does.
    * **Noise excluded from the shape scores.** Agreement (adjusted Rand / adjusted
      mutual information) is computed only over points that are non-noise (label ``!= -1``)
      in BOTH runs, so two HDBSCAN runs that merely agree on which points are noise do not
      inflate ARI. Agreement on the noise / non-noise split itself is reported separately
      as ``mean_noise_agreement`` (always ``1.0`` for methods that never emit ``-1``).

    ``n_boot`` defaults to ``config.bootstrap_n``. Returns ``{'mean_ari', 'std_ari',
    'mean_ami', 'mean_noise_agreement', 'n_boot'}``; degenerate inputs yield zeros.
    Deterministic given ``config.seed``.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n_boot is None:
        n_boot = int(getattr(config, "bootstrap_n", 50))
    zeros: StabilityMetrics = {
        "mean_ari": 0.0,
        "std_ari": 0.0,
        "mean_ami": 0.0,
        "mean_noise_agreement": 0.0,
        "n_boot": int(n_boot),
    }
    _ensure_cluster_method(method)
    if n < 4 or n_boot < 2:
        return zeros

    # Cluster the same reduced space run_clustering deploys; skip UMAP on tiny inputs.
    if n < max(config.umap_n_neighbors, 4):
        space = X
    else:
        space, _ = reduce_umap(X, config.umap_n_components_cluster, config)

    size = max(2, int(round(subsample * n)))
    size = min(size, n)

    runs: list[np.ndarray] = []  # full-length label arrays, -2 = not sampled
    for i in range(int(n_boot)):
        rng = np.random.default_rng(config.seed + i)
        idx = rng.choice(n, size=size, replace=False)
        # leiden clamps its kNN degree to the subsample size internally, so
        # every bootstrap graph stays valid.
        sub_labels = _cluster_with(method, space[idx], config)
        full = np.full(n, -2, dtype=int)
        full[idx] = sub_labels
        runs.append(full)

    aris: list[float] = []
    amis: list[float] = []
    noise_agrees: list[float] = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            shared = (runs[a] != -2) & (runs[b] != -2)
            if int(shared.sum()) < 2:
                continue
            la_all, lb_all = runs[a][shared], runs[b][shared]
            noise_agrees.append(float(np.mean((la_all == -1) == (lb_all == -1))))

            both = (la_all != -1) & (lb_all != -1)
            if int(both.sum()) < 2:
                continue
            la, lb = la_all[both], lb_all[both]
            try:
                aris.append(float(adjusted_rand_score(la, lb)))
                amis.append(float(adjusted_mutual_info_score(la, lb)))
            except Exception:
                continue

    mean_noise = float(np.mean(noise_agrees)) if noise_agrees else 0.0
    if not aris:
        # No pair shared >=2 co-clustered non-noise points (e.g. an all-noise partition):
        # cluster shape is unstable, but still surface the noise-split agreement measured.
        return {**zeros, "mean_noise_agreement": mean_noise}
    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "mean_ami": float(np.mean(amis)),
        "mean_noise_agreement": mean_noise,
        "n_boot": int(n_boot),
    }
