"""Novelty / out-of-distribution scoring on frozen CLAP embeddings (pure numpy, torch-free).

Where :func:`moodengine.cluster.outlier_scores` measures a *cluster-local* anomaly (``1 - cos`` to a
track's own cluster centroid), these score novelty *globally* against the whole library distribution —
"is this an authentic discovery, or more of the same?":

  * :func:`mahalanobis_scores` — distance to the global mean under a **shrinkage** covariance
    (Ledoit-Wolf-style structural shrinkage, Ledoit & Wolf 2004), the classic parametric OOD detector
    on deep features (Lee et al., NeurIPS 2018).
  * :func:`knn_distance_scores` — mean cosine distance to the ``k`` nearest neighbours, the
    non-parametric deep-kNN OOD detector (Sun et al., ICML 2022).

Both are deterministic and never raise on degenerate input. They read the same L2-normalized CLAP
matrix the triptych and the exact-cosine kNN already use; nothing is fabricated.
"""

from __future__ import annotations

import numpy as np

from moodengine._math import l2_normalize
from moodengine._validation import ensure_finite_2d


def mahalanobis_scores(
    X: np.ndarray, *, ref: np.ndarray | None = None, shrinkage: float = 0.10
) -> np.ndarray:
    """Mahalanobis distance of each row of ``X`` to the mean of ``ref`` under a shrinkage covariance.

    The covariance is estimated on ``ref`` (default ``ref = X``) and regularized toward a scaled
    identity — ``Σ̂ = (1−α)·S + α·(tr(S)/d)·I`` with ``α = shrinkage`` (Ledoit & Wolf 2004 structural
    shrinkage) — so it is well-conditioned even when ``d`` (512 for CLAP) exceeds the sample count.
    The per-row distance is ``sqrt((x−μ)ᵀ Σ̂⁻¹ (x−μ))``, solved via ``np.linalg.solve`` (falling back to
    the pseudo-inverse if ``Σ̂`` is singular). A higher score = further from the bulk of the library =
    more novel; it is the parametric complement of :func:`knn_distance_scores`.

    ``X`` ``(n, d)`` are the (L2-normalized) CLAP embeddings. Returns ``(n,)`` float32, ``>= 0``,
    deterministic. Guards degenerate input: fewer than 2 reference rows, an empty ``X``, or a zero-width
    matrix yield all-zeros (no distribution to score against — never a fabricated number). Inputs are
    not mutated.
    """
    X = np.asarray(X, dtype=np.float32)
    R = X if ref is None else np.asarray(ref, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    n_ref, d = (R.shape[0], R.shape[1]) if R.ndim == 2 else (0, 0)
    if n_ref < 2 or d == 0 or X.shape[1] != d:
        return np.zeros((X.shape[0],), dtype=np.float32)
    # Degenerate SIZES yield zeros (above); non-finite DATA raises — a NaN row would
    # otherwise poison the covariance and silently NaN every score.
    X = ensure_finite_2d(X, name="X")
    R = ensure_finite_2d(R, name="ref")

    mu = R.mean(axis=0)
    S = np.cov(R, rowvar=False).astype(np.float64)  # (d, d) unbiased sample covariance
    S = np.atleast_2d(S)
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    target = (np.trace(S) / d) * np.eye(d, dtype=np.float64)
    sigma = (1.0 - alpha) * S + alpha * target  # shrinkage estimator, well-conditioned

    D = (X - mu).astype(np.float64)  # (n, d)
    try:
        M = np.linalg.solve(sigma, D.T)  # (d, n) = Σ̂⁻¹ Dᵀ
    except np.linalg.LinAlgError:
        M = np.linalg.pinv(sigma) @ D.T
    dist2 = np.einsum("ij,ji->i", D, M)  # (n,) squared Mahalanobis
    return np.sqrt(np.maximum(dist2, 0.0)).astype(np.float32)


# Rows per cosine slab in knn_distance_scores: the peak allocation is one
# (block, m) float32 slab ≈ 40 MB at m = 10k — a full (n, m) block would OOM around
# 10-15k tracks on a 16 GB machine. On the wide path the select no longer runs in place on
# that slab: the chunk-max prefilter below adds a (block, kk·chunk) float32 gather and its
# intp index array on top of it — ~3.9 MB against a 40 MB slab at block=1024, kk=10,
# chunk=32, m=10k, so the slab still sets the peak. The narrow path keeps the in-place
# partition and stays exactly one slab.
_KNN_BLOCK_ROWS = 1024

#: Column width of one prefilter chunk. Picking the top-kk CHUNKS by their maximum reduces the
#: top-kk select from m columns to kk·32, which is where the win comes from at library scale.
#: Measured across widths on n=10000/d=512/k=10: 32 -> 2.67x, 64 -> 2.11x, 128 -> 1.42x,
#: 256 -> 1.01x. Narrower than 32 spends more on the chunk-max reduction than it saves.
_KNN_PREFILTER_CHUNK = 32


def _top_k_per_row(sims: np.ndarray, kk: int) -> np.ndarray:
    """The ``kk`` largest values of each row of ``sims``, as an unordered ``(rows, kk)`` block.

    Selecting over all ``m`` columns costs O(m) per row. Chunking first costs far less and is
    exact, by disjointness: split the columns into fixed-width chunks, take each chunk's maximum,
    and keep the ``kk`` chunks with the largest maxima. Those ``kk`` maxima are ``kk`` DISTINCT
    elements (the chunks share no column), and every element in a dropped chunk is ``<=`` its own
    chunk maximum ``<=`` the smallest kept one. So no dropped element can outrank all ``kk`` of
    them: at best it ties, and swapping a tie leaves the selected MULTISET — and therefore the
    mean this feeds — unchanged. Disjointness is what carries the argument, which is why the
    ragged tail is appended to the candidates unconditionally instead of being covered by a
    clamped final chunk: an overlapping chunk could make two chunk maxima the same physical
    element, and the count of distinct dominators would no longer be ``kk``.

    Falls back to the in-place full-width partition when chunking would not narrow the search --
    few chunks relative to ``kk``, or a candidate set no smaller than half the row. ``sims`` is a
    fresh slab nobody else holds, so that path partitions it in place rather than allocating a
    second ``(rows, m)`` array.
    """
    m = sims.shape[1]
    n_chunks = m // _KNN_PREFILTER_CHUNK
    body_width = n_chunks * _KNN_PREFILTER_CHUNK
    candidate_width = kk * _KNN_PREFILTER_CHUNK + (m - body_width)
    if n_chunks <= kk or candidate_width * 2 >= m:
        sims.partition(m - kk, axis=1)
        return sims[:, -kk:]

    rows = sims.shape[0]
    # A column slice of a C-contiguous slab reshapes to (rows, n_chunks, chunk) as a strided
    # VIEW, so the chunk maxima cost one reduction and no copy of the slab.
    chunk_max = sims[:, :body_width].reshape(rows, n_chunks, _KNN_PREFILTER_CHUNK).max(axis=2)
    keep = np.argpartition(chunk_max, n_chunks - kk, axis=1)[:, -kk:]  # (rows, kk) chunk ids
    cols = keep[:, :, None] * _KNN_PREFILTER_CHUNK + np.arange(_KNN_PREFILTER_CHUNK)
    candidates = np.take_along_axis(sims, cols.reshape(rows, -1), axis=1)
    if body_width < m:
        candidates = np.concatenate([candidates, sims[:, body_width:]], axis=1)
    candidates.partition(candidates.shape[1] - kk, axis=1)
    return candidates[:, -kk:]


def knn_distance_scores(X: np.ndarray, *, k: int = 10, ref: np.ndarray | None = None) -> np.ndarray:
    """Mean cosine distance (``1 − cos``) of each row of ``X`` to its ``k`` nearest rows of ``ref``.

    The non-parametric deep-kNN OOD score (Sun et al., ICML 2022): a point far from its ``k`` nearest
    library neighbours sits in a sparse region of the embedding space (novel). ``ref`` defaults to
    ``X``; when it does, each row excludes ITSELF from its neighbour set (its own cosine 1.0 would
    otherwise dominate). The cosine block is computed in row slabs (rows re-normalized defensively):
    compute is O(n·m·d) either way, but peak memory is O(block·m), not O(n·m), so large libraries
    never materialize an (n, m) matrix. Above a width threshold the per-row top-k select runs on a
    chunk-max prefiltered candidate set rather than the full row, which adds an O(block·k·32)
    gather on top of that slab — a few MB against tens — and returns the same k values.

    ``k`` is clamped to ``[1, n_neighbours]`` where ``n_neighbours`` excludes self. Returns ``(n,)``
    float32 in ``[0, 2]`` (cosine distance range), deterministic. Guards: empty ``X``/``ref`` or no
    available neighbour yields all-zeros. Inputs are not mutated.
    """
    X = np.asarray(X, dtype=np.float32)
    is_self = ref is None
    R = X if is_self else np.asarray(ref, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or R.ndim != 2 or R.shape[0] == 0:
        return np.zeros((X.shape[0] if X.ndim == 2 else 0,), dtype=np.float32)
    # Degenerate SIZES yield zeros (above); non-finite DATA raises — a NaN row would
    # silently propagate into every neighbour distance.
    X = ensure_finite_2d(X, name="X")
    R = X if is_self else ensure_finite_2d(R, name="ref")

    Xn = l2_normalize(X, axis=1)
    # `R is X` on the self path, and normalization is idempotent — reuse it rather than paying a
    # second pass and a second (n, d) copy.
    Rn = Xn if is_self else l2_normalize(R, axis=1)
    n_avail = R.shape[0] - (1 if is_self else 0)  # self excluded from its own neighbour pool
    if n_avail < 1:
        return np.zeros((X.shape[0],), dtype=np.float32)
    kk = int(np.clip(int(k), 1, n_avail))

    mean_cos = np.empty(X.shape[0], dtype=np.float32)
    for start in range(0, Xn.shape[0], _KNN_BLOCK_ROWS):
        stop = min(start + _KNN_BLOCK_ROWS, Xn.shape[0])
        sims = Xn[start:stop] @ Rn.T  # (block, m) — the only large allocation
        if is_self:
            rows = np.arange(start, stop)
            sims[rows - start, rows] = -np.inf  # a row is never its own neighbour
        part = _top_k_per_row(sims, kk)
        # Clamp the mean cosine to [-1, 1] before converting to a distance: a duplicate / re-encode
        # makes two rows bit-identical, and float32 ``X @ X.T`` then yields a self-cosine slightly
        # ABOVE 1.0, which would make ``1 - cos`` a (physically impossible) tiny NEGATIVE distance.
        # Cosine is mathematically in [-1, 1], so this is numerical hygiene — it keeps the result in
        # the documented [0, 2] and an exact duplicate reads distance 0 (not -1e-7).
        mean_cos[start:stop] = np.clip(part.mean(axis=1), -1.0, 1.0)
    return (1.0 - mean_cos).astype(np.float32, copy=False)  # cosine DISTANCE, in [0, 2]
