"""Cosine similarity search & text-to-playlist over CLAP track embeddings.

Pure numpy — torch-free. CLAP track embeddings are L2-normalized at pooling time
(``pooling.pool_clap``), so a plain dot product *is* the cosine similarity; the
helpers here defensively re-normalize anyway so they stay correct on any input.
Callers that hold a matrix already known to be row-normalized (e.g. a long-lived
search index serving many queries) can pass ``assume_normalized=True`` to skip
that O(n·d) re-normalization on every call; with unnormalized rows the flag
silently degrades scores to plain dot products, so it is strictly opt-in.

What this buys over a naive nearest-neighbour scan:
  * **One matmul** — ``similarity_matrix`` computes the full pairwise cosine block
    in a single BLAS call instead of a Python loop.
  * **Self-exclusion** — ``find_similar`` never returns the query track itself.
  * **Zero-shot text queries** — ``search_by_text`` / ``playlist_from_text`` embed
    a free-text mood description through the same CLAP text encoder used for
    labeling, so "dreamy nocturnal" ranks tracks directly in the shared space.

Every entry point guards empty matrices and out-of-range indices, returning an
empty result rather than raising, so callers (scripts, UI) can stay simple.
"""

from __future__ import annotations

import numpy as np

from moodengine._math import l2_normalize as _l2_normalize


def similarity_matrix(X: np.ndarray, *, assume_normalized: bool = False) -> np.ndarray:
    """Full ``(n, n)`` cosine-similarity matrix for the rows of ``X``.

    ``X`` is assumed L2-normalized (CLAP track embeddings), so ``X @ X.T`` is the
    cosine block; rows are re-normalized first to be robust to any input, unless
    ``assume_normalized`` (the caller then owns that guarantee — see the module
    docstring). Returns an empty ``(0, 0)`` array when ``X`` is empty.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xn = X if assume_normalized else _l2_normalize(X, axis=1)
    return Xn @ Xn.T


# Rows per cosine slab in near_duplicate_pairs — peak allocation is (block, n)
# float32 ≈ 40 MB at n = 10k instead of the full (n, n).
_NEARDUP_BLOCK_ROWS = 1024


def near_duplicate_pairs(
    X: np.ndarray,
    filenames: list[str],
    *,
    threshold: float = 0.98,
    max_pairs: int = 500,
    assume_normalized: bool = False,
) -> list[tuple[str, str, float]]:
    """Pairs of tracks whose cosine similarity is ``>= threshold`` — likely the same recording
    (alternate masters, live vs studio, re-encodes). A free hygiene / honest-similar pass: no model,
    no network, just the upper triangle of the cosine block.

    Scans the STRICT upper triangle (``i < j``) of the cosine block (rows re-normalized defensively,
    skipped under ``assume_normalized`` — see the module docstring),
    keeps pairs at or above ``threshold``, sorts them by descending cosine, and truncates to
    ``max_pairs``. The block is computed in row slabs: compute is O(n²·d) either way, but peak memory
    is O(block·n) — around 40 MB at n = 10k — instead of the O(n²) full matrix (+ triu index arrays)
    that OOMs around 10-15k tracks. Returns ``(filename_a, filename_b, cosine)`` — never a self-pair
    (``i == j``) and never a symmetric duplicate (only ``i < j``). A very high ``threshold``
    (≈0.98–1.0) keeps this to true near-duplicates; raising ``threshold`` can only SHRINK the returned
    set (monotone). Guards empty / degenerate ``X`` (→ ``[]``). Pure numpy, deterministic; inputs are
    not mutated.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0] if X.ndim == 2 else 0
    if n < 2:
        return []
    Xn = X if assume_normalized else _l2_normalize(X, axis=1)
    thr = float(threshold)
    cols = np.arange(n)

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    cc_parts: list[np.ndarray] = []
    for start in range(0, n, _NEARDUP_BLOCK_ROWS):
        stop = min(start + _NEARDUP_BLOCK_ROWS, n)
        sims = Xn[start:stop] @ Xn.T  # (block, n) — the only large allocation
        # Strict upper triangle → no self, no (b, a) twin; matches scan the block row-major, so the
        # accumulated pairs stay in ascending (i, j) order and the stable sort below is deterministic
        # for a given block size. (Cosines can differ from a full-matrix scan at float32-ULP level —
        # BLAS accumulation order depends on the slab shape.)
        keep = (sims >= thr) & (cols[None, :] > np.arange(start, stop)[:, None])
        r, j = np.nonzero(keep)
        if r.size:
            ii_parts.append(r + start)
            jj_parts.append(j)
            cc_parts.append(sims[r, j])

    if not ii_parts:
        return []
    ii = np.concatenate(ii_parts)
    jj = np.concatenate(jj_parts)
    cc = np.concatenate(cc_parts)
    order = np.argsort(-cc, kind="stable")  # descending cosine, stable for determinism
    order = order[: max(int(max_pairs), 0)]
    # Clamp the reported cosine to [-1, 1]: for a duplicate / re-encode the two rows are bit-identical,
    # and float32 ``X @ X.T`` yields a self-cosine slightly ABOVE 1.0 — mathematically impossible, and it
    # would break a downstream ``cosine <= 1.0`` contract. The threshold test above still uses the raw
    # value (an over-1 cosine is correctly kept); only the returned number is normalized to its true range.
    return [
        (filenames[int(ii[o])], filenames[int(jj[o])], float(np.clip(cc[o], -1.0, 1.0)))
        for o in order
    ]


def find_similar(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    top_k: int = 5,
    *,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """Top-``k`` tracks most similar to ``query_idx``, excluding the query itself.

    Returns ``(filename, cosine_score)`` pairs sorted by descending similarity.
    ``assume_normalized`` skips the defensive row re-normalization of ``X`` (see
    the module docstring). Yields ``[]`` for an empty ``X``, an out-of-range
    ``query_idx``, or ``top_k <= 0``.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0] if X.ndim == 2 else 0
    if n == 0 or not (0 <= int(query_idx) < n) or int(top_k) <= 0:
        return []

    Xn = X if assume_normalized else _l2_normalize(X, axis=1)
    sims = Xn @ Xn[int(query_idx)]
    sims[int(query_idx)] = -np.inf  # exclude self

    k = min(int(top_k), n - 1)
    if k <= 0:
        return []
    # argpartition for the top-k, then sort just those descending.
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(filenames[i], float(sims[i])) for i in top]


def find_neighbours(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    top_k: int = 20,
    spread: int = 1,
    *,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """Up to ``top_k`` neighbours of ``query_idx``, decimated by ``spread`` for diversity.

    ``spread`` is a *stride* over the ranked cosine-similarity list: we take the
    ``top_k * spread`` nearest tracks (via :func:`find_similar`, self excluded) and then keep
    indices ``0, spread, 2·spread, …`` up to ``top_k`` picks. So:

      * ``spread == 1`` reduces **exactly** to ``find_similar(query_idx, X, filenames, top_k)``
        — the closest ``top_k`` tracks;
      * larger ``spread`` samples a wider neighbourhood (the "Proche / Équilibré / Large"
        diversity control behind radio-by-similarity and the ambience journey), trading raw
        closeness for variety while staying in the same mood region.

    Returns ``(filename, cosine_score)`` pairs, descending by similarity. Yields ``[]`` for an
    empty ``X``, an out-of-range ``query_idx``, or ``top_k <= 0``. ``spread`` is clamped to ``>= 1``.
    ``assume_normalized`` is forwarded to :func:`find_similar`.
    """
    step = max(1, int(spread))
    k = int(top_k)
    if k <= 0:
        return []
    pool = find_similar(
        query_idx, X, filenames, top_k=k * step, assume_normalized=assume_normalized
    )
    return [pool[i * step] for i in range(k) if i * step < len(pool)]


def _camelot_harm(a: str | None, b: str | None) -> float:
    """Harmonic-mixing compatibility of two Camelot codes: ``1.0`` same key, ``0.5`` a wheel neighbour
    (``camelot_neighbors``: ±1 / relative), ``0.0`` otherwise or when either is unknown (``None``)."""
    if a is None or b is None:
        return 0.0
    if a == b:
        return 1.0
    from moodengine.signals import (
        camelot_neighbors,
    )  # lazy: keeps `import search` light (no librosa)

    try:
        return 0.5 if a in camelot_neighbors(b) else 0.0
    except Exception:  # noqa: BLE001 — a malformed code contributes no bonus, never an error
        return 0.0


def _tempo_compat(a: float, b: float, sigma: float) -> float:
    """Octave-aware BPM compatibility ``exp(−(d/σ)²/2)`` with ``d = min_{r∈{1,2,½}} |log2(a·r/b)|`` —
    so double-/half-time are treated as compatible. ``0.0`` when either BPM is NaN / non-positive."""
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0 or b <= 0.0:
        return 0.0
    d = min(abs(float(np.log2(a * r / b))) for r in (1.0, 2.0, 0.5))
    return float(np.exp(-((d / sigma) ** 2) / 2.0))


def _neighbours_greedy(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    *,
    top_k: int = 20,
    lambda_: float = 0.7,
    pool_mult: int = 5,
    camelot: list[str | None] | None = None,
    bpm: np.ndarray | None = None,
    harmonic_weight: float = 0.0,
    tempo_weight: float = 0.0,
    exclude: frozenset[int] = frozenset(),
    tempo_sigma: float = 0.05,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """Shared greedy for MMR (:func:`find_neighbours_mmr`) and its harmonic/tempo generalization
    (:func:`find_neighbours_harmonic`). Both are thin wrappers so there is a single greedy — with all
    bonus weights 0 and no ``exclude`` this is exactly MMR. See those two for the public contracts."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0] if X.ndim == 2 else 0
    if n == 0 or not (0 <= int(query_idx) < n) or int(top_k) <= 0:
        return []
    q = int(query_idx)
    Xn = X if assume_normalized else _l2_normalize(X, axis=1)
    rel_all = Xn @ Xn[q]  # (n,) cosine to the seed

    # Mask self + excluded (recent/queue) rows out of the pool — never selected. Out-of-range ids ignored.
    masked = {q} | {int(e) for e in exclude if 0 <= int(e) < n}
    rel_all[list(masked)] = -np.inf
    valid = n - len(masked)
    k = min(int(top_k), valid)
    if k <= 0:
        return []
    lam = float(lambda_)

    pool_size = min(max(k, k * max(1, int(pool_mult))), valid)
    pool = np.argpartition(-rel_all, pool_size - 1)[:pool_size]
    pool = pool[np.argsort(-rel_all[pool])]  # (pool_size,) original rows, rel descending
    rel = rel_all[pool].astype(np.float32)
    G = Xn[pool] @ Xn[pool].T  # (pool_size, pool_size) pairwise cosine
    use_harm = bool(camelot) and harmonic_weight != 0.0
    use_tempo = bpm is not None and tempo_weight != 0.0
    # Non-optional stand-ins for the closure below (empty when the feature is off):
    # every index guard stays simple and the Optional never leaks past this point.
    cam_list: list[str | None] = camelot if camelot is not None else []
    bpm_vec = np.asarray(bpm, dtype=np.float64).reshape(-1) if bpm is not None else np.empty(0)

    def _bonus(cand_positions: list[int], ref_row: int) -> np.ndarray:
        """Harmonic + tempo bonus for the remaining candidates, measured against ``ref_row`` (the seed
        on the first pick, then the previously-selected track → a continuous harmonic/tempo chain)."""
        out = np.zeros(len(cand_positions), dtype=np.float32)
        ref_cam = cam_list[ref_row] if (use_harm and 0 <= ref_row < len(cam_list)) else None
        ref_bpm = bpm_vec[ref_row] if (use_tempo and ref_row < bpm_vec.shape[0]) else np.nan
        for i, p in enumerate(cand_positions):
            r = int(pool[p])
            if use_harm:
                cam = cam_list[r] if r < len(cam_list) else None
                out[i] += harmonic_weight * _camelot_harm(cam, ref_cam)
            if use_tempo:
                a = bpm_vec[r] if r < bpm_vec.shape[0] else np.nan
                out[i] += tempo_weight * _tempo_compat(float(a), float(ref_bpm), tempo_sigma)
        return out

    # Greedy. selected/remaining are POSITIONS into `pool`; `max_sim[p]` is p's TRUE max cosine to any
    # chosen pick (may be negative). `max_sim is None` marks the empty chosen set → the first pick has no
    # diversity penalty; thereafter it is the genuine running max (never floored at 0, so an
    # anti-correlated candidate keeps its negative penalty, matching a from-scratch MMR). `ref_row`
    # chains: the seed for the first pick, then the last-selected track (smoothed harmonic/tempo chain).
    selected: list[int] = []
    remaining = list(range(pool_size))
    max_sim: np.ndarray | None = None
    ref_row = q
    while remaining and len(selected) < k:
        cand = np.asarray(remaining)
        div = np.zeros(len(remaining), dtype=np.float32) if max_sim is None else max_sim[cand]
        score = lam * rel[cand] - (1.0 - lam) * div
        if use_harm or use_tempo:
            score = score + _bonus(remaining, ref_row)
        best = remaining.pop(
            int(np.argmax(score))
        )  # first max on ties → the more-relevant candidate
        selected.append(best)
        max_sim = G[best].copy() if max_sim is None else np.maximum(max_sim, G[best])
        ref_row = int(pool[best])
    return [(filenames[int(pool[p])], float(rel[p])) for p in selected]


def find_neighbours_mmr(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    top_k: int = 20,
    lambda_: float = 0.7,
    pool_mult: int = 5,
    *,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """Up to ``top_k`` neighbours of ``query_idx`` by **Maximal Marginal Relevance** — relevant to the
    seed while penalizing redundancy with the already-chosen tracks (a smarter "diversity" than the
    coarse ``spread`` stride of :func:`find_neighbours`).

    Greedy over the ``top_k * pool_mult`` nearest candidates (self excluded). At each step pick

        ``argmax_i [ λ·sim(i, seed) − (1−λ)·max_{j∈chosen} sim(i, j) ]``

    where ``λ = lambda_``. The first pick has no chosen set, so its diversity penalty is 0 → it is the
    single most relevant track; ``lambda_ == 1`` degrades to a pure top-``k`` by relevance. Returns
    ``(filename, cosine_to_seed)`` — the score is always the REAL cosine to the seed, never the composite
    MMR objective (transparency). Deterministic (ties break toward the more-relevant candidate). Pure
    numpy; yields ``[]`` for an empty ``X``, an out-of-range ``query_idx``, or ``top_k <= 0``.
    ``assume_normalized`` skips the defensive row re-normalization (see the module docstring).
    """
    return _neighbours_greedy(
        query_idx,
        X,
        filenames,
        top_k=top_k,
        lambda_=lambda_,
        pool_mult=pool_mult,
        assume_normalized=assume_normalized,
    )


def find_neighbours_harmonic(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    *,
    top_k: int = 20,
    lambda_: float = 0.7,
    pool_mult: int = 5,
    camelot: list[str | None] | None = None,
    bpm: np.ndarray | None = None,
    harmonic_weight: float = 0.0,
    tempo_weight: float = 0.0,
    exclude: frozenset[int] = frozenset(),
    tempo_sigma: float = 0.05,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """MMR (:func:`find_neighbours_mmr`) enriched with two transparent, octave/harmony-aware bonuses and
    a ``recent`` exclusion — a "radio"-style continuous-playback ranking. At each greedy step pick

        ``argmax_i [ λ·rel(i,seed) − (1−λ)·max_{j∈chosen} sim(i,j)
                     + harmonic_weight·harm(key[i], key[ref]) + tempo_weight·tempo(bpm[i], bpm[ref]) ]``

    where ``ref`` is the seed on the first pick then the PREVIOUSLY-selected track (a continuous
    harmonic/tempo chain, not a comparison to the frozen seed). ``camelot`` / ``bpm`` are aligned to the
    ROWS of ``X`` (``camelot[row]`` may be ``None``; ``bpm[row]`` may be ``NaN``) — a track with a missing
    signal contributes 0 to the bonuses and is never removed, so nothing is fabricated. ``harm`` = 1.0
    same key / 0.5 a Camelot-wheel neighbour / 0.0 else (via ``moodengine.signals.camelot_neighbors``);
    ``tempo`` is the octave-aware Gaussian ``_tempo_compat``. ``exclude`` (row indices) drops recent /
    queued tracks from the pool before the greedy.

    Returns ``(filename, cosine_to_seed)`` — the real cosine, never the composite objective. Deterministic,
    pure numpy. With ``harmonic_weight == 0 ∧ tempo_weight == 0 ∧ exclude == ∅`` this is **exactly**
    :func:`find_neighbours_mmr` (both delegate to the same greedy)."""
    return _neighbours_greedy(
        query_idx,
        X,
        filenames,
        top_k=top_k,
        lambda_=lambda_,
        pool_mult=pool_mult,
        camelot=camelot,
        bpm=bpm,
        harmonic_weight=harmonic_weight,
        tempo_weight=tempo_weight,
        exclude=exclude,
        tempo_sigma=tempo_sigma,
        assume_normalized=assume_normalized,
    )


def search_by_text(
    query: str,
    X: np.ndarray,
    clap_embedder,
    filenames: list[str],
    top_k: int = 10,
    *,
    assume_normalized: bool = False,
) -> list[tuple[str, float]]:
    """Rank tracks by cosine similarity to a free-text ``query``.

    The query is embedded once via ``clap_embedder.embed_text([query])`` and
    L2-normalized into the shared CLAP space, then scored against every row of
    ``X``. ``assume_normalized`` skips the defensive re-normalization of ``X``
    only — the single query vector is always normalized (O(d), and text
    embeddings come straight from the model, outside the caller's guarantee).
    Returns ``(filename, score)`` pairs descending. Yields ``[]`` for an
    empty ``X`` or ``top_k <= 0``.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0] if X.ndim == 2 else 0
    if n == 0 or int(top_k) <= 0:
        return []

    q = np.asarray(clap_embedder.embed_text([query]), dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    q = _l2_normalize(q[0], axis=-1)

    Xn = X if assume_normalized else _l2_normalize(X, axis=1)
    sims = Xn @ q

    k = min(int(top_k), n)
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(filenames[i], float(sims[i])) for i in top]


def playlist_from_text(
    query: str,
    X: np.ndarray,
    clap_embedder,
    filenames: list[str],
    top_k: int = 20,
    *,
    assume_normalized: bool = False,
) -> list[str]:
    """Filenames only, descending by relevance to ``query`` (see :func:`search_by_text`)."""
    return [
        name
        for name, _ in search_by_text(
            query, X, clap_embedder, filenames, top_k, assume_normalized=assume_normalized
        )
    ]


def late_interaction_scores(
    query_segments: np.ndarray,
    candidate_segments: "list[np.ndarray]",
    *,
    aggregate: str = "sum",
) -> "list[tuple[int, float, int, int]]":
    """Rerank candidates by **MaxSim late interaction** (ColBERT — Khattab & Zaharia 2020).

    Each track is a *set* of per-segment embeddings (the structural sections from
    :func:`moodengine.signals.segment_structure`, CLAP-embedded).
    The query↔candidate similarity is ``MaxSim`` — for every query section, the best-matching candidate
    section, summed:

        ``score(c) = Σ_i max_j (q_i · c_j)``     (``aggregate="mean"`` averages instead of sums)

    ``query_segments`` is ``(nq, d)`` and each ``candidate_segments[k]`` is ``(nc_k, d)``, all assumed
    L2-normalized (so the dot product is a cosine). Returns ``[(candidate_index, score, best_q_seg,
    best_c_seg), …]`` sorted by descending score, where ``(best_q_seg, best_c_seg)`` is the single
    strongest section pair (``argmax_{i,j} q_i·c_j``) — "the section that matches". A candidate that is
    empty or whose embedding dim doesn't match the query is IGNORED (never scored from nothing).
    Pure numpy, torch-free, deterministic (ties keep the caller's candidate order). No I/O.
    """
    q = np.asarray(query_segments, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] == 0:
        return []
    d = q.shape[1]

    scored: list[tuple[int, float, int, int]] = []
    for idx, cand in enumerate(candidate_segments):
        c = np.asarray(cand, dtype=np.float32)
        if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != d:
            continue  # empty / dimension-mismatched candidate: ignored, never fabricated
        sim = q @ c.T  # (nq, nc) pairwise section cosines
        per_query_best = sim.max(axis=1)  # (nq,) each query section's best candidate section
        agg = float(per_query_best.sum()) if aggregate == "sum" else float(per_query_best.mean())
        flat = int(np.argmax(sim))  # the single strongest (query, candidate) section pair
        best_q, best_c = divmod(flat, sim.shape[1])
        scored.append((idx, agg, int(best_q), int(best_c)))

    scored.sort(key=lambda t: -t[1])  # stable → ties preserve candidate order (deterministic)
    return scored
