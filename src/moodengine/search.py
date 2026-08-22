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

from dataclasses import dataclass, field

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

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    cc_parts: list[np.ndarray] = []
    for start in range(0, n, _NEARDUP_BLOCK_ROWS):
        stop = min(start + _NEARDUP_BLOCK_ROWS, n)
        # Only columns j > i survive the triangle below, and every row of this slab has
        # i >= start — so columns 0..start-1 are provably dead. Multiplying and scanning them
        # was the bulk of the cost on a large library; the slab is (block, n - start), not
        # (block, n), and shrinks as the scan advances.
        sims = Xn[start:stop] @ Xn[start:].T  # (block, n - start) — the only large allocation
        # Strict upper triangle → no self, no (b, a) twin; matches scan the block row-major, so the
        # accumulated pairs stay in ascending (i, j) order and the stable sort below is deterministic
        # for a given block size. (Cosines can differ from a full-matrix scan at float32-ULP level —
        # BLAS accumulation order depends on the slab shape.)
        local_cols = np.arange(start, n)
        keep = (sims >= thr) & (local_cols[None, :] > np.arange(start, stop)[:, None])
        r, j = np.nonzero(keep)
        if r.size:
            ii_parts.append(r + start)
            jj_parts.append(j + start)
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
      * larger ``spread`` samples a wider neighbourhood (the "close / balanced / wide"
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
    except Exception:  # noqa: BLE001 — a malformed code contributes no bonus and never an error
        return 0.0


#: Playback-rate ratios the tempo term considers, so double- and half-time read as compatible.
#: Shared by the scalar `_tempo_compat` and the vectorized per-step `_tempo_bonus_row` —
#: one definition, so the two forms cannot drift apart.
_TEMPO_RATIOS: tuple[float, ...] = (1.0, 2.0, 0.5)


def _tempo_compat(a: float, b: float, sigma: float) -> float:
    """Octave-aware BPM compatibility ``exp(−(d/σ)²/2)`` with ``d = min_{r∈{1,2,½}} |log2(a·r/b)|`` —
    so double-/half-time are treated as compatible. ``0.0`` when either BPM is NaN / non-positive."""
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0.0 or b <= 0.0:
        return 0.0
    d = min(abs(float(np.log2(a * r / b))) for r in _TEMPO_RATIOS)
    return float(np.exp(-((d / sigma) ** 2) / 2.0))


# `eq=False`: the generated __eq__/__hash__ would walk the ndarray fields and raise
# ("truth value of an array is ambiguous"). These records are only ever attribute-read, so
# the comparison is not merely unused — it must not exist.
@dataclass(frozen=True, eq=False)
class _HarmonicBonus:
    """Per-call Camelot tables for one candidate pool.

    ``ids[p]`` is the position in ``distinct`` of pool member ``p``'s code, so a bonus computed once
    per DISTINCT code expands to the whole pool by fancy indexing. ``memo`` caches that expanded row
    per reference code and lives on this per-call record — a module-level memo would give this
    stateless core a hidden per-process memory.
    """

    weight: float
    codes: list[str | None]  # the caller's list, indexed by ROW (not by pool position)
    distinct: list[str | None]
    ids: np.ndarray  # (pool_size,) intp into `distinct`
    memo: dict[str | None, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class _TempoBonus:
    """Per-call BPM tables for one candidate pool.

    ``ok`` masks pool members whose BPM is unknown or non-positive; ``safe`` carries a 1.0
    placeholder under that mask so ``log2`` never sees a value it would warn about and whose result
    the mask discards anyway. ``memo`` caches the expanded row per reference BPM, per call.
    """

    weight: float
    sigma: float
    bpm: np.ndarray  # (n,) float64, indexed by ROW
    ok: np.ndarray  # (pool_size,) bool
    safe: np.ndarray  # (pool_size,) float64
    memo: dict[float, np.ndarray] = field(default_factory=dict)


def _build_harmonic_bonus(
    camelot: list[str | None] | None, weight: float, pool_rows: list[int]
) -> _HarmonicBonus | None:
    """Camelot tables for ``pool_rows``, or ``None`` when the harmonic term is off (no codes, or a
    zero weight) — the greedy then skips it entirely instead of adding a row of zeros."""
    if not camelot or weight == 0.0:
        return None

    pool_cam = [camelot[r] if r < len(camelot) else None for r in pool_rows]
    distinct = list(dict.fromkeys(pool_cam))
    index = {c: i for i, c in enumerate(distinct)}

    return _HarmonicBonus(
        weight=weight,
        codes=camelot,
        distinct=distinct,
        ids=np.array([index[c] for c in pool_cam], dtype=np.intp),
    )


def _build_tempo_bonus(
    bpm_vec: np.ndarray, has_bpm: bool, weight: float, sigma: float, pool_rows: list[int]
) -> _TempoBonus | None:
    """BPM tables for ``pool_rows``, or ``None`` when the tempo term is off (no BPMs, or a zero
    weight) — the greedy then skips it entirely instead of adding a row of zeros.

    ``bpm_vec`` arrives already converted, so a caller that supplies an unconvertible ``bpm``
    still fails at the same point it did before this was a separate function, whatever the weight.
    """
    if not has_bpm or weight == 0.0:
        return None

    pool_bpm = np.array(
        [bpm_vec[r] if r < bpm_vec.shape[0] else np.nan for r in pool_rows], dtype=np.float64
    )
    ok = np.isfinite(pool_bpm) & (pool_bpm > 0.0)

    # A placeholder under the mask keeps log2 off NaN and non-positive BPMs, which would
    # otherwise warn and produce values the mask discards anyway.
    return _TempoBonus(
        weight=weight, sigma=sigma, bpm=bpm_vec, ok=ok, safe=np.where(ok, pool_bpm, 1.0)
    )


def _harmonic_bonus_row(tables: _HarmonicBonus, ref_row: int) -> np.ndarray:
    """Weighted harmonic bonus of every pool member against ``ref_row``'s Camelot code.

    ``_camelot_harm`` is called once per DISTINCT pool code, so the ``a == b`` rule that scores two
    identical MALFORMED codes 1.0 is inherited rather than re-derived from a hand-written key map.
    """
    ref_cam = tables.codes[ref_row] if 0 <= ref_row < len(tables.codes) else None

    row = tables.memo.get(ref_cam)
    if row is None:
        per_code = np.array([_camelot_harm(c, ref_cam) for c in tables.distinct], dtype=np.float32)
        row = tables.memo[ref_cam] = per_code[tables.ids]

    return tables.weight * row


def _tempo_bonus_row(tables: _TempoBonus, ref_row: int) -> np.ndarray | None:
    """Weighted octave-aware tempo bonus of every pool member against ``ref_row``'s BPM, or ``None``
    when that reference BPM is unknown or non-positive — it then contributes nothing, exactly as the
    scalar :func:`_tempo_compat` does."""
    ref_bpm = float(tables.bpm[ref_row]) if ref_row < tables.bpm.shape[0] else float("nan")
    if not np.isfinite(ref_bpm) or ref_bpm <= 0.0:
        return None

    row = tables.memo.get(ref_bpm)
    if row is None:
        d = np.min(np.abs(np.log2(np.outer(tables.safe, _TEMPO_RATIOS) / ref_bpm)), axis=1)
        row = tables.memo[ref_bpm] = np.where(
            tables.ok, np.exp(-((d / tables.sigma) ** 2) / 2.0), 0.0
        ).astype(np.float32)

    return tables.weight * row


def _bonus_row(
    harmonic: _HarmonicBonus | None, tempo: _TempoBonus | None, ref_row: int, pool_size: int
) -> np.ndarray:
    """Harmonic + tempo bonus of EVERY pool member against ``ref_row`` (the seed on the first pick,
    then the previously-selected track → a continuous harmonic/tempo chain).

    The bonus depends on the candidate AND on ``ref_row``, and ``ref_row`` changes once per greedy
    STEP — so it is computed for the whole pool per step and indexed, instead of calling the scalar
    helpers once per (step, candidate).
    """
    out = np.zeros(pool_size, dtype=np.float32)

    if harmonic is not None:
        out += _harmonic_bonus_row(harmonic, ref_row)

    if tempo is not None:
        row = _tempo_bonus_row(tempo, ref_row)
        if row is not None:
            out += row

    return out


def _greedy_select(
    rel: np.ndarray,
    G: np.ndarray,
    pool: np.ndarray,
    *,
    k: int,
    lam: float,
    seed_row: int,
    harmonic: _HarmonicBonus | None,
    tempo: _TempoBonus | None,
) -> list[int]:
    """Greedy MMR over the candidate pool; returns the picks in order, as POSITIONS into ``pool``.

    ``max_sim[p]`` is p's TRUE max cosine to any chosen pick (may be negative). ``max_sim is None``
    marks the empty chosen set → the first pick has no diversity penalty; thereafter it is the
    genuine running max (never floored at 0, so an anti-correlated candidate keeps its negative
    penalty, matching a from-scratch MMR). ``ref_row`` chains: the seed for the first pick, then the
    last-selected track (a smoothed harmonic/tempo chain, not a comparison to the frozen seed).
    """
    pool_size = rel.shape[0]
    selected: list[int] = []
    remaining = list(range(pool_size))
    max_sim: np.ndarray | None = None
    ref_row = seed_row

    while remaining and len(selected) < k:
        cand = np.asarray(remaining)
        div = np.zeros(len(remaining), dtype=np.float32) if max_sim is None else max_sim[cand]
        score = lam * rel[cand] - (1.0 - lam) * div
        if harmonic is not None or tempo is not None:
            score = score + _bonus_row(harmonic, tempo, ref_row, pool_size)[cand]

        best = remaining.pop(int(np.argmax(score)))  # first max on ties → the more-relevant one
        selected.append(best)
        max_sim = G[best].copy() if max_sim is None else np.maximum(max_sim, G[best])
        ref_row = int(pool[best])

    return selected


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
    return find_neighbours_harmonic(
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
    :func:`find_neighbours_mmr`, which IS this function with both weights at 0 and no ``exclude``
    — one greedy, so the two can never drift apart."""
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

    pool_size = min(max(k, k * max(1, int(pool_mult))), valid)
    pool = np.argpartition(-rel_all, pool_size - 1)[:pool_size]
    pool = pool[np.argsort(-rel_all[pool])]  # (pool_size,) original rows, rel descending
    rel = rel_all[pool].astype(np.float32)
    G = Xn[pool] @ Xn[pool].T  # (pool_size, pool_size) pairwise cosine

    # Converted here, not inside `_build_tempo_bonus`: an unconvertible `bpm` must still fail on
    # the zero-weight path, exactly as it did when this was one function.
    bpm_vec = np.asarray(bpm, dtype=np.float64).reshape(-1) if bpm is not None else np.empty(0)
    pool_rows = pool.tolist()

    selected = _greedy_select(
        rel,
        G,
        pool,
        k=k,
        lam=float(lambda_),
        seed_row=q,
        harmonic=_build_harmonic_bonus(camelot, float(harmonic_weight), pool_rows),
        tempo=_build_tempo_bonus(
            bpm_vec, bpm is not None, float(tempo_weight), float(tempo_sigma), pool_rows
        ),
    )

    return [(filenames[int(pool[p])], float(rel[p])) for p in selected]


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
