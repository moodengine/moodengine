"""Tests for :mod:`moodengine.search` — cosine similarity search + text->playlist.

Pure numpy and torch-free; a tiny fake CLAP embedder maps query strings to
deterministic vectors so ``search_by_text`` / ``playlist_from_text`` are
reproducible without a real model.
"""

from __future__ import annotations

import hashlib

import numpy as np
from assertpy import assert_that

from moodengine.search import (
    _camelot_harm,
    _tempo_compat,
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
from moodengine.signals import camelot_neighbors


def _hash_unit_vec(text: str, dim: int) -> np.ndarray:
    """Deterministic, text-dependent unit vector in ``dim`` dims."""
    seed = int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _FakeCLAP:
    """Maps a query string to a chosen vector (overrides) or a hashed unit one."""

    def __init__(self, dim: int = 4, overrides: dict[str, np.ndarray] | None = None) -> None:
        self.dim = dim
        self.overrides = {k: np.asarray(v, dtype=np.float32) for k, v in (overrides or {}).items()}

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        rows = [
            self.overrides[p] if p in self.overrides else _hash_unit_vec(p, self.dim)
            for p in prompts
        ]
        return np.vstack(rows).astype(np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)).astype(np.float32)


def test_similarity_matrix_shape_and_unit_diagonal() -> None:
    """The pairwise cosine block is symmetric with a unit diagonal."""
    rng = np.random.default_rng(0)
    X = _l2(rng.standard_normal((5, 6)).astype(np.float32))
    S = similarity_matrix(X)
    assert_that(S.shape).is_equal_to((5, 5))
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-5)
    np.testing.assert_allclose(S, S.T, atol=1e-5)
    assert_that(bool(np.all(S <= 1.0 + 1e-4))).is_true()
    assert_that(bool(np.all(S >= -1.0 - 1e-4))).is_true()


def test_similarity_matrix_empty() -> None:
    """An empty matrix yields a ``(0, 0)`` block, not a crash."""
    S = similarity_matrix(np.empty((0, 0), dtype=np.float32))
    assert_that(S.shape).is_equal_to((0, 0))


def test_find_similar_excludes_self_and_sorted_desc() -> None:
    """``find_similar`` never returns the query and ranks strictly descending."""
    # Rows live on orthonormal axes: each track's nearest neighbour is determined
    # by the crafted overlaps below; the query itself must be excluded regardless.
    # Track 0 points mostly along axis 0 but leans toward axis 1 > 2 > 3.
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.4, 0.0, 0.0],  # closest to track 0
            [0.6, 0.0, 0.8, 0.0],  # next
            [0.2, 0.0, 0.0, 0.98],  # farthest
        ],
        dtype=np.float32,
    )
    X = _l2(X)
    files = ["a.wav", "b.wav", "c.wav", "d.wav"]
    out = find_similar(0, X, files, top_k=3)

    assert_that([name for name, _ in out]).is_equal_to(["b.wav", "c.wav", "d.wav"])
    assert_that([name for name, _ in out]).does_not_contain("a.wav")  # self excluded
    scores = [s for _, s in out]
    assert_that(scores).is_equal_to(sorted(scores, reverse=True))


def test_find_similar_top_k_caps_at_n_minus_one() -> None:
    """Asking for more neighbours than exist returns at most n-1 (self excluded)."""
    rng = np.random.default_rng(1)
    X = _l2(rng.standard_normal((3, 5)).astype(np.float32))
    out = find_similar(1, X, ["a", "b", "c"], top_k=10)
    assert_that(out).is_length(2)
    assert_that([name for name, _ in out]).does_not_contain("b")


def test_find_similar_guards_bad_inputs() -> None:
    """Empty X, out-of-range idx, or non-positive top_k all return []."""
    X = _l2(np.random.default_rng(2).standard_normal((3, 4)).astype(np.float32))
    files = ["a", "b", "c"]
    assert_that(find_similar(0, np.empty((0, 0), dtype=np.float32), [], top_k=3)).is_equal_to([])
    assert_that(find_similar(99, X, files, top_k=3)).is_equal_to([])
    assert_that(find_similar(-1, X, files, top_k=3)).is_equal_to([])
    assert_that(find_similar(0, X, files, top_k=0)).is_equal_to([])


def test_find_neighbours_spread1_equals_find_similar() -> None:
    """spread=1 is exactly the closest top_k (identical to find_similar)."""
    rng = np.random.default_rng(11)
    X = _l2(rng.standard_normal((40, 8)).astype(np.float32))
    files = [f"t{i}" for i in range(40)]
    for q in (0, 7, 39):
        assert_that(find_neighbours(q, X, files, top_k=6, spread=1)).is_equal_to(
            find_similar(q, X, files, top_k=6)
        )


def test_find_neighbours_strides_the_ranked_list() -> None:
    """spread=k picks indices 0, k, 2k, … of the top_k*spread ranked pool."""
    rng = np.random.default_rng(12)
    X = _l2(rng.standard_normal((60, 8)).astype(np.float32))
    files = [f"t{i}" for i in range(60)]
    k, spread = 5, 3
    pool = find_similar(0, X, files, top_k=k * spread)
    got = find_neighbours(0, X, files, top_k=k, spread=spread)
    assert_that(got).is_equal_to([pool[i * spread] for i in range(k)])
    # scores still descend (a strided subsequence of a descending list).
    scores = [s for _, s in got]
    assert_that(scores).is_equal_to(sorted(scores, reverse=True))


def test_find_neighbours_guards_and_clamps_spread() -> None:
    """Bad top_k -> []; spread<1 is clamped to 1 (== find_similar)."""
    X = _l2(np.random.default_rng(13).standard_normal((10, 4)).astype(np.float32))
    files = [f"t{i}" for i in range(10)]
    assert_that(find_neighbours(0, X, files, top_k=0, spread=2)).is_equal_to([])
    assert_that(find_neighbours(0, X, files, top_k=4, spread=0)).is_equal_to(
        find_similar(0, X, files, top_k=4)
    )
    assert_that(find_neighbours(99, X, files, top_k=4, spread=2)).is_equal_to(
        []
    )  # out of range (via find_similar guard)


# --------------------------------------------------------------------------- #
# MMR neighbours
# --------------------------------------------------------------------------- #
def _mean_pairwise_cos(names: list[str], Xn: np.ndarray, row: dict[str, int]) -> float:
    """Mean pairwise cosine among the returned tracks (lower = more diverse)."""
    idxs = [row[n] for n in names]
    if len(idxs) < 2:
        return 0.0
    G = Xn[idxs] @ Xn[idxs].T
    iu = np.triu_indices(len(idxs), k=1)
    return float(G[iu].mean())


def _reference_mmr(
    query_idx: int,
    X: np.ndarray,
    filenames: list[str],
    top_k: int,
    lambda_: float,
    pool_mult: int = 5,
) -> list[tuple[str, float]]:
    """An INDEPENDENT, obviously-correct MMR (explicit O(pool²) loops, true max cosine to the whole
    chosen set at each step) — the oracle that pins ``find_neighbours_mmr``'s greedy. A first-pick-only
    or wrong-index ``max_sim`` accumulation bug diverges from this on almost any matrix, which the
    hand-built diversity fixture alone cannot catch."""
    Xn = _l2(np.asarray(X, dtype=np.float32))
    n = Xn.shape[0]
    rel = Xn @ Xn[query_idx]
    rel[query_idx] = -np.inf
    k = min(int(top_k), n - 1)
    pool_size = min(max(k, k * pool_mult), n - 1)
    order = list(np.argsort(-rel)[:pool_size])  # top-(pool_size) by relevance, desc
    selected: list[int] = []
    while order and len(selected) < k:
        best, best_score = None, -np.inf
        for c in order:  # scan in rel-desc order → first max wins ties
            div = max(
                (float(Xn[c] @ Xn[s]) for s in selected), default=0.0
            )  # true max cos to chosen
            score = lambda_ * float(rel[c]) - (1.0 - lambda_) * div
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        order.remove(best)
    return [(filenames[s], float(rel[s])) for s in selected]


def test_find_neighbours_mmr_matches_independent_reference_greedy() -> None:
    # Pins the EXACT greedy selection against a hand-written reference (not the impl itself), so a
    # frozen-max_sim or wrong-index accumulation bug is caught immediately — on random matrices where
    # picks genuinely depend on the accumulated max cosine to the whole chosen set.
    for seed in (0, 1, 2):
        X = _l2(np.random.default_rng(seed).standard_normal((30, 6)).astype(np.float32))
        files = [f"t{i}" for i in range(30)]
        for q in (0, 7, 29):
            for lam in (0.4, 0.6, 0.8):
                got = find_neighbours_mmr(q, X, files, top_k=6, lambda_=lam)
                ref = _reference_mmr(q, X, files, top_k=6, lambda_=lam)
                assert_that([g[0] for g in got]).described_as(f"{seed=} {q=} {lam=}").is_equal_to(
                    [r[0] for r in ref]
                )
                for (gid, gs), (rid, rs) in zip(got, ref):
                    assert_that(gid).is_equal_to(rid)
                    assert_that(gs).is_close_to(rs, 1e-5)


def test_find_neighbours_mmr_excludes_self_and_score_is_cosine_to_seed() -> None:
    rng = np.random.default_rng(21)
    X = _l2(rng.standard_normal((60, 12)).astype(np.float32))
    files = [f"t{i}" for i in range(60)]
    out = find_neighbours_mmr(3, X, files, top_k=8, lambda_=0.6)
    names = [n for n, _ in out]
    assert_that(names).does_not_contain("t3")  # self excluded
    assert_that(names).is_length(8)  # no duplicates
    assert_that(set(names)).is_length(8)
    # the reported score is the REAL cosine to the seed, never the MMR objective
    row = {f"t{i}": i for i in range(60)}
    for name, score in out:
        assert_that(score).is_close_to(float(X[row[name]] @ X[3]), 1e-5)


def test_find_neighbours_mmr_lambda1_is_top_k_by_relevance() -> None:
    # λ=1 (diversity=0) drops the diversity term → a pure top-k by relevance == find_similar.
    rng = np.random.default_rng(22)
    X = _l2(rng.standard_normal((50, 10)).astype(np.float32))
    files = [f"t{i}" for i in range(50)]
    for q in (0, 9, 31):
        mmr = [n for n, _ in find_neighbours_mmr(q, X, files, top_k=7, lambda_=1.0)]
        top = [n for n, _ in find_similar(q, X, files, top_k=7)]
        assert_that(mmr).is_equal_to(top)


def test_find_neighbours_mmr_higher_diversity_lowers_intra_list_similarity() -> None:
    # A redundant cluster of near-duplicates near the seed + a diverse ring slightly farther. λ=1
    # grabs the whole redundant cluster (max relevance); a lower λ trades a little relevance for
    # spread, so the returned set's mean pairwise cosine strictly drops (crit #3/#4 monotonicity).
    d = 8
    seed = np.zeros(d, dtype=np.float32)
    seed[0] = 1.0
    cluster = [
        np.array([0.95, 0.05 + 0.001 * i, 0, 0, 0, 0, 0, 0], dtype=np.float32) for i in range(4)
    ]
    diverse = []
    for j in range(4):
        v = np.zeros(d, dtype=np.float32)
        v[0] = 0.85
        v[2 + j] = 0.53
        diverse.append(v)
    X = _l2(np.vstack([seed, *cluster, *diverse]).astype(np.float32))  # row 0 = seed
    files = [f"t{i}" for i in range(X.shape[0])]
    row = {f: i for i, f in enumerate(files)}

    dense = [n for n, _ in find_neighbours_mmr(0, X, files, top_k=4, lambda_=1.0)]
    spread = [n for n, _ in find_neighbours_mmr(0, X, files, top_k=4, lambda_=0.4)]
    div_dense = _mean_pairwise_cos(dense, X, row)
    div_spread = _mean_pairwise_cos(spread, X, row)
    # Measured, not fabricated — the whole point of the MMR mode:
    assert_that(div_spread).described_as(
        f"expected more diversity: {div_spread=} {div_dense=}"
    ).is_less_than(div_dense)


def test_find_neighbours_mmr_is_deterministic() -> None:
    rng = np.random.default_rng(23)
    X = _l2(rng.standard_normal((40, 9)).astype(np.float32))
    files = [f"t{i}" for i in range(40)]
    a = find_neighbours_mmr(5, X, files, top_k=6, lambda_=0.5)
    b = find_neighbours_mmr(5, X, files, top_k=6, lambda_=0.5)
    assert_that(a).is_equal_to(b)


def test_find_neighbours_mmr_guards_and_caps() -> None:
    X = _l2(np.random.default_rng(24).standard_normal((5, 4)).astype(np.float32))
    files = [f"t{i}" for i in range(5)]
    assert_that(
        find_neighbours_mmr(0, np.empty((0, 0), dtype=np.float32), [], top_k=3)
    ).is_equal_to([])
    assert_that(find_neighbours_mmr(99, X, files, top_k=3)).is_equal_to([])  # out of range
    assert_that(find_neighbours_mmr(0, X, files, top_k=0)).is_equal_to([])  # non-positive k
    assert_that(find_neighbours_mmr(0, X, files, top_k=100)).is_length(
        4
    )  # capped to n-1 (self excluded)


# --------------------------------------------------------------------------- #
# Harmonic / tempo neighbours
# --------------------------------------------------------------------------- #
def _consecutive_adjacent_rate(names: list[str], key: dict[str, str | None]) -> float:
    """Fraction of consecutive pairs whose Camelot codes are same-or-wheel-adjacent."""
    if len(names) < 2:
        return 0.0
    hits = 0
    for a, b in zip(names, names[1:]):
        ka, kb = key.get(a), key.get(b)
        if ka and kb and (ka == kb or ka in camelot_neighbors(kb)):
            hits += 1
    return hits / (len(names) - 1)


def _mean_octave_log2_delta(names: list[str], bpm: dict[str, float]) -> float:
    """Mean octave-aware |Δlog2 BPM| between consecutive pairs (lower = more tempo-compatible)."""
    if len(names) < 2:
        return 0.0
    ds = []
    for a, b in zip(names, names[1:]):
        va, vb = bpm[a], bpm[b]
        ds.append(min(abs(np.log2(va * r / vb)) for r in (1.0, 2.0, 0.5)))
    return float(np.mean(ds))


def _reference_harmonic(
    query_idx,
    X,
    filenames,
    *,
    top_k,
    lambda_,
    pool_mult=5,
    camelot=None,
    bpm=None,
    harmonic_weight=0.0,
    tempo_weight=0.0,
    exclude=frozenset(),
    tempo_sigma=0.05,
):
    """Independent reference greedy with an EXPLICIT chained ref (the seed, then the PREVIOUSLY-selected
    pick). Pins the chain in find_neighbours_harmonic: freezing ``ref`` at the seed (dropping the chain)
    diverges from this on a real fixture — the coverage the direction-only rate/delta tests lack."""
    Xn = _l2(np.asarray(X, dtype=np.float32))
    n = Xn.shape[0]
    rel = Xn @ Xn[query_idx]
    masked = {query_idx} | {int(e) for e in exclude if 0 <= int(e) < n}
    rel[list(masked)] = -np.inf
    valid = n - len(masked)
    k = min(int(top_k), valid)
    if k <= 0:
        return []
    pool_size = min(max(k, k * pool_mult), valid)
    order = list(np.argsort(-rel)[:pool_size])
    bpm_arr = np.asarray(bpm, dtype=np.float64).reshape(-1) if bpm is not None else None

    def _bonus(row: int, ref: int) -> float:
        h = (
            harmonic_weight * _camelot_harm(camelot[row], camelot[ref])
            if (camelot and harmonic_weight)
            else 0.0
        )
        t = (
            tempo_weight * _tempo_compat(float(bpm_arr[row]), float(bpm_arr[ref]), tempo_sigma)
            if (bpm_arr is not None and tempo_weight)
            else 0.0
        )
        return h + t

    selected: list[int] = []
    ref_row = query_idx
    while order and len(selected) < k:
        best, best_score = None, -np.inf
        for c in order:
            div = max((float(Xn[c] @ Xn[s]) for s in selected), default=0.0)
            score = lambda_ * float(rel[c]) - (1.0 - lambda_) * div + _bonus(c, ref_row)
            if score > best_score:
                best, best_score = c, score
        selected.append(best)
        order.remove(best)
        ref_row = best  # chain: the next bonus is measured against this just-picked track
    return [(filenames[s], float(rel[s])) for s in selected]


def test_find_neighbours_harmonic_matches_chained_reference_greedy() -> None:
    # Pins the CHAINED ref (seed → previous pick, as documented) against an independent reference greedy.
    # Freezing ref at the seed (collapsing the chain) diverges from this — coverage the aggregate
    # rate/delta tests miss. Also exercises harmonic + tempo + exclude together.
    codes = [f"{(i % 12) + 1}{'A' if i % 2 else 'B'}" for i in range(26)]
    for seed in (0, 1):
        X = _l2(np.random.default_rng(40 + seed).standard_normal((26, 10)).astype(np.float32))
        files = [f"t{i}" for i in range(26)]
        bpm = np.asarray([90.0 + float(i * 11 % 90) for i in range(26)], dtype=np.float32)
        for q in (0, 12):
            kw = dict(
                top_k=8,
                lambda_=0.6,
                camelot=codes,
                bpm=bpm,
                harmonic_weight=0.5,
                tempo_weight=0.4,
                exclude=frozenset({3, 5}),
            )
            got = find_neighbours_harmonic(q, X, files, **kw)
            ref = _reference_harmonic(q, X, files, **kw)
            assert_that([g[0] for g in got]).described_as(f"{seed=} {q=}").is_equal_to(
                [r[0] for r in ref]
            )
            for (gid, gs), (rid, rs) in zip(got, ref):
                assert_that(gid).is_equal_to(rid)
                assert_that(gs).is_close_to(rs, 1e-5)


def test_camelot_harm_and_tempo_compat_primitives() -> None:
    assert_that(_camelot_harm("8A", "8A")).is_equal_to(1.0)
    assert_that(_camelot_harm("8A", "9A")).is_equal_to(0.5)  # wheel neighbour (8A ∈ neighbours(9A))
    assert_that(_camelot_harm("8A", "8B")).is_equal_to(0.5)  # relative major/minor
    assert_that(_camelot_harm("8A", "3B")).is_equal_to(0.0)  # distant
    assert_that(_camelot_harm(None, "8A")).is_equal_to(0.0)
    assert_that(_camelot_harm("8A", None)).is_equal_to(0.0)
    # octave-aware tempo kernel
    assert_that(_tempo_compat(120, 120, 0.05)).is_close_to(1.0, 1e-6)
    assert_that(_tempo_compat(120, 240, 0.05)).is_close_to(1.0, 1e-6)  # double-time compatible
    assert_that(_tempo_compat(120, 60, 0.05)).is_close_to(1.0, 1e-6)  # half-time compatible
    assert_that(_tempo_compat(120, 121, 0.05)).is_greater_than(0.9)  # near
    assert_that(_tempo_compat(120, 150, 0.05)).is_less_than(0.1)  # far
    assert_that(_tempo_compat(float("nan"), 120, 0.05)).is_equal_to(0.0)
    assert_that(_tempo_compat(120, 0.0, 0.05)).is_equal_to(0.0)  # non-positive BPM


def test_find_neighbours_harmonic_equals_mmr_when_weights_zero() -> None:
    # Equivalence pin (crit #2): weights 0 + no exclude ⇒ EXACTLY find_neighbours_mmr (both delegate to
    # the same greedy). find_neighbours_mmr is itself pinned to an independent reference above.
    X = _l2(np.random.default_rng(31).standard_normal((40, 10)).astype(np.float32))
    files = [f"t{i}" for i in range(40)]
    for q in (0, 13, 39):
        for lam in (0.4, 0.7):
            h = find_neighbours_harmonic(
                q,
                X,
                files,
                top_k=8,
                lambda_=lam,
                harmonic_weight=0.0,
                tempo_weight=0.0,
                exclude=frozenset(),
            )
            m = find_neighbours_mmr(q, X, files, top_k=8, lambda_=lam)
            assert_that(h).is_equal_to(m)


def test_find_neighbours_harmonic_exclude_removes_from_pool() -> None:
    # crit #5: an id in `exclude` is absent from the result (removed before the greedy), self still out.
    X = _l2(np.random.default_rng(32).standard_normal((30, 8)).astype(np.float32))
    files = [f"t{i}" for i in range(30)]
    base = [n for n, _ in find_neighbours_harmonic(0, X, files, top_k=8)]
    excl_ids = {base[0], base[2]}
    excl_rows = frozenset(int(n[1:]) for n in excl_ids)
    after = [n for n, _ in find_neighbours_harmonic(0, X, files, top_k=8, exclude=excl_rows)]
    assert_that(after).does_not_contain(*excl_ids)  # excluded gone
    assert_that(after).does_not_contain("t0")  # self still excluded
    assert_that(after).is_length(8)  # still full, no dupes
    assert_that(set(after)).is_length(8)


def test_find_neighbours_harmonic_prefers_camelot_chain() -> None:
    # crit #3/#7: a harmonic weight raises the rate of consecutive Camelot-adjacent transitions, and it
    # is monotonic (rate at weight 0 ≤ 0.3 ≤ 0.6). Keys are split adjacent/distant across the pool.
    X = _l2(np.random.default_rng(33).standard_normal((24, 12)).astype(np.float32))
    files = [f"t{i}" for i in range(24)]
    adj = ["8A", "7A", "9A", "8B"]
    far = ["2B", "3B", "4B", "5B", "1A", "11A"]
    codes = ["8A"] + [adj[i % len(adj)] if i % 2 == 0 else far[i % len(far)] for i in range(1, 24)]
    key = {f"t{i}": codes[i] for i in range(24)}
    camelot = codes  # aligned to rows

    rates = []
    for hw in (0.0, 0.3, 0.6):
        names = [
            n
            for n, _ in find_neighbours_harmonic(
                0, X, files, top_k=8, lambda_=0.7, camelot=camelot, harmonic_weight=hw
            )
        ]
        rates.append(_consecutive_adjacent_rate(names, key))
    assert_that(rates[0]).is_less_than_or_equal_to(rates[1])  # monotonic non-decreasing (crit #7)
    assert_that(rates[1]).is_less_than_or_equal_to(rates[2])
    assert_that(rates[2]).is_greater_than(rates[0])  # measured harmonic effect (crit #3)


def test_find_neighbours_harmonic_prefers_tempo_octave_compatible() -> None:
    # crit #4: a tempo weight lowers the mean octave-aware |Δlog2 BPM| between consecutive picks.
    X = _l2(np.random.default_rng(34).standard_normal((24, 12)).astype(np.float32))
    files = [f"t{i}" for i in range(24)]
    compat = [120.0, 118.0, 122.0, 240.0, 60.0]  # octave-compatible with the seed's 120
    incompat = [95.0, 140.0, 175.0, 100.0]
    vals = [120.0] + [
        compat[i % len(compat)] if i % 2 == 0 else incompat[i % len(incompat)] for i in range(1, 24)
    ]
    bpm = {f"t{i}": vals[i] for i in range(24)}
    bpm_arr = np.asarray(vals, dtype=np.float32)

    d0 = _mean_octave_log2_delta(
        [
            n
            for n, _ in find_neighbours_harmonic(
                0, X, files, top_k=8, lambda_=0.7, bpm=bpm_arr, tempo_weight=0.0
            )
        ],
        bpm,
    )
    d1 = _mean_octave_log2_delta(
        [
            n
            for n, _ in find_neighbours_harmonic(
                0, X, files, top_k=8, lambda_=0.7, bpm=bpm_arr, tempo_weight=0.5
            )
        ],
        bpm,
    )
    assert_that(d1).is_less_than(d0)  # measured tempo effect


def test_find_neighbours_harmonic_missing_signals_are_safe() -> None:
    # crit #6 (anti-fabrication): a track with no key/BPM (None / NaN) is NEVER removed by the bonuses —
    # it stays eligible via CLAP relevance and contributes 0 to harmonic/tempo. Nothing is invented.
    X = _l2(np.random.default_rng(35).standard_normal((20, 8)).astype(np.float32))
    files = [f"t{i}" for i in range(20)]
    # The single most relevant neighbour of t0:
    top1 = find_similar(0, X, files, top_k=1)[0][0]
    camelot: list[str | None] = ["8A"] + [None] * 19  # every candidate's key unknown
    bpm = np.full(20, np.nan, dtype=np.float32)
    bpm[0] = 120.0  # every candidate's BPM unknown
    out = [
        n
        for n, _ in find_neighbours_harmonic(
            0, X, files, top_k=8, camelot=camelot, bpm=bpm, harmonic_weight=0.8, tempo_weight=0.8
        )
    ]
    assert_that(out).contains(top1)  # still eligible, not dropped
    # With ALL candidate signals missing, the bonuses are all 0 → identical to plain MMR.
    mmr = [n for n, _ in find_neighbours_mmr(0, X, files, top_k=8)]
    assert_that(out).is_equal_to(mmr)


def test_find_neighbours_harmonic_is_deterministic() -> None:
    X = _l2(np.random.default_rng(36).standard_normal((24, 10)).astype(np.float32))
    files = [f"t{i}" for i in range(24)]
    camelot = ["8A", "9A", "7A", "3B"] * 6
    bpm = np.asarray([120.0, 121.0, 240.0, 95.0] * 6, dtype=np.float32)
    a = find_neighbours_harmonic(
        0,
        X,
        files,
        top_k=6,
        camelot=camelot,
        bpm=bpm,
        harmonic_weight=0.5,
        tempo_weight=0.4,
        exclude=frozenset({3, 7}),
    )
    b = find_neighbours_harmonic(
        0,
        X,
        files,
        top_k=6,
        camelot=camelot,
        bpm=bpm,
        harmonic_weight=0.5,
        tempo_weight=0.4,
        exclude=frozenset({3, 7}),
    )
    assert_that(a).is_equal_to(b)


def test_search_by_text_ranks_by_query_alignment() -> None:
    """The track whose embedding aligns with the query ranks first."""
    e = np.eye(3, dtype=np.float32)
    X = e.copy()  # three orthonormal tracks
    files = ["x.wav", "y.wav", "z.wav"]
    # Query embedding points exactly at the second track.
    embedder = _FakeCLAP(dim=3, overrides={"query": e[1]})
    out = search_by_text("query", X, embedder, files, top_k=3)

    assert_that(out[0][0]).is_equal_to("y.wav")
    assert_that(out[0][1]).is_close_to(1.0, 1e-5)
    scores = [s for _, s in out]
    assert_that(scores).is_equal_to(sorted(scores, reverse=True))
    assert_that(out).is_length(3)


def test_search_by_text_respects_top_k() -> None:
    """``top_k`` truncates the result list."""
    rng = np.random.default_rng(3)
    X = _l2(rng.standard_normal((6, 4)).astype(np.float32))
    files = [f"t{i}.wav" for i in range(6)]
    out = search_by_text("dreamy nocturnal", X, _FakeCLAP(dim=4), files, top_k=2)
    assert_that(out).is_length(2)


def test_search_by_text_empty_guards() -> None:
    """Empty X or non-positive top_k -> [] without touching the embedder."""
    embedder = _FakeCLAP(dim=4)
    assert_that(
        search_by_text("q", np.empty((0, 0), dtype=np.float32), embedder, [], top_k=5)
    ).is_equal_to([])
    X = _l2(np.random.default_rng(4).standard_normal((3, 4)).astype(np.float32))
    assert_that(search_by_text("q", X, embedder, ["a", "b", "c"], top_k=0)).is_equal_to([])


def test_playlist_from_text_returns_filenames_only_desc() -> None:
    """``playlist_from_text`` returns just filenames, in the same descending order."""
    e = np.eye(3, dtype=np.float32)
    X = e.copy()
    files = ["x.wav", "y.wav", "z.wav"]
    embedder = _FakeCLAP(dim=3, overrides={"q": e[2]})
    names = playlist_from_text("q", X, embedder, files, top_k=3)

    ranked = search_by_text("q", X, embedder, files, top_k=3)
    assert_that(names).is_equal_to([n for n, _ in ranked])
    assert_that(names[0]).is_equal_to("z.wav")
    assert_that(all(isinstance(n, str) for n in names)).is_true()


# --------------------------------------------------------------------------- #
# MaxSim late interaction (ColBERT-style rerank)
# --------------------------------------------------------------------------- #
def _seg(*rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_late_interaction_maxsim_prefers_the_shared_section() -> None:
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0], e[1])  # two orthonormal query sections
    c1 = _seg(e[0], e[2])  # shares section 0 with the query
    c2 = _seg(e[2], e[3])  # shares nothing
    out = late_interaction_scores(query, [c1, c2])
    assert_that([i for i, *_ in out]).is_equal_to([0, 1])  # c1 ranked above c2
    idx, score, bq, bc = out[0]
    assert_that(idx).is_equal_to(0)
    assert_that(score).is_close_to(1.0, 1e-6)  # Σ max = (q0·e0=1) + (q1··=0)
    assert_that((bq, bc)).is_equal_to((0, 0))  # strongest pair = the shared section
    assert_that(out[1][1]).is_close_to(0.0, tolerance=1e-12)  # c2 scores 0 (orthogonal → exact 0)


def test_late_interaction_is_monotone_in_added_sections() -> None:
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0], e[1])
    base_score = late_interaction_scores(query, [_seg(e[0], e[2])])[0][1]
    for extra in (e[0], e[3], e[1]):  # duplicate / irrelevant / helpful
        more = late_interaction_scores(query, [_seg(e[0], e[2], extra)])[0][1]
        assert_that(more).is_greater_than_or_equal_to(
            base_score - 1e-6
        )  # adding never lowers MaxSim


def test_late_interaction_ignores_empty_and_dim_mismatch() -> None:
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0], e[1])
    empty = np.zeros((0, 4), dtype=np.float32)
    wrong_dim = np.ones((2, 3), dtype=np.float32)
    good = _seg(e[0])
    out = late_interaction_scores(query, [empty, wrong_dim, good])
    assert_that([i for i, *_ in out]).is_equal_to([2])  # only the valid candidate is scored
    assert_that(late_interaction_scores(np.zeros((0, 4), dtype=np.float32), [good])).is_equal_to(
        []
    )  # empty query


def test_late_interaction_deterministic_and_stable_ties() -> None:
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0])
    a, b = _seg(e[0]), _seg(e[0])  # identical candidates → a tie
    out = late_interaction_scores(query, [a, b])
    assert_that([i for i, *_ in out]).is_equal_to(
        [0, 1]
    )  # tie preserves caller order (stable sort)
    assert_that(late_interaction_scores(query, [a, b])).is_equal_to(out)  # deterministic


def test_late_interaction_is_torch_free() -> None:
    import subprocess
    import sys

    code = (
        "import sys, numpy as np, moodengine.search as s; "
        "e=np.eye(4,dtype='float32'); "
        "out=s.late_interaction_scores(e[:2], [e[[0,2]], e[[2,3]]]); "
        "assert out[0][0]==0; "
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.')]; "
        "sys.exit('torch loaded: '+repr(bad)) if bad else None"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert_that(r.returncode).described_as((r.stdout + r.stderr).strip()).is_equal_to(0)


def test_late_interaction_reranks_out_of_input_order() -> None:
    # The stronger-MaxSim candidate is passed SECOND (recall order); the rerank MUST reorder it to the
    # front — falsifies a "return recall order" passthrough (a deleted sort does zero reranking).
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0], e[1])
    weak = _seg(e[2], e[3])  # shares nothing → 0
    strong = _seg(e[0], e[1])  # shares both → 2
    out = late_interaction_scores(query, [weak, strong])
    assert_that([i for i, *_ in out]).is_equal_to(
        [1, 0]
    )  # strong (input index 1) reranked to front
    assert_that(out[0][1]).is_close_to(2.0, tolerance=1e-6)
    assert_that(out[1][1]).is_close_to(0.0, tolerance=1e-12)  # shares nothing → exact 0


def test_late_interaction_best_pair_is_off_diagonal() -> None:
    # The strongest pair is query-section 1 ↔ candidate-section 0 (OFF the diagonal) → (best_q, best_c) ==
    # (1, 0). Falsifies a best_q<->best_c swap or a transposed divmod (which every square/diagonal fixture
    # cannot catch).
    e = np.eye(4, dtype=np.float32)
    query = _seg(e[0], e[1])
    cand = _seg(e[1], e[2])  # sim = [[0,0],[1,0]] → argmax at (row 1, col 0)
    idx, score, bq, bc = late_interaction_scores(query, [cand])[0]
    assert_that((bq, bc)).is_equal_to((1, 0))
    assert_that(score).is_close_to(1.0, 1e-6)  # q1·c0 = 1, q0·· = 0


# --- near-duplicate detection ------------------------------------------------


def _names(n):
    return [f"t{i}.mp3" for i in range(n)]


def test_near_duplicate_pairs_flags_a_duplicated_row():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 8)).astype(np.float32)
    X = np.vstack([X, X[2][None, :]])  # row 6 duplicates row 2 → cosine 1.0
    pairs = near_duplicate_pairs(X, _names(7), threshold=0.98)
    # the duplicated pair (t2, t6) is present, in (i<j) order, at cosine ~1.0 and ranked first.
    assert_that([(a, b) for a, b, _ in pairs]).contains(("t2.mp3", "t6.mp3"))
    top = pairs[0]
    assert_that((top[0], top[1])).is_equal_to(("t2.mp3", "t6.mp3"))
    assert_that(top[2]).is_close_to(1.0, 1e-5)


def test_near_duplicate_threshold_is_monotone_and_wellformed():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 6)).astype(np.float32)
    X = np.vstack([X, X[0][None, :], X[3][None, :] * 1.0])  # a couple of exact dups
    names = _names(X.shape[0])
    lo = set((a, b) for a, b, _ in near_duplicate_pairs(X, names, threshold=0.90))
    hi = set((a, b) for a, b, _ in near_duplicate_pairs(X, names, threshold=0.999))
    assert_that(hi <= lo).is_true()  # raising the threshold can only shrink the set
    for a, b, c in near_duplicate_pairs(X, names, threshold=0.90):
        assert_that(a).is_not_equal_to(b)  # no self-pair
    all_pairs = [(a, b) for a, b, _ in near_duplicate_pairs(X, names, threshold=0.5)]
    assert_that(all_pairs).is_length(len(set(all_pairs)))  # no symmetric duplicate (i<j only)
    # sorted by descending cosine
    cs = [c for _, _, c in near_duplicate_pairs(X, names, threshold=0.5)]
    assert_that(cs).is_equal_to(sorted(cs, reverse=True))


def test_near_duplicate_cosine_never_exceeds_one():
    """Regression: an exact duplicate makes two 512-d float32 rows' self-cosine round ABOVE 1.0; the
    reported cosine must be clamped to the true [-1, 1] range (else a downstream ``<= 1.0`` bound 500s).
    Loops over seeds so the overflow is guaranteed — and asserts it occurred (non-vacuous)."""
    saw_overflow = False
    for seed in range(30):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((25, 512)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.vstack([X, X[3][None, :]]).astype(np.float32)  # exact duplicate of row 3 → row 25
        pairs = near_duplicate_pairs(X, _names(26), threshold=0.5)
        assert_that(all(-1.0 <= c <= 1.0 for _, _, c in pairs)).is_true()  # clamped to cosine range
        dup = [c for a, b, c in pairs if {a, b} == {"t3.mp3", "t25.mp3"}]
        assert_that(dup).is_not_empty()  # the duplicate pair is present
        assert_that(dup[0]).is_less_than_or_equal_to(1.0)  # at cosine <= 1.0
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        if float(Xn[3] @ Xn[-1]) > 1.0:
            saw_overflow = True
    assert_that(saw_overflow).described_as(
        "vacuous test: no float32 cosine overflow occurred across the seeds"
    ).is_true()


def test_near_duplicate_guards_and_max_pairs():
    assert_that(near_duplicate_pairs(np.zeros((0, 4), np.float32), [])).is_equal_to([])
    assert_that(near_duplicate_pairs(np.ones((1, 4), np.float32), ["a.mp3"])).is_equal_to(
        []
    )  # need >= 2 rows
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    assert_that(near_duplicate_pairs(X, _names(20), threshold=-1.0, max_pairs=3)).is_length(
        3
    )  # truncation


def test_near_duplicate_blockwise_equals_single_block(monkeypatch):
    """The row-slab scan is a memory optimization only: forcing a tiny block size
    that never divides n evenly must find the same pairs. Pair membership is
    compared exactly; cosines at float32-ULP tolerance (BLAS may accumulate a
    slab matmul in a different order than the full one)."""
    import moodengine.search as search

    rng = np.random.default_rng(5)
    base = rng.standard_normal((23, 8)).astype(np.float32)
    dups = base[:6] + 1e-4 * rng.standard_normal((6, 8)).astype(np.float32)
    X = np.vstack([base, dups])
    names = _names(X.shape[0])

    full = near_duplicate_pairs(X, names, threshold=0.9, max_pairs=100)
    monkeypatch.setattr(search, "_NEARDUP_BLOCK_ROWS", 5)
    chunked = near_duplicate_pairs(X, names, threshold=0.9, max_pairs=100)

    assert_that(sorted((a, b) for a, b, _ in chunked)).is_equal_to(
        sorted((a, b) for a, b, _ in full)
    )
    by_pair_full = {(a, b): c for a, b, c in full}
    for a, b, c in chunked:
        assert_that(c).is_close_to(by_pair_full[(a, b)], 2e-6)
    assert_that(len(full)).is_greater_than_or_equal_to(6)  # the injected near-duplicates are found


# --------------------------------------------------------------------------- #
# assume_normalized — skip the per-call re-normalization under the caller's
# guarantee that rows are already unit-norm
# --------------------------------------------------------------------------- #
def test_assume_normalized_ranking_helpers_match_default_on_unit_rows():
    """On unit-norm rows the flag changes nothing observable: same neighbours in the
    same order, scores equal to float32-ULP level (the default path divides each row
    by a norm of ~1.0, which is not bit-exactly 1.0)."""
    rng = np.random.default_rng(21)
    X = _l2(rng.standard_normal((30, 8)).astype(np.float32))
    names = _names(30)

    default_sim = find_similar(4, X, names, top_k=8)
    fast_sim = find_similar(4, X, names, top_k=8, assume_normalized=True)
    assert_that([n for n, _ in fast_sim]).is_equal_to([n for n, _ in default_sim])
    for (_, a), (_, b) in zip(fast_sim, default_sim):
        assert_that(a).is_close_to(b, 2e-6)

    default_mmr = find_neighbours_mmr(4, X, names, top_k=6, lambda_=0.7)
    fast_mmr = find_neighbours_mmr(4, X, names, top_k=6, lambda_=0.7, assume_normalized=True)
    assert_that([n for n, _ in fast_mmr]).is_equal_to([n for n, _ in default_mmr])

    default_spread = find_neighbours(4, X, names, top_k=5, spread=2)
    fast_spread = find_neighbours(4, X, names, top_k=5, spread=2, assume_normalized=True)
    assert_that([n for n, _ in fast_spread]).is_equal_to([n for n, _ in default_spread])


def test_assume_normalized_near_duplicates_and_block_match_default():
    """near_duplicate_pairs / similarity_matrix under the flag: same pairs, same block
    (float32-ULP tolerance on the cosines)."""
    rng = np.random.default_rng(22)
    base = rng.standard_normal((15, 6)).astype(np.float32)
    X = _l2(np.vstack([base, base[:3] + 1e-4 * rng.standard_normal((3, 6)).astype(np.float32)]))
    names = _names(X.shape[0])

    default_pairs = near_duplicate_pairs(X, names, threshold=0.9)
    fast_pairs = near_duplicate_pairs(X, names, threshold=0.9, assume_normalized=True)
    # Same pair SET; the desc-cosine ORDER may swap among near-equal cosines (the
    # default path divides by a ~1.0 norm — a float32-ULP perturbation, enough to
    # flip a sort between quasi-ties), so membership + per-pair cosine is the
    # honest comparison.
    assert_that(sorted((a, b) for a, b, _ in fast_pairs)).is_equal_to(
        sorted((a, b) for a, b, _ in default_pairs)
    )
    by_pair = {(a, b): c for a, b, c in default_pairs}
    for a, b, c in fast_pairs:
        assert_that(c).is_close_to(by_pair[(a, b)], 2e-6)

    np.testing.assert_allclose(
        similarity_matrix(X, assume_normalized=True), similarity_matrix(X), atol=2e-6
    )


def test_assume_normalized_text_search_matches_default():
    """search_by_text / playlist_from_text honour the flag (the query vector itself is
    still normalized — it comes straight from the text model)."""
    rng = np.random.default_rng(23)
    X = _l2(rng.standard_normal((12, 4)).astype(np.float32))
    names = _names(12)
    embedder = _FakeCLAP(dim=4, overrides={"calm night": 3.0 * X[7]})  # un-normalized on purpose

    default_hits = search_by_text("calm night", X, embedder, names, top_k=5)
    fast_hits = search_by_text("calm night", X, embedder, names, top_k=5, assume_normalized=True)
    assert_that([n for n, _ in fast_hits]).is_equal_to([n for n, _ in default_hits])
    assert_that(fast_hits[0][0]).is_equal_to("t7.mp3")  # query normalization still happened
    for (_, a), (_, b) in zip(fast_hits, default_hits):
        assert_that(a).is_close_to(b, 2e-6)

    assert_that(
        playlist_from_text("calm night", X, embedder, names, top_k=5, assume_normalized=True)
    ).is_equal_to([n for n, _ in fast_hits])


def test_assume_normalized_actually_skips_the_renormalization():
    """With UN-normalized rows the flag must yield raw dot products — proving the
    normalization is genuinely skipped (otherwise every flag test above is vacuous)."""
    X = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)  # norms 2 and 3

    S = similarity_matrix(X, assume_normalized=True)

    np.testing.assert_allclose(np.diag(S), [4.0, 9.0], atol=1e-6)  # raw <x, x> = ||x||²
    np.testing.assert_allclose(np.diag(similarity_matrix(X)), 1.0, atol=1e-5)
