"""Rigorous, falsifiable evaluation of the mood pipeline (torch-free).

The pipeline's outputs (mood labels, energy/valence axes, text-query playlists)
are only meaningful if we can *measure* them. This module supplies light,
self-contained metrics that lean on already-computed embeddings / DataFrames:

  * **Axis self-consistency** — does similarity to a pole prompt (e.g.
    "energetic") rank tracks the same way as the energy attribute axis? An AUC
    near 1.0 means the two views of the same construct agree.
  * **Retrieval quality** — precision@k and average precision for text->playlist
    search against hand-labelled relevant tracks.
  * **Gold comparison** — top-mood accuracy and energy/valence correlation
    against a human-labelled gold JSON (produced by ``viz.build_labeling_ui``),
    which is what makes the rest of the system falsifiable.

Everything here is numpy / pandas / sklearn. ``scipy`` is used only when present
(Spearman); otherwise we fall back to numpy ``corrcoef`` on ranks. The only
calls that could touch a heavy model are ``clap_embedder.embed_text`` inside
:func:`evaluate_text_queries`, mirroring :mod:`moodengine.labeling`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moodengine._typing import SupportsEmbedText


def axis_ranking_auc(scores: np.ndarray, axis_values: np.ndarray) -> float:
    """AUC that ``scores`` rank tracks the same way as ``axis_values``.

    A self-consistency check: ``scores`` is e.g. each track's similarity to an
    "energetic" prompt and ``axis_values`` is the energy attribute coordinate.
    ``axis_values`` is split at its median into hi (>= median) / lo labels and we
    report ``roc_auc_score(labels, scores)`` -> how well ``scores`` separates the
    high half from the low half. Returns ``0.5`` (chance) when the split is
    degenerate (one class only, or fewer than 2 samples). Never raises.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    v = np.asarray(axis_values, dtype=np.float64).ravel()
    n = min(s.shape[0], v.shape[0])
    if n < 2:
        return 0.5
    s, v = s[:n], v[:n]
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(v))):
        mask = np.isfinite(s) & np.isfinite(v)
        s, v = s[mask], v[mask]
        if s.shape[0] < 2:
            return 0.5
    labels = (v >= np.median(v)).astype(int)
    if labels.min() == labels.max():  # all one class -> undefined
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, s))
    except Exception:
        return 0.5


def retrieval_precision_at_k(ranked_idx: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-``k`` ranked items that are in ``relevant``.

    ``ranked_idx`` is a ranking (best first) of track indices; ``relevant`` is the
    gold set. Returns hits-in-top-k divided by ``k`` (the conventional P@k, which
    penalises short rankings). Returns ``0.0`` when ``k <= 0``.
    """
    k = int(k)
    if k <= 0:
        return 0.0
    top = list(ranked_idx)[:k]
    rel = set(relevant)
    hits = sum(1 for i in top if i in rel)
    return float(hits) / float(k)


def _average_precision(ranked_idx: list[int], relevant: set[int]) -> float:
    """Average precision of a ranking against ``relevant`` (0.0 if none relevant)."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    cumulative = 0.0
    for rank, idx in enumerate(ranked_idx, start=1):
        if idx in rel:
            hits += 1
            cumulative += hits / rank
    return float(cumulative / len(rel)) if hits else 0.0


def evaluate_text_queries(
    queries: dict[str, set[int]],
    X: np.ndarray,
    clap_embedder: SupportsEmbedText,
    k: int = 10,
) -> dict:
    """Score text->playlist retrieval against gold relevant sets.

    ``queries`` maps a text query to the set of relevant track indices into ``X``
    (CLAP track embeddings, assumed L2-normalized so dot == cosine). Each query is
    embedded via ``clap_embedder.embed_text([query])``, tracks are ranked by
    cosine similarity, and we compute precision@k and average precision. Returns
    ``{"per_query": {query: {"precision_at_k", "average_precision", "n_relevant"}},
    "macro_precision_at_k": float, "macro_map": float, "k": k}``. Robust to empty
    ``X`` / empty ``queries`` (macros default to 0.0).
    """
    Xa = np.asarray(X, dtype=np.float32)
    if Xa.ndim == 1:
        Xa = Xa[None, :]
    per_query: dict[str, dict] = {}
    if Xa.size == 0 or Xa.shape[0] == 0 or not queries:
        return {"per_query": per_query, "macro_precision_at_k": 0.0, "macro_map": 0.0, "k": int(k)}

    n = Xa.shape[0]
    for text, relevant in queries.items():
        q = np.asarray(clap_embedder.embed_text([text]), dtype=np.float32).ravel()
        # L2-normalize the query so the dot product is a cosine similarity.
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = Xa @ q  # (n,)
        ranked = np.argsort(-sims, kind="stable").tolist()
        rel = {int(i) for i in relevant if 0 <= int(i) < n}
        per_query[text] = {
            "precision_at_k": retrieval_precision_at_k(ranked, rel, k),
            "average_precision": _average_precision(ranked, rel),
            "n_relevant": len(rel),
        }

    pks = [d["precision_at_k"] for d in per_query.values()]
    aps = [d["average_precision"] for d in per_query.values()]
    return {
        "per_query": per_query,
        "macro_precision_at_k": float(np.mean(pks)) if pks else 0.0,
        "macro_map": float(np.mean(aps)) if aps else 0.0,
        "k": int(k),
    }


def load_gold(path) -> dict:
    """Load a gold-label JSON, or return ``{}`` if it is missing / unreadable.

    Expected shape: ``{filename: {"moods": [...], "energy": float, "valence":
    float}}`` (as written by ``viz.build_labeling_ui``). Never raises: a missing
    file, bad JSON, or a non-dict top level all yield ``{}``.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; ``nan`` if undefined (constant input / <2 points)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape[0] < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via scipy when available, else Pearson on ranks."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape[0] < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho = spearmanr(a, b).correlation
        return float(rho)
    except Exception:
        # numpy fallback: Pearson on the rank-transformed values.
        ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
        rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
        return _pearson(ra, rb)


def concordance_correlation_coefficient(pred: np.ndarray, gold: np.ndarray) -> tuple[float, int]:
    """Lin's concordance correlation coefficient between ``pred`` and ``gold``.

    The standard agreement metric for valence/arousal regression: unlike Pearson
    (invariant to shift and scale), CCC additionally penalises any location or
    scale mismatch, so ``pred`` must be on the SAME scale as ``gold`` to score
    well. ``CCC = 2·cov(p, g) / (var(p) + var(g) + (mean(p) − mean(g))²)`` with
    population (1/N) moments, in ``[-1, 1]`` (1 = perfect agreement on the y=x
    line). Returns ``(ccc, support)`` with ``support = min(len(pred), len(gold))``
    (the family convention of never reporting a metric without its sample count);
    ``(nan, n)`` when ``n < 2`` or both series are constant (denominator 0).
    Non-finite pairs are dropped before scoring. Never raises.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    g = np.asarray(gold, dtype=np.float64).ravel()
    n = min(p.shape[0], g.shape[0])
    if n < 2:
        return float("nan"), n
    p, g = p[:n], g[:n]

    mask = np.isfinite(p) & np.isfinite(g)
    p, g = p[mask], g[mask]
    if p.shape[0] < 2:
        return float("nan"), n

    mp, mg = float(p.mean()), float(g.mean())
    var_p = float(((p - mp) ** 2).mean())
    var_g = float(((g - mg) ** 2).mean())
    cov = float(((p - mp) * (g - mg)).mean())
    denom = var_p + var_g + (mp - mg) ** 2
    if denom == 0.0:  # both series constant -> agreement undefined
        return float("nan"), n
    return float(2.0 * cov / denom), n


def evaluate_against_gold(df: pd.DataFrame, gold: dict) -> dict:
    """Compare predicted labels/axes in ``df`` to a human gold set.

    Matches rows of ``df`` (by its ``filename`` column) to keys of ``gold``. For
    the overlap it reports top-mood accuracy (``df.top_mood`` present in the
    track's gold ``moods`` list) and Pearson + Spearman + CCC of predicted
    ``energy`` / ``valence`` against gold ``energy`` / ``valence``. Returns
    ``{"n_overlap", "top_mood_accuracy", "energy_pearson", "energy_spearman",
    "energy_ccc", "valence_pearson", "valence_spearman", "valence_ccc"}``; ``{}``
    when there is no overlap or ``df`` lacks a ``filename`` column. Correlations
    are ``nan`` when the gold field is absent or constant. CCC assumes gold and
    prediction share a scale (both in ``[0, 1]`` for the pipeline's axes). Never
    raises.
    """
    if not isinstance(df, pd.DataFrame) or "filename" not in df.columns or not gold:
        return {}

    names = df["filename"].astype(str).tolist()
    rows = [(i, name) for i, name in enumerate(names) if name in gold]
    if not rows:
        return {}

    summary: dict = {"n_overlap": len(rows)}

    # Top-mood accuracy.
    if "top_mood" in df.columns:
        top_moods = df["top_mood"].astype(str).tolist()
        correct = 0
        for i, name in rows:
            gold_moods = gold[name].get("moods", []) if isinstance(gold[name], dict) else []
            if top_moods[i] in set(gold_moods):
                correct += 1
        summary["top_mood_accuracy"] = float(correct) / float(len(rows))

    # Energy / valence correlations.
    for axis in ("energy", "valence"):
        pred, ref = [], []
        if axis in df.columns:
            col = df[axis].tolist()
            for i, name in rows:
                gv = gold[name].get(axis) if isinstance(gold[name], dict) else None
                if gv is not None:
                    try:
                        ref.append(float(gv))
                        pred.append(float(col[i]))
                    except (TypeError, ValueError):
                        continue
        if len(pred) >= 2:
            p_arr, r_arr = np.array(pred), np.array(ref)
            summary[f"{axis}_pearson"] = _pearson(p_arr, r_arr)
            summary[f"{axis}_spearman"] = _spearman(p_arr, r_arr)
            summary[f"{axis}_ccc"] = concordance_correlation_coefficient(p_arr, r_arr)[0]
        else:
            summary[f"{axis}_pearson"] = float("nan")
            summary[f"{axis}_spearman"] = float("nan")
            summary[f"{axis}_ccc"] = float("nan")

    return summary


# --- ranking metrics (binary relevance) -------------------------------------
# Composed by callers alongside the existing retrieval_precision_at_k / _average_precision to
# score retrieval gold sets (text->playlist, similar-track). Pure numpy, torch-free.


def ndcg_at_k(ranked_idx: list[int], relevant: set[int], k: int) -> float:
    """Normalized DCG at ``k`` for a binary-relevance ranking.

    ``ranked_idx`` is a ranking (best first) of item indices; ``relevant`` the gold set. Gain is
    binary (``rel ∈ {0, 1}``); discount is ``1 / log2(rank + 1)`` (rank 1-based, so the top item
    is undiscounted). Returns ``DCG@k / IDCG@k`` where the ideal ranking places every relevant
    item first. Returns ``0.0`` when ``k <= 0`` or nothing is relevant (IDCG would be 0). Never
    raises. ``1.0`` iff the top ``k`` are exactly the relevant items (up to ``min(|relevant|, k)``).
    """
    k = int(k)
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    top = list(ranked_idx)[:k]
    dcg = sum(1.0 / np.log2(rank + 1) for rank, idx in enumerate(top, start=1) if idx in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked_idx: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the gold set retrieved within the top ``k``: ``|top_k ∩ relevant| / |relevant|``.

    Returns ``0.0`` when ``relevant`` is empty or ``k <= 0``. Monotonically non-decreasing in ``k``
    (a larger window can only find more of a fixed gold set). Never raises.
    """
    k = int(k)
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    top = set(list(ranked_idx)[:k])
    return float(len(top & rel)) / float(len(rel))


# --- classification / clustering / calibration / drift ----------------------
# Each returns ``(value, support)`` so a caller never reports a metric without its sample count
# (anti-fabrication). sklearn / scipy are imported lazily with a numpy fallback, mirroring
# :func:`_spearman` — the module import stays light and torch-free.


def macro_f1(y_true: list, y_pred: list, labels: list | None = None) -> tuple[float, int]:
    """Macro-averaged F1 over ``labels`` (or the union of observed labels).

    Delegates to ``sklearn.metrics.f1_score(average="macro", zero_division=0)``. Returns
    ``(f1, support)`` with ``support = len(y_true)``. ``(0.0, 0)`` on empty / length-mismatched
    input. Never raises.
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return 0.0, 0
    from sklearn.metrics import f1_score

    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return f1, n


def nmi(labels_a, labels_b) -> tuple[float, int]:
    """Normalized mutual information between two label assignments (clusterings).

    Delegates to ``sklearn.metrics.normalized_mutual_info_score`` (same sklearn dependency as
    :mod:`moodengine.cluster`'s bootstrap stability). Returns ``(nmi, support=n)``; ``1.0`` for
    identical partitions (up to relabeling), ``≈0.0`` for independent ones. ``(0.0, 0)`` on empty /
    length-mismatched input. Never raises.
    """
    a = list(labels_a)
    b = list(labels_b)
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0, 0
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(a, b)), n


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, int]:
    """Expected Calibration Error (Guo et al. 2017), equal-width binning of ``confidences`` ∈ [0, 1].

    ``correct`` is a 0/1 array — whether each top-1 prediction was right. Bins the confidences into
    ``n_bins`` equal-width buckets and returns ``ECE = Σ_b (n_b / n) · |acc_b − conf_b|`` — a
    perfectly calibrated model → ``≈0``. Returns ``(ece, support=n)``; ``(0.0, 0)`` on empty input.
    Never raises.
    """
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correct, dtype=np.float64).ravel()
    n = min(conf.shape[0], corr.shape[0])
    if n == 0:
        return 0.0, 0
    conf, corr = conf[:n], corr[:n]
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # Assign each confidence to a bucket in [0, n_bins-1] (interior edges only; clip for safety).
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, int(n_bins) - 1)
    ece = 0.0
    for b in range(int(n_bins)):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        acc_b = float(corr[mask].mean())
        conf_b = float(conf[mask].mean())
        ece += (m / n) * abs(acc_b - conf_b)
    return float(ece), n


def procrustes_disparity(A: np.ndarray, B: np.ndarray) -> tuple[float, int]:
    """Procrustes disparity M² between two point clouds over shared rows (2D-layout drift).

    ``A`` and ``B`` are ``(n, d)`` with rows aligned (same tracks, same order). Uses
    ``scipy.spatial.procrustes`` when present — its disparity M² is the sum of squared differences
    after optimally translating / uniformly scaling / rotating the standardized clouds — and a
    numpy fallback otherwise (mean-center, Frobenius-normalize, optimal orthogonal alignment via
    SVD, then squared Frobenius of the residual). Same try/except-scipy pattern as :func:`_spearman`.
    Returns ``(disparity, n_shared)``; ``0.0`` for identical clouds, ``(nan, n)`` when ``n < 2``
    (undefined). Never raises.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = min(A.shape[0], B.shape[0]) if (A.ndim == 2 and B.ndim == 2) else 0
    if n < 2:
        return float("nan"), n
    A, B = A[:n], B[:n]
    try:
        from scipy.spatial import procrustes

        _, _, disparity = procrustes(A, B)
        return float(disparity), n
    except Exception:

        def _standardize(M: np.ndarray) -> np.ndarray:
            M = M - M.mean(axis=0, keepdims=True)
            norm = float(np.linalg.norm(M))
            return M / norm if norm > 0 else M

        As, Bs = _standardize(A), _standardize(B)
        u, w, vt = np.linalg.svd(Bs.T @ As)
        rot = u @ vt  # optimal orthogonal map aligning Bs to As (Kabsch/orthogonal Procrustes)
        # scipy applies the OPTIMAL uniform scale too, giving disparity 1 - s^2 (s = Σ singular
        # values); without it the residual would be 2(1 - s) — a different metric that diverges from
        # scipy by up to 2x. Scale Bs by s so the fallback matches scipy's disparity M^2.
        scale = float(w.sum())
        return float(np.sum((As - scale * (Bs @ rot)) ** 2)), n
