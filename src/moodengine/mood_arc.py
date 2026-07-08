"""Intra-track mood arc — score the CLAP embeddings of a track *per segment* into a mood curve.

A track is not a single mood: the pipeline pools its per-segment CLAP clip embeddings into one
track vector (``pooling.pool_clap`` = mean of segments → L2-norm) and only exposes the *global*
triptych. Here we keep the per-segment embeddings and score each one against the SAME label matrix,
through the EXACT same triptych (``sims → recenter → softmax``) — a mood trajectory in time (a calm
intro → an energetic drop).

Pure numpy, torch-free, deterministic. It owns no new algorithm: it reuses ``labeling.l2_normalize``
/ ``recenter_similarities`` / ``softmax`` / ``DEFAULT_TEMPERATURE`` and ``io_audio.segment_waveform``.
The per-segment embeddings come from the same ``embedder.extract`` the track pool is built from, so
``l2_normalize(mean(seg_embs)) == pool_clap(seg_embs)`` holds by construction.

Calibration (transparency): a caller passes ``mean_cosine`` = the *library* mean-cosine vector (the
same one ``recenter_similarities`` subtracts on the whole ``X_clap``), so each segment's scores live
on the same scale as the track-level ``top_score`` — a segment is "energetic" if it exceeds the
library mean, not the (degenerate at <5 segments) mean of the track's own segments. Passing
``mean_cosine = sims.mean(axis=0)`` instead recovers the self-contained recentering and is
byte-identical to ``recenter_similarities(sims) → softmax``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moodengine.config import Config
from moodengine.io_audio import _capped_indices, segment_waveform
from moodengine.labeling import (
    DEFAULT_TEMPERATURE,
    l2_normalize,
    recenter_similarities,
    softmax,
)


@dataclass(frozen=True)
class SegmentArc:
    """A track's per-segment mood scores. ``probs`` is the dense ``(n_seg, n_moods)`` softmax curve;
    ``top_moods`` / ``top_scores`` are the argmax mood + its probability per segment; ``top3`` is the
    ranked ``(mood, prob)`` list per segment (length ``top_k``). Frozen (inputs never mutated); compare
    fields explicitly (the numpy ``probs`` field makes a bare ``==`` ambiguous)."""

    probs: np.ndarray  # (n_seg, n_moods) float32, rows sum to 1
    top_moods: list[str]
    top_scores: list[float]
    top3: list[list[tuple[str, float]]]


def score_segment_arc(
    seg_embs: np.ndarray,
    label_matrix: np.ndarray,
    mood_names: list[str],
    *,
    mean_cosine: np.ndarray | None = None,
    recenter: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = 3,
) -> SegmentArc:
    """Score per-segment CLAP embeddings into a mood arc via the EXACT triptych.

    ``seg_embs`` ``(n_seg, d)`` are raw (un-normalized) per-segment CLAP clip embeddings;
    ``label_matrix`` ``(n_moods, d)`` is the L2-normalized ensembled prompt matrix; ``mood_names``
    labels its rows. Computes, segment by segment:

      ``X = l2_normalize(seg_embs, axis=1)`` → ``sims = X @ label_matrix.T`` → recenter → softmax.

    Recentering:
      * ``not recenter`` → ``rec = sims`` (no recentering);
      * ``mean_cosine is not None`` → ``rec = sims − mean_cosine`` (the **library-calibration** path:
        pass ``(X_clap @ label_matrix.T).mean(axis=0)`` so segment scores compare to the track score);
      * else → ``rec = recenter_similarities(sims, enable=True)`` (the **self-contained** path: the
        mean over this track's own segments, skipped for <5 segments — see ``recenter_similarities``).

    With ``mean_cosine = sims.mean(axis=0)`` and ``n_seg ≥ 5`` the output is byte-identical to
    ``recenter_similarities(sims) → softmax(...)``. Deterministic; the inputs are never mutated.
    """
    names = list(mood_names)
    n_moods = len(names)
    X = np.asarray(seg_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.size == 0:  # no segments → an honest empty arc (never a fabricated row)
        return SegmentArc(
            probs=np.zeros((0, n_moods), dtype=np.float32), top_moods=[], top_scores=[], top3=[]
        )
    if X.ndim != 2:
        raise ValueError(f"seg_embs must be (n_seg, d); got shape {X.shape}")

    X = l2_normalize(X, axis=1)
    sims = X @ np.asarray(label_matrix, dtype=np.float32).T  # (n_seg, n_moods) cosine
    if not recenter:
        rec = sims
    elif mean_cosine is not None:
        rec = sims - np.asarray(mean_cosine, dtype=np.float32).reshape(1, -1)
    else:
        rec = recenter_similarities(sims, enable=True)
    probs = softmax(rec, temperature=temperature, axis=1).astype(np.float32)  # (n_seg, n_moods)

    k = max(0, min(int(top_k), n_moods))
    top_moods: list[str] = []
    top_scores: list[float] = []
    top3: list[list[tuple[str, float]]] = []
    for row in probs:
        order = np.argsort(-row, kind="stable")  # ties break toward the earliest mood index
        top_moods.append(names[int(order[0])])
        top_scores.append(float(row[int(order[0])]))
        top3.append([(names[int(j)], float(row[int(j)])) for j in order[:k]])
    return SegmentArc(probs=probs, top_moods=top_moods, top_scores=top_scores, top3=top3)


def segment_embeddings(waveform: np.ndarray, embedder, config: Config) -> list[np.ndarray]:
    """Per-segment (raw, un-pooled) CLAP clip embeddings for one decoded waveform.

    The pre-pool half of ``pipeline.track_embedding_from_waveform``: split into the SAME fixed windows
    (``io_audio.segment_waveform``) the track pool is built from, then ``embedder.extract`` each →
    a list of ``(d,)`` float32 vectors. Pooling those with ``pool_clap`` reproduces the track vector,
    so the arc and the global triptych start from identical raw material. ``embedder.sample_rate`` is
    the rate the waveform must already be at (the caller decodes to it)."""
    sr = int(embedder.sample_rate)
    segments = segment_waveform(np.asarray(waveform, dtype=np.float32), sr, config)
    return [np.asarray(embedder.extract(seg, sr), dtype=np.float32).reshape(-1) for seg in segments]


def segment_bounds(n_samples: int, sr: int, config: Config) -> list[tuple[float, float]]:
    """Temporal bounds ``(t_start, t_end)`` in seconds for the fixed windows of a waveform of
    ``n_samples`` samples — a torch-free mirror of ``segment_waveform``'s geometry (same ``seg_len`` /
    ``overlap`` / ``min_segment`` / cap rules), for display alignment.

    Returns exactly ``len(segment_waveform(y, sr, config))`` spans, one per segment, in order (pinned
    equivalent by a test). Every value is COMPUTED from the real sample count — never a fabricated
    boundary. Empty for an empty waveform."""
    n = int(n_samples)
    if n <= 0:
        return []
    seg_len = max(1, int(round(config.segment_seconds * sr)))
    overlap_len = max(0, int(round(config.overlap_seconds * sr)))
    overlap_len = min(overlap_len, seg_len - 1)  # guarantee forward progress
    step = seg_len - overlap_len
    min_len = int(round(config.min_segment_seconds * sr))

    # Whole track shorter than the minimum: a single short window (never lost) — mirrors segment_waveform.
    if n < min_len:
        return [(0.0, n / sr)]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start + seg_len
        if end <= n:  # a full window
            spans.append((start, end))
        else:  # trailing partial: keep only if long enough
            if (n - start) >= min_len:
                spans.append((start, n))
            break
        start += step
    if not spans:  # safety net (parity with segment_waveform): a non-empty waveform yields ≥1 span
        spans.append((0, n))

    cap = config.max_segments_per_track
    keep = _capped_indices(len(spans), cap, config.segment_selection)
    if len(keep) != len(spans):
        spans = [spans[i] for i in keep]

    return [(s / sr, e / sr) for (s, e) in spans]
