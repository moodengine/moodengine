"""Implicit-feedback weak labels from listening behaviour — the personalization foundation.

Turns *how* a track was listened to (play / skip / seek / complete / replay) into a signed, bounded
weak label per track, WITHOUT any explicit rating. Inspired by implicit-feedback recommenders —
weighted matrix factorization (Hu-Koren-Volinsky 2008, confidence ``c = 1 + α·r``) and BPR
(Rendle 2009, relative preference) — but transposed to a single-user setting: no cohort, no matrix
factorization, just a per-track engagement weight that downstream consumers (e.g. few-shot mood
prototypes averaged over member tracks) can use to up/down-weight a track's contribution.

Signal design:
  * ``complete`` / ``replay`` → positive (the listener stayed / came back).
  * ``skip`` → negative, and STRONGLY so when early (before ``EARLY_SKIP_POS`` of the track);
    a late skip is close to neutral (they heard most of it).
  * ``play`` / ``seek`` → pure exposure, weight 0 (no preference signal).

Pure Python (stdlib ``math``), torch-free, deterministic, no I/O.
"""

from __future__ import annotations

import math
from typing import Iterable

# Per-event base weak label (before position weighting). Unknown events → 0 (neutral exposure).
BASE_WEIGHTS: dict[str, float] = {
    "complete": 1.0,
    "replay": 1.0,
    "play": 0.0,
    "seek": 0.0,
    "skip": -1.0,
}

# A skip before this fraction of the track is a full-strength negative; later skips attenuate to ~0.
EARLY_SKIP_POS: float = 0.2


def implicit_weight(event: str, position: float) -> float:
    """Signed weak label for ONE event, in ``[-1, 1]``.

    ``complete`` / ``replay`` → ``+1``; ``play`` / ``seek`` (and unknown events) → ``0``; ``skip`` →
    ``-1`` when it lands before ``EARLY_SKIP_POS``, attenuating linearly to ``0`` by the track's end
    (a late skip is nearly neutral — most of the track was heard). ``position`` is clamped to
    ``[0, 1]``. Deterministic and pure."""
    base = BASE_WEIGHTS.get(event, 0.0)
    if event != "skip" or base == 0.0:
        return base
    pos = min(max(float(position), 0.0), 1.0)
    if pos < EARLY_SKIP_POS:
        factor = 1.0  # full-strength negative for an early skip
    else:  # attenuate from 1.0 at EARLY_SKIP_POS to 0.0 at the track's end
        factor = max(0.0, 1.0 - (pos - EARLY_SKIP_POS) / (1.0 - EARLY_SKIP_POS))
    return base * factor


def aggregate_implicit(
    events: Iterable[tuple[str, str, float]], *, alpha: float = 0.5
) -> dict[str, float]:
    """Aggregate ``(track_id, event, position)`` triples into a per-track signed weight in ``[-1, 1]``.

    WMF-inspired: sum each track's per-event weak labels, then squash with ``tanh(alpha · Σ)`` so the
    weight saturates smoothly and can never leave ``[-1, 1]`` however many events accrue. Only tracks
    with at least one WEIGHING event (non-zero :func:`implicit_weight`) appear in the result — a
    track that was merely exposed (only ``play`` / ``seek``) is absent, never injected as a fake 0
    (anti-fabrication). No events → empty dict. Deterministic; order-independent."""
    sums: dict[str, float] = {}
    for track_id, event, position in events:
        w = implicit_weight(event, position)
        if w != 0.0:  # exposure-only tracks never enter the aggregate
            sums[track_id] = sums.get(track_id, 0.0) + w
    return {track_id: math.tanh(alpha * total) for track_id, total in sums.items()}
