"""Benchmark the text -> playlist path against DEAM ground truth.

The headline capability — type a mood, get tracks — had no measurement at all.
:func:`moodengine.evaluation.evaluate_text_queries` existed with zero callers, so a regression
from a prompt, pooling or recentering change would have been invisible. This is its caller.

**Where relevance comes from, and why it is not circular.** The obvious gold set is the engine's
own mood labels, which would score the labeller against itself. Instead the relevant set for each
query is drawn from DEAM's HUMAN valence/arousal ratings: "a high-energy track" is relevant for the
songs annotators rated most aroused, and so on for the other three poles. Nothing about the ranking
touches those ratings — the query is embedded by the text encoder and tracks are ranked by cosine
in the shared space — so the score measures whether the text tower and the audio tower agree with
people.

Reported per query and macro-averaged: precision@k and average precision, each with a 95 %
percentile bootstrap CI over songs, plus a random-ranking floor. Read the floor first: with the
top quartile relevant, a coin flip already scores P@10 = 0.25, and a number near it means the text
path is doing nothing.

``--embedder`` selects the text-capable backbone (``clap`` or ``mulan``), which is how the two are
compared on retrieval rather than on the regression task in ``bench_valence_arousal.py``.

    uv run --extra models python scripts/bench_text_retrieval.py \
        --data-dir ~/moodengine-bench/deam --limit 400 --embedder clap
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import typer

from moodengine.config import default_config
from moodengine.evaluation import evaluate_text_queries
from moodengine.pipeline import get_embedder, track_embedding

app = typer.Typer(add_completion=False, help=__doc__)

_SCALE_LO, _SCALE_HI = 1.0, 9.0
_BOOTSTRAP_RESAMPLES: int = 2000

# One query per pole of the two gold axes. Deliberately plain phrasings — the point is to measure
# the shipped text path, not to find the prompt that flatters it.
_QUERIES: dict[str, tuple[str, bool]] = {
    # query text: (which gold axis, whether HIGH values are the relevant ones)
    "a high-energy, intense, driving track": ("arousal", True),
    "a calm, quiet, low-energy track": ("arousal", False),
    "a happy, cheerful, positive track": ("valence", True),
    "a sad, gloomy, negative track": ("valence", False),
}


def _load_gold(data_dir: pathlib.Path) -> dict[int, tuple[float, float]]:
    """Map ``song_id -> (arousal01, valence01)`` from DEAM's averaged per-song CSVs."""
    gold: dict[int, tuple[float, float]] = {}
    span = _SCALE_HI - _SCALE_LO
    for csv_path in sorted(data_dir.rglob("static_annotations_averaged_songs_*.csv")):
        frame = pd.read_csv(csv_path, skipinitialspace=True)
        for _, row in frame.iterrows():
            gold[int(row["song_id"])] = (
                (float(row["arousal_mean"]) - _SCALE_LO) / span,
                (float(row["valence_mean"]) - _SCALE_LO) / span,
            )
    return gold


def _select(
    data_dir: pathlib.Path, gold: dict[int, tuple[float, float]], limit: int, seed: int
) -> list[tuple[pathlib.Path, float, float]]:
    """``(path, arousal01, valence01)`` for songs with both audio and gold, seeded-permuted.

    A permutation rather than a prefix: DEAM's song ids run by annotation campaign, so a prefix
    would evaluate one contiguous block of provenance.
    """
    audio_dirs = {p.parent for p in data_dir.rglob("*.mp3")}
    rows: list[tuple[pathlib.Path, float, float]] = []
    for song_id in sorted(gold):
        for d in audio_dirs:
            path = d / f"{song_id}.mp3"
            if path.is_file():
                arousal, valence = gold[song_id]
                rows.append((path, arousal, valence))
                break
    if not limit or limit <= 0 or limit >= len(rows):
        return rows
    order = np.random.default_rng(seed).permutation(len(rows))[:limit]
    return [rows[i] for i in sorted(order)]


def _relevant_sets(
    sel: list[tuple[pathlib.Path, float, float]], quantile: float
) -> dict[str, set[int]]:
    """Turn the gold axes into a relevant-row set per query — the non-circular half.

    A song is relevant to "high energy" when its HUMAN arousal rating sits in the top ``quantile``
    of this sample, and to "low energy" when it sits in the bottom one. Using a quantile of the
    sample rather than an absolute threshold keeps every query's relevant set the same size, so
    precision@k is comparable across queries and the random floor is a single number.
    """
    arousal = np.array([r[1] for r in sel], dtype=np.float64)
    valence = np.array([r[2] for r in sel], dtype=np.float64)
    axes = {"arousal": arousal, "valence": valence}

    relevant: dict[str, set[int]] = {}
    for text, (axis, high_is_relevant) in _QUERIES.items():
        values = axes[axis]
        cut = np.quantile(values, 1.0 - quantile if high_is_relevant else quantile)
        mask = values >= cut if high_is_relevant else values <= cut
        relevant[text] = {int(i) for i in np.flatnonzero(mask)}
    return relevant


class _MemoizedTextEmbedder:
    """``embed_text`` memoized per prompt — the bootstrap's only expensive repetition.

    :func:`evaluate_text_queries` embeds each query on every call, which is right for a general
    caller but is pure waste here: the four queries are fixed by design and the resampling varies
    only the corpus. Without this the honest resample count was unaffordable, and the CIs were
    quietly computed from a twentieth of it. Wrapping the embedder rather than pre-embedding keeps
    the shipped scoring function itself inside the measured path.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        missing = [p for p in prompts if p not in self._cache]
        if missing:
            fresh = np.asarray(self._embedder.embed_text(missing), dtype=np.float32)
            fresh = fresh.reshape(len(missing), -1)
            for prompt, row in zip(missing, fresh, strict=True):
                self._cache[prompt] = row
        return np.stack([self._cache[p] for p in prompts])


def _bootstrap_macro_ci(
    X: np.ndarray,
    embedder,
    sel: list[tuple[pathlib.Path, float, float]],
    quantile: float,
    k: int,
    seed: int,
    resamples: int = _BOOTSTRAP_RESAMPLES,
) -> dict[str, tuple[float, float]]:
    """95 % CI on the macro scores by resampling SONGS.

    Resampling songs (not queries) is what the interval has to be over: the queries are fixed by
    design, while the corpus is the sample. Relevance is recomputed inside each draw, because the
    quantile cut is a property of the resampled corpus — freezing it would leak the full sample
    into every replicate and understate the spread.

    ``resamples`` draws are actually taken. Each re-ranks the corpus against every query, but the
    query embeddings are memoized, so the per-draw cost is numpy alone.
    """
    n = len(sel)
    if n < 8:
        return {
            "macro_precision_at_k": (float("nan"), float("nan")),
            "macro_map": (float("nan"),) * 2,
        }
    rng = np.random.default_rng(seed)
    cached = _MemoizedTextEmbedder(embedder)
    precisions: list[float] = []
    maps: list[float] = []
    for _ in range(int(resamples)):
        draw = rng.integers(0, n, size=n)
        sub = [sel[i] for i in draw]
        scores = evaluate_text_queries(_relevant_sets(sub, quantile), X[draw], cached, k=k)
        precisions.append(float(scores["macro_precision_at_k"]))
        maps.append(float(scores["macro_map"]))
    return {
        "macro_precision_at_k": (
            float(np.quantile(precisions, 0.025)),
            float(np.quantile(precisions, 0.975)),
        ),
        "macro_map": (float(np.quantile(maps, 0.025)), float(np.quantile(maps, 0.975))),
    }


@app.command()
def main(
    data_dir: pathlib.Path = typer.Option(
        pathlib.Path.home() / "moodengine-bench" / "deam", "--data-dir", help="DEAM root."
    ),
    embedder: str = typer.Option(
        "clap", "--embedder", help="Text-capable backbone: 'clap' or 'mulan'."
    ),
    limit: int = typer.Option(400, "--limit", help="Max songs (0 = all); bounds embedding cost."),
    k: int = typer.Option(10, "--k", help="Cut-off for precision@k."),
    quantile: float = typer.Option(
        0.25, "--quantile", help="Gold quantile counted as relevant per pole."
    ),
    seed: int = typer.Option(0, "--seed", help="Seeds the subset permutation and the bootstrap."),
    force: bool = typer.Option(False, "--force", help="Recompute embeddings even on a cache hit."),
    out: pathlib.Path | None = typer.Option(None, "--out", help="Write results JSON here."),
) -> None:
    """Score text -> playlist retrieval against DEAM's human ratings."""
    config = default_config()
    gold = _load_gold(data_dir)
    sel = _select(data_dir, gold, limit, seed)
    if not sel:
        typer.echo(f"No annotated audio under {data_dir}. Run fetch_deam.py first.")
        raise typer.Exit(code=1)

    typer.echo(f"Scoring {len(sel)} DEAM songs with the {embedder!r} text tower.")
    model = get_embedder(embedder, config)
    X = np.vstack(
        [track_embedding(model, path, config, force=force).reshape(-1) for path, _, _ in sel]
    ).astype(np.float32)

    relevant = _relevant_sets(sel, quantile)
    scores = evaluate_text_queries(relevant, X, model, k=k)
    ci = _bootstrap_macro_ci(X, model, sel, quantile, k, seed)

    # The floor a coin flip already reaches: with the top `quantile` relevant, a random ranking
    # scores that in expectation. A result near it means the text path contributes nothing.
    floor = float(quantile)
    typer.echo(
        f"\ntext -> playlist ({embedder}, n={len(sel)}, k={k}, relevant = top {quantile:.0%})"
    )
    for text, per in scores["per_query"].items():
        typer.echo(
            f"  P@{k}={per['precision_at_k']:.3f}  AP={per['average_precision']:.3f}  "
            f"(n_rel={per['n_relevant']})  {text}"
        )
    p_lo, p_hi = ci["macro_precision_at_k"]
    m_lo, m_hi = ci["macro_map"]
    typer.echo(
        f"\n  macro P@{k} = {scores['macro_precision_at_k']:.3f} [{p_lo:.3f}, {p_hi:.3f}]"
        f"   (random floor {floor:.3f})"
    )
    typer.echo(f"  macro MAP  = {scores['macro_map']:.3f} [{m_lo:.3f}, {m_hi:.3f}]")

    if out is not None:
        payload = {
            "n": len(sel),
            "embedder": embedder,
            "k": k,
            "quantile": quantile,
            "seed": seed,
            "random_floor": floor,
            "macro_precision_at_k": scores["macro_precision_at_k"],
            "macro_precision_at_k_ci": list(ci["macro_precision_at_k"]),
            "macro_map": scores["macro_map"],
            "macro_map_ci": list(ci["macro_map"]),
            "per_query": scores["per_query"],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"\nWrote {out}")


if __name__ == "__main__":
    app()
