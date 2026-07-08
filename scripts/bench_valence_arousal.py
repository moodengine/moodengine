"""Benchmark the engine's energy/valence axes against DEAM ground-truth ratings.

This is the yardstick the production-readiness audit found missing: the engine had no
way to measure whether a change improves or regresses mood quality, so every "expected
gain" was unfalsifiable. It compares the pipeline against human valence/arousal ratings
(DEAM, fetched by ``fetch_deam.py``) two complementary ways:

* ``--mode zeroshot`` — the actual product path: CLAP zero-shot ``attribute_scores``
  (energy, valence) correlated with the gold ratings. Measures the labelling / prompt /
  recentering stack and any change to how CLAP pools a track's segments.
* ``--mode probe`` — a ridge linear probe (cross-validated) on frozen ``--embedder``
  embeddings regressed onto the gold ratings. This is the standard MARBLE-style protocol
  and the only way to see the quality of the MERT space itself, so it is what reveals an
  embedding-front-end change (e.g. the decode sample rate) that never touches the CLAP
  axes. Reports out-of-fold correlations, so it cannot overfit its own score.

Gold mapping: DEAM arousal -> the engine's ``energy`` axis, DEAM valence -> ``valence``;
the 1-9 rating scale is affine-mapped to [0, 1] so CCC (which penalises scale/shift) is
meaningful. Results (Pearson / Spearman / CCC per axis) are printed and written to JSON so
a before/after diff across an engine change is a file comparison. Runs on CPU; use
``--limit`` to bound the (dominant) embedding cost — a few hundred tracks already gives a
stable correlation.

    uv run --extra models python scripts/bench_valence_arousal.py \
        --data-dir ~/moodengine-bench/deam --mode both --embedder mert --limit 150
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import typer

from moodengine.config import default_config
from moodengine.evaluation import evaluate_against_gold
from moodengine.labeling import attribute_scores, l2_normalize
from moodengine.pipeline import get_embedder, track_embedding

app = typer.Typer(add_completion=False, help=__doc__)

# DEAM static ratings are on a 1-9 scale; the engine's axes live in [0, 1].
_SCALE_LO, _SCALE_HI = 1.0, 9.0


def _load_gold(data_dir: pathlib.Path) -> dict[int, tuple[float, float]]:
    """Map ``song_id -> (energy01, valence01)`` from DEAM's averaged per-song CSVs.

    Reads every ``static_annotations_averaged_songs_*.csv`` under ``data_dir`` (column
    names carry leading spaces, hence ``skipinitialspace``), maps DEAM arousal onto the
    engine's energy axis and valence onto valence, and affine-scales 1-9 -> [0, 1].
    """
    gold: dict[int, tuple[float, float]] = {}
    span = _SCALE_HI - _SCALE_LO
    for csv_path in sorted(data_dir.rglob("static_annotations_averaged_songs_*.csv")):
        frame = pd.read_csv(csv_path, skipinitialspace=True)
        for _, row in frame.iterrows():
            energy01 = (float(row["arousal_mean"]) - _SCALE_LO) / span
            valence01 = (float(row["valence_mean"]) - _SCALE_LO) / span
            gold[int(row["song_id"])] = (energy01, valence01)
    return gold


def _select(
    data_dir: pathlib.Path, gold: dict[int, tuple[float, float]], limit: int
) -> list[tuple[str, pathlib.Path, float, float]]:
    """Return ``(filename, path, energy01, valence01)`` for songs with both audio and gold.

    Sorted by numeric song id and truncated to ``limit`` (0 = all) so a run is
    deterministic and a subset is a stable prefix, not a random sample.
    """
    audio_dirs = {p.parent for p in data_dir.rglob("*.mp3")}
    rows: list[tuple[str, pathlib.Path, float, float]] = []
    for song_id in sorted(gold):
        for d in audio_dirs:
            path = d / f"{song_id}.mp3"
            if path.is_file():
                energy01, valence01 = gold[song_id]
                rows.append((path.name, path, energy01, valence01))
                break
    return rows[:limit] if limit and limit > 0 else rows


def _embed(paths: list[pathlib.Path], embedder_name: str, config, force: bool) -> np.ndarray:
    """Embed ``paths`` into ``(n, d)`` for ``'mert'`` / ``'clap'`` / ``'fused'`` (cached).

    ``fused`` block-L2-normalizes the MERT and CLAP matrices and stacks them scaled by
    ``config.fusion_weights`` — the same construction as the pipeline's fused space.
    Progress is printed because embedding is the dominant, minutes-scale cost on CPU.
    """
    if embedder_name == "fused":
        xm = _embed(paths, "mert", config, force)
        xc = _embed(paths, "clap", config, force)
        w_m, w_c = config.fusion_weights
        return np.hstack(
            [l2_normalize(xm, axis=1) * float(w_m), l2_normalize(xc, axis=1) * float(w_c)]
        ).astype(np.float32)

    embedder = get_embedder(embedder_name, config)
    vectors: list[np.ndarray] = []
    for i, path in enumerate(paths, start=1):
        vectors.append(track_embedding(embedder, path, config, force=force).reshape(-1))
        if i % 20 == 0 or i == len(paths):
            typer.echo(f"    {embedder_name}: embedded {i}/{len(paths)}")
    return np.vstack(vectors).astype(np.float32)


def _probe_oof(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int) -> np.ndarray:
    """Cross-validated out-of-fold ridge predictions for target ``y`` from features ``X``.

    Each fold fits ``RidgeCV`` (alpha chosen by efficient leave-one-out over a log grid) on
    the train rows and predicts the held-out rows, so the returned vector is never used to
    fit the model that produced it — the correlation computed on it is honest.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    preds = np.full(y.shape[0], np.nan, dtype=np.float64)
    kf = KFold(n_splits=min(n_splits, y.shape[0]), shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return preds


def _score(pred_energy: np.ndarray, pred_valence: np.ndarray, sel) -> dict:
    """Pearson/Spearman/CCC of predicted vs gold energy & valence via evaluate_against_gold."""
    df = pd.DataFrame(
        {"filename": [r[0] for r in sel], "energy": pred_energy, "valence": pred_valence}
    )
    gold = {r[0]: {"energy": r[2], "valence": r[3]} for r in sel}
    return evaluate_against_gold(df, gold)


def _print(title: str, metrics: dict) -> None:
    """Print one axis-metric block."""
    typer.echo(f"\n{title}  (n={metrics.get('n_overlap', 0)})")
    for axis in ("energy", "valence"):
        typer.echo(
            f"  {axis:8s} pearson={metrics.get(f'{axis}_pearson', float('nan')):.3f}  "
            f"spearman={metrics.get(f'{axis}_spearman', float('nan')):.3f}  "
            f"ccc={metrics.get(f'{axis}_ccc', float('nan')):.3f}"
        )


@app.command()
def main(
    data_dir: pathlib.Path = typer.Option(
        pathlib.Path.home() / "moodengine-bench" / "deam", "--data-dir", help="DEAM root."
    ),
    mode: str = typer.Option("both", "--mode", help="'zeroshot', 'probe' or 'both'."),
    embedder: str = typer.Option(
        "mert", "--embedder", help="Probe space: 'mert', 'clap' or 'fused'."
    ),
    limit: int = typer.Option(150, "--limit", help="Max tracks (0 = all); bounds embedding cost."),
    n_splits: int = typer.Option(5, "--folds", help="Probe cross-validation folds."),
    cache_dir: pathlib.Path | None = typer.Option(
        None, "--cache-dir", help="Embedding cache (defaults to the config cache dir)."
    ),
    force: bool = typer.Option(False, "--force", help="Recompute embeddings even on a cache hit."),
    out: pathlib.Path | None = typer.Option(None, "--out", help="Write results JSON here."),
) -> None:
    """Embed a DEAM subset and report axis correlation against gold valence/arousal."""
    # laion_clap parses sys.argv at import time; Typer has already bound our options, so
    # blank argv out before any embedder pulls laion_clap in, or its parser SystemExits.
    sys.argv = sys.argv[:1]

    config = default_config()
    if cache_dir is not None:
        config = replace(config, cache_dir=cache_dir)
    config.ensure_dirs()

    gold = _load_gold(data_dir)
    sel = _select(data_dir, gold, limit)
    if not sel:
        typer.echo(f"No annotated audio found under {data_dir}. Run fetch_deam.py first.")
        raise typer.Exit(code=1)
    typer.echo(f"Benchmarking {len(sel)} DEAM tracks (mode={mode}, embedder={embedder}).")

    energy_gold = np.array([r[2] for r in sel], dtype=np.float64)
    valence_gold = np.array([r[3] for r in sel], dtype=np.float64)
    results: dict = {"n": len(sel), "mode": mode, "embedder": embedder, "limit": limit}

    if mode in ("zeroshot", "both"):
        clap = get_embedder("clap", config)
        xc = _embed([r[1] for r in sel], "clap", config, force)
        attrs = attribute_scores(xc, clap)
        zs = _score(attrs["energy"].to_numpy(), attrs["valence"].to_numpy(), sel)
        _print("zero-shot (CLAP attribute_scores)", zs)
        results["zeroshot"] = zs

    if mode in ("probe", "both"):
        X = _embed([r[1] for r in sel], embedder, config, force)
        pe = _probe_oof(X, energy_gold, config.seed, n_splits)
        pv = _probe_oof(X, valence_gold, config.seed, n_splits)
        pr = _score(pe, pv, sel)
        _print(f"linear probe ({embedder}, {n_splits}-fold out-of-fold)", pr)
        results["probe"] = pr

    if out is not None:
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"\nWrote {out}")


if __name__ == "__main__":
    app()
