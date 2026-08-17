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
meaningful. Zero-shot is reported TWICE — raw, and affine-calibrated out-of-fold — because a raw
zero-shot score is a softmax whose spread is set by the labelling temperature, a free constant
CCC would otherwise reward. Track the calibrated block across engine changes, and read each CCC
next to its ``rho x c_b`` split.

Results (Pearson / Spearman / CCC per axis) are printed and written to JSON so
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
from moodengine.evaluation import (
    ccc_components,
    concordance_correlation_coefficient,
    evaluate_against_gold,
)
from moodengine.labeling import attribute_scores, l2_normalize
from moodengine.pipeline import get_embedder, track_embedding

app = typer.Typer(add_completion=False, help=__doc__)

# DEAM static ratings are on a 1-9 scale; the engine's axes live in [0, 1].
_SCALE_LO, _SCALE_HI = 1.0, 9.0


def _load_gold(data_dir: pathlib.Path) -> dict[int, tuple[float, float, float]]:
    """Map ``song_id -> (energy01, valence01, disagreement)`` from DEAM's averaged per-song CSVs.

    Reads every ``static_annotations_averaged_songs_*.csv`` under ``data_dir`` (column
    names carry leading spaces, hence ``skipinitialspace``), maps DEAM arousal onto the
    engine's energy axis and valence onto valence, and affine-scales 1-9 -> [0, 1].

    ``disagreement`` is ``max(arousal_std, valence_std)`` on the same [0, 1] scale — the spread
    between annotators, which sits in the same CSV row and was being discarded. A song the
    annotators themselves disagreed about cannot discriminate between two models, so a correlation
    computed over the whole set is partly measuring label noise. ``--max-disagreement`` filters on
    it. ``nan`` when the columns are absent, which filters to "keep everything" rather than
    silently dropping every song.
    """
    gold: dict[int, tuple[float, float, float]] = {}
    span = _SCALE_HI - _SCALE_LO
    for csv_path in sorted(data_dir.rglob("static_annotations_averaged_songs_*.csv")):
        frame = pd.read_csv(csv_path, skipinitialspace=True)
        has_std = {"arousal_std", "valence_std"} <= set(frame.columns)
        for _, row in frame.iterrows():
            energy01 = (float(row["arousal_mean"]) - _SCALE_LO) / span
            valence01 = (float(row["valence_mean"]) - _SCALE_LO) / span
            spread = (
                max(float(row["arousal_std"]), float(row["valence_std"])) / span
                if has_std
                else float("nan")
            )
            gold[int(row["song_id"])] = (energy01, valence01, spread)
    return gold


def _select(
    data_dir: pathlib.Path,
    gold: dict[int, tuple[float, float, float]],
    limit: int,
    seed: int = 0,
) -> list[tuple[str, pathlib.Path, float, float, float]]:
    """Return ``(filename, path, energy01, valence01, disagreement)`` for songs with audio + gold.

    ``limit`` takes a SEEDED PERMUTATION, not a prefix. DEAM's song ids run by annotation
    campaign, so ``sorted(gold)[:150]`` evaluated one contiguous block of provenance — a
    determinism that bought reproducibility at the cost of representativeness. Permuting under a
    fixed seed keeps both: the same 150 songs every run, drawn from the whole corpus.
    """
    audio_dirs = {p.parent for p in data_dir.rglob("*.mp3")}
    rows: list[tuple[str, pathlib.Path, float, float, float]] = []
    for song_id in sorted(gold):
        for d in audio_dirs:
            path = d / f"{song_id}.mp3"
            if path.is_file():
                energy01, valence01, disagreement = gold[song_id]
                rows.append((path.name, path, energy01, valence01, disagreement))
                break
    if not limit or limit <= 0 or limit >= len(rows):
        return rows
    order = np.random.default_rng(seed).permutation(len(rows))[:limit]
    return [rows[i] for i in sorted(order)]


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


def _affine_oof(pred: np.ndarray, gold: np.ndarray, seed: int, n_splits: int) -> np.ndarray:
    """Out-of-fold affine map of ``pred`` onto the gold scale — one slope/intercept per fold.

    Needed because a raw zero-shot axis is a softmax probability, whose SPREAD is set by
    ``labeling.DEFAULT_TEMPERATURE`` rather than by anything about the music. CCC penalises a scale
    mismatch, so it reads that free constant: with the ranking frozen (Spearman identical at every
    temperature) the reported CCC moved 0.367 → 0.503 → 0.578 → 0.378 across temperatures 0.02 /
    0.05 / 0.1 / 0.3, peaking exactly where the predicted spread happened to match the gold spread.
    A real prompt improvement could therefore register as a CCC regression, and a pure temperature
    tweak be banked as a gain.

    Fitting the two coefficients out-of-fold removes that degree of freedom without leaking: the
    map applied to a held-out row was fit only on other rows, so the calibrated CCC measures what
    the axis actually knows. Only two coefficients are fit, so each fold's map is order-preserving
    WITHIN that fold; across folds the slopes differ slightly, so Pearson and Spearman move a
    little between the raw and calibrated blocks (around a percent) rather than being identical. A
    LARGE gap means some fold fit a near-zero or negative slope — read its CCC with suspicion.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    out = np.full(gold.shape[0], np.nan, dtype=np.float64)
    kf = KFold(n_splits=min(n_splits, gold.shape[0]), shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(pred.reshape(-1, 1)):
        model = LinearRegression().fit(pred[train_idx].reshape(-1, 1), gold[train_idx])
        out[test_idx] = model.predict(pred[test_idx].reshape(-1, 1))
    return out


_BOOTSTRAP_RESAMPLES: int = 2000


def _bootstrap_ci(
    pred: np.ndarray, gold: np.ndarray, statistic, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for a paired statistic — the number that was missing entirely.

    Without an interval, two runs differing by 0.03 are indistinguishable from two runs differing
    by 0.30: both read as "it moved". Resampling the (pred, gold) PAIRS with replacement and
    re-computing the statistic gives the sampling spread directly, with no distributional
    assumption — which matters because Pearson and CCC are both bounded and skewed near their
    limits. 2000 resamples of a few hundred float pairs is milliseconds.
    """
    p_arr = np.asarray(pred, dtype=np.float64).ravel()
    g_arr = np.asarray(gold, dtype=np.float64).ravel()
    mask = np.isfinite(p_arr) & np.isfinite(g_arr)
    p_arr, g_arr = p_arr[mask], g_arr[mask]
    n = p_arr.shape[0]
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(_BOOTSTRAP_RESAMPLES, n))
    vals = np.array([statistic(p_arr[d], g_arr[d]) for d in draws], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def _paired_delta_ci(
    pred_a: np.ndarray, pred_b: np.ndarray, gold: np.ndarray, statistic, seed: int = 0
) -> tuple[float, float, float]:
    """CI on the DIFFERENCE between two predictors scored on the SAME songs.

    The comparison a benchmark exists for, and the one marginal intervals answer wrongly: both
    arms see identical audio and identical labels, so their errors are correlated and the paired
    interval is far tighter than either marginal one. Resample the song indices ONCE per draw and
    score both arms on that same resample, preserving the pairing. Returns
    ``(delta, lo, hi)``; an interval excluding 0 is a real difference.
    """
    a = np.asarray(pred_a, dtype=np.float64).ravel()
    b = np.asarray(pred_b, dtype=np.float64).ravel()
    g = np.asarray(gold, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(g)
    a, b, g = a[mask], b[mask], g[mask]
    n = a.shape[0]
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(_BOOTSTRAP_RESAMPLES, n))
    deltas = np.array(
        [statistic(a[d], g[d]) - statistic(b[d], g[d]) for d in draws], dtype=np.float64
    )
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size < 2:
        return float("nan"), float("nan"), float("nan")
    return (
        float(statistic(a, g) - statistic(b, g)),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def _pearson_stat(p: np.ndarray, g: np.ndarray) -> float:
    if p.size < 2 or float(p.std()) == 0.0 or float(g.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(p, g)[0, 1])


def _ccc_stat(p: np.ndarray, g: np.ndarray) -> float:
    return float(concordance_correlation_coefficient(p, g)[0])


def _score(pred_energy: np.ndarray, pred_valence: np.ndarray, sel) -> dict:
    """Pearson/Spearman/CCC of predicted vs gold energy & valence, plus Lin's CCC decomposition.

    ``<axis>_ccc_rho`` and ``<axis>_ccc_cb`` split each CCC into how well the axis ORDERS the gold
    values and how far it sits from the ``y = x`` line (``CCC = rho * c_b``). Reporting the pair
    makes a scale artifact legible: a CCC that moves while ``rho`` holds still is a spread change,
    not a quality change.
    """
    df = pd.DataFrame(
        {"filename": [r[0] for r in sel], "energy": pred_energy, "valence": pred_valence}
    )
    gold = {r[0]: {"energy": r[2], "valence": r[3]} for r in sel}
    metrics = evaluate_against_gold(df, gold)
    # Kept so a later run can pair against this one; a paired bootstrap needs the per-song values,
    # not the summary statistics.
    metrics["energy_pred"] = [float(v) for v in np.asarray(pred_energy).ravel()]
    metrics["valence_pred"] = [float(v) for v in np.asarray(pred_valence).ravel()]

    for axis, pred in (("energy", pred_energy), ("valence", pred_valence)):
        truth = np.array([g[axis] for g in (gold[r[0]] for r in sel)], dtype=np.float64)
        arr = np.asarray(pred, dtype=np.float64)
        rho, c_b, _ = ccc_components(arr, truth)
        metrics[f"{axis}_ccc_rho"] = rho
        metrics[f"{axis}_ccc_cb"] = c_b
        lo, hi = _bootstrap_ci(arr, truth, _pearson_stat)
        metrics[f"{axis}_pearson_ci"] = [lo, hi]
        lo, hi = _bootstrap_ci(arr, truth, _ccc_stat)
        metrics[f"{axis}_ccc_ci"] = [lo, hi]
    return metrics


def _print(title: str, metrics: dict) -> None:
    """Print one axis-metric block."""
    typer.echo(f"\n{title}  (n={metrics.get('n_overlap', 0)})")
    for axis in ("energy", "valence"):
        typer.echo(
            f"  {axis:8s} pearson={metrics.get(f'{axis}_pearson', float('nan')):.3f}  "
            f"spearman={metrics.get(f'{axis}_spearman', float('nan')):.3f}  "
            f"ccc={metrics.get(f'{axis}_ccc', float('nan')):.3f} "
            f"(rho={metrics.get(f'{axis}_ccc_rho', float('nan')):.3f} x "
            f"c_b={metrics.get(f'{axis}_ccc_cb', float('nan')):.3f})"
        )
        p_ci = metrics.get(f"{axis}_pearson_ci", [float("nan")] * 2)
        c_ci = metrics.get(f"{axis}_ccc_ci", [float("nan")] * 2)
        typer.echo(
            f"           95% CI  pearson [{p_ci[0]:.3f}, {p_ci[1]:.3f}]  "
            f"ccc [{c_ci[0]:.3f}, {c_ci[1]:.3f}]"
        )


def _report_paired(baseline_path: pathlib.Path, current: dict, energy_gold, valence_gold) -> None:
    """Compare this run against an earlier ``--out`` JSON with a PAIRED bootstrap.

    The question a benchmark exists to answer — "did that change help?" — and the one marginal
    intervals answer wrongly. Two runs over the same songs share their audio and their labels, so
    their errors are correlated and the paired interval on the DIFFERENCE is far tighter than the
    overlap of two separate intervals suggests. Two marginal CIs can overlap heavily while the
    paired difference excludes zero.

    Refuses to compare runs that did not score the same song set, since the pairing is the whole
    point: a mismatched comparison is a marginal one wearing a paired label.
    """
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        typer.echo(f"\nCannot read --compare baseline {baseline_path}: {exc}")
        return

    if baseline.get("songs") != current.get("songs"):
        typer.echo(
            f"\nRefusing to pair: {baseline_path.name} scored a different song set "
            f"({len(baseline.get('songs') or [])} vs {len(current.get('songs') or [])}). "
            "Re-run both arms with the same --limit/--seed/--max-disagreement."
        )
        return

    typer.echo(f"\nPaired vs {baseline_path.name}  (95% CI on the difference; excludes 0 = real)")
    for block in ("zeroshot", "zeroshot_calibrated", "probe"):
        if block not in baseline or block not in current:
            continue
        for axis, gold in (("energy", energy_gold), ("valence", valence_gold)):
            key = f"{axis}_pred"
            if key not in baseline[block] or key not in current[block]:
                continue
            delta, lo, hi = _paired_delta_ci(
                np.asarray(current[block][key], dtype=np.float64),
                np.asarray(baseline[block][key], dtype=np.float64),
                gold,
                _pearson_stat,
            )
            verdict = "significatif" if (lo > 0 or hi < 0) else "dans le bruit"
            typer.echo(
                f"  {block:<20} {axis:8s} pearson delta={delta:+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]  {verdict}"
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
    zeroshot_embedder: str = typer.Option(
        "clap",
        "--zeroshot-embedder",
        help="Text-capable backbone for the zero-shot block: 'clap' or 'mulan'.",
    ),
    limit: int = typer.Option(150, "--limit", help="Max tracks (0 = all); bounds embedding cost."),
    seed: int = typer.Option(0, "--seed", help="Seeds the subset permutation; fixes which songs."),
    max_disagreement: float = typer.Option(
        0.0,
        "--max-disagreement",
        help="Keep only songs whose annotators agreed within this (0-1 scale; 0 = keep all).",
    ),
    compare: pathlib.Path | None = typer.Option(
        None, "--compare", help="An earlier --out JSON; reports a PAIRED CI on the difference."
    ),
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
    sel = _select(data_dir, gold, limit, seed=seed)
    if max_disagreement > 0:
        kept = [r for r in sel if not np.isfinite(r[4]) or r[4] <= max_disagreement]
        typer.echo(
            f"Annotator-agreement filter <= {max_disagreement}: kept {len(kept)}/{len(sel)} songs."
        )
        sel = kept
    if not sel:
        typer.echo(f"No annotated audio found under {data_dir}. Run fetch_deam.py first.")
        raise typer.Exit(code=1)
    typer.echo(f"Benchmarking {len(sel)} DEAM tracks (mode={mode}, embedder={embedder}).")

    energy_gold = np.array([r[2] for r in sel], dtype=np.float64)
    valence_gold = np.array([r[3] for r in sel], dtype=np.float64)
    results: dict = {
        "n": len(sel),
        "mode": mode,
        "embedder": embedder,
        "limit": limit,
        "seed": seed,
        "max_disagreement": max_disagreement,
        # Songs, not just their count: a paired comparison is only valid across runs that scored
        # the SAME set, and this is what lets --compare check that instead of assuming it.
        "songs": sorted(r[0] for r in sel),
    }

    if mode in ("zeroshot", "both"):
        # Text-capable backbone, selectable: the zero-shot block used to be hard-wired to CLAP,
        # which made the one comparison this benchmark exists for — CLAP against MuQ-MuLan on
        # ACCURACY rather than on separability proxies — impossible to run.
        zs_name = zeroshot_embedder
        zs = get_embedder(zs_name, config)
        xc = _embed([r[1] for r in sel], zs_name, config, force)
        attrs = attribute_scores(xc, zs)
        e_raw, v_raw = attrs["energy"].to_numpy(), attrs["valence"].to_numpy()
        zs_block = _score(e_raw, v_raw, sel)
        _print(f"zero-shot ({zs_name} attribute_scores, raw softmax scale)", zs_block)
        results["zeroshot"] = zs_block
        results["zeroshot_embedder"] = zs_name

        # The raw CCC above is partly a readout of the softmax temperature (see _affine_oof).
        # This second block puts both axes on the gold scale out-of-fold, so its CCC reflects the
        # axis rather than that constant. Compare THIS one across engine changes. Pearson and
        # Spearman shift only slightly between the blocks (each fold's map preserves order, but the
        # folds' slopes differ); a large shift means a fold fit a degenerate slope.
        zs_cal = _score(
            _affine_oof(e_raw, energy_gold, config.seed, n_splits),
            _affine_oof(v_raw, valence_gold, config.seed, n_splits),
            sel,
        )
        _print(f"zero-shot, affine-calibrated out-of-fold ({n_splits}-fold)", zs_cal)
        results["zeroshot_calibrated"] = zs_cal

    if mode in ("probe", "both"):
        X = _embed([r[1] for r in sel], embedder, config, force)
        pe = _probe_oof(X, energy_gold, config.seed, n_splits)
        pv = _probe_oof(X, valence_gold, config.seed, n_splits)
        pr = _score(pe, pv, sel)
        _print(f"linear probe ({embedder}, {n_splits}-fold out-of-fold)", pr)
        results["probe"] = pr

    if compare is not None:
        _report_paired(compare, results, energy_gold, valence_gold)

    if out is not None:
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"\nWrote {out}")


if __name__ == "__main__":
    app()
