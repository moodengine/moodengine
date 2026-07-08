"""CLI: rigorous, falsifiable evaluation of the mood pipeline.

Thin :mod:`typer` wrapper that runs the (cached) pipeline and reports the
self-consistency / quality metrics from :mod:`moodengine.evaluation`,
:mod:`moodengine.labeling` and :mod:`moodengine.pipeline`:

  * :func:`moodengine.labeling.labeling_quality_metrics` — diversity / dominance and
    confidence of the zero-shot mood assignments.
  * :func:`moodengine.pipeline.compare_spaces` — clustering metrics for the MERT, CLAP
    and fused spaces, each augmented with ``silhouette_original`` (cosine on the
    pre-UMAP matrix) and bootstrap ``stability_ari``.
  * Axis self-consistency AUC (:func:`moodengine.evaluation.axis_ranking_auc`) — does
    similarity to an "energetic" / "positive" pole prompt rank tracks the same
    way as the predicted energy / valence axis? An AUC near 1.0 means the two
    views of the same construct agree.

It also writes a self-contained gold-labeling UI to ``outputs/label_ui.html``
(:func:`moodengine.viz.build_labeling_ui`) so a human can produce a gold set that makes
the rest of the evaluation falsifiable. The pipeline is run with labels (CLAP
provides them regardless of the clustering space). Degenerate inputs (no audio
found) are handled gracefully.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import typer

from moodengine.config import default_config
from moodengine.evaluation import axis_ranking_auc
from moodengine.labeling import DEFAULT_MOOD_PROMPTS, labeling_quality_metrics
from moodengine.pipeline import compare_spaces, extract_embeddings, get_embedder, run_pipeline
from moodengine.search import search_by_text
from moodengine.viz import build_labeling_ui

app = typer.Typer(add_completion=False, help=__doc__)


def _axis_auc(
    query: str, df, X, filenames: list[str], clap_embedder, axis_col: str
) -> float | None:
    """Self-consistency AUC: similarity to ``query`` vs the ``axis_col`` of ``df``.

    Embeds ``query`` through CLAP, scores every track by cosine similarity, then
    aligns those scores to ``df`` rows (by filename) and reports how well they
    rank-track the predicted ``axis_col`` via :func:`evaluation.axis_ranking_auc`.
    Returns ``None`` when the axis column is missing or nothing aligns.
    """
    if axis_col not in df.columns:
        return None
    results = search_by_text(query, X, clap_embedder, filenames, top_k=len(filenames))
    sim_by_name = {name: score for name, score in results}
    scores: list[float] = []
    axis_values: list[float] = []
    for _, row in df.iterrows():
        name = str(row["filename"])
        if name in sim_by_name:
            scores.append(float(sim_by_name[name]))
            axis_values.append(float(row[axis_col]))
    if len(scores) < 2:
        return None
    return axis_ranking_auc(np.asarray(scores), np.asarray(axis_values))


@app.command()
def main(
    input_dir: pathlib.Path | None = typer.Option(
        None, "--input-dir", help="Directory of audio files (defaults to config.raw_dir)."
    ),
    embedder: str = typer.Option(
        "mert", "--embedder", help="Clustering space: 'mert', 'clap' or 'fused'."
    ),
    method: str = typer.Option(
        "kmeans", "--method", help="Clustering method: 'hdbscan' or 'kmeans'."
    ),
    force: bool = typer.Option(False, "--force", help="Recompute embeddings even on a cache hit."),
) -> None:
    """Run the pipeline (cached) and print quality / self-consistency metrics."""
    config = default_config()
    if input_dir is not None:
        config = replace(config, raw_dir=input_dir)
    config.ensure_dirs()
    out_dir = pathlib.Path(config.output_dir)

    df = run_pipeline(
        config,
        embedder_name=embedder,
        method=method,
        with_labels=True,
        force=force,
    )
    typer.echo(f"Evaluated {len(df)} track(s)  (clustering space: {embedder}, method: {method}).")

    if len(df) == 0:
        typer.echo("No audio files found; nothing to evaluate.")
        # Still emit the (empty) labeling UI so the artifact set is stable.
        build_labeling_ui(
            [],
            [],
            list(DEFAULT_MOOD_PROMPTS.keys()),
            out_dir / "label_ui.html",
            audio_dir=config.raw_dir,
        )
        typer.echo(f"Wrote {out_dir / 'label_ui.html'}")
        return

    # --- labeling quality ---------------------------------------------------
    quality = labeling_quality_metrics(df)
    typer.echo("\nLabeling quality:")
    typer.echo(f"  distinct top moods: {quality['n_distinct_top_moods']}")
    typer.echo(f"  max mood share:     {quality['max_mood_share']:.3f}")
    typer.echo(f"  mean top1 - top2:   {quality['mean_top1_minus_top2']:.3f}")
    typer.echo(f"  mean top score:     {quality['mean_top_score']:.3f}")
    typer.echo(f"  histogram:          {quality['top_mood_histogram']}")

    # --- space comparison ---------------------------------------------------
    typer.echo("\nSpace comparison (clusters / noise / silhouette_orig / stability_ari):")
    spaces = compare_spaces(config, method=method, force=force)
    if not spaces:
        typer.echo("  (no spaces available)")
    for name, m in spaces.items():
        sil_o = m.get("silhouette_original")
        sil_o_str = f"{sil_o:.3f}" if sil_o is not None else "n/a"
        typer.echo(
            f"  {name:6s}: clusters={m.get('n_clusters', 0)} "
            f"noise={m.get('noise_ratio', 0.0):.3f} "
            f"sil_orig={sil_o_str} "
            f"stability_ari={m.get('stability_ari', 0.0):.3f}"
        )

    # --- axis self-consistency ----------------------------------------------
    # Score similarity to a pole prompt on the CLAP space and check it rank-tracks
    # the predicted energy / valence axes.
    typer.echo("\nAxis self-consistency (AUC, 1.0 = perfect agreement):")
    files_c, Xc = extract_embeddings(config, "clap", force=force)
    if Xc.shape[0] == 0:
        typer.echo("  (no CLAP embeddings available)")
    else:
        clap_embedder = get_embedder("clap", config)
        clap_filenames = [p.name for p in files_c]
        energy_auc = _axis_auc(
            "an energetic high-energy fast intense track",
            df,
            Xc,
            clap_filenames,
            clap_embedder,
            "energy",
        )
        valence_auc = _axis_auc(
            "a happy bright uplifting positive track",
            df,
            Xc,
            clap_filenames,
            clap_embedder,
            "valence",
        )
        typer.echo(
            f"  energy  ('energetic'): {energy_auc:.3f}"
            if energy_auc is not None
            else "  energy:  n/a"
        )
        typer.echo(
            f"  valence ('positive'):  {valence_auc:.3f}"
            if valence_auc is not None
            else "  valence: n/a"
        )

    # --- gold-labeling UI ---------------------------------------------------
    filenames = df["filename"].astype(str).tolist()
    paths = df["path"].astype(str).tolist() if "path" in df.columns else ["" for _ in filenames]
    build_labeling_ui(
        filenames,
        paths,
        list(DEFAULT_MOOD_PROMPTS.keys()),
        out_dir / "label_ui.html",
        audio_dir=config.raw_dir,
    )
    typer.echo(f"\nWrote gold-labeling UI to {out_dir / 'label_ui.html'}")


if __name__ == "__main__":
    app()
