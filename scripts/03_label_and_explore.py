"""CLI: full pipeline -> zero-shot mood labels, scatter HTML, and playlists.

Thin :mod:`typer` wrapper over :func:`moodengine.pipeline.run_pipeline` (with labels).
It prints each cluster's dominant mood, leaves ``clusters.html`` and
``assignments.parquet`` where the pipeline wrote them, and additionally exports
one playlist text file per cluster via :func:`moodengine.viz.export_playlists`.
Degenerate inputs (no audio found) are handled gracefully.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import typer

from moodengine.config import default_config
from moodengine.pipeline import run_pipeline
from moodengine.viz import export_playlists

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    input_dir: pathlib.Path | None = typer.Option(
        None, "--input-dir", help="Directory of audio files (defaults to config.raw_dir)."
    ),
    embedder: str = typer.Option("mert", "--embedder", help="Embedder to use: 'mert' or 'clap'."),
    method: str = typer.Option(
        "hdbscan", "--method", help="Clustering method: 'hdbscan' or 'kmeans'."
    ),
    force: bool = typer.Option(False, "--force", help="Recompute embeddings even on a cache hit."),
) -> None:
    """Label clusters with moods, write clusters.html, and export playlists."""
    config = default_config()
    if input_dir is not None:
        config = replace(config, raw_dir=input_dir)

    df = run_pipeline(
        config,
        embedder_name=embedder,
        method=method,
        with_labels=True,
        force=force,
    )

    out_dir = pathlib.Path(config.output_dir)
    typer.echo(f"Embedder: {embedder}  Method: {method}")
    typer.echo(f"Labeled {len(df)} track(s).")

    if len(df) == 0:
        typer.echo("No audio files found; nothing to label.")
    else:
        typer.echo("Per-cluster mood profile:")
        # One row per cluster, ascending; noise (-1) included when present.
        seen = df.drop_duplicates(subset="cluster").sort_values("cluster")
        for _, row in seen.iterrows():
            cluster = int(row["cluster"])
            name = "noise" if cluster == -1 else f"cluster {cluster}"
            mood = row["cluster_mood"] if "cluster_mood" in df.columns else ""
            profile = row["cluster_profile"] if "cluster_profile" in df.columns else ""
            count = int((df["cluster"] == cluster).sum())
            typer.echo(f"  {name}: {mood or '(none)'}  ({count} track(s))")
            if profile:
                typer.echo(f"    profile: {profile}")

    playlists = export_playlists(df, out_dir)
    typer.echo(f"Wrote {out_dir / 'clusters.html'}")
    typer.echo(f"Wrote {out_dir / 'cluster_report.md'}")
    typer.echo(f"Wrote {out_dir / 'mood_space.html'}")
    typer.echo(f"Wrote {len(playlists)} playlist file(s) to {out_dir}")


if __name__ == "__main__":
    app()
