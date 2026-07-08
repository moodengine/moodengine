"""CLI: cluster cached embeddings and save a per-track assignments parquet.

Thin :mod:`typer` wrapper that pulls (cached) embeddings via
:func:`moodengine.pipeline.extract_embeddings`, runs :func:`moodengine.cluster.run_clustering`,
prints the clustering metrics, and writes ``assignments.parquet`` (columns
``filename``, ``path``, ``cluster``, ``x``, ``y``) to ``config.output_dir``.
Degenerate inputs (no audio found) are handled gracefully.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import typer

from moodengine.cluster import run_clustering
from moodengine.config import default_config
from moodengine.pipeline import extract_embeddings

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
    """Cluster the embeddings, print metrics, and save an assignments parquet."""
    config = default_config()
    if input_dir is not None:
        config = replace(config, raw_dir=input_dir)
    config.ensure_dirs()

    files, X = extract_embeddings(config, embedder, force=force)
    out_path = pathlib.Path(config.output_dir) / "assignments.parquet"

    if len(files) == 0:
        typer.echo("No audio files found; nothing to cluster.")
        df = pd.DataFrame(
            {
                "filename": pd.Series([], dtype=str),
                "path": pd.Series([], dtype=str),
                "cluster": pd.Series([], dtype=int),
                "x": pd.Series([], dtype=float),
                "y": pd.Series([], dtype=float),
            }
        )
        df.to_parquet(out_path, index=False)
        typer.echo(f"Wrote {out_path}")
        return

    result = run_clustering(X, method, config)
    labels = result["labels"]
    coords2d = result["coords2d"]
    metrics = result["metrics"]

    df = pd.DataFrame(
        {
            "filename": [p.name for p in files],
            "path": [str(p) for p in files],
            "cluster": [int(c) for c in labels],
            "x": [float(v) for v in coords2d[:, 0]],
            "y": [float(v) for v in coords2d[:, 1]],
        }
    )
    df.to_parquet(out_path, index=False)

    typer.echo(f"Method: {result['method']}  (embedder: {embedder})")
    typer.echo(f"Clusters (excl. noise): {metrics['n_clusters']}")
    typer.echo(f"Noise ratio: {metrics['noise_ratio']:.3f}")
    typer.echo(f"Cluster sizes: {metrics['cluster_sizes']}")
    typer.echo(f"Silhouette: {metrics['silhouette']}")
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
