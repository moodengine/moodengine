"""CLI: extract (and cache) track-level audio embeddings.

Thin :mod:`typer` wrapper over :func:`moodengine.pipeline.extract_embeddings`. The heavy
lifting (decode -> segment -> embed -> pool -> cache) lives in ``src``; this
script only resolves the run :class:`~moodengine.config.Config`, kicks off extraction,
and prints how many tracks were embedded and the resulting matrix shape. The
embeddings themselves are persisted by the on-disk cache, not by this script.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import typer

from moodengine.config import default_config
from moodengine.pipeline import extract_embeddings

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    input_dir: pathlib.Path | None = typer.Option(
        None, "--input-dir", help="Directory of audio files (defaults to config.raw_dir)."
    ),
    embedder: str = typer.Option("mert", "--embedder", help="Embedder to use: 'mert' or 'clap'."),
    force: bool = typer.Option(False, "--force", help="Recompute embeddings even on a cache hit."),
) -> None:
    """Embed every audio file under ``--input-dir`` and report count + shape."""
    config = default_config()
    if input_dir is not None:
        config = replace(config, raw_dir=input_dir)

    files, X = extract_embeddings(config, embedder, force=force)
    typer.echo(f"Embedder: {embedder}")
    typer.echo(f"Source dir: {config.raw_dir}")
    typer.echo(f"Embedded {len(files)} track(s); matrix shape {X.shape}.")


if __name__ == "__main__":
    app()
