"""CLI: search the CLAP space by free text or by an example track.

Thin :mod:`typer` wrapper over :mod:`moodengine.search`. Both modes operate on the
(cached) CLAP track embeddings pulled via :func:`moodengine.pipeline.extract_embeddings`
(CLAP vectors are L2-normalized at pooling time, so dot products are cosines):

  * ``--query "dreamy nocturnal"`` embeds the text through the same CLAP text
    encoder used for labeling and prints the top-``--top-k``
    :func:`moodengine.search.playlist_from_text` ranking.
  * ``--similar-to <filename>`` finds the track whose filename matches and prints
    its nearest neighbours via :func:`moodengine.search.find_similar` (the query track is
    excluded from its own results).

Pass at least one of ``--query`` / ``--similar-to``. Degenerate inputs (no audio
found, unknown filename) are reported without raising.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import typer

from moodengine.config import default_config
from moodengine.pipeline import extract_embeddings, get_embedder
from moodengine.search import find_similar, search_by_text

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    query: str | None = typer.Option(
        None, "--query", "-q", help="Free-text mood query, e.g. 'dreamy nocturnal'."
    ),
    similar_to: str | None = typer.Option(
        None, "--similar-to", help="Filename of an embedded track to find neighbours of."
    ),
    input_dir: pathlib.Path | None = typer.Option(
        None, "--input-dir", help="Directory of audio files (defaults to config.raw_dir)."
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return."),
    force: bool = typer.Option(
        False, "--force", help="Recompute CLAP embeddings even on a cache hit."
    ),
) -> None:
    """Rank tracks in the CLAP space by a text query or by similarity to a track."""
    if query is None and similar_to is None:
        raise typer.BadParameter("pass at least one of --query / --similar-to.")

    config = default_config()
    if input_dir is not None:
        config = replace(config, raw_dir=input_dir)

    files, X = extract_embeddings(config, "clap", force=force)
    filenames = [p.name for p in files]
    typer.echo(f"CLAP space: {len(filenames)} track(s); matrix shape {X.shape}.")

    if len(filenames) == 0:
        typer.echo("No audio files found; nothing to search.")
        return

    if query is not None:
        clap_embedder = get_embedder("clap", config)
        results = search_by_text(query, X, clap_embedder, filenames, top_k=top_k)
        typer.echo(f"\nText query: {query!r}")
        if not results:
            typer.echo("  (no matches)")
        for rank, (name, score) in enumerate(results, start=1):
            typer.echo(f"  {rank:2d}. {name}  ({score:.3f})")

    if similar_to is not None:
        try:
            query_idx = filenames.index(similar_to)
        except ValueError:
            typer.echo(f"\nNo embedded track named {similar_to!r}.")
            typer.echo(
                "Known filenames: "
                + ", ".join(filenames[:10])
                + (" ..." if len(filenames) > 10 else "")
            )
        else:
            results = find_similar(query_idx, X, filenames, top_k=top_k)
            typer.echo(f"\nMost similar to: {similar_to}")
            if not results:
                typer.echo("  (no other tracks)")
            for rank, (name, score) in enumerate(results, start=1):
                typer.echo(f"  {rank:2d}. {name}  ({score:.3f})")


if __name__ == "__main__":
    app()
