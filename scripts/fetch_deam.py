"""Fetch the DEAM valence/arousal benchmark (audio + averaged per-song annotations).

DEAM (the MediaEval "Emotion in Music" database, a superset of the emoMusic 1000-songs
set) ships 1802 forty-five-second excerpts with human valence/arousal ratings — the
canonical, openly downloadable ground truth for the pipeline's energy/valence axes and
for embedding-quality probes. This is the yardstick behind ``bench_valence_arousal.py``:
without a labelled set, no change to the engine can be shown to help rather than regress.

Downloads two zips from the University of Geneva CVML mirror (~1.35 GB total), extracts
them under ``--data-dir``, and prints the resolved audio directory + static-annotations
CSV so the benchmark runner can be pointed straight at them. Idempotent: an already
downloaded zip (matching the server's advertised size) or an already extracted tree is
skipped, so re-running only fills what is missing. Pure stdlib networking — no extra
dependency lands in the package for a developer-only utility.

    uv run --extra models python scripts/fetch_deam.py --data-dir ~/moodengine-bench/deam
"""

from __future__ import annotations

import pathlib
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import typer

app = typer.Typer(add_completion=False, help=__doc__)

# CVML mirror (openly downloadable, no request form — verified July 2026). The audio zip
# extracts to MEMD_audio/<song_id>.mp3; the annotations zip to annotations/… with the
# averaged-per-song static CSVs the benchmark reads.
_AUDIO_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip"
_ANNOTATIONS_URL = "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip"


def _remote_size(url: str) -> int | None:
    """Content-Length the server advertises for ``url`` (``None`` if it does not)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:  # noqa: BLE001 - a missing size just disables the skip optimisation
        return None


def _download(url: str, dest: pathlib.Path) -> None:
    """Stream ``url`` to ``dest``, skipping when a fully downloaded file already exists.

    Writes to a ``.part`` sibling and renames on success so an interrupted run never
    leaves a truncated file masquerading as complete. Prints coarse progress because a
    multi-hundred-MB download over a plain request otherwise looks hung.
    """
    remote = _remote_size(url)
    if dest.exists() and remote is not None and dest.stat().st_size == remote:
        typer.echo(f"  {dest.name}: already downloaded ({remote / 1e6:.0f} MB), skipping.")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    typer.echo(f"  {dest.name}: downloading{f' ({remote / 1e6:.0f} MB)' if remote else ''}…")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        done = 0
        next_mark = 100_000_000  # log every ~100 MB
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if done >= next_mark:
                typer.echo(f"    …{done / 1e6:.0f} MB")
                next_mark += 100_000_000
    tmp.replace(dest)
    typer.echo(f"  {dest.name}: done ({done / 1e6:.0f} MB).")


def _extract(zip_path: pathlib.Path, root: pathlib.Path, sentinel: str) -> None:
    """Extract ``zip_path`` under ``root`` unless ``sentinel`` (a path fragment) already exists."""
    if any(root.rglob(sentinel)):
        typer.echo(f"  {zip_path.name}: already extracted (found {sentinel}), skipping.")
        return
    typer.echo(f"  {zip_path.name}: extracting…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)


@app.command()
def main(
    data_dir: pathlib.Path = typer.Option(
        pathlib.Path.home() / "moodengine-bench" / "deam",
        "--data-dir",
        help="Directory to download and extract DEAM into.",
    ),
    keep_zips: bool = typer.Option(
        True, "--keep-zips/--delete-zips", help="Keep the downloaded zips after extraction."
    ),
) -> None:
    """Download + extract DEAM, then print the audio dir and annotations CSV to use."""
    data_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"DEAM into {data_dir}")

    audio_zip = data_dir / "DEAM_audio.zip"
    ann_zip = data_dir / "DEAM_Annotations.zip"
    _download(_ANNOTATIONS_URL, ann_zip)
    _download(_AUDIO_URL, audio_zip)

    _extract(ann_zip, data_dir, "static_annotations_averaged_songs_1_2000.csv")
    _extract(audio_zip, data_dir, "*.mp3")

    if not keep_zips:
        audio_zip.unlink(missing_ok=True)
        ann_zip.unlink(missing_ok=True)

    audio_dirs = {p.parent for p in data_dir.rglob("*.mp3")}
    ann_csvs = sorted(data_dir.rglob("static_annotations_averaged_songs_*.csv"))
    typer.echo("\nReady. Point the benchmark at:")
    for d in sorted(audio_dirs):
        typer.echo(f"  audio:       {d}  ({len(list(d.glob('*.mp3')))} mp3)")
    for c in ann_csvs:
        typer.echo(f"  annotations: {c}")
    typer.echo("\nNext: uv run --extra models python scripts/bench_valence_arousal.py \\")
    typer.echo(f"        --data-dir {data_dir} --mode both --embedder mert --limit 150")


if __name__ == "__main__":
    app()
