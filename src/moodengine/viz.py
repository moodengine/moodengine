"""Visualization + playlist export for the music-mood POC.

Pure plotting / file-writing helpers built on plotly + pandas. This module is
deliberately torch-free (and free of any model deps) so it imports cleanly with
just the lightweight stack. It turns a 2-D cluster embedding into an interactive
scatter and writes one playlist text file per cluster.
"""

from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PathLike = Union[str, Path]

# Color reserved for HDBSCAN noise (label -1).
_NOISE_COLOR = "#9e9e9e"
# Qualitative palette cycled across the (non-noise) clusters.
_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _cluster_color(label: int, order: dict[int, int]) -> str:
    """Return a hex color for ``label`` (-1 -> gray noise, else from palette)."""
    if label == -1:
        return _NOISE_COLOR
    return _PALETTE[order[label] % len(_PALETTE)]


def _as_coords2d(coords2d: np.ndarray) -> np.ndarray:
    """``coords2d`` as a float ``(n, 2)`` array, folding a flat or single-column input into pairs.

    Degenerate SIZES degrade rather than raise (the module contract), so an empty input becomes
    ``(0, 2)`` instead of erroring on the reshape.
    """
    coords = np.asarray(coords2d, dtype=float)
    if coords.ndim != 2 or (coords.size and coords.shape[1] < 2):
        return coords.reshape(-1, 2) if coords.size else coords.reshape(0, 2)
    return coords


def _cluster_order(labels_arr: np.ndarray) -> tuple[list[int], dict[int, int]]:
    """The ascending cluster ids and their palette positions, noise (``-1``) excluded from both.

    Ascending order is what makes colors and legend deterministic; it also puts ``-1`` first, and
    plotly draws in add order, so noise ends up UNDER the colored points.
    """
    unique = sorted({int(v) for v in labels_arr.tolist()})
    non_noise = [c for c in unique if c != -1]
    return unique, {c: i for i, c in enumerate(non_noise)}


def _trace_name(cluster: int) -> str:
    """Legend entry for one cluster; ``-1`` reads as ``noise``, never as ``cluster -1``."""
    return "noise" if cluster == -1 else f"cluster {cluster}"


def _at(seq: Optional[Sequence[object]], i: int) -> Optional[object]:
    """``seq[i]`` when the sequence exists and is long enough, else ``None``.

    Every hover field is optional AND may be shorter than the coordinate array, so the
    ragged-input rule lives here once instead of being re-spelled at each lookup.
    """
    if seq is None or i >= len(seq):
        return None
    return seq[i]


def _point_hover(
    i: int,
    files: Sequence[str],
    mood_list: Optional[Sequence[str]],
    hover_list: Optional[Sequence[str]],
) -> str:
    """Hover string for point ``i``: an explicit ``hover_text`` entry wins, else filename + mood."""
    override = _at(hover_list, i)
    if override is not None:
        return str(override)

    name = _at(files, i)
    mood = _at(mood_list, i)
    if mood is not None:
        return f"{name if name is not None else ''}<br>mood: {mood}"
    return str(name if name is not None else "")


def _attr_hover(
    i: int, files: Sequence[str], mood_list: Optional[Sequence[str]], v: np.ndarray, e: np.ndarray
) -> str:
    """Hover string for the valence x energy scatter: filename, optional mood, then the coords."""
    name = _at(files, i)
    parts = [str(name if name is not None else "")]

    mood = _at(mood_list, i)
    if mood is not None:
        parts.append(f"mood: {mood}")

    parts.append(f"valence={v[i]:.2f} energy={e[i]:.2f}")
    return "<br>".join(parts)


def _medoid_trace(
    coords: np.ndarray, medoids: set[int], files: Sequence[str], n: int
) -> Optional[go.Scatter]:
    """Black-outlined diamond overlay for the cluster representatives, or ``None`` when there is
    nothing valid to draw — a non-integer or out-of-range member is dropped, never raised on."""
    med_idx = np.array(
        sorted(i for i in medoids if isinstance(i, (int, np.integer)) and 0 <= i < n),
        dtype=int,
    )
    if med_idx.size == 0:
        return None

    return go.Scatter(
        x=coords[med_idx, 0],
        y=coords[med_idx, 1],
        mode="markers",
        name="medoid",
        marker={
            "symbol": "diamond",
            "color": "rgba(0,0,0,0)",
            "size": 16,
            "line": {"width": 2, "color": "#000000"},
        },
        text=[f"medoid<br>{_at(files, i) if _at(files, i) is not None else ''}" for i in med_idx],
        hoverinfo="text",
    )


def _write_html(fig: go.Figure, out_html: Optional[PathLike]) -> None:
    """Write ``fig`` as a standalone self-contained page, or do nothing when no path is given.

    The ``out_html=None`` default is what keeps this module pure by default: file output is opt-in
    (see the module docstring), so plotting never touches the filesystem unless asked.
    """
    if out_html is None:
        return

    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)


def plot_clusters(
    coords2d: np.ndarray,
    labels: Sequence[int],
    filenames: Sequence[str],
    moods: Optional[Sequence[str]] = None,
    title: str = "Mood clusters",
    out_html: Optional[PathLike] = None,
    hover_text: Optional[Sequence[str]] = None,
    medoids: Optional[set[int]] = None,
) -> go.Figure:
    """2-D scatter colored by cluster label; hover shows filename + mood.

    ``coords2d`` is an ``(n, 2)`` array; ``labels`` / ``filenames`` / ``moods``
    (if given) are length-``n`` sequences. Label ``-1`` is rendered as gray
    ``noise``. ``hover_text`` (length ``n``), when provided, overrides the default
    ``filename + mood`` hover string per point (use it to surface top-k moods,
    energy/valence, etc.); ``<br>`` is honoured as a line break. ``medoids``, when
    given, is a set of point indices (cluster representatives) that are overdrawn
    with a black-outlined diamond marker so they stand out. If ``out_html`` is
    provided, a standalone self-contained HTML file (plotly.js inlined) is written.
    Returns the :class:`plotly.graph_objects.Figure`. Degenerate inputs (empty,
    all-noise, missing moods/hover) are handled gracefully.
    """
    coords = _as_coords2d(coords2d)
    n = coords.shape[0]

    labels_arr = np.asarray(list(labels), dtype=int) if len(labels) else np.empty(0, dtype=int)
    files = list(filenames)
    mood_list = list(moods) if moods is not None else None
    hover_list = list(hover_text) if hover_text is not None else None
    unique, order = _cluster_order(labels_arr)

    fig = go.Figure()
    for cluster in unique:
        # Only points that have coordinates: a labels/coords length mismatch degrades to a
        # shorter trace instead of raising IndexError.
        idx = np.flatnonzero(labels_arr == cluster)
        idx = idx[idx < n]
        if idx.size == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0],
                y=coords[idx, 1],
                mode="markers",
                name=_trace_name(cluster),
                marker={
                    "color": _cluster_color(cluster, order),
                    "size": 9,
                    "line": {"width": 0.5, "color": "#ffffff"},
                },
                text=[_point_hover(int(i), files, mood_list, hover_list) for i in idx],
                hoverinfo="text",
            )
        )

    if medoids:
        overlay = _medoid_trace(coords, medoids, files, n)
        if overlay is not None:
            fig.add_trace(overlay)  # added last, so the representatives sit on top

    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title="cluster",
        template="plotly_white",
    )
    _write_html(fig, out_html)

    return fig


def plot_attributes(
    energy: Sequence[float],
    valence: Sequence[float],
    labels: Sequence[int],
    filenames: Sequence[str],
    moods: Optional[Sequence[str]] = None,
    title: str = "Mood space (valence × energy)",
    out_html: Optional[PathLike] = None,
) -> go.Figure:
    """Scatter of tracks in the interpretable mood space: valence (x) × energy (y).

    All sequences are length-``n``. Points are colored by cluster label (``-1`` =
    gray ``noise``); hover shows filename, mood (if given) and the (valence,
    energy) coordinates. Axes are pinned to [0, 1] with mid-lines at 0.5 so the
    four quadrants (calm/positive, energetic/positive, ...) read clearly. Writes a
    self-contained HTML when ``out_html`` is given. Returns the Figure.
    """
    e = np.asarray(list(energy), dtype=float) if len(energy) else np.empty(0)
    v = np.asarray(list(valence), dtype=float) if len(valence) else np.empty(0)
    n = min(e.shape[0], v.shape[0])
    labels_arr = np.asarray(list(labels), dtype=int) if len(labels) else np.empty(0, dtype=int)
    files = list(filenames)
    mood_list = list(moods) if moods is not None else None

    unique, order = _cluster_order(labels_arr)

    fig = go.Figure()
    for cluster in unique:
        idx = np.flatnonzero(labels_arr == cluster)
        idx = idx[idx < n]
        if idx.size == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=v[idx],
                y=e[idx],
                mode="markers",
                name=_trace_name(cluster),
                marker={
                    "color": _cluster_color(cluster, order),
                    "size": 10,
                    "line": {"width": 0.5, "color": "#ffffff"},
                },
                text=[_attr_hover(int(i), files, mood_list, v, e) for i in idx],
                hoverinfo="text",
            )
        )

    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color="#cccccc")
    fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#cccccc")
    fig.update_layout(
        title=title,
        xaxis={"title": "valence  (dark/negative → bright/positive)", "range": [0, 1]},
        yaxis={"title": "energy  (calm/low → intense/high)", "range": [0, 1]},
        legend_title="cluster",
        template="plotly_white",
    )
    _write_html(fig, out_html)

    return fig


def _prepare_playlist_dir(out_dir: PathLike, pattern: str) -> Path:
    """Create ``out_dir`` and clear the playlists a previous run left there.

    Stale files are removed rather than overwritten, because a re-clustering can produce FEWER
    clusters than last time — the leftovers would otherwise read as current.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for old in out.glob(pattern):
        old.unlink()

    return out


def _has_columns(df: Optional[pd.DataFrame], *columns: str) -> bool:
    """``df`` is a non-empty frame carrying every one of ``columns``.

    The exporters promise an empty result rather than an exception on a frame they cannot group,
    so this is the single place that decides "nothing to write".
    """
    return df is not None and len(df) > 0 and all(c in df.columns for c in columns)


def _cluster_mood(sub: pd.DataFrame, has_mood: bool) -> str:
    """The cluster's mood label, or ``""`` when absent — a NaN cell reads as absent, not ``'nan'``."""
    if not has_mood or not len(sub):
        return ""

    value = sub["cluster_mood"].iloc[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _m3u_suffix(mood: str, cluster: int) -> str:
    """Filename suffix for one cluster's playlist, in precedence order.

    A mood wins when there is one, so a labelled noise cluster reads by its mood rather than as
    ``_noise``; ``_noise`` is the fallback for an unlabelled ``-1``; everything else gets nothing.
    """
    if mood:
        return f"_{_slugify(mood)}"
    if cluster == -1:
        return "_noise"
    return ""


def _track_paths(sub: pd.DataFrame) -> list[str]:
    """The cluster's track paths as strings, dropping the missing ones (``None`` / NaN)."""
    return [
        str(p)
        for p in sub["path"].tolist()
        if p is not None and not (isinstance(p, float) and np.isnan(p))
    ]


def export_playlists(df: pd.DataFrame, out_dir: PathLike) -> list[Path]:
    """Write one playlist text file per cluster and return the written paths.

    ``df`` must have at least ``'cluster'`` and ``'filename'`` columns. Files are
    named ``cluster_00.txt`` (zero-padded, ascending), with noise written to
    ``cluster_-1_noise.txt``. Each file lists the cluster's filenames, one per
    line. Returns the list of written :class:`pathlib.Path` (ascending by cluster).
    """
    out = _prepare_playlist_dir(out_dir, "cluster_*.txt")
    if not _has_columns(df, "cluster", "filename"):
        return []

    written: list[Path] = []
    clusters = sorted({int(c) for c in df["cluster"].tolist()})
    for cluster in clusters:
        names = df.loc[df["cluster"] == cluster, "filename"].astype(str).tolist()
        fname = "cluster_-1_noise.txt" if cluster == -1 else f"cluster_{cluster:02d}.txt"
        path = out / fname
        path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        written.append(path)

    return written


# --------------------------------------------------------------------------- #
# TRACK 3 — restitution / UX
# --------------------------------------------------------------------------- #

# Columns shown (in order) in the dashboard table, when present in the df.
_TABLE_COLUMNS: tuple[str, ...] = (
    "filename",
    "cluster",
    "cluster_mood",
    "top_mood",
    "top_score",
    "mood_top3",
    "energy",
    "valence",
)


def _cell_str(value: object) -> str:
    """Render a df cell as a compact, HTML-escaped string for a table cell.

    ``np.floating`` is matched alongside ``float`` because ``np.float64`` IS a ``float`` subclass
    and ``np.float32`` is not — so without it the same number renders as ``0.262`` from a float64
    column and ``0.26161215`` from a float32 one, and a float32 NaN renders as the literal
    ``'nan'`` where a float64 NaN renders as empty. This library is float32 end to end and
    ``build_dashboard`` takes an arbitrary DataFrame, so both were reachable from a public entry
    point. Formatting now depends on the VALUE, not on how pandas happened to box it.
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        text = f"{value:.3f}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(v) for v in value)
    elif isinstance(value, np.ndarray):
        text = ", ".join(str(v) for v in value.tolist())
    else:
        text = str(value)
    return _html.escape(text)


def _slugify(text: str) -> str:
    """Filesystem-safe slug (lowercase, non-alnum -> ``_``) for playlist names."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in str(text))
    out = "_".join(part for part in out.split("_") if part)
    return out or "unknown"


def _scatter_figs(df: pd.DataFrame) -> tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Build the (valence×energy, UMAP-cluster) figures from a df, or None each."""
    labels = df["cluster"].tolist() if "cluster" in df.columns else [0] * len(df)
    files = df["filename"].astype(str).tolist() if "filename" in df.columns else [""] * len(df)
    moods = df["cluster_mood"].tolist() if "cluster_mood" in df.columns else None

    attr_fig = None
    if "energy" in df.columns and "valence" in df.columns:
        attr_fig = plot_attributes(
            df["energy"].tolist(), df["valence"].tolist(), labels, files, moods=moods
        )

    umap_fig = None
    if "x" in df.columns and "y" in df.columns:
        coords = np.column_stack([df["x"].to_numpy(float), df["y"].to_numpy(float)])
        medoids = None
        if "is_medoid" in df.columns:
            medoids = {int(i) for i, m in enumerate(df["is_medoid"].tolist()) if bool(m)}
        umap_fig = plot_clusters(coords, labels, files, moods=moods, medoids=medoids)

    return attr_fig, umap_fig


def _plots_html(df: pd.DataFrame) -> str:
    """The two scatter sections as HTML, or a stand-in line when neither can be built.

    Only the FIRST figure inlines plotly.js; the second reuses it. Two inlined copies would roughly
    double a page that is already the largest artefact this module writes.
    """
    attr_fig, umap_fig = _scatter_figs(df)

    blocks: list[str] = []
    for label, fig in (("Mood space (valence × energy)", attr_fig), ("UMAP clusters", umap_fig)):
        if fig is None:
            continue
        body = fig.to_html(full_html=False, include_plotlyjs="inline" if not blocks else False)
        blocks.append(f'<section class="plot"><h2>{_html.escape(label)}</h2>{body}</section>')

    return "\n".join(blocks) or "<p>No plottable columns available.</p>"


def _table_head(columns: Sequence[str], show_audio: bool) -> str:
    """Header cells, each wired to the client-side sort by its column index."""
    cells = "".join(
        f'<th onclick="sortTable({i})">{_html.escape(c)}</th>' for i, c in enumerate(columns)
    )
    return cells + "<th>preview</th>" if show_audio else cells


def _audio_cell(path: object) -> str:
    """One ``<audio>`` cell for a track ``path``, or an empty cell when the path is missing.

    An absolute path becomes a ``file://`` URI so the browser can actually open it; a relative one
    is emitted as given, since resolving it here would guess at the reader's working directory.
    """
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return "<td></td>"

    as_path = Path(str(path))
    src = _html.escape(as_path.as_uri() if as_path.is_absolute() else str(path))
    return f'<td><audio controls preload="none" src="{src}"></audio></td>'


def _table_rows(df: pd.DataFrame, columns: Sequence[str], show_audio: bool) -> str:
    """The table body, built COLUMN-wise.

    ``iterrows()`` materializes a fresh Series per row, which is what this loop actually spends its
    time on once a library is more than a few hundred tracks. Which element type the columns yield
    no longer matters: ``_cell_str`` formats float32 and float64 identically, so ``tolist()``
    (Python scalars, and the cheaper of the two) renders exactly what the row-wise path did. That
    equivalence is what the dtype fix in ``_cell_str`` buys — ``iterrows()`` unboxed to Python
    scalars on a MIXED frame but kept ``np.float32`` on a homogeneous one, so no single accessor
    reproduced it before.
    """
    if not len(df):
        return ""

    per_col = [df[c].tolist() for c in columns]
    cell_rows = (
        ["".join(f"<td>{_cell_str(v)}</td>" for v in values) for values in zip(*per_col)]
        if per_col  # a frame carrying none of _TABLE_COLUMNS still gets its rows
        else [""] * len(df)
    )

    if not show_audio:
        return "\n".join(f"<tr>{cells}</tr>" for cells in cell_rows)

    paths = df["path"].tolist()
    return "\n".join(
        f"<tr>{cells}{_audio_cell(paths[i])}</tr>" for i, cells in enumerate(cell_rows)
    )


def _dashboard_page(title: str, plots_html: str, head_cells: str, body_html: str) -> str:
    """The whole self-contained page: inline CSS, the plots, the table, and the filter/sort JS.

    Everything is inlined on purpose — no CDN, no external asset — so the file still opens from a
    local disk with no network, which is the point of handing someone a single HTML artefact.
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; color: #222; }}
  h1 {{ margin-top: 0; }}
  section.plot {{ margin-bottom: 2rem; }}
  #filter {{ padding: .4rem .6rem; width: 320px; max-width: 100%; margin-bottom: .6rem;
            border: 1px solid #ccc; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #e0e0e0; padding: .35rem .5rem; text-align: left;
           vertical-align: middle; }}
  th {{ background: #f5f5f5; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
  audio {{ height: 32px; }}
  .wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>{_html.escape(title)}</h1>
{plots_html}
<section>
  <h2>Tracks</h2>
  <input id="filter" type="text" placeholder="filter rows…" oninput="filterTable()">
  <div class="wrap">
  <table id="tracks">
    <thead><tr>{head_cells}</tr></thead>
    <tbody>
{body_html}
    </tbody>
  </table>
  </div>
</section>
<script>
function filterTable() {{
  var q = document.getElementById('filter').value.toLowerCase();
  var rows = document.querySelectorAll('#tracks tbody tr');
  rows.forEach(function(r) {{
    r.style.display = r.textContent.toLowerCase().indexOf(q) > -1 ? '' : 'none';
  }});
}}
function sortTable(col) {{
  var tbody = document.querySelector('#tracks tbody');
  var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
  var asc = tbody.getAttribute('data-sort-col') != col || tbody.getAttribute('data-sort-asc') != '1';
  rows.sort(function(a, b) {{
    var x = a.children[col] ? a.children[col].textContent.trim() : '';
    var y = b.children[col] ? b.children[col].textContent.trim() : '';
    var nx = parseFloat(x), ny = parseFloat(y);
    var cmp;
    if (!isNaN(nx) && !isNaN(ny)) cmp = nx - ny;
    else cmp = x.localeCompare(y);
    return asc ? cmp : -cmp;
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
  tbody.setAttribute('data-sort-col', col);
  tbody.setAttribute('data-sort-asc', asc ? '1' : '0');
}}
</script>
</body>
</html>
"""


def build_dashboard(
    df: pd.DataFrame,
    out_html: PathLike,
    title: str = "Mood explorer",
    audio_dir: Optional[PathLike] = None,
) -> Path:
    """Write ONE self-contained HTML dashboard (inline CSS+JS, no external/CDN).

    Combines a valence×energy scatter, a UMAP cluster scatter, and a
    sortable/text-filterable table of the tracks (filename, cluster, cluster_mood,
    top_mood, top_score, mood_top3, energy, valence). When ``audio_dir`` is given,
    each row gets an inline ``<audio controls>`` whose ``src`` is the track ``path``
    (``file://``) so the user can preview it; missing paths degrade gracefully.

    Robust to missing columns and empty frames. Returns the written :class:`Path`.
    """
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    show_audio = audio_dir is not None and "path" in df.columns
    columns = [c for c in _TABLE_COLUMNS if c in df.columns]

    page = _dashboard_page(
        title,
        _plots_html(df),
        _table_head(columns, show_audio),
        _table_rows(df, columns, show_audio),
    )

    out.write_text(page, encoding="utf-8")
    return out


def export_m3u(df: pd.DataFrame, out_dir: PathLike) -> list[Path]:
    """Write one ``.m3u`` playlist per cluster, returning the written paths.

    Each file is named ``cluster_<id>_<cluster_mood>.m3u`` and lists the absolute
    ``path`` of its member tracks (one per line) under an ``#EXTM3U`` header.
    ``df`` needs at least ``'cluster'`` and ``'path'`` columns; missing columns or
    an empty frame yield an empty list. Robust; returns paths ascending by cluster.
    """
    out = _prepare_playlist_dir(out_dir, "cluster_*.m3u")
    if not _has_columns(df, "cluster", "path"):
        return []

    has_mood = "cluster_mood" in df.columns
    written: list[Path] = []
    clusters = sorted({int(c) for c in df["cluster"].tolist()})
    for cluster in clusters:
        sub = df[df["cluster"] == cluster]
        mood = _cluster_mood(sub, has_mood)
        path = out / f"cluster_{cluster}{_m3u_suffix(mood, cluster)}.m3u"

        lines = ["#EXTM3U"] + _track_paths(sub)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)

    return written


# --------------------------------------------------------------------------- #
# TRACK 4 — gold-set labeling UI (lives here so viz stays the HTML surface)
# --------------------------------------------------------------------------- #


def build_labeling_ui(
    filenames: Sequence[str],
    paths: Sequence[str],
    moods: Sequence[str],
    out_html: PathLike,
    audio_dir: Optional[PathLike] = None,
) -> Path:
    """Write a self-contained gold-set labeling form (inline CSS+JS, no network).

    For each track it renders an ``<audio>`` element (``file://`` from ``paths``),
    a row of mood checkboxes (from ``moods``) and energy/valence sliders in [0, 1].
    A "Download JSON" button serializes ``{filename: {moods, energy, valence}}`` to
    a Blob entirely client-side (no backend), making evaluation falsifiable.

    Robust to ragged ``paths`` / empty inputs. Returns the written :class:`Path`.

    ``audio_dir`` is accepted for symmetry with :func:`build_dashboard`, whose trailing parameter
    it mirrors, and is never read: every ``<audio>`` src here comes from ``paths``.
    """
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in filenames]
    path_list = [str(p) for p in paths]
    mood_list = [str(m) for m in moods]

    # `<` escaped to its \u003c form before the JSON reaches the <script> element. json.dumps
    # leaves `/` alone, so a mood containing `</script>` would otherwise CLOSE the script tag and
    # everything after it would be parsed as markup. Escaping `<` covers every way out of a script
    # element, and it is invisible to the client: JSON.parse restores the original character.
    mood_json = _json.dumps(mood_list).replace("<", "\\u003c")

    cards: list[str] = []
    for i, fn in enumerate(files):
        esc_fn = _html.escape(fn)
        raw_path = path_list[i] if i < len(path_list) else ""
        audio = ""
        if raw_path:
            try:
                src = Path(raw_path).as_uri() if Path(raw_path).is_absolute() else raw_path
            except (ValueError, OSError):
                src = raw_path
            audio = f'<audio controls preload="none" src="{_html.escape(src)}"></audio>'
        checks = "".join(
            f'<label class="mood"><input type="checkbox" data-track="{i}" '
            f'value="{_html.escape(m)}"> {_html.escape(m)}</label>'
            for m in mood_list
        )
        cards.append(
            f'<div class="card" data-filename="{esc_fn}">'
            f'<div class="fn">{esc_fn}</div>'
            f"{audio}"
            f'<div class="moods">{checks}</div>'
            f'<div class="sliders">'
            f'<label>energy <input type="range" min="0" max="1" step="0.01" value="0.5" '
            f'class="energy" data-track="{i}" oninput="this.nextElementSibling.textContent=this.value">'
            f"<span>0.50</span></label>"
            f'<label>valence <input type="range" min="0" max="1" step="0.01" value="0.5" '
            f'class="valence" data-track="{i}" oninput="this.nextElementSibling.textContent=this.value">'
            f"<span>0.50</span></label>"
            f"</div></div>"
        )
    cards_html = "\n".join(cards) or "<p>No tracks provided.</p>"

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold-set labeling</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; color: #222; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: .8rem 1rem;
          margin-bottom: 1rem; }}
  .fn {{ font-weight: 600; margin-bottom: .4rem; }}
  .moods {{ margin: .5rem 0; }}
  label.mood {{ display: inline-block; margin-right: .8rem; white-space: nowrap; }}
  .sliders label {{ display: inline-block; margin-right: 1.5rem; }}
  audio {{ height: 32px; display: block; margin: .3rem 0; }}
  #download {{ position: sticky; top: 0; padding: .5rem 1rem; font-size: 15px;
              background: #1f77b4; color: #fff; border: none; border-radius: 4px;
              cursor: pointer; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>Gold-set labeling</h1>
<button id="download" onclick="downloadJSON()">Download JSON</button>
<div id="cards">
{cards_html}
</div>
<script>
var MOODS = {mood_json};
function downloadJSON() {{
  var result = {{}};
  var cards = document.querySelectorAll('.card');
  cards.forEach(function(card, i) {{
    var fn = card.getAttribute('data-filename');
    var moods = [];
    card.querySelectorAll('input[type=checkbox]:checked').forEach(function(cb) {{
      moods.push(cb.value);
    }});
    var energy = parseFloat(card.querySelector('.energy').value);
    var valence = parseFloat(card.querySelector('.valence').value);
    result[fn] = {{ moods: moods, energy: energy, valence: valence }};
  }});
  var blob = new Blob([JSON.stringify(result, null, 2)], {{ type: 'application/json' }});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'gold.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""
    out.write_text(page, encoding="utf-8")
    return out
