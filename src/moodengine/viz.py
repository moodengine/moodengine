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
    coords = np.asarray(coords2d, dtype=float)
    if coords.ndim != 2 or (coords.size and coords.shape[1] < 2):
        coords = coords.reshape(-1, 2) if coords.size else coords.reshape(0, 2)
    n = coords.shape[0]

    labels_arr = np.asarray(list(labels), dtype=int) if len(labels) else np.empty(0, dtype=int)
    files = list(filenames)
    mood_list = list(moods) if moods is not None else None
    hover_list = list(hover_text) if hover_text is not None else None

    # Stable cluster ordering (ascending) so colors/legend are deterministic.
    unique = sorted({int(v) for v in labels_arr.tolist()})
    non_noise = [c for c in unique if c != -1]
    order = {c: i for i, c in enumerate(non_noise)}

    fig = go.Figure()
    # Draw noise last so real clusters sit on top? Plotly draws in add order;
    # iterate ascending which puts -1 first (under the colored points).
    for cluster in unique:
        mask = labels_arr == cluster
        idx = np.flatnonzero(mask)
        # Only plot points that have coordinates; guards against a labels/coords
        # length mismatch so we degrade gracefully instead of raising IndexError.
        idx = idx[idx < n]
        if idx.size == 0:
            continue
        name = "noise" if cluster == -1 else f"cluster {cluster}"
        hover = []
        for i in idx:
            if hover_list is not None and i < len(hover_list) and hover_list[i] is not None:
                hover.append(str(hover_list[i]))
                continue
            fn = files[i] if i < len(files) else ""
            if mood_list is not None and i < len(mood_list) and mood_list[i] is not None:
                hover.append(f"{fn}<br>mood: {mood_list[i]}")
            else:
                hover.append(str(fn))
        fig.add_trace(
            go.Scatter(
                x=coords[idx, 0],
                y=coords[idx, 1],
                mode="markers",
                name=name,
                marker=dict(
                    color=_cluster_color(cluster, order),
                    size=9,
                    line=dict(width=0.5, color="#ffffff"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    # Overdraw cluster representatives (medoids) so they stand out.
    if medoids:
        med_idx = np.array(
            sorted(i for i in medoids if isinstance(i, (int, np.integer)) and 0 <= i < n),
            dtype=int,
        )
        if med_idx.size:
            med_hover = []
            for i in med_idx:
                fn = files[i] if i < len(files) else ""
                med_hover.append(f"medoid<br>{fn}")
            fig.add_trace(
                go.Scatter(
                    x=coords[med_idx, 0],
                    y=coords[med_idx, 1],
                    mode="markers",
                    name="medoid",
                    marker=dict(
                        symbol="diamond",
                        color="rgba(0,0,0,0)",
                        size=16,
                        line=dict(width=2, color="#000000"),
                    ),
                    text=med_hover,
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title="cluster",
        template="plotly_white",
    )

    if out_html is not None:
        out = Path(out_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs=True, full_html=True)

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

    unique = sorted({int(x) for x in labels_arr.tolist()})
    non_noise = [c for c in unique if c != -1]
    order = {c: i for i, c in enumerate(non_noise)}

    fig = go.Figure()
    for cluster in unique:
        idx = np.flatnonzero(labels_arr == cluster)
        idx = idx[idx < n]
        if idx.size == 0:
            continue
        hover = []
        for i in idx:
            fn = files[i] if i < len(files) else ""
            parts = [str(fn)]
            if mood_list is not None and i < len(mood_list) and mood_list[i] is not None:
                parts.append(f"mood: {mood_list[i]}")
            parts.append(f"valence={v[i]:.2f} energy={e[i]:.2f}")
            hover.append("<br>".join(parts))
        fig.add_trace(
            go.Scatter(
                x=v[idx],
                y=e[idx],
                mode="markers",
                name="noise" if cluster == -1 else f"cluster {cluster}",
                marker=dict(
                    color=_cluster_color(cluster, order),
                    size=10,
                    line=dict(width=0.5, color="#ffffff"),
                ),
                text=hover,
                hoverinfo="text",
            )
        )

    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color="#cccccc")
    fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#cccccc")
    fig.update_layout(
        title=title,
        xaxis=dict(title="valence  (dark/negative → bright/positive)", range=[0, 1]),
        yaxis=dict(title="energy  (calm/low → intense/high)", range=[0, 1]),
        legend_title="cluster",
        template="plotly_white",
    )

    if out_html is not None:
        out = Path(out_html)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs=True, full_html=True)

    return fig


def export_playlists(df: pd.DataFrame, out_dir: PathLike) -> list[Path]:
    """Write one playlist text file per cluster and return the written paths.

    ``df`` must have at least ``'cluster'`` and ``'filename'`` columns. Files are
    named ``cluster_00.txt`` (zero-padded, ascending), with noise written to
    ``cluster_-1_noise.txt``. Each file lists the cluster's filenames, one per
    line. Returns the list of written :class:`pathlib.Path` (ascending by cluster).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Clear stale playlists from a previous run (which may have had more/other
    # clusters) so the directory always reflects the current clustering.
    for old in out.glob("cluster_*.txt"):
        old.unlink()

    if df is None or len(df) == 0 or "cluster" not in df.columns or "filename" not in df.columns:
        return []

    written: list[Path] = []
    clusters = sorted({int(c) for c in df["cluster"].tolist()})
    for cluster in clusters:
        names = df.loc[df["cluster"] == cluster, "filename"].astype(str).tolist()
        if cluster == -1:
            fname = "cluster_-1_noise.txt"
        else:
            fname = f"cluster_{cluster:02d}.txt"
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
    """Render a df cell as a compact, HTML-escaped string for a table cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
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
    attr_fig, umap_fig = _scatter_figs(df)

    # First plot inlines plotly.js; the second reuses it (include_plotlyjs=False).
    plot_blocks: list[str] = []
    first = True
    for label, fig in (("Mood space (valence × energy)", attr_fig), ("UMAP clusters", umap_fig)):
        if fig is None:
            continue
        inc = "inline" if first else False
        body = fig.to_html(full_html=False, include_plotlyjs=inc)
        plot_blocks.append(f'<section class="plot"><h2>{_html.escape(label)}</h2>{body}</section>')
        first = False
    plots_html = "\n".join(plot_blocks) or "<p>No plottable columns available.</p>"

    show_audio = audio_dir is not None and "path" in df.columns
    columns = [c for c in _TABLE_COLUMNS if c in df.columns]

    # Build the table head.
    head_cells = "".join(
        f'<th onclick="sortTable({i})">{_html.escape(c)}</th>' for i, c in enumerate(columns)
    )
    if show_audio:
        head_cells += "<th>preview</th>"

    # Build the table body.
    rows_html: list[str] = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_cell_str(row.get(c))}</td>" for c in columns)
        if show_audio:
            path = row.get("path")
            if path is not None and not (isinstance(path, float) and np.isnan(path)):
                src = (
                    _html.escape(Path(str(path)).as_uri())
                    if Path(str(path)).is_absolute()
                    else _html.escape(str(path))
                )
                cells += f'<td><audio controls preload="none" src="{src}"></audio></td>'
            else:
                cells += "<td></td>"
        rows_html.append(f"<tr>{cells}</tr>")
    body_html = "\n".join(rows_html)

    page = f"""<!doctype html>
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
    out.write_text(page, encoding="utf-8")
    return out


def export_m3u(df: pd.DataFrame, out_dir: PathLike) -> list[Path]:
    """Write one ``.m3u`` playlist per cluster, returning the written paths.

    Each file is named ``cluster_<id>_<cluster_mood>.m3u`` and lists the absolute
    ``path`` of its member tracks (one per line) under an ``#EXTM3U`` header.
    ``df`` needs at least ``'cluster'`` and ``'path'`` columns; missing columns or
    an empty frame yield an empty list. Robust; returns paths ascending by cluster.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Clear stale playlists from a previous run.
    for old in out.glob("cluster_*.m3u"):
        old.unlink()

    if df is None or len(df) == 0 or "cluster" not in df.columns or "path" not in df.columns:
        return []

    has_mood = "cluster_mood" in df.columns
    written: list[Path] = []
    clusters = sorted({int(c) for c in df["cluster"].tolist()})
    for cluster in clusters:
        sub = df[df["cluster"] == cluster]
        mood = ""
        if has_mood and len(sub):
            mv = sub["cluster_mood"].iloc[0]
            mood = "" if mv is None or (isinstance(mv, float) and np.isnan(mv)) else str(mv)
        suffix = f"_{_slugify(mood)}" if mood else ("_noise" if cluster == -1 else "")
        fname = f"cluster_{cluster}{suffix}.m3u"
        paths = [
            str(p)
            for p in sub["path"].tolist()
            if p is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        lines = ["#EXTM3U"] + paths
        path = out / fname
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
    """
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in filenames]
    path_list = [str(p) for p in paths]
    mood_list = [str(m) for m in moods]

    mood_json = _json.dumps(mood_list)

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
