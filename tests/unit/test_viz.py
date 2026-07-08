"""Tests for :mod:`moodengine.viz` — cluster scatter plot + playlist export."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from assertpy import assert_that

from moodengine.viz import (
    build_dashboard,
    export_m3u,
    export_playlists,
    plot_attributes,
    plot_clusters,
)


def _toy_coords(n: int = 6) -> np.ndarray:
    """Deterministic 2-D coordinates for ``n`` points."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2)).astype(np.float32)


def test_plot_clusters_returns_figure() -> None:
    """A small input yields a plotly Figure with one trace per cluster label."""
    coords = _toy_coords(6)
    labels = [0, 0, 1, 1, -1, -1]
    files = [f"track_{i}.wav" for i in range(6)]
    fig = plot_clusters(coords, labels, files, title="t")
    assert_that(fig).is_instance_of(go.Figure)
    # One trace per distinct label (0, 1, noise).
    assert_that(fig.data).is_length(3)
    assert_that(fig.layout.title.text).is_equal_to("t")
    # Contract: label -1 is rendered as a distinct "noise" trace. Pin the name (the
    # docstring promises that exact word) but not the incidental hex/naming of the real
    # clusters, so a correct implementation isn't over-constrained.
    trace_names = [tr.name for tr in fig.data]
    assert_that(trace_names).contains("noise")
    # The two real clusters get their own, distinct, non-noise traces.
    non_noise_names = [n for n in trace_names if n != "noise"]
    assert_that(non_noise_names).is_length(2)
    assert_that(set(non_noise_names)).is_length(2)


def test_plot_clusters_hover_includes_mood() -> None:
    """When moods are supplied the hover text mentions the mood."""
    coords = _toy_coords(2)
    fig = plot_clusters(
        coords, labels=[0, 0], filenames=["a.wav", "b.wav"], moods=["happy", "calm"]
    )
    hover_text = " ".join(str(t) for tr in fig.data for t in (tr.text or []))
    assert_that(hover_text).contains("happy")
    assert_that(hover_text).contains("a.wav")


def test_plot_clusters_writes_self_contained_html(tmp_path) -> None:
    """``out_html`` writes a standalone HTML file with plotly.js inlined."""
    coords = _toy_coords(4)
    out = tmp_path / "nested" / "clusters.html"
    fig = plot_clusters(coords, [0, 1, 0, 1], ["a", "b", "c", "d"], out_html=out)
    assert_that(fig).is_instance_of(go.Figure)
    assert_that(out.exists()).is_true()
    html = out.read_text(encoding="utf-8")
    assert_that(html.lower()).contains("<html")
    # plotly.js inlined -> the bundle text is present, not just a CDN <script src>.
    assert_that(html).contains("Plotly")


def test_plot_clusters_hover_text_overrides_default(tmp_path) -> None:
    """An explicit ``hover_text`` replaces the default filename/mood hover string."""
    coords = _toy_coords(2)
    custom = ["a.wav<br>moods: happy 53%, calm 21%", "b.wav<br>energy 0.42 · valence 0.31"]
    fig = plot_clusters(
        coords,
        labels=[0, 0],
        filenames=["a.wav", "b.wav"],
        moods=["happy", "calm"],  # should be ignored in favor of hover_text
        hover_text=custom,
    )
    hover = [t for tr in fig.data for t in (tr.text or [])]
    assert_that(set(hover)).is_equal_to(set(custom))
    # The override is verbatim — the default "mood: <m>" wording is NOT injected.
    assert_that(all("mood: " not in h for h in hover)).is_true()
    assert_that(any("53%" in h for h in hover)).is_true()


def test_plot_clusters_empty_input() -> None:
    """An empty dataset produces an empty figure without raising."""
    fig = plot_clusters(np.zeros((0, 2), dtype=np.float32), [], [])
    assert_that(fig).is_instance_of(go.Figure)
    assert_that(fig.data).is_length(0)


def test_plot_attributes_returns_figure_one_trace_per_cluster() -> None:
    """``plot_attributes`` makes a valence×energy scatter, one trace per cluster."""
    energy = [0.1, 0.2, 0.8, 0.9, 0.5]
    valence = [0.9, 0.8, 0.2, 0.1, 0.5]
    labels = [0, 0, 1, 1, -1]
    files = [f"t{i}.wav" for i in range(5)]
    fig = plot_attributes(energy, valence, labels, files, moods=["happy"] * 5, title="space")
    assert_that(fig).is_instance_of(go.Figure)
    # One trace per distinct label: clusters 0, 1 and noise.
    assert_that(fig.data).is_length(3)
    assert_that(fig.layout.title.text).is_equal_to("space")
    trace_names = [tr.name for tr in fig.data]
    assert_that(trace_names).contains("noise")
    # Hover surfaces the (valence, energy) coordinates and the filename.
    hover = " ".join(str(t) for tr in fig.data for t in (tr.text or []))
    assert_that(hover).contains("valence")
    assert_that(hover).contains("energy")
    assert_that(hover).contains("t0.wav")


def test_plot_attributes_writes_self_contained_html(tmp_path) -> None:
    """``out_html`` writes a standalone HTML file with plotly.js inlined."""
    out = tmp_path / "nested" / "mood_space.html"
    fig = plot_attributes([0.2, 0.7], [0.3, 0.6], [0, 1], ["a.wav", "b.wav"], out_html=out)
    assert_that(fig).is_instance_of(go.Figure)
    assert_that(out.exists()).is_true()
    html = out.read_text(encoding="utf-8")
    assert_that(html.lower()).contains("<html")
    assert_that(html).contains("Plotly")


def test_plot_attributes_empty_input() -> None:
    """An empty mood space produces an empty figure without raising."""
    fig = plot_attributes([], [], [], [])
    assert_that(fig).is_instance_of(go.Figure)
    assert_that(fig.data).is_length(0)


def test_export_playlists_writes_one_file_per_cluster(tmp_path) -> None:
    """One text file per cluster, names contents, and noise named distinctly."""
    df = pd.DataFrame(
        {
            "cluster": [0, 0, 1, -1],
            "filename": ["a.wav", "b.wav", "c.wav", "d.wav"],
        }
    )
    paths = export_playlists(df, tmp_path)
    assert_that(paths).is_length(3)
    names = sorted(p.name for p in paths)
    assert_that(names).is_equal_to(["cluster_-1_noise.txt", "cluster_00.txt", "cluster_01.txt"])

    c0 = (tmp_path / "cluster_00.txt").read_text(encoding="utf-8").splitlines()
    assert_that(c0).is_equal_to(["a.wav", "b.wav"])
    noise = (tmp_path / "cluster_-1_noise.txt").read_text(encoding="utf-8").splitlines()
    assert_that(noise).is_equal_to(["d.wav"])


def test_export_playlists_creates_out_dir(tmp_path) -> None:
    """A missing output directory is created before writing."""
    out_dir = tmp_path / "playlists"
    df = pd.DataFrame({"cluster": [0], "filename": ["x.wav"]})
    paths = export_playlists(df, out_dir)
    assert_that(out_dir.is_dir()).is_true()
    assert_that(paths).is_length(1)
    assert_that(paths[0].read_text(encoding="utf-8").splitlines()).is_equal_to(["x.wav"])


def test_export_playlists_empty_df_returns_empty(tmp_path) -> None:
    """An empty / column-less DataFrame yields no files and no error."""
    assert_that(export_playlists(pd.DataFrame(), tmp_path)).is_equal_to([])
    assert_that(
        export_playlists(pd.DataFrame({"cluster": [], "filename": []}), tmp_path)
    ).is_equal_to([])


# --------------------------------------------------------------------------- #
# plot_clusters — medoids highlight
# --------------------------------------------------------------------------- #


def test_plot_clusters_medoids_adds_highlight_trace() -> None:
    """Passing ``medoids`` overdraws a dedicated 'medoid' trace on the given points."""
    coords = _toy_coords(6)
    labels = [0, 0, 1, 1, 2, 2]
    files = [f"t{i}.wav" for i in range(6)]
    plain = plot_clusters(coords, labels, files)
    with_med = plot_clusters(coords, labels, files, medoids={1, 4})

    assert_that([tr.name for tr in plain.data]).does_not_contain("medoid")
    med_traces = [tr for tr in with_med.data if tr.name == "medoid"]
    assert_that(med_traces).is_length(1)
    # Two medoid points highlighted, with a distinctive diamond marker.
    assert_that(med_traces[0].x).is_length(2)
    assert_that(med_traces[0].marker.symbol).is_equal_to("diamond")


def test_plot_clusters_medoids_none_is_backward_compatible() -> None:
    """The default ``medoids=None`` leaves the figure unchanged (no extra trace)."""
    coords = _toy_coords(4)
    fig = plot_clusters(coords, [0, 0, 1, 1], ["a", "b", "c", "d"], medoids=None)
    assert_that([tr.name for tr in fig.data]).does_not_contain("medoid")


# --------------------------------------------------------------------------- #
# build_dashboard
# --------------------------------------------------------------------------- #


def _dashboard_df(n: int = 5) -> pd.DataFrame:
    """A small df carrying every column the dashboard knows how to render."""
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((n, 2))
    return pd.DataFrame(
        {
            "filename": [f"track_{i}.wav" for i in range(n)],
            "path": [f"/music/track_{i}.wav" for i in range(n)],
            "cluster": [i % 2 for i in range(n)],
            "cluster_mood": ["happy" if i % 2 == 0 else "dark" for i in range(n)],
            "top_mood": ["happy", "calm", "dark", "epic", "groovy"][:n],
            "top_score": [0.6, 0.5, 0.7, 0.4, 0.55][:n],
            "mood_top3": [["happy", "calm", "epic"]] * n,
            "energy": [0.2, 0.4, 0.6, 0.8, 0.5][:n],
            "valence": [0.9, 0.7, 0.3, 0.1, 0.5][:n],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "is_medoid": [i == 0 for i in range(n)],
        }
    )


def test_build_dashboard_writes_self_contained_html(tmp_path) -> None:
    """The dashboard is one non-trivial, fully self-contained HTML file."""
    out = tmp_path / "nested" / "dashboard.html"
    path = build_dashboard(_dashboard_df(), out)
    assert_that(path).is_equal_to(out)
    assert_that(out.exists()).is_true()
    html = out.read_text(encoding="utf-8")

    assert_that(len(html)).is_greater_than(5000)  # non-trivial
    assert_that(html.lower()).contains("<html")
    assert_that(html).contains("Plotly")  # the scatter is embedded
    # Self-contained: no external/CDN resource *tags*. (URL string literals baked
    # into the inlined plotly.js bundle are not network fetches by the page.)
    flat = html.replace(" ", "")
    assert_that(flat).does_not_contain("<scriptsrc=")
    assert_that(flat).does_not_contain("<linkhref=")
    # Table content + the inline filter/sort JS are present.
    assert_that(html).contains("track_0.wav")
    assert_that(html).contains("filterTable")
    assert_that(html).contains("sortTable")


def test_build_dashboard_inline_audio_when_audio_dir(tmp_path) -> None:
    """With ``audio_dir`` each row gets an inline file:// <audio> preview.

    Paths must be absolute *for the running platform* (tmp_path-based): the
    factory's POSIX-style ``/music/...`` strings are not absolute on Windows,
    and a bare ``file://`` grep would only match plotly's own bundled JS."""
    out = tmp_path / "dash.html"
    df = _dashboard_df(3)
    df["path"] = [str(tmp_path / name) for name in df["filename"]]
    build_dashboard(df, out, audio_dir=tmp_path)
    html = out.read_text(encoding="utf-8")
    assert_that(html).contains("<audio")
    assert_that(html.count('src="file://')).is_equal_to(
        3
    )  # one real preview per track, from our rows


def test_build_dashboard_robust_to_missing_columns(tmp_path) -> None:
    """A bare df (only filename/cluster) still produces a valid HTML page."""
    out = tmp_path / "min_dash.html"
    df = pd.DataFrame({"filename": ["a.wav", "b.wav"], "cluster": [0, 1]})
    path = build_dashboard(df, out)
    assert_that(path.exists()).is_true()
    html = out.read_text(encoding="utf-8")
    assert_that(html.lower()).contains("<html")
    assert_that(html).contains("a.wav")


# --------------------------------------------------------------------------- #
# export_m3u
# --------------------------------------------------------------------------- #


def test_export_m3u_one_file_per_cluster_with_mood_names(tmp_path) -> None:
    """One ``.m3u`` per cluster, named with the cluster mood, listing abs paths."""
    df = pd.DataFrame(
        {
            "cluster": [0, 0, 1],
            "cluster_mood": ["happy groove", "happy groove", "dark"],
            "path": ["/m/a.wav", "/m/b.wav", "/m/c.wav"],
        }
    )
    paths = export_m3u(df, tmp_path)
    assert_that(paths).is_length(2)
    names = sorted(p.name for p in paths)
    assert_that(names).is_equal_to(["cluster_0_happy_groove.m3u", "cluster_1_dark.m3u"])

    c0 = (tmp_path / "cluster_0_happy_groove.m3u").read_text(encoding="utf-8").splitlines()
    assert_that(c0[0]).is_equal_to("#EXTM3U")
    assert_that(c0[1:]).is_equal_to(["/m/a.wav", "/m/b.wav"])


def test_export_m3u_missing_columns_returns_empty(tmp_path) -> None:
    """No ``path`` (or no ``cluster``) column -> no files, no error."""
    assert_that(export_m3u(pd.DataFrame(), tmp_path)).is_equal_to([])
    assert_that(
        export_m3u(pd.DataFrame({"cluster": [0], "filename": ["a.wav"]}), tmp_path)
    ).is_equal_to([])
