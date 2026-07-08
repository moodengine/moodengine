# moodengine

[![CI](https://github.com/moodengine/moodengine/actions/workflows/ci.yml/badge.svg)](https://github.com/moodengine/moodengine/actions/workflows/ci.yml)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

**Music mood analysis engine** — turn a folder of audio files into mood-labeled, explorable clusters.

moodengine is a pure, stateless Python library (MIR toolbox) that extracts audio embeddings
(**MERT**, **CLAP**), pools them to track level, clusters tracks by mood/ambience
(UMAP + HDBSCAN / KMeans / spherical / Leiden), names the clusters with **zero-shot
mood labels** (calibrated CLAP text↔audio similarities), and ships the surrounding
toolbox: text→audio and audio→audio search, evaluation metrics, calibration,
playlist export, and interactive HTML visualizations.

Design principles:

- **Functional core** — pure functions + small frozen dataclasses, sklearn-style.
  No global state; compute functions never write to disk.
- **Light by default** — `import moodengine` is torch-free. Clustering, labeling
  math, search, evaluation and viz run on precomputed embeddings with only the
  scientific-Python stack. The deep-learning backbones are an optional extra.
- **Cached, content-addressed embeddings** — each track's embedding is cached as
  a `.npy` keyed by file content hash + model variant + pooling params, so nothing
  is ever recomputed unless the audio or the config actually changed.
- Fully typed (`py.typed`), tested on a torch-free install, Python 3.11+.

## Installation

```bash
pip install moodengine            # light core (not yet on PyPI: pip install "moodengine @ git+https://github.com/moodengine/moodengine")
pip install "moodengine[models]"  # + MERT/CLAP backbones (pulls torch, several GB)
```

| Extra | Enables | Pulls |
|---|---|---|
| *(none)* | clustering, labeling math, search, eval, viz on precomputed embeddings | numpy/pandas/sklearn/umap/hdbscan/librosa/plotly |
| `[models]` | embedding real audio with MERT + CLAP | torch, transformers, laion-clap (~GBs, downloads model checkpoints on first use) |
| `[ot]` | optimal-transport journey morphing | POT |
| `[cluster-graph]` | Leiden community detection | leidenalg, python-igraph |
| `[pacmap]` | PaCMAP 2-D projection (`projection_method="pacmap"`) | pacmap |
| `[explain]` | TreeSHAP attribution backend | shap |

`ffmpeg` is recommended so librosa can decode MP3/M4A.

## Quickstart

### Cluster precomputed embeddings (light install)

```python
import numpy as np
from moodengine import default_config, run_clustering

X = np.load("embeddings.npy")  # (n_tracks, dim) float32 track embeddings

result = run_clustering(X, "kmeans", default_config())
print(result["labels"])        # (n,) cluster id per track
print(result["metrics"])       # silhouette, sizes, noise ratio…
print(result["coords2d"])      # (n, 2) map coordinates for plotting
```

### Full pipeline: audio in, labeled clusters out (needs `[models]`)

```python
from dataclasses import replace
from pathlib import Path

from moodengine import default_config, run_pipeline

config = replace(
    default_config(),
    raw_dir=Path("~/Music/chill").expanduser(),   # your audio files
    output_dir=Path("moodengine-out"),
)

# Cluster in the CLAP space, auto-pick k by silhouette, zero-shot label the result.
df = run_pipeline(config, embedder_name="clap", method="kmeans", with_labels=True)
print(df[["filename", "cluster", "top_mood", "energy", "valence", "cluster_profile"]])
```

`run_pipeline` writes its artifacts under `config.output_dir`: `assignments.parquet`
(one row per track: cluster, calibrated top-3 moods, energy/valence in [0, 1],
medoid + outlier scores), interactive `dashboard.html` and `clusters.html` /
`mood_space.html` scatters, a `cluster_report.md`, an annotation UI
(`label_ui.html`) and per-cluster `.m3u` playlists.

The clustering space and the labeling model are independent: cluster with
`embedder_name="mert"`, `"clap"` or `"fused"` — zero-shot labels always come
from CLAP. Device is auto-detected (CUDA > Apple MPS > CPU). On Apple Silicon,
export `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op unsupported by MPS falls back
to CPU instead of crashing.

### Paths and caching

- Embeddings are cached under the per-user platform cache directory by default
  (`%LOCALAPPDATA%\moodengine` on Windows, `~/Library/Caches/moodengine` on macOS,
  `~/.cache/moodengine` on Linux) — override with `Config.cache_dir`.
- `Config.raw_dir` / `Config.output_dir` default to `./data/raw` and `./outputs`
  relative to the working directory; pass explicit paths in real use.
- The cache key includes the audio content hash, the model variant and the
  pooling/segmentation params — changing any of them cleanly invalidates only
  the affected entries.

## What's in the box

| Module | Purpose |
|---|---|
| `embeddings` | `Embedder` interface, MERT (frame-level) + CLAP (clip-level + text) wrappers, on-disk cache |
| `pooling` | frame/segment → track vectors (mean / mean+std, MERT layer weighting), L2 discipline |
| `cluster` | UMAP reduction, HDBSCAN/KMeans/spherical/Leiden, auto-k, metrics, bootstrap stability, medoids, sub-clustering |
| `labeling` | zero-shot mood taxonomy (prompt ensembling, customizable), softmax calibration, similarity recentering, energy/valence axes, cluster mood profiles |
| `search` | text→audio and audio→audio cosine search (pure numpy) |
| `evaluation`, `calibration` | retrieval P@k, axis AUC, gold-set tooling, score calibration |
| `novelty`, `signals`, `sequence`, `journey`, `adapt`, `explain`, `feedback` | OOD/near-duplicate detection, BPM/key signals, next-track models, playlist morphing, metric adapters, SHAP explanations, feedback loops |
| `viz` | self-contained HTML dashboard/scatters, annotation UI, playlist export |
| `pipeline` | end-to-end orchestration with caching (the only module that writes files) |

Practical guidance from our own evaluations: cluster in the **CLAP or fused**
space rather than MERT alone (bootstrap stability strongly favors them), and
keep `recenter_labels=True` — recentering cancels CLAP's modality gap, which
otherwise skews every track toward the same few moods.

## Example scripts

`scripts/` contains CLI walkthroughs (extract → cluster → label → search →
evaluate). Their `typer` dependency ships with the dev group (`uv sync`):

```bash
uv run python scripts/01_extract_embeddings.py --input-dir path/to/audio --embedder clap
uv run python scripts/03_label_and_explore.py --embedder clap --method kmeans
```

## Model licensing & network access

- The `[models]` extra downloads checkpoints from the Hugging Face hub and the
  laion-clap release page on first use.
- **Offline / air-gapped use**: pre-download the checkpoints once
  (`huggingface-cli download m-a-p/MERT-v1-95M` and `huggingface-cli download
  lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt`), point `HF_HOME` at
  the cache location if needed, and set `HF_HUB_OFFLINE=1` to force cache-only
  resolution. A load failure raises `ModelLoadError` naming the exact artifact
  and this remediation.
- **MERT-v1-95M weights are CC-BY-NC-4.0 (non-commercial)** — a separate grant
  from this package's own [license](#license). Any commercial use of the weights
  requires licensing them separately or configuring another model
  (`Config.mert_model_name`).
- MERT executes custom modeling code from the hub (`trust_remote_code`); moodengine
  pins the default model to a reviewed revision. Override with `Config.mert_revision`.

## Error handling

Every library-specific failure derives from `moodengine.MoodengineError`:
`AudioDecodeError` (an existing file cannot be decoded — a missing path raises
stdlib `FileNotFoundError` instead), `MissingDependencyError` (an optional
backend is absent; the message names the exact `pip install "moodengine[...]"`
command), `ModelLoadError` (a checkpoint could not be fetched or loaded).
Argument errors stay plain `ValueError`, following the numpy/sklearn convention.

## Development

```bash
git clone https://github.com/moodengine/moodengine && cd moodengine
uv sync              # light install + dev tooling (pytest, ruff)
uv run pytest        # default suite — torch-free by contract
uv run pytest --cov  # same suite + coverage floor (fail_under in pyproject)
uv run pytest -m model   # opt-in: real MERT/CLAP tests (needs `uv sync --extra models`)
uv run ruff format . && uv run ruff check .
uv run mypy && uv run deptry .
```

The default test suite must pass on a light install — anything touching torch
or an optional backend skips itself cleanly. See
[CONTRIBUTING.md](CONTRIBUTING.md) for conventions (conventional commits,
SemVer policy, test layout) and [docs/](docs/) for developer guides (e.g.
[benchmarking against ground truth](docs/benchmarking.md)). API documentation
lives in the source docstrings — every public symbol is fully documented there.

## Compatibility

- Python 3.11+ · numpy ≥ 1.26 (tested under numpy 2.x) · all dependencies ship
  cross-platform wheels (macOS arm64 / Windows / Linux).
- `transformers` is capped `<5`: MERT's remote code targets the 4.x API.
- The support window follows the scientific-Python ecosystem schedule
  ([SPEC 0](https://scientific-python.org/specs/spec-0000/)): Python and numpy
  versions are dropped on that calendar, in a minor release, noted in the
  release notes.

### Concurrency

- All compute functions are pure and re-entrant — safe to call concurrently
  from multiple threads or processes on your own arrays.
- **Embedder instances (MERT/CLAP) are not thread-safe.** Create one per
  worker, or serialize calls to a shared instance.
- The on-disk embedding cache is safe to share across processes: writes are
  atomic (temp file + rename), partial or corrupt entries are treated as cache
  misses and recomputed, and several processes may fill the same cache
  directory concurrently.

## License

The code is licensed under the **PolyForm Noncommercial License 1.0.0**: free to
use, modify and share for any noncommercial purpose (personal projects, research,
teaching, nonprofits). Commercial use is not granted by the base license, with one
exception written into the [LICENSE](LICENSE) file — Aymeric Pasco and any product
or service he creates or distributes (including `moodengine-desktop`) may use it
commercially. Need a commercial license? Open an issue to get in touch.

Model weights carry their own, separate licenses — see
[Model licensing](#model-licensing--network-access).
