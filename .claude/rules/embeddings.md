---
paths:
  - "src/moodengine/embeddings/**"
---

# Embedders — the torch boundary

This module is the ONE place the deep-learning stack is allowed. The rules here protect the
"importing moodengine is torch-free" invariant and the model-loading contract.

**Keep `base.py` torch-free.** It holds the `Embedder` ABC and the on-disk cache, and is
imported by the lightweight pipeline and the test suite — so it imports only numpy + stdlib.
torch / transformers / laion_clap are imported eagerly at the top of the *concrete* modules
(`mert.py`, `clap.py`) only; that is the sanctioned exception to the torch-free rule. The
package root never imports the concrete embedders — they are constructed lazily via
`get_embedder` (in `pipeline.py`), so `import moodengine` stays light.

**Implementing an embedder.** Subclass `Embedder`; set `name` (used in cache keys and
DataFrame columns) and `sample_rate` (the rate the model expects); implement
`extract(waveform, sr) -> np.ndarray`. Document the exact output shape (MERT →
`(n_layers, n_frames, hidden)`; CLAP → `(hidden,)`). `extract` may assume the caller
resampled to `self.sample_rate`, but validates it and raises on a mismatch rather than
feeding the model off-rate audio. Coerce outputs to float32 at the boundary (detach → cpu →
numpy for any tensor).

**Model loading.** Wrap any hub/torch load failure in `ModelLoadError`, naming the exact
artifact and the remediation (offline pre-download, `HF_HUB_OFFLINE=1`, or a config
override) — the raw hub/torch exception never names what failed. Log an info line before a
first-construction download (hundreds of MB to GBs) or a cold start just looks hung. Move
the model to `config.device` and put it in eval mode.

**Device.** Honor `config.device` (auto-detected CUDA > MPS > CPU) and pass it through to
the backbone — laion-clap otherwise silently falls back to CPU whenever CUDA is absent
(e.g. on Apple Silicon, where MPS is wanted). Note that MPS users set
`PYTORCH_ENABLE_MPS_FALLBACK=1` so an op MPS lacks falls back to CPU instead of crashing.

**`trust_remote_code` models are pinned.** MERT executes custom modeling code fetched from
the hub, so it loads at a reviewed, pinned revision — arbitrary hub Python is frozen to
exactly what was reviewed. Bump the pin deliberately, after reviewing the upstream diff;
expose an override (`config.mert_revision`) but default to the reviewed snapshot.

**Weights carry their own license,** separate from this package's code license (MERT-v1-95M
is CC-BY-NC-4.0 / non-commercial). State that wherever a default model is chosen; never
imply the code license covers the weights.

**No unit tests here by contract.** `clap.py` and `mert.py` are pure torch wrappers covered
only by the opt-in `-m model` suite, and are `omit`-ted from coverage. Don't add mocked unit
tests for them (see `testing.md`) — pin real behavior in the model suite instead. The cache
helpers and the ABC in `base.py` ARE torch-free and unit-tested normally.
