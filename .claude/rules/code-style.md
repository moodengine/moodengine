---
paths:
  - "src/**/*.py"
---

# Core code style — typed, airy, fast

**Shape of the code — functional core.**
- Pure functions + small frozen dataclasses. No stateful service classes, no inheritance hierarchies; the only sanctioned OOP is the `Embedder` ABC (N real implementations) and structural `Protocol`s.
- One responsibility per function. If a function computes *and* formats *and* writes, split it.
- Airy layout: blank line between logical steps, no dense one-liner chains. Readability beats cleverness; `ruff format` is the floor, not the ceiling.

**Typing — every public signature is fully annotated.**
- `config: Config` on every function taking a config (no duck-typed `config` parameters).
- Closed string vocabularies are `Literal` aliases, backed by an explicit runtime check that raises with the received value and the valid options. The alias lives in `moodengine/_typing.py`, is exported from the package root, and the runtime check derives its tuple via `typing.get_args(...)` so the two can never drift (e.g. `ClusterMethod`, checked in `cluster.run_clustering`; the `Config`-field aliases like `pooling_mode` are checked in `Config.__post_init__`).
- Structural interfaces are `typing.Protocol` (`SupportsEmbedText`, `Reducer2D` in `_typing.py` — `@runtime_checkable`), not implicit duck typing.
- Dict-shaped returns get a `TypedDict` in `_typing.py` + root export (`ClusteringResult`, `StabilityMetrics`…); multi-value returns a `NamedTuple` or frozen dataclass — never anonymous tuples/dicts on public APIs. `tests/unit/test__typing.py` holds the drift guards (alias ↔ runtime vocabulary, TypedDict ↔ returned keys) — extend it with every addition.
- Arrays are `numpy.typing.NDArray[np.float32]` where dtype is guaranteed. The `py.typed` marker must remain in the package.

**float32 discipline.** The engine promises float32 end-to-end. Watch silent float64 upcasts (`np.mean`/`np.sum` reductions, scipy/sklearn round-trips) — cast back at function boundaries and say so in the docstring.

**Performance.** Hot-path and optimization discipline — hot vs cold paths, never
re-deriving what the caller can pass, `argpartition` over `argsort`, vectorizing over
`n_tracks`, memory envelopes, and benchmark-before-you-optimize — lives in `performance.md`,
which loads alongside this file when you edit `src/`.

**Errors, logging, warnings.**
- Raise from `moodengine.exceptions` (`MoodengineError` root; `AudioDecodeError`, `MissingDependencyError`, `ModelLoadError`) for failures a caller reacts to; keep plain `ValueError`/`TypeError` for argument errors (numpy/sklearn convention). Messages state what was received *and* what to do. Never bare `except Exception` in the core.
- Input matrices at public entry points go through `moodengine._validation.ensure_finite_2d` (NaN/Inf named at the boundary, not three stacks deep). Degenerate SIZES follow each function's documented contract (zeros/empty, never raise); non-finite DATA raises.
- `Config` validates in `__post_init__` — new fields with a silent-failure mode get a check there in the same change.
- Silent behavior switches (fallbacks, clamps, skipped steps) must at least `logger.info`/`warning` — module-level `logging.getLogger(__name__)`, never `print`.
- Deprecations: keep the old symbol working one minor version with `DeprecationWarning`, then remove.
