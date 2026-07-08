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
- Closed string vocabularies are `Literal` aliases, backed by an explicit runtime check that raises with the received value and the valid options. The alias lives in `moodengine/_typing.py`, is exported from the package root, and the runtime check derives its tuple via `typing.get_args(...)` so the two can never drift (model: `ClusterMethod`, `Config.__post_init__`).
- Structural interfaces are `typing.Protocol` (`SupportsEmbedText`, `Reducer2D` in `_typing.py` — `@runtime_checkable`), not implicit duck typing.
- Dict-shaped returns get a `TypedDict` in `_typing.py` + root export (`ClusteringResult`, `StabilityMetrics`…); multi-value returns a `NamedTuple` or frozen dataclass — never anonymous tuples/dicts on public APIs. `tests/unit/test__typing.py` holds the drift guards (alias ↔ runtime vocabulary, TypedDict ↔ returned keys) — extend it with every addition.
- Arrays are `numpy.typing.NDArray[np.float32]` where dtype is guaranteed. The `py.typed` marker must remain in the package.

**float32 discipline.** The engine promises float32 end-to-end. Watch silent float64 upcasts (`np.mean`/`np.sum` reductions, scipy/sklearn round-trips) — cast back at function boundaries and say so in the docstring.

**Performance — hot paths are sacred, cold paths are simple.**
- Hot = called per-track, per-pair, or per-segment (search, pooling, labeling math, novelty). Cold = one-shot orchestration, viz, I/O. Optimize hot, keep cold obvious.
- Never re-derive what the caller can pass: no repeated L2 normalization of the same matrix (normalize once, document the expectation "rows must be L2-normalized"), no recomputed similarity matrices. L2 normalization itself has ONE implementation: `moodengine._math.l2_normalize` (re-exported by `pooling`/`labeling`) — never write a local copy.
- Top-k uses `np.argpartition` (+ local sort of k items), not full `argsort`.
- No Python loops over `n_tracks` when a vectorized form exists (loops over `n_clusters` are fine). No pandas in hot paths — DataFrames are for presentation at the pipeline boundary.
- Don't allocate dense `(n, n)` matrices when top-k or blockwise computation serves; state the memory envelope in the docstring when a function is O(n²).

**Errors, logging, warnings.**
- Raise from `moodengine.exceptions` (`MoodengineError` root; `AudioDecodeError`, `MissingDependencyError`, `ModelLoadError`) for failures a caller reacts to; keep plain `ValueError`/`TypeError` for argument errors (numpy/sklearn convention). Messages state what was received *and* what to do. Never bare `except Exception` in the core.
- Input matrices at public entry points go through `moodengine._validation.ensure_finite_2d` (NaN/Inf named at the boundary, not three stacks deep). Degenerate SIZES follow each function's documented contract (zeros/empty, never raise); non-finite DATA raises.
- `Config` validates in `__post_init__` — new fields with a silent-failure mode get a check there in the same change.
- Silent behavior switches (fallbacks, clamps, skipped steps) must at least `logger.info`/`warning` — module-level `logging.getLogger(__name__)`, never `print`.
- Deprecations: keep the old symbol working one minor version with `DeprecationWarning`, then remove.
