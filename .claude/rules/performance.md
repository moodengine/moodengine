---
paths:
  - "src/**/*.py"
---

# Performance & optimization — hot paths sacred, cold paths simple

Loads alongside `code-style.md` when you edit `src/`. Optimize where it counts, keep the
rest obvious, and treat every speedup claim as something to measure — never assert by feel.

**Hot vs cold.** Hot = called per-track, per-pair, or per-segment (search, pooling,
labeling math, novelty). Cold = one-shot orchestration, viz, I/O. Make hot paths fast;
keep cold paths readable — don't trade clarity for speed the caller will never feel.

**Never re-derive what the caller can pass.**
- L2-normalize once. A function needing unit-norm rows documents "rows must be
  L2-normalized" and trusts it, instead of renormalizing on every call. Normalization has
  ONE implementation — `moodengine._math.l2_normalize` (re-exported by `pooling` /
  `labeling`); never write a local copy.
- Never recompute a similarity matrix the caller already holds — take it as an argument.

**Vectorize.**
- No Python loop over `n_tracks` where a numpy/BLAS expression exists (a loop over
  `n_clusters` is fine — that dimension is small). Push the work into numpy.
- Top-k is `np.argpartition` (+ a local sort of the k winners), never a full `argsort` of
  all n.
- No pandas on a hot path — DataFrames are presentation, built once at the pipeline boundary.

**Mind memory, not just time.**
- Don't allocate a dense `(n, n)` matrix when top-k or a blockwise pass serves. When a
  function is unavoidably O(n²) in time or memory, state the envelope in its docstring so a
  caller sizing an input isn't surprised.
- Keep hot-path arrays float32 and contiguous; avoid needless copies (`np.asarray`,
  `astype(..., copy=False)` where safe). Watch silent float64 upcasts (numpy reductions,
  sklearn/scipy round-trips) — the dtype contract itself lives in `code-style.md`.

**Measure — never optimize on a hunch.**
- Hot-path benchmarks live in `tests/benchmarks/` (pytest-benchmark, `-m benchmark`,
  deselected by default). Save a baseline before optimizing (`--benchmark-save`), compare
  after (`--benchmark-compare`); a docstring speedup claim must trace to such a run.
- A chunked/blockwise rewrite of numeric code needs an equivalence test against the
  single-block path at float32-ULP tolerance (`atol ≈ 2e-6`), NOT exact equality: BLAS
  accumulation order changes with slab shape (see `testing.md`).
