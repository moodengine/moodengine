---
paths:
  - "tests/**/*.py"
  - "conftest.py"
---

# Testing conventions

**Layout — strict 1:1 mirror.** Every package module has exactly one test file at the mirrored path (`src/moodengine/cluster.py` → `tests/unit/test_cluster.py`). New module ⇒ new mirrored test file in the same change. No test code inside the package, ever. Acted exception: `embeddings/clap.py` and `embeddings/mert.py` are pure torch wrappers — no unit-test files; they are covered by the opt-in `-m model` integration suite. Do not "fix" this by adding mocked unit tests for them.

**Structure — AAA, visually separated.** Arrange / Act / Assert as distinct blocks separated by blank lines (comments optional when blocks are obvious). One behavior per test. Name tests `test_<unit>_<condition>_<expected>` — the name alone should state the property being verified.

**Assertions — fluent, property-level.**
- `assert_that(...)` (assertpy) for all assertions; `pytest.raises(SomeError, match="...")` for error paths — always assert on the message. The native `assert` statement is banned in `tests/` and a CI guard (`static` job) enforces it; `np.testing.assert_*` array helpers are the sanctioned exception (they are function calls, not the statement form).
- Never widen a tolerance when phrasing a check: `== pytest.approx(0.0)` maps to `is_close_to(0.0, tolerance=1e-12)` (approx's zero-tolerance), NOT a looser `1e-6`. Preserve the exact literal and comparison direction.
- Prefer mathematical properties and invariants (bounds, monotonicity, idempotence, known analytic results on tiny inputs) over shape-only or "doesn't crash" checks.
- Numeric comparisons use explicit tolerances (`is_close_to`, `np.testing.assert_allclose(..., rtol=...)`) — never exact float equality.

**Doubles — centralized, minimal.**
- Shared fakes and array factories live in `conftest.py` (nearest-directory) — never duplicate a fake across test files.
- `pytest-mock` (`mocker`) for patching; hand-written fakes for behavioral doubles (e.g. a fake embedder returning deterministic arrays). No faker/polyfactory — inputs here are numpy arrays, not business entities; seeded `np.random.default_rng` factories are the right tool.

**Isolation — the default suite runs anywhere.**
- Must pass on a light install: no torch, no optional extras. Anything importing torch is either marked `model` or guarded with `pytest.importorskip("torch")`; optional backends (`pot`, `leidenalg`, `pacmap`, `shap`) likewise.
- Deterministic: seed everything through `Config.seed` or explicit `rng`; no reliance on dict ordering, wall clock, or network.

**Economy.** `@pytest.mark.parametrize` over copy-pasted variants. Keep unit tests fast (<1 s each); anything slower gets a marker and a reason.

**Coverage — a floor, not a target.** `uv run pytest --cov` enforces `fail_under` (`[tool.coverage.report]`) on the torch-free default suite; the pure-torch wrappers (`embeddings/clap.py`, `embeddings/mert.py`) are `omit`-ted because they have no unit tests by contract. Write tests because a behavior needs pinning, never to move the number — and never lower the floor to make a change pass. Torch/POT-gated paths are covered by the extras/model CI jobs, not this measurement.

**Benchmarks — measured, never asserted.**
- Hot-path benchmarks live in `tests/benchmarks/` (pytest-benchmark), carry the `benchmark` marker and are deselected by default (`uv run pytest -m benchmark` to run).
- Never optimize a hot path without a saved baseline: `--benchmark-save=<name>` before, `--benchmark-compare` after. A perf claim in a docstring must trace to such a run.
- Chunked/blockwise rewrites of numeric code need an equivalence test against the single-block path — tolerance at float32-ULP level (`atol≈2e-6`), NOT exact equality: BLAS accumulation order changes with slab shape.
