---
name: new-module
description: Scaffolds a new moodengine engine module and its mirrored test file, wired into the public API and following every repo standard. Use when adding a new algorithm or capability to the core.
argument-hint: '<module> "<one-line purpose>"'
arguments: [module, purpose]
---

Scaffold a new engine module named `$module` (purpose: $purpose). Follow every step — the
value of this skill is that nothing gets forgotten.

## 1. Module file — `src/moodengine/$module.py`

- Module docstring: what it computes, for whom, and the core idea in plain language
  (2–6 self-contained sentences — no citations to unshareable docs, English).
- `from __future__ import annotations` first.
- **Light imports only at top level** (numpy, stdlib). torch or an optional backend:
  lazy import *inside* the function that needs it, guarded by an actionable `ImportError`
  naming the extra (`pip install "moodengine[...]"`).
- Pure functions + frozen dataclasses for structured results. Full annotations:
  `config: Config` where applicable; `Literal` aliases for closed string options (declared
  in `_typing.py`, exported from the root, runtime-checked via `get_args`); `TypedDict` in
  `_typing.py` for dict-shaped returns; `NDArray[np.float32]` where the dtype is guaranteed.
- Docstrings state the contract: shapes, dtypes, degenerate cases (empty, `n < 2`, NaN),
  and complexity/memory when O(n²) or worse.
- `logging.getLogger(__name__)` if the module ever makes a silent decision (fallback,
  clamp, skipped step) — never `print`.

## 2. Mirrored test file — `tests/unit/test_$module.py`

- AAA blocks separated by blank lines; names `test_<unit>_<condition>_<expected>`.
- `assert_that` (assertpy) + `pytest.raises(..., match=...)`; explicit float tolerances.
- Test mathematical properties (bounds, invariances, analytic results on tiny inputs) and
  every degenerate case the docstrings promise — not just shapes or "doesn't crash".
- Seeded `np.random.default_rng` factories; reuse fakes from `tests/conftest.py`, never
  duplicate them.
- Optional backend? Guard the import with `pytest.importorskip` so the light suite stays green.

## 3. Wiring

- Export the public symbols in **both** `src/moodengine/__init__.py` imports **and**
  `__all__` (a contract test asserts they match).
- New config knobs? Add typed fields with defaults to `Config` (a comment stating the *why*
  of each default), and validation in `__post_init__` if the domain is bounded.
- New third-party dependency? Stop and check `.claude/rules/dependencies.md` first
  (wheels-only, extras for anything heavy, no upper bounds without a documented breakage).
- No separate API doc page — the module and function docstrings *are* the reference. Add a
  `docs/` guide only if the module needs a walkthrough beyond its docstrings.

## 4. Verify

Run `uv run pytest tests/unit/test_$module.py -q`, then the full default suite, then
`uv run ruff format . && uv run ruff check .`. Report results honestly — red output is a
finding, not a detail to omit.
