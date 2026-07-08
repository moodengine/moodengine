---
name: new-module
description: Scaffold a new moodengine engine module plus its mirrored test file, wired into the public API following all repo standards. Use when adding a new algorithm/capability to the core.
argument-hint: "[module-name] [one-line purpose]"
---

Scaffold a new engine module named `$0` (purpose: $ARGUMENTS). Follow every step — the value of this skill is that nothing is forgotten.

## 1. Module file

Create `src/moodengine/$0.py`:
- Module docstring: what the module computes, for whom, and the core idea in plain language (2–6 sentences, self-contained — no internal doc citations, English).
- `from __future__ import annotations` first.
- Only light imports at top level (numpy, stdlib). torch or optional backends: lazy import inside functions, guarded with an actionable `ImportError` naming the extra.
- Pure functions + frozen dataclasses for structured results. Full annotations: `config: Config` where applicable, `Literal` aliases for string options (declared in `src/moodengine/_typing.py` + exported from the root, runtime check via `get_args`), `TypedDict` in `_typing.py` for dict-shaped returns, `NDArray[np.float32]` where dtype is guaranteed.
- Docstrings state the contract: shapes, dtypes, degenerate cases (empty, n<2, NaN), and complexity/memory if O(n²) or worse.
- `logging.getLogger(__name__)` if the module makes silent decisions (fallbacks, clamps, skips).

## 2. Mirrored test file

Create `tests/unit/test_$0.py`:
- AAA blocks separated by blank lines; names `test_<unit>_<condition>_<expected>`.
- assertpy `assert_that` + `pytest.raises(..., match=...)`; explicit float tolerances.
- Test mathematical properties (bounds, invariances, analytic results on tiny inputs) and every degenerate case documented in the docstrings — not just shapes.
- Seeded `np.random.default_rng` factories; reuse fakes from `tests/conftest.py`, never duplicate them.
- If the module has an optional backend: `pytest.importorskip` guard so the light suite stays green.

## 3. Wiring

- Export the public symbols: add to `src/moodengine/__init__.py` imports **and** `__all__` (keep both in sync — there is/will be a contract test asserting it).
- If new config knobs are needed: add typed fields with defaults to `Config` (grouped with a comment stating the *why* of each default), and validation in `__post_init__` if bounded.
- New third-party dependency? Stop and check `.claude/rules/dependencies.md` first (wheels-only, extras for anything heavy, no upper bounds without documented breakage).
- No separate API doc page to create: the module docstring + function docstrings ARE the API reference. `docs/` only holds standalone guides (e.g. benchmarking) — add one there only if the module needs a developer walkthrough beyond its docstrings.

## 4. Verify

Run `uv run pytest tests/unit/test_$0.py -q`, then the full default suite, then `uvx ruff format . && uvx ruff check .`. Report results honestly — red output is a finding, not a detail to omit.
