---
name: quality-gate
description: Run the full local quality gate (format, lint, types, tests, dependency hygiene) and report a pass/fail table with fixes. Use before committing, tagging a release, or when asked whether the repo is green.
disable-model-invocation: false
---

Run every gate below **even if an early one fails** — the report must show the whole picture. For each gate record: pass/fail, the exact failing output (trimmed), and the concrete fix.

## Gates

1. **Format** — `uv run ruff format --check .` (ruff is in the PEP 735 `dev` group).
2. **Lint** — `uv run ruff check .`.
3. **Types** — `uv run mypy` (configured in pyproject `[tool.mypy]`, default mode; the package must stay at zero errors).
4. **Tests (light install contract)** — `uv run pytest -q`. This suite must pass without torch; if a test fails with `ModuleNotFoundError` on an optional dep, that's a missing `importorskip` guard — a real defect, report it as such.
5. **Public API contract** — verify `python -c "import moodengine; assert all(hasattr(moodengine, s) for s in moodengine.__all__)"` succeeds (every `__all__` symbol importable).
6. **Dependency hygiene** — `uv run deptry .` (configured in pyproject `[tool.deptry]`; per-rule ignores are documented there — extend them only with a comment stating why).
7. **Lockfile** — `uv lock --check` (lockfile up to date with pyproject), if `uv.lock` exists.

## Report

A single table: gate | status | one-line detail. Below it, the prioritized fix list (worst first, `file:line`). End with a one-line verdict: **green / green-with-warnings / red**. Never soften a red: if a gate fails, the verdict is red and the first fix is the blocker.
