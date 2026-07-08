---
name: quality-gate
description: Runs moodengine's full local quality gate — format, lint, types, the torch-free test suite, public-API contract, dependency hygiene, and lockfile — and reports a pass/fail table with concrete fixes. Use before committing, before tagging a release, or whenever asked whether the repo is green.
disable-model-invocation: false
allowed-tools: Bash(uv run ruff *) Bash(uv run mypy *) Bash(uv run pytest *) Bash(uv run deptry *) Bash(uv lock *) Bash(uv run python *)
---

Run **every** gate below, even if an early one fails — the report must show the whole
picture, not stop at the first red. For each gate record: pass/fail, the exact failing
output (trimmed to the relevant lines), and the concrete fix.

## Gates

1. **Format** — `uv run ruff format --check .`
2. **Lint** — `uv run ruff check .`
3. **Types** — `uv run mypy` (config in `[tool.mypy]`; the package must stay at zero errors).
4. **Tests (light-install contract)** — `uv run pytest -q`. This suite must pass with no
   torch and no optional extras. A failure with `ModuleNotFoundError` on an optional
   backend is a **missing `pytest.importorskip` guard** — a real defect; report it as one,
   don't wave it away.
5. **Public-API contract** — `uv run python -c "import moodengine; assert all(hasattr(moodengine, s) for s in moodengine.__all__)"` must succeed (every `__all__` symbol is importable).
6. **Dependency hygiene** — `uv run deptry .` (config in `[tool.deptry]`; per-rule ignores
   are documented there — extend them only with a comment stating why).
7. **Lockfile** — `uv lock --check` (lockfile in sync with `pyproject.toml`).

## Report

One table: `gate | status | one-line detail`. Below it, the prioritized fix list, worst
first, each anchored at `file:line`. End with a one-line verdict: **green /
green-with-warnings / red**. Never soften a red — if any gate fails, the verdict is red and
the first fix is the blocker.
