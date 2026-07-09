---
paths:
  - "pyproject.toml"
  - "uv.lock"
  - ".python-version"
  - ".github/**/*"
---

# Dependency & packaging policy

**Adding a runtime dependency is a big deal** for a library — it lands in every consumer's tree. Checklist before adding one:
- Ships cross-platform wheels (macOS arm64 / Windows / Linux) for all supported CPythons (cp311+). No source-build-only deps — hnswlib was rejected on this bar.
- Can't be an optional extra instead? Anything heavy, niche, or backend-like goes in `[project.optional-dependencies]` with a lazy, guarded import (see below).
- Justify it in the commit/PR message: what it does, why stdlib/numpy can't.

**Version bounds.**
- Lower bound = the oldest version actually tested (CI job with `uv run --resolution lowest-direct` keeps us honest).
- **No upper bounds** unless a breakage is known and reproduced; every cap carries an inline comment stating the exact reason and what unlocks removal — and is lifted once that condition clears (as the former `transformers<5` cap was, once MERT and laion-clap were verified on the transformers 5.x line). The project currently ships no upper caps.
- Support window follows the scientific-Python ecosystem (SPEC 0): drop Python/numpy versions on schedule, in a minor release, noted in the changelog.

**Tooling layout.**
- Dev tooling (pytest, ruff, mypy, deptry, pre-commit…) lives in `[dependency-groups]` (PEP 735) — never in `project.dependencies`, never in a published extra.
- `uv.lock` is committed (it pins dev/CI only — never constrains consumers). Refresh it as a deliberate, reviewed change, not as a side effect.
- `.python-version` pins the dev interpreter.

**Optional imports** are guarded at use-site with `MissingDependencyError` (from
`moodengine.exceptions`), which names the feature, the missing distribution, and the pip
extra so the message spells out the exact install command:
`raise MissingDependencyError("leiden clustering", "leidenalg + python-igraph", "cluster-graph") from exc`
Every optional backend follows this exact pattern — one style, no silent fallbacks between
backends, and never a bare `ImportError` (see the error taxonomy in `code-style.md`).

**Never** `import` a package that isn't declared (scripts included). Declared-but-unused and used-but-undeclared are both defects; `uv run deptry .` is the wired guard (CI `static` job). Intentional exceptions live in `[tool.deptry.per_rule_ignores]` with a comment stating why — extend them only that way.

**uv pitfalls (all bitten in practice).**
- `uv lock` **preserves** already-locked versions; upgrading requires `uv lock --upgrade-package <name>` (or `--upgrade`).
- Testing lowest bounds: `rm uv.lock && uv lock --resolution lowest-direct`, then **`--frozen` on every subsequent sync/run** — a plain `uv sync` ignores a lowest lock and silently re-locks at highest, un-testing the minimums.
- `astral-sh/setup-uv` v8 enables caching by default — set `enable-cache: false` in any workflow that publishes artifacts (cache poisoning).
- Raising a floor? It must be *provable*: the CI `test-lowest` job runs the whole suite at the declared minimums (this is what exposed hdbscan 0.8.33 / umap 0.5.0 / librosa 0.10.0 / scipy < 1.15 as dishonest floors).
