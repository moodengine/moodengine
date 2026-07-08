# CLAUDE.md — moodengine

Guidance for anyone changing this repo. It holds the cross-cutting invariants and
the commands you can't guess by reading the code. Per-area detail loads on demand
from `.claude/rules/` when you touch matching files; multi-step procedures live in
`.claude/skills/`. Run `/memory` to see what's currently loaded.

<!--
Maintainer note (stripped before this file enters context — costs no tokens):
Keep this file under ~200 lines and free of anything inferable from the code.
It loads in full every session, and length costs adherence. Put per-area
conventions in .claude/rules/ (path-scoped) and procedures in .claude/skills/.
-->

## What this is

A **pure, stateless** music-mood library (music information retrieval). The pipeline:
audio → embeddings (MERT / CLAP, optional `[models]` extra) → pooling → clustering →
zero-shot mood labeling → search / evaluation / calibration / visualization. It is a
library for external adopters — every public choice must make sense to a stranger who
has only this repo, no other context.

## Invariants — read before changing anything

These are the rules a change most often breaks. None is obvious from a single file, so
hold them in every session.

- **Importing the package is torch-free.** `import moodengine` MUST NOT pull in
  torch / transformers / laion_clap. Those load lazily *inside* the embedders
  (`get_embedder`, `embeddings/`). The default test suite runs on an install with no
  torch — a top-level heavy import breaks it everywhere.
- **The core is stateless and never writes to disk.** Compute functions are pure: they
  return data, they don't persist it. File output is opt-in (the `out_html=None`
  pattern in `viz.py`) or lives in `pipeline.py` and `scripts/`. Never anchor a path on
  `__file__`.
- **`Config` is a frozen dataclass.** Derive variants with `dataclasses.replace(...)`;
  never mutate it, never read a module-level global config. New fields with a bounded
  domain get validated in `__post_init__` in the same change.
- **float32 end to end.** Watch silent float64 upcasts (numpy reductions, sklearn/scipy
  round-trips) and cast back at function boundaries; say so in the docstring.
- **Flat module graph.** Modules are independent; only `pipeline.py` orchestrates across
  them. Don't introduce cross-module imports that turn the graph into a web.

## Architecture

Functional core, sklearn-style: pure functions + small frozen dataclasses. The only
sanctioned OOP is the `Embedder` ABC (its concrete backbones) and structural
`typing.Protocol`s. No service/manager classes, no inheritance hierarchies, no Pydantic,
no dependency-injection framework, no global state. One responsibility per function — if
one computes *and* formats *and* writes, split it.

## Commands

Everything runs through `uv`; a plain `uv sync` is the torch-free "light" install.

- `uv sync` — light install + dev tooling (PEP 735 `dev` group).
  `uv sync --extra models` adds torch (~GBs; only needed to embed real audio).
- `uv run pytest` — the default suite; torch-free by contract. `-m model` runs the
  real-model tests (needs `--extra models`); `-m benchmark` runs hot-path benchmarks.
  Both markers are deselected by default.
- `uv run pytest --cov` — same suite with the coverage floor (`fail_under` in
  `[tool.coverage.report]`).
- `uv run ruff format . && uv run ruff check .` — format, then lint.
- `uv run mypy` — types; must stay at zero errors. `uv run deptry .` — dependency hygiene.
- `/quality-gate` — a skill that runs all of the above and reports a pass/fail table.

## Repo etiquette

- **Conventional Commits** (`type(scope): summary`; `feat` `fix` `perf` `refactor`
  `docs` `test` `chore` `ci` `build` `style`; `!` marks a breaking change). The summary
  becomes the changelog entry (release notes are generated from it), and CI checks every
  commit in a PR.
- **Sign off every commit** with `git commit -s` — the DCO check blocks any unsigned
  commit on a PR.
- **`main` is protected.** Changes land via PR with green CI, an approving review,
  resolved conversations, and linear history (squash merge). Don't commit to `main`
  directly.
- **Releases are automated** — don't bump the version or tag by hand. `__version__` in
  `src/moodengine/__init__.py` is the single source of truth; the release tooling bumps
  it and the changelog from the merged commits.

## Gotchas

- **Optional backends.** The compute extras (`[ot]`, `[cluster-graph]`, `[pacmap]`,
  `[explain]`) import lazily at use-site and raise `MissingDependencyError` naming the exact
  `pip install "moodengine[...]"` when absent; their tests self-skip via
  `pytest.importorskip`, so the light suite stays green. The `[models]` backbones differ —
  torch is imported at the top of the embedder modules, so without it `get_embedder`
  surfaces a plain `ModuleNotFoundError`, not the friendly error.
- **Public API surface is a contract.** A new export goes in both the
  `src/moodengine/__init__.py` imports *and* `__all__`; keep them in sync (a test checks it).
- **No volatile stats** in `CLAUDE.md` or `README.md` (test/file counts, timings,
  durations) — they drift. Describe the contract instead.
- **English only** in code, comments, and docstrings.

## Where the rest lives

- **`.claude/rules/*.md`** — per-area conventions, auto-loaded when you touch matching
  files: code style & typing plus hot-path performance (`src/`), the embedders/torch
  boundary (`src/moodengine/embeddings/`), testing (`tests/`), comments & docstrings,
  dependency & packaging policy, and CI/workflow conventions (`.github/workflows/`).
- **`.claude/skills/`** — procedures: `/quality-gate` (run every local gate),
  `/new-module` (scaffold an engine module + its mirrored test, wired into the API).
- **`.claude/agents/standards-reviewer`** — a read-only reviewer that checks a diff
  against these standards.
- **`docs/*.md`** — standalone developer guides. The API reference *is* the source
  docstrings; don't hand-write duplicate per-module doc pages.
