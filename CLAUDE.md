# CLAUDE.md — moodengine (core engine)

Guidance for AI agents working in this repo. **Read this first** — it encodes the invariants every change must respect. Path-scoped conventions live in `.claude/rules/`; the maintainer keeps an improvement roadmap and engineering standards in `plans/`, which is git-ignored and not shipped with the repo.

## What this is

Pure, **stateless** music-mood computation library (MIR): audio → embeddings (MERT/CLAP, optional `[models]` extra) → pooling → clustering (UMAP/HDBSCAN/KMeans/Leiden) → zero-shot mood labeling → search / evaluation / calibration / viz. Two consumers, equal rank: the sibling app `moodengine-desktop` (git dependency) and **any external OSS adopter** — every API decision must hold for a stranger with no access to our plans or history.

## Architecture — functional core, imperative shell (non-negotiable)

- **No hexagonal/clean architecture here.** That layer lives in `moodengine-desktop` (`app/ports` + `app/adapters`). The core *is* the domain: pure functions + small frozen dataclasses, sklearn-style. No Pydantic, no stamina/pybreaker, no OOP hierarchies (decided after a full architecture audit; rationale and explicit vetoes are recorded in the maintainer's internal notes).
- **Importing the package is torch-free.** torch/transformers/laion_clap load lazily inside the embedders only. Never import them at module top level; the default test suite must pass on a light install.
- **Compute functions never write to disk.** I/O is opt-in (the `out_html=None` pattern in `viz.py`) or lives in the pipeline shell and scripts. Never anchor paths on `__file__`.
- **`Config` is a frozen dataclass** — derive variants with `dataclasses.replace(...)`, never mutate, never read a global config.
- Modules are flat and independent; only `pipeline.py` orchestrates across modules. Keep the dependency graph flat.

## Commands

- `uv sync` — light install (no torch; dev tooling comes from the PEP 735 `dev` group); `--extra models` pulls torch (~GBs, only to embed real audio).
- `uv run pytest` — default suite (torch-free by contract). Real-model tests: `-m model` (opt-in). Coverage floor: `uv run pytest --cov` (`fail_under` in `[tool.coverage.report]`; pure-torch wrappers omitted).
- `uv run ruff format . && uv run ruff check .` — format + lint (ruff lives in the PEP 735 `dev` group).
- `uv run mypy` — types, default mode, must stay at zero errors; `uv run deptry .` — dependency hygiene (documented ignores in pyproject).
- CI/CD: `ci.yml` (tests 3 OS × py3.11–3.14, `static` job for format/lint/mypy/deptry, coverage floor, lowest-direct, extras, conventional-commit + DCO sign-off gates, aggregated into one required `CI success` check), `security.yml` (weekly pip-audit + zizmor), `release-please.yml` (conventional commits → a release PR that bumps the version + `CHANGELOG.md`; merging it cuts the tag + GitHub release, then builds & attaches sdist/wheel). Prefer commands that exist; don't invent gates that aren't wired yet.

## Engineering standards (details in `.claude/rules/`)

- **Comments/docstrings**: self-contained, explain the *why*; **never cite internal docs** (R&D specs, `plans/`) — an external dev has none of them. English only.
- **No volatile stats in CLAUDE.md or README.md** (test/file/symbol counts, durations) — they drift; describe contracts instead.
- **Tests**: `tests/` mirrors the package layout 1:1; AAA; fluent assertions (assertpy); shared fakes in conftest.
- **Dependencies**: state-of-the-art hygiene — minimal runtime deps, wheels-only, justified bounds, PEP 735 groups, committed `uv.lock`.
- **Code**: airy, fully typed (`Literal`/`Protocol`/`TypedDict`/`NDArray`), float32 discipline, performance-conscious on hot paths.
- **Docs**: `docs/` holds plain, standalone `.md` guides (e.g. `docs/benchmarking.md`) — no static-site generator, no build step, GitHub renders them directly. The API reference is the source docstrings; don't hand-write a duplicate per-module doc page.

## Two-repo workflow (desktop lives downstream)

- Algorithms are implemented + unit-tested **here**, then the desktop wraps them in thin adapters. If the desktop must re-implement or contort around something (progress callbacks, re-normalization, error taxonomy…), that is an API defect of the core — fix it here.
- The desktop repo path is machine-dependent — never hardcode it in code or docs; use relative sibling references (the desktop repo sits alongside this one).
- Release flow: tag here (`vX.Y.Z`) → desktop bumps `rev` in its `[tool.uv.sources]` → `uv sync` → desktop tests.
- New dependency bar (both repos): must ship cross-platform wheels (macOS arm64 / Windows / Linux, cp311+).

## Working on the roadmap

The `plans/` directory is **internal and git-ignored** — it lives only on the maintainer's machine and is not part of the published repo. When present, it holds:

- `plans/production-readiness-audit.md` — prioritized P0→P3 backlog with file:line evidence, plus explicit **vetoes** (things we decided NOT to do). Reconcile against current code before executing any item; never redo what exists.
- `plans/engineering-standards.md` — the normative standards this repo is converging to.

If `plans/` is absent (e.g. a fresh clone), treat this file and `.claude/rules/` as the authoritative guidance.
