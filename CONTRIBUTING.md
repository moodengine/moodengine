# Contributing to moodengine

> **License of contributions.** By contributing to this project, you agree that your
> contribution is distributed under the terms of the [LICENSE](LICENSE) file, including
> the commercial-use exception granted to Aymeric Pasco therein.
>
> **Sign-off required (DCO).** Every commit must be signed off with `git commit -s`. See
> [Developer Certificate of Origin](#developer-certificate-of-origin-dco) below — CI blocks
> any PR that contains an unsigned commit.

## Setup

```bash
git clone https://github.com/moodengine/moodengine && cd moodengine
uv sync              # light install + dev tooling (pytest, ruff) — no torch
uv run pytest        # default suite, must pass on this light install
uv run ruff format . && uv run ruff check .
```

Optional stacks: `uv sync --extra models` (torch backbones, several GB),
`--extra ot`, `--extra cluster-graph`. Real-model tests: `uv run pytest -m model`.

## Ground rules

- **The default test suite is torch-free.** Anything importing torch or an
  optional backend guards itself (`pytest.importorskip`) or carries the `model`
  marker. CI enforces this on a light install across all supported OSes and
  Python versions.
- **Pure core.** Compute functions never write to disk; I/O is opt-in and lives
  at the pipeline boundary. `Config` is a frozen dataclass — derive variants
  with `dataclasses.replace`.
- **Tests mirror the package** (`src/moodengine/cluster.py` → `tests/unit/test_cluster.py`),
  AAA structure, fluent assertions, shared fakes and audio synthesis in
  `tests/conftest.py`.
- **Dependencies**: cross-platform wheels required (macOS arm64 / Windows /
  Linux); anything heavy or niche goes in an optional extra; no upper bounds
  without a documented, reproduced breakage.

## Commit messages — Conventional Commits

Every commit message follows [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): summary` with types `feat`, `fix`, `perf`, `refactor`, `docs`,
`test`, `chore`, `ci`, `build`, `style`. A `!` after the type marks a breaking
change. CI checks every PR commit, and release notes are generated from them
(release-please), so the message you write is the changelog entry users read.

## Versioning & releases — SemVer

Releases follow [Semantic Versioning](https://semver.org/). While the project
is on `0.x`:

- **minor** (`0.X.0`) may contain breaking API changes — always called out in
  the release notes;
- **patch** (`0.x.Y`) never breaks anything.

From `1.0.0` on, the public API (`moodengine.__all__` + documented signatures)
only breaks on a major bump. During `0.x`, a breaking change (`feat!`/`fix!`)
bumps the minor; every other `feat`/`fix` bumps the patch.

Release flow (automated by **release-please**): you don't bump the version or tag
by hand. As conventional commits land on `main`, release-please keeps a "release
PR" open that bumps `__version__` (single source of truth in
`src/moodengine/__init__.py`) and updates `CHANGELOG.md`. Merging that PR cuts the
`vX.Y.Z` tag and publishes the GitHub release; a follow-up job then re-runs the
suite, builds sdist + wheel, and attaches them to the release.

## Developer Certificate of Origin (DCO)

This project uses the [Developer Certificate of Origin](DCO) instead of a CLA. It is a
lightweight statement that you wrote — or otherwise have the right to submit — the code
you contribute, made by adding a `Signed-off-by` line to each commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

`git commit -s` adds it for you automatically (configure `user.name` / `user.email` once).
A CI check verifies every commit in a pull request carries a sign-off matching its author;
unsigned commits block the merge. To fix them:

```bash
git commit --amend -s        # the latest commit
git rebase --signoff main    # every commit on your branch
```

## Pull requests

`main` is protected: all changes land through pull requests. A PR must have a passing CI
run (all jobs), at least one approving review, resolved conversations, and a linear history
(squash merge). Keep commit messages conventional — the squash-merge title becomes the
changelog entry.
