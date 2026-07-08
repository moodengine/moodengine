---
paths:
  - ".github/workflows/**"
---

# CI/CD & GitHub Actions conventions

The pipeline is deliberately hardened and minimal-supply-chain. A new or edited workflow
keeps every convention below; `security.yml` runs `uvx zizmor .github/workflows/` and it
MUST stay clean.

**Pin every action to a full commit SHA, with the version in a trailing comment** —
`uses: actions/checkout@9c091bb...dddfe3e0 # v7.0.0`. Never a floating tag (`@v7`) or a
branch. Bump the SHA and the comment together, after checking the upstream release.

**Least-privilege tokens.**
- Every workflow declares top-level `permissions: contents: read`. Elevate per-*job*, never
  globally, and only to what that job needs (e.g. the artifact-publish job gets
  `contents: write` solely to upload release assets).
- Every `actions/checkout` sets `persist-credentials: false` — don't leave the token on disk
  for later steps.
- release-please authenticates with a scoped GitHub App token: the minted token is
  down-scoped via `permission-contents` / `permission-pull-requests` / `permission-issues`
  inputs rather than inheriting the App's full installation grant (zizmor flags the
  un-scoped form as High), and the job's own `GITHUB_TOKEN` stays at `contents: read`.

**Prefer self-contained gates over third-party actions.** The conventional-commit, DCO
sign-off, no-internal-doc-citation, and no-native-assert checks are plain `bash` steps —
nothing extra to pin, audit, or trust. Add a new gate the same way unless a maintained,
SHA-pinnable action is clearly better.

**One required status check: `CI success`.** The `ci-success` job (`if: always()`, with
`needs:` listing every real job) collapses the whole OS × Python matrix plus all gates into
a single status context, so branch protection never has to enumerate individual job names.
When you add a real job, add it to that `needs:` list. PR-only gates (`commits`, `dco`) are
gated on `github.event_name == 'pull_request'` and are tolerated as "skipped" on
push-to-`main`. The manual `test-models` job (multi-GB torch download) is
`workflow_dispatch`-only and is intentionally excluded from `ci-success`.

**Tooling & structure.** Install with `uv sync --locked`; run tools via `uv run` / `uvx`.
Concurrency groups cancel superseded runs on CI (`cancel-in-progress: true`) but never on a
release (`false`). Weekly `pip-audit` runs against the exported lockfile and must stay
clean. See `dependencies.md` for the uv/lock, lowest-resolution, and setup-uv-cache pitfalls
that also bite in CI.

**Don't invent gates that aren't wired.** Reference the checks that exist; if a capability
isn't set up, say so rather than assuming a command or job is present.
