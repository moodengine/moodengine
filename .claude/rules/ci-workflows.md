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

**Pin `uv` itself too, via the `UV_VERSION` workflow-level env each `setup-uv` step reads.** With no
version, setup-uv resolves "latest" by fetching a manifest from `raw.githubusercontent.com` on every
job — an unpinned network call per job, and a real source of run failures. The version must be one
the *pinned* setup-uv knows: it verifies the download against a checksum table baked into the
action, and for an unknown version it skips validation **silently** rather than failing. So
`UV_VERSION` and the setup-uv SHA move together — raising one without the other leaves the pin
unverified with nothing to say so.

**And pin every `uvx` tool.** `uvx <tool>` resolves the newest release at run time, so a CI gate can
change behaviour without a commit — the exact thing pinning actions by SHA exists to prevent. Write
`uvx <tool>==<version>` and bump it deliberately.

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
push-to-`main`.

**Suites too slow to gate a PR live in `scheduled.yml`, not behind an `if:` in `ci.yml`.**
The real-model suite (multi-GB torch download) and the hot-path benchmarks run weekly there,
plus `workflow_dispatch`. Keeping them in their own workflow makes their exclusion from
`CI success` structural rather than something a reader has to infer from a condition. A
suite that runs on no trigger at all is the failure this repo has already hit once — about
twenty torch-guarded unit tests executed nowhere until the extras job was given a CPU torch,
so a new opt-in marker needs a job that actually selects it. The benchmark job deliberately
does not assert on timings (`testing.md`: measured, never asserted); it exists so the
benchmarks stay executable and so there is a durable record to compare by hand.

**Tooling & structure.** Install with `uv sync --locked`; run tools via `uv run` / `uvx`.
Concurrency groups cancel superseded runs on CI (`cancel-in-progress: true`) but never on a
release (`false`). Weekly `pip-audit` runs against the exported lockfile and must stay
clean. See `dependencies.md` for the uv/lock, lowest-resolution, and setup-uv-cache pitfalls
that also bite in CI.

**Don't invent gates that aren't wired.** Reference the checks that exist; if a capability
isn't set up, say so rather than assuming a command or job is present.
