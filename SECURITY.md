# Security policy

## Supported versions

Only the latest release receives security fixes. While the project is on
`0.x`, upgrade to the newest version before reporting — the fix will land
there.

## Reporting a vulnerability

Please use **GitHub private vulnerability reporting**: on the repository page,
*Security → Report a vulnerability*. Do not open a public issue for anything
exploitable. You should receive an acknowledgement within a week.

## Scope notes for this library

- moodengine is a local computation library: it opens audio files you point it
  at and writes artifacts where you tell it to. It runs no server and phones
  nothing home.
- The `[models]` extra downloads model checkpoints from the Hugging Face hub,
  and MERT executes **remote modeling code** from the hub. The default model is
  pinned to a reviewed revision precisely to keep that code immutable —
  anything that weakens or bypasses this pinning is in scope and worth
  reporting.
- Dependency advisories are monitored continuously (weekly `pip-audit` over the
  committed lockfile, Renovate update PRs).
