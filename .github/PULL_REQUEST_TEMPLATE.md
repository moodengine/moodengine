<!-- Thanks for contributing to moodengine! Keep the PR focused on one change. -->

## What & why

<!-- What does this change do, and why is it needed? Link any related issue: Closes #123 -->

## Checklist

- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`type(scope): summary`)
- [ ] Every commit is signed off (`git commit -s`) — the [DCO](../DCO) check must pass
- [ ] Local gates pass: `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run deptry . && uv run pytest`
- [ ] Tests added/updated for the change (mirroring `src/` layout, AAA, assertpy)
- [ ] Public API changes are reflected in docstrings and `__all__`
- [ ] Docs updated if user-facing behavior changed (`README.md` / `docs/`)

<!-- By opening this PR you agree your contribution is distributed under the LICENSE file,
     including the commercial-use exception granted to Aymeric Pasco therein. -->
