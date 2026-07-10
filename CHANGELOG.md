# Changelog

## [0.2.0](https://github.com/moodengine/moodengine/compare/v0.1.2...v0.2.0) (2026-07-10)


### ⚠ BREAKING CHANGES

* **deps:** the [models] extra now requires transformers >= 5.3.0 and torch >= 2.4 (previously transformers 4.x / torch 2.1). Consumers installing moodengine[models] against a pinned transformers 4.x or torch < 2.4 must upgrade those. The default install is unaffected.

### Fixes

* **deps:** require transformers &gt;= 5.3.0 for the models extra (CVE-2026-4372, CVE-2026-1839) ([#10](https://github.com/moodengine/moodengine/issues/10)) ([5f0a89f](https://github.com/moodengine/moodengine/commit/5f0a89f93625e69f4cffecc294b22f38bfd1ab8c))

## [0.1.2](https://github.com/moodengine/moodengine/compare/v0.1.1...v0.1.2) (2026-07-08)


### Fixes

* **embeddings:** raise MissingDependencyError when the models extra is absent ([#4](https://github.com/moodengine/moodengine/issues/4)) ([a0be9da](https://github.com/moodengine/moodengine/commit/a0be9daff2343c8c954cc12a49a00c2aa89f15b1))

## [0.1.1](https://github.com/moodengine/moodengine/compare/moodengine-v0.1.0...moodengine-v0.1.1) (2026-07-08)


### Fixes

* **embeddings:** raise MissingDependencyError when the models extra is absent ([#4](https://github.com/moodengine/moodengine/issues/4)) ([a0be9da](https://github.com/moodengine/moodengine/commit/a0be9daff2343c8c954cc12a49a00c2aa89f15b1))
