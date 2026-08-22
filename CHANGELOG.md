# Changelog

## [0.4.0](https://github.com/moodengine/moodengine/compare/v0.3.1...v0.4.0) (2026-08-22)


### ⚠ BREAKING CHANGES

* **ml:** `ProbeHead` gained a required `cv_score` field, and `fit_linear_probe`'s `C` now defaults to None (sweep by cross-validation) instead of 1.0. Pass `C=1.0` for the previous behaviour. Probe states saved before this change still load, with `cv_score` all-nan.

### Features

* **bench:** measure the text -&gt; playlist path, which had no measurement at all ([#41](https://github.com/moodengine/moodengine/issues/41)) ([551bc03](https://github.com/moodengine/moodengine/commit/551bc033ffb5dbffb95ea4ea32bfdb686c060a09))
* **cluster:** expose the geometry choices the pipeline was making silently ([#38](https://github.com/moodengine/moodengine/issues/38)) ([47e4141](https://github.com/moodengine/moodengine/commit/47e41414cff58b7eb0fdef52371c2e5b95a35534))
* **cluster:** surface the HDBSCAN diagnostics the fit already computed ([#34](https://github.com/moodengine/moodengine/issues/34)) ([f4393e2](https://github.com/moodengine/moodengine/commit/f4393e2fb038d1d07e2bf5b479629d44f89cff0b))
* **embeddings:** add MuQ-MuLan as a second audio-text backbone ([#36](https://github.com/moodengine/moodengine/issues/36)) ([3d142bb](https://github.com/moodengine/moodengine/commit/3d142bb2b4eedad56d6d37766a1e6d6f3eb460fd))
* **journey:** make both morph modes return playlists, and add the missing ordering objective ([#42](https://github.com/moodengine/moodengine/issues/42)) ([d7adfc9](https://github.com/moodengine/moodengine/commit/d7adfc902fe690bb80cf1b1c463bcafab9403ca8))
* **labeling:** ground the mood vocabulary in affect, and check the two label streams agree ([#40](https://github.com/moodengine/moodengine/issues/40)) ([2691e5f](https://github.com/moodengine/moodengine/commit/2691e5f77e4590de8d3be4763244318c22ee705b))
* **labeling:** measure label-direction redundancy, and keep the prompts it vindicates ([#35](https://github.com/moodengine/moodengine/issues/35)) ([432fefe](https://github.com/moodengine/moodengine/commit/432fefee15b11510974726cef4128fa84879f331))


### Fixes

* **bench:** make the benchmark able to answer "did that help?" ([#39](https://github.com/moodengine/moodengine/issues/39)) ([e2178b2](https://github.com/moodengine/moodengine/commit/e2178b235b23e879556f0bbe8dd38cd6eb74e406))
* **bench:** take the number of bootstrap draws the constant advertises ([#47](https://github.com/moodengine/moodengine/issues/47)) ([776b1bd](https://github.com/moodengine/moodengine/commit/776b1bd231c95ce40aa0e74f919e8a21a356c5c9))
* **cluster:** publish the structure verdict the engine already computes ([#43](https://github.com/moodengine/moodengine/issues/43)) ([70eece7](https://github.com/moodengine/moodengine/commit/70eece7333f58ce2fd24692c075784a9db04ced7))
* **config:** refuse the settings that silently produce unusable vectors ([#33](https://github.com/moodengine/moodengine/issues/33)) ([ea9898d](https://github.com/moodengine/moodengine/commit/ea9898da524f98f14bf1b33730082850131f0b31))
* **embeddings:** restore the guards the lazy and batched paths bypassed ([#44](https://github.com/moodengine/moodengine/issues/44)) ([671af30](https://github.com/moodengine/moodengine/commit/671af30c1ff90fa114fb00d2a4d3e5db4d4a27bf))
* **evaluation:** read the gold JSON as UTF-8 so a non-ASCII filename is not silently dropped ([093c372](https://github.com/moodengine/moodengine/commit/093c372747a540ef8e8aac7755e99bd41e6bd2dc))
* **journey:** make smooth_order's start parameter honest at both ends ([#46](https://github.com/moodengine/moodengine/issues/46)) ([d884974](https://github.com/moodengine/moodengine/commit/d884974be94622ba7c984bd2c8dadc9ca256bb76))
* **labeling:** extend the fixed prior to the energy and valence axes ([#45](https://github.com/moodengine/moodengine/issues/45)) ([310f528](https://github.com/moodengine/moodengine/commit/310f5283bd985a0c2502a95265eace4038536998))
* **labeling:** guard the two-pole contract, and correct two measured notes ([#52](https://github.com/moodengine/moodengine/issues/52)) ([a09b41a](https://github.com/moodengine/moodengine/commit/a09b41aad9a79b71fbc4021050704ba95460e015))
* **labeling:** refuse to profile a single cluster instead of ranking rounding noise ([#56](https://github.com/moodengine/moodengine/issues/56)) ([7707219](https://github.com/moodengine/moodengine/commit/77072199e885108391148e6679de23ef38c9405b))
* **ml:** correct seven measurements that reported the wrong quantity ([#31](https://github.com/moodengine/moodengine/issues/31)) ([b7d1037](https://github.com/moodengine/moodengine/commit/b7d103774bb9d317a3e5ef0fdf0711e392bbb3a6))
* **pipeline:** correct three contracts the last series left behind it ([#50](https://github.com/moodengine/moodengine/issues/50)) ([ab874f4](https://github.com/moodengine/moodengine/commit/ab874f4363a84943df6ce3e49c51259211c7b933))


### Performance

* **calibration:** compute the APS score as a masked row sum instead of a per-row argsort ([ba4a3d5](https://github.com/moodengine/moodengine/commit/ba4a3d5562d26cf477a9fc4c27d26081274dba88))
* **calibration:** evaluate the temperature objective in logsumexp form and bin the diagram with bincount ([1ad8502](https://github.com/moodengine/moodengine/commit/1ad8502579996163a8c82ca6cc8c140fff4f7585))
* **cluster:** rebuild spherical centroids with one BLAS call and probe degeneracy without a gram ([8c6fd67](https://github.com/moodengine/moodengine/commit/8c6fd67b094a779e6779e33d319c06ee04525b85))
* **cluster:** reuse the k-selection fit and compute the sub-cluster silhouette once ([ac00bbd](https://github.com/moodengine/moodengine/commit/ac00bbd58c958d95dd2239361a0930dc47c3f9fb))
* cut the work three hot paths were doing and never using ([#51](https://github.com/moodengine/moodengine/issues/51)) ([98ec50d](https://github.com/moodengine/moodengine/commit/98ec50d4adc4b238baf7e3f9ca4df4ab27381cd8))
* **explain:** batch the Shapley coalition payoffs into one predict_proba ([ff63792](https://github.com/moodengine/moodengine/commit/ff6379285dea166d56b68b67cbb33bfff654ee18))
* **explain:** score the counterfactual candidate grid in one predict_proba ([a3f9223](https://github.com/moodengine/moodengine/commit/a3f9223b7fb99e13330b71eb988ea7873f5717c5))
* **journey:** vectorize the free-start seed and the 2-opt sweep ([feedaa4](https://github.com/moodengine/moodengine/commit/feedaa4f37fd4e17add6a2ce48100ad3291db6d8))
* **labeling:** build the affect keep-mask with vectorized finiteness checks ([cca7c10](https://github.com/moodengine/moodengine/commit/cca7c1067e80869062c1f9c20df16a03504dc938))
* **novelty:** prefilter candidate columns by chunk max before the top-k select ([7378c74](https://github.com/moodengine/moodengine/commit/7378c74c0e0a6f96e3138eb7300c776ecfa7d1cd))
* **pipeline:** stop loading weights a cached run never needs, and batch what is left ([#37](https://github.com/moodengine/moodengine/issues/37)) ([39e7e1e](https://github.com/moodengine/moodengine/commit/39e7e1ee8657b9641fa798bc06a853afbe438fe6))
* **search:** hoist the harmonic and tempo bonus out of the greedy loop ([f66f86d](https://github.com/moodengine/moodengine/commit/f66f86d38120f368c1fcddfc45053d25a6117fad))
* **viz:** build the dashboard table and hover text column-wise instead of per row ([271ca9d](https://github.com/moodengine/moodengine/commit/271ca9d3012e8a1d5d31ef6f6d14a74352b3d2c3))

## [0.3.1](https://github.com/moodengine/moodengine/compare/v0.3.0...v0.3.1) (2026-08-17)


### Fixes

* **errors:** install hints now resolve (GitHub-only distribution) ([#26](https://github.com/moodengine/moodengine/issues/26)) ([2f93ce5](https://github.com/moodengine/moodengine/commit/2f93ce5fd8dc4f36a2d0a34bbd3eff32bdd5f495))

## [0.3.0](https://github.com/moodengine/moodengine/compare/v0.2.4...v0.3.0) (2026-08-17)


### ⚠ BREAKING CHANGES

* **deps:** clear all 20 open Dependabot advisories ([#20](https://github.com/moodengine/moodengine/issues/20))

### Fixes

* **deps:** clear all 20 open Dependabot advisories ([#20](https://github.com/moodengine/moodengine/issues/20)) ([df196ab](https://github.com/moodengine/moodengine/commit/df196abe88b7fb1ace0ec77738992b8d7edaacc8))

## [0.2.4](https://github.com/moodengine/moodengine/compare/v0.2.3...v0.2.4) (2026-07-16)


### Features

* **evaluation:** promote average_precision to the public API ([#18](https://github.com/moodengine/moodengine/issues/18)) ([7b16582](https://github.com/moodengine/moodengine/commit/7b165822c59357691d5e2a5d2d577ea8d331479a))

## [0.2.3](https://github.com/moodengine/moodengine/compare/v0.2.2...v0.2.3) (2026-07-12)


### Fixes

* **cluster:** coverage_entropy 0.0 for a single region; guard NaN cophenetic ([#16](https://github.com/moodengine/moodengine/issues/16)) ([6dfab98](https://github.com/moodengine/moodengine/commit/6dfab9819c6279b037c79f0abab2a78da6b04b19))

## [0.2.2](https://github.com/moodengine/moodengine/compare/v0.2.1...v0.2.2) (2026-07-11)


### Features

* **io:** add recursive flag to discover_audio_files ([#14](https://github.com/moodengine/moodengine/issues/14)) ([57f7671](https://github.com/moodengine/moodengine/commit/57f7671075ef2e92db05823a95fa5a47c60fd8ef))

## [0.2.1](https://github.com/moodengine/moodengine/compare/v0.2.0...v0.2.1) (2026-07-10)


### Fixes

* **embeddings:** keep laion_clap single-prompt tokenizer output 2-D for transformers 5 ([#12](https://github.com/moodengine/moodengine/issues/12)) ([c850b72](https://github.com/moodengine/moodengine/commit/c850b72b5560adf6c76754450a5c0016b579e58b))

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
