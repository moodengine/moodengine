# Benchmarking against ground truth

The engine's mood outputs are only trustworthy if you can *measure* them. This guide runs
the pipeline against human valence/arousal ratings so a change can be shown to improve or
regress quality, rather than asserted to. It needs the model backbones
(`uv sync --extra models`) and is a developer workflow run from a clone, not part of the
library surface.

## The dataset

[DEAM](https://cvml.unige.ch/databases/DEAM/) (the MediaEval "Emotion in Music" database,
a superset of the emoMusic 1000-songs set) provides 1802 forty-five-second excerpts with
averaged human valence and arousal ratings on a 1–9 scale. `fetch_deam.py` downloads and
extracts it (~1.35 GB), skipping anything already present:

```bash
uv run python scripts/fetch_deam.py --data-dir ~/moodengine-bench/deam
```

It prints the audio directory and the static-annotations CSV to point the runner at.

## Running the benchmark

```bash
uv run --extra models --extra muq python scripts/bench_valence_arousal.py \
    --data-dir ~/moodengine-bench/deam --mode both --embedder mert --limit 200
```

DEAM arousal maps to the engine's **energy** axis and valence to **valence**; the 1–9
ratings are affine-scaled to `[0, 1]` so the reported metrics are comparable to the
pipeline's `[0, 1]` outputs. Two modes measure different layers of the stack:

- **`zeroshot`** — the product path: zero-shot `attribute_scores` (energy, valence) correlated
  with the gold ratings. Measures the labelling / prompt / recentering stack.
  `--zeroshot-embedder` picks the text-capable backbone (`clap` or `mulan`), which is how the
  two are compared on accuracy rather than on separability proxies.
- **`probe`** — a cross-validated ridge linear probe on frozen `--embedder` embeddings
  (`mert`, `clap` or `fused`) regressed onto the gold ratings. This is the standard
  MARBLE-style protocol and the only view of the MERT embedding space itself, so it is
  what reveals an embedding-front-end change (for example the decode sample rate). The
  correlations are computed out-of-fold, so the probe cannot inflate its own score.

Each axis reports Pearson, Spearman and CCC (`moodengine.evaluation.concordance_correlation_coefficient`,
Lin's concordance correlation, which unlike Pearson penalises scale/offset mismatch and
is the standard valence/arousal metric). CCC also arrives split into its two factors,
`<axis>_ccc_rho` (how well the shapes agree) and `<axis>_ccc_cb` (how much of the loss is
scale/offset), because they call for opposite responses: a low `rho` means the model does not
track the axis, while a low `c_b` at a healthy `rho` means it does and only needs rescaling.
`zeroshot` runs additionally emit a `zeroshot_calibrated` block — the same scores after an affine
map fitted OUT OF FOLD — which is what separates "the ordering is wrong" from "the ordering is
right and the range is not". `--out results.json` writes the numbers, so a before/after comparison
across an engine change is a plain file diff.

## Reading the numbers honestly

Three habits, each of which the protocol used to make impossible:

**`--limit` takes a seeded permutation, not a prefix.** DEAM's song ids run by annotation
campaign, so a prefix evaluated one contiguous block of provenance. `--seed` fixes *which*
songs, so a run stays reproducible without being unrepresentative.

**Every statistic carries a 95 % bootstrap CI**, and comparisons use `--compare`, which
runs a *paired* bootstrap against an earlier `--out` JSON. This is the difference between
"it moved" and "it improved": two arms scored on the same songs share their audio and
their labels, so their errors are correlated and the paired interval on the difference is
much tighter than the overlap of two marginal intervals suggests. `--compare` refuses to
pair runs that scored different song sets, because a mismatched comparison is a marginal
one wearing a paired label.

**`--max-disagreement` filters on annotator spread.** DEAM ships `arousal_std` /
`valence_std` per song. A song the annotators themselves disagreed about cannot
discriminate between two models, so a correlation over the whole set is partly measuring
label noise.

### What this has already settled

Zero-shot, CLAP against MuQ-MuLan, paired Pearson difference on the same songs:

| n | energy Δ (95 % CI) | valence Δ (95 % CI) |
| --- | --- | --- |
| 150 | +0.087 [−0.015, +0.196] — in the noise | −0.096 [−0.238, +0.045] — in the noise |
| **400** | **+0.073 [+0.013, +0.136]** — real | **−0.097 [−0.182, −0.030]** — real |

**Neither backbone dominates.** MuQ-MuLan reads arousal significantly better and valence
significantly worse, and the split is by axis rather than a sampling artifact — at n=150
both intervals straddled zero, at n=400 both exclude it in opposite directions. Pick by
which axis your application leans on, or keep both.

Two things this pair of rows demonstrates about the protocol itself. The n=150 row is why
an interval is mandatory: the point estimates there (+0.087, −0.096) look decisive and are
not. And the sample size needed to resolve a ~0.08 difference on this data is somewhere
between 150 and 400 songs — worth knowing before trusting a quick `--limit 100` run.

It also shows why separability is not accuracy. MuQ-MuLan is clearly ahead on
`label_direction_redundancy` (mean mutual cosine 0.400 against CLAP's 0.568) yet loses
half the accuracy comparison.

### A gap that remains

There is no artist grouping. `KFold(shuffle=True)` splits at clip level, so two clips by
the same artist can straddle a fold and inflate the probe — the album effect, MIR's oldest
known source of optimism. `fetch_deam.py` downloads audio and annotations only; DEAM's
per-song metadata is not fetched, so the groups a `GroupKFold` needs are not on disk. This
affects the **probe** block's absolute value, not the zero-shot block (which fits nothing)
and not the paired comparisons (both arms inherit the same inflation).

`--limit` bounds the (dominant) embedding cost; a few hundred tracks already give a stable
correlation on CPU. The cap of 12 ten-second segments never bites on DEAM's 45-second
clips, so this benchmark exercises temporal *pooling* but not the long-track segment
*selection* policy — that behaviour is pinned by the `io_audio` unit tests instead.

## Text -> playlist retrieval

`scripts/bench_text_retrieval.py` measures the headline capability — type a mood, get
tracks — which had no measurement at all: `evaluate_text_queries` shipped with zero callers,
so a regression from a prompt, pooling or recentering change would have been invisible.

```bash
uv run --extra models --extra muq python scripts/bench_text_retrieval.py \
    --data-dir ~/moodengine-bench/deam --limit 400 --embedder clap
```

**Relevance comes from DEAM's human ratings, not from the engine.** Scoring against the
engine's own mood labels would grade the labeller with its own answers. Instead each of four
pole queries takes as relevant the top (or bottom) quartile of songs by *annotator*
arousal/valence, while the ranking comes from the text encoder alone. `--quantile` sets that
cut, and it is also the random-ranking floor: at the default 0.25 a coin flip scores
P@10 = 0.25, so read that first.

Measured on 400 songs, `k=10`:

| query | CLAP P@10 | MuQ-MuLan P@10 |
| --- | --- | --- |
| high-energy, intense, driving | 0.700 | 0.600 |
| calm, quiet, low-energy | 0.700 | 0.500 |
| **happy, cheerful, positive** | **0.300** | **0.100** |
| sad, gloomy, negative | 0.700 | 0.600 |
| **macro P@10** | **0.600** [0.425, 0.750] | 0.450 [0.325, 0.650] |
| macro MAP | **0.448** [0.408, 0.494] | 0.420 [0.379, 0.460] |

Two things worth acting on. **Positive valence is the stack's weak spot**: "happy" retrieves
at 0.300 against a 0.250 floor for CLAP, and at 0.100 — *below* chance, i.e. anti-correlated
— for MuQ-MuLan, while every other pole clears 0.500. And **CLAP wins retrieval** even though
MuQ-MuLan wins arousal regression, which is the third independent measurement agreeing that
neither backbone dominates.
