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
uv run --extra models python scripts/fetch_deam.py --data-dir ~/moodengine-bench/deam
```

It prints the audio directory and the static-annotations CSV to point the runner at.

## Running the benchmark

```bash
uv run --extra models python scripts/bench_valence_arousal.py \
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
is the standard valence/arousal metric). `--out results.json` writes the numbers, so a
before/after comparison across an engine change is a plain file diff.

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

Measured on 150 DEAM songs (`--seed 0`), zero-shot, CLAP against MuQ-MuLan:

| axis | CLAP | MuQ-MuLan | paired Δ (95 % CI) |
| --- | --- | --- | --- |
| energy  | 0.513 | **0.600** | +0.087 [−0.015, +0.196] |
| valence | **0.483** | 0.387 | −0.096 [−0.238, +0.045] |

MuQ-MuLan reads arousal better and valence worse, and **neither difference clears the
noise floor at this sample size**. That is worth stating plainly, because MuQ-MuLan looks
clearly ahead on the separability diagnostics (`label_direction_redundancy` reports mean
mutual cosine 0.400 against CLAP's 0.568) — and separability is not accuracy. Keep both
backbones and measure on your own library.

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
