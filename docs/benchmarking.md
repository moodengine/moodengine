# Benchmarking against ground truth

The engine's mood outputs are only trustworthy if you can *measure* them. This guide runs
the pipeline against human valence/arousal ratings so a change can be shown to improve or
regress quality, rather than asserted to. It needs the model backbones
(`pip install "moodengine[models]"`) and is a developer workflow, not part of the library
surface.

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

- **`zeroshot`** — the product path: CLAP zero-shot `attribute_scores` (energy, valence)
  correlated with the gold ratings. Measures the labelling / prompt / recentering stack.
- **`probe`** — a cross-validated ridge linear probe on frozen `--embedder` embeddings
  (`mert`, `clap` or `fused`) regressed onto the gold ratings. This is the standard
  MARBLE-style protocol and the only view of the MERT embedding space itself, so it is
  what reveals an embedding-front-end change (for example the decode sample rate). The
  correlations are computed out-of-fold, so the probe cannot inflate its own score.

Each axis reports Pearson, Spearman and CCC (`moodengine.evaluation.concordance_correlation_coefficient`,
Lin's concordance correlation, which unlike Pearson penalises scale/offset mismatch and
is the standard valence/arousal metric). `--out results.json` writes the numbers, so a
before/after comparison across an engine change is a plain file diff.

`--limit` bounds the (dominant) embedding cost; a few hundred tracks already give a stable
correlation on CPU. The cap of 12 ten-second segments never bites on DEAM's 45-second
clips, so this benchmark exercises temporal *pooling* but not the long-track segment
*selection* policy — that behaviour is pinned by the `io_audio` unit tests instead.
