"""Zero-shot mood labeling, attribute scoring + cluster mood profiles.

Pure aggregation over already-computed embeddings. The only stages that need a
real model are the ones that call ``clap_embedder.embed_text`` to turn prompts
into text embeddings; everything else is numpy/pandas, so this module is
torch-free and imports cleanly with just numpy/pandas.

Quality levers over a naive single-prompt / top-1 scheme:
  * **Prompt ensembling** — each mood/pole is described by several prompt
    templates whose text embeddings are averaged, which de-noises the direction.
  * **Softmax calibration** — raw CLAP cosine similarities sit in a narrow band;
    a temperature-scaled softmax turns them into spread-out, comparable scores.
  * **Attribute axes** — two-pole energy & valence prompts give each track an
    interpretable [0, 1] coordinate, independent of the discrete mood vocabulary.
  * **Cluster mood profiles** — averaging per-track mood affinities inside a
    cluster describes it with a ranked profile instead of a single majority word.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from moodengine._math import l2_normalize
from moodengine._typing import SupportsEmbedText
from moodengine._validation import ensure_finite_2d

logger = logging.getLogger(__name__)

# Default temperature for the softmax that turns similarities into per-track scores.
#
# Read what it actually divides: `score_moods` softmaxes the RECENTERED similarities, not the raw
# cosines. Measured over 400 DEAM tracks against the shipped vocabulary, raw CLAP audio/text
# cosines run -0.157 to +0.380 between the 1st and 99th percentiles (full range -0.316 to +0.485)
# — not the tidy positive band this note used to claim — and recentering trims the mean per-track
# spread only from 0.302 to 0.260, about 14 %, nowhere near the halving it used to claim. So this
# value is tuned against a quantity one step removed from the one it describes, and both halves of
# that description were wrong. It is a spread aesthetic, not a statistical
# optimum — a 0.95 score does NOT mean "right 95 % of the time"; fitting an honest one is
# `moodengine.calibration.fit_temperature` on a gold set. The principled starting point, if you
# want one without gold labels, is `1 / exp(logit_scale_a)` read off the loaded CLAP model: the
# temperature its own contrastive objective was trained at.
DEFAULT_TEMPERATURE: float = 0.05

# --- prompt vocabulary -------------------------------------------------------------------------
# Mood name -> list of natural-language prompts, averaged into one direction per mood by
# `build_label_matrix`. Hand-written, multi-adjective, three per mood.
#
# That shape looks wrong against the published zero-shot recipes, which cross class words with
# many caption templates (CLIP's uses 80) on the theory that averaging over phrasings cancels the
# phrasing and leaves the concept. It was tested here rather than assumed: a 4-descriptor x
# 6-template cross product, 24 prompts per mood, measured against this exact table on the LAION
# music checkpoint. It lost on every axis.
#
#   direction separability   raw pairwise cosine 0.840 vs 0.568 (worse — the six shared carrier
#                            phrases dominate the embedding), and after removing that shared
#                            component the two are equivalent (-0.0566 vs -0.0573) with a LOWER
#                            effective rank, 11 of 18 dims against 12
#   labels on 60 real tracks 15 distinct top moods vs 16, top mood share 0.15 vs 0.13,
#                            top1-top2 margin 0.054 vs 0.096, mean entropy 2.662 vs 2.360
#
# So the templates cost more (a shared carrier direction every mood inherits) than the larger
# ensemble buys. Keep the hand-written table, and re-measure with
# `label_direction_redundancy` + `labeling_quality_metrics` before changing it again.
DEFAULT_MOOD_PROMPTS: dict[str, list[str]] = {
    "energetic": [
        "an energetic high-energy upbeat song",
        "a lively driving track full of energy",
        "an exciting pumped-up high-tempo tune",
    ],
    "calm": [
        "a calm peaceful relaxing track",
        "soothing gentle mellow music",
        "a quiet laid-back soft song",
    ],
    "melancholic": [
        "a melancholic sad emotional track",
        "a wistful sorrowful melancholy song",
        "music that feels longing and bittersweet",
    ],
    "happy": [
        "a happy cheerful feel-good song",
        "a joyful bright sunny track",
        "an upbeat positive carefree tune",
    ],
    "dark": [
        "a dark brooding ominous track",
        "a gloomy sinister song",
        "music that feels cold and menacing",
    ],
    "aggressive": [
        "an aggressive heavy intense track",
        "a fierce hard-hitting powerful song",
        "loud angry forceful music",
    ],
    "romantic": [
        "a romantic tender loving song",
        "a sensual intimate warm track",
        "music that feels affectionate and heartfelt",
    ],
    "epic": [
        "an epic cinematic dramatic track",
        "a grand powerful orchestral-feeling song",
        "triumphant heroic sweeping music",
    ],
    "dreamy": [
        "a dreamy ethereal atmospheric track",
        "a hazy floating ambient song",
        "soft shimmering otherworldly music",
    ],
    "groovy": [
        "a groovy danceable funky song",
        "a rhythmic head-nodding groove",
        "a smooth infectious dance track",
    ],
    "funky": [
        "a funky soulful bass-driven track",
        "a syncopated funk groove",
        "a slinky funky rhythm song",
    ],
    "jazzy": [
        "a jazzy smooth sophisticated track",
        "a lounge jazz song with swing",
        "music with jazzy chords and improvisation",
    ],
    "hypnotic": [
        "a hypnotic repetitive trance-like track",
        "a looping mesmerizing groove",
        "steady pulsing hypnotic music",
    ],
    "nostalgic": [
        "a nostalgic retro wistful track",
        "a vintage warm reminiscent song",
        "music that evokes fond memories",
    ],
    "uplifting": [
        "an uplifting inspiring hopeful track",
        "a euphoric soaring positive song",
        "music that feels uplifting and motivating",
    ],
    "tense": [
        "a tense suspenseful ominous track",
        "an anxious eerie unsettling song",
        "nervous edgy foreboding music",
    ],
    "spacey": [
        "a spacey cosmic psychedelic track",
        "a deep-space ambient drifting song",
        "trippy interstellar electronic music",
    ],
    "playful": [
        "a playful quirky lighthearted track",
        "a whimsical fun bouncy song",
        "cheeky cartoonish playful music",
    ],
}


def build_prompt_table(
    descriptors: dict[str, tuple[str, ...]], templates: tuple[str, ...]
) -> dict[str, list[str]]:
    """Cross every label's descriptors with every caption template -> a prompt table.

    Returns ``{label: [prompt, ...]}``, ``len(descriptors[label]) * len(templates)`` prompts per
    label in a stable descriptor-major order, shaped for :func:`build_label_matrix`. Each template
    must contain exactly one ``{}`` slot. Pure; the inputs are never mutated.

    For building a vocabulary of your own without hand-writing every combination. Note that this
    construction is NOT how :data:`DEFAULT_MOOD_PROMPTS` is built, and deliberately so — sharing
    caption templates across labels gives every label direction a common component, which measured
    worse here on both separability and label quality (see the note above that table). Check yours
    with :func:`label_direction_redundancy` rather than assuming a bigger ensemble is better.
    """
    for template in templates:
        if template.count("{}") != 1:
            raise ValueError(
                f"template {template!r} must contain exactly one '{{}}' slot for the descriptor"
            )
    return {
        label: [template.format(word) for word in words for template in templates]
        for label, words in descriptors.items()
    }


# --- affect grounding ---------------------------------------------------------------------------
# Where each mood sits on Russell's circumplex — the two-dimensional model (valence x arousal) that
# affective science uses to place emotion words — expressed on the SAME [0, 1] scale as
# `attribute_scores`, so a predicted mood and a measured axis are directly comparable.
#
# `kind` records something the vocabulary was mixing silently. Eleven entries are affect words, whose
# circumplex position is what the word means. The rest are TEXTURE words — genre or production
# descriptors that carry a loose affective connotation but are not primarily emotions; "jazzy" is
# not a feeling. Their coordinates are the typical affect of music so described, which is a weaker
# claim, so `mood_affect_consistency` reports the two groups separately rather than pooling them.
#
# Measured against DEAM's human ratings over 400 songs, the engine's own labels order by gold
# arousal about as this table predicts at the extremes — `energetic` highest at 0.672, `romantic`
# lowest at 0.317 — with two clear disagreements. Tracks the engine calls `tense` sit at 0.319,
# among the three LOWEST of any mood, where the circumplex puts tense high; and `aggressive` lands
# mid-range at 0.528 rather than near the top. Read the first with its sample size: only 10 of the
# 400 tracks were labelled `tense`, against 36 for `aggressive`. Those are the disagreements this
# table exists to surface; do not silently "fix" them by moving the coordinates to match.
MOOD_AFFECT: dict[str, tuple[float, float, str]] = {
    # mood: (valence, arousal, kind)
    "energetic": (0.60, 0.90, "affect"),
    "calm": (0.60, 0.15, "affect"),
    "melancholic": (0.20, 0.30, "affect"),
    "happy": (0.90, 0.70, "affect"),
    "dark": (0.15, 0.40, "affect"),
    "aggressive": (0.20, 0.95, "affect"),
    "romantic": (0.80, 0.35, "affect"),
    "tense": (0.20, 0.75, "affect"),
    "uplifting": (0.90, 0.75, "affect"),
    "playful": (0.85, 0.65, "affect"),
    "nostalgic": (0.50, 0.30, "affect"),
    "epic": (0.70, 0.85, "texture"),
    "dreamy": (0.60, 0.20, "texture"),
    "groovy": (0.75, 0.70, "texture"),
    "funky": (0.80, 0.75, "texture"),
    "jazzy": (0.65, 0.45, "texture"),
    "hypnotic": (0.50, 0.50, "texture"),
    "spacey": (0.50, 0.25, "texture"),
}

#: How far a track's measured axis may sit from its mood's expected coordinate before the pair is
#: called incoherent. 0.35 on a [0, 1] axis is deliberately loose: the goal is to catch
#: "top_mood='calm' with energy 0.9", not to police ordinary spread.
MOOD_AFFECT_TOLERANCE: float = 0.35


def mood_affect_consistency(
    label_df: pd.DataFrame,
    affect: dict[str, tuple[float, float, str]] = MOOD_AFFECT,
    tolerance: float = MOOD_AFFECT_TOLERANCE,
) -> dict:
    """Does the discrete mood agree with the continuous axes? Nothing checked this before.

    The two outputs are computed from the same CLAP embeddings by independent prompt sets, so
    ``top_mood="calm"`` alongside ``energy=0.9`` is possible and was never flagged. This scores
    that agreement.

    ``label_df`` needs ``top_mood`` plus ``energy`` / ``valence`` columns — the shape
    :func:`moodengine.pipeline.run_pipeline_core` produces by joining :func:`label_tracks` with
    :func:`attribute_scores`. Returns ``{"arousal_pearson", "valence_pearson", "incoherent_share",
    "worst_moods", "n_scored", "n_affect", "n_texture"}``, where the two correlations are between
    each track's EXPECTED coordinate (looked up from its mood) and its MEASURED axis, and
    ``worst_moods`` lists the moods with the largest mean gap, descending.

    Affect words and texture words are counted separately (``n_affect`` / ``n_texture``) because
    the claim differs: a texture word's coordinate describes the typical affect of music so
    described, not the meaning of the word. Moods missing from ``affect`` are skipped, not guessed.
    Pure; robust to an empty frame.
    """
    empty = {
        "arousal_pearson": float("nan"),
        "valence_pearson": float("nan"),
        "incoherent_share": 0.0,
        "worst_moods": [],
        "n_scored": 0,
        "n_affect": 0,
        "n_texture": 0,
    }
    needed = {"top_mood", "energy", "valence"}
    if not needed <= set(label_df.columns) or len(label_df) == 0:
        return empty

    moods = [str(m) for m in label_df["top_mood"]]
    measured_a = np.asarray(label_df["energy"], dtype=np.float64)
    measured_v = np.asarray(label_df["valence"], dtype=np.float64)
    # `np.isfinite` on a SCALAR pays the whole ufunc dispatch, and the comprehension this replaces
    # paid it 2n times: 17.9 ms of this function's 38.5 ms at 20 000 tracks, over half the runtime,
    # for a check that vectorizes to 1.0 ms. The `and` used to short-circuit past both checks for an
    # unknown mood; `&` evaluates them regardless, which cannot change the result — a row with a
    # non-finite axis is dropped whether or not its mood is in the vocabulary.
    known = np.fromiter((m in affect for m in moods), dtype=bool, count=len(moods))
    keep = known & np.isfinite(measured_a) & np.isfinite(measured_v)
    if not bool(keep.any()):
        return empty

    kept_moods = [m for m, k in zip(moods, keep) if k]
    expected_v = np.array([affect[m][0] for m in kept_moods], dtype=np.float64)
    expected_a = np.array([affect[m][1] for m in kept_moods], dtype=np.float64)
    got_a, got_v = measured_a[keep], measured_v[keep]

    def _r(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    gap = np.maximum(np.abs(expected_a - got_a), np.abs(expected_v - got_v))
    per_mood: dict[str, list[float]] = {}
    for m, g in zip(kept_moods, gap):
        per_mood.setdefault(m, []).append(float(g))

    return {
        "arousal_pearson": _r(expected_a, got_a),
        "valence_pearson": _r(expected_v, got_v),
        "incoherent_share": float(np.mean(gap > float(tolerance))),
        # Descending by mean gap. `sorted` is stable and `per_mood` is insertion-ordered, so moods
        # tied on the gap come back in order of FIRST APPEARANCE in the frame — not alphabetically,
        # and not in vocabulary order. Grouping this with `np.bincount` would silently switch to the
        # latter, which is why that rewrite is not worth its 2.8 ms.
        "worst_moods": sorted(
            ((m, float(np.mean(g))) for m, g in per_mood.items()), key=lambda t: -t[1]
        )[:5],
        "n_scored": int(keep.sum()),
        "n_affect": sum(1 for m in kept_moods if affect[m][2] == "affect"),
        "n_texture": sum(1 for m in kept_moods if affect[m][2] == "texture"),
    }


# Two-pole attribute axes. Each pole is ensembled like the moods above; the score
# is the softmax probability of the positive pole -> a [0, 1] coordinate.
ENERGY_PROMPTS: dict[str, list[str]] = {
    "low": [
        "a calm low-energy slow track",
        "relaxed mellow gentle music",
        "a quiet sparse laid-back song",
    ],
    "high": [
        "an energetic high-energy fast track",
        "an intense driving powerful song",
        "a loud pumping high-tempo tune",
    ],
}
VALENCE_PROMPTS: dict[str, list[str]] = {
    "negative": [
        "a dark sad gloomy track",
        "a melancholic depressing tense song",
        "music with a negative heavy mood",
    ],
    "positive": [
        "a happy bright uplifting track",
        "a cheerful joyful warm song",
        "music with a positive feel-good mood",
    ],
}


def softmax(
    scores: np.ndarray, temperature: float = DEFAULT_TEMPERATURE, axis: int = -1
) -> np.ndarray:
    """Temperature-scaled softmax along ``axis`` (numerically stable)."""
    s = np.asarray(scores, dtype=np.float32) / max(float(temperature), 1e-6)
    s = s - np.max(s, axis=axis, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=axis, keepdims=True)


def label_prior(sims: np.ndarray) -> NDArray[np.float32]:
    """Per-label mean cosine over a reference corpus — the FIXED offset :func:`recenter_similarities`
    should subtract.

    ``sims`` is ``(n, n_labels)`` cosine similarities for the reference set (typically the whole
    library, scored once against the vocabulary from :func:`build_label_matrix`). Returns the
    ``(n_labels,)`` float32 column means. Persist it next to the label matrix and pass it back as
    ``prior=`` from then on: the estimate is only as good as the corpus behind it, so it must
    be computed on a set that represents the listening domain, not on whatever batch happens to be
    in flight. Empty or mis-shaped input yields zeros — a no-op offset, never a fabricated one.
    """
    s = np.asarray(sims, dtype=np.float32)
    if s.ndim != 2 or s.shape[0] == 0:
        width = s.shape[1] if s.ndim == 2 else 0
        return np.zeros((width,), dtype=np.float32)
    return s.mean(axis=0, dtype=np.float32)


#: Rows below which a per-label mean is not an estimate worth subtracting. Centering on fewer
#: than this leaves the offset dominated by the very rows it is meant to correct — in the limit of
#: one row it IS that row, so subtracting it yields exactly zero and every score collapses to the
#: uniform softmax. Shared by :func:`recenter_similarities` and by any caller deciding whether to
#: derive a prior at all, so the two cannot drift.
RECENTER_MIN_N: int = 5


def recenter_similarities(
    sims: np.ndarray,
    enable: bool = True,
    min_n: int = RECENTER_MIN_N,
    prior: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Subtract each label's mean cosine to cancel its modality-gap offset.

    ``sims`` is ``(n, n_labels)`` cosine similarities. With ``prior`` — a ``(n_labels,)`` reference
    vector from :func:`label_prior` — that FIXED offset is subtracted and ``min_n`` does not apply:
    one row alone is corrected exactly as it would be inside any other batch. Without one, the
    offset falls back to this batch's own column means when there are at least ``min_n`` rows,
    and ``sims`` passes through unchanged below that.

    **Prefer a prior.** The batch-mean fallback makes a track's label depend on which other tracks
    were scored alongside it: measured on CLAP-like geometry, 60 % of tracks get a different
    ``top_mood`` in a 5-track batch than in their full corpus, 40 % at n=10, still 9 % at n=100.
    It also fights the corpus it is meant to describe — on a library where 80 % of tracks genuinely
    share one mood, centering removes exactly that shared component and the mood is predicted for
    only ~8 % of them. The published account of the modality gap is a near-constant offset
    direction, so estimating it ONCE on a reference set is the correction that matches the
    phenomenon; a per-batch mean is only valid when the batch represents the whole domain, which a
    playlist never does. Pure; does not mutate the input.
    """
    s = np.asarray(sims, dtype=np.float32)
    if not enable or s.ndim != 2:
        return s
    if prior is not None:
        offset = np.asarray(prior, dtype=np.float32).reshape(1, -1)
        if offset.shape[1] != s.shape[1]:
            raise ValueError(
                f"label_prior has {offset.shape[1]} entries but sims has {s.shape[1]} labels; "
                "the prior must come from the same vocabulary (see label_prior)"
            )
        return (s - offset).astype(np.float32, copy=False)
    if s.shape[0] < int(min_n):
        logger.info(
            "recenter: %d rows < min_n=%d and no prior given; leaving similarities uncentered.",
            s.shape[0],
            int(min_n),
        )
        return s
    logger.info(
        "recenter: no prior given; falling back to this batch's own column means over %d rows, "
        "so these labels depend on the batch composition (pass prior= from label_prior to fix).",
        s.shape[0],
    )
    return s - s.mean(axis=0, keepdims=True)


def build_label_matrix(
    clap_embedder: SupportsEmbedText, prompts: dict[str, list[str]]
) -> tuple[list[str], np.ndarray]:
    """Encode ensembled prompts into one L2-normalized vector per label.

    For each label, all its prompt templates are embedded with
    ``clap_embedder.embed_text`` (one batched call), averaged, and re-normalized.
    Returns ``(label_names, matrix)`` where ``matrix`` is ``(n_labels, dim)``.
    Accepts ``str`` prompt values too (treated as a single-element list).
    """
    names = list(prompts.keys())
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for name in names:
        vals = prompts[name]
        if isinstance(vals, str):
            vals = [vals]
        start = len(flat)
        flat.extend(vals)
        spans.append((start, len(flat)))

    text_emb = np.asarray(clap_embedder.embed_text(flat), dtype=np.float32)
    if text_emb.ndim == 1:
        text_emb = text_emb[None, :]
    dim = text_emb.shape[1]

    matrix = np.zeros((len(names), dim), dtype=np.float32)
    for i, (start, end) in enumerate(spans):
        matrix[i] = text_emb[start:end].mean(axis=0)
    return names, l2_normalize(matrix, axis=1)


@dataclass(frozen=True)
class MoodScores:
    """The mood-scoring triptych for a batch of tracks, as one immutable result.

    ``mood_names`` labels the columns of the three ``(n, n_moods)`` float32
    arrays, which are the same signal at three calibration stages: ``sims`` —
    raw cosine similarities (audio rows × label matrix); ``recentered`` —
    per-mood centered similarities (equal to ``sims``, possibly sharing memory,
    when recentering was disabled or ``n < 5``), the right signal for
    cross-track aggregation such as cluster profiles; ``probs`` —
    temperature-softmax over ``recentered``, each row summing to 1, the
    calibrated per-track label distribution.
    """

    mood_names: list[str]
    sims: NDArray[np.float32]
    recentered: NDArray[np.float32]
    probs: NDArray[np.float32]


def score_moods(
    audio_embs: np.ndarray,
    mood_names: list[str],
    label_matrix: np.ndarray,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    recenter: bool = True,
    prior: np.ndarray | None = None,
) -> MoodScores:
    """Score tracks against a precomputed label matrix: sims → recenter → softmax.

    The pure core behind :func:`label_tracks` and :func:`cluster_mood_profiles`,
    exposed so a caller holding a :func:`build_label_matrix` result (e.g. a
    long-lived app scoring many batches against one vocabulary) can score
    without a live embedder. ``audio_embs`` is ``(n, d)`` (a single ``(d,)``
    track is promoted to ``(1, d)``); ``label_matrix`` is ``(n_moods, d)``;
    both are assumed L2-normalized so the matmul is cosine similarity.
    ``recenter`` applies :func:`recenter_similarities`: with ``prior`` — a ``(n_moods,)`` fixed
    offset from :func:`label_prior` — at ANY ``n`` including 1; without one it falls back to this
    batch's own column means and is active only for ``n >= 5``, which makes the result depend on
    the batch. Non-finite audio embeddings raise ``ValueError`` naming the
    offending rows — a NaN row would otherwise poison the per-mood recentering
    means for every track. Returns a :class:`MoodScores` — see it for the
    semantics of each stage. Pure numpy, deterministic; inputs are never mutated.
    """
    X = np.asarray(audio_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    X = ensure_finite_2d(X, name="audio_embs")
    M = np.asarray(label_matrix, dtype=np.float32)

    sims = X @ M.T  # (n, n_moods) cosine similarities
    recentered = recenter_similarities(sims, enable=recenter, prior=prior)
    probs = softmax(recentered, temperature=temperature, axis=1)
    return MoodScores(mood_names=list(mood_names), sims=sims, recentered=recentered, probs=probs)


def _resolve_label_matrix(
    clap_embedder: SupportsEmbedText | None,
    prompts: dict[str, list[str]],
    label_matrix: tuple[list[str], np.ndarray] | None,
) -> tuple[list[str], np.ndarray]:
    """A precomputed ``(names, matrix)`` pair wins; otherwise encode ``prompts`` via the embedder.

    Raises :class:`ValueError` when neither an embedder nor a precomputed pair
    is available — scoring has no label directions to compare against.
    """
    if label_matrix is not None:
        names, matrix = label_matrix
        return list(names), np.asarray(matrix, dtype=np.float32)
    if clap_embedder is None:
        raise ValueError(
            "no label directions to score against: pass clap_embedder "
            "or a precomputed label_matrix=(mood_names, matrix) from build_label_matrix"
        )
    return build_label_matrix(clap_embedder, prompts)


def label_direction_redundancy(
    mood_names: list[str], label_matrix: np.ndarray, *, top_k: int = 10
) -> dict:
    """How distinguishable the label DIRECTIONS are from each other — the blind spot in
    :func:`labeling_quality_metrics`.

    That function measures the diversity of the assignments a vocabulary produced; this measures
    whether the vocabulary could produce diverse assignments at all. Two moods whose ensembled
    directions sit at cosine 0.97 cannot be told apart by any input, so every track that leans
    toward one leans equally toward the other, and which of them wins the argmax is decided by
    noise. That is invisible downstream: the labels look confident and vary across tracks.

    ``label_matrix`` ``(n_moods, d)`` is the L2-normalized output of :func:`build_label_matrix`,
    ``mood_names`` labels its rows. Returns ``{"mean_cosine", "max_cosine", "most_similar_pairs",
    "n_moods"}`` over the STRICT upper triangle — no self-pairs, no mirrored duplicates —
    with ``most_similar_pairs`` the ``top_k`` ``(mood_a, mood_b, cosine)`` triples in descending
    cosine. Rows are re-normalized defensively so the values are true cosines on any input.

    Read it when adding or renaming a mood: a new entry that lands near an existing direction adds
    a word to the output without adding a distinction. Pure numpy; fewer than 2 moods yields zeroed
    metrics and an empty pair list rather than raising.
    """
    M = np.asarray(label_matrix, dtype=np.float32)
    n = M.shape[0] if M.ndim == 2 else 0
    if n < 2 or len(mood_names) < n:
        return {"mean_cosine": 0.0, "max_cosine": 0.0, "most_similar_pairs": [], "n_moods": int(n)}

    sims = l2_normalize(M, axis=1) @ l2_normalize(M, axis=1).T
    iu = np.triu_indices(n, k=1)  # strict upper triangle: each unordered pair exactly once
    pair_cos = sims[iu]
    order = np.argsort(-pair_cos, kind="stable")[: max(int(top_k), 0)]
    return {
        "mean_cosine": float(pair_cos.mean()),
        "max_cosine": float(pair_cos.max()),
        "most_similar_pairs": [
            (mood_names[int(iu[0][o])], mood_names[int(iu[1][o])], float(pair_cos[o]))
            for o in order
        ],
        "n_moods": int(n),
    }


def compose_mood_vector(
    label_matrix: np.ndarray, mood_names: list[str], terms: list[tuple[str, float]]
) -> np.ndarray:
    """Mood-vector arithmetic: a signed, weighted combination of existing mood directions.

    ``label_matrix`` ``(n_moods, d)`` is the L2-normalized ensembled prompt matrix from
    :func:`build_label_matrix`; ``mood_names`` labels its rows; ``terms`` is ``[(mood_name, weight),
    …]`` where a positive weight pulls toward a mood and a negative weight pushes away from it —
    "calm but not melancholic" is ``[("calm", 1.0), ("melancholic", -1.0)]``. Returns
    ``l2_normalize(Σ_k w_k · label_matrix[idx(name_k)])`` ``(d,)`` float32 — one query vector in the
    shared CLAP space, rankable by the same cosine kNN as any mood. Names not in ``mood_names`` are
    ignored (the UI only offers real vocab); empty ``terms`` or a net-zero / cancelling combination
    yields ``np.zeros((d,))``. Pure numpy, torch-free, deterministic; the input is never mutated.
    """
    M = np.asarray(label_matrix, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] == 0:
        d = M.shape[1] if M.ndim == 2 else 0
        return np.zeros((d,), dtype=np.float32)
    idx = {name: i for i, name in enumerate(mood_names)}
    acc = np.zeros((M.shape[1],), dtype=np.float32)
    for name, weight in terms:
        i = idx.get(name)
        if i is not None:
            acc = acc + np.float32(weight) * M[i]
    if float(np.linalg.norm(acc)) < 1e-8:  # empty / unknown-only / perfectly cancelling
        return np.zeros((M.shape[1],), dtype=np.float32)
    return l2_normalize(acc, axis=-1).astype(np.float32)


def zero_shot_moods(
    audio_emb: np.ndarray,
    text_emb: np.ndarray,
    mood_names: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Rank moods for one track by cosine similarity.

    ``audio_emb`` (d,) and ``text_emb`` (n_moods, d) are assumed L2-normalized,
    so the dot product is the cosine similarity. Returns the ``top_k`` highest
    ``(mood, score)`` pairs sorted by score descending (ties break toward the
    earliest index). Pure numpy.
    """
    audio = np.asarray(audio_emb, dtype=np.float32).ravel()
    text = np.asarray(text_emb, dtype=np.float32)
    if text.ndim == 1:
        text = text[None, :]
    sims = text @ audio  # (n_moods,)
    k = max(0, min(int(top_k), len(mood_names)))
    if k == 0:
        return []
    order = np.argsort(-sims, kind="stable")[:k]
    return [(mood_names[i], float(sims[i])) for i in order]


def label_tracks(
    audio_embs: np.ndarray,
    clap_embedder: SupportsEmbedText | None = None,
    prompts: dict[str, list[str]] = DEFAULT_MOOD_PROMPTS,
    top_k: int = 3,
    temperature: float = DEFAULT_TEMPERATURE,
    recenter: bool = True,
    label_matrix: tuple[list[str], np.ndarray] | None = None,
    prior: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assign calibrated zero-shot mood labels to a batch of CLAP embeddings.

    ``audio_embs`` (n, d) are CLAP track embeddings. Prompts are ensembled into a
    mood matrix via :func:`build_label_matrix`, cosine similarities are turned
    into per-track probabilities with a temperature softmax (:func:`score_moods`),
    and the ``top_k`` moods are reported. When ``recenter`` (and n>=5), per-mood
    similarities are centered via :func:`recenter_similarities` before the softmax
    to cancel each mood's modality-gap prior. ``label_matrix`` accepts a
    precomputed ``(mood_names, matrix)`` pair as returned by
    :func:`build_label_matrix`; when given, the embedder is never consulted (it
    may be ``None``) and ``prompts`` is ignored — this is how a caller scoring
    several batches (or several stages) against one vocabulary avoids re-encoding
    the prompts each time. With neither an embedder nor ``label_matrix``,
    raises :class:`ValueError`. Returns a DataFrame (index ``0..n-1``) with columns:
    ``top_mood`` (str), ``top_score`` (float, softmax prob), ``mood_topk``
    (list[str]) and ``mood_topk_scores`` (list[float], probs aligned to topk).
    """
    X = np.asarray(audio_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    mood_names, mood_matrix = _resolve_label_matrix(clap_embedder, prompts, label_matrix)
    n_moods = len(mood_names)
    k = max(0, min(int(top_k), n_moods))

    probs = score_moods(
        X, mood_names, mood_matrix, temperature=temperature, recenter=recenter, prior=prior
    ).probs  # (n, n_moods)

    n = probs.shape[0]
    if k == 0:
        return pd.DataFrame(
            {
                "top_mood": [""] * n,
                "top_score": [float("nan")] * n,
                "mood_topk": [[] for _ in range(n)],
                "mood_topk_scores": [[] for _ in range(n)],
            },
            columns=["top_mood", "top_score", "mood_topk", "mood_topk_scores"],
        )

    # One vectorized rank over the whole (n, n_moods) block rather than a Python loop with a
    # per-row argsort: the loop was the bulk of this function's cost on a real library, and
    # `n_tracks` is exactly the dimension the performance rules say never to loop over. Stable
    # ordering is preserved, so ties still resolve to the lower mood index.
    order = np.argsort(-probs, axis=1, kind="stable")[:, :k]  # (n, k)
    names = np.asarray(mood_names, dtype=object)
    topk_names = names[order]  # (n, k)
    topk_scores = np.take_along_axis(probs, order, axis=1).astype(float)  # (n, k)

    return pd.DataFrame(
        {
            "top_mood": topk_names[:, 0].tolist(),
            "top_score": topk_scores[:, 0].tolist(),
            # `.tolist()` rather than `list(...)`: it unboxes to Python `str` / `float`, which is
            # what the documented `list[str]` / `list[float]` contract says. Wrapping the numpy
            # rows would leak `np.float64` into every cell.
            "mood_topk": topk_names.tolist(),
            "mood_topk_scores": topk_scores.tolist(),
        },
        columns=["top_mood", "top_score", "mood_topk", "mood_topk_scores"],
    )


def _resolve_axis_matrix(
    clap_embedder: SupportsEmbedText,
    axis_prompts: dict[str, list[str]],
    label_matrix: tuple[list[str], np.ndarray] | None,
    poles: list[str],
) -> NDArray[np.float32]:
    """The ``(2, d)`` pole matrix for a two-pole axis, built or taken from the caller.

    Shared by :func:`score_axis` and :func:`axis_prior` so the two-pole contract is enforced on
    every entry point that accepts a prebuilt matrix. Without the width check, handing over a
    vocabulary of another width — the 18-mood matrix, say — returns a plausible value in [0, 1]
    with no error at all, and the adjacent ``energy_matrix=`` / ``valence_matrix=`` parameters
    make that a one-token slip. The check is on the WIDTH, not the pole names: a caller may
    legitimately score a custom two-pole axis whose names differ from the shipped table.
    """
    if label_matrix is None:
        label_matrix = build_label_matrix(clap_embedder, axis_prompts)
    matrix = np.asarray(label_matrix[1], dtype=np.float32)  # (2, d): [neg, pos]
    if matrix.ndim != 2 or matrix.shape[0] != 2:
        raise ValueError(
            f"label_matrix must hold exactly 2 pole vectors for axis {poles}; got shape "
            f"{matrix.shape}. Build it from the SAME two-pole vocabulary you are scoring "
            "(e.g. ENERGY_PROMPTS for the energy axis)."
        )
    return matrix


def score_axis(
    audio_embs: np.ndarray,
    clap_embedder: SupportsEmbedText,
    axis_prompts: dict[str, list[str]],
    temperature: float = DEFAULT_TEMPERATURE,
    recenter: bool = True,
    prior: np.ndarray | None = None,
    label_matrix: tuple[list[str], np.ndarray] | None = None,
) -> np.ndarray:
    """Score tracks on a two-pole axis as the softmax prob of the positive pole.

    ``axis_prompts`` must have exactly two entries ``{negative_pole, positive_pole}``
    (insertion order = [negative, positive]). When ``recenter``, the two pole similarities are
    centered via :func:`recenter_similarities` first — against ``prior`` (a ``(2,)`` offset from
    :func:`axis_prior`, applying at any ``n``) when given, else this batch's own means for
    ``n >= 5``. Returns a (n,) array in [0, 1]: 0 = fully negative pole, 1 = fully
    positive pole.

    ``label_matrix`` reuses an already-built ``(names, (2, d))`` pair from
    :func:`build_label_matrix` instead of paying the text-encoder forward again — the same escape
    hatch :func:`label_tracks` offers, and what lets a caller derive the prior and the scores from
    ONE build.
    """
    poles = list(axis_prompts.keys())
    if len(poles) != 2:
        raise ValueError(f"axis_prompts must have exactly 2 poles; got {poles}")
    X = np.asarray(audio_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    matrix = _resolve_axis_matrix(clap_embedder, axis_prompts, label_matrix, poles)
    sims = X @ matrix.T  # (n, 2)
    sims = recenter_similarities(sims, enable=recenter, prior=prior)
    probs = softmax(sims, temperature=temperature, axis=1)
    return probs[:, 1].astype(np.float32)  # P(positive pole)


def axis_prior(
    audio_embs: np.ndarray,
    clap_embedder: SupportsEmbedText,
    axis_prompts: dict[str, list[str]],
    label_matrix: tuple[list[str], np.ndarray] | None = None,
) -> NDArray[np.float32]:
    """The FIXED ``(2,)`` offset :func:`score_axis` should subtract for one two-pole axis.

    The axis counterpart of :func:`label_prior`: scores ``audio_embs`` — the REFERENCE corpus,
    typically the whole library — against the axis vocabulary and returns the per-pole mean cosine.
    Persist it and pass it back as ``prior=``, or a later call re-estimates the offset from
    whatever rows it was handed and can place the same track differently on the axis.

    ``label_matrix`` reuses an already-built pair, so the prior and the scores it feeds can come
    from one text-encoder forward.
    """
    X = np.asarray(audio_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    matrix = _resolve_axis_matrix(clap_embedder, axis_prompts, label_matrix, list(axis_prompts))
    return label_prior(X @ matrix.T)


def attribute_priors(
    audio_embs: np.ndarray,
    clap_embedder: SupportsEmbedText,
    energy_matrix: tuple[list[str], np.ndarray] | None = None,
    valence_matrix: tuple[list[str], np.ndarray] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """``(energy_prior, valence_prior)`` for :func:`attribute_scores` over a reference corpus.

    Two separate ``(2,)`` offsets because each axis is built from its own two-pole vocabulary and
    carries its own modality-gap bias. Feed the pair straight into ``attribute_scores(...,
    energy_prior=..., valence_prior=...)`` and keep it alongside the mood prior from
    :func:`label_prior`: the three together are what make a single-track re-score reproduce the
    numbers a whole-library run produced. Pass the same matrices to both calls to build each axis
    vocabulary once.
    """
    return (
        axis_prior(audio_embs, clap_embedder, ENERGY_PROMPTS, label_matrix=energy_matrix),
        axis_prior(audio_embs, clap_embedder, VALENCE_PROMPTS, label_matrix=valence_matrix),
    )


def attribute_scores(
    audio_embs: np.ndarray,
    clap_embedder: SupportsEmbedText,
    temperature: float = DEFAULT_TEMPERATURE,
    recenter: bool = True,
    energy_prior: np.ndarray | None = None,
    valence_prior: np.ndarray | None = None,
    energy_matrix: tuple[list[str], np.ndarray] | None = None,
    valence_matrix: tuple[list[str], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Per-track interpretable attributes from two-pole axes.

    Returns a DataFrame (index ``0..n-1``) with ``energy`` and ``valence`` in
    [0, 1] (0 = low-energy / negative, 1 = high-energy / positive). ``recenter``
    is forwarded to :func:`score_axis` for both axes, as are the per-axis ``(2,)``
    ``energy_prior`` / ``valence_prior`` reference offsets (see
    :func:`recenter_similarities` for why a fixed prior beats the batch mean). The two axes have
    separate priors because each is built from its own two-pole vocabulary; :func:`attribute_priors`
    computes the pair over a reference corpus. Without them the offsets come from whatever batch is
    in flight (and only for ``n >= 5``), so a track's score depends on what it was scored with.
    ``energy_matrix`` / ``valence_matrix`` reuse already-built vocabularies, so deriving the priors
    and the scores costs one text-encoder forward per axis rather than two.
    """
    energy = score_axis(
        audio_embs,
        clap_embedder,
        ENERGY_PROMPTS,
        temperature,
        recenter=recenter,
        prior=energy_prior,
        label_matrix=energy_matrix,
    )
    valence = score_axis(
        audio_embs,
        clap_embedder,
        VALENCE_PROMPTS,
        temperature,
        recenter=recenter,
        prior=valence_prior,
        label_matrix=valence_matrix,
    )
    return pd.DataFrame({"energy": energy, "valence": valence})


def cluster_mood_profiles(
    audio_embs: np.ndarray,
    cluster_labels: np.ndarray,
    clap_embedder: SupportsEmbedText | None = None,
    prompts: dict[str, list[str]] = DEFAULT_MOOD_PROMPTS,
    top_k: int = 3,
    recenter: bool = True,
    label_matrix: tuple[list[str], np.ndarray] | None = None,
    prior: np.ndarray | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """Describe each cluster by its average mood affinity.

    Computes per-track cosine similarities to every mood, averages them within
    each cluster, and returns ``{cluster_id: [(mood, mean_score), ...]}`` with the
    ``top_k`` moods per cluster (noise cluster -1 included when present). When
    ``recenter`` (and n>=5), the per-mood similarities are centered via
    :func:`recenter_similarities` before averaging, which makes every profile a
    CONTRAST against the batch — and a single cluster spanning the whole batch has no
    contrast to report, so that combination raises :class:`ValueError` rather than
    ranking the rounding noise it would otherwise produce. Pass ``prior`` (a fixed
    offset, so the contrast is against it rather than against this batch) or
    ``recenter=False`` to profile one cluster. ``label_matrix`` accepts a
    precomputed ``(mood_names, matrix)`` pair as returned by
    :func:`build_label_matrix` — same contract as in :func:`label_tracks`: the
    embedder is then never consulted and may be ``None``; with neither, raises
    :class:`ValueError`. Pure aside from the single ``embed_text`` call inside
    :func:`build_label_matrix` (none at all with a precomputed matrix).
    """
    X = np.asarray(audio_embs, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    labels = np.asarray(cluster_labels).astype(int)
    # One group covering the whole batch has NO contrast to report when the centering
    # offset is the batch's own mean: cluster mean minus batch mean is then zero by
    # construction, so the averaged similarities are float rounding noise and ranking
    # them returns arbitrary moods. Measured on the same generator at 1e-8 magnitude,
    # the top-3 came back ['dreamy', 'energetic', 'uplifting'] at n=5,
    # ['jazzy', 'groovy', 'dark'] at n=100 and ['uplifting', 'dark', 'calm'] at
    # n=2000 — three different answers, none of them evidence. sklearn's
    # silhouette_score refuses the same degeneracy for the same reason. Both escapes
    # stay valid at one cluster: a `prior` makes the offset external to the batch, and
    # `recenter=False` removes it. Below RECENTER_MIN_N no centering happens at all,
    # so the profile is well defined there and the guard must not fire.
    if recenter and prior is None and X.shape[0] >= RECENTER_MIN_N and len(np.unique(labels)) == 1:
        raise ValueError(
            "cannot profile a single cluster spanning the whole batch while recentering "
            "on the batch mean: every cluster mean is then zero by construction and the "
            "ranking would be rounding noise. Pass prior=label_prior(...) to center on a "
            "fixed offset, or recenter=False to rank raw similarities."
        )
    mood_names, mood_matrix = _resolve_label_matrix(clap_embedder, prompts, label_matrix)
    # Cluster profiles aggregate the CENTERED similarities (comparable across
    # moods), not the per-track softmax probabilities.
    sims = score_moods(X, mood_names, mood_matrix, recenter=recenter, prior=prior).recentered
    k = max(0, min(int(top_k), len(mood_names)))

    profiles: dict[int, list[tuple[str, float]]] = {}
    for cid in sorted(set(labels.tolist())):
        mask = labels == cid
        mean_sims = sims[mask].mean(axis=0)  # (n_moods,)
        order = np.argsort(-mean_sims, kind="stable")[:k]
        profiles[int(cid)] = [(mood_names[i], float(mean_sims[i])) for i in order]
    return profiles


def name_clusters(cluster_labels: np.ndarray, top_moods: list[str]) -> dict:
    """Name each cluster by the majority top-mood of its tracks.

    ``cluster_labels`` (n,) and ``top_moods`` (length n) align by index. Returns
    ``{cluster_id: dominant_mood}`` via per-cluster majority vote, including the
    noise cluster (-1) when present. Ties break deterministically toward the mood
    that appears first. Pure.
    """
    labels = np.asarray(cluster_labels)
    moods = list(top_moods)
    groups: dict[int, list[str]] = {}
    for lbl, mood in zip(labels.tolist(), moods):
        groups.setdefault(int(lbl), []).append(mood)

    named: dict[int, str] = {}
    for cid, members in groups.items():
        counts = Counter(members)
        named[cid] = counts.most_common(1)[0][0]
    return named


def _mean_top1_minus_top2(label_df: pd.DataFrame) -> float:
    """Mean gap between each row's best and second-best mood score.

    A row with fewer than two scores contributes 0.0 rather than being dropped: the metric is a
    mean over TRACKS, so skipping the thin rows would quietly raise it on exactly the batches
    where confidence is least established.
    """
    if "mood_topk_scores" not in label_df:
        return 0.0

    margins = []
    for scores in label_df["mood_topk_scores"]:
        seq = list(scores) if scores is not None else []
        margins.append(float(seq[0] - seq[1]) if len(seq) >= 2 else 0.0)

    return float(np.mean(margins)) if margins else 0.0


def labeling_quality_metrics(label_df: pd.DataFrame, mood_names: list[str] | None = None) -> dict:
    """Health metrics for a :func:`label_tracks` output — three temperature-invariant, two not.

    Reports how well-spread the assignments are (diversity / dominance) and how
    confident/separated the top picks are. ``mood_names`` is accepted for API
    symmetry but unused by these metrics. Returns
    ``{"n_distinct_top_moods", "top_mood_histogram", "max_mood_share",
    "mean_top1_minus_top2", "mean_top_score"}``. Pure; robust to an empty df.

    **Only the three argmax-derived metrics survive a temperature change.** The softmax temperature
    is a monotone rescaling, so it never reorders a row: ``n_distinct_top_moods``,
    ``top_mood_histogram`` and ``max_mood_share`` are identical at any ``temperature``. The two
    that read the probability VALUES are not, and they move enormously — swept on a fixed 200-track
    batch across ``temperature`` 0.5 / 0.05 / 0.005, ``mean_top1_minus_top2`` went 0.001 → 0.027 →
    0.590 and ``mean_top_score`` 0.060 → 0.121 → 0.745. Comparing either across runs is only
    meaningful at a FIXED temperature; to compare across temperatures, work from
    :attr:`MoodScores.recentered` (whose scale the temperature does not touch) or calibrate first
    with :func:`moodengine.calibration.fit_temperature`.
    """
    top_moods = list(label_df["top_mood"]) if "top_mood" in label_df else []
    n = len(top_moods)
    counts = Counter(top_moods)
    n_distinct = len(counts)
    max_share = (max(counts.values()) / n) if n else 0.0

    top_scores = [
        float(s)
        for s in (label_df["top_score"] if "top_score" in label_df else [])
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    ]
    mean_top_score = float(np.mean(top_scores)) if top_scores else 0.0

    return {
        "n_distinct_top_moods": int(n_distinct),
        "top_mood_histogram": {m: int(c) for m, c in counts.items()},
        "max_mood_share": float(max_share),
        "mean_top1_minus_top2": _mean_top1_minus_top2(label_df),
        "mean_top_score": mean_top_score,
    }
