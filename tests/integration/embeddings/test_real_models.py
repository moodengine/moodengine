"""Integration test against the REAL MERT/CLAP models (torch required).

Opt-in only: marked ``model`` and deselected by default (see pyproject addopts
``-m 'not model'``). Run explicitly with::

    pytest -m model

Needs the ``models`` extra (``uv sync --extra models``) and the checkpoints downloaded
(MERT-v1-95M + the CLAP music checkpoint). Validates that the real embedders
honor the Embedder contract end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
from assertpy import assert_that

from moodengine.config import default_config
from moodengine.pipeline import get_embedder

pytestmark = pytest.mark.model


def test_real_clap_text_and_audio_shapes(synth_clip):
    # Arrange
    cfg = default_config()
    clap = get_embedder("clap", cfg)
    wav = synth_clip("tone", seconds=2.0, sr=cfg.clap_sample_rate)

    # Act
    audio = clap.extract(wav, cfg.clap_sample_rate)
    text = clap.embed_text(["an energetic upbeat song", "a calm ambient track"])

    # Assert
    assert_that(audio.ndim).is_equal_to(1)  # (hidden,) clip embedding
    assert_that(text.shape[0]).is_equal_to(2)  # (n_prompts, hidden)
    assert_that(text.shape[1]).is_equal_to(audio.shape[0])


def test_real_clap_single_prompt_does_not_crash():
    """Regression: a SINGLE text prompt must embed cleanly. laion_clap's tokenizer squeezes the batch
    dim, so N=1 reaches transformers-5 RoBERTa with a 1-D ``input_ids`` and used to raise ``IndexError``
    in ``create_position_ids_from_input_ids`` (``cumsum(mask, dim=1)``). The N=2 test above never
    exercised this (squeeze is a no-op for a batch), which is how the break shipped. ``embed_text`` now
    re-batches (``_ensure_batched``)."""
    cfg = default_config()
    clap = get_embedder("clap", cfg)

    text = clap.embed_text(["a single mellow jazz tune"])  # N=1 — the case that regressed

    assert_that(text.ndim).is_equal_to(2)  # (1, hidden), not a crash
    assert_that(text.shape[0]).is_equal_to(1)


def test_real_mulan_text_and_audio_share_one_space(synth_clip):
    """MuQ-MuLan is the alternative audio-text backbone: its audio and text embeddings must land in
    the SAME 512-d space, or the zero-shot labeling stack cannot use it at all."""
    mulan = pytest.importorskip("moodengine.embeddings.mulan", reason="needs the muq extra")
    cfg = default_config()
    embedder = mulan.MuLanEmbedder(cfg)
    wav = synth_clip("tone", seconds=2.0, sr=cfg.mulan_sample_rate)

    audio = embedder.extract(wav, cfg.mulan_sample_rate)
    text = embedder.embed_text(["an energetic upbeat song", "a calm ambient track"])

    assert_that(audio.ndim).is_equal_to(1)  # (hidden,) clip embedding, like CLAP
    assert_that(text.shape[0]).is_equal_to(2)
    assert_that(text.shape[1]).is_equal_to(audio.shape[0])


def test_real_mulan_rows_are_unit_norm(synth_clip):
    """The model L2-normalizes each internal 10 s clip then averages them WITHOUT re-normalizing,
    so anything longer comes back with norm < 1 and a dot product against text would not be a
    cosine. The embedder restores unit norm at the boundary; assert it on audio LONGER than one
    clip, which is the only case that can regress."""
    mulan = pytest.importorskip("moodengine.embeddings.mulan", reason="needs the muq extra")
    cfg = default_config()
    embedder = mulan.MuLanEmbedder(cfg)
    wav = synth_clip("tone", seconds=25.0, sr=cfg.mulan_sample_rate)  # > the 10 s internal clip

    audio = embedder.extract(wav, cfg.mulan_sample_rate)

    assert_that(float(np.linalg.norm(audio))).is_close_to(1.0, tolerance=1e-5)


def test_real_mulan_rejects_off_rate_audio(synth_clip):
    """24 kHz, not CLAP's 48 kHz. Off-rate audio is time/pitch-warped in a way nothing downstream
    can detect, so it must fail rather than embed."""
    mulan = pytest.importorskip("moodengine.embeddings.mulan", reason="needs the muq extra")
    cfg = default_config()
    embedder = mulan.MuLanEmbedder(cfg)

    with pytest.raises(ValueError, match=r"received audio at 48000 Hz but expects 24000 Hz"):
        embedder.extract(synth_clip("tone", seconds=1.0, sr=48_000), 48_000)


def test_real_clap_batched_extract_matches_the_per_segment_path(synth_clip):
    """Batching must not change the numbers. Segments are RAGGED — a track's trailing window is
    short — so `np.stack` over all of them raises; grouping by length keeps each group rectangular
    and leaves the arithmetic alone. Assert at float32-ULP tolerance, not exact equality: BLAS
    accumulation order depends on the batch shape."""
    cfg = default_config()
    clap = get_embedder("clap", cfg)
    segments = [
        synth_clip("tone", seconds=10.0, sr=cfg.clap_sample_rate),
        synth_clip("percussive", seconds=10.0, sr=cfg.clap_sample_rate),
        synth_clip("tone", seconds=3.0, sr=cfg.clap_sample_rate),  # the ragged tail
    ]

    one_at_a_time = np.vstack([clap.extract(s, cfg.clap_sample_rate) for s in segments])
    batched = np.vstack(clap.extract_batch(segments, cfg.clap_sample_rate))

    np.testing.assert_allclose(batched, one_at_a_time, atol=2e-6)


def test_real_mert_layered_shape(synth_clip):
    # Arrange
    cfg = default_config()
    mert = get_embedder("mert", cfg)
    wav = synth_clip("tone", seconds=2.0, sr=cfg.mert_sample_rate)

    # Act
    emb = mert.extract(wav, cfg.mert_sample_rate)

    # Assert
    assert_that(emb.ndim).is_equal_to(3)  # (n_layers, n_frames, hidden)
