"""Integration test against the REAL MERT/CLAP models (torch required).

Opt-in only: marked ``model`` and deselected by default (see pyproject addopts
``-m 'not model'``). Run explicitly with::

    pytest -m model

Needs ``pip install "moodengine[models]"`` and the checkpoints downloaded
(MERT-v1-95M + the CLAP music checkpoint). Validates that the real embedders
honor the Embedder contract end-to-end.
"""

from __future__ import annotations

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


def test_real_mert_layered_shape(synth_clip):
    # Arrange
    cfg = default_config()
    mert = get_embedder("mert", cfg)
    wav = synth_clip("tone", seconds=2.0, sr=cfg.mert_sample_rate)

    # Act
    emb = mert.extract(wav, cfg.mert_sample_rate)

    # Assert
    assert_that(emb.ndim).is_equal_to(3)  # (n_layers, n_frames, hidden)
