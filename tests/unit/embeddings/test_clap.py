"""Unit tests for the torch-only, model-free helpers in ``moodengine.embeddings.clap``.

The full CLAP embedder needs the checkpoint (see ``tests/integration/embeddings/test_real_models.py``,
``-m model``); here we pin only ``_ensure_batched``, the shape fix that keeps laion_clap's single-prompt
tokenizer output 2-D for transformers-5 RoBERTa. Importing ``clap`` eagerly pulls in ``laion_clap``, so
skip cleanly when the ``models`` extra is absent.
"""

from __future__ import annotations

import pytest
from assertpy import assert_that

torch = pytest.importorskip("torch")
# Import via the clap module (which clears sys.argv around the laion_clap import — importing laion_clap
# directly runs its argparse at module load and SystemExits under pytest). Skips if models extra absent.
clap = pytest.importorskip("moodengine.embeddings.clap")
_ensure_batched = clap._ensure_batched


def test_ensure_batched_readds_squeezed_batch_dim():
    # laion_clap's tokenizer squeezes a single prompt to 1-D (input_ids, attention_mask); restore (1, seq).
    tok = {
        "input_ids": torch.zeros(7, dtype=torch.long),
        "attention_mask": torch.ones(7, dtype=torch.long),
    }
    out = _ensure_batched(tok)
    assert_that(out["input_ids"].dim()).is_equal_to(2)
    assert_that(tuple(out["input_ids"].shape)).is_equal_to((1, 7))
    assert_that(out["attention_mask"].dim()).is_equal_to(2)


def test_ensure_batched_leaves_a_real_batch_untouched():
    # For N>1 the tokenizer's squeeze is a no-op, so the input is already (n, seq) — must pass through.
    tok = {"input_ids": torch.zeros(3, 7, dtype=torch.long)}
    out = _ensure_batched(tok)
    assert_that(tuple(out["input_ids"].shape)).is_equal_to((3, 7))
    assert_that(out["input_ids"]).is_same_as(tok["input_ids"])  # untouched, not copied
