"""MERT frame-level audio embedder.

Wraps the MERT-v1-95M self-supervised music model. Loading is lazy: the model
and feature extractor are fetched from the Hugging Face hub on first
construction, so importing this module is cheap apart from the (eager) torch +
transformers imports at the top. :meth:`MERTEmbedder.extract` returns the full
stack of hidden states for one segment; track-level pooling and caching live in
:mod:`moodengine.pooling` / :mod:`moodengine.pipeline`.

Licensing note: the MERT-v1-95M *weights* are CC-BY-NC-4.0 (non-commercial) —
a separate grant from this package's own code license. Any commercial use of the
weights requires licensing them separately or configuring a different model.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from moodengine.config import Config
from moodengine.embeddings.base import Embedder
from moodengine.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

# Reviewed snapshot of m-a-p/MERT-v1-95M (upstream last modified 2025-05-25).
# MERT ships custom modeling code executed via trust_remote_code — arbitrary
# Python pulled from the hub — so the revision is pinned to freeze exactly
# which code runs. Bump deliberately, after reviewing the upstream diff.
_DEFAULT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


class MERTEmbedder(Embedder):
    """Frame-level embedder backed by ``m-a-p/MERT-v1-95M``.

    ``extract`` yields ``(n_layers, n_frames, hidden)`` float32 hidden states
    (all transformer layers, including the embedding layer) for a single mono
    waveform segment. ``config.mert_sample_rate`` must equal the rate the loaded
    model's feature extractor expects (24 kHz for MERT-v1); the I/O layer decodes
    to it and ``extract`` raises on any mismatch rather than feeding the model
    off-rate audio.
    """

    name = "mert"

    def __init__(self, config: Config) -> None:
        """Store ``config`` and eagerly load the MERT model + feature extractor.

        The model is moved to ``config.device`` and put in eval mode. MERT-v1
        ships custom modeling code, so both the model and feature extractor are
        loaded with ``trust_remote_code=True`` — and therefore at a pinned
        revision: the default model uses the reviewed snapshot above unless
        ``config.mert_revision`` overrides it; custom models load the hub's
        latest unless a revision is configured.
        """
        self.config = config
        self.sample_rate = config.mert_sample_rate
        self.device = config.device
        revision = config.mert_revision
        if revision is None and config.mert_model_name == "m-a-p/MERT-v1-95M":
            revision = _DEFAULT_REVISION
        # The first construction downloads the weights (hundreds of MB) — say so,
        # or a cold-start run just looks hung.
        logger.info(
            "Loading MERT model %s (revision %s) on %s",
            config.mert_model_name,
            revision or "latest",
            self.device,
        )
        try:
            # transformers 5.x defaults the load dtype to "auto" (whatever precision
            # the checkpoint declares); pin float32 explicitly so the engine's
            # float32-end-to-end contract holds regardless of the model's config.
            self.model = AutoModel.from_pretrained(
                config.mert_model_name,
                revision=revision,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                config.mert_model_name, revision=revision, trust_remote_code=True
            )
        except Exception as exc:  # noqa: BLE001 — hub/torch errors never name the artifact
            raise ModelLoadError(
                f"could not load MERT model {config.mert_model_name!r} "
                f"(revision {revision or 'latest'}): {exc}. If you are offline, pre-download it "
                f"with `huggingface-cli download {config.mert_model_name}` (HF_HOME chooses the "
                f"cache location; HF_HUB_OFFLINE=1 forces cache-only resolution)."
            ) from exc
        self.model.to(self.device).eval()
        logger.info("MERT model %s ready on %s", config.mert_model_name, self.device)

    def _extractor_sample_rate(self) -> int:
        """Return the sampling rate the feature extractor expects.

        Falls back to ``self.sample_rate`` when the extractor exposes no
        ``sampling_rate`` attribute.
        """
        return int(getattr(self.feature_extractor, "sampling_rate", self.sample_rate))

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed one mono float32 waveform into stacked hidden states.

        Returns ``(n_layers, n_frames, hidden)`` float32 on CPU. ``sr`` must equal
        the rate the feature extractor expects (24 kHz for MERT-v1); a mismatch
        raises ``ValueError`` instead of being silently relabeled, because feeding
        off-rate audio time/pitch-warps every embedding. Very short or empty inputs
        are padded up to the extractor's minimum so the model never receives a
        zero-length frame.
        """
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        expected_sr = self._extractor_sample_rate()
        # Pass the ACTUAL decode rate so the extractor's own check fires on a mismatch; raising here
        # first gives an actionable message. Silently relabeling off-rate audio (the pre-fix bug)
        # made the model perceive every track as time/pitch-warped.
        if int(sr) != expected_sr:
            raise ValueError(
                f"MERT received audio at {int(sr)} Hz but its feature extractor expects "
                f"{expected_sr} Hz; set config.mert_sample_rate={expected_sr} so audio is decoded "
                f"at the model's rate (off-rate audio silently warps every embedding)."
            )

        # Guard the degenerate empty/very-short case: pad to a small floor so the
        # convolutional front-end always has something to chew on.
        min_len = max(expected_sr // 100, 1)  # ~10 ms of samples
        if wav.size < min_len:
            wav = np.pad(wav, (0, min_len - wav.size))

        inputs = self.feature_extractor(wav, sampling_rate=int(sr), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (batch=1, n_frames, hidden) per layer.
        hidden = torch.stack(outputs.hidden_states, dim=0)  # (n_layers, 1, n_frames, hidden)
        hidden = hidden.squeeze(1)  # (n_layers, n_frames, hidden)
        return hidden.detach().to("cpu").numpy().astype(np.float32)
