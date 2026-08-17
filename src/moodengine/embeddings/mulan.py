"""MuQ-MuLan clip-level audio + text embedder — the second audio-text backbone.

Wraps Tencent AI Lab's MuQ-MuLan, which maps audio and natural-language prompts into a shared
512-d space, exactly the surface :mod:`moodengine.labeling` needs. It is an ALTERNATIVE to
:mod:`moodengine.embeddings.clap`, not a replacement: both stay selectable, because which one
labels a given library better is an empirical question and the two disagree in ways that matter.

Measured on 60 tracks of a real personal library against the shipped mood vocabulary, MuQ-MuLan
gave more decisive labels (mean top1-top2 margin 0.120 vs 0.096, mean entropy 2.220 vs 2.360) and
a markedly more separable label space (mean mutual cosine between mood directions 0.400 vs 0.568 —
see :func:`moodengine.labeling.label_direction_redundancy`). Those are separability proxies, NOT
accuracy: no gold labels were involved, and a sharper distribution can also be confidently wrong.
Settle it on your own library with ``scripts/bench_valence_arousal.py`` before switching.

Two caveats that belong next to any comparison with the published numbers:

* The paper's +5.4 ROC-AUC over LAION-CLAP on MagnaTagATune zero-shot tagging was measured with a
  130K-hour checkpoint. The downloadable weights are trained on the Million Song Dataset instead,
  and the model card says outright they "may not achieve the same level of performance as reported
  in the paper". Do not carry the paper's number over to these weights.
* That benchmark scores a fixed TAG vocabulary. Independent work on perceptual timbre semantics
  reports LAION-CLAP aligning better with human judgement on free-form descriptive adjectives —
  which is what a mood prompt is. Hence: both backbones stay, and you measure.

Licensing: the ``muq`` package code is MIT, but the MuQ-MuLan and MuQ-large weights are
**CC-BY-NC-4.0 (non-commercial)** — a separate grant from this package's own code license, and the
same posture as the MERT weights. Commercial use requires licensing the weights separately.

MuQ-MuLan expects 24 kHz mono float32 audio (the same rate as MERT, and NOT CLAP's 48 kHz);
callers resample upstream.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from muq import MuQMuLan

from moodengine._math import l2_normalize as _l2_normalize
from moodengine.config import Config
from moodengine.embeddings.base import Embedder
from moodengine.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

#: Seconds of audio MuQ-MuLan embeds per internal clip. Its own config; audio longer than this is
#: split, embedded clip by clip and averaged. Named here because it drives the re-normalization
#: below rather than being an arbitrary constant.
_CLIP_SECONDS: int = 10


class MuLanEmbedder(Embedder):
    """Clip-level embedder backed by MuQ-MuLan.

    ``extract`` yields a single ``(512,)`` float32 audio embedding for one mono waveform segment,
    and :meth:`embed_text` maps prompts into the same space as ``(n_prompts, 512)``. Both are
    L2-normalized, so a dot product between them is a cosine.

    The model expects ``config.mulan_sample_rate`` (24 kHz); callers resample upstream, and
    ``extract`` raises on a mismatch rather than embedding time/pitch-warped audio.
    """

    name = "mulan"

    def __init__(self, config: Config) -> None:
        """Store ``config`` and eagerly load MuQ-MuLan onto ``config.device``.

        Constructing this downloads roughly 5 GB across THREE hub repos, not one: the MuLan
        weights (~2.65 GB), plus the MuQ audio tower and ``xlm-roberta-base`` that its constructor
        builds before the MuLan state dict overwrites them. Budget for that on a cold start.
        """
        self.config = config
        self.sample_rate = config.mulan_sample_rate
        self.device = config.device
        logger.info(
            "Loading MuQ-MuLan %s on %s (first run downloads ~5 GB across 3 hub repos)",
            config.mulan_model_name,
            self.device,
        )
        try:
            self.model = MuQMuLan.from_pretrained(config.mulan_model_name)
            self.model = self.model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 — hub/torch errors never name the artifact
            # A network failure surfaces upstream as `__init__() missing 1 required positional
            # argument: 'config'`, because the hub mixin only injects `config=` once it has
            # actually fetched config.json. Name the real cause rather than passing that on.
            raise ModelLoadError(
                f"could not load MuQ-MuLan {config.mulan_model_name!r}: {exc}. This model builds "
                "its audio and text towers from two further hub repos, so it needs network access "
                "on a cold start; a download failure surfaces from upstream as a confusing "
                "missing-'config'-argument TypeError. If you are offline, pre-download it on a "
                "connected machine (HF_HOME chooses the cache location; HF_HUB_OFFLINE=1 forces "
                "cache-only resolution)."
            ) from exc
        logger.info("MuQ-MuLan ready on %s", self.device)

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed one mono float32 waveform into a clip-level vector.

        Returns a 1-D ``(512,)`` float32 array, L2-normalized. ``sr`` must equal
        ``self.sample_rate`` (24 kHz); a mismatch raises ``ValueError`` instead of being silently
        relabeled, because off-rate audio time/pitch-warps every embedding. Inputs shorter than
        one internal clip are wrap-padded by the model itself, so no floor is applied here.
        """
        if int(sr) != int(self.sample_rate):
            raise ValueError(
                f"MuQ-MuLan received audio at {int(sr)} Hz but expects {int(self.sample_rate)} Hz; "
                f"set config.mulan_sample_rate={int(self.sample_rate)} so audio is decoded at the "
                "model's rate (off-rate audio silently warps every embedding)."
            )
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        with torch.inference_mode():
            emb = self.model(wavs=torch.from_numpy(wav).to(self.device)[None, :])
        # Re-normalize. The model L2-normalizes each internal clip and then averages them WITHOUT
        # re-normalizing, so anything past `_CLIP_SECONDS` comes back with norm < 1 and a raw dot
        # product against a unit-norm text vector would not be a cosine. Everything downstream
        # (labeling, search) assumes unit rows, so this is restored at the boundary.
        return _l2_normalize(_as_float32(emb).reshape(-1), axis=-1)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Embed text prompts into the shared MuQ-MuLan space.

        Returns ``(n_prompts, 512)`` float32, L2-normalized (the model already returns unit rows
        for text; the normalize is defensive and free). An empty prompt list yields an empty
        ``(0,)`` array rather than crashing.

        The tokenizer neither truncates nor sets a ``max_length``, so a prompt beyond XLM-R's
        512-token limit raises inside the model rather than being cut. Mood prompts are far below
        that; a caller supplying long free text should truncate first.
        """
        if not prompts:
            return np.empty((0,), dtype=np.float32)
        with torch.inference_mode():
            emb = self.model(texts=list(prompts))
        return _l2_normalize(_as_float32(emb).reshape(len(prompts), -1), axis=-1)


def _as_float32(emb) -> np.ndarray:
    """Coerce a MuQ-MuLan output to a float32 numpy array (detach and move off the device first)."""
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)
