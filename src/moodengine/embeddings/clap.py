"""CLAP clip-level audio + text embedder.

Wraps the LAION CLAP model, which maps audio and natural-language prompts into a
shared embedding space (the basis for zero-shot mood labelling). Loading is
lazy: the checkpoint is fetched on first construction, so importing this module
is cheap apart from the (eager) ``torch`` + ``laion_clap`` imports at the top.
:meth:`CLAPEmbedder.extract` returns one clip-level audio vector per segment;
track-level pooling and caching live in :mod:`moodengine.pooling` / :mod:`moodengine.pipeline`.
CLAP expects 48 kHz mono float32 audio; callers resample upstream.
"""

from __future__ import annotations

import logging

import laion_clap
import numpy as np
import torch

from moodengine._math import l2_normalize as _l2_normalize
from moodengine.config import Config
from moodengine.embeddings.base import Embedder
from moodengine.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


def _as_float32(emb) -> np.ndarray:
    """Coerce a laion_clap embedding to a float32 numpy array.

    Across laion_clap versions the getters return either a numpy array or a torch tensor
    (older builds took a ``use_tensor`` flag that newer ones dropped), so detach/move any
    tensor to CPU before converting rather than assuming one return type.
    """
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


class CLAPEmbedder(Embedder):
    """Clip-level embedder backed by LAION CLAP.

    ``extract`` yields a single ``(hidden,)`` float32 audio embedding for one
    mono waveform segment, and :meth:`embed_text` maps prompts into the same
    space as ``(n_prompts, hidden)``. The model expects
    ``config.clap_sample_rate`` (48 kHz); callers resample upstream.
    """

    name = "clap"

    def __init__(self, config: Config) -> None:
        """Store ``config`` and eagerly load the CLAP model + checkpoint.

        The fusion flag and audio backbone come from ``config.clap_enable_fusion``
        / ``config.clap_amodel``. ``config.clap_checkpoint`` selects a specific
        checkpoint path; ``None`` loads LAION's default pretrained weights.
        """
        self.config = config
        self.sample_rate = config.clap_sample_rate
        self.device = config.device
        # Pass our resolved device so CLAP honours MPS/CUDA — laion-clap
        # otherwise falls back to CPU whenever CUDA is absent (e.g. on Apple
        # Silicon, where we want MPS). Set PYTORCH_ENABLE_MPS_FALLBACK=1 in the
        # environment so any op MPS lacks falls back to CPU instead of erroring.
        self.model = laion_clap.CLAP_Module(
            enable_fusion=config.clap_enable_fusion,
            amodel=config.clap_amodel,
            device=config.device,
        )
        # The first construction downloads the checkpoint (~GB-scale) — say so,
        # or a cold-start run just looks hung.
        logger.info("Loading CLAP checkpoint (amodel=%s) on %s", config.clap_amodel, self.device)
        try:
            self._load_ckpt_trusted(self._resolve_checkpoint())
        except Exception as exc:  # noqa: BLE001 — hub/torch errors never name the artifact
            raise ModelLoadError(
                f"could not load the CLAP checkpoint for amodel={config.clap_amodel!r} "
                f"(clap_checkpoint={config.clap_checkpoint!r}): {exc}. If you are offline, "
                f"pre-download the default music checkpoint with `huggingface-cli download "
                f"lukewys/laion_clap music_audioset_epoch_15_esc_90.14.pt` (HF_HOME chooses the "
                f"cache location; HF_HUB_OFFLINE=1 forces cache-only resolution)."
            ) from exc
        logger.info("CLAP checkpoint ready on %s", self.device)

    def _load_ckpt_trusted(self, checkpoint) -> None:
        """Call ``laion_clap``'s ``load_ckpt`` across modern torch / transformers versions.

        Two forward-compat shims, both scoped to this call and restored in ``finally`` so no
        global torch state leaks:

        * **torch.load** — PyTorch >= 2.6 flipped its ``weights_only`` default to ``True``,
          which rejects the numpy scalars pickled in LAION's published checkpoints;
          ``laion_clap`` does not pass the flag. The checkpoint is LAION's own music weights
          (the pinned HF repo, or an explicit ``clap_checkpoint`` the caller supplied) — a
          trusted source — so restoring ``weights_only=False`` is safe.
        * **load_state_dict strict=False** — transformers >= ~4.31 dropped the derived,
          non-persistent ``text_branch.embeddings.position_ids`` buffer that older LAION
          checkpoints still carry, so a strict load rejects that one benign extra key while
          every learned weight still matches. Tolerating it lets the shipped checkpoint load
          against a current transformers without silently dropping real parameters.
        """
        original_load = torch.load
        original_load_state_dict = torch.nn.Module.load_state_dict

        def _trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        def _lenient_load_state_dict(self, state_dict, *args, **kwargs):
            kwargs["strict"] = False
            return original_load_state_dict(self, state_dict, *args, **kwargs)

        torch.load = _trusted_load
        torch.nn.Module.load_state_dict = _lenient_load_state_dict  # type: ignore[method-assign]
        try:
            self.model.load_ckpt(checkpoint)
        finally:
            torch.load = original_load
            torch.nn.Module.load_state_dict = original_load_state_dict  # type: ignore[method-assign]

    def _resolve_checkpoint(self):
        """Pick a checkpoint compatible with ``config.clap_amodel``.

        An explicit ``config.clap_checkpoint`` always wins. Otherwise the choice
        must match the audio backbone, or laion-clap raises a state-dict size
        mismatch: ``HTSAT-base`` pairs with LAION's *music* checkpoint (downloaded
        once from the HF hub), while ``HTSAT-tiny`` uses laion-clap's built-in
        default (``load_ckpt(None)`` fetches the 630k-audioset weights).
        """
        if self.config.clap_checkpoint is not None:
            return self.config.clap_checkpoint
        if str(self.config.clap_amodel).lower() == "htsat-base":
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id="lukewys/laion_clap",
                filename="music_audioset_epoch_15_esc_90.14.pt",
            )
        return None

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed one mono float32 waveform into a clip-level vector.

        Returns a 1-D ``(hidden,)`` float32 array. Empty/degenerate inputs are
        padded to a small floor so the model never receives a zero-length clip.
        Quantization is disabled so embeddings stay in raw float space.
        """
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)

        # Guard the degenerate empty/very-short case: pad to a small floor.
        min_len = max(self.sample_rate // 100, 1)  # ~10 ms of samples
        if wav.size < min_len:
            wav = np.pad(wav, (0, min_len - wav.size))

        with torch.no_grad():
            emb = self.model.get_audio_embedding_from_data(x=wav[None, :])
        return _as_float32(emb).reshape(-1)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Embed text prompts into the shared CLAP space.

        Returns ``(n_prompts, hidden)`` float32, L2-normalized (downstream
        labelling assumes unit-norm text embeddings). An empty prompt list
        yields an empty ``(0,)`` array rather than crashing.
        """
        if not prompts:
            return np.empty((0,), dtype=np.float32)

        with torch.no_grad():
            emb = self.model.get_text_embedding(list(prompts))
        emb = _as_float32(emb).reshape(len(prompts), -1)
        return _l2_normalize(emb, axis=-1)
