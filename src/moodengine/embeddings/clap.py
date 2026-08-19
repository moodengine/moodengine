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
import sys

import numpy as np
import torch

from moodengine._math import l2_normalize as _l2_normalize
from moodengine.config import Config, ensure_clap_fusion_supported
from moodengine.embeddings.base import Embedder
from moodengine.exceptions import ModelLoadError

# laion_clap parses sys.argv (argparse) at IMPORT time — its training/data.py runs a
# module-level parse_args() — so importing it from a process that carries its own CLI
# flags (pytest, or any host application) aborts with SystemExit before our code runs.
# Import it behind a cleared argv; laion_clap's parsed training args are unused by the
# inference-only path we drive. Restored in finally so no global argv state leaks.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    import laion_clap  # noqa: E402 — eager but argv-guarded (see the note above)
finally:
    sys.argv = _saved_argv

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


def _ensure_batched(tokens: dict) -> dict:
    """Restore a leading batch dim to any 1-D tensor in a tokenizer output.

    laion_clap's default text tokenizer squeezes the batch dim (``{k: v.squeeze(0) ...}`` in its
    ``hook.py``), so a *single* prompt yields a 1-D ``input_ids``. transformers < 5 reshaped 1-D → 2-D
    inside RoBERTa, so this was harmless; transformers >= 5 (required by moodengine's CVE-fix pin) does
    not, so a single-prompt text embed crashes in ``create_position_ids_from_input_ids`` — ``cumsum(mask,
    dim=1)`` on a 1-D mask raises ``IndexError: Dimension out of range``. Re-adding the batch dim keeps
    ``input_ids`` ``(n, seq)`` for any ``n``, which is byte-identical to the pre-5 reshape path.
    """
    return {
        k: (v.unsqueeze(0) if hasattr(v, "dim") and v.dim() == 1 else v) for k, v in tokens.items()
    }


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

        Refuses (``ValueError``) a config whose segments exceed
        :data:`~moodengine.config.CLAP_FUSION_SAMPLE_LIMIT` — see
        :func:`~moodengine.config.ensure_clap_fusion_supported`. Checked before any weight is
        touched, so the refusal costs nothing.
        """
        ensure_clap_fusion_supported(config)
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
                f"pre-download the default music checkpoint with `hf download "
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

    def _model_sample_rate(self) -> int | None:
        """The rate the LOADED checkpoint's audio config declares, or ``None`` if it cannot be read.

        Checked against ``config.clap_sample_rate`` rather than trusting it: the pipeline calls
        ``extract(seg, sr=embedder.sample_rate)``, so ``sr`` echoes the config and comparing the two
        would always agree — including when the config itself is wrong. Only the model can say what
        it was trained at.

        laion-clap keeps the loaded audio config in TWO places, neither a published API:
        ``CLAP_Module.model_cfg`` (a plain dict parsed from the checkpoint's json) and
        ``CLAP_Module.model.audio_cfg`` (a ``CLAPAudioCfp`` dataclass). Both are tried — reading
        only one of them is how this check silently returned ``None`` on the shipped default
        checkpoint, which made the guard inert exactly where it was supposed to apply. Still read
        defensively (``None`` skips the check): a future layout change must not stop a
        correctly-configured run.
        """
        outer = getattr(self.model, "model_cfg", None)
        audio_cfg = outer.get("audio_cfg") if isinstance(outer, dict) else None
        for source in (audio_cfg, getattr(getattr(self.model, "model", None), "audio_cfg", None)):
            rate = (
                source.get("sample_rate")
                if isinstance(source, dict)
                else getattr(source, "sample_rate", None)
            )
            if isinstance(rate, (int, float)):
                return int(rate)
        return None

    def _ensure_model_rate(self, sr: int) -> None:
        """Raise unless ``sr`` matches the loaded checkpoint's declared rate.

        Split out of :meth:`extract` so :meth:`extract_batch` cannot silently skip it — the
        pipeline probes for ``extract_batch`` and takes it whenever present, which made this the
        live path and left the checkpoint-rate guard unreachable on it.
        """
        expected = self._model_sample_rate()
        if expected is not None and int(sr) != expected:
            raise ValueError(
                f"CLAP received audio at {int(sr)} Hz but the loaded checkpoint "
                f"(amodel={self.config.clap_amodel!r}) declares {expected} Hz; set "
                f"config.clap_sample_rate={expected} so audio is decoded at the model's rate "
                "(off-rate audio silently warps every embedding)."
            )

    def _padded(self, waveform: np.ndarray) -> np.ndarray:
        """1-D float32 view of ``waveform``, padded up to a ~10 ms floor.

        The floor keeps the model from ever receiving a zero-length clip. Shared with
        :meth:`extract_batch` so the base contract holds — a batched override must not change the
        numbers — and ``io_audio.segment_waveform`` does return a sub-floor trailing window on very
        short files.
        """
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        min_len = max(self.sample_rate // 100, 1)  # ~10 ms of samples
        if wav.size < min_len:
            wav = np.pad(wav, (0, min_len - wav.size))
        return wav

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed one mono float32 waveform into a clip-level vector.

        Returns a 1-D ``(hidden,)`` float32 array. ``sr`` must be the rate the waveform was decoded
        at and must match the loaded checkpoint's declared rate; a mismatch raises ``ValueError``
        rather than embedding time/pitch-warped audio, mirroring the MERT embedder. Empty/degenerate
        inputs are padded to a small floor so the model never receives a zero-length clip.
        """
        self._ensure_model_rate(sr)
        wav = self._padded(waveform)

        with torch.inference_mode():
            emb = self.model.get_audio_embedding_from_data(x=wav[None, :])
        return _as_float32(emb).reshape(-1)

    def extract_batch(self, waveforms: list[np.ndarray], sr: int) -> list[np.ndarray]:
        """Embed several segments in one forward pass per distinct LENGTH.

        laion-clap accepts ``x`` as ``(N, T)``, but a track's segments are not all ``T``: the
        trailing window is short (down to ``config.min_segment_seconds``), so a single
        ``np.stack`` raises. Grouping by length keeps every group rectangular and leaves the
        numerics untouched — measured max deviation from the one-at-a-time path is 1.5e-8, i.e.
        float32 rounding.

        Applies the SAME two guards as :meth:`extract` — the checkpoint-rate check and the
        short-clip pad — because the pipeline takes this path whenever it exists, and the base
        contract is that a batched override must not change the numbers.

        Measured on 8 real tracks (96 segments) on Apple Silicon: 1.24x on the forward pass and
        1.12x end to end, since decoding is 45 % of the per-track cost. Peak memory grows with the
        group, bounded by ``config.max_segments_per_track`` (12 by default, ~23 MB of input).
        """
        self._ensure_model_rate(sr)
        # Pad BEFORE grouping: the floor can lift a sub-floor trailing window into another
        # group's length, and grouping on the raw sizes would then stack ragged rows.
        prepared = [self._padded(wav) for wav in waveforms]

        by_length: dict[int, list[int]] = {}
        for i, wav in enumerate(prepared):
            by_length.setdefault(int(wav.size), []).append(i)

        out: list[np.ndarray | None] = [None] * len(prepared)
        for _length, idxs in sorted(by_length.items()):
            stack = np.stack([prepared[i] for i in idxs])
            with torch.inference_mode():
                emb = _as_float32(self.model.get_audio_embedding_from_data(x=stack))
            for slot, i in enumerate(idxs):
                out[i] = emb[slot].reshape(-1)
        return [np.asarray(v, dtype=np.float32) for v in out if v is not None]

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Embed text prompts into the shared CLAP space.

        Returns ``(n_prompts, hidden)`` float32, L2-normalized (downstream
        labelling assumes unit-norm text embeddings). An empty prompt list
        yields an empty ``(0,)`` array rather than crashing.
        """
        if not prompts:
            return np.empty((0,), dtype=np.float32)

        with torch.inference_mode():
            emb = self.model.get_text_embedding(list(prompts), tokenizer=self._tokenizer_batched)
        emb = _as_float32(emb).reshape(len(prompts), -1)
        return _l2_normalize(emb, axis=-1)

    def _tokenizer_batched(self, prompts: list[str]) -> dict:
        """laion_clap's own RoBERTa tokenizer, but with the batch dim preserved (see
        :func:`_ensure_batched`). Passed through ``get_text_embedding(..., tokenizer=)`` so a
        single-prompt embed no longer emits a 1-D ``input_ids`` that transformers-5 RoBERTa rejects.
        """
        return _ensure_batched(self.model.tokenizer(prompts))
