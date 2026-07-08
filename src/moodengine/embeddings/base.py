"""Common embedder interface + on-disk embedding cache.

This module is intentionally torch-free so that the cache helpers and the
abstract interface can be imported by the lightweight pipeline stages and the
test suite without pulling in the deep-learning stack. Concrete embedders
(:mod:`moodengine.embeddings.mert`, :mod:`moodengine.embeddings.clap`) import torch lazily.
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Embedding cache (key = content hash of the audio file + model + extra tag)
# --------------------------------------------------------------------------- #
def file_fingerprint(path: PathLike) -> str:
    """Return a short, stable content hash for an audio file (streamed sha1)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(path: PathLike, model_name: str, extra: str = "") -> str:
    """Build a filesystem-safe cache key from file content + model + extra tag."""
    fp = file_fingerprint(path)
    safe_model = model_name.replace("/", "_")
    suffix = f"__{extra}" if extra else ""
    return f"{safe_model}__{fp}{suffix}"


def provenance_cache_key(model_name: str, version: str, input_hash: str, config_hash: str) -> str:
    """``node__version__input_hash__config_hash`` — the generalized content-addressed cache key.

    The legacy :func:`cache_key` form (``model__fp__extra``) is a special case of this triplet,
    with algorithm version and config folded into ``extra``. This helper spells the triplet out
    so a downstream provenance layer can mint strings of the same shape. ``track_embedding`` still
    uses the legacy :func:`cache_key`, so existing on-disk ``.npy`` caches stay byte-identical.
    """
    return f"{model_name.replace('/', '_')}__{version}__{input_hash}__{config_hash}"


def cache_path(cache_dir: PathLike, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npy"


def load_cached(cache_dir: PathLike, key: str) -> Optional[np.ndarray]:
    """Return the cached array for ``key``, or ``None`` if absent or unreadable.

    A truncated / corrupt ``.npy`` (crash mid-write, disk hiccup) is treated as
    a cache MISS, not an error: it is logged, deleted so it cannot shadow the
    same key forever, and ``None`` is returned so the caller recomputes and
    rewrites. Without this, one bad cache entry would silently make a perfectly
    decodable track vanish from every future run.
    """
    p = cache_path(cache_dir, key)
    if not p.exists():
        return None
    try:
        return np.load(p, allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        logger.warning("Corrupt cache entry %s (%s); dropping it and recomputing.", p, exc)
        p.unlink(missing_ok=True)
        return None


def save_cached(cache_dir: PathLike, key: str, array: np.ndarray) -> None:
    """Persist ``array`` under ``key`` (creates the cache dir if needed).

    The write is atomic — serialized to a sibling temp file, then moved into
    place with ``os.replace`` — so a crash mid-write or a concurrent reader
    (several processes may share one cache dir) can never observe a
    half-written ``.npy``.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    final = cache_path(cache_dir, key)
    # PID-unique temp name: two processes saving the same key must not collide
    # on the temp file. Opened explicitly because np.save appends ".npy" to
    # plain paths that lack the suffix.
    tmp = final.with_name(f"{final.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(array))
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Embedder interface
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    """Abstract audio embedder.

    Subclasses set ``name`` (used in cache keys + DataFrame columns) and
    ``sample_rate`` (the SR their model expects; the I/O layer resamples to it).

    The single required method, :meth:`extract`, embeds one already-decoded mono
    waveform segment. Per-model output shapes:

      * MERT -> ``(n_layers, n_frames, hidden)`` frame-level hidden states.
      * CLAP -> ``(hidden,)`` clip-level audio embedding.

    Track-level pooling and disk caching are orchestrated in
    :mod:`moodengine.pipeline`, keeping this interface free of pooling policy.
    """

    name: str
    sample_rate: int

    @abstractmethod
    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed a single mono float32 waveform sampled at ``sr``.

        Implementations may assume ``sr == self.sample_rate`` (the caller
        resamples), but should validate and be robust to short inputs.
        """
        raise NotImplementedError
